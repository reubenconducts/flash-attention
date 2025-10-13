from typing import Tuple, Optional, Callable
import torch

"""
Computes block sparse masks for use in CuTe DSL Flash Attention Flex Attention
for some common mask_mod functions. To be replaced by a preprocessing kernel eventually
"""

def compute_block_sparsity(
    config, 
    mask_mod_flex: Callable, 
    device: str,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    buffers: Optional[list] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute block sparsity tensors from mask_mod function
    
    Args:
        config: Benchmark configuration
        mask_mod_flex: Mask function for flex attention
        device: Device to create tensors on
        cu_seqlens_q: Optional cumulative sequence lengths for Q (varlen)
        cu_seqlens_k: Optional cumulative sequence lengths for K (varlen)
    
    Returns:
        Tuple of (full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx)
    """

    if not config.use_mask_mod or mask_mod_flex is None:
        return None, None, None, None

    use_varlen = cu_seqlens_q is not None
    num_heads = config.nheads

    if use_varlen:
        # Varlen: compute max blocks and iterate per sequence
        assert cu_seqlens_k is not None, "cu_seqlens_k required when using cu_seqlens_q"
        assert cu_seqlens_q.shape[0] == config.batch_size + 1
        assert cu_seqlens_k.shape[0] == config.batch_size + 1
        
        # Compute max m_blocks across all sequences
        max_m_blocks = 0
        for seq_idx in range(config.batch_size):
            seq_len_q = (cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]).item()
            n_blocks_q = (seq_len_q + config.m_block_size - 1) // config.m_block_size
            max_m_blocks = max(max_m_blocks, n_blocks_q)
        
        # Allocate padded tensors with (batch, head, ...) ordering
        total_k = cu_seqlens_k[-1].item()
        max_n_blocks = (total_k + config.n_block_size - 1) // config.n_block_size
        
        full_block_cnt = torch.zeros(
            (config.batch_size, num_heads, max_m_blocks), device=device, dtype=torch.int32
        )
        mask_block_cnt = torch.zeros(
            (config.batch_size, num_heads, max_m_blocks), device=device, dtype=torch.int32
        )
        full_block_idx = torch.zeros(
            (config.batch_size, num_heads, max_m_blocks, max_n_blocks), device=device, dtype=torch.int32
        )
        mask_block_idx = torch.zeros(
            (config.batch_size, num_heads, max_m_blocks, max_n_blocks), device=device, dtype=torch.int32
        )
        
        # Process each sequence
        for seq_idx in range(config.batch_size):
            seq_start_q = cu_seqlens_q[seq_idx].item()
            seq_end_q = cu_seqlens_q[seq_idx + 1].item()
            seq_len_q = seq_end_q - seq_start_q
            
            seq_start_k = cu_seqlens_k[seq_idx].item()
            seq_end_k = cu_seqlens_k[seq_idx + 1].item()
            seq_len_k = seq_end_k - seq_start_k
            
            n_blocks_q = (seq_len_q + config.m_block_size - 1) // config.m_block_size
            n_blocks_k = (seq_len_k + config.n_block_size - 1) // config.n_block_size
            
            # Global block indices for this sequence
            first_m_block_global = seq_start_q // config.m_block_size
            first_n_block_global = seq_start_k // config.n_block_size
            
            # Apply mask pattern (treating indices as sequence-local)
            if config.mask_mod_name == "causal":
                _compute_causal_varlen_blocks(
                    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx,
                    seq_idx, num_heads, n_blocks_q, n_blocks_k,
                    seq_start_q, seq_end_q, seq_start_k, seq_end_k,
                    first_n_block_global, config.m_block_size, config.n_block_size,
                    device
                )
            elif config.mask_mod_name == "identity":
                _compute_identity_varlen_blocks(
                    full_block_cnt, full_block_idx,
                    seq_idx, num_heads, n_blocks_q, n_blocks_k,
                    first_n_block_global, device
                )
            else:
                # Generic: sample-based classification
                _compute_generic_varlen_blocks(
                    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx,
                    mask_mod_flex, seq_idx, num_heads, n_blocks_q, n_blocks_k,
                    seq_start_q, seq_end_q, seq_start_k, seq_end_k,
                    seq_len_q, seq_len_k, first_n_block_global,
                    config.m_block_size, config.n_block_size, config.nheads_kv,
                    device
                )
        
        return full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx
    
    else:
        # Non-varlen: use (batch, head, ...) ordering
        n_blocks_k = (config.seqlen_k + config.n_block_size - 1) // config.n_block_size
        n_blocks_q = (config.seqlen_q + config.m_block_size - 1) // config.m_block_size

        full_block_cnt = torch.zeros(
            (config.batch_size, num_heads, n_blocks_q), device=device, dtype=torch.int32
        )
        mask_block_cnt = torch.zeros(
            (config.batch_size, num_heads, n_blocks_q), device=device, dtype=torch.int32
        )
        full_block_idx = torch.zeros(
            (config.batch_size, num_heads, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
        )
        mask_block_idx = torch.zeros(
            (config.batch_size, num_heads, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
        )

        if config.mask_mod_name == "identity":
            k_blocks = torch.arange(n_blocks_k, device=device)
            for q_block in range(n_blocks_q):
                full_block_cnt[:, :, q_block] = n_blocks_k
                full_block_idx[:, :, q_block, :n_blocks_k] = k_blocks

        elif config.mask_mod_name == "identity_partial":
            k_blocks = torch.arange(n_blocks_k, device=device)
            for q_block in range(n_blocks_q):
                mask_block_cnt[:, :, q_block] = n_blocks_k
                mask_block_idx[:, :, q_block, :n_blocks_k] = k_blocks

        elif config.mask_mod_name == "block_causal":
            k_blocks = torch.arange(n_blocks_k, device=device)
            for q_block in range(n_blocks_q):
                full_indices = k_blocks[k_blocks <= q_block]
                if len(full_indices) > 0:
                    full_block_cnt[:, :, q_block] = len(full_indices)
                    full_block_idx[:, :, q_block, : len(full_indices)] = full_indices

        elif config.mask_mod_name == "causal":
            q_blocks = torch.arange(n_blocks_q, device=device)
            k_blocks = torch.arange(n_blocks_k, device=device)

            q_starts = q_blocks * config.m_block_size
            q_ends = torch.minimum(
                (q_blocks + 1) * config.m_block_size, torch.tensor(config.seqlen_q, device=device)
            )
            k_starts = k_blocks * config.n_block_size
            k_ends = torch.minimum(
                (k_blocks + 1) * config.n_block_size, torch.tensor(config.seqlen_k, device=device)
            )

            q_starts = q_starts.unsqueeze(1)
            q_ends = q_ends.unsqueeze(1)
            k_starts = k_starts.unsqueeze(0)
            k_ends = k_ends.unsqueeze(0)

            offset = config.seqlen_k - config.seqlen_q
            is_full = (k_ends - 1) <= (q_starts + offset)
            is_partial = (k_starts <= (q_ends - 1 + offset)) & ((k_ends - 1) > (q_starts + offset)) & ~is_full

            for q_block in range(n_blocks_q):
                full_indices = k_blocks[is_full[q_block]]
                partial_indices = k_blocks[is_partial[q_block]]

                if len(full_indices) > 0:
                    full_block_cnt[:, :, q_block] = len(full_indices)
                    full_block_idx[:, :, q_block, : len(full_indices)] = full_indices

                if len(partial_indices) > 0:
                    mask_block_cnt[:, :, q_block] = len(partial_indices)
                    mask_block_idx[:, :, q_block, : len(partial_indices)] = partial_indices

        elif config.mask_mod_name == "document":
            # doc_ids shape: (batch_size, nheads, seqlen_q)
            doc_ids = buffers[0]
            for b in range(config.batch_size):
                for h in range(num_heads):
                    for q_block in range(n_blocks_q):
                        q_start = q_block * config.m_block_size
                        q_end = min((q_block + 1) * config.m_block_size, config.seqlen_q) - 1
                        
                        # Since monotone non-decreasing, just check corners
                        doc_q_start = doc_ids[b, h, q_start]
                        doc_q_end = doc_ids[b, h, q_end]
                        
                        for k_block in range(n_blocks_k):
                            k_start = k_block * config.n_block_size
                            k_end = min((k_block + 1) * config.n_block_size, config.seqlen_k) - 1
                            
                            doc_k_start = doc_ids[b, h, k_start]
                            doc_k_end = doc_ids[b, h, k_end]
                            
                            # Check if all four corners are in the same document
                            if doc_q_start == doc_q_end == doc_k_start == doc_k_end:
                                # Fully computed block
                                cnt = full_block_cnt[b, h, q_block].item()
                                full_block_idx[b, h, q_block, cnt] = k_block
                                full_block_cnt[b, h, q_block] += 1
                            elif not (doc_q_start == doc_q_end and doc_k_start == doc_k_end and doc_q_start != doc_k_start):
                                # Partially masked block (some corners match, some don't)
                                cnt = mask_block_cnt[b, h, q_block].item()
                                mask_block_idx[b, h, q_block, cnt] = k_block
                                mask_block_cnt[b, h, q_block] += 1
                            # else: fully masked block (all q in one doc, all k in different doc) - skip

        return full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx


def _compute_causal_varlen_blocks(
    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx,
    seq_idx, num_heads, n_blocks_q, n_blocks_k,
    seq_start_q, seq_end_q, seq_start_k, seq_end_k,
    first_n_block_global, m_block_size, n_block_size, device
):
    """Compute causal block sparsity for one varlen sequence"""
    for m_local in range(n_blocks_q):
        m_start_global = seq_start_q + m_local * m_block_size
        m_end_global = min(seq_start_q + (m_local + 1) * m_block_size, seq_end_q)
        
        full_blocks = []
        partial_blocks = []
        
        for n_local in range(n_blocks_k):
            n_block_global = first_n_block_global + n_local
            n_start_global = seq_start_k + n_local * n_block_size
            n_end_global = min(seq_start_k + (n_local + 1) * n_block_size, seq_end_k)
            
            # Convert to sequence-local coordinates for causal check
            m_start_local = m_start_global - seq_start_q
            m_end_local = m_end_global - seq_start_q
            n_start_local = n_start_global - seq_start_k
            n_end_local = n_end_global - seq_start_k
            
            # Causal: q_pos >= kv_pos (in local coordinates)
            # Full block: all positions satisfy causal
            # Partial block: some positions satisfy causal
            
            # Block is full if: min(q) >= max(kv)
            # Block is partial if: max(q) >= min(kv) and not full
            is_full = (m_start_local >= n_end_local - 1)
            is_partial = (m_end_local - 1 >= n_start_local) and not is_full
            
            # Also check if block is at sequence boundary (needs masking)
            at_boundary = (m_end_global > seq_end_q) or (n_end_global > seq_end_k)
            
            if is_full and not at_boundary:
                full_blocks.append(n_block_global)
            elif is_partial or at_boundary:
                partial_blocks.append(n_block_global)
        
        if full_blocks:
            full_block_cnt[seq_idx, :, m_local] = len(full_blocks)
            full_block_idx[seq_idx, :, m_local, :len(full_blocks)] = torch.tensor(
                full_blocks, device=device, dtype=torch.int32
            )
        
        if partial_blocks:
            mask_block_cnt[seq_idx, :, m_local] = len(partial_blocks)
            mask_block_idx[seq_idx, :, m_local, :len(partial_blocks)] = torch.tensor(
                partial_blocks, device=device, dtype=torch.int32
            )


def _compute_identity_varlen_blocks(
    full_block_cnt, full_block_idx,
    seq_idx, num_heads, n_blocks_q, n_blocks_k,
    first_n_block_global, device
):
    """Compute identity (all attend) block sparsity for one varlen sequence"""
    n_blocks_global = torch.arange(
        first_n_block_global, first_n_block_global + n_blocks_k,
        device=device, dtype=torch.int32
    )
    
    for m_local in range(n_blocks_q):
        full_block_cnt[seq_idx, :, m_local] = n_blocks_k
        full_block_idx[seq_idx, :, m_local, :n_blocks_k] = n_blocks_global


def _compute_generic_varlen_blocks(
    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx,
    mask_mod_flex, seq_idx, num_heads, n_blocks_q, n_blocks_k,
    seq_start_q, seq_end_q, seq_start_k, seq_end_k,
    seq_len_q, seq_len_k, first_n_block_global,
    m_block_size, n_block_size, nheads_kv, device
):
    """Generic sampling-based block classification for varlen sequences"""
    qhead_per_kvhead = num_heads // nheads_kv
    
    for h_q in range(num_heads):
        h_kv = h_q // qhead_per_kvhead
        
        for m_local in range(n_blocks_q):
            m_start_local = m_local * m_block_size
            m_end_local = min((m_local + 1) * m_block_size, seq_len_q)
            
            full_blocks = []
            partial_blocks = []
            
            for n_local in range(n_blocks_k):
                n_block_global = first_n_block_global + n_local
                n_start_local = n_local * n_block_size
                n_end_local = min((n_local + 1) * n_block_size, seq_len_k)
                
                # Sample positions (using sequence-local coordinates for mask_mod)
                sample_positions = [
                    (m_start_local, n_start_local),
                    (m_start_local, n_end_local - 1),
                    (m_end_local - 1, n_start_local),
                    (m_end_local - 1, n_end_local - 1),
                    ((m_start_local + m_end_local) // 2, (n_start_local + n_end_local) // 2),
                ]
                
                unmasked_count = sum(
                    1 for q_pos, k_pos in sample_positions
                    if mask_mod_flex(seq_idx, h_q, q_pos, k_pos, seq_len_q, seq_len_k)
                )
                
                if unmasked_count == len(sample_positions):
                    full_blocks.append(n_block_global)
                elif unmasked_count > 0:
                    partial_blocks.append(n_block_global)
            
            if full_blocks:
                full_block_cnt[seq_idx, h_q, m_local] = len(full_blocks)
                full_block_idx[seq_idx, h_q, m_local, :len(full_blocks)] = torch.tensor(
                    full_blocks, device=device, dtype=torch.int32
                )
            
            if partial_blocks:
                mask_block_cnt[seq_idx, h_q, m_local] = len(partial_blocks)
                mask_block_idx[seq_idx, h_q, m_local, :len(partial_blocks)] = torch.tensor(
                    partial_blocks, device=device, dtype=torch.int32
                )