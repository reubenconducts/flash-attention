from typing import Tuple, Optional, Callable
import torch

"""
Computes block sparse masks for use in CuTe DSL Flash Attention Flex Attention
for some common mask_mod functions. To be replaced by a preprocessing kernel eventually
"""

def compute_block_sparsity(
    config, mask_mod_flex: Callable, device: str
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute block sparsity tensors from mask_mod function"""

    if not config.use_mask_mod or mask_mod_flex is None:
        return None, None, None, None

    n_blocks_k = (config.seqlen_k + config.n_block_size - 1) // config.n_block_size
    n_blocks_q = (config.seqlen_q + config.m_block_size - 1) // config.m_block_size
    num_heads = config.nheads_kv

    full_block_cnt = torch.zeros(
        (num_heads, config.batch_size, n_blocks_q), device=device, dtype=torch.int32
    )
    mask_block_cnt = torch.zeros(
        (num_heads, config.batch_size, n_blocks_q), device=device, dtype=torch.int32
    )
    full_block_idx = torch.zeros(
        (num_heads, config.batch_size, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
    )
    mask_block_idx = torch.zeros(
        (num_heads, config.batch_size, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
    )

    # Identity: all blocks are full
    if config.mask_mod_name == "identity":
        k_blocks = torch.arange(n_blocks_k, device=device)
        for q_block in range(n_blocks_q):
            full_block_cnt[:, :, q_block] = n_blocks_k
            full_block_idx[:, :, q_block, :n_blocks_k] = k_blocks

    # Identity partial: all blocks are partial
    elif config.mask_mod_name == "identity_partial":
        k_blocks = torch.arange(n_blocks_k, device=device)
        for q_block in range(n_blocks_q):
            mask_block_cnt[:, :, q_block] = n_blocks_k
            mask_block_idx[:, :, q_block, :n_blocks_k] = k_blocks

    # Block causal: only full blocks below diagonal
    elif config.mask_mod_name == "block_causal":
        k_blocks = torch.arange(n_blocks_k, device=device)
        for q_block in range(n_blocks_q):
            full_indices = k_blocks[k_blocks <= q_block]
            if len(full_indices) > 0:
                full_block_cnt[:, :, q_block] = len(full_indices)
                full_block_idx[:, :, q_block, : len(full_indices)] = full_indices

    # Causal: analytical computation
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

        is_full = (k_ends - 1) <= q_starts
        is_partial = (k_starts <= (q_ends - 1)) & ((k_ends - 1) > q_starts) & ~is_full

        for q_block in range(n_blocks_q):
            full_indices = k_blocks[is_full[q_block]]
            partial_indices = k_blocks[is_partial[q_block]]

            if len(full_indices) > 0:
                full_block_cnt[:, :, q_block] = len(full_indices)
                full_block_idx[:, :, q_block, : len(full_indices)] = full_indices

            if len(partial_indices) > 0:
                mask_block_cnt[:, :, q_block] = len(partial_indices)
                mask_block_idx[:, :, q_block, : len(partial_indices)] = partial_indices

    # Half identity: even k-blocks are full, odd k-blocks are partial
    elif config.mask_mod_name == "half_identity":
        for q_block in range(n_blocks_q):
            full_blocks = [k for k in range(n_blocks_k) if k % 2 == 0]
            partial_blocks = [k for k in range(n_blocks_k) if k % 2 == 1]

            if full_blocks:
                full_block_cnt[:, :, q_block] = len(full_blocks)
                full_block_idx[:, :, q_block, : len(full_blocks)] = torch.tensor(
                    full_blocks, device=device, dtype=torch.int32
                )

            if partial_blocks:
                mask_block_cnt[:, :, q_block] = len(partial_blocks)
                mask_block_idx[:, :, q_block, : len(partial_blocks)] = torch.tensor(
                    partial_blocks, device=device, dtype=torch.int32
                )

    # Block diagonal
    elif config.mask_mod_name == "block_diagonal":
        block_size = 64

        for q_block in range(n_blocks_q):
            q_start = q_block * config.m_block_size
            q_end = min((q_block + 1) * config.m_block_size, config.seqlen_q)

            full_blocks = []
            partial_blocks = []

            for k_block in range(n_blocks_k):
                k_start = k_block * config.n_block_size
                k_end = min((k_block + 1) * config.n_block_size, config.seqlen_k)

                has_any_match = False
                all_match = True

                for q_pos in range(q_start, q_end):
                    for k_pos in range(k_start, k_end):
                        matches = (q_pos // block_size) == (k_pos // block_size)
                        if matches:
                            has_any_match = True
                        else:
                            all_match = False

                if all_match and has_any_match:
                    full_blocks.append(k_block)
                elif has_any_match:
                    partial_blocks.append(k_block)

            if full_blocks:
                full_block_cnt[:, :, q_block] = len(full_blocks)
                full_block_idx[:, :, q_block, : len(full_blocks)] = torch.tensor(
                    full_blocks, device=device, dtype=torch.int32
                )

            if partial_blocks:
                mask_block_cnt[:, :, q_block] = len(partial_blocks)
                mask_block_idx[:, :, q_block, : len(partial_blocks)] = torch.tensor(
                    partial_blocks, device=device, dtype=torch.int32
                )

    # Generic sampling for other patterns
    else:
        qhead_per_kvhead = config.nheads // config.nheads_kv
        for h_kv in range(num_heads):
            h_q = h_kv * qhead_per_kvhead
            for b in range(config.batch_size):
                for q_block in range(n_blocks_q):
                    q_start = q_block * config.m_block_size
                    q_end = min((q_block + 1) * config.m_block_size, config.seqlen_q)

                    full_blocks = []
                    partial_blocks = []

                    for k_block in range(n_blocks_k):
                        k_start = k_block * config.n_block_size
                        k_end = min((k_block + 1) * config.n_block_size, config.seqlen_k)

                        sample_positions = [
                            (q_start, k_start),
                            (q_start, k_end - 1),
                            (q_end - 1, k_start),
                            (q_end - 1, k_end - 1),
                            ((q_start + q_end) // 2, (k_start + k_end) // 2),
                        ]

                        unmasked_count = sum(
                            1
                            for q_pos, k_pos in sample_positions
                            if mask_mod_flex(b, h_q, q_pos, k_pos)
                        )

                        if unmasked_count == len(sample_positions):
                            full_blocks.append(k_block)
                        elif unmasked_count > 0:
                            partial_blocks.append(k_block)

                    if full_blocks:
                        full_block_cnt[h_kv, b, q_block] = len(full_blocks)
                        full_block_idx[h_kv, b, q_block, : len(full_blocks)] = torch.tensor(
                            full_blocks, device=device, dtype=torch.int32
                        )

                    if partial_blocks:
                        mask_block_cnt[h_kv, b, q_block] = len(partial_blocks)
                        mask_block_idx[h_kv, b, q_block, : len(partial_blocks)] = torch.tensor(
                            partial_blocks, device=device, dtype=torch.int32
                        )

    return full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx
