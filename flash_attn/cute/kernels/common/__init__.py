"""Device-side building blocks shared across the attention kernels.

- ``softmax`` — online softmax with row_max/row_sum tracking, score-mod support.
- ``mask`` — ``AttentionMask``: causal, local/sliding window, mask_mod application.
- ``block_info`` / ``seqlen_info`` — tile ranges and (varlen) sequence bookkeeping.
- ``pipeline`` — circular-buffer index/phase management for pipelined loads.
- ``tile_scheduler`` — tile scheduling strategies (single tile, varlen-aware, persistent).
- ``copy_utils`` — type-converting copies, shared-to-register loads, TMA copy atoms.
- ``barrier`` / ``named_barrier`` — barrier helpers and named-barrier enums.
- ``pack_gqa`` — packs multiple Q heads per KV head for efficient GQA.
- ``paged_kv`` — ``PagedKVManager``: paged KV cache with TMA support.
- ``topk_gather_kv`` — cp.async gather of top-k KV blocks (sparse MLA).
- ``fast_math`` — exp2 polynomial coefficients.

Modules here are imported directly (no re-exports): they are traced into
kernels, not part of the host-facing API.
"""
