"""Device-side kernel code. Everything under this package traces into kernels
(device code); the ``flash_attn/cute/`` top level is host-side machinery and API.

Subpackages:

- ``forward/`` — forward attention kernels per architecture, plus SplitKV combine.
- ``backward/`` — backward attention kernels plus pre/postprocess.
- ``common/`` — building blocks shared across kernels (softmax, masking,
  scheduling, pipelining, paged KV, ...).
- ``arch/`` — architecture-specific MMA and smem-layout helpers.
- ``block_sparse/`` — device-side block sparsity: the tensor view kernels
  receive, in-kernel block iteration, and the block-tensor computation kernel.
"""
