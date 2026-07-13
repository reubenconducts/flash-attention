"""Device-side block sparsity.

- ``block_sparse_utils`` — ``BlockSparseTensors`` (the view kernels receive)
  and in-kernel block iteration helpers. The host-side model
  (``BlockSparseTensorsTorch``, normalization, cute conversion) lives in
  ``flash_attn.cute.block_sparsity``.
- ``compute_block_sparsity`` — standalone kernel program computing block
  tensors from a ``mask_mod``.
"""

from flash_attn.cute.kernels.block_sparse.block_sparse_utils import BlockSparseTensors

__all__ = [
    "BlockSparseTensors",
]
