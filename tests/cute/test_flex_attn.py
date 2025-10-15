"""
Tests for FlexAttention (mask_mod + score_mod with block sparsity and arbitrary buffer tensors)
in CuTe DSL
"""

import math
import operator

import cuda.bindings.driver as cuda
import cutlass
import cutlass
from cutlass._mlir.dialects import math as mlir_math
import cutlass.cute as cute
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import pytest
import torch
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import flex_attention
import torch.nn.functional as F

from flash_attn.cute.block_sparsity import compute_block_sparsity
from flash_attn.cute.flash_fwd import (
    FlashAttentionForwardSm80,
    FlashAttentionForwardSm90,
)
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.mask_definitions import MASK_FUNCTIONS, flex_causal_mask
from test_score_mod import (
    score,
    score_mod_1,
    score_mod_10,
    score_mod_11,
    score_mod_2,
    score_mod_3,
    score_mod_4,
    score_mod_5,
    score_mod_6,
    score_mod_7,
    score_mod_8,
    score_mod_9,
)


def main():
    raise NotImplementedError()


if __name__ == "__main__":
    main()
