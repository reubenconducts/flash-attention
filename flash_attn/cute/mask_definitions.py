from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch


MaskModCallable = Optional[
    Callable[
        ["cutlass.Int32", "cutlass.Int32", "cutlass.Int32", "cutlass.Int32"],
        "cutlass.Boolean",
    ]
]


# Flex Attention mask functions (PyTorch signatures for reference implementation)


def flex_identity_mask(b, h, q_idx, kv_idx):
    if torch.is_tensor(q_idx):
        return torch.ones_like(q_idx, dtype=torch.bool)
    return True


def flex_identity_partial_mask(b, h, q_idx, kv_idx):
    if torch.is_tensor(q_idx):
        return torch.ones_like(q_idx, dtype=torch.bool)
    return True


def flex_causal_mask(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


def flex_block_causal_mask(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


def flex_sliding_window_mask(b, h, q_idx, kv_idx, window_size=256):
    result = abs(q_idx - kv_idx) <= window_size // 2
    return result


def flex_block_diagonal_mask(b, h, q_idx, kv_idx, block_size=64):
    return (q_idx // block_size) == (kv_idx // block_size)


def flex_mini_causal_mask(b, h, q_idx, kv_idx):
    return (q_idx % 128) >= (kv_idx % 128)


def flex_half_identity_mask(b, h, q_idx, kv_idx):
    """Even k-blocks are full blocks, odd k-blocks are masked blocks (both return True)"""
    if torch.is_tensor(kv_idx):
        return torch.ones_like(kv_idx, dtype=torch.bool)
    return True


# CuTe versions for kernel compilation


@cute.jit
def cute_identity_mask(
    head: cutlass.Int32, batch: cutlass.Int32, m_idx: cutlass.Int32, n_idx: cutlass.Int32
) -> cutlass.Boolean:
    return cutlass.Boolean(True)


@cute.jit
def cute_identity_partial_mask(
    head: cutlass.Int32, batch: cutlass.Int32, m_idx: cutlass.Int32, n_idx: cutlass.Int32
) -> cutlass.Boolean:
    return cutlass.Boolean(True)


@cute.jit
def cute_causal_mask(
    head: cutlass.Int32, batch: cutlass.Int32, m_idx: cutlass.Int32, n_idx: cutlass.Int32
) -> cutlass.Boolean:
    return cutlass.Boolean(n_idx <= m_idx)


@cute.jit
def cute_block_causal_mask(
    head: cutlass.Int32, batch: cutlass.Int32, m_idx: cutlass.Int32, n_idx: cutlass.Int32
) -> cutlass.Boolean:
    return cutlass.Boolean(n_idx <= m_idx)


@cute.jit
def cute_sliding_window_mask(
    head: cutlass.Int32, batch: cutlass.Int32, m_idx: cutlass.Int32, n_idx: cutlass.Int32
) -> cutlass.Boolean:
    return cutlass.Boolean(m_idx - n_idx <= 128 and m_idx - n_idx >= -128)


@cute.jit
def cute_block_diagonal_mask(
    head: cutlass.Int32, batch: cutlass.Int32, m_idx: cutlass.Int32, n_idx: cutlass.Int32
) -> cutlass.Boolean:
    return cutlass.Boolean((m_idx // 64) == (n_idx // 64))


@cute.jit
def cute_mini_causal_mask(
    head: cutlass.Int32, batch: cutlass.Int32, m_idx: cutlass.Int32, n_idx: cutlass.Int32
) -> cutlass.Boolean:
    """Each tile is locally causal-masked"""
    m_mod = m_idx % 128
    n_mod = n_idx % 128
    return cutlass.Boolean(m_mod >= n_mod)


@cute.jit
def cute_half_identity_mask(
    head: cutlass.Int32, batch: cutlass.Int32, m_idx: cutlass.Int32, n_idx: cutlass.Int32
) -> cutlass.Boolean:
    return cutlass.Boolean(True)


MASK_FUNCTIONS = {
    "identity": (cute_identity_mask, flex_identity_mask),
    "identity_partial": (cute_identity_partial_mask, flex_identity_partial_mask),
    "causal": (cute_causal_mask, flex_causal_mask),
    "block_causal": (cute_block_causal_mask, flex_block_causal_mask),
    "sliding_window": (cute_sliding_window_mask, flex_sliding_window_mask),
    "block_diagonal": (cute_block_diagonal_mask, flex_block_diagonal_mask),
    "mini_causal": (cute_mini_causal_mask, flex_mini_causal_mask),
    "half_identity": (cute_half_identity_mask, flex_half_identity_mask),
}
