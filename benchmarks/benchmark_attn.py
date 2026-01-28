from collections import namedtuple
from functools import partial
import math
import operator
import os
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import cutlass.cute as cute
import cutlass
from cutlass.cute import Float32
from flash_attn.cute.seqlen_info import SeqlenInfoQK


import time

try:
    import cudnn
except ImportError:
    cudnn = None
# cudnn = None

Timing = NamedTuple("timing", [("mean", float)])


from einops import rearrange, repeat

# from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.cute.benchmark import (
    benchmark_forward,
    benchmark_backward,
    benchmark_combined,
    benchmark_all,
    benchmark_fwd_bwd,
    pytorch_profiler,
)

try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None
from flash_attn.cute.interface import flash_attn_func as flash_attn_func_python
from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_python

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    flash_attn_func_v3 = None
    flash_attn_varlen_func_v3 = None

if torch.cuda.get_device_capability()[0] != 9:
    flash_attn_func_v3 = None
# flash_attn_func_v3 = None

flash_attn_func = None

from triton.testing import do_bench


def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    # # Warmup
    # for _ in range(5):
    #     func(*args, **kwargs)
    # time.sleep(1)
    # return benchmark_forward(func, *args, **kwargs, repeats=repeats, verbose=verbose, desc=desc)[1]
    # s = torch.cuda.Stream()
    # s.wait_stream(torch.cuda.current_stream())
    # with torch.cuda.stream(s):
    #     for _ in range(2):
    #         out = func(*args, **kwargs)
    # torch.cuda.current_stream().wait_stream(s)
    # graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(graph):
    #     out = func(*args, **kwargs)
    # time_f = benchmark_forward(lambda: graph.replay(), repeats=repeats, verbose=verbose, desc=desc)
    # # return time_f[1].mean
    # return time_f[1]
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=5, rep=repeats) * 1e-3)


def flops(
    batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(None, None)
):
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (None, None):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device="cuda")
            col_left = (
                torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0))
                if window_size[0] is not None
                else torch.zeros_like(row_idx)
            )
            col_right = (
                torch.minimum(
                    row_idx + seqlen_k - seqlen_q + window_size[1], torch.tensor(seqlen_k - 1)
                )
                if window_size[1] is not None
                else torch.full_like(row_idx, seqlen_k - 1)
            )
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")


def get_fa4_score_mod(rel_extent: int, vec_size: int):
    import cutlass.cute as cute
    from cutlass.cute import Float32
    from flash_attn.cute.seqlen_info import SeqlenInfoQK

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        rel_logits = aux_tensors[0]

        # seqlen_q: total Q length for this sequence
        # seqlen_k: total KV length for this sequence
        # seqlen_local_offset: index of the first Q token in the sequence
        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q

        # q_idx: index of the Q token in the query chunk
        # kv_idx: index of the KV token in the sequence
        # rel_dist: distance between the Q token and the KV token
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx

        # offset_q: offset of the query chunk in the ragged batch
        # q_idx: index of the Q token in the query chunk
        # global_q_idx: index of the Q token in the ragged batch
        global_q_idx = seqlen_info.offset_q + q_idx

        # FA4 score mod always uses vec_size = 1
        rel_dist_0 = rel_dist[0]

        # Clamp index to valid range, ternary is fast in cute
        rel_idx = rel_dist_0 if rel_dist_0 >= 0 else 0
        rel_idx = rel_idx if rel_idx < rel_extent else (rel_extent - 1)

        # Load rel_bias from rel_logits
        rel_bias = rel_logits[global_q_idx[0], h_idx[0], rel_idx]  # pyright: ignore

        # Only apply rel_bias if the index is valid
        rel_bias = Float32(rel_bias) if rel_dist_0 == rel_idx else Float32(0.0)

        return scores + rel_bias

    return score_mod_rel_bias


"""
For vectorizing this rel bias:
    need to check if kv_idx[0] is oob. oob means
    - < -vec_size or
    - > rel_extent
    then can ensure kv_idx - i reads in bounds for all i < vec_size.
"""


def get_fa4_score_mod_vec(rel_extent: int, vec_size: int):
    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        mBias = aux_tensors[0]
        vec_size = cute.size(kv_idx.shape)

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        q_idx0 = q_idx[0]
        kv_idx0 = kv_idx[0]
        h_idx0 = h_idx[0]
        global_q_idx = seqlen_info.offset_q + q_idx0
        rel_bias = cute.make_rmem_tensor(kv_idx.shape, dtype=Float32)
        dist0 = (q_idx0 + seqlen_local_offset) - kv_idx0
        in_bounds = dist0 >= vec_size and dist0 < rel_extent + vec_size
        if in_bounds:
            mBias_cur = mBias[global_q_idx, h_idx0, None]
            mBias_cur = cute.make_tensor(
                mBias_cur.iterator,
                cute.make_layout(
                    (cute.size(mBias_cur.shape) // vec_size, vec_size), stride=(vec_size, 1)
                ),
            )
            gBias = mBias_cur[kv_idx0, None]
            rel_logits = cute.make_rmem_tensor_like(gBias)
            cute.autovec_copy(gBias, rel_logits)

            for i in cutlass.range_constexpr(vec_size):
                rel_bias_i = rel_logits[vec_size - i - 1]
                rel_bias[i] = Float32(rel_bias_i)
        else:
            rel_bias.fill(0.0)
        return scores + rel_bias.load()

    score_mod_rel_bias.__vec_size__ = vec_size

    return score_mod_rel_bias


def get_causal_mask_mod(vec_size):
    @cute.jit
    def score_mod(
        scores,
        b_idx,
        h_idx,
        q_idx,
        kv_idx,
        seqlen_info,
        aux_tensors,
    ):
        mask = cute.make_rmem_tensor(kv_idx.shape, dtype=cutlass.Boolean)
        kv_idx0 = kv_idx[0]
        for i in cutlass.range_constexpr(cute.size(mask.shape)):
            mask[i] = q_idx[0] >= kv_idx0 + i
        mask_ssa = mask.load()
        return cute.where(mask_ssa, scores, cute.full_like(scores, float(0.0)))

    score_mod.__vec_size__ = vec_size

    return score_mod


def get_kv_rel_bias_score_mod(vec_size):
    @cute.jit
    def kv_rel_bias_score_mod(
        scores,
        b_idx,
        h_idx,
        q_idx,
        kv_idx,
        seqlen_info,
        aux_tensors,
    ):
        q_idx0 = q_idx[0]
        kv_idx0 = kv_idx[0]
        offset = kv_idx0 - q_idx0
        vec_size = cute.size(kv_idx.shape)
        rBias = cute.make_rmem_tensor((vec_size,), dtype=cutlass.Float32)
        for i in cutlass.range(vec_size, unroll_full=True):
            rBias[i] = Float32(offset + i)

        return scores + rBias.load()

    kv_rel_bias_score_mod.__vec_size__ = vec_size

    return kv_rel_bias_score_mod


def get_kv_bias_score_mod(vec_size):
    @cute.jit
    def kv_bias_score_mod(
        scores,
        b_idx,
        h_idx,
        q_idx,
        kv_idx,
        seqlen_info,
        aux_tensors,
    ):
        mBias = aux_tensors[0]
        kv_idx0 = kv_idx[0]
        q_idx0 = q_idx[0]
        offset = kv_idx0 - q_idx0
        vec_size = cute.size(kv_idx.shape)
        # rRelBias = cute.make_rmem_tensor(kv_idx.shape, dtype=cutlass.Float32)
        # for i in cutlass.range(vec_size, unroll_full=True):
        #     rRelBias[i] = Float32(offset + i)
        mBias = cute.make_tensor(
            mBias.iterator,
            layout=cute.make_ordered_layout((mBias.shape[0] // vec_size, vec_size), order=(1, 0)),
        )
        rBias = cute.make_rmem_tensor(kv_idx.shape, dtype=mBias.element_type)
        cute.autovec_copy(mBias[kv_idx0 // vec_size, None], rBias)

        return scores + rBias.load()  # + rRelBias.load()

    kv_bias_score_mod.__vec_size__ = vec_size

    return kv_bias_score_mod


def get_dual_buffer_score_mod(vec_size):
    @cute.jit
    def score_mod_dual_buffer_vectorized(
        tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    ):
        head_bias = aux_tensors[0]
        pos_bias = aux_tensors[1]
        dtype = head_bias.element_type

        head_val_frag = cute.make_fragment(1, dtype)
        head_val_frag[0] = head_bias[h_idx[0]]
        head_val = (head_val_frag.load()).to(cutlass.Float32)

        pos_val_frag = cute.make_fragment(1, dtype)
        pos_val_frag[0] = pos_bias[q_idx[0]]
        pos_val = (pos_val_frag.load()).to(cutlass.Float32)

        return tSrS_ssa + head_val + pos_val

    score_mod_dual_buffer_vectorized.__vec_size__ = vec_size

    return score_mod_dual_buffer_vectorized


def cudnn_spda_setup(q, k, v, causal=False, window_size_left=None):
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    headdim_v = v.shape[-1]
    assert v.shape == (b, nheads_k, seqlen_k, headdim_v)
    assert cudnn is not None, "CUDNN is not available"
    q_gpu, k_gpu, v_gpu = q, k, v
    o_gpu = torch.empty((b, nheads, seqlen_q, headdim_v), dtype=q.dtype, device=q.device)
    stats_gpu = torch.empty(b, nheads, seqlen_q, 1, dtype=torch.float32, device=q.device)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())

    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        is_inference=False,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left is not None,
        use_causal_mask=causal or window_size_left is not None,
        sliding_window_length=window_size_left
        if window_size_left is not None and not causal
        else None,
    )

    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        stats: stats_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return o_gpu

    return run


def cudnn_spda_bwd_setup(q, k, v, o, g, lse, causal=False, window_size_left=None):
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    headdim_v = v.shape[-1]
    assert v.shape == (b, nheads_k, seqlen_k, headdim_v)
    assert g.shape == (b, nheads, seqlen_q, headdim_v)
    assert o.shape == (b, nheads, seqlen_q, headdim_v)
    assert lse.shape == (b, nheads, seqlen_q, 1)
    assert cudnn is not None, "CUDNN is not available"
    q_gpu, k_gpu, v_gpu, o_gpu, g_gpu = q, k, v, o, g
    dq_gpu = torch.empty_like(q_gpu)
    dk_gpu = torch.empty_like(k_gpu)
    dv_gpu = torch.empty_like(v_gpu)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())
    o = graph.tensor_like(o_gpu.detach())
    g = graph.tensor_like(g_gpu.detach())
    stats = graph.tensor_like(lse.detach())

    dq, dk, dv = graph.sdpa_backward(
        name="sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=o,
        dO=g,
        stats=stats,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left is not None,
        use_causal_mask=causal or window_size_left is not None,
        sliding_window_length=window_size_left
        if window_size_left is not None and not causal
        else None,
        use_deterministic_algorithm=False,
    )

    dq.set_output(True).set_dim(dq_gpu.shape).set_stride(dq_gpu.stride())
    dk.set_output(True).set_dim(dk_gpu.shape).set_stride(dk_gpu.stride())
    dv.set_output(True).set_dim(dv_gpu.shape).set_stride(dv_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        g: g_gpu,
        stats: lse,
        dq: dq_gpu,
        dk: dk_gpu,
        dv: dv_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return dq_gpu, dk_gpu, dv_gpu

    return run


def jagged_shift_pad(t: torch.Tensor, offset: int) -> torch.Tensor:
    """
    t: shape (D0, D1, ..., D_{n-2}, C)
    returns: (D0, D1, ..., D_{n-2}, C + offset)
    shift for slices with dim0=i is (i % offset), applied to the last dim.
    """
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if offset == 0:
        return t

    *prefix, C = t.shape
    D0 = prefix[0] if prefix else 1  # if t is 1D, treat as D0=1

    out = t.new_zeros((*prefix, C + offset))

    # shifts depends only on dim0
    shifts0 = torch.arange(D0, device=t.device) % offset  # (D0,)

    # reshape for broadcasting to prefix dims
    # shifts shape: (D0, 1, 1, ..., 1)
    view = (D0,) + (1,) * (t.ndim - 2) + (1,)
    shifts = shifts0.view(*view)

    # indices along last dim for each element: shifts + k
    k = torch.arange(C, device=t.device).view((1,) * (t.ndim - 1) + (C,))
    idx = shifts + k  # shape (*prefix, C)
    idx = idx.expand(*prefix, C)  # ensure expanded

    out.scatter_(dim=t.ndim - 1, index=idx, src=t)
    return out


torch.manual_seed(0)
repeats = 10
dropout_p = 0.0
causal = False
dtype = torch.bfloat16
# dtype = torch.float8_e4m3fn
dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
device = "cuda"
verbose = True
varlen = True
has_backward = False
page_size = None
# page_size = 128
softcap = 0.0
V_colmajor = False
deterministic = False
batch_size = 2
# seqlen = 2048
seqlen = 8192
# seqlen = 4096
# seqlen = 2047
dim = 2048
# headdim = 128
# headdim = 64
headdim = 256
# for headdim in [64, 128, 256]:
# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
# bs_seqlen_vals = [(32, 1024), (16, 2048), (8, 4096), (4, 8192), (2, 16384), (1, 32768)]
# bs_seqlen_vals = [(32, 512), (16, 1024)]
# bs_seqlen_vals = [(2, 64 * 132)]
bs_seqlen_vals = [(2, 16384)]  # , (4, 16384), (4, 32 * 1024)]
# bs_seqlen_vals = [(1, 16 * 1024)]
time_f = {}
time_b = {}
rel_extent = 512
vec_size = 8
vec_size_vals = [1, 2, 4, 8, 16, 32, 64, 128]
# vec_size_vals = [1]
check_reference = True

score_mod_factory = partial(get_fa4_score_mod_vec, rel_extent=rel_extent)
# score_mod_factory = get_causal_mask_mod
# score_mod_factory = get_kv_bias_score_mod
# score_mod_factory = get_kv_rel_bias_score_mod
# for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192]:
# for headdim in [64, 96, 128, 192, 256]:
# for headdim in [64, 96, 128]:
# for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192, 256]:
for headdim in [128]:
    # nheads = dim // headdim
    nheads = 32 if headdim <= 64 else 16 if headdim <= 192 else 8
    # nheads = 128
    # headdim = 64
    # batch_size = 64
    # seqlen = 512
    # nheads = 8
    # headdim = 128
    nheads_kv = nheads
    # nheads_kv = nheads // 8
    # nheads_kv = 1
    # headdim_v = headdim
    headdim_v = 128 if headdim == 192 else headdim
    # headdim_v = 512
    has_qv = headdim == 64 and headdim_v == 512
    # has_qv = False
    # sinks = torch.randn(nheads, dtype=torch.bfloat16, device=device)
    sinks = None

    for batch_size, seqlen in bs_seqlen_vals:
        num_splits = 0
        # window_size = (-1, -1)
        window_size = (None, None)
        window_size_fa = (-1, -1)
        # window_size = (seqlen // 2 - 1, 0)
        pack_gqa = None
        # seqlen_q = 64
        seqlen_q = seqlen
        leftpad_k = None
        # leftpad_k = torch.full((batch_size,), 0, device=device, dtype=torch.int32)
        q = torch.randn(
            batch_size,
            seqlen_q,
            nheads,
            headdim,
            device=device,
            dtype=dtype_gen,
            requires_grad=has_backward,
        )
        k = torch.randn(
            batch_size,
            seqlen,
            nheads_kv,
            headdim,
            device=device,
            dtype=dtype_gen,
            requires_grad=has_backward,
        )
        v = torch.randn(
            batch_size,
            seqlen,
            nheads_kv,
            headdim_v,
            device=device,
            dtype=dtype_gen,
            requires_grad=has_backward,
        )
        q, k, v = [x.detach().to(dtype).requires_grad_(has_backward) for x in [q, k, v]]
        v_colmajor = (
            v.detach().transpose(-1, -3).contiguous().transpose(-1, -3).requires_grad_(has_backward)
        )
        v_fa3 = v if not V_colmajor else v_colmajor
        qv = (
            torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen)
            if has_qv
            else None
        )
        bias = torch.randn(
            batch_size, seqlen_q, nheads, rel_extent, device=device, dtype=torch.bfloat16
        )
        bias_jagged = jagged_shift_pad(bias, vec_size - 1)
        # q = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # k = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # v = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim_v), device=device, dtype=torch.int32).to(dtype)
        g = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen)
        o = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen)
        stats = torch.randn(batch_size, seqlen_q, nheads, 1, device=device, dtype=torch.float32)
        if varlen:
            q_unpad, k_unpad, v_unpad, bias_unpad, bias_jagged_unpad = [
                rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_(has_backward)
                for x in [q, k, v, bias, bias_jagged]
            ]
            cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q
            cu_seqlens_k = (
                torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen
                if page_size is None
                else None
            )
            # cu_seqlens_q = torch.tensor([0, 248, 249, 250, 251, 252, 253, 254, 255, 256], device=device, dtype=torch.int32)
            # q_unpad = q_unpad[:256]
            # seqlen_q = 256
            # cu_seqlens_q = torch.tensor([0, 376, 377, 378, 379, 380, 381, 382, 383, 384], device=device, dtype=torch.int32)
            # q_unpad = q_unpad[:384]
            # seqlen_q = 384
        if page_size is not None:
            assert seqlen % page_size == 0
            k_paged, v_paged = [
                rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k, v]
            ]
            page_table = rearrange(
                torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                "(b s) -> b s",
                s=seqlen // page_size,
            )
        else:
            page_table = None

        # bias = torch.randn(
        #     seqlen, device=device, dtype=torch.bfloat16
        # )
        bias_jagged_unpad.__assumed_align__ = 16
        bias_jagged_unpad.__leading_dim__ = -1

        for causal in [True]:
            # for causal in [True]:
            print(
                f"\n### {headdim = }, {causal = }, {seqlen_q = }, {seqlen = }, {batch_size = }, {nheads = }, {varlen = } ###"
            )
            out_ref = None
            nFLOPS = flops(
                batch_size,
                nheads,
                seqlen_q,
                seqlen,
                headdim if not has_qv else headdim + headdim_v,
                headdim_v,
                causal=causal,
                window_size=window_size,
            )
            if cudnn is not None:
                # if False:
                if headdim <= 256 and dtype != torch.float8_e4m3fn:
                    cudnn_spda = cudnn_spda_setup(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        causal=causal,
                        window_size_left=window_size[0],
                    )
                    if has_backward and headdim == headdim_v:
                        cudnn_spda_bwd = cudnn_spda_bwd_setup(
                            q.transpose(1, 2),
                            k.transpose(1, 2),
                            v.transpose(1, 2),
                            o.transpose(1, 2),
                            g.transpose(1, 2),
                            stats.transpose(1, 2),
                            causal=causal,
                            window_size_left=window_size[0],
                        )
            if (
                dtype != torch.float8_e4m3fn
                and headdim == headdim_v
                and flash_attn_func is not None
            ):
                # if False:
                if not varlen:
                    m0 = time_fwd(
                        flash_attn_func,
                        q,
                        k,
                        v,
                        dropout_p,
                        causal=causal,
                        window_size=window_size,
                        softcap=softcap,
                        repeats=repeats,
                        verbose=verbose,
                        desc="Fav2",
                    )
                else:
                    m0 = time_fwd(
                        flash_attn_varlen_func,
                        q_unpad,
                        k_unpad,
                        v_unpad,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        seqlen_q,
                        seqlen,
                        dropout_p,
                        causal=causal,
                        window_size=window_size,
                        softcap=softcap,
                        repeats=repeats,
                        verbose=verbose,
                        desc="Fav2",
                    )
                time_f[(causal, headdim, batch_size, seqlen), "Flash2"] = m0.mean
                if has_backward:
                    time.sleep(1)
                    if not varlen:
                        _, m0b = benchmark_backward(
                            flash_attn_func,
                            q,
                            k,
                            v,
                            dropout_p,
                            causal=causal,
                            window_size=window_size,
                            softcap=softcap,
                            deterministic=deterministic,
                            repeats=repeats,
                            verbose=False,
                            desc="Fav2",
                        )
                    else:
                        _, m0b = benchmark_backward(
                            flash_attn_varlen_func,
                            q_unpad,
                            k_unpad,
                            v_unpad,
                            cu_seqlens_q,
                            cu_seqlens_k,
                            seqlen_q,
                            seqlen,
                            dropout_p,
                            causal=causal,
                            window_size=window_size,
                            softcap=softcap,
                            deterministic=deterministic,
                            repeats=repeats,
                            verbose=False,
                            desc="Fav2",
                        )
                    time_b[(causal, headdim, batch_size, seqlen), "Flash2"] = m0b.mean
            # pytorch_profiler(flash_attn_func, q, k, v, dropout_p, causal=causal, backward=True)

            if cudnn is not None:
                # if False:
                if headdim <= 256 and dtype != torch.float8_e4m3fn:
                    time.sleep(
                        1
                    )  # Sleep to avoid residual power throttling from the previous benchmark
                    m2 = time_fwd(cudnn_spda, repeats=repeats, verbose=verbose, desc="CuDNN")
                    time_f[(causal, headdim, batch_size, seqlen), "cuDNN"] = m2.mean
                    if has_backward:
                        time.sleep(1)
                        m2b = time_fwd(
                            cudnn_spda_bwd, repeats=repeats, verbose=verbose, desc="CuDNN"
                        )
                        time_b[(causal, headdim, batch_size, seqlen), "cuDNN"] = m2b.mean
                # pytorch_profiler(cudnn_spda, backward=False)
                # pytorch_profiler(cudnn_spda_bwd, backward=False)
            time.sleep(1)
            if flash_attn_func_v3 is not None:
                if not varlen:
                    # m1 = time_fwd(flash_attn_func_v3, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, cache_leftpad = leftpad_k, page_table=page_table, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3')
                    m1 = time_fwd(
                        flash_attn_func_v3,
                        q,
                        k if page_size is None else k_paged,
                        v_fa3 if page_size is None else v_paged,
                        causal=causal,
                        window_size=window_size_fa,
                        softcap=softcap,
                        num_splits=num_splits,
                        pack_gqa=pack_gqa,
                        repeats=repeats,
                        verbose=verbose,
                        desc="Fav3",
                    )
                    # pytorch_profiler(flash_attn_func_v3, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, page_table=page_table, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa)
                else:
                    m1 = time_fwd(
                        flash_attn_varlen_func_v3,
                        q_unpad,
                        k_unpad,
                        v_unpad,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        seqlen_q,
                        seqlen,
                        causal=causal,
                        window_size=window_size_fa,
                        softcap=softcap,
                        num_splits=num_splits,
                        pack_gqa=pack_gqa,
                        repeats=repeats,
                        verbose=verbose,
                        desc="Fav3",
                    )
                    # pytorch_profiler(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits)
                time_f[(causal, headdim, batch_size, seqlen), "Flash3"] = m1.mean
            if flash_attn_func_python is not None:
                if not varlen:
                    if check_reference:
                        pass
                    m1_py = time_fwd(
                        flash_attn_func_python,
                        q,
                        k if page_size is None else k_paged,
                        v_fa3 if page_size is None else v_paged,
                        causal=causal,
                        window_size=window_size,
                        learnable_sink=sinks,
                        softcap=softcap,
                        pack_gqa=pack_gqa,
                        repeats=repeats,
                        verbose=verbose,
                        desc="Fav3 python",
                    )
                else:
                    out, _ = flash_attn_varlen_func_python(
                        q_unpad,
                        k_unpad if page_size is None else k_paged,
                        v_unpad if page_size is None else v_paged,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        page_table=page_table,
                        causal=causal,
                        window_size=window_size,
                        softcap=softcap,
                        pack_gqa=pack_gqa,
                        score_mod=score_mod_factory(vec_size=vec_size),
                        aux_tensors=[bias_jagged_unpad],
                    )
                    if check_reference:
                        out_ref, _ = flash_attn_varlen_func_python(
                            q_unpad,
                            k_unpad if page_size is None else k_paged,
                            v_unpad if page_size is None else v_paged,
                            cu_seqlens_q,
                            cu_seqlens_k,
                            page_table=page_table,
                            causal=causal,
                            window_size=window_size,
                            softcap=softcap,
                            pack_gqa=pack_gqa,
                            score_mod=get_fa4_score_mod(rel_extent=rel_extent, vec_size=vec_size),
                            aux_tensors=[bias_unpad],
                        )
                        torch.testing.assert_allclose(out_ref, out)
                        if not torch.equal(out_ref, out):
                            print(f"For vec_size {vec_size}, output is not equal to reference.")
                    m1_py = time_fwd(
                        flash_attn_varlen_func_python,
                        q_unpad,
                        k_unpad if page_size is None else k_paged,
                        v_unpad if page_size is None else v_paged,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        page_table=page_table,
                        causal=causal,
                        window_size=window_size,
                        softcap=softcap,
                        pack_gqa=pack_gqa,
                        score_mod=score_mod_factory(vec_size=vec_size),
                        aux_tensors=[bias_jagged_unpad],
                        repeats=repeats,
                        verbose=verbose,
                        desc="Fav3 python",
                    )
            if (
                dtype != torch.float8_e4m3fn
                and headdim == headdim_v
                and flash_attn_func_v3 is not None
                and has_backward
            ):
                time.sleep(1)
                if not varlen:
                    _, m1b = benchmark_backward(
                        flash_attn_func_v3,
                        q,
                        k,
                        v,
                        causal=causal,
                        softcap=softcap,
                        repeats=repeats,
                        verbose=False,
                        desc="Fav3",
                    )
                else:
                    _, m1b = benchmark_backward(
                        flash_attn_varlen_func_v3,
                        q_unpad,
                        k_unpad,
                        v_unpad,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        seqlen_q,
                        seqlen,
                        causal=causal,
                        window_size=window_size,
                        softcap=softcap,
                        deterministic=deterministic,
                        repeats=repeats,
                        verbose=False,
                        desc="Fav3",
                    )
                time_b[(causal, headdim, batch_size, seqlen), "Flash3"] = m1b.mean
                time.sleep(1)
                # if not varlen:
                #     pytorch_profiler(flash_attn_func_v3, q, k, v, causal=causal, deterministic=deterministic, backward=True)
                # else:
                #     pytorch_profiler(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, deterministic=deterministic, backward=True)
            # benchmark_forward(torch.clone, k, repeats=repeats, verbose=verbose, desc='Memcpy')
            if (
                dtype != torch.float8_e4m3fn
                and headdim == headdim_v
                and flash_attn_func_python is not None
                and has_backward
            ):
                if not varlen:
                    _, m1b_py = benchmark_backward(
                        flash_attn_func_python,
                        q,
                        k,
                        v,
                        causal=causal,
                        softcap=softcap,
                        deterministic=deterministic,
                        repeats=repeats,
                        verbose=False,
                        desc="Fav4 python",
                    )
                else:
                    _, m1b_py = benchmark_backward(
                        flash_attn_varlen_func_python,
                        q_unpad,
                        k_unpad,
                        v_unpad,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        causal=causal,
                        softcap=softcap,
                        deterministic=deterministic,
                        repeats=repeats,
                        verbose=False,
                        desc="Fav4 python",
                    )

            if (
                dtype != torch.float8_e4m3fn
                and headdim == headdim_v
                and flash_attn_func is not None
            ):
                # if False:
                print(f"FAv2 fwd: {m0.mean * 1e3:.3f}ms, {(nFLOPS / m0.mean * 1e-12):.1f} TFLOPS")
                if has_backward:
                    print(
                        f"FAv2 bwd: {m0b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m0b.mean * 1e-12):.1f} TFLOPS"
                    )
            if cudnn is not None:
                print(f"CuDNN fwd: {m2.mean * 1e3:.3f}ms, {(nFLOPS / m2.mean * 1e-12):.1f} TFLOPS")
                if has_backward:
                    print(
                        f"CuDNN bwd: {m2b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m2b.mean * 1e-12):.1f} TFLOPS"
                    )
            if flash_attn_func_v3 is not None:
                print(f"FAv3 fwd: {m1.mean * 1e3:.3f}ms, {(nFLOPS / m1.mean * 1e-12):.1f} TFLOPS")
                if dtype != torch.float8_e4m3fn and headdim == headdim_v and has_backward:
                    print(
                        f"FAv3 bwd: {m1b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m1b.mean * 1e-12):.1f} TFLOPS"
                    )

            if flash_attn_func_python is not None:
                print(
                    f"FA Python fwd with vec_size {vec_size}: {m1_py.mean * 1e3:.3f}ms, {(nFLOPS / m1_py.mean * 1e-12):.1f} TFLOPS"
                )
                if dtype != torch.float8_e4m3fn and headdim == headdim_v and has_backward:
                    print(
                        f"FA Python bwd: {m1b_py.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m1b_py.mean * 1e-12):.1f} TFLOPS"
                    )
