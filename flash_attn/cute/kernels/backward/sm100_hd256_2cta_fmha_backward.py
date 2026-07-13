# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.


"""Fused multi-head attention (FMHA) backward for the SM100 architecture using CUTE DSL.

Constraints:
* Supported head dimensions: 256 only
* mma_tiler_mn must be 64,64
* Batch size must be the same for Q, K, and V tensors
"""

from dataclasses import dataclass
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.typing import Int32

import torch

from flash_attn.cute.kernels.backward.sm100_hd256_2cta_fmha_backward_dqkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDQKernel,
)
from flash_attn.cute.kernels.backward.sm100_hd256_2cta_fmha_backward_dkdvkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
)
from flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned,
    aux_data_bwd_compile_arg,
    aux_data_bwd_fingerprint,
    aux_data_call_arg,
)
from flash_attn.cute.kernel_compiler import (
    KernelArgs,
    KernelParams,
    composite,
    fa_tensor,
    mod_field,
    no_sym,
    scalar,
    sym_expr,
    sym_tag,
    sym_val,
)
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    block_sparse_call_arg,
    block_sparse_compile_arg,
    block_sparse_fingerprint,
)
from flash_attn.cute.kernels.backward.flash_bwd import _hdim_sym, _semaphore_arg
from flash_attn.cute.kernels.block_sparse.block_sparse_utils import BlockSparseTensors
from flash_attn.cute.utils import AuxData, as_bshkrd_tensor, as_shhb_tensor


class BlackwellFusedMultiHeadAttentionBackward:
    """FMHA backward class for executing CuTeDSL kernel."""

    KERNEL_NAME = "flash_bwd_hd256"
    compile_mode = "real"
    call_convention = "args_struct"

    @dataclass(frozen=True)
    class Params(KernelParams):
        """Compile-time specialization: mirrors __init__ field-for-field.

        dtype is inferred from the tensors and keyed via Args.spec()."""

        head_dim: int = sym_expr(_hdim_sym)
        head_dim_v: Optional[int] = no_sym(default=None)
        is_causal: bool = sym_tag("causal", default=False)
        is_local: bool = sym_tag("local", default=False)
        qhead_per_kvhead: int = sym_val("gqa{}", skip=1, default=1)
        is_persistent: bool = sym_tag("persistent", default=False)
        deterministic: bool = sym_tag("det", default=False)
        cluster_size: int = sym_val("cluster{}", skip=1, default=1)
        use_2cta_instrs: bool = sym_tag("2cta", default=False)
        score_mod: Optional[Callable] = mod_field("score_mod")
        score_mod_bwd: Optional[Callable] = mod_field("score_mod_bwd")
        mask_mod: Optional[Callable] = mod_field("mask_mod")
        has_aux_tensors: bool = sym_tag("aux", default=False)
        q_subtile_factor: int = sym_val("qsub{}", skip=1, default=1)
        tile_m_dq: int = sym_expr(lambda p: f"tdq{p.tile_m_dq}x{p.tile_n_dq}", default=128)
        tile_n_dq: int = no_sym(default=128)
        tile_m_dkdv: int = sym_expr(lambda p: f"tdkv{p.tile_m_dkdv}x{p.tile_n_dkdv}", default=128)
        tile_n_dkdv: int = no_sym(default=64)
        window_size_left: Optional[int] = no_sym(default=None)
        window_size_right: Optional[int] = no_sym(default=None)
        use_clc_scheduler: bool = sym_tag("clc", default=False)

    @dataclass
    class Args(KernelArgs):
        """Runtime __call__ arguments: mirrors __call__ (minus stream)
        field-for-field. Same order as FlashAttentionBackwardSm80.Args, so the
        launcher constructs both positionally."""

        Q: torch.Tensor = fa_tensor(detach=True, keyed_broadcast=True, sym_dtype=True)
        K: torch.Tensor = fa_tensor(detach=True, keyed_broadcast=True)
        V: torch.Tensor = fa_tensor(detach=True, keyed_broadcast=True)
        dO: torch.Tensor = fa_tensor(keyed_broadcast=True)
        lse_log2: torch.Tensor = fa_tensor()
        dpsum: torch.Tensor = fa_tensor()
        dQ_accum: Optional[torch.Tensor] = fa_tensor()
        dK: torch.Tensor = fa_tensor()
        dV: torch.Tensor = fa_tensor()
        scale_softmax: float = scalar(Float32)
        cumulative_s_q: Optional[torch.Tensor] = fa_tensor(align=4, sym="varlen")
        cumulative_s_k: Optional[torch.Tensor] = fa_tensor(align=4, sym="varlen")
        seqused_q: Optional[torch.Tensor] = fa_tensor(align=4, sym="varlen", default=None)
        seqused_k: Optional[torch.Tensor] = fa_tensor(align=4, sym="varlen", default=None)
        window_size_left: Optional[int] = scalar(Int32, default=None)
        window_size_right: Optional[int] = scalar(Int32, default=None)
        dQ_semaphore: Optional[torch.Tensor] = fa_tensor(convert=_semaphore_arg, default=None)
        dK_semaphore: Optional[torch.Tensor] = fa_tensor(convert=_semaphore_arg, default=None)
        dV_semaphore: Optional[torch.Tensor] = fa_tensor(convert=_semaphore_arg, default=None)
        aux_data: AuxData = composite(
            fingerprint=aux_data_bwd_fingerprint,
            compile_build=aux_data_bwd_compile_arg,
            call_build=aux_data_call_arg,
            carrier_type=AuxData,
            default=AuxData(),
        )
        block_sparse_tensors: Optional[BlockSparseTensorsTorch] = composite(
            fingerprint=block_sparse_fingerprint,
            compile_build=block_sparse_compile_arg,
            call_build=block_sparse_call_arg,
            carrier_type=BlockSparseTensors,
            sym="blksparse",
            default=None,
        )

    def __init__(self, params: "BlackwellFusedMultiHeadAttentionBackward.Params"):
        """Initialization."""
        head_dim_v = params.head_dim if params.head_dim_v is None else params.head_dim_v
        assert params.head_dim == 256 and head_dim_v == 256, (
            "SM100 dedicated backward kernel only supports (params.head_dim, head_dim_v) = (256, 256)"
        )
        assert not params.is_local, (
            "SM100 backward with head_dim=256 does not support local attention"
        )
        assert params.tile_m_dq == 128 and params.tile_n_dq == 128, (
            "SM100 dedicated backward kernel only supports tile_m_dq=128 and tile_n_dq=128"
        )
        assert params.tile_m_dkdv == 128 and params.tile_n_dkdv == 64, (
            "SM100 dedicated backward kernel only supports tile_m_dkdv=128 and tile_n_dkdv=64"
        )
        assert (
            params.score_mod is None and params.score_mod_bwd is None and params.mask_mod is None
        ), "SM100 backward with head_dim=256 does not support params.score_mod/params.mask_mod"
        assert not params.deterministic, (
            "SM100 backward with head_dim=256 does not support params.deterministic mode"
        )
        assert not params.has_aux_tensors, (
            "SM100 backward with head_dim=256 does not support aux_tensors"
        )
        assert params.cluster_size in (1, 2), (
            "SM100 backward with head_dim=256 only supports params.cluster_size in {1, 2}"
        )
        assert params.use_2cta_instrs, (
            "SM100 backward with head_dim=256 requires use_2cta_instrs=True"
        )
        # params.q_subtile_factor is accepted for interface parity with FlashAttentionBackwardSm100,
        # but this dedicated kernel uses fixed internal behavior.

        self.acc_dtype = cutlass.Float32
        self.is_causal = params.is_causal
        self.window_size_left = (
            None
            if (params.window_size_left is None or params.window_size_left < 0)
            else params.window_size_left
        )
        self.window_size_right = (
            None
            if (params.window_size_right is None or params.window_size_right < 0)
            else params.window_size_right
        )
        self.tile_m_dq = params.tile_m_dq
        self.tile_n_dq = params.tile_n_dq
        self.tile_m_dkdv = params.tile_m_dkdv
        self.tile_n_dkdv = params.tile_n_dkdv
        self.use_clc_scheduler = params.use_clc_scheduler

        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            self.acc_dtype,
            (self.tile_m_dq, self.tile_n_dq, 256),
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            False,  # params.is_persistent
            False,  # split_head
            use_clc_scheduler=self.use_clc_scheduler,
        )
        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            self.acc_dtype,
            (self.tile_m_dkdv, self.tile_n_dkdv, 256),
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            use_clc_scheduler=self.use_clc_scheduler,
        )

    @cute.jit
    def __call__(
        self,
        args,  # Args carrier (see kernel_compiler._args_carrier_cls)
        # Always keep stream as the last parameter.
        stream: cuda.CUstream = None,
    ):
        Q = args.Q
        K = args.K
        V = args.V
        dO = args.dO
        lse_log2 = args.lse_log2
        dpsum = args.dpsum
        dQ_accum = args.dQ_accum
        dK = args.dK
        dV = args.dV
        scale_softmax = args.scale_softmax
        cumulative_s_q = args.cumulative_s_q
        cumulative_s_k = args.cumulative_s_k
        seqused_q = args.seqused_q
        seqused_k = args.seqused_k
        window_size_left = args.window_size_left
        window_size_right = args.window_size_right
        dQ_semaphore = args.dQ_semaphore
        dK_semaphore = args.dK_semaphore
        dV_semaphore = args.dV_semaphore
        aux_data = args.aux_data
        block_sparse_tensors = args.block_sparse_tensors
        """Host function to launch CuTeDSL kernel."""
        assert seqused_q is None and seqused_k is None, (
            "SM100 backward with head_dim=256 does not support seqused_q/seqused_k"
        )
        assert window_size_left is None and window_size_right is None, (
            "SM100 backward with head_dim=256 uses constructor-provided window sizes"
        )
        assert dQ_semaphore is None and dK_semaphore is None and dV_semaphore is None, (
            "SM100 backward with head_dim=256 does not use semaphores"
        )
        assert block_sparse_tensors is None, (
            "SM100 backward with head_dim=256 does not support block sparse tensors"
        )
        assert aux_data.tensors is None or len(aux_data.tensors) == 0, (
            "SM100 backward with head_dim=256 does not support aux_tensors"
        )
        assert aux_data.scalars is None or len(aux_data.scalars) == 0, (
            "SM100 backward with head_dim=256 does not support aux_scalars"
        )
        assert dQ_accum is not None, (
            "SM100 backward with head_dim=256 expects dQ tensor at dQ_accum slot"
        )
        dQ = dQ_accum
        varlen = cumulative_s_q is not None or cumulative_s_k is not None
        q_rank = cute.rank(Q.layout)
        k_rank = cute.rank(K.layout)
        if cutlass.const_expr(q_rank == 5):
            h_q = Q.shape[2] * Q.shape[3]
        elif cutlass.const_expr(q_rank == 4):
            h_q = Q.shape[2]
        else:
            h_q = Q.shape[1]
        if cutlass.const_expr(k_rank == 5):
            h_k = K.shape[2]
        elif cutlass.const_expr(k_rank == 4):
            h_k = K.shape[2]
        else:
            h_k = K.shape[1]
        h_r = h_q // h_k
        if cutlass.const_expr(cumulative_s_q is not None):
            b = cumulative_s_q.shape[0] - 1
        elif cutlass.const_expr(cumulative_s_k is not None):
            b = cumulative_s_k.shape[0] - 1
        else:
            b = Q.shape[0]

        Q, K, V, dQ, dK, dV, dO = [assume_tensor_aligned(t) for t in (Q, K, V, dQ, dK, dV, dO)]

        Q = as_bshkrd_tensor(Q, h_k, h_r, varlen)
        K = as_bshkrd_tensor(K, h_k, 1, varlen)
        V = as_bshkrd_tensor(V, h_k, 1, varlen)
        dQ = as_bshkrd_tensor(dQ, h_k, h_r, varlen)
        dK = as_bshkrd_tensor(dK, h_k, 1, varlen)
        dV = as_bshkrd_tensor(dV, h_k, 1, varlen)
        dO = as_bshkrd_tensor(dO, h_k, h_r, varlen)
        scaled_LSE = as_shhb_tensor(lse_log2, h_k, h_r, b, varlen)
        sum_OdO = as_shhb_tensor(dpsum, h_k, h_r, b, varlen)

        # Keep original order: dQ first, then dKdV.
        self.dq_kernel(
            Q,
            K,
            V,
            dQ,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            stream,
        )
        self.dkdv_kernel(
            Q,
            K,
            V,
            dK,
            dV,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            stream,
        )
