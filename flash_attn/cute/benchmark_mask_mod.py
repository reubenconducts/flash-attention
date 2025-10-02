"""
FlashAttention benchmarking script with flex_attention reference implementation.
Supports mask_mod functions for sparse attention patterns.
"""

import argparse
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F

from flash_fwd import FlashAttentionForwardSm80, FlashAttentionForwardSm90
from flash_fwd_sm100 import FlashAttentionForwardSm100
from block_sparsity import compute_block_sparsity
from mask_definitions import MASK_FUNCTIONS, flex_causal_mask


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""

    # Model parameters
    headdim: int
    headdim_v: int
    nheads: int
    nheads_kv: int
    dtype: torch.dtype

    # Sequence parameters
    batch_size: int = 2
    seqlen_q: int = 8192
    seqlen_k: int = 8192

    # Mask parameters
    use_mask_mod: bool = False
    mask_mod_name: str = "identity"

    # Attention parameters
    causal: bool = False
    softcap: Optional[float] = None

    # Kernel configuration
    m_block_size: int = 128
    n_block_size: int = 128
    num_stages: int = 2
    num_threads: int = 384
    intra_wg_overlap: bool = True
    mma_pv_is_rs: bool = True

    # Benchmark parameters
    warmup_iters: int = 5
    benchmark_iters: int = 20
    ref_check: bool = False
    atol: float = 1e-2
    verbose: bool = False
    seed: int = 42


class FlashAttentionBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if config.use_mask_mod:
            self.mask_mod_cute, self.mask_mod_flex = MASK_FUNCTIONS[config.mask_mod_name]
        else:
            self.mask_mod_cute = None
            self.mask_mod_flex = None

        self._validate_config()

        if config.verbose:
            self._print_config()

    def _validate_config(self):
        config = self.config

        assert config.headdim <= 256, "headdim must be <= 256"
        assert config.headdim_v <= 256, "headdim_v must be <= 256"
        assert config.nheads % config.nheads_kv == 0, "nheads must be divisible by nheads_kv"

        alignment = 16 // config.dtype.itemsize
        assert config.headdim % alignment == 0, f"headdim must be divisible by {alignment}"
        assert config.headdim_v % alignment == 0, f"headdim_v must be divisible by {alignment}"

    def _print_config(self):
        print(f"\n{'=' * 60}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute Capability: {torch.cuda.get_device_capability()}")
        print(f"{'=' * 60}\n")

    def _create_tensors(self) -> Dict[str, torch.Tensor]:
        config = self.config
        device = "cuda"

        q = torch.randn(
            config.batch_size,
            config.seqlen_q,
            config.nheads,
            config.headdim,
            dtype=config.dtype,
            device=device,
        )
        k = torch.randn(
            config.batch_size,
            config.seqlen_k,
            config.nheads_kv,
            config.headdim,
            dtype=config.dtype,
            device=device,
        )
        v = torch.randn(
            config.batch_size,
            config.seqlen_k,
            config.nheads_kv,
            config.headdim_v,
            dtype=config.dtype,
            device=device,
        )
        out = torch.empty(
            config.batch_size,
            config.seqlen_q,
            config.nheads,
            config.headdim_v,
            dtype=config.dtype,
            device=device,
        )
        lse = torch.empty(
            config.batch_size, config.nheads, config.seqlen_q, dtype=torch.float32, device=device
        )

        tensors = {
            "q": q.contiguous(),
            "k": k.contiguous(),
            "v": v.contiguous(),
            "out": out.contiguous(),
            "lse": lse.contiguous(),
        }

        # Compute block sparsity when using mask_mod
        if config.use_mask_mod:
            full_cnt, full_idx, mask_cnt, mask_idx = compute_block_sparsity(
                config=self.config, mask_mod_flex=self.mask_mod_flex, device=device
            )
            if all(t is not None for t in [full_cnt, full_idx, mask_cnt, mask_idx]):
                tensors["full_block_cnt"] = full_cnt.contiguous()
                tensors["full_block_idx"] = full_idx.contiguous()
                tensors["mask_block_cnt"] = mask_cnt.contiguous()
                tensors["mask_block_idx"] = mask_idx.contiguous()

                if config.verbose:
                    total_full = full_cnt.sum().item()
                    total_partial = mask_cnt.sum().item()
                    n_blocks_k = (config.seqlen_k + config.n_block_size - 1) // config.n_block_size
                    n_blocks_q = (config.seqlen_q + config.m_block_size - 1) // config.m_block_size
                    max_blocks = n_blocks_k * n_blocks_q * config.nheads * config.batch_size
                    skipped = max_blocks - total_full - total_partial
                    print(
                        f"Block stats: Full={total_full}, Partial={total_partial}, "
                        f"Skipped={skipped}/{max_blocks}"
                    )

        return tensors

    def _compile_kernel(self, tensors: Dict[str, torch.Tensor]) -> Tuple[Any, tuple]:
        config = self.config

        dtype_map = {
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
            torch.float32: cutlass.Float32,
        }
        cute_dtype = dtype_map[config.dtype]

        compute_capability = torch.cuda.get_device_capability()
        if compute_capability >= (10, 0):
            kernel_class = FlashAttentionForwardSm100
        elif compute_capability >= (9, 0):
            kernel_class = FlashAttentionForwardSm90
        else:
            kernel_class = FlashAttentionForwardSm80

        qhead_per_kvhead = config.nheads // config.nheads_kv
        kernel = kernel_class(
            cute_dtype,
            config.headdim,
            config.headdim_v,
            qhead_per_kvhead,
            is_causal=config.causal,
            is_local=False,
            pack_gqa=qhead_per_kvhead > 1,
            m_block_size=config.m_block_size,
            n_block_size=config.n_block_size,
            num_stages=config.num_stages,
            num_threads=config.num_threads,
            intra_wg_overlap=config.intra_wg_overlap,
            mma_pv_is_rs=config.mma_pv_is_rs,
            Q_in_regs=False,
        )

        softmax_scale = 1.0 / math.sqrt(config.headdim)
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        q_cute = from_dlpack(tensors["q"].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=tensors["q"].ndim - 1
        )
        k_cute = from_dlpack(tensors["k"].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=tensors["k"].ndim - 1
        )
        v_cute = from_dlpack(tensors["v"].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=tensors["v"].ndim - 1
        )
        out_cute = from_dlpack(tensors["out"].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=tensors["out"].ndim - 1
        )
        lse_cute = from_dlpack(tensors["lse"].detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=tensors["lse"].ndim - 1
        )

        # Block sparsity tensors
        full_block_cnt_cute = (
            from_dlpack(tensors["full_block_cnt"].detach(), assumed_align=4)
            if "full_block_cnt" in tensors
            else None
        )
        full_block_idx_cute = (
            from_dlpack(tensors["full_block_idx"].detach(), assumed_align=4)
            if "full_block_idx" in tensors
            else None
        )
        mask_block_cnt_cute = (
            from_dlpack(tensors["mask_block_cnt"].detach(), assumed_align=4)
            if "mask_block_cnt" in tensors
            else None
        )
        mask_block_idx_cute = (
            from_dlpack(tensors["mask_block_idx"].detach(), assumed_align=4)
            if "mask_block_idx" in tensors
            else None
        )

        if self.mask_mod_cute is not None:
            compiled = cute.compile(
                kernel,
                q_cute,
                k_cute,
                v_cute,
                out_cute,
                lse_cute,
                self.mask_mod_cute,
                False,
                softmax_scale,
                current_stream,
                full_block_cnt=full_block_cnt_cute,
                full_block_idx=full_block_idx_cute,
                mask_block_cnt=mask_block_cnt_cute,
                mask_block_idx=mask_block_idx_cute,
            )
        else:
            compiled = cute.compile(
                kernel,
                q_cute,
                k_cute,
                v_cute,
                out_cute,
                lse_cute,
                None,
                False,
                softmax_scale,
                current_stream,
                full_block_cnt=None,
                full_block_idx=None,
                mask_block_cnt=None,
                mask_block_idx=None,
            )

        args = (
            q_cute,
            k_cute,
            v_cute,
            out_cute,
            lse_cute,
            softmax_scale,
            current_stream,
            full_block_cnt_cute,
            full_block_idx_cute,
            mask_block_cnt_cute,
            mask_block_idx_cute,
        )

        return compiled, args

    def _compute_reference(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        config = self.config

        q = tensors["q"].transpose(1, 2)  # (B, H, Q, D)
        k = tensors["k"].transpose(1, 2)
        v = tensors["v"].transpose(1, 2)

        if config.nheads != config.nheads_kv:
            repeat_factor = config.nheads // config.nheads_kv
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        scale = 1.0 / math.sqrt(config.headdim)

        if config.use_mask_mod:
            mask_fn = self.mask_mod_flex
        elif config.causal:
            mask_fn = flex_causal_mask
        else:
            mask_fn = None

        if mask_fn is not None:
            if config.mask_mod_name == "block_causal":
                # Block-level causal mask
                m_block_size = config.m_block_size
                n_block_size = config.n_block_size
                n_blocks_q = (config.seqlen_q + m_block_size - 1) // m_block_size
                n_blocks_k = (config.seqlen_k + n_block_size - 1) // n_block_size

                mask = torch.zeros(
                    config.seqlen_q, config.seqlen_k, dtype=torch.bool, device=q.device
                )

                for q_block in range(n_blocks_q):
                    q_start = q_block * m_block_size
                    q_end = min((q_block + 1) * m_block_size, config.seqlen_q)
                    for k_block in range(n_blocks_k):
                        if k_block <= q_block:
                            k_start = k_block * n_block_size
                            k_end = min((k_block + 1) * n_block_size, config.seqlen_k)
                            mask[q_start:q_end, k_start:k_end] = True

                attn_mask = (
                    mask.unsqueeze(0).unsqueeze(0).expand(config.batch_size, config.nheads, -1, -1)
                )
                out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
            else:
                block_mask = create_block_mask(
                    mask_fn,
                    B=config.batch_size,
                    H=config.nheads,
                    Q_LEN=config.seqlen_q,
                    KV_LEN=config.seqlen_k,
                ).to(q.device)
                out_ref = flex_attention(q, k, v, block_mask=block_mask, scale=scale)
        else:
            out_ref = F.scaled_dot_product_attention(q, k, v, scale=scale)

        return out_ref.transpose(1, 2).contiguous()  # Back to (B, Q, H, D_v)

    def _check_accuracy(self, tensors: Dict[str, torch.Tensor]) -> Tuple[bool, float]:
        out_ref = self._compute_reference(tensors)
        out_cute = tensors["out"]

        if self.config.verbose:
            print(f"Reference output sample [0,0,0,:5]: {out_ref[0, 0, 0, :5]}")
            print(f"Kernel output sample [0,0,0,:5]: {out_cute[0, 0, 0, :5]}")
            print(f"Reference mean: {out_ref.mean():.6f}, std: {out_ref.std():.6f}")
            print(f"Kernel mean: {out_cute.mean():.6f}, std: {out_cute.std():.6f}")

            # Check for nan/inf
            if torch.isnan(out_cute).any():
                num_nan = torch.isnan(out_cute).sum().item()
                print(f"⚠️  Found {num_nan} nan values")
            if torch.isneginf(out_cute).any():
                num_neginf = torch.isneginf(out_cute).sum().item()
                print(f"⚠️  Found {num_neginf} -inf values")

        abs_diff = torch.abs(out_cute - out_ref)
        max_error = abs_diff.max().item()
        mean_error = abs_diff.mean().item()

        noise_baseline = 2.0 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        allowed_error = noise_baseline + self.config.atol
        passed = max_error <= allowed_error

        if self.config.verbose:
            print(f"Max error: {max_error:.6f}")
            print(f"Mean error: {mean_error:.6f}")
            print(f"Allowed: {allowed_error:.6f}")

        return passed, max_error

    def _calculate_flops(self) -> float:
        config = self.config

        # Estimate sparsity for known mask patterns
        if config.use_mask_mod:
            if config.mask_mod_name in ["identity", "identity_partial"]:
                sparsity_ratio = 1.0
            elif config.mask_mod_name in ["causal", "block_causal"]:
                sparsity_ratio = 0.5
            elif config.mask_mod_name == "sliding_window":
                window_size = 128
                sparsity_ratio = min(1.0, window_size / config.seqlen_k)
            elif config.mask_mod_name == "block_diagonal":
                block_size = 64
                num_blocks = (config.seqlen_k + block_size - 1) // block_size
                sparsity_ratio = 1.0 / num_blocks if num_blocks > 1 else 1.0
            else:
                sparsity_ratio = 1.0
        elif config.causal:
            sparsity_ratio = 0.5
        else:
            sparsity_ratio = 1.0

        num_cells = int(config.seqlen_q * config.seqlen_k * sparsity_ratio)

        if config.headdim == config.headdim_v:
            flops_per_batch = 4 * config.nheads * num_cells * config.headdim
        else:
            flops_per_batch = (
                2 * config.nheads * num_cells * config.headdim
                + 2 * config.nheads * num_cells * config.headdim_v
            )

        return flops_per_batch * config.batch_size

    def benchmark(self) -> Dict[str, Any]:
        config = self.config

        tensors = self._create_tensors()
        compiled_kernel, args = self._compile_kernel(tensors)

        # Warmup
        for _ in range(config.warmup_iters):
            compiled_kernel(
                *args[:7],
                full_block_cnt=args[7],
                full_block_idx=args[8],
                mask_block_cnt=args[9],
                mask_block_idx=args[10],
            )
        torch.cuda.synchronize()

        # Accuracy check
        ref_passed, max_error = None, None
        if config.ref_check:
            print("Checking accuracy...")
            ref_passed, max_error = self._check_accuracy(tensors)
            status = "PASS" if ref_passed else f"FAIL (error: {max_error:.6f})"
            print(f"Accuracy: {status}")

        # Benchmark
        times = []
        for _ in range(config.benchmark_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            compiled_kernel(
                *args[:7],
                full_block_cnt=args[7],
                full_block_idx=args[8],
                mask_block_cnt=args[9],
                mask_block_idx=args[10],
            )
            end.record()
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))

        times_tensor = torch.tensor(times)
        mean_time = times_tensor.mean().item()
        std_time = times_tensor.std().item() if len(times) > 1 else 0.0

        total_flops = self._calculate_flops()
        tflops = total_flops / (mean_time * 1e-3) / 1e12

        bytes_per_element = config.dtype.itemsize
        memory_accessed = (
            config.batch_size * config.seqlen_q * config.nheads * config.headdim * bytes_per_element
            + config.batch_size
            * config.seqlen_k
            * config.nheads_kv
            * config.headdim
            * bytes_per_element
            + config.batch_size
            * config.seqlen_k
            * config.nheads_kv
            * config.headdim_v
            * bytes_per_element
            + config.batch_size
            * config.seqlen_q
            * config.nheads
            * config.headdim_v
            * bytes_per_element
        )
        bandwidth_gbps = memory_accessed / (mean_time * 1e-3) / 1e9

        return {
            "mean_time_ms": mean_time,
            "std_time_ms": std_time,
            "tflops": tflops,
            "bandwidth_gbps": bandwidth_gbps,
            "ref_passed": ref_passed,
            "max_error": max_error,
        }

    def print_results(self, results: Dict[str, Any]):
        config = self.config

        print(f"\n{'=' * 60}")
        print(f"Configuration:")
        print(f"  Shape: B={config.batch_size}, Q={config.seqlen_q}, K={config.seqlen_k}")
        print(
            f"  Model: HD={config.headdim}, NH={config.nheads}, NKV={config.nheads_kv}, Causal={config.causal}"
        )
        print(
            f"  Kernel: M={config.m_block_size}, N={config.n_block_size}, Stages={config.num_stages}"
        )
        if config.use_mask_mod:
            print(f"  Mask: {config.mask_mod_name}")

        print(f"\nPerformance:")
        print(f"  Time: {results['mean_time_ms']:.3f} ± {results['std_time_ms']:.3f} ms")
        print(f"  Throughput: {results['tflops']:.2f} TFLOPS")
        print(f"  Bandwidth: {results['bandwidth_gbps']:.1f} GB/s")

        if results["ref_passed"] is not None:
            status = (
                "PASS" if results["ref_passed"] else f"FAIL (max error: {results['max_error']:.6f})"
            )
            print(f"\nAccuracy: {status}")

        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="FlashAttention Benchmark")

    # Model parameters
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--headdim_v", type=int, default=None)
    parser.add_argument("--nheads", type=int, default=16)
    parser.add_argument("--nheads_kv", type=int, default=16)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )

    # Sequence parameters
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seqlen_q", type=int, default=8192)
    parser.add_argument("--seqlen_k", type=int, default=8192)

    # Attention parameters
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--softcap", type=float, default=None)

    # Mask parameters
    parser.add_argument("--mask-mod", action="store_true")
    parser.add_argument(
        "--mask-mod-name", type=str, default="identity", choices=list(MASK_FUNCTIONS.keys())
    )

    # Kernel config
    parser.add_argument("--m_block", type=int, default=128)
    parser.add_argument("--n_block", type=int, default=128)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--num_threads", type=int, default=384)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--mma_pv_rs", action="store_true")

    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    # Accuracy and output
    parser.add_argument("-r", "--ref-check", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    config = BenchmarkConfig(
        headdim=args.headdim,
        headdim_v=args.headdim_v or args.headdim,
        nheads=args.nheads,
        nheads_kv=args.nheads_kv,
        dtype=dtype_map[args.dtype],
        batch_size=args.batch_size,
        seqlen_q=args.seqlen_q,
        seqlen_k=args.seqlen_k,
        use_mask_mod=args.mask_mod,
        mask_mod_name=args.mask_mod_name,
        causal=args.causal,
        softcap=args.softcap,
        m_block_size=args.m_block,
        n_block_size=args.n_block,
        num_stages=args.num_stages,
        num_threads=args.num_threads,
        intra_wg_overlap=args.overlap,
        mma_pv_is_rs=args.mma_pv_rs,
        warmup_iters=args.warmup,
        benchmark_iters=args.iters,
        ref_check=args.ref_check,
        atol=args.atol,
        verbose=args.verbose,
        seed=args.seed,
    )

    benchmark = FlashAttentionBenchmark(config)
    results = benchmark.benchmark()
    benchmark.print_results(results)


if __name__ == "__main__":
    main()
