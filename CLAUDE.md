# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashAttention-4 (FA4) — fast, memory-efficient exact attention kernels written in Python using CuTeDSL (NVIDIA CUTLASS DSL). Kernels are compiled to PTX/CUBIN at runtime. Targets Hopper (SM90) and Blackwell (SM100/SM110) GPUs. Package name: `flash-attn-4`.

The repository also contains older generations (FA2 in top-level `csrc/`, FA3 in `hopper/`) but active development is on FA4 in `flash_attn/cute/`.

## Agent Scratch Space

Use `agent_space/` for project-local scratch work such as lab notes, profiling outputs, temporary repro scripts, and experiment artifacts. Treat it as disposable workspace rather than product code.

## Build & Install

```bash
pip install flash-attn-4
# or dev install:
pip install -e "flash_attn/cute[dev]"
```

Dependencies: `nvidia-cutlass-dsl>=4.5.2`, `torch`, `einops`, `apache-tvm-ffi`, `quack-kernels>=0.5.0`.

## Running Tests

```bash
pytest tests/cute/test_flash_attn.py
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x  # single test
pytest tests/cute/test_flash_attn_varlen.py
pytest tests/cute/test_mask_mod.py
pytest tests/cute/test_score_mod.py
pytest tests/cute/test_block_sparsity.py
```

### Fast two-pass testing

Compilation dominates test time. The fast workflow separates compilation (parallel, no GPU needed) from execution (uses cached binaries):

```bash
# Pass 1: compile all kernels in parallel using FakeTensorMode (no GPU memory allocation)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: run tests using cached compiled kernels
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -x tests/cute/test_flash_attn.py
```

- `FLASH_ATTENTION_FAKE_TENSOR=1` — uses PyTorch FakeTensorMode to compile kernels without allocating GPU memory or running them.
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` — enables persistent disk cache at `/tmp/${USER}/flash_attention_cute_dsl_cache/`.
- `-n 256` — pytest-xdist parallel workers (only useful in the compilation pass).

Tests are parametrized over dtype (fp16/bf16), head dimension (64, 96, 128), sequence length, causal/non-causal, and MHA/GQA/MQA.

If you get OOM errors running tests or benchmarks, use `nvidia-smi` to find a free GPU and select it with `CUDA_VISIBLE_DEVICES=<id>`.

## Linting

Pre-commit uses ruff on `flash_attn/cute/` files. Large kernel files (`kernels/backward/flash_bwd.py`, `kernels/forward/flash_fwd.py`, `kernels/forward/flash_fwd_sm100.py`, `interface.py`) are excluded from auto-formatting.

```bash
ruff check flash_attn/cute/ --fix
ruff format flash_attn/cute/
```

## Code Architecture

### Public API (`flash_attn/cute/interface.py`)

Two entry points exported from `flash_attn/cute/__init__.py`:
- `flash_attn_func(q, k, v, ...)` — standard attention
- `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)` — variable-length

Key parameters: `causal`, `window_size_left/right`, `softmax_scale`, `softcap`, `score_mod`, `mask_mod`, `block_sparse_tensors`, `num_splits`, `pack_gqa`, `m_block_size`, `n_block_size`, `num_threads`.

Tensor layout: `(batch, seqlen, num_heads, head_dim)`, last dim contiguous, 16-byte aligned.

### Forward Kernels (`flash_attn/cute/kernels/forward/`)

- `flash_fwd.py` — `FlashAttentionForwardSm80` (+ `FlashAttentionForwardBase`): Ampere forward (base).
- `flash_fwd_sm90.py` — `FlashAttentionForwardSm90`: Hopper forward. No SplitKV or paged KV.
- `flash_fwd_sm100.py` — `FlashAttentionForwardSm100`: Blackwell forward. Full features including SplitKV, paged KV cache, persistent kernels, 2CTA instructions.
- `flash_fwd_sm120.py` — `FlashAttentionForwardSm120`: Blackwell GeForce (SM80 MMA, SM120 SMEM).
- `flash_fwd_mla_sm100.py` — `FlashAttentionMLAForwardSm100`: MLA (qv) forward.
- `sm100_hd256_2cta_fmha_forward.py` — dedicated head_dim=256 2CTA forward.
- `flash_fwd_combine.py` — `FlashAttentionForwardCombine`: merges SplitKV partial results.

### Backward Kernels (`flash_attn/cute/kernels/backward/`)

- `flash_bwd.py` — `FlashAttentionBackwardSm80`: Ampere backward (base).
- `flash_bwd_sm90.py` — `FlashAttentionBackwardSm90`: Hopper backward.
- `flash_bwd_sm100.py` — `FlashAttentionBackwardSm100`: Blackwell backward with 2CTA and block sparse support.
- `flash_bwd_sm120.py` — `FlashAttentionBackwardSm120`: Blackwell GeForce backward.
- `flash_bwd_preprocess.py` / `flash_bwd_postprocess.py` — auxiliary backward kernels.
- `flash_bwd_mla_*.py`, `sm100_hd256_2cta_fmha_backward*.py` — sparse MLA and head_dim=256 backward kernels.

### Device-Side Building Blocks (`flash_attn/cute/kernels/common/`)

Everything under `flash_attn/cute/kernels/` traces into the kernels (device code);
the `flash_attn/cute/` top level is host-side machinery and API.

- `softmax.py` — Online softmax with row_max/row_sum tracking, score modifier support.
- `mask.py` — `AttentionMask`: causal, local/sliding window, block sparse, mask_mod application.
- `block_info.py` — `BlockInfo`: tile dimensions, n/m block range computation for causal/local masking.
- `seqlen_info.py` — `SeqlenInfoQK`: sequence length and offset tracking for varlen.
- `pipeline.py` — `PipelineStateSimple`: circular buffer index/phase management for pipelined loads.
- `tile_scheduler.py` — Tile scheduling strategies (single tile, varlen-aware, persistent).
- `copy_utils.py` — Type-converting copies, shared-to-register loads, TMA copy atoms.
- `barrier.py` / `named_barrier.py` — Barrier helpers and named barrier enums for warp synchronization.
- `pack_gqa.py` — Packs multiple Q heads per KV head for efficient GQA.
- `paged_kv.py` — `PagedKVManager`: paged KV cache with TMA support.
- `topk_gather_kv.py` — cp.async gather of top-k KV blocks (sparse MLA).
- `fast_math.py` — exp2 polynomial coefficients.

### Architecture-Specific Helpers (`flash_attn/cute/kernels/arch/`)

- `ampere_helpers.py` — SM80 MMA and smem layout helpers.
- `blackwell_helpers.py` — SM100 UMMA-based GEMM, PTX-optimized paths, 2CTA support.
- `mma_sm100_desc.py` — Hardware MMA descriptor enums (formats, saturation, scaling).
- (SM90 warp-group GEMM helpers come from `quack.sm90_utils`.)

### Block Sparsity

- `flash_attn/cute/block_sparsity.py` — host-side model: `BlockSparseTensorsTorch`,
  normalization, cute conversion, compile-key fingerprints.
- `flash_attn/cute/kernels/block_sparse/block_sparse_utils.py` — device-side:
  `BlockSparseTensors` (what kernels receive) and in-kernel block iteration.
- `flash_attn/cute/kernels/block_sparse/compute_block_sparsity.py` — standalone
  kernel program computing block tensors from a `mask_mod`.

### Host-Side Machinery (`flash_attn/cute/` top level)

- `kernel_compiler.py` — self-describing kernel compilation: per-kernel `Params`/`Args`
  declarations, cache keys, kernel-symbol naming, `run_kernel`.
- `utils.py` — Hash functions for compile cache keys, warp reductions, predicates.
- `cache_utils.py` — JIT compilation cache management.
- `cute_dsl_utils.py` — torch↔cute tensor conversion; patched `cute.compile` that optionally dumps SASS.
- `bench/` — benchmark and tuning tooling (not part of the attention library proper).

### Compilation & Caching

Kernels are JIT-compiled. Cache key includes dtype, head_dim, causal, mask/score_mod hashes, architecture, block sizes. Caching levels: in-memory LRU + optional disk cache via `get_jit_cache()`.

Env vars: `CUTE_CUBIN_PATH` (dump CUBIN/SASS), `CUTE_DSL_KEEP_PTX=1` (inspect PTX), `CUTE_DSL_PTXAS_PATH` (custom ptxas).

## Key Patterns

- Compile-time constants use `cutlass.Constexpr[type]` for kernel specialization.
- Score/mask modifiers are user-defined `@cute.jit` callables injected into the kernel at compile time.
- Forward execution: load Q tile → loop over K/V blocks (pipelined) → online softmax accumulation → store O and LSE.
- 2CTA instructions (SM100, hdim=128): both CTAs in a cluster coordinate via shared mbarriers; tx_count must be multiplied by `cta_group_size`.

## Debugging GPU Kernels

See `AI/DEBUG_2CTA.md` for kernel hang/deadlock debugging (printf bisection, pipeline barrier analysis, 2CTA pitfalls). See `AI/RACECHECK_TMA_HAZARD.md` for `compute-sanitizer` false positives with `cp.async.bulk`. See `AI/CLC_TRACE_DEBUG.md` for visualization of CLC scheduling.

Key tools:
- `cute.printf` with thread guards (`tidx % 32 == 0`, `elect_one()`) for targeted output
- `compute-sanitizer --tool=racecheck` (beware false positives with raw TMA)
- `CUTE_DSL_KEEP_PTX=1` and `CUTE_DSL_LINEINFO=1` for PTX inspection and sanitizer source mapping
