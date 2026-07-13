"""Architecture-specific helpers.

- ``ampere_helpers`` — SM80 MMA and smem layout helpers.
- ``blackwell_helpers`` — SM100 UMMA-based GEMM, PTX-optimized paths, 2CTA support.
- ``mma_sm100_desc`` — hardware MMA descriptor enums (formats, saturation, scaling).

(SM90 warp-group GEMM helpers come from ``quack.sm90_utils``.)
"""
