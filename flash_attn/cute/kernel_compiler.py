# Copyright (c) 2025, Tri Dao.

"""Self-describing kernel compilation and caching.

Establishes one invariant for every FA4 kernel launch path:

    compilation is a pure function of (kernel_cls, params, spec(args))

Kernel classes opt in by declaring two dataclasses next to the kernel class
(see ``check_kernel_contract``, which enforces both mirrors):

- ``Params`` — a frozen dataclass that IS the constructor signature:
  ``__init__(self, params)`` receives it whole, so the parameter list exists
  exactly once (fields, defaults, and naming metadata all on the dataclass).
  ``params.key()`` is the init-part of the cache key: callable fields
  (score_mod / mask_mod) are keyed by ``utils.hash_callable``, every other
  value must be a tensor-free picklable host value (the persistent cache
  pickles keys; a tensor in a key changes its pickle bytes every call and
  silently forces recompiles — the #2507 bug class).
- ``Args`` — a dataclass mirroring ``__call__`` field-for-field, in call order
  (minus the trailing stream). Its *values* are runtime-only; its
  compile-relevant *structure* — presence, dtypes, ranks, keyed broadcast
  patterns, composite fingerprints — is projected out mechanically by
  ``args.spec()`` and forms the call-part of the key. There are no hand-set
  presence bits, so the call-part cannot desync from what is actually passed.

Field declarations carry their own metadata (``sym_val``/``sym_tag`` for
Params, ``fa_tensor``/``scalar``/``composite`` for Args), from which three things
are derived that previously required hand-maintained parallel encodings:
the cache key, the compile/runtime argument lists, and the readable GPU
kernel symbol prefix (``kernel_name_prefix``) stamped onto the CUBIN so
nsys/ncu traces map back to the exact cache entry that produced them.

Compilation modes (``kernel_cls.compile_mode``):

- ``"symbolic"`` — compile against fake tensors built from each Args field's
  shape template plus the runtime tensor's rank/dtype (both keyed via
  ``spec()``); dynamic dims never enter the key.
- ``"real"`` — compile against the caller's tensors via ``to_cute_tensor``
  (transitional).

Call conventions (``kernel_cls.call_convention``):

- ``"flat"`` (default) — ``__call__`` takes one parameter per Args field
  (checked as an exact mirror) plus the trailing stream; the stream is the
  implicit TVM FFI env stream.
- ``"args_struct"`` — ``__call__(self, args, stream)`` takes the bundle whole
  via a typed-NamedTuple carrier generated from the Args declaration, so the
  parameter list exists only once; the stream is an explicit argument (env
  detection cannot see tensors inside the struct). Composite fields must
  declare ``carrier_type``.

The runner compiles lazily and the compiled function is executed immediately
after (unless under FakeTensorMode). Do not batch-precompile ahead of execution
in a live process: a later ``cute.compile`` can invalidate compiled-but-not-yet-
executed functions.
"""

import functools
import inspect
import os
import re
import typing
from contextlib import contextmanager
from dataclasses import MISSING, dataclass, field, fields
from functools import lru_cache
from typing import ClassVar, Optional, Protocol, Union

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute

from flash_attn.cute import fa_logging
from flash_attn.cute.cute_dsl_utils import (
    get_broadcast_dims,
    to_cute_tensor,
    torch2cute_dtype_map,
)
from flash_attn.cute.fa_logging import fa_log
from flash_attn.cute.testing import is_fake_mode
from flash_attn.cute.utils import hash_callable


def parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      FLASH_ATTENTION_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


# ---------------------------------------------------------------------------
# Kernel symbol names
# ---------------------------------------------------------------------------

_DTYPE_SHORT_NAMES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float8_e4m3fn: "e4m3",
    torch.float8_e5m2: "e5m2",
    torch.int32: "i32",
    torch.int64: "i64",
    cutlass.Float16: "f16",
    cutlass.BFloat16: "bf16",
    cutlass.Float32: "f32",
    cutlass.Float8E4M3FN: "e4m3",
    cutlass.Float8E5M2: "e5m2",
    cutlass.Int32: "i32",
    cutlass.Int64: "i64",
}


def short_dtype_name(dtype):
    """Return a compact dtype label suitable for a kernel symbol."""
    name = _DTYPE_SHORT_NAMES.get(dtype)
    if name is None:
        name = str(getattr(dtype, "__name__", dtype)).replace("torch.", "").replace(".", "_")
    return name


# ---------------------------------------------------------------------------
# Params field declarations
# ---------------------------------------------------------------------------

_UNSET = object()


def _make_field(metadata, default):
    if default is _UNSET:
        return field(metadata=metadata)
    return field(default=default, metadata=metadata)


def sym_val(fmt, *, skip=_UNSET, default=_UNSET):
    """A Params field rendered into the kernel symbol by value.

    ``fmt`` is a format string (``"d{}"``) or a callable ``value -> str|None``.
    The part is suppressed when the value is None, equals ``skip``, or —
    for cross-field conditions — when a callable ``skip(params)`` is true
    (e.g. omit head_dim_v when it equals head_dim).
    """
    return _make_field({"sym": ("val", fmt, skip)}, default)


def sym_tag(tag, *, invert=False, default=_UNSET):
    """A Params field rendered as a bare tag when truthy (``invert``: falsy)."""
    return _make_field({"sym": ("tag", tag, invert)}, default)


def sym_expr(fn, *, default=_UNSET):
    """A Params field whose symbol is computed from the whole Params instance —
    for cross-field renders like ``t128x64`` or ``d192x128`` that belong to one
    field but read several. ``fn(params) -> str | None``."""
    return _make_field({"sym": ("expr", fn)}, default)


def no_sym(default=_UNSET):
    """A Params field deliberately absent from the kernel symbol."""
    return _make_field({"sym": None}, default)


def mod_field(tag, default=None):
    """A score/mask-mod callable field: keyed by hash_callable, named by tag."""
    return _make_field({"sym": ("tag", tag, False)}, default)


def _render_sym(symspec, value, params):
    if symspec is None:
        return None
    kind = symspec[0]
    if kind == "expr":
        return symspec[1](params)
    if kind == "val":
        _, fmt, skip = symspec
        # NB: types (e.g. skip=Float32) are callable but are equality sentinels.
        if callable(skip) and not isinstance(skip, type):
            skipped = skip(params)
        else:
            skipped = skip is not _UNSET and value == skip
        if value is None or skipped:
            return None
        return fmt(value) if callable(fmt) else fmt.format(value)
    _, tag, invert = symspec
    return tag if bool(value) != bool(invert) else None


def _canon_key_value(path, v):
    """Canonicalize one Params value for the cache key.

    Callables (score_mod / mask_mod) are keyed by their compile-significant
    hash; types (cutlass dtypes) pass through; torch.Tensors are rejected
    loudly because they pickle differently on every call.
    """
    if isinstance(v, torch.Tensor):
        raise TypeError(
            f"compile key field {path} is a torch.Tensor; keys must be tensor-free "
            "host values (a tensor here forces a recompile on every call)"
        )
    if callable(v) and not isinstance(v, type):
        return ("callable", hash_callable(v))
    if isinstance(v, (tuple, list, frozenset)):
        return tuple(_canon_key_value(f"{path}[{i}]", x) for i, x in enumerate(v))
    if isinstance(v, dict):
        return tuple(sorted((k, _canon_key_value(f"{path}[{k!r}]", x)) for k, x in v.items()))
    return v


@dataclass(frozen=True)
class KernelParams:
    """Base for per-kernel Params dataclasses (the field-for-field mirror of
    ``__init__``; see module docstring)."""

    def as_kwargs(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def key(self) -> tuple:
        return tuple(
            (f.name, _canon_key_value(f"{type(self).__name__}.{f.name}", getattr(self, f.name)))
            for f in fields(self)
        )

    def name_parts(self) -> list:
        parts = [
            _render_sym(f.metadata.get("sym"), getattr(self, f.name), self) for f in fields(self)
        ]
        return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Args field declarations
# ---------------------------------------------------------------------------


class _Static:
    """Sentinel for shape-template entries whose (compile-time static) value is
    read off the runtime tensor — e.g. head_dim as ``t.shape[-1]``. Sound
    because such dims are keyed elsewhere (Params) or fixed by construction."""

    def __repr__(self):
        return "STATIC"


STATIC = _Static()

# A dimension in a symbolic tensor shape: a static int, a shared-sym name, or
# STATIC. Specs using the same name within one compilation share one
# cute.sym_int, encoding cross-tensor shape constraints structurally
# (e.g. LSE's seqlen == O's seqlen).
Dim = Union[int, str, _Static]


def fa_tensor(
    *,
    align=16,
    divisibility=None,
    shape=None,
    leading_dim=-1,
    broadcast_dims=(),
    keyed_broadcast=False,
    detach=False,
    fp8_as_uint8=False,
    convert=None,
    sym=None,
    sym_absent=None,
    sym_dtype=False,
    default=_UNSET,
):
    """Declare a (possibly absent) tensor argument of an Args class.

    align: assumed alignment in bytes (real mode and, unless ``divisibility``
        is given, symbolic mode where it converts to elements via the dtype).
    divisibility: symbolic-mode alignment in elements, overriding ``align``.
    shape: symbolic-mode template of ``Dim`` entries; rank must match the
        runtime tensor.
    leading_dim: index of the static stride-1 dim (-1 = last, None = fully
        dynamic).
    broadcast_dims: dims compiled with static stride 0.
    keyed_broadcast: fingerprint the runtime broadcast pattern into the key.
    detach / fp8_as_uint8: runtime call transforms.
    convert: real-mode compile conversion override, ``convert(t) -> cute
        tensor``, for tensors that need something other than to_cute_tensor
        (e.g. leading-static semaphore layouts).
    sym / sym_absent: kernel-symbol tag emitted when present / absent.
    sym_dtype: also emit the tensor's short dtype name into the kernel symbol
        (for kernels whose __init__ takes no dtype; duplicates are deduped).

    ``shape``, ``leading_dim``, ``broadcast_dims`` and ``divisibility`` may be
    callables ``(args, params) -> value`` for layouts that depend on other
    arguments (e.g. varlen vs batched) or on the kernel specialization
    (e.g. nheads_major).
    """
    return _make_field(
        {
            "kind": "tensor",
            "align": align,
            "divisibility": divisibility,
            "shape": shape,
            "leading_dim": leading_dim,
            "broadcast_dims": broadcast_dims,
            "keyed_broadcast": keyed_broadcast,
            "detach": detach,
            "fp8_as_uint8": fp8_as_uint8,
            "convert": convert,
            "sym": sym,
            "sym_absent": sym_absent,
            "sym_dtype": sym_dtype,
        },
        default,
    )


def scalar(typ, *, sym=None, sym_absent=None, default=_UNSET):
    """Declare a non-tensor argument (softmax_scale, window sizes). Values are
    dynamic at runtime and never enter the cache key (only presence does);
    symbolic mode compiles against a ``typ(0)`` placeholder."""
    return _make_field(
        {"kind": "scalar", "typ": typ, "sym": sym, "sym_absent": sym_absent}, default
    )


def composite(
    *, fingerprint, compile_build, call_build, carrier_type=None, sym=None, default=_UNSET
):
    """Declare a composite argument (DescaleTensors, BlockSparseTensors, AuxData).

    fingerprint(value): the value's compile-relevant structure, entered into
        the key (e.g. sub-tensor presence, aux metadata descriptors).
    compile_build(pool, value) / call_build(value): build the compile-time and
        runtime argument from the field's value; both live next to the kernel
        so composite construction stays colocated. Only invoked when the field
        is not None.
    carrier_type: the compile-side type (what compile_build returns), used as
        the field's type hint when the kernel takes its arguments as one
        struct (call_convention="args_struct").
    """
    return _make_field(
        {
            "kind": "composite",
            "fingerprint": fingerprint,
            "compile_build": compile_build,
            "call_build": call_build,
            "carrier_type": carrier_type,
            "sym": sym,
            "sym_absent": None,
        },
        default,
    )


def _resolve(maybe_callable, args, params):
    return maybe_callable(args, params) if callable(maybe_callable) else maybe_callable


class SymPool:
    """Maps named dims to shared cute.sym_int instances within one compilation."""

    def __init__(self):
        self._syms: dict = {}

    def dim(self, d):
        if isinstance(d, int):
            return d
        if d not in self._syms:
            self._syms[d] = cute.sym_int()
        return self._syms[d]


def symbolic_tensor(dtype, shape, pool, *, divisibility=1, leading_dim=-1, broadcast_dims=()):
    """Build a fake cute tensor for compilation from pure metadata.

    Mirrors ``to_cute_tensor(t).mark_layout_dynamic(leading_dim)``: one static
    stride-1 dim (or none if leading_dim is None), static-0 broadcast dims
    (mark_layout_dynamic keeps stride-0 static, so broadcast patterns are a
    compile-time property), symbolic strides elsewhere. ``divisibility`` is in
    elements; the assumed alignment is ``divisibility * dtype.width // 8`` bytes.
    """
    if dtype is None:
        return None
    if leading_dim is not None and leading_dim < 0:
        leading_dim += len(shape)
    stride = tuple(
        0
        if i in broadcast_dims
        else (1 if i == leading_dim else cute.sym_int64(divisibility=divisibility))
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        tuple(pool.dim(d) for d in shape),
        stride=stride,
        assumed_align=divisibility * dtype.width // 8,
    )


@functools.lru_cache(maxsize=None)
def _args_carrier_cls(args_cls):
    """NamedTuple carrier for kernels declaring ``call_convention =
    "args_struct"``: ``__call__(self, args, stream)`` receives the whole
    bundle as one argument.

    The DSL natively flattens NamedTuple arguments for tracing and TVM FFI
    marshals them at runtime (the DescaleTensors path); the reconstruction
    hook reinserts the None pattern, one MLIR value per present leaf. The
    FFI args-spec converter requires per-field type hints, so the carrier is
    a typed NamedTuple derived from the Args field metadata. Nested
    composites are not yet supported under this convention — they would
    consume several values and misalign the hook."""
    spec = []
    for f in fields(args_cls):
        kind = f.metadata["kind"]
        if kind == "scalar":
            leaf_type = f.metadata["typ"]
        elif kind == "composite":
            leaf_type = f.metadata["carrier_type"]
            assert leaf_type is not None, (
                f"{args_cls.__name__}.{f.name}: composite fields need carrier_type= to be "
                "used under call_convention='args_struct'"
            )
        else:
            leaf_type = cute.Tensor
        spec.append((f.name, Optional[leaf_type]))
    # No custom MLIR-value hooks: the DSL's generic tuple handling flattens the
    # carrier per typed field and reconstructs it inside the traced function,
    # chunking by get_mlir_types() counts (None -> 0, python statics -> 0) and
    # dispatching to nested composites' own hooks.
    return typing.NamedTuple(args_cls.__name__ + "Carrier", spec)


@dataclass
class KernelArgs:
    """Base for per-kernel Args dataclasses (the field-for-field mirror of
    ``__call__``; see module docstring)."""

    @classmethod
    def pick(cls, src, **overrides):
        """Build from any object carrying same-named attributes (e.g. a
        comprehensive per-launcher args bundle), with keyword overrides."""
        kwargs = {f.name: getattr(src, f.name) for f in fields(cls) if f.name not in overrides}
        return cls(**kwargs, **overrides)

    def spec(self) -> tuple:
        """The compile-relevant fingerprint: presence, dtype and rank of each
        tensor (plus its broadcast pattern when keyed), presence of each
        scalar, and each composite's fingerprint. Derived from the same values
        that are marshaled at launch, so key and call cannot desync."""
        entries = []
        for f in fields(self):
            v, m = getattr(self, f.name), f.metadata
            if m["kind"] == "tensor":
                entries.append(
                    (
                        f.name,
                        v is not None,
                        short_dtype_name(v.dtype) if v is not None else None,
                        v.dim() if v is not None else None,
                        get_broadcast_dims(v) if m["keyed_broadcast"] and v is not None else None,
                    )
                )
            elif m["kind"] == "scalar":
                entries.append((f.name, v is not None))
            else:
                entries.append((f.name, m["fingerprint"](v)))
        return tuple(entries)

    def dtype_name_parts(self) -> list:
        """Short dtype names of sym_dtype-flagged tensors; placed right after
        the arch in the kernel symbol (before the Params parts) so the dtype
        sits in the same position whether it comes from Params or Args."""
        return [
            short_dtype_name(getattr(self, f.name).dtype)
            for f in fields(self)
            if f.metadata.get("sym_dtype") and getattr(self, f.name) is not None
        ]

    def name_parts(self) -> list:
        parts = []
        for f in fields(self):
            v, m = getattr(self, f.name), f.metadata
            tag = m["sym"] if v is not None else m["sym_absent"]
            if tag:
                parts.append(tag)
        return parts

    def compile_arguments(self, mode, pool, params=None) -> list:
        """The ``cute.compile`` argument list (minus kernel object and stream)."""
        out = []
        for f in fields(self):
            v, m = getattr(self, f.name), f.metadata
            if m["kind"] == "composite":
                out.append(m["compile_build"](pool, v) if v is not None else None)
            elif m["kind"] == "scalar":
                out.append(None if v is None else (m["typ"](0) if mode == "symbolic" else v))
            elif v is None:
                out.append(None)
            elif mode == "symbolic":
                out.append(self._symbolic_tensor(f, v, pool, params))
            elif m["convert"] is not None:
                out.append(m["convert"](v))
            else:
                leading_dim = _resolve(m["leading_dim"], self, params)
                out.append(
                    to_cute_tensor(
                        v,
                        assumed_align=m["align"],
                        leading_dim=leading_dim if leading_dim is not None else -1,
                        fully_dynamic=leading_dim is None,
                    )
                )
        return out

    def _symbolic_tensor(self, f, v, pool, params):
        m = f.metadata
        shape_template = _resolve(m["shape"], self, params)
        assert shape_template is not None, (
            f"{type(self).__name__}.{f.name}: symbolic compile_mode requires a shape template"
        )
        assert len(shape_template) == v.dim(), (
            f"{type(self).__name__}.{f.name}: shape template {shape_template} has rank "
            f"{len(shape_template)} but the runtime tensor has rank {v.dim()}"
        )
        dtype = torch2cute_dtype_map[v.dtype]
        divisibility = _resolve(m["divisibility"], self, params)
        if divisibility is None:
            divisibility = max(1, m["align"] * 8 // dtype.width)
        shape = tuple(int(v.shape[i]) if d is STATIC else d for i, d in enumerate(shape_template))
        return symbolic_tensor(
            dtype,
            shape,
            pool,
            divisibility=divisibility,
            leading_dim=_resolve(m["leading_dim"], self, params),
            broadcast_dims=_resolve(m["broadcast_dims"], self, params),
        )

    def call_arguments(self) -> list:
        """The positional runtime argument list, from the same fields that
        produced the compile-time arguments (compiled callables are
        positional-only: disk-cache loads return tvm_ffi.Function)."""
        out = []
        for f in fields(self):
            v, m = getattr(self, f.name), f.metadata
            if m["kind"] == "composite":
                out.append(m["call_build"](v) if v is not None else None)
                continue
            if m["kind"] == "tensor" and v is not None:
                if m["detach"]:
                    v = v.detach()
                if m["fp8_as_uint8"] and v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    v = v.view(torch.uint8)
            out.append(v)
        return out


# ---------------------------------------------------------------------------
# Compile / cache / run
# ---------------------------------------------------------------------------


class KernelProtocol(Protocol):
    """What every kernel class provides to be compiled/cached/run by this
    module (the analogue of ``TileSchedulerProtocol`` for tile schedulers).

    Structural and mostly documentary: the fine print that makes the cache key
    trustworthy (field metadata totality, exact ``__init__``/``__call__``
    signatures) is enforced at runtime by ``check_kernel_contract``, which the
    unit tests apply to every kernel class.

    Kernels may additionally define ``can_implement_params(params) ->
    Optional[str]`` (a classmethod returning a rejection reason) to fail fast
    before tracing.
    """

    KERNEL_NAME: ClassVar[str]
    # "real": compile against the call's actual tensor layouts (transitional).
    # "symbolic": compile against fake tensors built from the Args shape
    # templates, so the compiled artifact is a pure function of the cache key.
    compile_mode: ClassVar[str]
    # "args_struct": __call__(self, args, stream) with the Args instance passed
    # whole (as a generated NamedTuple carrier) and the stream explicit.
    # "flat" (transitional): __call__ mirrors the Args fields one by one.
    call_convention: ClassVar[str]
    Params: ClassVar[type[KernelParams]]
    Args: ClassVar[type[KernelArgs]]

    def __init__(self, params): ...

    def __call__(self, args, stream): ...


def kernel_name_prefix(kernel_cls: type[KernelProtocol], params, args) -> str:
    """Readable prefix for CuTeDSL's normally mangled kernel symbol, derived
    from the cache key's own sources so symbol and key cannot disagree."""
    parts = [
        getattr(kernel_cls, "KERNEL_NAME", kernel_cls.__name__),
        f"sm{get_device_arch()}",
        *args.dtype_name_parts(),
        *params.name_parts(),
        *args.name_parts(),
    ]
    return "_".join(dict.fromkeys(parts))


def _is_device_entry(fn):
    """True for @cute.kernel-decorated functions (device entry points).

    Only those handle the DSL's ``_name_prefix`` kwarg — stamping a plain
    @cute.jit helper would forward the kwarg into the Python function during
    tracing. The decorator kind lives in the jit wrapper's closure; be
    conservative if the DSL's internals ever change shape.
    """
    if not callable(fn) or not hasattr(fn, "set_name_prefix"):
        return False
    try:
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
        return cells["executor_name"].cell_contents == "_kernel_helper"
    except (AttributeError, KeyError):
        return False


@contextmanager
def _kernel_symbol_prefix(kernel_cls, prefix):
    """Stamp ``prefix`` onto every @cute.kernel device entry point of the
    class for the duration of one compile (the DSL mangles it into the CUBIN
    symbol, visible in nsys/ncu/SASS). The DSL records the prefix on shared
    state during compilation, so scrub it afterwards."""
    stamped = []
    for name in dir(kernel_cls):
        fn = inspect.getattr_static(kernel_cls, name, None)
        if _is_device_entry(fn):
            fn.set_name_prefix(prefix)
            stamped.append(fn)
    try:
        yield
    finally:
        for fn in stamped:
            fn.set_name_prefix(None)
            dsl = getattr(fn, "__wrapped__", fn).__dict__.get("_dsl_object")
            if dsl is not None and getattr(dsl, "_name_prefix", None):
                dsl._name_prefix = None


def compile_kernel(kernel_cls: type[KernelProtocol], params, args):
    """Compile one kernel specialization. Pure function of
    (kernel_cls, params, spec(args)) when kernel_cls.compile_mode is
    "symbolic"; in "real" mode the args tensors' full layouts are compiled
    against (transitional)."""
    can_implement = getattr(kernel_cls, "can_implement_params", None)
    if can_implement is not None:
        reason = can_implement(params)
        if reason is not None:
            raise RuntimeError(f"{kernel_cls.__name__} cannot be implemented: {reason}")
    kernel = kernel_cls(params)
    mode = getattr(kernel_cls, "compile_mode", "symbolic")
    compile_args = args.compile_arguments(mode, SymPool(), params)
    if getattr(kernel_cls, "call_convention", "flat") == "args_struct":
        # __call__ takes the Args instance whole; scalars must be DSL-typed so
        # the struct flattening treats them as dynamic values, not constexprs.
        compile_args = [
            f.metadata["typ"](v)
            if f.metadata["kind"] == "scalar" and isinstance(v, (bool, int, float))
            else v
            for f, v in zip(fields(args), compile_args)
        ]
        compile_args = [_args_carrier_cls(type(args))(*compile_args)]
        # With every tensor inside the struct, the implicit TVM FFI env-stream
        # cannot bind (detection only scans top-level tensor params), so the
        # stream is an explicit argument for struct-convention kernels.
        stream = cute.runtime.make_fake_stream()
    else:
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    prefix = kernel_name_prefix(kernel_cls, params, args)
    fa_log(1, f"Compiling {prefix}")
    with _kernel_symbol_prefix(kernel_cls, prefix):
        return cute.compile(
            kernel,
            *compile_args,
            stream,
            options="--enable-tvm-ffi",
        )


def cache_key(kernel_cls: type[KernelProtocol], params, args, extra_key=()) -> tuple:
    """The full cache key: the two declared parts plus ambient fields that
    affect codegen for every kernel (compile target arch, device-printf log
    level).

    ``extra_key`` is an escape hatch for host-derived compile-relevant facts
    that are neither ``__init__`` values nor mechanical call structure (e.g.
    the single-block TVM-stride-poisoning guards in the backward launcher,
    which depend on rounded seqlens the tensors alone don't determine). Use
    sparingly: anything here is a hand-maintained key part again.
    """
    return (
        kernel_cls.__name__,
        get_device_arch(),
        fa_logging.get_fa_log_level(),
        params.key(),
        args.spec(),
        *extra_key,
    )


def run_kernel(cache, kernel_cls: type[KernelProtocol], params, args, extra_key=()) -> None:
    """Get-or-compile the specialization described by (params, spec(args)),
    then invoke it with arguments marshaled from the same Args instance
    (skipped under FakeTensorMode)."""
    key = cache_key(kernel_cls, params, args, extra_key)
    if key not in cache:
        cache[key] = compile_kernel(kernel_cls, params, args)
    if not is_fake_mode():
        call_args = args.call_arguments()
        if getattr(kernel_cls, "call_convention", "flat") == "args_struct":
            call_args = [
                _args_carrier_cls(type(args))(*call_args),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            ]
        cache[key](*call_args)


# ---------------------------------------------------------------------------
# Contract checks (call from tests: one line per kernel class)
# ---------------------------------------------------------------------------


def check_kernel_contract(kernel_cls: type[KernelProtocol]) -> None:
    """Verify the declarations that make the cache key trustworthy:

    - The class satisfies ``KernelProtocol`` (KERNEL_NAME, compile_mode,
      call_convention, nested Params/Args).
    - ``__init__`` takes exactly ``(self, params)`` — the Params dataclass is
      the constructor signature, so the parameter list exists once.
    - ``Args`` fields == ``__call__`` parameters (minus the trailing stream),
      same order, same defaults.
    - Every field declares its metadata explicitly (sym or no_sym for Params;
      fa_tensor/scalar/composite for Args), so naming and marshaling are total.
    """
    assert isinstance(getattr(kernel_cls, "KERNEL_NAME", None), str), (
        f"{kernel_cls.__name__} must declare KERNEL_NAME"
    )
    assert getattr(kernel_cls, "compile_mode", None) in ("real", "symbolic"), (
        f"{kernel_cls.__name__} must declare compile_mode ('real' or 'symbolic')"
    )
    assert getattr(kernel_cls, "call_convention", None) in ("flat", "args_struct"), (
        f"{kernel_cls.__name__} must declare call_convention ('args_struct', or 'flat' transitionally)"
    )
    params_cls, args_cls = kernel_cls.Params, kernel_cls.Args
    assert issubclass(params_cls, KernelParams) and issubclass(args_cls, KernelArgs), (
        f"{kernel_cls.__name__}.Params/Args must derive from KernelParams/KernelArgs"
    )

    init_params = [p.name for p in inspect.signature(kernel_cls.__init__).parameters.values()][1:]
    assert init_params == ["params"], (
        f"{kernel_cls.__name__}.__init__ must take exactly (self, params); got {init_params}"
    )
    for f in fields(params_cls):
        assert "sym" in f.metadata, (
            f"{params_cls.__name__}.{f.name} must be declared via sym_val/sym_tag/"
            f"mod_field/no_sym so its kernel-symbol rendering is an explicit choice"
        )

    call_params = list(inspect.signature(kernel_cls.__call__).parameters.values())[1:]
    assert call_params and call_params[-1].name == "stream", (
        f"{kernel_cls.__name__}.__call__ must keep the stream as its last parameter"
    )
    if getattr(kernel_cls, "call_convention", "flat") == "args_struct":
        call_names = [p.name for p in call_params]
        assert call_names == ["args", "stream"], (
            f"{kernel_cls.__name__}.__call__ must take exactly (self, args, stream) "
            f"under call_convention='args_struct'; got {call_names}"
        )
    else:
        _check_mirror(kernel_cls, args_cls, call_params[:-1], "__call__")
    for f in fields(args_cls):
        assert f.metadata.get("kind") in ("tensor", "scalar", "composite"), (
            f"{args_cls.__name__}.{f.name} must be declared via fa_tensor()/scalar()/composite()"
        )


def _check_mirror(kernel_cls, decl_cls, sig_params, target):
    decl_fields = fields(decl_cls)
    decl_names = [f.name for f in decl_fields]
    sig_names = [p.name for p in sig_params]
    assert decl_names == sig_names, (
        f"{decl_cls.__name__} does not mirror {kernel_cls.__name__}.{target}: "
        f"fields {decl_names} vs parameters {sig_names}"
    )
    for f, p in zip(decl_fields, sig_params):
        decl_default = None if f.default is MISSING else (f.default,)
        sig_default = None if p.default is inspect.Parameter.empty else (p.default,)
        assert decl_default == sig_default, (
            f"{decl_cls.__name__}.{f.name} default {decl_default} does not mirror "
            f"{kernel_cls.__name__}.{target}'s default {sig_default}"
        )
