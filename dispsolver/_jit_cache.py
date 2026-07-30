"""
_jit_cache.py
=============
Central on-disk JIT cache configuration for both Numba (@njit) and JAX
(@jax.jit) kernels, with an explicit escape hatch for deployment/CI
environments where a stale on-disk cache would be a correctness risk
(e.g. deploying to a machine with an older cache dir still present, or a
CI runner reusing a Docker layer cache across code changes).

Safety design
-------------
Numba's own cache invalidation (source hash + numba/llvmlite version) and
JAX's persistent compilation cache (jaxpr hash) both auto-invalidate on
most code changes. But neither is airtight for this codebase's pattern of
module-level numeric constants baked into a jitted function's closure
without being traced arguments (e.g. q4_reduced_jax.py's `_ALPHA_HG`,
q4_eas_jax.py's `_ALPHA_MAX`) -- changing one of those doesn't always
change the function's bytecode/jaxpr hash in a way the cache notices.

Controls, all read once at import time:

- `DISPFOLD_JIT_CACHE` (default "1"): set to "0" to disable Numba's
  on-disk caching entirely -- every process start recompiles from
  scratch. Use this in CI/deployment pipelines where reproducible,
  cache-free builds matter more than startup speed, or when debugging a
  suspected stale-cache issue.
- `DISPFOLD_JAX_CACHE` (default "0", i.e. OFF -- opt-in only): JAX's
  persistent compilation cache stores CPU-feature-targeted AOT-compiled
  code (XLA:CPU AOT). Observed in practice: loading a cache entry
  compiled for a different CPU feature set than the executing host logs
  "Target machine feature ... not supported on the host machine ...
  could lead to execution errors such as SIGILL" -- this is an XLA-level
  warning, not a code-version problem, and neither the version tag nor
  the platform tag below catches it (both machines can report the same
  `platform.machine()` and Python version while still having different
  CPU instruction-set extensions, e.g. AVX-512 variants). Numba's cache
  has no equivalent failure mode (it caches LLVM-compiled-for-host code,
  recompiled per-host at first use, not cross-host AOT snapshots), which
  is why it defaults on while this defaults off. Set to "1" only after
  confirming every machine that will load the cache has an identical CPU
  feature set to the one that built it (e.g. a homogeneous deployment
  fleet), or accept the SIGILL risk knowingly.
- `DISPFOLD_CACHE_VERSION` (default "v1"): a manual cache-busting tag.
  The cache directory path includes this string, so bumping it (e.g. to
  "v2") after changing a module-level constant inside a jitted function
  guarantees a fresh cache directory -- no reliance on hash-based
  invalidation catching the change. Bump this whenever you change a
  numeric constant, default parameter, or algorithm inside any
  @njit(cache=True) or @jax.jit function without changing its traced
  argument list.

Usage
-----
    from .._jit_cache import njit_cached, configure_jax_cache
    configure_jax_cache()  # call once, before any jax.jit is defined

    @njit_cached(fastmath=True)
    def my_kernel(...): ...
"""

from __future__ import annotations

import os
import platform
import functools

_CACHE_ENABLED = os.environ.get("DISPFOLD_JIT_CACHE", "1") not in ("0", "false", "False")
_JAX_CACHE_ENABLED = os.environ.get("DISPFOLD_JAX_CACHE", "0") not in ("0", "false", "False")
_CACHE_VERSION = os.environ.get("DISPFOLD_CACHE_VERSION", "v1")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CACHE_ROOT = os.path.join(_REPO_ROOT, ".cache")

_PLATFORM_TAG = f"{platform.system()}-{platform.machine()}-py{platform.python_version()}"


def numba_cache_enabled() -> bool:
    """Whether Numba @njit(cache=True) should actually cache to disk."""
    return _CACHE_ENABLED


def njit_cached(*args, **kwargs):
    """Drop-in replacement for @numba.njit that adds cache=True, gated by
    DISPFOLD_JIT_CACHE. Also sets NUMBA_CACHE_DIR (once, on first import
    of this module) to a version+platform-tagged directory so a manual
    DISPFOLD_CACHE_VERSION bump gets a guaranteed-fresh cache location
    regardless of Numba's own hash-based invalidation.

    Usage identical to numba.njit: @njit_cached(fastmath=True) or bare
    @njit_cached (no parens) both work.
    """
    import numba

    kwargs.setdefault("cache", _CACHE_ENABLED)

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return numba.njit(cache=_CACHE_ENABLED)(args[0])

    def _decorator(fn):
        return numba.njit(*args, **kwargs)(fn)
    return _decorator


_numba_cache_dir_set = False


def _ensure_numba_cache_dir():
    """Point Numba's on-disk cache at our version+platform-tagged directory.

    NUMBA_CACHE_DIR as an environment variable is NOT reliable here:
    numba.config resolves all NUMBA_* env vars once, at `import numba`
    time, and every element module does `import numba` before importing
    this module -- by the time this function runs, numba.config has
    already finished reading (and possibly caching the absence of) the
    env var. Setting os.environ afterward is silently too late. Instead,
    write directly to the already-imported numba.config.CACHE_DIR
    attribute, which every subsequent @njit(cache=True) compile reads
    live (Numba's cache locator consults numba.config.CACHE_DIR at
    compile time, not at import time).
    """
    global _numba_cache_dir_set
    if _numba_cache_dir_set or not _CACHE_ENABLED:
        return
    cache_dir = os.path.join(_CACHE_ROOT, "numba", _CACHE_VERSION, _PLATFORM_TAG)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        import numba
        numba.config.CACHE_DIR = cache_dir
    except ImportError:
        pass
    _numba_cache_dir_set = True


_jax_cache_configured = False


def configure_jax_cache():
    """Enable JAX's persistent compilation cache, version+platform-tagged
    the same way as the Numba cache. Call once, as early as possible
    (before the first @jax.jit function is traced) -- typically at the
    top of dispsolver/solver/dynamic.py or dispsolver/__init__.py.

    No-op if DISPFOLD_JIT_CACHE=0.
    """
    global _jax_cache_configured
    if _jax_cache_configured or not _JAX_CACHE_ENABLED:
        return
    import jax

    cache_dir = os.path.join(_CACHE_ROOT, "jax", _CACHE_VERSION, _PLATFORM_TAG)
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Cache every compiled executable regardless of estimated compile
    # cost/size -- this codebase's kernels are all small-to-medium element
    # routines where even a "cheap" compile is still tens of ms x hundreds
    # of unique (pid, element-type) combinations.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    _jax_cache_configured = True


_ensure_numba_cache_dir()
