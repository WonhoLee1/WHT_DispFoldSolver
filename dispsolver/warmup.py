"""Pre-compile every Numba kernel of the package into the on-disk JIT cache.

    python -m dispsolver.warmup                 # compile everything in the manifest (this machine)
    python -m dispsolver.warmup --if-needed     # same, but return at once when the cache is current
    python -m dispsolver.warmup --status        # what is covered, what is new, what is stale
    python -m dispsolver.warmup --record        # run the kernel tests, refresh jit_manifest.json
    python -m dispsolver.warmup --record-run -- examples/ex13_unified_model_io.py ...   # record a REAL run
    python -m dispsolver.warmup --clean         # delete this machine's cache directory, then warm

Why: a Numba kernel compiles on its first call (seconds to minutes for the big 3D ones). After a
move to another machine, a Python / Numba / NumPy change, or a new kernel, the first solve is silent
for minutes and looks like a hang. A cache that does not match the machine is rebuilt in-line.

How it finds kernels without hand-written inputs:
  * discovery  -- every module under ``dispsolver`` is imported and every module-level Numba
    dispatcher is collected, so a kernel added tomorrow is found with no registration;
  * signatures -- ``jit_manifest.json`` (committed) holds the argument types each kernel was seen
    with; warming calls ``dispatcher.compile(types)``, which compiles and writes the cache without
    running the kernel (no inputs, no side effects, nothing to hang on);
  * ``--record`` runs the kernel tests in-process and writes down what was actually compiled, so new
    code is covered as soon as it has a test. ``--status`` lists discovered kernels that have no
    recorded signature ("uncovered") so a gap is visible rather than silent.

Not covered: JAX (its persistent cache is opt-in, DISPFOLD_JAX_CACHE=1, and its functions do not
expose a signature to record) and dispatchers created inside factory functions (not module-level).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import inspect
import json
import os
import pkgutil
import sys
import time
from typing import Dict, List, Optional

from . import _jit_cache

PACKAGE = "dispsolver"
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jit_manifest.json")
_SKIP_PARTS = {"cli", "postprocess", "postprocess3d", "export", "warmup"}   # GUI / viewer / this module

_DEFAULT_RECORD_PATTERNS = (
    "test_*3d*.py", "test_*2d*.py", "test_*element*.py", "test_cpe4*.py", "test_c3d*.py",
    "test_*numba*.py", "test_hourglass*.py", "test_*hybrid*.py", "test_section*.py",
    "test_*material*.py", "test_plastic*.py",
)


# --------------------------------------------------------------------------------------- discovery
def import_all() -> Dict[str, str]:
    """Import every non-GUI module of the package. Returns {module: error} for the ones that failed."""
    errors: Dict[str, str] = {}
    pkg = importlib.import_module(PACKAGE)
    for info in pkgutil.walk_packages(pkg.__path__, PACKAGE + ".", onerror=lambda n: errors.setdefault(n, "walk")):
        parts = set(info.name.split("."))
        if parts & _SKIP_PARTS or info.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except Exception as exc:  # a module that needs an optional dependency must not stop the warmup
            errors[info.name] = f"{type(exc).__name__}: {exc}"
    return errors


def discover() -> Dict[str, object]:
    """{'module:qualname': dispatcher} for every module-level Numba dispatcher under the package."""
    from numba.core.dispatcher import Dispatcher
    found: Dict[str, object] = {}
    for modname, mod in list(sys.modules.items()):
        if mod is None or not (modname == PACKAGE or modname.startswith(PACKAGE + ".")):
            continue
        for obj in list(vars(mod).values()):
            if isinstance(obj, Dispatcher):
                py = obj.py_func
                found.setdefault(f"{py.__module__}:{py.__qualname__}", obj)
    return found


# ----------------------------------------------------------------- numba types <-> JSON descriptors
def _describe(t) -> Optional[dict]:
    from numba.core import types
    if isinstance(t, types.Array):
        return {"k": "array", "dtype": str(t.dtype), "ndim": t.ndim, "layout": t.layout,
                "ro": not t.mutable, "aligned": bool(t.aligned)}
    if isinstance(t, types.Omitted):
        return {"k": "omitted"}
    if isinstance(t, types.NoneType):
        return {"k": "none"}
    if isinstance(t, (types.Boolean, types.Integer, types.Float, types.Complex)):
        return {"k": "scalar", "t": str(t)}
    if isinstance(t, (types.UniTuple, types.Tuple)):
        items = [_describe(x) for x in t.types]
        return None if any(i is None for i in items) else {"k": "tuple", "items": items}
    return None   # dispatcher / function / record / literal types: not reproducible, signature skipped


def _scalar(name: str):
    from numba.core import types
    return types.boolean if name == "bool" else getattr(types, name)


def _build(desc: dict, default):
    from numba.core import types
    k = desc["k"]
    if k == "array":
        return types.Array(_scalar(desc["dtype"]), desc["ndim"], desc["layout"],
                           readonly=desc["ro"], aligned=desc["aligned"])
    if k == "scalar":
        return _scalar(desc["t"])
    if k == "none":
        return types.none
    if k == "omitted":
        return types.Omitted(default)
    if k == "tuple":
        items = [_build(i, None) for i in desc["items"]]
        return types.UniTuple(items[0], len(items)) if len(set(items)) == 1 else types.Tuple(items)
    raise ValueError(f"unknown descriptor {desc!r}")


def _defaults(dispatcher) -> list:
    return [p.default for p in inspect.signature(dispatcher.py_func).parameters.values()]


# ---------------------------------------------------------------------------------------- manifest
def load_manifest(path: str = MANIFEST_PATH) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"numba": None, "kernels": {}}


manifest_sha = _jit_cache.manifest_sha


def collect_signatures(found: Dict[str, object]) -> Dict[str, List[List[dict]]]:
    out: Dict[str, List[List[dict]]] = {}
    for key, d in sorted(found.items()):
        sigs = []
        for sig in d.signatures:
            descs = [_describe(t) for t in sig]
            if all(x is not None for x in descs) and descs not in sigs:
                sigs.append(descs)
        if sigs:
            out[key] = sigs
    return out


# ---------------------------------------------------------------------------------------- warming
def _compile_keys(keys: List[str], kernels: Dict[str, List[List[dict]]]) -> dict:
    """Compile the manifest signatures of ``keys`` in this process (modules already imported)."""
    found = discover()
    done = failed = 0
    errors: List[str] = []
    missing: List[str] = []
    for key in keys:
        d = found.get(key)
        if d is None:
            missing.append(key)
            continue
        defaults = _defaults(d)
        for descs in kernels[key]:
            try:
                argtypes = tuple(_build(x, defaults[i] if i < len(defaults) else None)
                                 for i, x in enumerate(descs))
                if argtypes in d.signatures:
                    continue
                d.compile(argtypes)
                done += 1
            except Exception as exc:
                failed += 1
                errors.append(f"{key}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
    return {"done": done, "failed": failed, "errors": errors, "missing": missing}


def _worker(args):
    keys, kernels = args
    import_all()
    return _compile_keys(keys, kernels)


def warm(jobs: int = 1, verbose: bool = True) -> int:
    manifest = load_manifest()
    kernels = manifest.get("kernels", {})
    if not kernels:
        print("[warmup] jit_manifest.json is empty -- run `python -m dispsolver.warmup --record` first.")
        return 1
    t0 = time.time()
    errs = import_all()
    keys = sorted(kernels)
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        chunks = [keys[i::jobs] for i in range(jobs)]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            parts = list(ex.map(_worker, [(c, {k: kernels[k] for k in c}) for c in chunks]))
        res = {"done": sum(p["done"] for p in parts), "failed": sum(p["failed"] for p in parts),
               "errors": sum((p["errors"] for p in parts), []), "missing": sum((p["missing"] for p in parts), [])}
    else:
        res = _compile_keys(keys, kernels)
    if verbose:
        print(f"[warmup] {len(keys)} kernels, compiled {res['done']} new signature(s), {res['failed']} failed, "
              f"{len(res['missing'])} in the manifest but no longer in the code, "
              f"{len(errs)} module(s) not importable, {time.time() - t0:.1f} s")
        for e in res["errors"][:10]:
            print("  FAILED", e)
        for m in res["missing"][:10]:
            print("  STALE ", m)
        for m, e in list(errs.items())[:10]:
            print("  IMPORT", m, "->", e)
    if res["failed"] == 0:
        _jit_cache.write_warm_stamp(manifest_sha())
    return 0 if res["failed"] == 0 else 2


# ----------------------------------------------------------------------------------------- status
def status() -> int:
    manifest = load_manifest()
    kernels = manifest.get("kernels", {})
    errs = import_all()
    found = discover()
    uncovered = sorted(k for k in found if k not in kernels)
    stale = sorted(k for k in kernels if k not in found)
    nsig = sum(len(v) for v in kernels.values())
    print(f"[status] discovered {len(found)} kernels | manifest {len(kernels)} kernels / {nsig} signatures "
          f"(numba {manifest.get('numba')}, running {_jit_cache.numba_version()})")
    print(f"[status] uncovered (no recorded signature -- add a test, then --record): {len(uncovered)}")
    for k in uncovered[:40]:
        print("   ", k)
    print(f"[status] stale (in manifest, not in code): {len(stale)}")
    for k in stale[:20]:
        print("   ", k)
    for m, e in errs.items():
        print("[status] not importable:", m, "->", e)
    reason = _jit_cache.warm_state_reason(manifest_sha())
    print("[status] this machine's cache:", reason or "current")
    return 0


# ----------------------------------------------------------------------------------------- record
def _merge_and_write(fresh: bool, label: str, rc) -> None:
    import_all()   # modules the workload pulled in lazily
    found = discover()
    new = collect_signatures(found)
    old = {} if fresh else load_manifest().get("kernels", {})
    merged = dict(old)
    added = 0
    for key, sigs in new.items():
        have = merged.setdefault(key, [])
        for s_ in sigs:
            if s_ not in have:
                have.append(s_)
                added += 1
    manifest = {"numba": _jit_cache.numba_version(), "kernels": dict(sorted(merged.items()))}
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=False)
        fh.write(os.linesep)
    uncovered = sorted(k for k in found if k not in merged)
    print(f"[{label}] rc={rc}; +{added} signature(s); manifest {len(merged)} of {len(found)} discovered kernels, "
          f"{sum(len(v) for v in merged.values())} signatures -> {MANIFEST_PATH}")
    print(f"[{label}] discovered but still uncovered: {len(uncovered)}")


def record(pytest_args: List[str], fresh: bool) -> int:
    import pytest
    import_all()
    if not pytest_args:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
        files = sorted({f for pat in _DEFAULT_RECORD_PATTERNS for f in glob.glob(os.path.join(root, pat))})
        pytest_args = ["-q", "--no-header", "-p", "no:cacheprovider", *files]
    rc = pytest.main(pytest_args)
    _merge_and_write(fresh, "record", rc)
    return 0


def record_run(script_and_args: List[str], fresh: bool) -> int:
    """Run a real analysis script in-process and record the kernel signatures it actually compiled.

    This is the faithful source: the signatures a production solve needs are exactly the ones it
    compiles (array layouts, defaulted arguments and all). Use short runs, e.g. the first steps of
    ex12 / ex13 / ex14 -- compilation happens on the first call, not at step 100.
    """
    import runpy
    if not script_and_args:
        print("[record-run] give a script: --record-run examples/ex13_unified_model_io.py [args]")
        return 1
    script, *argv = script_and_args
    import_all()
    sys.argv = [script, *argv]
    rc = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        rc = exc.code or 0
    except Exception as exc:   # a run that aborts late still compiled what it used
        rc = f"{type(exc).__name__}: {exc}"
    _merge_and_write(fresh, "record-run", rc)
    return 0


# -------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dispsolver.warmup", description=__doc__.split("\n\n")[0])
    ap.add_argument("--status", action="store_true", help="coverage and cache state, compile nothing")
    ap.add_argument("--record", action="store_true",
                    help="run pytest (default: the kernel test files; or pass pytest args after --) and write the manifest")
    ap.add_argument("--record-run", action="store_true",
                    help="run the script given after -- (real analysis, short) and record what it compiled")
    ap.add_argument("--fresh", action="store_true", help="with --record: replace the manifest instead of merging")
    ap.add_argument("--clean", action="store_true", help="delete this machine's Numba cache directory first")
    ap.add_argument("--if-needed", action="store_true", help="do nothing when this machine's cache is current")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for warming (Numba compiles hold the GIL)")
    args, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if args.status:
        return status()
    if args.record:
        return record(rest, args.fresh)
    if args.record_run:
        return record_run(rest, args.fresh)
    if args.clean:
        _jit_cache.clean_cache_dir()
    elif args.if_needed and _jit_cache.warm_state_reason(manifest_sha()) is None:
        print("[warmup] cache is current.")
        return 0
    return warm(jobs=max(1, args.jobs))


if __name__ == "__main__":
    sys.exit(main())
