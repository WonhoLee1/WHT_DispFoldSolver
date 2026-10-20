"""dispsolver.warmup -- discovery, signature round trip, compile-without-running, cold-cache detection."""
import numpy as np
import pytest
from numba import njit
from numba.core import types

from dispsolver import _jit_cache, warmup


def _fresh_kernel():
    @njit
    def kern(a, b, c=2.5, flag=True):
        return a.sum() + b[0, 0] + c

    return kern


def test_signature_round_trip_including_omitted_default_and_readonly():
    k = _fresh_kernel()
    ro = np.ones((2, 2))
    ro.flags.writeable = False
    k(np.ones(3), ro)                 # c and flag omitted
    k(np.ones(3), np.ones((2, 2)), 1.0, False)
    sigs = list(k.signatures)
    assert len(sigs) == 2
    for sig in sigs:
        descs = [warmup._describe(t) for t in sig]
        assert all(d is not None for d in descs)
        rebuilt = tuple(warmup._build(d, dflt) for d, dflt in zip(descs, warmup._defaults(k)))
        assert rebuilt == sig


def test_warming_compiles_without_running_the_kernel(monkeypatch):
    calls = []

    @njit
    def probe(x):
        return x[0] / 0.0   # would be inf, never reached: we only compile

    desc = [{"k": "array", "dtype": "float64", "ndim": 1, "layout": "C", "ro": False, "aligned": True}]
    monkeypatch.setattr(warmup, "discover", lambda: {"m:probe": probe})
    assert probe.signatures == []
    res = warmup._compile_keys(["m:probe"], {"m:probe": [desc]})
    assert res["done"] == 1 and res["failed"] == 0 and not res["missing"]
    assert len(probe.signatures) == 1
    assert not calls
    # idempotent: a second pass has nothing left to compile
    assert warmup._compile_keys(["m:probe"], {"m:probe": [desc]})["done"] == 0


def test_a_manifest_entry_without_code_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(warmup, "discover", lambda: {})
    res = warmup._compile_keys(["gone:kernel"], {"gone:kernel": [[]]})
    assert res["missing"] == ["gone:kernel"] and res["failed"] == 0


def test_unreproducible_signatures_are_skipped():
    assert warmup._describe(types.unicode_type) is None


def test_discovery_finds_package_kernels_without_registration():
    warmup.import_all()
    found = warmup.discover()
    assert "dispsolver.element3d.c3d8r_numba:compute_c3d8r_element_numba" in found
    assert len(found) > 100


def test_cold_cache_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(_jit_cache, "numba_cache_dir", lambda: str(tmp_path))
    assert "never warmed" in _jit_cache.warm_state_reason("abc")
    _jit_cache.write_warm_stamp("abc")
    assert _jit_cache.warm_state_reason("abc") is None
    assert "manifest changed" in _jit_cache.warm_state_reason("def")
    real = _jit_cache.fingerprint
    monkeypatch.setattr(_jit_cache, "fingerprint", lambda: {**real(), "cpu": "another-machine"})
    reason = _jit_cache.warm_state_reason("abc")
    assert "machine or software changed" in reason and "cpu" in reason


def test_clean_removes_the_cache_directory(tmp_path, monkeypatch):
    d = tmp_path / "numba"
    d.mkdir()
    (d / "x.nbi").write_text("stale")
    monkeypatch.setattr(_jit_cache, "numba_cache_dir", lambda: str(d))
    _jit_cache.clean_cache_dir()
    assert d.is_dir() and not list(d.iterdir())
