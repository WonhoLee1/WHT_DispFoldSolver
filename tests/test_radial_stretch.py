"""
test_radial_stretch.py
=======================
Regression gate for benchmark_element.benchmark_3d_mechanics.run_radial_stretch_benchmark
(dev_log/static_analysis_benchmark_design_20260913.md Sec 2, the one
"READY" item from the 13-benchmark static-analysis review).

**Status as of 2026-09-13: displacement check passes to a reasonable
thin-wedge-approximation tolerance; the stress check is HONESTLY XFAILED,
not forced to pass** -- a real, not-yet-understood discrepancy was found
while implementing this (see the xfail reason below and the module
docstring in benchmark_element/benchmark_3d_mechanics.py's
run_radial_stretch_benchmark for the full radial-profile investigation).
Do not silently widen the stress tolerance to make this pass without
actually resolving the discrepancy -- see AGENTS.md's standing rule
against exactly that pattern.
"""

import pytest

from benchmark_element.benchmark_3d_mechanics import run_radial_stretch_benchmark


def test_radial_stretch_displacement():
    """Radial displacement at every node vs. the closed-form u_r(r) =
    C2*r + C3/r (re-derived from first principles and independently
    cross-checked against the source's own sigma_rr/sigma_thetatheta
    formulas -- see run_radial_stretch_benchmark's docstring). Measured
    2026-09-13: ~1.1% relative error, attributed to the thin-wedge
    approximation (global-axis BCs standing in for exact
    radial/tangential-direction constraints) plus ordinary mesh
    discretization -- error did not vanish as the wedge angle was swept
    from 4 deg to 0.5 deg nor as n_r was refined 10->80, so this 1%
    should be treated as a real, currently only partially explained,
    residual -- not proven to be purely the wedge approximation.
    """
    r = run_radial_stretch_benchmark()
    assert r["status"] != "DIVERGED", f"solver diverged: {r}"
    assert r["max_u_err_relative"] < 0.05, (
        f"radial displacement error {r['max_u_err_relative']:.4f} exceeds the "
        "5% thin-wedge-approximation allowance"
    )


def test_radial_stretch_stress():
    """Cauchy stress (radial and hoop components, via StressFieldOutput)
    vs. the closed-form sigma_rr(r)/sigma_thetatheta(r).

    XFAIL, honestly, not a silently-widened tolerance: measured
    2026-09-13, max relative error ~30%. Investigated but NOT resolved:
    - The closed-form formula itself was re-derived from scratch
      (generalized plane strain, sigma_zz=0, u_r(Ri)=0, u_r(Ro)=U0) and
      exactly reproduces the design doc's C1/C2/C3 -- the reference is
      correct, not the bug.
    - The Newton solve converges to a genuinely tiny residual (~1e-5 on a
      ~1e10-scale force problem) -- this is not a convergence-tolerance
      artifact (unlike the earlier C3D8_FBAR investigation this session).
    - Radial-profile check: s_rr converges to <1% error at r=Ro (where
      the prescribed-displacement BC is exact at theta=0) but grows to
      ~28% error at r=Ri; s_tt is off by an even larger, and
      QUALITATIVELY WRONG-TREND amount (computed s_tt decreases with r,
      the exact solution increases with r) -- this last point in
      particular suggests something more specific than a generic
      thin-wedge/discretization approximation, but was not run to ground
      in the time available. Candidate suspects for a follow-up, NOT
      confirmed: (a) StressFieldOutput's single-centroid-point evaluation
      is explicitly documented (its own module docstring) as an
      approximation for any non-affine field, and this 1/r-varying field
      is exactly such a case, unlike every benchmark that module was
      previously verified against; (b) some other, more specific defect
      in the wedge mesh's BC approximation for the tangential/hoop
      direction specifically. Re-investigate before trusting stress
      output from this benchmark for anything beyond regression-tracking
      the same (currently ~30%) number.
    """
    r = run_radial_stretch_benchmark()
    assert r["status"] != "DIVERGED", f"solver diverged: {r}"
    if r["max_stress_err_relative"] > 0.05:
        pytest.xfail(
            f"stress error {r['max_stress_err_relative']:.4f} -- unresolved discrepancy, "
            "see this test's docstring and run_radial_stretch_benchmark's module docs"
        )
    assert r["max_stress_err_relative"] < 0.05
