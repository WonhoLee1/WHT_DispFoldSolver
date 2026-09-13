"""
test_2d_elements.py
===================
Unit and regression tests for 2D solid finite element formulations,
rank sufficiency, tangent consistency, and multilayer 2-point bending.
"""

import pytest
import numpy as np

from benchmark_element.benchmark_2d_elements import (
    ALL_2D_ELEMENTS,
    run_rank_sufficiency_test,
    run_tangent_consistency_test,
)
from benchmark_element.two_point_bending_multilayer_theory import (
    MultilayerTwoPointBendingTheory,
    get_standard_display_stackup,
)
from benchmark_element.mechanics_patches_2d import (
    make_multilayer_two_point_bending_mesh_2d,
    make_distorted_patch_mesh_2d,
)


@pytest.mark.parametrize("elem_type", ALL_2D_ELEMENTS)
def test_2d_element_rank_sufficiency(elem_type):
    """Verify each 2D element exhibits exactly 3 zero rigid-body modes (no spurious modes)."""
    res = run_rank_sufficiency_test(elem_type)
    assert res["passed"], f"Element {elem_type} failed rank test: zero_modes={res['n_zero']} (expected 3)"


@pytest.mark.parametrize("elem_type", ALL_2D_ELEMENTS)
def test_2d_element_tangent_consistency(elem_type):
    """Verify relative discrepancy between Ke and central difference Jacobian < 1e-4."""
    res = run_tangent_consistency_test(elem_type)
    assert res["passed"], f"Element {elem_type} failed tangent test: rel_err={res['rel_err']:.2e}"


def test_multilayer_theory_bending_stiffness():
    """Verify multilayer composite bending stiffness bounds: (EI)_slip < (EI)_eff < (EI)_full."""
    layers = get_standard_display_stackup("3layer")
    theory = MultilayerTwoPointBendingTheory(layers=layers, D=20.0, shear_efficiency=0.45)

    assert theory.EI_slip < theory.EI_eff < theory.EI_full
    assert theory.total_thickness == pytest.approx(0.150, abs=1e-5)
    assert theory.D_eff == pytest.approx(20.0 - 0.150, abs=1e-5)

    # Check Elastica peak stress calculation
    s_peak = theory.peak_stress()
    assert s_peak > 0.0

    # Check thickness stress profile continuity
    ys, sigmas, names = theory.compute_thickness_stress_profile()
    assert len(ys) == len(sigmas) == len(names)
    assert np.all(np.isfinite(sigmas))


@pytest.mark.parametrize("elem_type", ["CPE4", "CPE4I", "CPE4R", "CPE8"])
def test_multilayer_mesh_generation(elem_type):
    """Verify structured 2D multilayer mesh generation."""
    layers = get_standard_display_stackup("3layer")
    mesh, groups, pids = make_multilayer_two_point_bending_mesh_2d(
        elem_type, layers=layers, L=60.0, nx=20, ny_per_layer=1
    )

    assert len(mesh.nodes) > 0
    assert len(mesh.elements) > 0
    assert len(groups["left"]) > 0
    assert len(groups["right"]) > 0
    assert len(groups["apex_nodes"]) > 0
    assert set(pids.values()) == {1, 2, 3}
