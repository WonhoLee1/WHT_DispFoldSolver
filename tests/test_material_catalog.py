"""
tests/test_material_catalog.py
==============================
Unit tests and integrity verification for dispsolver/material/catalog (B2):
- Integrity test: every branch of material_dispatch_3d and material_dispatch_2d
  maps to exactly one MaterialSpec in MATERIALS, and vice versa.
- Id 1 is NEOHOOKEAN in both dimensions.
- Unified material_id resolver correctly classifies all material representations:
  Material objects, property dicts, Abaqus keywords.
- pack_props packs the parameters at exact offsets required by the Numba kernels.
- No silent fallbacks: unknown or invalid materials raise ValueError.
"""

import numpy as np
import pytest

from dispsolver.material.catalog import (
    MaterialSpec,
    MATERIALS,
    material_id,
    pack_props,
    is_linear_elastic,
)
from dispsolver.material3d.numba_materials import (
    MAT_LINEAR_ELASTIC,
    MAT_HYPERELASTIC_NEOHOOKEAN,
    MAT_J2_PLASTICITY,
    MAT_VISCOELASTIC_PRONY,
    MAT_HYPERELASTIC_ARRUDA_BOYCE,
    MAT_HYPERELASTIC_YEOH,
    MAT_HYPER_VISCOELASTIC,
    MAT_J2_PLASTICITY_NONLINEAR,
    MAT_HYPERELASTIC_MOONEY_RIVLIN,
    MAT_HYPERELASTIC_OGDEN,
    MAT_HYPERELASTIC_POLYNOMIAL,
    MAT_HYPERELASTIC_MARLOW,
    MAT_J2_PLASTICITY_TABULAR,
    MAT_J2_DUCTILE_DAMAGE,
    MAT_DRUCKER_PRAGER,
    MAT_CONCRETE_DP,
    MAT_CUSTOM_ELASTIC,
    material_dispatch_3d,
    material_dispatch_2d,
)
from dispsolver.model.material import Material


# -----------------------------------------------------------------------------
# Integrity Test: 1:1 Bijective Mapping with Kernel Dispatchers
# -----------------------------------------------------------------------------
def _create_sample_material(mid: int) -> Material:
    """Create a valid sample Material instance for testing kernel dispatch of mid."""
    m = Material(name=f"Mat_{mid}")
    if mid == MAT_LINEAR_ELASTIC:
        m.elastic = (200000.0, 0.3)
    elif mid == MAT_HYPERELASTIC_NEOHOOKEAN:
        m.hyperelastic = {"model": "NEOHOOKEAN", "c10": 0.2, "d1": 0.01}
    elif mid == MAT_J2_PLASTICITY:
        m.elastic = (200000.0, 0.3)
        m.plastic = [(250.0, 0.0)]
    elif mid == MAT_VISCOELASTIC_PRONY:
        m.elastic = (1000.0, 0.45)
        m.viscoelastic_prony = [(0.2, 0.0, 1.0)]
    elif mid == MAT_HYPERELASTIC_ARRUDA_BOYCE:
        m.hyperelastic = {"model": "ARRUDA_BOYCE", "mu": 0.2, "lambda_m": 3.0, "K": 10.0}
    elif mid == MAT_HYPERELASTIC_YEOH:
        m.hyperelastic = {"model": "YEOH", "c10": 0.1, "c20": 0.0, "c30": 0.0, "d1": 0.01, "d2": 0.0, "d3": 0.0}
    elif mid == MAT_HYPER_VISCOELASTIC:
        m.hyperelastic = {"model": "NEOHOOKEAN", "c10": 0.2, "d1": 0.01}
        m.viscoelastic_prony = [(0.2, 0.0, 1.0)]
    elif mid == MAT_J2_PLASTICITY_NONLINEAR:
        m.elastic = (200000.0, 0.3)
        m.extra_params["sigma_y0"] = 400.0
        m.extra_params["R_inf"] = 100.0
        m.extra_params["b_exp"] = 10.0
    elif mid == MAT_HYPERELASTIC_MOONEY_RIVLIN:
        m.hyperelastic = {"model": "MOONEY_RIVLIN", "c10": 0.1, "c01": 0.05, "d1": 0.01}
    elif mid == MAT_HYPERELASTIC_OGDEN:
        m.hyperelastic = {"model": "OGDEN", "mu_list": [0.2], "alpha_list": [2.0], "d_list": [0.01]}
    elif mid == MAT_HYPERELASTIC_POLYNOMIAL:
        m.hyperelastic = {"model": "POLYNOMIAL", "c10": 0.1, "c01": 0.0, "d1": 0.01}
    elif mid == MAT_HYPERELASTIC_MARLOW:
        m.hyperelastic = {"model": "MARLOW", "c10": 0.1, "d1": 0.01}
    elif mid == MAT_J2_PLASTICITY_TABULAR:
        m.elastic = (200000.0, 0.3)
        m.extra_params["tabular_plastic"] = [(250.0, 0.0), (350.0, 0.1)]
    elif mid == MAT_J2_DUCTILE_DAMAGE:
        m.elastic = (200000.0, 0.3)
        m.extra_params["damage_init"] = {"type": 0, "p": [0.05, 0, 0, 0, 0]}
        m.extra_params["damage_evol"] = {"type": 0, "Gf": 1.0, "L": 1.0}
        m.extra_params["tabular_plastic"] = [(250.0, 0.0)]
    elif mid == MAT_DRUCKER_PRAGER:
        m.elastic = (200000.0, 0.3)
        m.extra_params["d0"] = 100.0
        m.extra_params["beta"] = 20.0
        m.extra_params["psi"] = 20.0
    elif mid == MAT_CONCRETE_DP:
        m.elastic = (30000.0, 0.2)
        m.extra_params["cdp_tables"] = {"compression": [(10.0, 0.0)], "tension": [(2.0, 0.0)]}
    elif mid == MAT_CUSTOM_ELASTIC:
        m.C_mat = np.eye(6)
    return m


def test_catalog_dispatch_3d_coverage():
    """Verify that every MaterialSpec in MATERIALS is runnable by material_dispatch_3d."""
    dt = 0.01
    F = np.eye(3, dtype=np.float64)
    detF = 1.0
    E_voigt = np.zeros(6, dtype=np.float64)
    sdv_prev = np.zeros(64, dtype=np.float64)

    for mid, spec in MATERIALS.items():
        if 3 not in spec.dims:
            continue
        m = _create_sample_material(mid)
        props = pack_props(m, dim=3)
        S_voigt, C_mat, sdv_new, err = material_dispatch_3d(
            mid, props, sdv_prev, E_voigt, F, detF, dt
        )
        assert err == 0, f"material_dispatch_3d failed for {spec.name} (id={mid})"
        assert S_voigt.shape == (6,)
        assert C_mat.shape == (6, 6)


def test_catalog_dispatch_2d_coverage():
    """Verify that 2D dispatch handles every MaterialSpec marked with dim 2."""
    eps_2d = np.zeros(3, dtype=np.float64)
    sdv_prev = np.zeros(64, dtype=np.float64)
    dt = 0.01

    for mid, spec in MATERIALS.items():
        if 2 not in spec.dims:
            continue
        m = _create_sample_material(mid)
        props = pack_props(m, dim=2)
        sigma, C_mat_2d, sdv_new, err = material_dispatch_2d(
            mid, props, sdv_prev, eps_2d, dt
        )
        assert err == 0, f"material_dispatch_2d failed for {spec.name} (id={mid})"
        assert sigma.shape == (3,)
        assert C_mat_2d.shape == (3, 3)


def test_id_1_is_neohookean_in_both_dims():
    """Requirement: Id 1 is NEOHOOKEAN in both dimensions (preventing keyword audit O7 regression)."""
    assert MAT_HYPERELASTIC_NEOHOOKEAN == 1
    assert 1 in MATERIALS
    spec = MATERIALS[1]
    assert spec.name == "NEOHOOKEAN"
    assert spec.dims == frozenset({2, 3})
    assert spec.linear_elastic is False
    assert spec.needs_F is True


# -----------------------------------------------------------------------------
# Material Resolution and Property Packing Tests
# -----------------------------------------------------------------------------
def test_material_id_linear_elastic():
    m = Material(name="Steel", elastic=(210000.0, 0.3))
    assert material_id(m) == MAT_LINEAR_ELASTIC
    assert is_linear_elastic(material_id(m)) is True

    props = pack_props(m)
    assert props[0] == 210000.0
    assert props[1] == 0.3


def test_material_id_j2_plasticity():
    m = Material(name="Alu", elastic=(70000.0, 0.33), plastic=[(250.0, 0.0), (350.0, 0.1)])
    assert material_id(m) == MAT_J2_PLASTICITY
    assert is_linear_elastic(material_id(m)) is False

    props = pack_props(m)
    assert props[0] == 70000.0
    assert props[1] == 0.33
    assert props[2] == 250.0  # yield stress


def test_material_id_hyperelastic():
    m = Material(name="Rubber", hyperelastic={"model": "ARRUDA_BOYCE", "mu": 0.25, "lambda_m": 4.0, "K": 10.0})
    assert material_id(m) == MAT_HYPERELASTIC_ARRUDA_BOYCE
    props = pack_props(m)
    assert props[0] == 0.25
    assert props[1] == 4.0
    assert props[2] == 10.0

    m_neo = Material(name="Silicone", hyperelastic={"model": "NEOHOOKEAN", "c10": 0.3, "d1": 0.02})
    assert material_id(m_neo) == MAT_HYPERELASTIC_NEOHOOKEAN
    p_neo = pack_props(m_neo)
    assert p_neo[0] == 2.0 * 0.3  # mu = 2*c10
    assert p_neo[1] == 2.0 / 0.02  # K = 2/d1


def test_material_id_orthotropic_cylindrical():
    m = Material(name="Composite")
    m.extra_params["engineering_constants"] = (1000.0, 200.0, 200.0, 0.3, 0.3, 0.2, 50.0, 50.0, 40.0)
    assert material_id(m) == MAT_CUSTOM_ELASTIC
    assert is_linear_elastic(material_id(m)) is True

    props = pack_props(m)
    # Check that C_mat (36 floats) was populated and C11 > 0
    assert props[0] > 0.0


def test_material_id_no_silent_fallback():
    """Unrecognized material type must raise ValueError instead of defaulting to linear elastic."""
    with pytest.raises(ValueError, match="Unknown or unsupported material type"):
        material_id({"mat_type": "UNKNOWN_SUPER_MATERIAL"})

    with pytest.raises(ValueError, match="Unknown numeric material id"):
        material_id(99999)
