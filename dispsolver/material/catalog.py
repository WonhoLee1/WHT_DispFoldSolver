"""
dispsolver/material/catalog.py
==============================
The single source of truth for material models, their numeric IDs, and property packing.

Provides unified dispatch metadata across 2D and 3D solvers:
- MaterialSpec: metadata (ID, name, linear elasticity, supported dimensions, deformation gradient requirement)
- MATERIALS: dictionary mapping mat_id -> MaterialSpec
- material_id: canonical resolver from Material model / dict to integer mat_id
- pack_props: packs constitutive parameters into the flat float64 vector read by Numba kernels
- is_linear_elastic: fast predicate for constant-stiffness assembly paths
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

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
)

# Orthotropic cylindrical / engineering constants (used in dynamic3d:642)
MAT_ORTHOTROPIC_CYLINDRICAL = 17


@dataclass(frozen=True)
class MaterialSpec:
    """Specification of a constitutive material model supported by the solver."""
    mat_id: int                 # numeric ID matching numba_materials.MAT_*
    name: str                   # canonical uppercase string identifier
    linear_elastic: bool        # True if constant tangent modulus, no history evolution
    dims: frozenset[int]        # {2, 3} dimensions where the kernel dispatch really executes it
    needs_F: bool               # True if constitutive law requires deformation gradient F


_SPECS: list[MaterialSpec] = [
    MaterialSpec(MAT_LINEAR_ELASTIC, "LINEAR_ELASTIC", linear_elastic=True, dims=frozenset({2, 3}), needs_F=False),
    MaterialSpec(MAT_HYPERELASTIC_NEOHOOKEAN, "NEOHOOKEAN", linear_elastic=False, dims=frozenset({2, 3}), needs_F=True),
    MaterialSpec(MAT_J2_PLASTICITY, "J2_PLASTICITY", linear_elastic=False, dims=frozenset({2, 3}), needs_F=False),
    MaterialSpec(MAT_VISCOELASTIC_PRONY, "VISCOELASTIC_PRONY", linear_elastic=False, dims=frozenset({2, 3}), needs_F=False),
    MaterialSpec(MAT_HYPERELASTIC_ARRUDA_BOYCE, "ARRUDA_BOYCE", linear_elastic=False, dims=frozenset({2, 3}), needs_F=True),
    MaterialSpec(MAT_HYPERELASTIC_YEOH, "YEOH", linear_elastic=False, dims=frozenset({2, 3}), needs_F=True),
    MaterialSpec(MAT_HYPER_VISCOELASTIC, "HYPER_VISCOELASTIC", linear_elastic=False, dims=frozenset({2, 3}), needs_F=True),
    MaterialSpec(MAT_J2_PLASTICITY_NONLINEAR, "J2_PLASTICITY_NONLINEAR", linear_elastic=False, dims=frozenset({2, 3}), needs_F=False),
    MaterialSpec(MAT_HYPERELASTIC_MOONEY_RIVLIN, "MOONEY_RIVLIN", linear_elastic=False, dims=frozenset({2, 3}), needs_F=True),
    MaterialSpec(MAT_HYPERELASTIC_OGDEN, "OGDEN", linear_elastic=False, dims=frozenset({2, 3}), needs_F=True),
    MaterialSpec(MAT_HYPERELASTIC_POLYNOMIAL, "POLYNOMIAL", linear_elastic=False, dims=frozenset({2, 3}), needs_F=True),
    MaterialSpec(MAT_HYPERELASTIC_MARLOW, "MARLOW", linear_elastic=False, dims=frozenset({2, 3}), needs_F=True),
    MaterialSpec(MAT_J2_PLASTICITY_TABULAR, "J2_PLASTICITY_TABULAR", linear_elastic=False, dims=frozenset({2, 3}), needs_F=False),
    MaterialSpec(MAT_J2_DUCTILE_DAMAGE, "J2_DUCTILE_DAMAGE", linear_elastic=False, dims=frozenset({2, 3}), needs_F=False),
    MaterialSpec(MAT_DRUCKER_PRAGER, "DRUCKER_PRAGER", linear_elastic=False, dims=frozenset({2, 3}), needs_F=False),
    MaterialSpec(MAT_CONCRETE_DP, "CONCRETE_DP", linear_elastic=False, dims=frozenset({2, 3}), needs_F=False),
    MaterialSpec(MAT_CUSTOM_ELASTIC, "CUSTOM_ELASTIC", linear_elastic=True, dims=frozenset({2, 3}), needs_F=False),
]

MATERIALS: Dict[int, MaterialSpec] = {s.mat_id: s for s in _SPECS}


def is_linear_elastic(mat_id: int) -> bool:
    """Return True if the material model is linear elastic (constant tangent modulus)."""
    spec = MATERIALS.get(int(mat_id))
    return spec.linear_elastic if spec is not None else False


def material_id(material: Any) -> int:
    """Resolve an Abaqus Material object, property dictionary, or identifier into a canonical mat_id.

    Raises ValueError when the material is unrecognized (No silent fallback).
    """
    if isinstance(material, int):
        if material in MATERIALS:
            return material
        raise ValueError(f"Unknown numeric material id: {material}")

    # Check for direct object properties or dict
    if hasattr(material, "extra_params") and isinstance(material.extra_params, dict):
        extra = material.extra_params
    elif isinstance(material, dict):
        extra = material
    else:
        extra = {}

    # Orthotropic engineering constants -> MAT_CUSTOM_ELASTIC
    if "engineering_constants" in extra:
        return MAT_CUSTOM_ELASTIC

    # Custom elasticity matrix
    c_mat = getattr(material, "C_mat", None)
    if c_mat is not None or "C_mat" in extra:
        return MAT_CUSTOM_ELASTIC

    # Ductile damage
    if "damage_init" in extra or "damage_evol" in extra or "damage_init_type" in extra:
        return MAT_J2_DUCTILE_DAMAGE

    # Concrete damaged plasticity
    if "cdp_tables" in extra:
        return MAT_CONCRETE_DP

    # Drucker-Prager
    if ("d0" in extra and "beta" in extra) or extra.get("mat_type") == MAT_DRUCKER_PRAGER:
        return MAT_DRUCKER_PRAGER

    # Tabular plasticity / Radioss LAW36
    if "tabular_plastic" in extra or "tabular" in extra:
        return MAT_J2_PLASTICITY_TABULAR

    # Nonlinear Voce J2 plasticity
    if "R_inf" in extra or "b_exp" in extra:
        return MAT_J2_PLASTICITY_NONLINEAR

    # Hyperelastic or Viscoelastic
    hyper = getattr(material, "hyperelastic", None) or extra.get("hyperelastic")
    prony = getattr(material, "viscoelastic_prony", None) or extra.get("prony")

    if hyper is not None and prony is not None:
        return MAT_HYPER_VISCOELASTIC

    if hyper is not None:
        model = str(hyper.get("model", "")).upper()
        if model in ("ARRUDA_BOYCE", "ARRUDA-BOYCE", "ARRUDA"):
            return MAT_HYPERELASTIC_ARRUDA_BOYCE
        if model == "YEOH":
            return MAT_HYPERELASTIC_YEOH
        if model in ("MOONEY_RIVLIN", "MOONEY-RIVLIN", "MOONEY"):
            return MAT_HYPERELASTIC_MOONEY_RIVLIN
        if model == "OGDEN":
            return MAT_HYPERELASTIC_OGDEN
        if model == "POLYNOMIAL":
            return MAT_HYPERELASTIC_POLYNOMIAL
        if model == "MARLOW":
            return MAT_HYPERELASTIC_MARLOW
        if model in ("NEOHOOKEAN", "NEO_HOOKEAN"):
            return MAT_HYPERELASTIC_NEOHOOKEAN
        raise ValueError(f"Unknown hyperelastic model: {model!r}")

    if prony is not None:
        return MAT_VISCOELASTIC_PRONY

    # String mat_type dispatch
    mtype_raw = getattr(material, "mat_type", None) or extra.get("mat_type") or extra.get("type", "")
    if isinstance(mtype_raw, int):
        if mtype_raw in MATERIALS:
            return mtype_raw
        raise ValueError(f"Unknown numeric mat_type: {mtype_raw}")

    mtype = str(mtype_raw).upper().strip()

    if mtype in ("ELASTIC", "LINEAR_ELASTIC", "ISOTROPIC", ""):
        # Check if plastic yield stress is defined despite type string
        if getattr(material, "plastic", None) or "sigma_y0" in extra or "yield_stress" in extra:
            return MAT_J2_PLASTICITY
        # Check if hyperelastic parameters (c10) are present
        if "c10" in extra or "C10" in extra:
            return MAT_HYPERELASTIC_NEOHOOKEAN
        return MAT_LINEAR_ELASTIC

    if mtype in ("NEOHOOKEAN", "NEO_HOOKEAN"):
        return MAT_HYPERELASTIC_NEOHOOKEAN
    if mtype in ("J2_PLASTICITY", "PLASTIC", "J2"):
        return MAT_J2_PLASTICITY
    if mtype in ("J2_PLASTICITY_NONLINEAR", "VOCE"):
        return MAT_J2_PLASTICITY_NONLINEAR
    if mtype in ("TABULAR_PLASTIC", "RADIOSS_LAW36", "J2_PLASTICITY_TABULAR"):
        return MAT_J2_PLASTICITY_TABULAR
    if mtype in ("VISCOELASTIC", "PRONY", "VISCOELASTIC_PRONY"):
        return MAT_VISCOELASTIC_PRONY
    if mtype in ("ARRUDA_BOYCE", "ARRUDA-BOYCE", "ARRUDA"):
        return MAT_HYPERELASTIC_ARRUDA_BOYCE
    if mtype == "YEOH":
        return MAT_HYPERELASTIC_YEOH
    if mtype in ("MOONEY_RIVLIN", "MOONEY-RIVLIN", "MOONEY"):
        return MAT_HYPERELASTIC_MOONEY_RIVLIN
    if mtype == "OGDEN":
        return MAT_HYPERELASTIC_OGDEN
    if mtype == "POLYNOMIAL":
        return MAT_HYPERELASTIC_POLYNOMIAL
    if mtype == "MARLOW":
        return MAT_HYPERELASTIC_MARLOW
    if mtype == "DRUCKER_PRAGER":
        return MAT_DRUCKER_PRAGER
    if mtype == "CONCRETE_DP":
        return MAT_CONCRETE_DP

    raise ValueError(f"Unknown or unsupported material type: {mtype_raw!r}")


def pack_props(material: Any, dim: int = 3) -> np.ndarray:
    """Pack constitutive parameters of `material` into a 64-element float64 vector.

    Matches the exact layout consumed by 2D and 3D Numba kernel dispatchers.
    """
    props = np.zeros(64, dtype=np.float64)
    mid = material_id(material)

    # Extract extra parameters dictionary
    if hasattr(material, "extra_params") and isinstance(material.extra_params, dict):
        extra = material.extra_params
    elif isinstance(material, dict):
        extra = material
    else:
        extra = {}

    # Extract E and nu with defaults
    E_val = getattr(material, "E", None)
    nu_val = getattr(material, "nu", None)
    elastic = getattr(material, "elastic", None)
    if elastic is not None and len(elastic) >= 2:
        E_val, nu_val = elastic[0], elastic[1]
    if E_val is None:
        E_val = extra.get("E", 200000.0)
    if nu_val is None:
        nu_val = extra.get("nu", 0.3)
    E_val = float(E_val)
    nu_val = float(nu_val)

    if mid == MAT_LINEAR_ELASTIC:
        props[0] = E_val
        props[1] = nu_val
        return props

    if mid == MAT_CUSTOM_ELASTIC:
        c_mat = getattr(material, "C_mat", None)
        if c_mat is None:
            c_mat = extra.get("C_mat")
        if c_mat is not None:
            flat = np.asarray(c_mat, dtype=np.float64).ravel()
            n = min(len(flat), 36)
            props[:n] = flat[:n]
            return props
        if "engineering_constants" in extra:
            ec = [float(x) for x in extra["engineering_constants"]]
            E1, E2, E3 = ec[0], ec[1], ec[2]
            nu12, nu13, nu23 = ec[3], ec[4], ec[5]
            G12, G13, G23 = ec[6], ec[7], ec[8]
            nu21 = nu12 * E2 / E1
            nu31 = nu13 * E3 / E1
            nu32 = nu23 * E3 / E2
            # 6x6 compliance matrix
            S = np.zeros((6, 6), dtype=np.float64)
            S[0, 0] = 1.0 / E1;    S[0, 1] = -nu21 / E2; S[0, 2] = -nu31 / E3
            S[1, 0] = -nu12 / E1; S[1, 1] = 1.0 / E2;   S[1, 2] = -nu32 / E3
            S[2, 0] = -nu13 / E1; S[2, 1] = -nu23 / E2; S[2, 2] = 1.0 / E3
            S[3, 3] = 1.0 / G12
            S[4, 4] = 1.0 / G23
            S[5, 5] = 1.0 / G13
            try:
                C_6x6 = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                C_6x6 = np.eye(6, dtype=np.float64) * E1
            props[:36] = C_6x6.ravel()[:36]
            return props
        return props

    if mid == MAT_HYPERELASTIC_NEOHOOKEAN:
        hyper = getattr(material, "hyperelastic", None) or extra.get("hyperelastic", {})
        c10 = float(hyper.get("c10", extra.get("c10", extra.get("C10", 0.1))))
        d1 = float(hyper.get("d1", extra.get("d1", extra.get("D1", 0.01))))
        mu = 2.0 * c10
        K = 2.0 / max(d1, 1e-12)
        props[0] = mu
        props[1] = K
        props[2] = c10
        props[3] = d1
        return props

    if mid == MAT_HYPERELASTIC_ARRUDA_BOYCE:
        hyper = getattr(material, "hyperelastic", None) or extra.get("hyperelastic", {})
        mu = float(hyper.get("mu", extra.get("mu", 0.16785)))
        lambda_m = float(hyper.get("lambda_m", extra.get("lambda_m", 3.0)))
        K = float(hyper.get("K", extra.get("K", 8.3333)))
        props[0] = mu
        props[1] = lambda_m
        props[2] = K
        return props

    if mid == MAT_HYPERELASTIC_YEOH:
        hyper = getattr(material, "hyperelastic", None) or extra.get("hyperelastic", {})
        props[0] = float(hyper.get("c10", extra.get("c10", extra.get("C10", 0.1))))
        props[1] = float(hyper.get("c20", extra.get("c20", extra.get("C20", 0.0))))
        props[2] = float(hyper.get("c30", extra.get("c30", extra.get("C30", 0.0))))
        props[3] = float(hyper.get("d1", extra.get("d1", extra.get("D1", 0.01))))
        props[4] = float(hyper.get("d2", extra.get("d2", extra.get("D2", 0.0))))
        props[5] = float(hyper.get("d3", extra.get("d3", extra.get("D3", 0.0))))
        return props

    if mid == MAT_HYPERELASTIC_MOONEY_RIVLIN:
        hyper = getattr(material, "hyperelastic", None) or extra.get("hyperelastic", {})
        props[0] = float(hyper.get("c10", extra.get("c10", 0.1)))
        props[1] = float(hyper.get("c01", extra.get("c01", 0.05)))
        props[2] = float(hyper.get("d1", extra.get("d1", 0.01)))
        return props

    if mid == MAT_HYPERELASTIC_OGDEN:
        hyper = getattr(material, "hyperelastic", None) or extra.get("hyperelastic", {})
        mu_list = hyper.get("mu_list", extra.get("mu_list", [0.2]))
        alpha_list = hyper.get("alpha_list", extra.get("alpha_list", [2.0]))
        d_list = hyper.get("d_list", extra.get("d_list", [0.01]))
        n_terms = min(len(mu_list), 6)
        props[0] = float(n_terms)
        props[1] = float(d_list[0] if len(d_list) > 0 else 0.01)
        for p in range(n_terms):
            props[2 + 2 * p] = float(mu_list[p])
            props[3 + 2 * p] = float(alpha_list[p])
        return props

    if mid == MAT_HYPERELASTIC_POLYNOMIAL:
        hyper = getattr(material, "hyperelastic", None) or extra.get("hyperelastic", {})
        props[0] = float(hyper.get("c10", extra.get("c10", 0.1)))
        props[1] = float(hyper.get("c01", extra.get("c01", 0.0)))
        props[2] = float(hyper.get("c20", extra.get("c20", 0.0)))
        props[3] = float(hyper.get("c11", extra.get("c11", 0.0)))
        props[4] = float(hyper.get("c02", extra.get("c02", 0.0)))
        props[5] = float(hyper.get("d1", extra.get("d1", 0.01)))
        props[6] = float(hyper.get("d2", extra.get("d2", 0.0)))
        return props

    if mid == MAT_HYPERELASTIC_MARLOW:
        hyper = getattr(material, "hyperelastic", None) or extra.get("hyperelastic", {})
        props[0] = float(hyper.get("c10", extra.get("c10", 0.1)))
        props[1] = float(hyper.get("c20", extra.get("c20", 0.0)))
        props[2] = float(hyper.get("c30", extra.get("c30", 0.0)))
        props[3] = float(hyper.get("d1", extra.get("d1", 0.01)))
        props[4] = float(hyper.get("d2", extra.get("d2", 0.0)))
        props[5] = float(hyper.get("d3", extra.get("d3", 0.0)))
        return props

    if mid == MAT_J2_PLASTICITY:
        sy = getattr(material, "sigma_y0", getattr(material, "yield_stress", None))
        if sy is None:
            plastic = getattr(material, "plastic", None)
            if plastic:
                sy = plastic[0][0]
            else:
                sy = extra.get("yield_stress", extra.get("sigma_y0", 400.0))
        H = getattr(material, "H", getattr(material, "hardening_modulus", None))
        if H is None:
            H = extra.get("hardening_modulus", extra.get("H", 0.0))
        props[0] = E_val
        props[1] = nu_val
        props[2] = float(sy)
        props[3] = float(H)
        return props

    if mid == MAT_J2_PLASTICITY_NONLINEAR:
        sy = float(getattr(material, "sigma_y0", extra.get("yield_stress", extra.get("sigma_y0", 400.0))))
        H = float(getattr(material, "H", extra.get("hardening_modulus", extra.get("H", 0.0))))
        R_inf = float(extra.get("R_inf", 100.0))
        b_exp = float(extra.get("b_exp", 10.0))
        props[0] = E_val
        props[1] = nu_val
        props[2] = sy
        props[3] = H
        props[4] = R_inf
        props[5] = b_exp
        return props

    if mid == MAT_J2_PLASTICITY_TABULAR:
        table = extra.get("tabular_plastic", extra.get("tabular", getattr(material, "plastic", []))) or [(100.0, 0.0)]
        eps_fail = float(extra.get("eps_fail", 1e9))
        sigma_fail = float(extra.get("sigma_fail", 0.0))
        n_pts = min(len(table), 10)
        props[0] = E_val
        props[1] = nu_val
        props[2] = float(n_pts)
        props[3] = eps_fail
        props[4] = sigma_fail
        for p_idx in range(n_pts):
            props[5 + 2 * p_idx] = float(table[p_idx][0])
            props[6 + 2 * p_idx] = float(table[p_idx][1])
        return props

    if mid == MAT_J2_DUCTILE_DAMAGE:
        di = extra.get("damage_init", {})
        de = extra.get("damage_evol", {})
        dp = list(extra.get("damage_init_p", di.get("p", [0.05, 0, 0, 0, 0])))
        while len(dp) < 5:
            dp.append(0.0)
        props[0] = E_val
        props[1] = nu_val
        props[2] = float(extra.get("yield_stress", 250.0))
        props[3] = float(extra.get("hardening_modulus", 0.0))
        if isinstance(di, dict):
            init_type = di.get("type", 0)
        else:
            init_type = extra.get("damage_init_type", di)
        props[4] = float(init_type)

        for i in range(5):
            props[5 + i] = float(dp[i])

        gf = de.get("Gf", 1.0) if isinstance(de, dict) else extra.get("damage_Gf", 1.0)
        L_val = de.get("L", 1.0) if isinstance(de, dict) else extra.get("damage_L", 1.0)
        evol_type = de.get("type", 0) if isinstance(de, dict) else extra.get("damage_evol", 0)

        props[10] = float(gf)
        props[11] = float(L_val)
        props[12] = float(evol_type)
        props[13] = float(extra.get("damage_T", 20.0))
        props[14] = float(di.get("Troom", 20.0) if isinstance(di, dict) else extra.get("damage_Troom", 20.0))
        props[15] = float(di.get("Tmelt", 1000.0) if isinstance(di, dict) else extra.get("damage_Tmelt", 1000.0))
        props[16] = float(di.get("eps0", 1.0) if isinstance(di, dict) else extra.get("damage_eps0", 1.0))
        return props

    if mid == MAT_DRUCKER_PRAGER:
        dp = extra.get("drucker_prager", {}) if isinstance(extra.get("drucker_prager"), dict) else {}
        props[0] = E_val
        props[1] = nu_val
        props[2] = float(dp.get("d0", extra.get("d0", 100.0)))
        props[3] = float(dp.get("H", extra.get("H", 0.0)))
        props[4] = float(dp.get("beta", extra.get("beta", 20.0)))
        props[5] = float(dp.get("psi", extra.get("psi", 20.0)))
        return props

    if mid == MAT_CONCRETE_DP:
        tabs = extra.get("cdp_tables", {})
        flat = [E_val, nu_val, float(extra.get("beta", 20.0)), float(extra.get("psi", 15.0))]
        for key in ("compression", "tension", "dt_table", "dc_table"):
            rows = list(tabs.get(key, [(0.0, 0.0)]))[:6]
            flat.append(float(len(rows)))
            for (a, b) in rows:
                flat.append(float(a))
                flat.append(float(b))
        n = min(len(flat), 64)
        props[:n] = flat[:n]
        return props

    if mid == MAT_VISCOELASTIC_PRONY:
        prony = getattr(material, "viscoelastic_prony", None) or extra.get("prony", [])
        g1 = float(prony[0][0]) if prony else float(extra.get("g1", 0.2))
        tau1 = float(prony[0][2] if len(prony[0]) > 2 else prony[0][1]) if prony else float(extra.get("tau1", 1.0))
        props[0] = E_val
        props[1] = nu_val
        props[2] = g1
        props[3] = tau1
        return props

    if mid == MAT_HYPER_VISCOELASTIC:
        prony = getattr(material, "viscoelastic_prony", None) or extra.get("prony", [])
        base_m = extra.get("base_model", MAT_HYPERELASTIC_NEOHOOKEAN)
        props[0] = float(base_m)
        props[1] = float(extra.get("mu", extra.get("c10", 0.2)))
        props[2] = float(extra.get("K", extra.get("lambda_m", 10.0)))
        props[3] = float(extra.get("p2", 0.0))
        n_terms = min(len(prony), 6)
        props[4] = float(n_terms)
        for m_idx in range(n_terms):
            item = prony[m_idx]
            g_val = float(item[0])
            tau_val = float(item[2] if len(item) > 2 else item[1])
            props[5 + 2 * m_idx] = g_val
            props[6 + 2 * m_idx] = tau_val
        return props

    # Fallback to direct props attribute if available
    raw_p = getattr(material, "props", None) or extra.get("props")
    if raw_p is not None:
        p_arr = np.asarray(raw_p, dtype=np.float64).ravel()
        n = min(len(p_arr), 64)
        props[:n] = p_arr[:n]
        return props

    return props
