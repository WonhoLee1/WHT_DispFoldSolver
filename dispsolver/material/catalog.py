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
    MAT_COHESIVE,
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
    MaterialSpec(MAT_COHESIVE, "COHESIVE", linear_elastic=False, dims=frozenset({2, 3}), needs_F=False),
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

    # Cohesive / Traction-separation
    if "cohesive" in extra or extra.get("mat_type") in ("COHESIVE", "TRACTION") or extra.get("elastic_type") == "TRACTION":
        return MAT_COHESIVE
    if "damage_init" in extra and isinstance(extra.get("damage_init"), dict) and str(extra["damage_init"].get("criterion", "")).upper() in ("MAXS", "MAXPE", "QUADS", "QUADE"):
        return MAT_COHESIVE

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

    # Kinematic & Combined J2 plasticity (takes precedence over isotropic Voce)
    if extra.get("hardening_type") in ("KINEMATIC", "COMBINED"):
        return MAT_J2_PLASTICITY

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


def prony_check(prony, capacity: int, kernel: str) -> None:
    """Refuse what the Prony kernels cannot carry: bulk relaxation (k_i != 0 is never packed) and more
    terms than the kernel packs."""
    if not prony:
        return
    if any(len(t) > 2 and float(t[1]) != 0.0 for t in prony):
        raise NotImplementedError(f"{kernel}: bulk relaxation (k_i != 0) is not represented; the Prony kernels "
                                  f"relax the deviatoric response only")
    if len(prony) > capacity:
        raise NotImplementedError(f"{kernel}: {len(prony)} Prony terms, the kernel packs {capacity}")


def moduli_factor(extra: dict, prony) -> float:
    """1 / (1 - sum g_i) when the deck's elastic constants are LONG TERM (Keywords Guide default of
    *ELASTIC / *HYPERELASTIC MODULI), else 1. Only a deck-read Material carries extra['moduli']; a
    Material built through the Python API keeps its constants as instantaneous (the kernels' meaning)."""
    if not prony or str(extra.get("moduli", "INSTANTANEOUS")).upper() != "LONG TERM":
        return 1.0
    g_sum = sum(float(t[0]) for t in prony)
    if g_sum >= 1.0:
        raise NotImplementedError(f"sum of Prony g_i = {g_sum} >= 1: the long-term modulus would vanish")
    return 1.0 / (1.0 - g_sum)


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
            from dispsolver.material.orientation import orthotropic_stiffness
            props[:36] = orthotropic_stiffness(extra["engineering_constants"]).ravel()
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
        if lambda_m <= 0.0:
            raise NotImplementedError(
                f"Arruda-Boyce lambda_m = {lambda_m} <= 0 (the kernel would silently use 3.0): "
                f"locking stretch must be positive")
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
        if any(float(d) != 0.0 for d in list(d_list)[1:]):
            raise NotImplementedError("Ogden D2..DN != 0 are not represented (the kernel packs D1 only)")
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
        h_type = str(extra.get("hardening_type", "ISOTROPIC")).upper()
        if h_type == "KINEMATIC":
            h_code = 1.0
            C_kin = float(extra.get("C_kin", H if H is not None else 0.0))
            H_iso = 0.0
        elif h_type == "COMBINED":
            h_code = 2.0
            C_kin = float(extra.get("C_kin", 0.0))
            H_iso = float(extra.get("H_iso", H if H is not None else 0.0))
        else:
            h_code = 0.0
            C_kin = 0.0
            H_iso = float(H if H is not None else 0.0)

        props[0] = E_val
        props[1] = nu_val
        props[2] = float(sy)
        props[3] = float(H_iso)
        props[4] = float(C_kin)
        props[5] = float(h_code)
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
        if len(table) > 10:
            raise NotImplementedError(f"tabular plasticity holds 10 points, the material has {len(table)}")
        n_pts = len(table)
        props[0] = E_val
        props[1] = nu_val
        props[2] = float(n_pts)
        props[3] = eps_fail
        props[4] = sigma_fail
        for p_idx in range(n_pts):
            props[5 + 2 * p_idx] = float(table[p_idx][0])
            props[6 + 2 * p_idx] = float(table[p_idx][1])
        props[40] = 1.0 if str(extra.get("plastic_extrapolation", "CONSTANT")).upper() == "LINEAR" else 0.0
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
        # Material.ConcreteDP stores extra['concrete_dp']; the flattened property dict carries
        # cdp_tables + beta/psi at the top level. Either source, never a default.
        cd = extra.get("concrete_dp") if isinstance(extra.get("concrete_dp"), dict) else None
        if cd is not None:
            tabs, beta_cd, psi_cd = cd, cd["beta"], cd["psi"]
        else:
            tabs, beta_cd, psi_cd = extra.get("cdp_tables", {}), extra.get("beta", 20.0), extra.get("psi", 15.0)
        flat = [E_val, nu_val, float(beta_cd), float(psi_cd)]
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
        prony_check(prony, capacity=1, kernel="plain Prony (id 3)")
        g1 = float(prony[0][0]) if prony else float(extra.get("g1", 0.2))
        tau1 = float(prony[0][2] if len(prony[0]) > 2 else prony[0][1]) if prony else float(extra.get("tau1", 1.0))
        # Abaqus *ELASTIC default MODULI=LONG TERM: the constants are long-term, instantaneous shear =
        # long-term / (1 - sum g_i); the kernel wants the instantaneous ones (bulk is not relaxed)
        f_mod = moduli_factor(extra, prony)
        if f_mod != 1.0:
            mu_inst = f_mod * E_val / (2.0 * (1.0 + nu_val))
            K_inst = E_val / (3.0 * (1.0 - 2.0 * nu_val))
            E_val = 9.0 * K_inst * mu_inst / (3.0 * K_inst + mu_inst)
            nu_val = (3.0 * K_inst - 2.0 * mu_inst) / (2.0 * (3.0 * K_inst + mu_inst))
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

    if mid == MAT_COHESIVE:
        coh = extra.get("cohesive", {}) if isinstance(extra.get("cohesive"), dict) else {}
        di = extra.get("damage_init", {}) if isinstance(extra.get("damage_init"), dict) else {}
        de = extra.get("damage_evol", {}) if isinstance(extra.get("damage_evol"), dict) else {}

        Knn = float(coh.get("Knn", extra.get("Knn", extra.get("K_nn", E_val))))
        Kss = float(coh.get("Kss", extra.get("Kss", extra.get("K_ss", Knn))))
        Nmax = float(coh.get("Nmax", di.get("Nmax", extra.get("Nmax", extra.get("N_max", 0.0)))))
        Smax = float(coh.get("Smax", di.get("Smax", extra.get("Smax", extra.get("S_max", Nmax)))))
        Gc = float(coh.get("Gc", de.get("Gf", extra.get("Gc", extra.get("G_c", extra.get("Gf", 0.0))))))
        crit_str = str(coh.get("criterion", di.get("criterion", "MAXS"))).upper()
        init_crit = 1.0 if crit_str in ("QUADS", "QUADE") else 0.0
        evol_type = 0.0
        delta_f = float(coh.get("delta_f", de.get("delta_f", extra.get("delta_f", 0.0))))
        visc = float(coh.get("viscosity", extra.get("viscosity", 0.0)))
        thick = float(coh.get("thickness", extra.get("thickness", 1.0)))

        props[0] = Knn
        props[1] = Kss
        props[2] = Nmax
        props[3] = Smax
        props[4] = Gc
        props[5] = init_crit
        props[6] = evol_type
        props[7] = delta_f
        props[8] = visc
        props[9] = thick
        return props

    # Fallback to direct props attribute if available
    raw_p = getattr(material, "props", None) or extra.get("props")
    if raw_p is not None:
        p_arr = np.asarray(raw_p, dtype=np.float64).ravel()
        n = min(len(p_arr), 64)
        props[:n] = p_arr[:n]
        return props

    return props
