"""Shared pieces of the HHT / Newmark implicit-dynamics branch of DynamicSolver2D and
DynamicSolver3D (Abaqus/Standard *DYNAMIC).

* step_time_integration(): the integrator a step asks for (step.time_integration, set
  by the *STATIC / *DYNAMIC reader) applied to a solver, the way solve(step) applies
  NLGEOM. A step without it (Python API) leaves the solver's constructor setting alone.
* lumped_mass(): diagonal mass, m_a = rho * int N_a dV per element (row-sum lumping;
  loads.integration.lumped_mass_weights), summed into a DOF vector.
* require_density(): Abaqus requires a density for every material in a dynamic step;
  the solvers' steel default (7.85e-9) is only ever a Python-API convenience.
* initial_acceleration(): M a0 = f_ext(t0) - f_int(u0) [- C v0] on the free DOFs, which
  Abaqus computes at the start of every *DYNAMIC step unless INITIAL=NO.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def material_density(mat: Any) -> float:
    """The density a material object gives (0.0 when it gives none)."""
    if mat is None:
        return 0.0
    if isinstance(mat, dict):
        return float(mat.get("density", mat.get("rho", 0.0)) or 0.0)
    for attr in ("density", "rho"):
        v = getattr(mat, attr, None)
        if v is not None:
            return float(v or 0.0)
    return 0.0


def material_label(mat: Any, pid: Any) -> str:
    """The material's name (a MaterialProps dict carries its Material), for messages."""
    for obj in (mat, getattr(mat, "material", None)):
        name = getattr(obj, "name", None)
        if name:
            return str(name)
    return f"of pid {pid}"


def step_time_integration(solver: Any, step: Any) -> Optional[Dict[str, Any]]:
    """Apply step.time_integration to `solver` and return it (None: nothing to apply)."""
    ti = getattr(step, "time_integration", None)
    if not ti:
        return None
    from dispsolver.solver.time_integrator_presets import resolve_time_integrator
    name = str(ti.get("name", "")).lower()
    if name == "quasistatic":
        cfg = resolve_time_integrator("quasistatic")
    else:
        cfg = resolve_time_integrator(name, alpha_hht=ti.get("alpha"), beta=ti.get("beta"),
                                      gamma=ti.get("gamma"))
    solver.config_integrator = cfg
    solver.time_integrator = cfg.name
    solver.alpha_hht = cfg.alpha_hht
    solver.beta_hht = cfg.beta
    solver.gamma_hht = cfg.gamma
    solver.alpha_m = cfg.alpha_m
    solver.alpha_f = cfg.alpha_f
    return ti


def is_dynamic(solver: Any) -> bool:
    cfg = solver.config_integrator
    return not cfg.is_quasistatic and not cfg.is_emcs


def require_density(solver: Any, step: Any) -> None:
    """Refuse a dynamic step when an element's material has no *DENSITY (Abaqus: a
    density is required for every material in a dynamic analysis)."""
    missing = getattr(solver, "_pids_without_density", None) or {}
    if missing:
        names = ", ".join(sorted({str(n) for n in missing.values()}))
        raise ValueError(f"*DYNAMIC step {getattr(step, 'name', '?')!r}: material(s) {names} have no *DENSITY; "
                         f"Abaqus requires a density for a dynamic step (the solver's default density "
                         f"would be a different model)")


def lumped_mass(solver: Any, dim: int) -> np.ndarray:
    """Diagonal mass vector (num_dofs) over the elements the solver assembles, cached on
    the solver (reference configuration, so it does not change during the analysis)."""
    cached = getattr(solver, "_lumped_mass_vec", None)
    if cached is not None and cached.shape[0] == solver.num_dofs:
        return cached
    from dispsolver.loads.integration import lumped_mass_weights
    elements = list(solver.mesh.elements.values())
    coords = np.asarray(solver.mesh.nodes_array(), dtype=np.float64)[:, :dim]
    nid_map = solver.mesh.node_id_to_index()
    m = np.zeros(solver.num_dofs, dtype=np.float64)
    for g_key in sorted(solver.elem_kernel_groups.keys()):
        for e_idx in solver.elem_kernel_groups[g_key]:
            elem = elements[int(e_idx)]
            idx = np.array([nid_map[n] for n in elem.node_ids], dtype=np.int64)
            w = lumped_mass_weights(coords[idx], elem.elem_type) * float(solver.elem_density[int(e_idx)])
            for d in range(dim):
                np.add.at(m, dim * idx + d, w)
    solver._lumped_mass_vec = m
    return m


def initial_acceleration(solver: Any, f_ext: np.ndarray, dt: float) -> np.ndarray:
    """a0 on the free DOFs from M a0 = f_ext - f_int(u0) - C v0 (lumped M), zero on the
    prescribed ones. Sets solver.a and returns it."""
    m = lumped_mass(solver, solver.num_dofs // solver.num_nodes)
    K0, f_int0 = solver.assemble_system(solver.u, dt=dt, update_state=False)
    r = np.asarray(f_ext, dtype=np.float64) - f_int0
    ray_a = float(getattr(solver, "alpha_m_damp", 0.0))
    ray_b = float(getattr(solver, "beta_k_damp", 0.0))
    if getattr(solver, "_rayleigh_in_hht", False) and (ray_a != 0.0 or ray_b != 0.0):
        r = r - ray_a * (m * solver.v) - ray_b * (K0 @ solver.v)
    free = np.ones(solver.num_dofs, dtype=bool)
    if solver.fixed_dofs:
        free[list(solver.fixed_dofs.keys())] = False
    a0 = np.zeros(solver.num_dofs, dtype=np.float64)
    ok = free & (m > 0.0)
    a0[ok] = r[ok] / m[ok]
    solver.a[:] = a0
    return a0
