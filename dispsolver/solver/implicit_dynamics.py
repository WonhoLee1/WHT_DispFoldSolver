"""Implicit dynamics (Abaqus/Standard *DYNAMIC, HHT / Newmark) for DynamicSolver2D and
DynamicSolver3D.

There is ONE Newton loop per solver -- the static one. Dynamics is a modification of the
equations it solves, not a second loop: while an increment runs, the solver's
assemble_system() returns, through HHTIncrement.transform(),

    f_eff(u) = (1+alpha) f_int(u) - alpha f_int,n + M a(u) + [(1+alpha) C v(u) - alpha C v_n]
    K_eff    = (1+alpha) K + M / (beta dt^2) + [(1+alpha) gamma/(beta dt) C]

and the load passed to the loop is f_ext,eff = (1+alpha) f_ext,n+1 - alpha f_ext,n
(Hilber, Hughes & Taylor 1977; Abaqus Theory Guide "Implicit dynamic analysis").
a(u), v(u) are the Newmark kinematics. Boundary conditions, convergence criteria,
contact SDI / freeze / friction commit, line search, hooks and cutbacks are the static
loop's own. a and v are updated only when the increment is accepted (run_increment).

Also here:
* step_time_integration(): the integrator a step asks for (step.time_integration, set by
  the *STATIC / *DYNAMIC reader) applied to a solver, the way solve(step) applies NLGEOM.
  A step without it (Python API) leaves the solver's constructor setting alone.
* lumped_mass(): m_a = rho * int N_a dV (loads.integration.lumped_mass_weights) over the
  nodes each element's kernel actually uses.
* require_density(): Abaqus requires a density for every material in a dynamic step.
* initial_acceleration(): M a0 = f_ext(t0) - f_int(u0) - C v0 at the start of every
  *DYNAMIC step (unless INITIAL=NO), with the step's constraints and prescribed motion.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags


# Element types the solvers skip (model/element_catalog._SKIP). The couplings, springs and
# gap elements carry no mass; MASS / ROTARYI do, and their magnitude (*MASS / *ROTARY
# INERTIA) is not read, so a dynamic step refuses them instead of running them massless.
_MASSLESS_SKIPPED = ("DCOUP3D", "DCOUP2D", "SPRING1", "SPRING2", "GAPUNI")
_MASS_ELEMENTS = ("MASS", "ROTARYI")

_CATEGORIES = {2: ("CPE3", "CPE4", "CPE6", "CPE8", "CAX4", "CAX8"),
               3: ("C3D4", "C3D6", "C3D8", "C3D10", "C3D20")}


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


def _kernel_type_name(etype: str, n_use: int, dim: int) -> str:
    """Element type whose shape functions the kernel integrates: the element's own when
    the kernel uses all its nodes, else the first-order type on the nodes it uses."""
    et = etype.strip().upper()
    if dim == 3:
        return f"C3D{n_use}"
    return f"{'CAX' if et.startswith(('CAX', 'CGAX')) else 'CPE'}{n_use}"


def lumped_mass(solver: Any, dim: int) -> np.ndarray:
    """Diagonal mass vector (num_dofs), rho * int N_a dV over the nodes each element's
    kernel uses. Cached per kernel grouping (reference configuration); a regrouping
    (_build_kernel_groups) or a different mesh rebuilds it."""
    key = (id(solver.elem_kernel_groups), solver.num_dofs)
    cached = getattr(solver, "_lumped_mass_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    from dispsolver.loads.integration import _classify_element, lumped_mass_weights
    elements = list(solver.mesh.elements.values())
    coords = np.asarray(solver.mesh.nodes_array(), dtype=np.float64)[:, :dim]
    m = np.zeros(solver.num_dofs, dtype=np.float64)
    grouped = np.zeros(len(elements), dtype=bool)
    for g_key in sorted(solver.elem_kernel_groups.keys()):
        conn = solver.elem_conn_groups[g_key]
        for local, e_idx in enumerate(solver.elem_kernel_groups[g_key]):
            e_idx = int(e_idx)
            grouped[e_idx] = True
            elem = elements[e_idx]
            idx = np.asarray(conn[local], dtype=np.int64)
            n_use = len(idx)
            et = elem.elem_type if n_use == len(elem.node_ids) else _kernel_type_name(elem.elem_type, n_use, dim)
            try:
                cat = _classify_element(et, n_use)
            except NotImplementedError:
                cat = None
            if cat not in _CATEGORIES[dim]:
                raise NotImplementedError(
                    f"element {getattr(elem, 'id', e_idx)} type {elem.elem_type}: no lumped mass in the {dim}D "
                    f"dynamic solver (solid continuum elements only); a dynamic step cannot run it")
            w = lumped_mass_weights(coords[idx], et) * float(solver.elem_density[e_idx])
            for d in range(dim):
                np.add.at(m, dim * idx + d, w)
    for e_idx, elem in enumerate(elements):
        if grouped[e_idx]:
            continue
        et = elem.elem_type.strip().upper()
        if et in _MASS_ELEMENTS:
            raise NotImplementedError(f"element {getattr(elem, 'id', e_idx)} type {et}: the point mass / rotary "
                                      f"inertia (*MASS / *ROTARY INERTIA) is not read, so a dynamic step would "
                                      f"run it massless")
        if et not in _MASSLESS_SKIPPED:
            raise NotImplementedError(f"element {getattr(elem, 'id', e_idx)} type {et} is not assembled; "
                                      f"its mass would be missing from the dynamic step")
    solver._lumped_mass_cache = (key, m)
    return m


def _damping(solver: Any) -> Tuple[float, float]:
    return float(getattr(solver, "alpha_m_damp", 0.0)), float(getattr(solver, "beta_k_damp", 0.0))


def _split(res):
    return res[0], res[1], tuple(res[2:])


class HHTIncrement:
    """One HHT / Newmark increment from the accepted state (u_n, v_n, a_n)."""

    def __init__(self, solver: Any, dim: int, dt: float, f_ext: np.ndarray, f_ext_n: np.ndarray):
        self.solver = solver
        self.dt = float(dt)
        self.alpha = float(solver.alpha_hht)
        self.beta = float(solver.beta_hht)
        self.gamma = float(solver.gamma_hht)
        self.u_n = solver.u.copy()
        self.v_n = solver.v.copy()
        self.a_n = solver.a.copy()
        self.m = lumped_mass(solver, dim)
        K_n, f_int_n = solver.assemble_system(self.u_n, dt=dt, update_state=False)[:2]
        self.f_int_n = np.array(f_int_n, dtype=np.float64, copy=True)
        self.ray_a, self.ray_b = _damping(solver)
        self.damped = self.ray_a != 0.0 or self.ray_b != 0.0
        self.f_damp_n = (self.ray_a * self.m * self.v_n + self.ray_b * (K_n @ self.v_n)) if self.damped else 0.0
        self.f_ext = np.asarray(f_ext, dtype=np.float64)
        self.f_ext_eff = (1.0 + self.alpha) * self.f_ext - self.alpha * np.asarray(f_ext_n, dtype=np.float64)
        self.c_m = 1.0 / (self.beta * self.dt ** 2)
        self.c_v = self.gamma / (self.beta * self.dt)
        self.inertia_norm = 0.0
        self.f_dyn = np.zeros_like(self.f_int_n)

    def kinematics(self, u: np.ndarray):
        a = (self.c_m * (u - self.u_n) - self.v_n / (self.beta * self.dt)
             - (0.5 / self.beta - 1.0) * self.a_n)
        v = self.v_n + self.dt * ((1.0 - self.gamma) * self.a_n + self.gamma * a)
        return a, v

    def transform(self, res, u: np.ndarray):
        """(K, f_int[, err]) of the physical system at u -> the HHT effective system."""
        K, f, rest = _split(res)
        a, v = self.kinematics(u)
        f_in = self.m * a
        f_eff = (1.0 + self.alpha) * f - self.alpha * self.f_int_n + f_in
        k_fac = 1.0 + self.alpha
        m_fac = self.c_m
        f_dyn = f_in
        if self.damped:
            f_damp = self.ray_a * self.m * v + self.ray_b * (K @ v)
            f_eff = f_eff + (1.0 + self.alpha) * f_damp - self.alpha * self.f_damp_n
            k_fac += (1.0 + self.alpha) * self.c_v * self.ray_b
            m_fac += (1.0 + self.alpha) * self.c_v * self.ray_a
            f_dyn = f_in + f_damp
        self.f_dyn = f_dyn
        self.inertia_norm = float(np.linalg.norm(f_in))
        K_eff = (csr_matrix(K) * k_fac + diags(m_fac * self.m, format="csr")).tocsr()
        return (K_eff, f_eff) + rest

    def commit(self) -> None:
        """Accepted increment: Newmark a, v at the accepted u; reactions (solver.f_int at
        the prescribed DOFs) include the inertia and damping force of those DOFs."""
        s = self.solver
        a, v = self.kinematics(s.u)
        s.a[:] = a
        s.v[:] = v
        if s.fixed_dofs and getattr(s, "f_int", None) is not None and len(s.f_int) == len(a):
            fixed = np.fromiter(s.fixed_dofs.keys(), dtype=np.int64, count=len(s.fixed_dofs))
            f_dyn = self.m * a
            if self.damped:
                K, _ = s._assemble_physical(s.u, dt=self.dt, update_state=False)[:2]
                f_dyn = f_dyn + self.ray_a * self.m * v + self.ray_b * (K @ v)
            f_int = np.array(s.f_int, dtype=np.float64, copy=True)
            f_int[fixed] += f_dyn[fixed]
            s.f_int = f_int
        s._hht_f_ext_prev = np.array(self.f_ext, copy=True)


def run_increment(solver: Any, dim: int, newton: Callable, dt: float, f_ext: Optional[np.ndarray],
                  f_ext_n: Optional[np.ndarray], **kw):
    """Run the solver's Newton loop `newton(dt=, f_ext=, **kw)` for one increment: as is
    for a static integrator, on the HHT effective system for a dynamic one.
    `f_ext_n` (load at t_n) defaults to the f_ext of the last accepted dynamic increment,
    else to f_ext itself."""
    if not is_dynamic(solver) or getattr(solver, "_dyn_ctx", None) is not None:
        return newton(dt=dt, f_ext=f_ext, **kw)
    if f_ext is None:
        applied = getattr(solver, "f_ext_applied", None)
        f_ext = applied.copy() if applied is not None else np.zeros(solver.num_dofs, dtype=np.float64)
    if f_ext_n is None:
        prev = getattr(solver, "_hht_f_ext_prev", None)
        f_ext_n = prev if prev is not None and prev.shape == f_ext.shape else f_ext
    ctx = HHTIncrement(solver, dim, dt, f_ext, f_ext_n)
    solver.M_global = diags(ctx.m, format="csc")
    solver._dyn_ctx = ctx
    try:
        converged, iters = newton(dt=dt, f_ext=ctx.f_ext_eff, **kw)
    finally:
        solver._dyn_ctx = None
    if converged:
        ctx.commit()
    return converged, iters


def _constraint_stiffness(solver: Any, u: np.ndarray, dt: float):
    """Stiffness the step's constraints (ties, MPCs, couplings, contact -- all enter the
    2D/3D solvers as penalty terms of assemble_system) add at u."""
    if not solver.constraints:
        return None
    saved = solver.constraints
    try:
        solver.constraints = []
        K0 = solver._assemble_physical(u, dt=dt, update_state=False)[0]
    finally:
        solver.constraints = saved
    K1 = solver._assemble_physical(u, dt=dt, update_state=False)[0]
    Kc = (csr_matrix(K1) - csr_matrix(K0)).tocsr()
    Kc.eliminate_zeros()
    return Kc if Kc.nnz else None


def prescribed_acceleration(plan: Any, t0: float, dt: float, u_start: np.ndarray) -> Dict[int, float]:
    """Second time derivative at t0+ of each prescribed DOF's history (one-sided second
    difference of plan.prescribed). A displacement applied at once (no amplitude in a
    dynamic step) has zero acceleration here: the jump is imposed over the first
    increment, as Abaqus/Standard does."""
    h = 1e-3 * dt
    vals = []
    for t in (t0 + h, t0 + 2 * h, t0 + 3 * h):
        d, v = plan.prescribed(t, u_start)
        vals.append(dict(zip(np.asarray(d).tolist(), np.asarray(v, dtype=np.float64).tolist())))
    return {dof: (vals[2][dof] - 2.0 * vals[1][dof] + vals[0][dof]) / h ** 2 for dof in vals[0]}


def initial_acceleration(solver: Any, f_ext: np.ndarray, dt: float,
                         a_prescribed: Optional[Dict[int, float]] = None) -> np.ndarray:
    """a0 from M a0 = f_ext - f_int(u0) - C v0, at the step start (solver.u, solver.v).

    Prescribed DOFs take `a_prescribed` (0 when not given). Constraints (penalty terms
    of assemble_system) are enforced on a0 the same way the step enforces them on u:
    (M + p K_c) a0 = r with p making the constraint stiffness dominate the mass, i.e.
    the accelerations the constraint couples move together. Sets and returns solver.a."""
    dim = solver.num_dofs // solver.num_nodes
    m = lumped_mass(solver, dim)
    K0, f_int0 = solver._assemble_physical(solver.u, dt=dt, update_state=False)[:2]
    r = np.asarray(f_ext, dtype=np.float64) - f_int0
    ray_a, ray_b = _damping(solver)
    if ray_a != 0.0 or ray_b != 0.0:
        r = r - ray_a * (m * solver.v) - ray_b * (K0 @ solver.v)
    n = solver.num_dofs
    a0 = np.zeros(n, dtype=np.float64)
    fixed = np.zeros(n, dtype=bool)
    if solver.fixed_dofs:
        fixed[list(solver.fixed_dofs.keys())] = True
    for dof, val in (a_prescribed or {}).items():
        if fixed[dof]:
            a0[dof] = val
    Kc = _constraint_stiffness(solver, solver.u, dt)
    free = ~fixed
    if Kc is None:
        ok = free & (m > 0.0)
        a0[ok] = r[ok] / m[ok]
    else:
        kc_max = float(np.max(np.abs(Kc.diagonal()))) or float(np.max(np.abs(Kc.data)))
        p = 1e8 * max(float(np.max(m)), 1e-300) / kc_max
        A = (diags(m, format="csr") + p * Kc).tocsr()
        fi = np.flatnonzero(free)
        rhs = r[fi] - (A[fi][:, np.flatnonzero(fixed)] @ a0[fixed] if fixed.any() else 0.0)
        Aff = A[fi][:, fi].tocsc()
        # DOFs with neither mass nor constraint stiffness (e.g. an unused node) stay at 0
        live = np.asarray(abs(Aff).sum(axis=1)).ravel() > 0.0
        from scipy.sparse.linalg import spsolve
        sol = np.zeros(len(fi))
        if live.any():
            sol[live] = spsolve(Aff[live][:, live].tocsc(), rhs[live])
        a0[fi] = sol
    solver.a[:] = a0
    return a0
