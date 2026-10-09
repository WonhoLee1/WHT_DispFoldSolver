# -*- coding: utf-8 -*-
"""Review round of the *DYNAMIC (HHT) work: dynamics now runs inside the solvers' single
Newton loop (implicit_dynamics.run_increment), so prescribed displacements, contact
acceptance / friction commit and the static convergence criteria apply to it.

Each test here failed on 1e214a4 unless its docstring says otherwise.
"""
import numpy as np
import pytest

from dispsolver.io.abaqus_reader import execute_inp
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d import Mesh3D
from dispsolver.model.material import Material
from dispsolver.solver.implicit_dynamics import initial_acceleration, lumped_mass
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

E, RHO = 1000.0, 1e-3


def _mat(E=E, nu=0.0, density=RHO):
    m = Material(name="BAR")
    m.Elastic(E=E, nu=nu)
    m.density = density
    return m


# ── prescribed displacement history in a *DYNAMIC deck ─────────────────────

def _base_deck(dim, Tr, U, dt):
    """SDOF rod (L = 5, section 2 x 3 in 3D / height 2 in 2D), lateral DOFs fixed, root x
    driven by U * ramp(t / Tr) (tabular amplitude), tip free. Base excitation:
    M x'' + K (x - u_b) = 0 with u_b = U t / Tr from rest -> x = U/Tr (t - sin(w t)/w)."""
    if dim == 3:
        nodes = ("1, 0., 0., 0.\n2, 5., 0., 0.\n3, 5., 2., 0.\n4, 0., 2., 0.\n"
                 "5, 0., 0., 3.\n6, 5., 0., 3.\n7, 5., 2., 3.\n8, 0., 2., 3.\n")
        elem = "*ELEMENT, TYPE=C3D8, ELSET=ROD\n1, 1, 2, 3, 4, 5, 6, 7, 8\n"
        sets = "*NSET, NSET=ROOT\n1, 4, 5, 8\n*NSET, NSET=ALLN\n1, 2, 3, 4, 5, 6, 7, 8\n"
        lat, sec = "ALLN, 2, 3, 0.0\n", ""
    else:
        nodes = "1, 0., 0.\n2, 5., 0.\n3, 5., 2.\n4, 0., 2.\n"
        elem = "*ELEMENT, TYPE=CPE4, ELSET=ROD\n1, 1, 2, 3, 4\n"
        sets = "*NSET, NSET=ROOT\n1, 4\n*NSET, NSET=ALLN\n1, 2, 3, 4\n"
        lat, sec = "ALLN, 2, 2, 0.0\n", "1.0\n"
    return (f"*HEADING\nbase excitation\n*NODE\n{nodes}{elem}{sets}"
            f"*MATERIAL, NAME=RODMAT\n*ELASTIC\n1000.0, 0.0\n*DENSITY\n1.0e-3\n"
            f"*SOLID SECTION, ELSET=ROD, MATERIAL=RODMAT\n{sec}"
            f"*AMPLITUDE, NAME=RAMP\n0., 0., {Tr}, 1.\n"
            f"*BOUNDARY\n{lat}"
            f"*STEP, NAME=DYN, NLGEOM=NO\n*DYNAMIC, ALPHA=0.0\n{dt}, {Tr}, , {dt}\n"
            f"*BOUNDARY, AMPLITUDE=RAMP\nROOT, 1, 1, {U}\n*END STEP\n")


@pytest.mark.parametrize("dim", [2, 3])
def test_deck_prescribed_displacement_history_drives_the_body(dim):
    K = E * (6.0 if dim == 3 else 2.0) / 5.0
    M = RHO * (30.0 if dim == 3 else 10.0) / 2.0
    om = np.sqrt(K / M)
    T = 2 * np.pi / om
    Tr, U, dt = 5 * T, 0.01, T / 40
    res = execute_inp(_base_deck(dim, Tr, U, dt), dim=dim, verbose=False)
    assert res["converged"]
    tip = float(res["u"][dim * 1])                    # node 2, x
    root = float(res["u"][0])
    assert root == pytest.approx(U, rel=1e-9)          # 1e214a4: prescribed DOF never moved (0.0)
    exact = U / Tr * (Tr - np.sin(om * Tr) / om)
    assert abs(tip - exact) < 2e-3 * U                 # 1e214a4: 0.0
    assert tip == pytest.approx(U, rel=0.05)           # slow drive: close to the static limit


def test_hht_branch_applies_prescribed_dof_in_one_step():
    """The reviewer's probe: a prescribed tip displacement on a body at rest."""
    mesh = Mesh3D()
    for i in range(3):
        for j, (y, z) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            mesh.add_node(4 * i + j + 1, float(i), y, z)
    mesh.add_element(1, [1, 5, 6, 2, 4, 8, 7, 3], "C3D8", pid=0)
    mesh.add_element(2, [5, 9, 10, 6, 8, 12, 11, 7], "C3D8", pid=0)
    s = DynamicSolver3D(mesh, materials={0: _mat()}, nlgeom=False, time_integrator="hht-tf")
    for n in (1, 2, 3, 4):
        for d in range(3):
            s.fix_dof(n, d, 0.0)
    nm = mesh.node_id_to_index()
    for n in (9, 10, 11, 12):
        s.fix_dof(n, 0, 0.01)
    ok, _ = s.solve_step(dt=1e-3)
    assert ok
    assert s.u[3 * nm[9]] == pytest.approx(0.01, rel=1e-12)   # 1e214a4: 0.0


# ── contact + dynamics: acceptance path with friction commit ───────────────

def _contact_run(mu, n=10, dt=1e-3):
    from dispsolver.constraint3d.surface_contact3d_deformable import DeformableSurfaceContactConstraint3D
    mesh = Mesh3D()
    lo = [(0, 0, -1), (1, 0, -1), (1, 1, -1), (0, 1, -1), (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    up = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for i, p in enumerate(lo + up, 1):
        mesh.add_node(i, *p)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    mesh.add_element(2, list(range(9, 17)), "C3D8", pid=0)
    s = DynamicSolver3D(mesh, materials={0: _mat(nu=0.3, density=1e-6)}, nlgeom=False, time_integrator="hht-tf")
    for nid in range(1, 9):
        for d in range(3):
            s.fix_dof(nid, d, 0.0)
    nm = mesh.node_id_to_index()
    c = DeformableSurfaceContactConstraint3D([9, 10, 11, 12], [(5, 6, 7, 8)], nm, mesh.nodes_array(),
                                             penalty_stiffness=1e4, friction_coefficient=mu,
                                             master_interior={(5, 6, 7, 8): np.array([0.5, 0.5, -0.5])})
    s.add_constraint(c)
    commits = []
    orig = c.commit_friction_state
    c.commit_friction_state = lambda u: (commits.append(1), orig(u))[1]
    f = np.zeros(s.num_dofs)
    for nid in (13, 14, 15, 16):
        f[3 * nm[nid] + 2] = -1.0                      # press down: 4 N
        f[3 * nm[nid] + 0] = 0.2                       # push sideways: 0.8 N
    initial_acceleration(s, f, dt)
    for _ in range(n):
        ok, _ = s.solve_step(dt=dt, f_ext=f, f_ext_n=f)
        assert ok
    return len(commits), float(s.u[3 * nm[9]]), s


def test_contact_dynamic_step_commits_friction_and_sticks():
    n, x_stick, s = _contact_run(0.5)
    assert n == 10                                     # 1e214a4: 0 (HHT branch bypassed acceptance)
    assert s.last_accept is not None
    assert abs(x_stick) < 1e-3                         # mu N = 2 N > 0.8 N: sticks
    n, x_slide, _ = _contact_run(0.0)
    # frictionless: rigid-body slide of the upper block, x = F t^2 / (2 M), M = rho V = 1e-6
    assert x_slide == pytest.approx(0.8 * 0.01 ** 2 / (2 * 1e-6), rel=0.05)


# ── tiny loads in N-mm-tonne ───────────────────────────────────────────────

def test_tiny_load_in_n_mm_tonne_units_moves_the_body():
    """Steel rod in N-mm-tonne (E = 2e5 MPa, rho = 7.85e-9 t/mm^3), 1e-6 N tip load.
    1e214a4 accepted iteration 1 on the absolute residual 1e-6 < 1e-5: u stayed 0."""
    mesh = Mesh3D()
    xyz = [(0, 0, 0), (5, 0, 0), (5, 2, 0), (0, 2, 0), (0, 0, 3), (5, 0, 3), (5, 2, 3), (0, 2, 3)]
    for i, p in enumerate(xyz, 1):
        mesh.add_node(i, *p)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    s = DynamicSolver3D(mesh, materials={0: _mat(E=2e5, density=7.85e-9)}, nlgeom=False, time_integrator="newmark")
    for n in range(1, 9):
        s.fix_dof(n, 1, 0.0)
        s.fix_dof(n, 2, 0.0)
    for n in (1, 4, 5, 8):
        s.fix_dof(n, 0, 0.0)
    nm = mesh.node_id_to_index()
    tip = [3 * nm[n] for n in (2, 3, 6, 7)]
    f = np.zeros(s.num_dofs)
    f[tip] = 0.25e-6
    K, M = 2e5 * 6 / 5, 7.85e-9 * 30 / 2
    dt = 2 * np.pi / np.sqrt(K / M) / 40
    initial_acceleration(s, f, dt)
    ok, _ = s.solve_step(dt=dt, f_ext=f, f_ext_n=f)
    assert ok
    x1 = 2 * 1e-6 / (K + 4 * M / dt ** 2)             # trapezoidal step from rest, a0 = F/M
    assert s.u[tip[0]] == pytest.approx(x1, rel=1e-6, abs=0.0)


# ── lumped mass: more element families; refusals ───────────────────────────

def test_total_mass_c3d4_c3d6():
    """(Passes on 1e214a4 too -- the per-family row sums were already right there.)"""
    mesh = Mesh3D()
    for i, p in enumerate([(0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 5)], 1):
        mesh.add_node(i, *p)
    mesh.add_element(1, [1, 2, 3, 4], "C3D4", pid=0)
    s = DynamicSolver3D(mesh, materials={0: _mat()}, nlgeom=False, time_integrator="newmark")
    assert lumped_mass(s, 3)[0::3].sum() == pytest.approx(RHO * 5.0, rel=1e-12)
    mesh = Mesh3D()
    for i, p in enumerate([(0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 5), (2, 0, 5), (0, 3, 5)], 1):
        mesh.add_node(i, *p)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6], "C3D6", pid=0)
    s = DynamicSolver3D(mesh, materials={0: _mat()}, nlgeom=False, time_integrator="newmark")
    assert lumped_mass(s, 3)[0::3].sum() == pytest.approx(RHO * 15.0, rel=1e-12)


def test_total_mass_cax4_ring():
    """(Passes on 1e214a4 too: body_force_weights already integrated the ring volume.)"""
    mesh = Mesh2D()
    for i, (r, z) in enumerate([(1, 0), (3, 0), (3, 2), (1, 2)], 1):
        mesh.add_node(i, r, z)
    mesh.add_element(1, [1, 2, 3, 4], "CAX4", pid=0)
    s = DynamicSolver2D(mesh, materials={0: _mat()}, nlgeom=False, time_integrator="newmark")
    ring = 2 * np.pi * 2.0 * 4.0                       # 2 pi r_c A
    assert lumped_mass(s, 2)[0::2].sum() == pytest.approx(RHO * ring, rel=1e-12)


def test_point_mass_element_is_refused_in_dynamics():
    mesh = Mesh3D()
    xyz = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for i, p in enumerate(xyz, 1):
        mesh.add_node(i, *p)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    mesh.add_element(2, [7], "MASS", pid=0)
    s = DynamicSolver3D(mesh, materials={0: _mat()}, nlgeom=False, time_integrator="newmark")
    for n in (1, 4, 5, 8):
        for d in range(3):
            s.fix_dof(n, d, 0.0)
    with pytest.raises(NotImplementedError, match="MASS"):
        s.solve_step(dt=1e-3)                          # 1e214a4: ran with the point mass missing


def test_lumped_mass_cache_follows_regrouping():
    mesh = Mesh3D()
    xyz = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for i, p in enumerate(xyz, 1):
        mesh.add_node(i, *p)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    s = DynamicSolver3D(mesh, materials={0: _mat()}, nlgeom=True, time_integrator="newmark")
    m1 = lumped_mass(s, 3)
    s.nlgeom = False
    s._build_kernel_groups()
    assert s._lumped_mass_cache is None
    assert lumped_mass(s, 3) is not m1


# ── initial acceleration with constraints and prescribed motion ────────────

def test_initial_acceleration_respects_tie_constraint():
    """Two unit cubes side by side, the right cube's left face tied (*TIE, penalty) to the
    left cube's right face, a load f on each tied slave node. Each tied pair must start
    with ONE acceleration, f / (m_slave + m_master); per-DOF M^-1 f gave the slave
    f / m_slave and its master 0 (1e214a4)."""
    from dispsolver.constraint3d.surface_tie3d import SurfaceTieConstraint3D
    mesh = Mesh3D()
    left = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    right = [(1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 0, 1), (2, 0, 1), (2, 1, 1), (1, 1, 1)]
    for i, p in enumerate(left + right, 1):
        mesh.add_node(i, *p)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    mesh.add_element(2, list(range(9, 17)), "C3D8", pid=0)
    s = DynamicSolver3D(mesh, materials={0: _mat()}, nlgeom=False, time_integrator="hht-tf")
    nm = mesh.node_id_to_index()
    s.add_constraint(SurfaceTieConstraint3D([9, 12, 13, 16], [(2, 3, 7, 6)], nm, mesh.nodes_array(),
                                            penalty_stiffness=1e6))
    for n in mesh.nodes:                               # 1D in x: lateral DOFs fixed
        s.fix_dof(n, 1, 0.0)
        s.fix_dof(n, 2, 0.0)
    f = np.zeros(s.num_dofs)
    for n in (9, 12, 13, 16):
        f[3 * nm[n]] = 0.1
    a0 = initial_acceleration(s, f, 1e-3)
    want = 0.1 / (2 * RHO / 8)
    for slave, master in ((9, 2), (12, 3), (13, 6), (16, 7)):
        assert a0[3 * nm[slave]] == pytest.approx(want, rel=1e-6)
        assert a0[3 * nm[master]] == pytest.approx(want, rel=1e-6)
    assert a0[3 * nm[1]] == pytest.approx(0.0, abs=1e-6 * want)   # untied, unloaded node


def test_initial_acceleration_prescribed_dof_takes_bc_acceleration():
    """A prescribed DOF driven by u = U t^2 (smooth start) has a0 = 2U at t0+."""
    from dispsolver.solver.implicit_dynamics import prescribed_acceleration

    class _Plan:
        def prescribed(self, t, u0):
            return np.array([0]), np.array([0.5 * 3.0 * t * t])

    acc = prescribed_acceleration(_Plan(), 0.0, 1e-2, None)
    assert acc[0] == pytest.approx(3.0, rel=1e-6)


# ── 2D Rayleigh damping is applied, not ignored ────────────────────────────

def test_2d_mass_proportional_damping_decays_at_theory_rate():
    """CPE4 SDOF rod (K = E H / L, M = rho L H / 2), mass-proportional damping
    C = a M, xi = a / (2 w). 1e214a4's 2D branch ignored alpha_m_damp (no decay)."""
    mesh = Mesh2D()
    for i, (x, y) in enumerate([(0, 0), (5, 0), (5, 2), (0, 2)], 1):
        mesh.add_node(i, x, y)
    mesh.add_element(1, [1, 2, 3, 4], "CPE4", pid=0)
    K, M = E * 2.0 / 5.0, RHO * 10.0 / 2.0
    om = np.sqrt(K / M)
    xi = 0.02
    s = DynamicSolver2D(mesh, materials={0: _mat()}, nlgeom=False, time_integrator="newmark",
                        alpha_m_damp=2 * xi * om)
    for n in (1, 2, 3, 4):
        s.fix_dof(n, 1, 0.0)
    for n in (1, 4):
        s.fix_dof(n, 0, 0.0)
    nm = mesh.node_id_to_index()
    tip = [2 * nm[2], 2 * nm[3]]
    s.u[tip] = 1e-4
    dt = 2 * np.pi / om / 80
    initial_acceleration(s, np.zeros(s.num_dofs), dt)
    h = []
    for _ in range(80 * 4):
        ok, _ = s.solve_step(dt=dt, f_ext=np.zeros(s.num_dofs), f_ext_n=np.zeros(s.num_dofs))
        assert ok
        h.append(s.u[tip[0]])
    h = np.array(h)
    peaks = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0]
    assert len(peaks) >= 3
    ratio = np.mean([h[peaks[i + 1]] / h[peaks[i]] for i in range(len(peaks) - 1)])
    assert ratio == pytest.approx(np.exp(-2 * np.pi * xi / np.sqrt(1 - xi ** 2)), rel=1e-2)
