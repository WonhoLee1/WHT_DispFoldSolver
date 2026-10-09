# -*- coding: utf-8 -*-
"""Abaqus/Standard *DYNAMIC (HHT) end to end -- dev_log/plan_dynamic_hht_20261009.md.

Physics, not just convergence: every solver test compares against a closed form or
against the HHT operator's own amplification matrix. The D3-D6 tests each fail on the
pre-fix code (numbers in the plan's hand-off report).
"""
import numpy as np
import pytest

from dispsolver.io.abaqus_reader import execute_inp, read_inp
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d import Mesh3D
from dispsolver.model.material import Material
from dispsolver.solver.implicit_dynamics import initial_acceleration, lumped_mass
from dispsolver.solver.time_integrator_presets import resolve_time_integrator
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

E, RHO = 1000.0, 1e-3


def _mat(density=RHO):
    m = Material(name="BAR")
    m.Elastic(E=E, nu=0.0)
    if density:
        m.density = density
    return m


def _rod3d(n_el, h=1.0, by=1.0, bz=1.0, integ="newmark", **kw):
    """Axial rod of n_el C3D8 along x; y/z fixed everywhere, x fixed at x=0 (1D rod)."""
    mesh = Mesh3D()
    nid = 0
    ids = []
    for i in range(n_el + 1):
        x = i * h
        layer = []
        for (y, z) in ((0, 0), (by, 0), (by, bz), (0, bz)):
            nid += 1
            mesh.add_node(nid, x, y, z)
            layer.append(nid)
        ids.append(layer)
    for i in range(n_el):
        a, b = ids[i], ids[i + 1]
        mesh.add_element(i + 1, [a[0], b[0], b[1], a[1], a[3], b[3], b[2], a[2]], "C3D8", pid=0)
    s = DynamicSolver3D(mesh, materials={0: _mat()}, nlgeom=False, time_integrator=integ, **kw)
    for n in mesh.nodes:
        s.fix_dof(n, 1, 0.0)
        s.fix_dof(n, 2, 0.0)
    for n in ids[0]:
        s.fix_dof(n, 0, 0.0)
    nmap = mesh.node_id_to_index()
    tip = [3 * nmap[n] for n in ids[-1]]
    return s, tip


# ── D7: unknown integrator names ───────────────────────────────────────────

def test_unknown_time_integrator_name_raises():
    with pytest.raises(ValueError, match="unknown time_integrator"):
        resolve_time_integrator("hhtt")
    with pytest.raises(ValueError):
        resolve_time_integrator("generalized-abc")
    tf = resolve_time_integrator("hht-tf")
    md = resolve_time_integrator("hht-md")
    assert (tf.alpha_hht, tf.beta, tf.gamma) == (-0.05, 0.275625, 0.55)
    assert md.alpha_hht == -0.41421 and abs(md.beta - 0.5) < 1e-5 and abs(md.gamma - 0.91421) < 1e-12


# ── D3: lumped mass = rho * V ──────────────────────────────────────────────

def test_d3_lumped_mass_3d_bar_2x3x5():
    s, _ = _rod3d(1, h=2.0, by=3.0, bz=5.0)
    s.solve_step(dt=1e-3)                       # populates M_global (old code: rho/8 per DOF)
    m = s.M_global.diagonal()
    for d in range(3):
        assert m[d::3].sum() == pytest.approx(RHO * 30.0, rel=1e-12)


def test_d3_lumped_mass_2d_quad_area_6():
    mesh = Mesh2D()
    for i, (x, y) in enumerate([(0, 0), (2, 0), (2, 3), (0, 3)], 1):
        mesh.add_node(i, x, y)
    mesh.add_element(1, [1, 2, 3, 4], "CPE4", pid=0)
    s = DynamicSolver2D(mesh, materials={0: _mat()}, nlgeom=False, time_integrator="newmark")
    s.fix_dof(1, 0, 0.0)
    s.fix_dof(1, 1, 0.0)
    s.fix_dof(4, 0, 0.0)
    s.solve_step(dt=1e-3)                       # old code: elem_density never filled -> M = 0
    m = s.M_global.diagonal()
    assert m[0::2].sum() == pytest.approx(RHO * 6.0, rel=1e-12)
    assert m[1::2].sum() == pytest.approx(RHO * 6.0, rel=1e-12)


def test_d3_quadratic_elements_get_positive_lumped_mass():
    from dispsolver.loads.integration import lumped_mass_weights
    t = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 5]], float)
    mid = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    t10 = np.vstack([t, [(t[a] + t[b]) / 2 for a, b in mid]])
    w = lumped_mass_weights(t10, "C3D10")         # row sum would give corners -V/20
    assert np.all(w > 0) and w.sum() == pytest.approx(5.0, rel=1e-12)
    q8 = np.array([[0, 0], [2, 0], [2, 3], [0, 3], [1, 0], [2, 1.5], [1, 3], [0, 1.5]], float)
    w = lumped_mass_weights(q8, "CPE8")
    assert np.all(w > 0) and w.sum() == pytest.approx(6.0, rel=1e-12)


# ── D4: HHT weights f_ext like f_int ───────────────────────────────────────

@pytest.mark.parametrize("integ", ["hht-md", "moderate-4"])
def test_d4_linear_ramp_particular_solution_exact(integ):
    """x = F(t)/K, v = Fdot/K, a = 0 solves M a + K x = F exactly for a linear ramp, and
    HHT reproduces it to round-off -- but only if f_ext carries the same (1+alpha, -alpha)
    weights as f_int. Mass-independent (a = 0), so it isolates D4."""
    s, tip = _rod3d(1, h=5.0, by=2.0, bz=3.0, integ=integ)
    K = E * 6.0 / 5.0
    om = np.sqrt(K / (RHO * 30.0 / 2))
    dt = 2 * np.pi / om / 20
    F0, Tr = 1.0, 200 * dt
    for d in tip:
        s.v[d] = F0 / (K * Tr)

    def f(t):
        v = np.zeros(s.num_dofs)
        v[tip] = F0 * t / Tr / 4
        return v

    for n in range(1, 41):
        ok, _ = s.solve_step(dt=dt, f_ext=f(n * dt), f_ext_n=f((n - 1) * dt), tol=1e-12)
        assert ok
    exact = F0 * 40 * dt / (K * Tr)
    assert abs(s.u[tip[0]] - exact) / exact < 1e-9      # old code: 6.3e-3 (alpha=-0.22)


# ── axial bar free vibration: period, energy, HHT decay ────────────────────

def _first_mode(s):
    K, _ = s.assemble_system(s.u, dt=1.0, update_state=False)
    K = K.toarray()
    m = lumped_mass(s, 3)
    free = np.array([i for i in range(s.num_dofs) if i not in s.fixed_dofs])
    Kf, mf = K[np.ix_(free, free)], m[free]
    from scipy.linalg import eigh
    w2, vec = eigh(Kf, np.diag(mf))
    phi = np.zeros(s.num_dofs)
    phi[free] = vec[:, 0]
    return float(np.sqrt(w2[0])), phi, K, m


def _hht_amplification(Om, alpha, beta, gamma):
    """HHT amplification matrix on x'' + Om^2 x = 0 in the scaled state (x, v dt, a dt^2)
    (Hilber, Hughes & Taylor 1977), and its spectral radius (oscillatory pair)."""
    A = np.zeros((3, 3))
    for j in range(3):
        x, v, a = np.eye(3)[j]
        c = 1.0 / beta
        # a1 = c(x1 - x) - c v - (c/2 - 1) a ; a1 + (1+alpha) Om^2 x1 - alpha Om^2 x = 0
        x1 = (c * x + c * v + (c / 2 - 1) * a + alpha * Om ** 2 * x) / (c + (1 + alpha) * Om ** 2)
        a1 = c * (x1 - x) - c * v - (c / 2 - 1) * a
        v1 = v + (1 - gamma) * a + gamma * a1
        A[:, j] = (x1, v1, a1)
    lam = np.linalg.eigvals(A)
    osc = lam[np.abs(lam.imag) > 1e-14]
    return A, float(max(abs(osc)))


def _run_mode(integ, n_el=10, steps_per_period=40, periods=3):
    s, tip = _rod3d(n_el, integ=integ)
    om, phi, K, m = _first_mode(s)                    # phi is M-orthonormal
    T = 2 * np.pi / om
    dt = T / steps_per_period
    s.u[:] = 1e-4 * phi / np.abs(phi).max()
    s.v[:] = 0.0
    free = m > 0
    s.a[:] = 0.0
    s.a[free] = -(K @ s.u)[free] / m[free]
    for d in s.fixed_dofs:
        s.a[d] = 0.0
    s.q0 = float(phi @ (m * s.u))
    q, energy = [], []
    for _ in range(steps_per_period * periods):
        ok, _ = s.solve_step(dt=dt, tol=1e-12)
        assert ok
        q.append(float(phi @ (m * s.u)))
        energy.append(0.5 * s.u @ (K @ s.u) + 0.5 * s.v @ (m * s.v))
    return np.array(q), np.array(energy), om, dt, s


def test_bar_free_vibration_newmark_period_and_energy():
    q, energy, om, dt, s = _run_mode("newmark", steps_per_period=40, periods=3)
    # trapezoidal rule period elongation: Om_bar = 2 atan(Om/2)
    Om = om * dt
    T_num = 2 * np.pi / (2 * np.arctan(Om / 2) / dt)
    up = [i for i in range(1, len(q)) if q[i - 1] < 0 <= q[i]]
    xs = [(i - 1) + q[i - 1] / (q[i - 1] - q[i]) for i in up]     # linear zero crossings
    T_meas = float(np.mean(np.diff(xs))) * dt
    assert T_meas == pytest.approx(T_num, rel=2e-3)
    # discrete vs continuous rod (fixed-free, L = 10): T = 4 L / c, lumped 10-element mesh
    T_cont = 4 * 10.0 / np.sqrt(E / RHO)
    assert 2 * np.pi / om == pytest.approx(T_cont, rel=1e-2)
    # Newmark trapezoidal rule conserves the energy of a linear system
    assert np.max(np.abs(energy - energy[0])) / energy[0] < 1e-8


def test_bar_free_vibration_hht_decay_matches_spectral_radius():
    """The rod's first mode, integrated with HHT alpha = -0.3, follows the SDOF HHT
    amplification matrix step by step, so its amplitude decays at the spectral radius."""
    alpha = -0.3
    spp, periods = 6, 8
    q, _, om, dt, s = _run_mode("moderate-5", steps_per_period=spp, periods=periods)
    A, rho = _hht_amplification(om * dt, alpha, (1 - alpha) ** 2 / 4, 0.5 - alpha)
    assert rho < 0.99                                  # genuinely dissipative at Om = 2 pi / 6
    q0 = s.q0
    state = np.array([q0, 0.0, -(om * dt) ** 2 * q0])  # (q, q' dt, q'' dt^2), consistent a0
    pred = []
    for _ in range(spp * periods):
        state = A @ state
        pred.append(state[0])
    pred = np.array(pred)
    assert np.max(np.abs(q - pred)) / q0 < 1e-8
    # amplitude after 8 periods ~ rho^48 (the spurious root has long died out)
    env = np.max(np.abs(q[-spp:])) / np.max(np.abs(q[:spp]))
    assert env == pytest.approx(rho ** (spp * (periods - 1)), rel=0.1)


# ── D5: initial acceleration ───────────────────────────────────────────────

def test_d5_initial_acceleration_is_minv_f_on_free_dofs():
    s, tip = _rod3d(1, h=5.0, by=2.0, bz=3.0, integ="hht-tf")
    f = np.zeros(s.num_dofs)
    f[tip] = 0.25
    a0 = initial_acceleration(s, f, dt=1e-4)
    m = lumped_mass(s, 3)
    for d in range(s.num_dofs):
        if d in s.fixed_dofs:
            assert a0[d] == 0.0
        elif m[d] > 0:
            assert a0[d] == pytest.approx(f[d] / m[d], rel=1e-12)
    assert a0[tip[0]] == pytest.approx(0.25 / (RHO * 30.0 / 8), rel=1e-12)


def _bar_deck(dyn_line, data_line, density=True, extra_step="", dim=3):
    """2 x 3 cross-section, L = 5 single C3D8 rod: lateral DOFs fixed, root x fixed,
    CLOAD F = 1 on the tip face. SDOF: K = E A / L = 1200, M = rho V / 2 = 0.015."""
    dens = "*DENSITY\n1.0e-3\n" if density else ""
    return f"""*HEADING
SDOF rod
*NODE
1, 0., 0., 0.
2, 5., 0., 0.
3, 5., 2., 0.
4, 0., 2., 0.
5, 0., 0., 3.
6, 5., 0., 3.
7, 5., 2., 3.
8, 0., 2., 3.
*ELEMENT, TYPE=C3D8, ELSET=ROD
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=ROOT
1, 4, 5, 8
*NSET, NSET=TIP
2, 3, 6, 7
*NSET, NSET=ALLN
1, 2, 3, 4, 5, 6, 7, 8
*MATERIAL, NAME=RODMAT
*ELASTIC
1000.0, 0.0
{dens}*SOLID SECTION, ELSET=ROD, MATERIAL=RODMAT
*BOUNDARY
ALLN, 2, 3, 0.0
ROOT, 1, 1, 0.0
*STEP, NAME=DYN, NLGEOM=NO
{dyn_line}
{data_line}
*CLOAD
TIP, 1, 0.25
{extra_step}*END STEP
"""


K_SDOF, M_SDOF = 1200.0, 0.015
OM_SDOF = np.sqrt(K_SDOF / M_SDOF)


def _tip(res):
    return float(res["u"][3 * 1])                     # node 2, x


def test_d5_deck_single_increment_uses_initial_acceleration():
    dt = 2 * np.pi / OM_SDOF / 40
    res = execute_inp(_bar_deck("*DYNAMIC, ALPHA=0.0", f"{dt}, {dt}, , {dt}"), dim=3, verbose=False)
    assert res["converged"]
    # trapezoidal rule, one step from rest with a0 = F/M: (K + 4M/dt^2) x1 = 2F
    x1 = 2.0 / (K_SDOF + 4 * M_SDOF / dt ** 2)
    assert _tip(res) == pytest.approx(x1, rel=1e-8)
    exact = (1 - np.cos(OM_SDOF * dt)) / K_SDOF
    assert _tip(res) == pytest.approx(exact, rel=1e-2)
    # INITIAL=NO: a0 stays 0 -> (K + 4M/dt^2) x1 = F, half the response
    res0 = execute_inp(_bar_deck("*DYNAMIC, ALPHA=0.0, INITIAL=NO", f"{dt}, {dt}, , {dt}"),
                       dim=3, verbose=False)
    assert _tip(res0) == pytest.approx(1.0 / (K_SDOF + 4 * M_SDOF / dt ** 2), rel=1e-8)


def test_deck_dynamic_step_runs_dynamically_with_application_and_alpha():
    T = 2 * np.pi / OM_SDOF
    dt = T / 80
    res = execute_inp(_bar_deck("*DYNAMIC, APPLICATION=MODERATE DISSIPATION, ALPHA=-0.02",
                                f"{dt}, {T / 2}, , {dt}"), dim=3, verbose=False)
    s = res["solver"]
    assert res["converged"]
    assert s.time_integrator == "hht-md"
    assert s.alpha_hht == -0.02
    assert s.beta_hht == pytest.approx(1.02 ** 2 / 4) and s.gamma_hht == pytest.approx(0.52)
    # step load from rest: x(T/2) = 2 F / K (dynamic amplification 2; static would be F/K)
    assert _tip(res) == pytest.approx(2.0 / K_SDOF, rel=1e-2)


def test_d6_dynamic_step_refuses_material_without_density():
    dt = 1e-4
    with pytest.raises(ValueError, match="RODMAT.*DENSITY"):
        execute_inp(_bar_deck("*DYNAMIC", f"{dt}, {10 * dt}", density=False), dim=3, verbose=False)
    deck2d = """*HEADING
*NODE
1, 0., 0.
2, 2., 0.
3, 2., 3.
4, 0., 3.
*ELEMENT, TYPE=CPE4, ELSET=Q
1, 1, 2, 3, 4
*MATERIAL, NAME=NODENS
*ELASTIC
1000.0, 0.0
*SOLID SECTION, ELSET=Q, MATERIAL=NODENS
1.0
*BOUNDARY
1, 1, 2
4, 1, 1
*STEP, NAME=DYN, NLGEOM=NO
*DYNAMIC
0.001, 0.01
*CLOAD
2, 1, 1.0
*END STEP
"""
    with pytest.raises(ValueError, match="NODENS.*DENSITY"):
        execute_inp(deck2d, dim=2, verbose=False)


def test_static_step_after_python_api_keeps_constructor_integrator():
    from dispsolver.solver.implicit_dynamics import step_time_integration

    class _S:                                         # Python-API step: no time_integration
        time_integration = None

    s, _ = _rod3d(1, integ="moderate-3")
    assert step_time_integration(s, _S()) is None
    assert s.time_integrator == "moderate-3" and s.alpha_hht == -0.15


# ── reader ─────────────────────────────────────────────────────────────────

def _step(text):
    m = read_inp(text, dim=3, strict=False)
    return m.steps["DYN"]


def test_reader_dynamic_defaults_and_overrides():
    ti = _step(_bar_deck("*DYNAMIC", "0.001, 0.1")).time_integration
    assert ti == {"name": "hht-tf", "application": "TRANSIENT FIDELITY", "alpha": -0.05,
                  "beta": 0.275625, "gamma": 0.55, "initial_acceleration": True}
    ti = _step(_bar_deck("*DYNAMIC, APPLICATION=MODERATE DISSIPATION", "0.001, 0.1")).time_integration
    assert ti["name"] == "hht-md" and ti["alpha"] == -0.41421
    assert ti["beta"] == pytest.approx(0.5, abs=1e-5) and ti["gamma"] == pytest.approx(0.91421)
    ti = _step(_bar_deck("*DYNAMIC, TIME INTEGRATOR=HHT-MD, ALPHA=-0.3, GAMMA=0.9", "0.001, 0.1")).time_integration
    assert (ti["name"], ti["alpha"], ti["gamma"]) == ("hht-md", -0.3, 0.9)
    assert ti["beta"] == pytest.approx(1.3 ** 2 / 4)
    st = _step(_bar_deck("*DYNAMIC, INITIAL=NO, NOHAF, IMPACT=NO", "0.001, 0.1"))
    assert st.time_integration["initial_acceleration"] is False
    assert st.dt_max == pytest.approx(0.1 / 100)       # TRANSIENT FIDELITY: T / 100
    st = _step(_bar_deck("*DYNAMIC, APPLICATION=MODERATE DISSIPATION", "0.001, 0.1"))
    assert st.dt_max == pytest.approx(0.1 / 10)        # MODERATE DISSIPATION: T / 10


def test_reader_static_step_is_quasistatic():
    st = _step(_bar_deck("*STATIC", "0.1, 1.0"))
    assert st.time_integration == {"name": "quasistatic"}


@pytest.mark.parametrize("line,level,code", [
    ("*DYNAMIC, ALPHA=-0.6", "error", None),
    ("*DYNAMIC, ALPHA=0.1", "error", None),
    ("*DYNAMIC, BETA=0.0", "error", None),
    ("*DYNAMIC, GAMMA=0.4", "error", None),
    ("*DYNAMIC, APPLICATION=QUASI-STATIC", "gap", "unsupported-value"),
    ("*DYNAMIC, TIME INTEGRATOR=BWE", "gap", "unsupported-value"),
    ("*DYNAMIC, EXPLICIT", "gap", "unsupported-param"),
])
def test_reader_refuses_what_it_cannot_represent(line, level, code):
    from dispsolver.io.abaqus_reader import _parse_source
    reader, model = _parse_source(_bar_deck(line, "0.001, 0.1"), dim=3, strict=False)
    diags = [d for d in reader.diagnostics if "DYNAMIC" in str(d.message)]
    assert diags and diags[0].level == level
    if code:
        assert diags[0].code == code
    assert model.steps["DYN"].procedure == "STATIC"    # the block is refused, not half-read
    assert model.steps["DYN"].time_integration is None


def test_reader_application_default_follows_contact():
    from dispsolver.io.keywords import steps as steps_kw

    class _R:
        class model:
            interactions = {"c": object()}
            steps = {}

    class _St:
        _dynamic_request = {}

    assert steps_kw._dynamic_application(_R, _St) == "MODERATE DISSIPATION"


def test_ex12_deck_step_reads_dynamic_alpha():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "ex12_rigid_plate_display_fold.inp")
    st = [s for n, s in read_inp(path, dim=2, strict=False).steps.items() if n != "Initial"][0]
    assert st.procedure == "DYNAMIC"
    assert st.time_integration["alpha"] == -0.22
    assert st.time_integration["name"] == "hht-md"


def test_legacy_adapter_forwards_hht_parameters():
    from dispsolver.io.legacy_adapter import model_to_builder_result
    model = read_inp(_bar_deck("*DYNAMIC, APPLICATION=MODERATE DISSIPATION, ALPHA=-0.2", "0.001, 0.1"),
                     dim=3, strict=False)
    cfg = model_to_builder_result(model).solver_config
    assert cfg["alpha"] == -0.2
    assert cfg["beta"] == pytest.approx(1.44 / 4) and cfg["gamma"] == pytest.approx(0.7)
