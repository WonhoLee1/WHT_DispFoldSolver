"""
test_viscoelastic.py
====================
Phase 3 tests: Viscoelastic Prony series + WLF TTS.

Verification:
1. Single-step response matches exact linearized midpoint rule γ·(1-exp(-Δt/τ))/(Δt/τ)
2. Multiple small steps converge to continuous-time exp(-t/τ)
3. WLF shift factor matches known reference values
4. Instantaneous and equilibrium limits correct
5. Internal variable initialization and evolution

Reference: Simo & Hughes (1998) Ch. 10.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from dispsolver.material import NeoHookean, ViscoelasticMaterial
from dispsolver.material.viscoelastic import wlf_shift
from dispsolver.state import TemperatureField, InternalVarStore


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def nh_base():
    return NeoHookean()


@pytest.fixture
def nh_params():
    return {"E": 1000.0, "nu": 0.3}


@pytest.fixture
def sp(nh_base):
    """Single Prony term: g_1 = 0.3, tau_1 = 1.0."""
    return ViscoelasticMaterial(nh_base, g_i=[0.3], tau_i=[1.0])


@pytest.fixture
def tp(nh_base):
    """Three Prony terms."""
    return ViscoelasticMaterial(nh_base, g_i=[0.2, 0.15, 0.1],
                                tau_i=[0.1, 1.0, 10.0])


@pytest.fixture
def wlf_mat(nh_base):
    """Single Prony + WLF."""
    return ViscoelasticMaterial(nh_base, g_i=[0.3], tau_i=[1.0],
                                wlf_params={"C1": 17.44, "C2": 51.6, "T_ref": 20.0})


# =====================================================================
# Helpers
# =====================================================================

def _split_voigt(mat, F, params):
    """Return (S_vol_voigt, S_dev_voigt) for a deformation."""
    S3 = np.asarray(mat.base.pk2_tensor(F, params))
    from dispsolver.material.viscoelastic import _volumetric_stress
    S_vol = _volumetric_stress(F, params)
    S_dev = S3 - S_vol
    def v(S): return np.array([S[0, 0], S[1, 1], S[0, 1]])
    return v(S_vol), v(S_dev)


# =====================================================================
# Section 1 — Prony series: stress relaxation
# =====================================================================

class TestPronySingleStep:
    """Single-step response: the discrete integration formula is exact
    for a linear variation of S_dev_el over the step."""

    F = np.array([[1.1, 0.0], [0.0, 0.95]], dtype=np.float64)

    def test_instantaneous_equals_elastic(self, sp, nh_params):
        """At dt/tau -> 0, gamma -> 1, so S_eff = S_el."""
        S_vol_v, S_dev_v = _split_voigt(sp, self.F, nh_params)
        S_el = S_vol_v + S_dev_v

        h0 = sp.initial_internal_vars()
        S_eff, _ = sp.pk2_voigt(self.F, nh_params, h0, dt=1e-14)
        np.testing.assert_allclose(S_eff, S_el, rtol=1e-10)

    def test_equilibrium_stress(self, sp, nh_params):
        """After many steps with large dt, h_i -> 0."""
        h = sp.initial_internal_vars()
        for _ in range(30):
            _, h = sp.pk2_voigt(self.F, nh_params, h, dt=50.0)
        for i in range(sp.M):
            assert np.linalg.norm(h[i]) < 1e-12, f"h[{i}] not decayed"
        S_vol_v, S_dev_v = _split_voigt(sp, self.F, nh_params)
        S_eq = S_vol_v + sp.g_inf * S_dev_v
        S_eff, _ = sp.pk2_voigt(self.F, nh_params, h, dt=50.0)
        np.testing.assert_allclose(S_eff, S_eq, rtol=1e-10)

    def test_single_step_gamma_exact(self, sp, nh_params):
        """Single-step stress matches the linearized midpoint rule exactly."""
        S_vol_v, S_dev_v = _split_voigt(sp, self.F, nh_params)
        h0 = sp.initial_internal_vars()

        dts = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
        for dt in dts:
            gamma = (1.0 - np.exp(-dt/sp.tau_i[0])) / (dt/sp.tau_i[0])
            S_ref = S_vol_v + (sp.g_inf + sp.g_i[0] * gamma) * S_dev_v
            S_eff, _ = sp.pk2_voigt(self.F, nh_params, h0, dt=dt)
            np.testing.assert_allclose(S_eff, S_ref, rtol=1e-12,
                                       err_msg=f"dt={dt} single-step mismatch")


class TestPronyMultiStep:
    """Multi-step convergence: as dt -> 0, numerical -> continuous Prony."""

    F = np.array([[1.1, 0.0], [0.0, 0.95]], dtype=np.float64)

    @staticmethod
    def _gamma(dt, tau):
        return (1.0 - np.exp(-dt / tau)) / (dt / tau) if dt > 0 else 1.0

    def test_convergence_small_dt(self, sp, nh_params):
        """Numerical and analytic converge as dt → 0."""
        S_vol_v, S_dev_v = _split_voigt(sp, self.F, nh_params)
        tau = float(sp.tau_i[0])

        n_steps = 200
        dt = 5e-3

        h = sp.initial_internal_vars()
        t = 0.0
        errors = []
        for step in range(n_steps):
            t += dt
            S_eff, h = sp.pk2_voigt(self.F, nh_params, h, dt)
            if step < 10:
                continue
            g_t = sp.g_inf + sp.g_i[0] * np.exp(-t / tau)
            S_ref = S_vol_v + g_t * S_dev_v
            denom = max(np.linalg.norm(S_ref), 1e-30)
            errors.append(np.linalg.norm(S_eff - S_ref) / denom)

        avg_err = np.mean(errors)
        # dt=5e-3 gives O(dt) ~ 5e-3 in the h recurrence
        assert avg_err < 5e-3, f"Convergence L2 = {avg_err:.2e} > 5e-3"

    def test_triple_prony_convergence(self, tp, nh_params):
        """Three Prony terms converge with small dt."""
        F = np.array([[1.15, 0.0], [0.0, 0.90]], dtype=np.float64)
        S_vol_v, S_dev_v = _split_voigt(tp, F, nh_params)

        n_steps = 200
        dt = 5e-3

        h = tp.initial_internal_vars()
        t = 0.0
        errors = []
        for step in range(n_steps):
            t += dt
            S_eff, h = tp.pk2_voigt(F, nh_params, h, dt)
            if step < 10:
                continue
            g_t = tp.g_inf + np.sum(tp.g_i * np.exp(-t / tp.tau_i))
            S_ref = S_vol_v + g_t * S_dev_v
            denom = max(np.linalg.norm(S_ref), 1e-30)
            errors.append(np.linalg.norm(S_eff - S_ref) / denom)

        assert np.mean(errors) < 5e-3, f"Triple Prony L2 = {np.mean(errors):.2e} > 5e-3"

    def test_overstress_decays(self, sp, nh_params):
        """At constant F+large dt, overstress h_i -> 0."""
        F = np.array([[1.05, 0.01], [0.01, 1.02]], dtype=np.float64)
        h = sp.initial_internal_vars()
        for _ in range(30):
            _, h = sp.pk2_voigt(F, nh_params, h, dt=50.0)
        for i in range(sp.M):
            assert np.linalg.norm(h[i]) < 1e-12

    def test_sdev_stored(self, sp, nh_params):
        """h[M] stores the current S_dev_el for next step's increment."""
        F = np.array([[1.05, 0.01], [0.01, 1.02]], dtype=np.float64)
        _, h1 = sp.pk2_voigt(F, nh_params, sp.initial_internal_vars(), dt=0.5)
        assert np.linalg.norm(h1[sp.M]) > 0


# =====================================================================
# Section 2 — WLF
# =====================================================================

class TestWLFShift:
    def test_at_reference(self):
        assert wlf_shift(20.0, 17.44, 51.6, 20.0) == pytest.approx(1.0)

    def test_higher_temp_faster(self):
        assert wlf_shift(10.0, 17.44, 51.6, 20.0) > 1.0
        assert wlf_shift(30.0, 17.44, 51.6, 20.0) < 1.0

    def test_known_values(self):
        aT = wlf_shift(40.0, 17.44, 51.6, 20.0)
        log_ref = -17.44 * 20.0 / (51.6 + 20.0)
        assert aT == pytest.approx(10.0 ** log_ref, rel=1e-10)

    def test_accelerates(self, wlf_mat, nh_params):
        F = np.array([[1.1, 0.0], [0.0, 0.95]], dtype=np.float64)
        h_c = wlf_mat.initial_internal_vars()
        h_h = wlf_mat.initial_internal_vars()
        _, h_c = wlf_mat.pk2_voigt(F, nh_params, h_c, dt=1.0, temperature=10.0)
        _, h_h = wlf_mat.pk2_voigt(F, nh_params, h_h, dt=1.0, temperature=60.0)
        n_c = np.linalg.norm(h_c[:wlf_mat.M])
        n_h = np.linalg.norm(h_h[:wlf_mat.M])
        assert n_h < n_c

    def test_temperaturefield(self, wlf_mat, nh_params):
        F = np.array([[1.1, 0.0], [0.0, 0.95]], dtype=np.float64)
        h = wlf_mat.initial_internal_vars()
        tf = TemperatureField(default_temp=80.0)
        T = tf.get_node_temperature(1)
        S_hot, _ = wlf_mat.pk2_voigt(F, nh_params, h, dt=1.0, temperature=T)
        S_amb, _ = wlf_mat.pk2_voigt(F, nh_params, h, dt=1.0)
        assert not np.allclose(S_hot, S_amb, rtol=1e-3)


# =====================================================================
# Section 3 — Tangent
# =====================================================================

class TestTangent:
    def test_instantaneous_equals_elastic(self, sp, nh_params):
        F = np.eye(2, dtype=np.float64)
        C_el = np.asarray(sp.base.tangent_voigt(F, nh_params))
        C_eff = sp.tangent_voigt(F, nh_params, dt=1e-14)
        np.testing.assert_allclose(C_eff, C_el, rtol=1e-10)

    def test_equilibrium_scaling(self, sp, nh_params):
        F = np.eye(2, dtype=np.float64)
        C_el = np.asarray(sp.base.tangent_voigt(F, nh_params))
        C_eff = sp.tangent_voigt(F, nh_params, dt=1e12)
        m = np.array([1.0, 1.0, 0.0])
        K = (C_el[0, 0] + C_el[0, 1] + C_el[1, 0] + C_el[1, 1]) / 4.0
        C_vol = K * np.outer(m, m)
        C_ref = C_vol + sp.g_inf * (C_el - C_vol)
        np.testing.assert_allclose(C_eff, C_ref, rtol=1e-10)


# =====================================================================
# Section 4 — Internal var store
# =====================================================================

class TestInternalVarStore:
    def test_init(self):
        s = InternalVarStore(n_elem=10, n_gp=4, n_vars=6)
        assert s.data.shape == (10, 4, 6)
        assert np.all(s.data == 0.0)

    def test_get_set(self):
        s = InternalVarStore(n_elem=3, n_gp=2, n_vars=3)
        v = np.random.randn(2, 3)
        s.set_elem(1, v)
        np.testing.assert_array_equal(s.get_elem(1), v)

    def test_copy(self):
        s = InternalVarStore(n_elem=5, n_gp=3, n_vars=9)
        s.data[:] = np.random.randn(5, 3, 9)
        c = s.copy()
        c.data[0, 0, 0] = 999.0
        assert s.data[0, 0, 0] != 999.0

    def test_var_size(self, sp):
        assert sp.n_internal_vars == 6 * (sp.M + 1)

    def test_var_names(self, sp):
        names = sp.internal_var_names(sp.M)
        assert len(names) == 6 * (sp.M + 1)
        assert "Sdev_prev_11" in names

    def test_initial_shape(self, sp):
        h0 = sp.initial_internal_vars()
        assert h0.shape == (sp.M + 1, 3, 3)
        assert np.all(h0 == 0.0)

    def test_evolution(self, sp, nh_params):
        F = np.array([[1.05, 0.01], [0.01, 1.02]], dtype=np.float64)
        _, h1 = sp.pk2_voigt(F, nh_params, sp.initial_internal_vars(), dt=0.5)
        assert np.any(h1 != 0.0)


# =====================================================================
# Section 5 — TemperatureField
# =====================================================================

class TestTemperatureField:
    def test_default(self):
        tf = TemperatureField(default_temp=25.0)
        assert tf.get_node_temperature(42) == 25.0

    def test_nodal(self):
        tf = TemperatureField(nodal_temperatures={1: 50.0})
        assert tf.get_node_temperature(1) == 50.0
        assert tf.get_node_temperature(99) == 20.0

    def test_element(self):
        tf = TemperatureField(element_temperatures={5: 80.0})
        assert tf.get_element_temperature(5) == 80.0
        assert tf.get_element_temperature(99) == 20.0

    def test_node_array(self):
        tf = TemperatureField(nodal_temperatures={0: 10.0, 1: 20.0, 3: 30.0})
        arr = tf.node_temperature_array([0, 1, 2, 3, 4], {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})
        np.testing.assert_array_equal(arr, [10.0, 20.0, 20.0, 30.0, 20.0])

    def test_elem_array(self):
        tf = TemperatureField(element_temperatures={0: 15.0, 2: 25.0})
        np.testing.assert_array_equal(tf.element_temperature_array([0, 1, 2]),
                                       [15.0, 20.0, 25.0])


# =====================================================================
# Section 6 — Imports
# =====================================================================

class TestImports:
    def test_ve_export(self):
        from dispsolver.material import ViscoelasticMaterial
        assert ViscoelasticMaterial is not None

    def test_state_export(self):
        from dispsolver.state import TemperatureField, InternalVarStore
        assert TemperatureField is not None and InternalVarStore is not None
