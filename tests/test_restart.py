"""*RESTART (PRD: dev_log/prd_restart_20261018.md).

The expected value is an invariant, not a number from the code under test: a solve that is written out,
read back into a NEW solver and continued must end where the uninterrupted solve ends.
"""
import numpy as np
import pytest

from dispsolver.io import read_inp
from dispsolver.solver.step_executor import MultiStepExecutor

_NODES_3D = "*NODE\n1,0,0,0\n2,1,0,0\n3,1,1,0\n4,0,1,0\n5,0,0,1\n6,1,0,1\n7,1,1,1\n8,0,1,1\n"


def _deck_3d(nlgeom: str = "NO", extra_step_lines: str = "", model_lines: str = "") -> str:
    """Unit cube pulled 5 % in x through an elastic-plastic law: SDV (plastic strain) evolves."""
    return (
        "*HEADING\nrestart probe\n" + _NODES_3D
        + "*ELEMENT, TYPE=C3D8, ELSET=EL\n1, 1,2,3,4,5,6,7,8\n"
        "*MATERIAL, NAME=M\n*ELASTIC\n1000.0, 0.3\n*PLASTIC\n10.0, 0.0\n20.0, 0.1\n"
        "*SOLID SECTION, ELSET=EL, MATERIAL=M\n"
        + model_lines
        + f"*STEP, NAME=S1, NLGEOM={nlgeom}\n*STATIC\n0.1, 1.0, 1e-4, 0.1\n"
        + extra_step_lines
        + "*BOUNDARY\n1, 1, 3, 0.0\n4, 1, 1, 0.0\n4, 3, 3, 0.0\n5, 1, 2, 0.0\n8, 1, 1, 0.0\n"
        "2, 1, 1, 0.05\n3, 1, 1, 0.05\n6, 1, 1, 0.05\n7, 1, 1, 0.05\n*END STEP\n"
    )


def _make(tmp_path, deck: str, dim: int = 3):
    path = tmp_path / "deck.inp"
    path.write_text(deck)
    model = read_inp(str(path), dim=dim)
    ex = MultiStepExecutor(model, verbose=False, dim=dim)
    ex.initialize_solver()
    return model, ex


def test_s1_state_roundtrips_through_a_file_into_a_new_solver(tmp_path):
    from dispsolver.solver import restart

    model, ex = _make(tmp_path, _deck_3d())
    ex.solver.solve(step=model.steps["S1"], sys=ex.sys, verbose=False)
    assert float(np.abs(ex.solver.elem_sdvs).max()) > 0.0       # the probe really carries history

    state = restart.capture(ex.solver, step_name="S1", step_time=1.0, step_inc=10, inc_no=11,
                            total_iters=0, dt_ctrl=None)
    path = tmp_path / "job.rst"
    restart.save(state, str(path))

    model2, ex2 = _make(tmp_path, _deck_3d())
    restart.apply(ex2.solver, restart.load(str(path)))
    np.testing.assert_array_equal(ex2.solver.u, ex.solver.u)
    np.testing.assert_array_equal(ex2.solver.v, ex.solver.v)
    np.testing.assert_array_equal(ex2.solver.elem_sdvs, ex.solver.elem_sdvs)


def _solve(model, ex, **kwargs):
    return ex.solver.solve(step=model.steps["S1"], sys=ex.sys, verbose=False, **kwargs)


@pytest.mark.parametrize("nlgeom", ["NO", "YES"])
def test_s2_resume_mid_step_matches_the_uninterrupted_solve(tmp_path, nlgeom):
    from dispsolver.solver import restart

    model, ex = _make(tmp_path, _deck_3d(nlgeom))
    assert _solve(model, ex)
    u_ref, sdv_ref = ex.solver.u.copy(), ex.solver.elem_sdvs.copy()

    # the same solve, asked to write a restart record every 5 converged increments
    model, ex = _make(tmp_path, _deck_3d(nlgeom))
    model.steps["S1"].restart_write = {"frequency": 5, "path": str(tmp_path / "job"), "overlay": False}
    assert _solve(model, ex)
    mid = tmp_path / "job_S1_0005.rst"
    assert mid.exists() and (tmp_path / "job_S1_0010.rst").exists()
    np.testing.assert_array_equal(ex.solver.u, u_ref)           # writing must not change the answer

    # a NEW solver continues from increment 5
    model, ex = _make(tmp_path, _deck_3d(nlgeom))
    assert _solve(model, ex, restart=restart.load(str(mid)))
    np.testing.assert_allclose(ex.solver.u, u_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ex.solver.elem_sdvs, sdv_ref, rtol=0, atol=1e-12)
    assert ex.solver.last_solve_stats["increments"] == 5        # only the remaining increments ran


def _deck_2d(nlgeom: str = "NO") -> str:
    """Plane-strain square pulled 5 % in x through an elastic-plastic law."""
    return (
        "*HEADING\nrestart probe 2D\n*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n"
        "*ELEMENT, TYPE=CPE4I, ELSET=EL\n1, 1,2,3,4\n"
        "*MATERIAL, NAME=M\n*ELASTIC\n1000.0, 0.3\n*PLASTIC\n10.0, 0.0\n20.0, 0.1\n"
        "*SOLID SECTION, ELSET=EL, MATERIAL=M\n"
        f"*STEP, NAME=S1, NLGEOM={nlgeom}\n*STATIC\n0.1, 1.0, 1e-4, 0.1\n"
        "*BOUNDARY\n1, 1, 2, 0.0\n4, 1, 1, 0.0\n2, 1, 1, 0.05\n3, 1, 1, 0.05\n*END STEP\n"
    )


@pytest.mark.parametrize("nlgeom", ["NO", "YES"])
def test_s2_resume_mid_step_2d(tmp_path, nlgeom):
    from dispsolver.solver import restart

    model, ex = _make(tmp_path, _deck_2d(nlgeom), dim=2)
    assert _solve(model, ex)
    u_ref, sdv_ref = ex.solver.u.copy(), ex.solver.elem_sdvs.copy()
    assert float(np.abs(sdv_ref).max()) > 0.0

    model, ex = _make(tmp_path, _deck_2d(nlgeom), dim=2)
    model.steps["S1"].restart_write = {"frequency": 5, "path": str(tmp_path / "job2d"), "overlay": False}
    assert _solve(model, ex)
    np.testing.assert_array_equal(ex.solver.u, u_ref)
    mid = tmp_path / "job2d_S1_0005.rst"
    assert mid.exists()

    model, ex = _make(tmp_path, _deck_2d(nlgeom), dim=2)
    assert _solve(model, ex, restart=restart.load(str(mid)))
    np.testing.assert_allclose(ex.solver.u, u_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ex.solver.elem_sdvs, sdv_ref, rtol=0, atol=1e-12)
    assert ex.solver.last_solve_stats["increments"] == 5


# ---- S3: the *RESTART keyword and execute_inp ------------------------------------------------------------

_STEP2 = (
    "*STEP, NAME=S2, NLGEOM=NO\n*STATIC\n0.1, 1.0, 1e-4, 0.1\n"
    "*BOUNDARY\n2, 1, 1, 0.0\n3, 1, 1, 0.0\n6, 1, 1, 0.0\n7, 1, 1, 0.0\n*END STEP\n"
)


def _two_step_deck(model_lines: str = "") -> str:
    return _deck_3d(model_lines=model_lines) + _STEP2


def _run(tmp_path, deck: str, **kwargs):
    from dispsolver.io import execute_inp
    path = tmp_path / "deck.inp"
    path.write_text(deck)
    return execute_inp(str(path), dim=3, verbose=False, **kwargs)


def test_s3_reader_stores_write_and_read_requests(tmp_path):
    path = tmp_path / "w.inp"
    path.write_text(_deck_3d(model_lines="*RESTART, WRITE, FREQUENCY=5, OVERLAY\n"))
    assert read_inp(str(path), dim=3).restart_write == {"frequency": 5, "overlay": True}
    path.write_text(_deck_3d(model_lines="*RESTART, READ, STEP=1, INC=5\n"))
    assert read_inp(str(path), dim=3).restart_read == {"step": 1, "inc": 5, "end_step": False}
    path.write_text(_deck_3d(model_lines="*RESTART, WRITE\n"))
    assert read_inp(str(path), dim=3).restart_write == {"frequency": 1, "overlay": False}   # Keywords Guide default


def test_s3_write_then_read_continues_to_the_same_answer(tmp_path):
    ref = _run(tmp_path, _deck_3d())
    wrote = _run(tmp_path, _deck_3d(model_lines="*RESTART, WRITE, FREQUENCY=5\n"), job="job", restart_dir=str(tmp_path))
    assert (tmp_path / "job_S1_0005.rst").exists() and (tmp_path / "job_S1_0010.rst").exists()
    np.testing.assert_array_equal(wrote["u"], ref["u"])
    cont = _run(tmp_path, _deck_3d(model_lines="*RESTART, READ, STEP=1, INC=5\n"),
                oldjob="job", restart_dir=str(tmp_path))
    np.testing.assert_allclose(cont["u"], ref["u"], rtol=0, atol=1e-12)
    assert cont["increments"] == 5


def test_s3_read_at_the_end_of_a_step_runs_only_the_later_steps(tmp_path):
    ref = _run(tmp_path, _two_step_deck())
    _run(tmp_path, _two_step_deck("*RESTART, WRITE, FREQUENCY=5\n"), job="job", restart_dir=str(tmp_path))
    cont = _run(tmp_path, _two_step_deck("*RESTART, READ, STEP=1, END STEP\n"), oldjob="job", restart_dir=str(tmp_path))
    assert cont["steps_executed"] == ["S2"]
    np.testing.assert_allclose(cont["u"], ref["u"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(cont["solver"].elem_sdvs, ref["solver"].elem_sdvs, rtol=0, atol=1e-12)


def test_s4_unsupported_write_parameter_is_reported_not_dropped(tmp_path):
    # NUMBER INTERVAL only changes WHEN records are written, never the answer, so the deck still runs; it
    # must not vanish silently (the project rule for unsupported parameters: the warning ledger)
    res = _run(tmp_path, _deck_3d(model_lines="*RESTART, WRITE, NUMBER INTERVAL=4\n"))
    assert any("NUMBER INTERVAL" in w and "unsupported" in w for w in res["warnings"])


def test_s4_read_without_a_record_says_which_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="job_S1_0005"):
        _run(tmp_path, _deck_3d(model_lines="*RESTART, READ, STEP=1, INC=5\n"), oldjob="job", restart_dir=str(tmp_path))


def test_s4_record_of_another_model_is_refused(tmp_path):
    _run(tmp_path, _deck_3d(model_lines="*RESTART, WRITE, FREQUENCY=5\n"), job="job", restart_dir=str(tmp_path))
    other = _deck_3d(model_lines="*RESTART, READ, STEP=1, INC=5\n").replace("1000.0, 0.3", "1500.0, 0.3")
    with pytest.raises(ValueError, match="different model"):
        _run(tmp_path, other, oldjob="job", restart_dir=str(tmp_path))


class _Stateful:
    name = "STATEFUL"

    def __init__(self):
        self.slip = 0.0

    def capture_state(self):
        return {"slip": self.slip}

    def restore_state(self, st):
        self.slip = st["slip"]


class _Stateless:
    name = "STATELESS"


def test_s4_constraint_state_round_trips_and_stateless_constraints_are_named(tmp_path):
    from dispsolver.solver import restart

    model, ex = _make(tmp_path, _deck_3d())
    a, b = _Stateful(), _Stateless()
    ex.solver.constraints.extend([a, b])
    a.slip = 0.37
    state = restart.capture(ex.solver, "S1", 0.5, 5, 6, 0, None)
    assert state["constraints_without_state"] == ["STATELESS"]

    model2, ex2 = _make(tmp_path, _deck_3d())
    a2, b2 = _Stateful(), _Stateless()
    ex2.solver.constraints.extend([a2, b2])
    warnings = restart.apply(ex2.solver, state)
    assert a2.slip == 0.37
    assert any("STATELESS" in w and "restarts fresh" in w for w in warnings)


def test_s4_other_version_and_other_model_are_refused(tmp_path):
    from dispsolver.solver import restart

    model, ex = _make(tmp_path, _deck_3d())
    state = restart.capture(ex.solver, "S1", 0.5, 5, 6, 0, None)
    model2, ex2 = _make(tmp_path, _deck_3d())
    with pytest.raises(ValueError, match="version"):
        restart.apply(ex2.solver, dict(state, version=restart.VERSION + 1))
    model3, ex3 = _make(tmp_path, _deck_3d().replace("2, 1, 1, 0.05", "2, 1, 1, 0.04"))     # same mesh: accepted
    restart.apply(ex3.solver, state)
    model4, ex4 = _make(tmp_path, _deck_2d(), dim=2)
    with pytest.raises(ValueError, match="different model"):
        restart.apply(ex4.solver, state)


# ---- the forms real Abaqus decks use (benchmark_inp holds hundreds of *RESTART lines) -----------------------

def _read_with(tmp_path, restart_line: str):
    path = tmp_path / "r.inp"
    path.write_text(_deck_3d(model_lines=restart_line + "\n"))
    return read_inp(str(path), dim=3)


def test_frequency_zero_means_no_restart_data(tmp_path):
    # Keywords Guide: FREQUENCY=0 writes no restart data
    assert _read_with(tmp_path, "*RESTART,WRITE,FREQ=0").restart_write is None


def test_abbreviated_parameters_are_understood(tmp_path):
    assert _read_with(tmp_path, "*RESTART,WRITE,FREQ=8").restart_write == {"frequency": 8, "overlay": False}
    assert _read_with(tmp_path, "*Restart, Write, Freque=4, Overlay").restart_write == {"frequency": 4, "overlay": True}


def test_read_and_write_on_one_keyword(tmp_path):
    m = _read_with(tmp_path, "*RESTART,TIMEMARKS=YES,READ,STEP=1,INTE=3,END STEP,WRITE,NUM=1")
    assert m.restart_read == {"step": 1, "inc": None, "end_step": True}
    assert m.restart_write == {"frequency": 1, "overlay": False}      # NUM / TIMEMARKS only pick WHEN: reported
