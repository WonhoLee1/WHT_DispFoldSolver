"""*OUTPUT TIME MARKS=YES (the documented default): increments end exactly on the requested times."""
import numpy as np
import pytest

from dispsolver.io import read_inp
from dispsolver.output.requests import FieldOutputRequest, OutputTimingController
from dispsolver.solver.step_executor import MultiStepExecutor


# ---- controller ----------------------------------------------------------------------------------

def _ctrl(**kw):
    return OutputTimingController(FieldOutputRequest(name="F", frequency=None, **kw), step_time_period=1.0)


def test_next_time_point_after_the_current_time():
    c = _ctrl(time_points=[0.15, 0.45, 0.9])
    assert c.next_exact_time(0.0) == pytest.approx(0.15)
    assert c.next_exact_time(0.15) == pytest.approx(0.45)        # a point already reached is not "next"
    assert c.next_exact_time(0.46) == pytest.approx(0.9)
    assert c.next_exact_time(0.9) is None


def test_number_of_intervals_and_time_interval():
    c = _ctrl(num_intervals=4)
    assert c.next_exact_time(0.1) == pytest.approx(0.25)
    assert c.next_exact_time(0.25) == pytest.approx(0.5)
    t = _ctrl(time_interval=0.3)
    assert t.next_exact_time(0.0) == pytest.approx(0.3)
    assert t.next_exact_time(0.3) == pytest.approx(0.6)
    assert t.next_exact_time(0.95) is None                       # 1.2 lies beyond the step


def test_time_marks_no_and_no_time_trigger():
    assert _ctrl(time_points=[0.2], exact_time_points=False).next_exact_time(0.0) is None
    assert _ctrl().next_exact_time(0.0) is None                  # frequency-only request


# ---- solver loops --------------------------------------------------------------------------------

class _Recorder:
    def __init__(self, marks):
        self.marks = np.asarray(marks, dtype=float)
        self.times = []

    def next_exact_time(self, t):
        later = self.marks[self.marks > t + 1e-9]
        return float(later.min()) if later.size else None

    def record_step(self, step_time, **kw):
        self.times.append(float(step_time))


_DECK_3D = (
    "*HEADING\nmarks\n*NODE\n1,0,0,0\n2,1,0,0\n3,1,1,0\n4,0,1,0\n5,0,0,1\n6,1,0,1\n7,1,1,1\n8,0,1,1\n"
    "*ELEMENT, TYPE=C3D8, ELSET=EL\n1, 1,2,3,4,5,6,7,8\n"
    "*MATERIAL, NAME=M\n*ELASTIC\n1000.0, 0.3\n*SOLID SECTION, ELSET=EL, MATERIAL=M\n"
    "*STEP, NAME=S1, NLGEOM=NO\n*STATIC\n0.1, 1.0, 1e-4, 0.1\n"
    "*BOUNDARY\n1, 1, 3, 0.0\n4, 1, 1, 0.0\n4, 3, 3, 0.0\n5, 1, 2, 0.0\n8, 1, 1, 0.0\n"
    "2, 1, 1, 0.05\n3, 1, 1, 0.05\n6, 1, 1, 0.05\n7, 1, 1, 0.05\n*END STEP\n"
)
_DECK_2D = (
    "*HEADING\nmarks\n*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n"
    "*ELEMENT, TYPE=CPE4, ELSET=EL\n1, 1,2,3,4\n"
    "*MATERIAL, NAME=M\n*ELASTIC\n1000.0, 0.3\n*SOLID SECTION, ELSET=EL, MATERIAL=M\n"
    "*STEP, NAME=S1, NLGEOM=NO\n*STATIC\n0.1, 1.0, 1e-4, 0.1\n"
    "*BOUNDARY\n1, 1, 2, 0.0\n4, 1, 1, 0.0\n2, 1, 1, 0.05\n3, 1, 1, 0.05\n*END STEP\n"
)


def _times(tmp_path, deck, dim, marks):
    path = tmp_path / f"d{dim}.inp"
    path.write_text(deck)
    model = read_inp(str(path), dim=dim)
    ex = MultiStepExecutor(model, verbose=False, dim=dim)
    ex.initialize_solver()
    rec = _Recorder(marks)
    assert ex.solver.solve(step=model.steps["S1"], sys=ex.sys, verbose=False, output_mgr=rec)
    return np.array(rec.times)


@pytest.mark.parametrize("dim,deck", [(3, _DECK_3D), (2, _DECK_2D)])
def test_increments_end_on_the_marked_times(tmp_path, dim, deck):
    marks = [0.123, 0.456]
    t = _times(tmp_path, deck, dim, marks)
    for m in marks:
        assert np.isclose(t, m, atol=1e-12).any(), (dim, m, t)
    assert np.isclose(t[-1], 1.0)
    assert np.all(np.diff(t) > 0)


@pytest.mark.parametrize("dim,deck", [(3, _DECK_3D), (2, _DECK_2D)])
def test_without_marks_the_increments_are_unchanged(tmp_path, dim, deck):
    t = _times(tmp_path, deck, dim, [])
    assert not np.isclose(t, 0.123, atol=1e-12).any()
    assert np.isclose(t[0], 0.1)                                  # the step's own initial increment


# ---- reader --------------------------------------------------------------------------------------

def _requests(deck_tail: str, dim: int = 2):
    from tests.test_p1_reader_keywords import _parse_source
    deck = ("*HEADING\nt\n*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n"
            "*ELEMENT, TYPE=CPE4, ELSET=EL\n1, 1,2,3,4\n*NSET, NSET=N1\n1\n"
            "*TIME POINTS, NAME=TP1\n0.1, 0.5\n0.9\n"
            "*STEP, NAME=S1\n*STATIC\n0.1, 1.0, 1e-4, 0.1\n" + deck_tail + "*END STEP\n")
    reader, model = _parse_source(deck, dim=dim)
    # the step also carries a default F-Output-1; the requests this deck defines are F-Output-Node/Elem-*
    reqs = [r for n, r in model.steps["S1"].field_output_requests.items() if n.startswith(("F-Output-Node", "F-Output-Elem"))]
    assert reqs, list(model.steps["S1"].field_output_requests)
    return reader, reqs


def test_time_points_table_and_output_request():
    reader, reqs = _requests("*OUTPUT, FIELD, TIME POINTS=TP1\n*NODE OUTPUT\nU\n")
    assert reader.time_point_tables["TP1"] == [0.1, 0.5, 0.9]
    r = reqs[0]
    assert r.time_points == [0.1, 0.5, 0.9] and r.exact_time_points is True
    assert r.frequency is None                                   # a time-based request is not also per-increment


def test_generate_form():
    reader, _ = _requests("*OUTPUT, FIELD, NUMBER INTERVAL=2\n*NODE OUTPUT\nU\n")
    from tests.test_p1_reader_keywords import _parse_source
    deck = ("*HEADING\nt\n*NODE\n1,0,0\n*TIME POINTS, NAME=G, GENERATE\n0.0, 1.0, 0.25\n"
            "*STEP, NAME=S1\n*STATIC\n*END STEP\n")
    rdr, _ = _parse_source(deck, dim=2)
    assert rdr.time_point_tables["G"] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_time_interval_and_time_marks_no():
    _, reqs = _requests("*OUTPUT, FIELD, TIME INTERVAL=0.2, TIME MARKS=NO\n*ELEMENT OUTPUT\nS\n")
    r = reqs[0]
    assert r.time_interval == pytest.approx(0.2) and r.exact_time_points is False
    assert r.frequency is None


def test_frequency_request_is_unchanged():
    _, reqs = _requests("*OUTPUT, FIELD, FREQUENCY=3\n*NODE OUTPUT\nU\n")
    r = reqs[0]
    assert r.frequency == 3 and r.time_interval is None and r.time_points is None
