"""The single diagnostics channel of the .inp pipeline (design M1 / spec A1)."""
import warnings

import pytest

from dispsolver.io.abaqus_reader import UnsupportedDeckError, _parse_source
from dispsolver.io.diagnostics import BlockingDiagnostics, Diagnostics, Loc

CUBE = """*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT, TYPE=C3D8, ELSET=E
1,1,2,3,4,5,6,7,8
*MATERIAL, NAME=M
*ELASTIC
1000., 0.3
*SOLID SECTION, ELSET=E, MATERIAL=M
"""


def _read(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _parse_source(text, dim=3, strict=False)


def test_collector_levels_dedupe_and_policy():
    d = Diagnostics(echo=False)
    d.info("x", "note")
    d.warning("y", "caveat")
    d.gap("z", "not represented", Loc("deck.inp", 3))
    d.gap("z", "not represented", Loc("deck.inp", 3))           # duplicate
    assert len(d) == 3
    d.raise_if_blocking(strict=False)                           # gaps do not block a non-strict run
    with pytest.raises(BlockingDiagnostics) as e:
        d.raise_if_blocking(strict=True)
    assert e.value.diagnostics[0].loc == Loc("deck.inp", 3) and str(e.value.diagnostics[0].loc) == "deck.inp:3"
    d.error("e", "wrong")
    with pytest.raises(BlockingDiagnostics):
        d.raise_if_blocking(strict=False)                       # errors always block


def test_reader_levels():
    text = CUBE + ("*EL PRINT\nS\n"                              # output only     -> info
                   "*FOOBAR\n1\n"                                # undocumented     -> gap
                   "*RADIATION VIEW FACTOR\n"                    # Abaqus, unhandled -> gap
                   "*STEP, UNSYMM=YES\n*STATIC, DIRECT\n0.1, 1.\n"   # controls -> info
                   "*BOUNDARY, TYPE=VELOCITY\nE, 1, 1\n"         # unimplemented choice -> gap
                   "*BOUNDARY\nE, X, 1\n"                        # bad line -> error
                   "*END STEP\n")
    reader, _ = _read(text)
    by = {lv: [x.code for x in reader.diagnostics.of(lv)] for lv in ("error", "gap", "warning", "info")}
    assert "ignored-keyword" in by["info"] and "control-param-ignored" in by["info"]
    assert "unsupported-keyword" in by["gap"] and "unsupported-value" in by["gap"]
    assert "unknown-keyword" in by["gap"] and "bad-input" in by["error"]
    gaps = reader.physics_gaps()
    assert any("*FOOBAR" in g for g in gaps) and not any("EL PRINT" in g for g in gaps)
    assert all(x.loc is not None for x in reader.diagnostics.of("error", "gap"))


def test_skipped_input_is_an_error_not_a_warning():
    # used to be a warning "... skipped" that no strict run ever saw
    reader, _ = _read("*DENSITY\n7.8e-9\n" + CUBE)
    assert any(x.code == "bad-context" and x.level == "error" for x in reader.diagnostics)
    with pytest.raises(UnsupportedDeckError):
        reader.require_complete()


def test_legacy_views_and_notice_sink():
    reader, _ = _read(CUBE + "*FOOBAR\n1\n")
    assert reader.warnings == [x.message for x in reader.diagnostics]
    sink = reader.diagnostics.notice_sink()
    sink.append("solver says hello")
    assert any(x.code == "solver-notice" and x.message == "solver says hello" for x in reader.diagnostics)
