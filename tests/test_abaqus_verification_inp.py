"""SIMAINPRefResources verification-deck reading checks.

Decks in benchmark_inp/verification/ were copied from the local
SIMAINPRefResources archive named by the SIMACAEVERRefMap pages.
Reader coverage only; material constitutive implementation belongs
to the material team (unsupported material keywords below are their
honest backlog, not reader defects).
"""
import os

import pytest

from dispsolver.io.abaqus_reader import _parse_source, _AbortParse

VERIF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "benchmark_inp", "verification")
OFFICIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "benchmark_inp", "official")


def _read(path, dim=3):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _parse_source(path, dim=dim)


def test_jcrateplasticuni_reads():
    reader, model = _read(os.path.join(VERIF, "jcrateplasticuni.inp"))
    assert sum(len(p.nodes) for p in model.parts.values()) == 8
    assert "MAT" in model.materials
    assert "RATE DEPENDENT" in reader.unsupported_keywords


def test_concrete_damage_reads_structure():
    reader, model = _read(os.path.join(VERIF, "cdpse_cpe4r.inp"))
    assert sum(len(p.nodes) for p in model.parts.values()) == 4
    for kw in ["CONCRETE DAMAGED PLASTICITY",
               "CONCRETE COMPRESSION HARDENING",
               "CONCRETE TENSION STIFFENING"]:
        assert kw in reader.unsupported_keywords


def test_honest_aborts():
    cases = [
        (os.path.join(VERIF, "annealtemp.inp"), "ORIENTATION"),
        (os.path.join(OFFICIAL, "anisoplate_s9r5_orient_sect.inp"), "TRANSFORM"),
        (os.path.join(OFFICIAL, "compositeshells_c3d20r.inp"), "ORIENTATION"),
        (os.path.join(OFFICIAL, "thickcompcyl_8el.inp"), "ORIENTATION"),
        (os.path.join(OFFICIAL, "elasticsheet_cps8r_biderman.inp"), "EQUATION"),
        (os.path.join(OFFICIAL, "hertzcontact_2d_rsurf.inp"), "EQUATION"),
        (os.path.join(OFFICIAL, "rubberdisk_cax4ih_poly.inp"), "EQUATION"),
    ]
    for path, needle in cases:
        with pytest.raises(_AbortParse) as exc:
            _read(path)
        assert needle in str(exc.value), path
