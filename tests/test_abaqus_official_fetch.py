"""Official Abaqus decks (fetched via abaqus fetch) reading checks.

Source: benchmark_inp/official/*.inp, fetched from the release archives
per SIMACAEEXCRefMap fetchproc. Cook/cantilever job names come from the
SIMACAEBMKRefMap benchmark pages.
"""
import glob
import os
import warnings

import pytest

from dispsolver.io.abaqus_reader import _AbortParse, _parse_source

OFFICIAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "benchmark_inp", "official")

OK_DECKS = {
    "beamgap.inp": (6, 2),
    "compositeshells_c3d20r.inp": (4, 2),
    "concreteslump_beta30.inp": (4, 1),
    "cook_2d.inp": (17328, 12240),
    "cook_3d.inp": (101693, 85242),
    "elasticsheet_cps8r_biderman.inp": (6, 1),
    "eref3ksg.inp": (15, 2),
    "hertzcontact_2d_rsurf.inp": (25, 8),
    "neckingtensilebar_cax4r.inp": (10, 1),
    "nlgeocantilever_c3d8_coarse.inp": (9, 2),
    "nlgeocantilever_c3d8i_coarse.inp": (9, 2),
    "nlgeocantilever_c3d8r_coarse_eh.inp": (9, 2),
    "pipecrushing_cpe4i.inp": (5, 1),
    "radstr_holcyl_cax4r.inp": (441, 400),
    "roofsnapthrough_s4r.inp": (4, 1),
    "rubberdisk_cax4ih_poly.inp": (6, 1),
    "thickcompcyl_8el.inp": (4, 2),
    "unifcollapspipe_str_elbelem.inp": (2, 1),
}

ABORT_DECKS = {
    "anisoplate_s9r5_orient_sect.inp",
}


def _read(name, dim=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _parse_source(os.path.join(OFFICIAL_DIR, name), dim=dim)


@pytest.mark.parametrize("name,counts", sorted(OK_DECKS.items()))
def test_official_deck_reads(name, counts):
    min_nodes, min_elems = counts
    reader, model = _read(name)
    n_nodes = sum(len(p.nodes) for p in model.parts.values())
    n_elems = sum(len(p.elements) for p in model.parts.values())
    assert n_nodes == min_nodes, name
    assert n_elems == min_elems, name
    assert len(model.steps) >= 1, name


@pytest.mark.parametrize("name", sorted(ABORT_DECKS))
def test_official_deck_honest_abort(name):
    with pytest.raises(_AbortParse):
        _read(name)


def test_official_inventory_complete():
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(OFFICIAL_DIR, "*.inp"))}
    assert on_disk == set(OK_DECKS) | ABORT_DECKS
