"""FoldModelConfig.set_path must parse text for fields whose current value is None.

`--set solver.max_displacement_corr=0.03` used to store the string "0.03" (the field is
Optional[float] and defaults to None, so the declared type is not visible at runtime) and
the solver later failed with `TypeError: can't multiply sequence by non-int of type 'float'`.
"""
import pytest

from dispsolver.fold_model_config import FoldModelConfig


def _cfg():
    cfg = FoldModelConfig()
    cfg.solver.max_displacement_corr = None
    return cfg


@pytest.mark.parametrize("text, expected", [
    ("0.03", 0.03),
    ("5", 5),
    ("1e-3", 1e-3),
    ("none", None),
    ("None", None),
    ("null", None),
    ("true", True),
    ("False", False),
])
def test_none_valued_field_parses_text(text, expected):
    cfg = _cfg()
    cfg.set_path("solver.max_displacement_corr", text)
    value = cfg.solver.max_displacement_corr
    assert value == expected and type(value) is type(expected)


def test_non_numeric_text_stays_text():
    cfg = _cfg()
    cfg.set_path("solver.max_displacement_corr", "abc")
    assert cfg.solver.max_displacement_corr == "abc"


def test_typed_fields_still_coerce():
    cfg = FoldModelConfig()
    cfg.set_path("solver.tie_method", "penalty")
    assert cfg.solver.tie_method == "penalty"
    cfg.set_path("solver.distortion_control", "false")
    assert cfg.solver.distortion_control is False
    cfg.set_path("solver.distortion_j_crit", "0.5")
    assert cfg.solver.distortion_j_crit == 0.5
