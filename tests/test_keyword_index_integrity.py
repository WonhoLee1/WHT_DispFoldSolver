"""The reader's keyword registry agrees with the Abaqus Keywords Guide index.

Every registered keyword is an Abaqus keyword, and every declared parameter
name (and alias) is an Abaqus parameter of that keyword under Abaqus's
abbreviation rule. A declaration Abaqus does not define means the handler is
reading something no valid deck contains -- typically a data-line item
declared as a parameter (*MPC's type), or a name that differs from the manual
(*OUTPUT, NUMBER INTERVALS for NUMBER INTERVAL, which hid the real one).
"""
import dispsolver.io.keywords  # noqa: F401
from dispsolver.io.keyword_index import AmbiguousName, abaqus_index, resolve_name
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY
from dispsolver.io.schema import SCHEMA_KEYWORDS, check_against_index

# Known, owned, not yet fixed: (keyword, declared name) -> reason
ALLOWED = {
    # material keywords move to schema classes in B4 (agent B, after sync S2);
    # these are data-line items declared as parameters
    ("ELASTIC", "TEMPERATURE"): "B4: data item",
    ("ELASTIC", "TEMP"): "B4: data item",
    ("DRUCKER PRAGER", "ANGLE"): "B4: data item",
    ("DRUCKER PRAGER", "DILATION ANGLE"): "B4: data item",
    ("CONCRETE DAMAGED PLASTICITY", "ANGLE"): "B4: data item",
    ("CONCRETE DAMAGED PLASTICITY", "ECCENTRICITY"): "B4: data item",
    ("CONCRETE DAMAGED PLASTICITY", "FB0/FC0"): "B4: data item",
    ("CONCRETE DAMAGED PLASTICITY", "FB0 FC0"): "B4: data item",
    ("CONCRETE DAMAGED PLASTICITY", "K"): "B4: data item",
    ("CONCRETE DAMAGED PLASTICITY", "VISCOSITY"): "B4: data item",
}


def test_every_registered_keyword_is_an_abaqus_keyword():
    ix = abaqus_index()
    missing = sorted(n for n in GLOBAL_KEYWORD_REGISTRY.all_specs() if n not in ix)
    assert not missing, missing


def test_every_declared_parameter_is_an_abaqus_parameter():
    ix = abaqus_index()
    bad = []
    for name, spec in GLOBAL_KEYWORD_REGISTRY.all_specs().items():
        entry = ix.get(name)
        for key in spec.params:
            try:
                ok = resolve_name(key, entry.params) is not None
            except AmbiguousName:
                ok = False
            if not ok and (name, key) not in ALLOWED:
                bad.append((name, key))
    assert not bad, bad


def test_allow_list_has_no_stale_entries():
    ix = abaqus_index()
    stale = [k for k in ALLOWED
             if GLOBAL_KEYWORD_REGISTRY.get_handler(k[0]) is None
             or k[1] not in GLOBAL_KEYWORD_REGISTRY.get_handler(k[0]).params
             or resolve_name(k[1], ix.get(k[0]).params) is not None]
    assert not stale, stale


def test_schema_keywords_match_the_index():
    assert SCHEMA_KEYWORDS
    for cls in SCHEMA_KEYWORDS.values():
        assert check_against_index(cls) == [], cls.name
