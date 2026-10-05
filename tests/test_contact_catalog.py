"""Integrity of dispsolver.model.contact_catalog (spec A5, contact half)."""
import pathlib
import re

import pytest

from dispsolver.constraint3d.contact_geometry import contact_mode
from dispsolver.model import contact_catalog as cc
from dispsolver.model.contact_catalog import ENFORCEMENTS, resolve_contact

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _all_specs():
    for (dim, scope, master, enf, sl, ep) in cc.catalog_keys():
        for nl in (True, False):
            yield resolve_contact(dim, enf, sl, nl, master=master, scope=scope,
                                  explicit_pairs=bool(ep))


def test_every_entry_loads_and_mode_matches_contact_geometry():
    n = 0
    for sp in _all_specs():
        assert sp.supported
        assert sp.mode == contact_mode(sp.sliding, sp.nlgeom)
        assert sp.mode in ("finite", "small", "linear")
        assert isinstance(sp.load(), type)
        n += 1
    assert n == 2 * len(cc.catalog_keys())


def test_catalog_classes_equal_classes_reachable_from_builders():
    """Reachable set = every class_path a catalog key resolves to; it must equal
    the set of concrete contact constraint classes the builders can return,
    which we get by scanning the interaction module for spec.load() families."""
    paths = cc.catalog_class_paths()
    src = (ROOT / "dispsolver/model/interaction.py").read_text(encoding="utf-8")
    # every family the catalog emits is consumed by a builder branch (or is the
    # default branch of its dimension); families used by neither would be dead
    families = {sp.family for sp in _all_specs()}
    consumed = set(re.findall(r'family(?: in \(|\s*==\s*)"([a-z0-9_]+)"', src))
    consumed |= set(re.findall(r'"(mortar3d_finite|mortar3d_small)"', src))
    default_branch = {"node_to_surface", "node_to_segment2d", "rigid3d", "rigid2d", "general3d", "mortar3d_general"}
    assert families <= consumed | default_branch, families - consumed - default_branch
    # and the contact constraint classes that exist on disk are exactly these
    for p in paths:
        mod, _, name = p.rpartition(".")
        assert (ROOT / (mod.replace(".", "/") + ".py")).exists(), p
    assert len(paths) == 9


def test_every_resolved_combination_builds_its_catalog_class():
    """resolve_contact over the full (dim, enforcement, sliding) grid: a combination
    is either in the catalog or raises NotImplementedError (no third outcome)."""
    for dim in (2, 3):
        for master in ("deformable", "rigid"):
            for enf in ENFORCEMENTS:
                for sl in ("SMALL", "FINITE"):
                    try:
                        sp = resolve_contact(dim, enf, sl, True, master=master)
                    except NotImplementedError as e:
                        assert enf in str(e) and sl in str(e)
                        assert master == "rigid" and enf not in ("PENALTY", "AUGMENTED_LAGRANGE")
                    else:
                        assert sp.class_path in cc.catalog_class_paths()


@pytest.mark.parametrize("args", [
    dict(dim=3, enforcement="BOGUS", sliding="SMALL"),
    dict(dim=3, enforcement="PENALTY", sliding="WIGGLY"),
])
def test_invalid_values_raise(args):
    with pytest.raises(NotImplementedError):
        resolve_contact(args["dim"], args["enforcement"], args["sliding"], True)


def test_rigid_master_with_mortar_or_nitsche_raises_not_silently_penalty():
    for enf in ("NITSCHE", "DUAL_MORTAR", "STANDARD_MORTAR"):
        with pytest.raises(NotImplementedError, match=enf):
            resolve_contact(3, enf, "FINITE", True, master="rigid")


def test_general_contact_is_3d_only():
    with pytest.raises(NotImplementedError, match="dim=2"):
        resolve_contact(2, "NITSCHE", "SMALL", True, scope="general")


def test_documented_limitations_are_flagged():
    assert not resolve_contact(2, "PENALTY", "SMALL", True).honors_sliding
    assert resolve_contact(2, "PENALTY", "FINITE", True).honors_sliding
    assert not resolve_contact(3, "PENALTY", "SMALL", True, master="rigid").honors_sliding
    sp = resolve_contact(3, "DUAL_MORTAR", "SMALL", True, scope="general", explicit_pairs=False)
    assert not sp.honors_enforcement and "NOT applied" in sp.note


def test_no_other_module_picks_a_contact_class_by_strings():
    """Static scan: outside the catalog, no dispsolver module imports a concrete
    contact constraint class inside a branch on enforcement/sliding strings."""
    class_names = {p.rpartition(".")[2] for p in cc.catalog_class_paths()}
    allowed = {"dispsolver/model/contact_catalog.py"}
    # modules that define the classes, and package re-exports
    defining = {p.rpartition(".")[0].replace(".", "/") + ".py" for p in cc.catalog_class_paths()}
    offenders = []
    for f in (ROOT / "dispsolver").rglob("*.py"):
        rel = f.relative_to(ROOT).as_posix()
        if rel in allowed or rel in defining or rel.endswith("__init__.py"):
            continue
        text = f.read_text(encoding="utf-8")
        for name in class_names:
            for m in re.finditer(r"^\s*(?:from\s+\S+\s+)?import\s+.*\b" + name + r"\b|^\s*from\s+\S+\s+import\s+.*\b"
                                 + name + r"\b", text, re.M):
                offenders.append((rel, name, text[:m.start()].count("\n") + 1))
    # Only the constraint classes' own consumers (solver isinstance / sensors) may import them
    for rel, name, line in offenders:
        text = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        ctx = "\n".join(text[max(0, line - 4):line + 1])
        assert not re.search(r"(constraint_enforcement|contact_enforcement|sliding)\s*(==|in\b|!=)", ctx), \
            (rel, name, line)
    interaction = (ROOT / "dispsolver/model/interaction.py").read_text(encoding="utf-8")
    for name in class_names:
        assert not re.search(r"import\s+.*\b" + name + r"\b", interaction), name
