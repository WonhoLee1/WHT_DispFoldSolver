"""
dispsolver/loads/catalog.py
===========================
Declarative catalog of load types supported across DLOAD, DSLOAD, and CLOAD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple


LoadKind = Literal["face_pressure", "body_force", "gravity", "nodal", "unsupported"]


@dataclass(frozen=True)
class LoadTypeSpec:
    """Specification of an Abaqus load type under a given load keyword."""
    keyword: str                 # "DLOAD" | "DSLOAD" | "CLOAD"
    load_type: str               # "P1".."P6", "P", "BX", "BY", "BZ", "GRAV", ...
    kind: LoadKind
    face: Optional[int] = None   # 1-based face index if face pressure
    component: Optional[int] = None   # 0-based DOF component index if directional load
    reason: Optional[str] = None      # explanation when kind == "unsupported"


_SPECS: list[LoadTypeSpec] = [
    # ------------------------------------------------------------------ DLOAD
    # Body forces
    LoadTypeSpec("DLOAD", "BX", "body_force", component=0),
    LoadTypeSpec("DLOAD", "BY", "body_force", component=1),
    LoadTypeSpec("DLOAD", "BZ", "body_force", component=2),
    LoadTypeSpec("DLOAD", "GRAV", "gravity"),
    LoadTypeSpec("DLOAD", "CENTRIF", "unsupported", reason="centrifugal body force is not implemented"),
    LoadTypeSpec("DLOAD", "ROTGRAV", "unsupported", reason="rotary gravity is not implemented"),
    # Pressures by element face
    LoadTypeSpec("DLOAD", "P", "face_pressure"),
    LoadTypeSpec("DLOAD", "P1", "face_pressure", face=1),
    LoadTypeSpec("DLOAD", "P2", "face_pressure", face=2),
    LoadTypeSpec("DLOAD", "P3", "face_pressure", face=3),
    LoadTypeSpec("DLOAD", "P4", "face_pressure", face=4),
    LoadTypeSpec("DLOAD", "P5", "face_pressure", face=5),
    LoadTypeSpec("DLOAD", "P6", "face_pressure", face=6),
    # Aliases with S* on DLOAD
    LoadTypeSpec("DLOAD", "S1", "face_pressure", face=1),
    LoadTypeSpec("DLOAD", "S2", "face_pressure", face=2),
    LoadTypeSpec("DLOAD", "S3", "face_pressure", face=3),
    LoadTypeSpec("DLOAD", "S4", "face_pressure", face=4),
    LoadTypeSpec("DLOAD", "S5", "face_pressure", face=5),
    LoadTypeSpec("DLOAD", "S6", "face_pressure", face=6),
    # Unsupported DLOAD options
    *[LoadTypeSpec("DLOAD", f"HP{i}", "unsupported", face=i, reason="hydrostatic face pressure is not implemented")
      for i in range(1, 7)],
    *[LoadTypeSpec("DLOAD", f"TRVEC{i}", "unsupported", face=i, reason="vector traction is not implemented")
      for i in range(1, 7)],

    # ------------------------------------------------------------------ DSLOAD
    LoadTypeSpec("DSLOAD", "P", "face_pressure"),
    LoadTypeSpec("DSLOAD", "TRNORM", "face_pressure"),
    LoadTypeSpec("DSLOAD", "P1", "face_pressure", face=1),
    LoadTypeSpec("DSLOAD", "P2", "face_pressure", face=2),
    LoadTypeSpec("DSLOAD", "P3", "face_pressure", face=3),
    LoadTypeSpec("DSLOAD", "P4", "face_pressure", face=4),
    LoadTypeSpec("DSLOAD", "P5", "face_pressure", face=5),
    LoadTypeSpec("DSLOAD", "P6", "face_pressure", face=6),
    LoadTypeSpec("DSLOAD", "S1", "face_pressure", face=1),
    LoadTypeSpec("DSLOAD", "S2", "face_pressure", face=2),
    LoadTypeSpec("DSLOAD", "S3", "face_pressure", face=3),
    LoadTypeSpec("DSLOAD", "S4", "face_pressure", face=4),
    LoadTypeSpec("DSLOAD", "S5", "face_pressure", face=5),
    LoadTypeSpec("DSLOAD", "S6", "face_pressure", face=6),
    LoadTypeSpec("DSLOAD", "HP", "unsupported", reason="hydrostatic surface pressure is not implemented"),
    LoadTypeSpec("DSLOAD", "TRVEC", "unsupported", reason="surface vector traction is not implemented"),
    LoadTypeSpec("DSLOAD", "EDNOR", "unsupported", reason="edge normal traction is not implemented"),
    LoadTypeSpec("DSLOAD", "EDTRA", "unsupported", reason="edge transverse traction is not implemented"),
    LoadTypeSpec("DSLOAD", "EDSHE", "unsupported", reason="edge shear traction is not implemented"),

    # ------------------------------------------------------------------ CLOAD
    LoadTypeSpec("CLOAD", "1", "nodal", component=0),
    LoadTypeSpec("CLOAD", "2", "nodal", component=1),
    LoadTypeSpec("CLOAD", "3", "nodal", component=2),
    LoadTypeSpec("CLOAD", "4", "nodal", component=3),
    LoadTypeSpec("CLOAD", "5", "nodal", component=4),
    LoadTypeSpec("CLOAD", "6", "nodal", component=5),
    LoadTypeSpec("CLOAD", "CF1", "nodal", component=0),
    LoadTypeSpec("CLOAD", "CF2", "nodal", component=1),
    LoadTypeSpec("CLOAD", "CF3", "nodal", component=2),
    LoadTypeSpec("CLOAD", "CM1", "nodal", component=3),
    LoadTypeSpec("CLOAD", "CM2", "nodal", component=4),
    LoadTypeSpec("CLOAD", "CM3", "nodal", component=5),
]

LOAD_TYPES: Dict[Tuple[str, str], LoadTypeSpec] = {
    (s.keyword.upper(), s.load_type.upper()): s for s in _SPECS
}


def load_type(keyword: str, ltype: str) -> LoadTypeSpec:
    """Look up a LoadTypeSpec. Unknown combinations return kind='unsupported'."""
    kw = keyword.strip().upper()
    lt = ltype.strip().upper()
    spec = LOAD_TYPES.get((kw, lt))
    if spec is not None:
        return spec
    return LoadTypeSpec(
        keyword=kw,
        load_type=lt,
        kind="unsupported",
        reason=f"load type '{lt}' under *{kw} is not recognized or not implemented"
    )
