"""
material_factory.py
====================
Single dispatch point from a `fold_model_config.MaterialDef` to either a
live dispsolver material object (`build_material_instance`, used by
ex13_unified_model_io.py::run_build()) or Abaqus .inp *MATERIAL text
(`emit_abaqus_material_block`, used by gen_ex12_inp.py). Both dispatch
on `mdef.type` in exactly one place each -- previously this branch was
duplicated independently in both files as magic-string family checks
(`if family == "PET"` / `elif family == "PSA"`), which is exactly the
kind of drift fold_model_config.py's own docstring already warns about
for geometry/material numbers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from dispsolver.fold_model_config import MaterialDef
from dispsolver.material.type_tags import J2_PLASTIC, ARRUDA_BOYCE_VISCO, NEOHOOKEAN
from dispsolver.material.plastic import J2Plasticity
from dispsolver.material.arruda_boyce import ArrudaBoyce
from dispsolver.material.neohookean import NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial


def build_material_instance(mdef: MaterialDef) -> Tuple[object, dict]:
    """Return (live material object, material_params dict) for one
    MaterialDef. Materials are stateless (constructor args only, no
    per-element mutable state lives on the object -- solve-time
    internal variables live in solver.state instead), so callers
    should build this ONCE per distinct name and reuse the same
    (obj, params) pair for every pid that references it -- do not
    rebuild per pid.
    """
    p = mdef.params
    if mdef.type == J2_PLASTIC:
        obj = J2Plasticity(E=p["E"], nu=p["nu"], sigma_y0=p["sigma_y0"], H=p["H"])
        params = dict(p)
        return obj, params

    if mdef.type == ARRUDA_BOYCE_VISCO:
        base = ArrudaBoyce()
        wlf = {"T_ref": p["wlf_T_ref"], "C1": p["wlf_C1"], "C2": p["wlf_C2"], "definition": "WLF"}
        prony = list(zip(p["prony_g"], [0.0] * len(p["prony_g"]), p["prony_tau"]))
        obj = ViscoelasticMaterial(base, p["prony_g"], p["prony_tau"], wlf_params=wlf)
        params = {"mu": p["mu"], "lambda_m": p["lambda_m"], "K": p["K"],
                  "base": base, "prony": prony, "wlf": wlf}
        return obj, params

    if mdef.type == NEOHOOKEAN:
        obj = NeoHookean()
        params = {"E": p["E"], "nu": p["nu"]}
        return obj, params

    raise ValueError(f"Unknown material type {mdef.type!r} for {mdef.name!r}")


def emit_abaqus_material_block(mdef: MaterialDef) -> List[str]:
    """Return the .inp text lines for one *MATERIAL block (no trailing
    newline join -- caller appends each line to its own `lines` list).
    """
    p = mdef.params
    lines = [f"*MATERIAL, NAME={mdef.name}"]

    if mdef.type == J2_PLASTIC:
        lines += [
            "*ELASTIC",
            f"{p['E']}, {p['nu']}",
            "*PLASTIC",
            f"{p['sigma_y0']}, 0.0",
            f"{p['sigma_y0'] + p['H']}, 1.0",
        ]
    elif mdef.type == ARRUDA_BOYCE_VISCO:
        d_param = 2.0 / p["K"]
        lines += ["*HYPERELASTIC, ARRUDA-BOYCE", f"{p['mu']}, {p['lambda_m']}, {d_param}",
                  "*VISCOELASTIC, TIME=PRONY"]
        for g_i, tau_i in zip(p["prony_g"], p["prony_tau"]):
            lines.append(f"{g_i}, 0.0, {tau_i}")
        lines += ["*TRS, DEFINITION=WLF", f"{p['wlf_T_ref']}, {p['wlf_C1']}, {p['wlf_C2']}"]
    elif mdef.type == NEOHOOKEAN:
        lines += ["*ELASTIC", f"{p['E']}, {p['nu']}"]
    else:
        raise ValueError(f"Unknown material type {mdef.type!r} for {mdef.name!r}")

    lines.append("**")
    return lines
