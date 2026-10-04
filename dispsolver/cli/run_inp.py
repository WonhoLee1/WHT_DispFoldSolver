"""
run_inp.py
==========
Production CLI runner and launcher engine for Abaqus .inp decks.

Reads, parses, executes, and exports FE analysis results into VTKHDF (.vtkhdf),
VTU (.vtu), Matplotlib/PyVista figures, and summary JSON reports.

Usage:
    python -m dispsolver.cli.run_inp path/to/model.inp [options]
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

from dispsolver.io.abaqus_reader import AbaqusInputReader, _parse_source
from dispsolver.solver.step_executor import MultiStepExecutor
from dispsolver.export.vtkhdf_exporter import export_vtkhdf


def peek_dim_from_file(inp_path: str) -> int:
    """2D vs 3D from the deck's element types (abaqus_reader.detect_dim: the
    tokenized deck, so *INCLUDE files and indented keyword lines count)."""
    from dispsolver.io.abaqus_reader import detect_dim
    return detect_dim(inp_path, default=2)


def print_preflight_summary(reader: AbaqusInputReader, model: Any, dim: int, out_dir: str):
    """Print diagnostic pre-flight table of parsed Abaqus model."""
    print("=" * 100)
    print(f" Abaqus .inp Solver Runner Pre-Flight Summary [Model: {model.name}]")
    print("=" * 100)
    print(f" Spatial Dimension : {dim}D (Auto-Detected / Configured)")
    print(f" Target Output Dir : {os.path.abspath(out_dir)}")
    print(f" Parts Count       : {len(model.parts)}")
    
    total_nodes = 0
    total_elems = 0
    elem_types: Dict[str, int] = {}
    
    for p_name, part in model.parts.items():
        total_nodes += len(part.nodes)
        total_elems += len(part.elements)
        for etype, _ in part.elements.values():
            elem_types[etype] = elem_types.get(etype, 0) + 1
            
    print(f" Total Nodes       : {total_nodes}")
    print(f" Total Elements    : {total_elems}")
    print(f" Element Types     : {dict(elem_types)}")
    print(f" Materials         : {list(model.materials.keys())}")
    print(f" Steps Sequence    : {[s for s in model.steps.keys() if s != 'Initial']}")
    num_ties = len(model.root_assembly.ties) + len([c for c in model.constraints.values() if type(c).__name__ == "Tie"])
    num_cp = len(getattr(model.root_assembly, 'contact_pairs', [])) + len([i for i in model.interactions.values() if type(i).__name__ == "ContactPair"])
    print(f" Surface Ties      : {num_ties}")
    print(f" Contact Pairs     : {num_cp}")
    print(f" Nodal Transforms  : {list(model.transforms.keys())}")

    gaps = reader.physics_gaps()
    if reader.unsupported_keywords or reader.warnings:
        print(f" Warnings          : {len(reader.warnings)} warning(s)")
        print(f" Unsupported KWs   : {list(reader.unsupported_keywords)}")
    if gaps:
        # Not "100% handled": these change the model being solved.
        print(" " + "!" * 98)
        print(f" MODEL GAPS        : {len(gaps)} item(s) NOT represented in this solve "
              f"(rerun with --strict to refuse):")
        for g in gaps[:20]:
            print(f"   - {g}")
        print(" " + "!" * 98)
    elif not reader.warnings:
        print(" Compatibility     : 100% Keywords Handled")
    print("-" * 100)


def run_inp_file(
    inp_path: str,
    dim: Union[int, str] = "auto",
    elem_jit: str = "numba",
    export_format: str = "vtkhdf",
    out_dir: Optional[str] = None,
    step_name: Optional[str] = None,
    dt: Optional[float] = None,
    verbose: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """Execute an Abaqus .inp file deck end-to-end and export results.

    Parameters
    ----------
    inp_path : str
        Path to .inp input deck file
    dim : int or "auto"
        Spatial dimension (2, 3, or "auto")
    elem_jit : str
        Assembly backend ("numba" or "jax")
    export_format : str
        Field export format ("vtkhdf", "vtu", "both", "none")
    out_dir : str, optional
        Output directory path (defaults to ./output_<inp_basename>)
    step_name : str, optional
        Specific step to run
    dt : float, optional
        Time increment override
    verbose : bool
        Print live terminal progress
    strict : bool
        Strict parsing mode

    Returns
    -------
    summary : dict
        Execution summary dictionary
    """
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Abaqus .inp file not found: {inp_path}")

    inp_name = os.path.splitext(os.path.basename(inp_path))[0]
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), f"output_{inp_name}")
    os.makedirs(out_dir, exist_ok=True)

    start_time = time.time()

    # 1. Dimension detection
    if str(dim).lower() == "auto":
        actual_dim = peek_dim_from_file(inp_path)
    else:
        actual_dim = int(dim)

    # 2. Parse .inp deck with correct dim
    reader, model = _parse_source(inp_path, dim=actual_dim, strict=strict)

    if verbose:
        print_preflight_summary(reader, model, actual_dim, out_dir)
    if strict:
        reader.require_complete()

    # 3. Determine steps to run
    if step_name is None:
        targets = [s for s in model.steps.keys() if s != "Initial"]
    else:
        targets = [step_name]

    if dt is not None:
        for sname in targets:
            if sname in model.steps:
                model.steps[sname].dt_init = float(dt)

    # 4. MultiStep Execution
    from dispsolver.io.cae_logger import CAELogger
    logger = CAELogger(job_name=inp_name, out_dir=out_dir, dim=actual_dim)

    executor = MultiStepExecutor(model, verbose=verbose)
    executor.initialize_solver(notices=reader.warnings)
    
    # Configure element JIT backend if solver supports it
    if hasattr(executor.solver, "elem_jit"):
        executor.solver.elem_jit = str(elem_jit).lower()

    from dispsolver.output import Odb, OutputManager, FieldOutputRequest

    vtkhdf_path = None
    if export_format in ("vtkhdf", "both", "auto"):
        vtkhdf_path = os.path.join(out_dir, f"{inp_name}_result.vtkhdf")

    odb = Odb(name=inp_name, mesh=executor.solver.mesh)
    output_mgr: Optional[OutputManager] = None

    ok_all = True
    step_results = []
    t_accum_current = 0.0

    try:
        for s_idx, sname in enumerate(targets, start=1):
            step_obj = model.steps[sname]
            step_start_t = time.time()

            field_reqs = list(getattr(step_obj, "field_output_requests", {}).values())
            hist_reqs = list(getattr(step_obj, "history_output_requests", {}).values())
            if not field_reqs and export_format in ("vtkhdf", "both", "auto"):
                field_reqs = [FieldOutputRequest(name="DefaultField", variables=["U", "RF", "S", "E"])]

            time_period = float(getattr(step_obj, "time_period", 1.0))

            if output_mgr is None:
                output_mgr = OutputManager(
                    odb=odb,
                    step_name=sname,
                    time_period=time_period,
                    field_requests=field_reqs,
                    history_requests=hist_reqs,
                    dim=actual_dim,
                    vtkhdf_path=vtkhdf_path,
                    time_offset=t_accum_current,
                )
            else:
                output_mgr.switch_step(
                    step_name=sname,
                    time_period=time_period,
                    field_requests=field_reqs,
                    history_requests=hist_reqs,
                    time_offset=t_accum_current,
                )

            ok = executor.execute_step(step_obj, from_parent_checkpoint=True, output_mgr=output_mgr)
            ok_all = ok_all and ok
            step_wall = time.time() - step_start_t
            t_accum_current += time_period

            # One .sta row per converged increment and one cutback line per
            # failed attempt, from the solver's own per-attempt record (this
            # used to write a single hard-coded INC=1/ATT=1 row per step,
            # whatever actually happened).
            t_step_start = t_accum_current - time_period
            inc_log = getattr(executor.solver, "increment_log", None) or []
            for k, rec in enumerate(inc_log):
                logger.total_iters += rec["n_iter"]
                logger.total_sdis += rec["n_sdi"]
                if rec["converged"]:
                    logger.log_increment(
                        step_idx=s_idx, inc_idx=rec["inc"], att_num=rec["att"],
                        n_iters=rec["n_iter"], n_sdis=rec["n_sdi"],
                        t_total=t_step_start + rec["t_step"], t_step=rec["t_step"],
                        dt=rec["dt"], status_str="CONVERGED",
                    )
                else:
                    logger.log_cutback(step_idx=s_idx, inc_idx=rec["inc"], old_dt=rec["dt"],
                                       new_dt=inc_log[k + 1]["dt"] if k + 1 < len(inc_log) else 0.0,
                                       reason=f"inc {rec['inc']} att {rec['att']} not converged in {rec['n_iter']} iters")
            if not ok:
                logger.log_increment(step_idx=s_idx, inc_idx=0, att_num=0, n_iters=0,
                                     t_total=0.0, t_step=0.0, dt=0.0, status_str="FAILED")

            step_results.append({
                "step_name": sname,
                "converged": ok,
                "wall_time": step_wall,
            })
    finally:
        if output_mgr is not None:
            output_mgr.finalize()
        logger.close(status_str="THE ANALYSIS HAS COMPLETED SUCCESSFULLY" if ok_all else "ANALYSIS TERMINATED WITH ERRORS")

    wall_time = time.time() - start_time
    solver, sys_obj = executor.solver, executor.sys

    # 5. Field Exports (VTKHDF / VTU)
    exported_files = []
    if vtkhdf_path and os.path.exists(vtkhdf_path):
        exported_files.append(vtkhdf_path)
        if verbose:
            print(f"[Export] Saved transient VTKHDF binary field data: {vtkhdf_path}")
    elif solver is not None and hasattr(solver, "u") and export_format in ("vtkhdf", "both", "auto"):
        nodes_arr = sys_obj.coords if sys_obj is not None else solver.mesh.nodes_array()
        elem_conn = sys_obj.elem_conn_0based if sys_obj is not None else solver.mesh.elements_array()
        etype = sys_obj.elem_types[0] if (sys_obj is not None and len(sys_obj.elem_types) > 0) else "CPE4"
        stresses = getattr(solver, "elem_stresses", None)
        sdvs = getattr(solver, "elem_sdvs", None)
        vtkhdf_path = os.path.join(out_dir, f"{inp_name}_result.vtkhdf")
        saved_file = export_vtkhdf(
            file_path=vtkhdf_path,
            nodes=nodes_arr,
            elements=elem_conn,
            u=solver.u,
            element_type=etype,
            stresses=stresses,
            sdvs=sdvs,
            time_value=1.0,
        )
        exported_files.append(saved_file)
        if verbose:
            print(f"[Export] Saved fallback VTKHDF binary field data: {saved_file}")

        if export_format in ("vtu", "both", "auto"):
            from dispsolver.export.vtu_exporter import export_vtu
            vtu_path = os.path.join(out_dir, f"{inp_name}_result.vtu")
            saved_vtu = export_vtu(
                file_path=vtu_path,
                nodes=nodes_arr,
                elements=elem_conn,
                u=solver.u,
                element_type=etype,
                stresses=stresses,
                sdvs=sdvs,
                time_value=1.0,
            )
            exported_files.append(saved_vtu)
            if verbose:
                print(f"[Export] Saved VTK XML (.vtu) field data: {saved_vtu}")

    # 6. Save JSON Summary Report
    summary = {
        "inp_path": os.path.abspath(inp_path),
        "model_name": model.name,
        "dimension": actual_dim,
        "converged": ok_all,
        "wall_time_seconds": wall_time,
        "steps_executed": targets,
        "step_results": step_results,
        "exported_files": exported_files,
        "unsupported_keywords": list(reader.unsupported_keywords),
        "warnings": list(reader.warnings),
    }

    json_path = os.path.join(out_dir, f"{inp_name}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print("=" * 100)
        print(f" Execution Complete: Converged={ok_all} | Total Wall Time={wall_time:.2f}s")
        print(f" Summary JSON Report: {json_path}")
        print("=" * 100)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Abaqus .inp File Solver Runner & Visualizer (VTKHDF + VTU Export)"
    )
    parser.add_argument("inp_path", type=str, help="Path to Abaqus .inp input deck file")
    parser.add_argument("--dim", type=str, default="auto", choices=["auto", "2", "3"],
                        help="Spatial dimension (default: auto)")
    parser.add_argument("--elem-jit", type=str, default="numba", choices=["numba", "jax"],
                        help="Element kernel assembly backend (default: numba)")
    parser.add_argument("--export-format", type=str, default="vtkhdf", choices=["vtkhdf", "vtu", "both", "none"],
                        help="Field export format (default: vtkhdf)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory path")
    parser.add_argument("--step", type=str, default=None,
                        help="Specific step name to run")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--print-json", action="store_true",
                        help="Print JSON summary of the run to stdout")
    parser.add_argument("--strict", action="store_true",
                        help="Refuse to solve a deck with unsupported/unreadable physics keywords")

    args = parser.parse_args()
    try:
        summary = run_inp_file(
            inp_path=args.inp_path,
            dim=args.dim,
            elem_jit=args.elem_jit,
            export_format=args.export_format,
            out_dir=args.out_dir,
            step_name=args.step,
            verbose=not args.quiet,
            strict=args.strict,
        )
        # If user requested JSON output on stdout, emit the summary dict as JSON
        if getattr(args, "print_json", False):
            import json as _json
            print(_json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"\n[Error] Failed executing Abaqus .inp deck: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
