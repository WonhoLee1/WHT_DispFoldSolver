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
    """Fast inspection of .inp file text to detect 2D vs 3D dimension."""
    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            l_upper = line.strip().upper()
            if l_upper.startswith("*ELEMENT") and "TYPE=" in l_upper:
                match = re.search(r"TYPE\s*=\s*([A-Z0-9]+)", l_upper)
                if match:
                    et = match.group(1)
                    if et.startswith("C3D") or et in ("C3D8", "C3D8I", "C3D8H", "C3D8R", "C3D4", "C3D10", "C3D6"):
                        return 3
    return 2


def print_preflight_summary(reader: AbaqusInputReader, model: Any, dim: int, out_dir: str):
    """Print diagnostic pre-flight table of parsed Abaqus model."""
    print("=" * 72)
    print(f" Abaqus .inp Solver Runner Pre-Flight Summary [Model: {model.name}]")
    print("=" * 72)
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
    print(f" Surface Ties      : {len(model.root_assembly.ties)}")
    print(f" Contact Pairs     : {len(getattr(model.root_assembly, 'contact_pairs', []))}")
    print(f" Nodal Transforms  : {list(model.transforms.keys())}")

    if reader.unsupported_keywords:
        print(f" Warnings          : {len(reader.warnings)} warning(s)")
        print(f" Unsupported KWs   : {list(reader.unsupported_keywords)}")
    else:
        print(" Compatibility     : 100% Keywords Handled")
    print("-" * 72)


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
    executor = MultiStepExecutor(model, verbose=verbose)
    executor.initialize_solver(notices=reader.warnings)
    
    # Configure element JIT backend if solver supports it
    if hasattr(executor.solver, "elem_jit"):
        executor.solver.elem_jit = str(elem_jit).lower()

    ok_all = True
    step_results = []

    for sname in targets:
        step_obj = model.steps[sname]
        step_start_t = time.time()
        ok = executor.execute_step(step_obj, from_parent_checkpoint=True)
        ok_all = ok_all and ok
        step_wall = time.time() - step_start_t

        step_results.append({
            "step_name": sname,
            "converged": ok,
            "wall_time": step_wall,
        })

    wall_time = time.time() - start_time
    solver, sys_obj = executor.solver, executor.sys

    # 5. Field Exports (VTKHDF / VTU)
    exported_files = []
    if solver is not None and hasattr(solver, "u"):
        nodes_arr = sys_obj.coords if sys_obj is not None else solver.mesh.nodes_array()
        elem_conn = sys_obj.elem_conn_0based if sys_obj is not None else solver.mesh.elements_array()
        etype = sys_obj.elem_types[0] if (sys_obj is not None and len(sys_obj.elem_types) > 0) else "CPE4"
        stresses = getattr(solver, "elem_stresses", None)
        sdvs = getattr(solver, "elem_sdvs", None)

        if export_format in ("vtkhdf", "both", "auto"):
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
                print(f"[Export] Saved VTKHDF binary field data: {saved_file}")

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
        print("=" * 72)
        print(f" Execution Complete: Converged={ok_all} | Total Wall Time={wall_time:.2f}s")
        print(f" Summary JSON Report: {json_path}")
        print("=" * 72)

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

    args = parser.parse_args()
    try:
        run_inp_file(
            inp_path=args.inp_path,
            dim=args.dim,
            elem_jit=args.elem_jit,
            export_format=args.export_format,
            out_dir=args.out_dir,
            step_name=args.step,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\n[Error] Failed executing Abaqus .inp deck: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
