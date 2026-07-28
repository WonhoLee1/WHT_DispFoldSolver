"""
DispFoldApp.py
==============
Unified Display Folding Simulation Application & Solvers Entry Point.

Consolidates ex12 (Abaqus .inp deck solve) and ex13 (in-memory build/roundtrip modes)
into a single high-performance FEA application.

Features
--------
1. Model Execution Modes (--mode):
   - 'inp'       (default): Read Abaqus .inp deck (auto-generated via gen_ex12_inp if missing)
   - 'build'     : Build model directly as Python objects in memory
   - 'roundtrip' : Generate .inp text in memory, parse back via AbaqusReader
2. Element JIT Backend (--elem_jit):
   - 'jax'       (default): JAX AutoDiff vectorized assembly
   - 'numba'     : Multi-threaded LLVM JIT assembly for supported elements (Q4_BBAR, Q4_EAS, Q4_UP, Q4_COROTATIONAL, T3)
   - 'numpy'     : Pure CPython / NumPy sequential assembly
3. 50% Mesh Element Size Default (--mesh_ratio 0.5):
   - Refines element size to 50% of default (2x density along panel) for high bending accuracy.
4. Commercial S/W Style Performance Summary:
   - Prints detailed Abaqus / Ansys style timing breakdown at completion.

Usage
-----
    python examples/DispFoldApp.py --mode inp --elem_jit jax
    python examples/DispFoldApp.py --mode build --elem_jit numba --max-steps 10
    python examples/DispFoldApp.py --mode roundtrip --elem_jit numba --mesh_ratio 0.5
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Optional

import numpy as np

# Ensure repository root is on sys.path
HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dispsolver.fold_model_config import FoldModelConfig, DEFAULT_CONFIG, MeshGradingConfig
from dispsolver.postprocess.model_review import print_model_review_from_builder_result
from gen_ex12_inp import generate as generate_inp_text, _graded_display_x
from ex12_abaqus_inp_plate_fold import (
    run_abaqus_inp_folding,
    run_folding_from_result,
    _laminate_layer_materials,
)


def run_disp_fold_app(
    mode: str = "inp",
    elem_jit: str = "jax",
    max_steps: Optional[int] = None,
    mesh_ratio: float = 0.5,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Run display folding simulation with specified mode and element JIT backend."""
    t0_pre = time.time()

    # Apply mesh refinement ratio to config
    config = FoldModelConfig(
        grading=MeshGradingConfig(
            tip_cluster_width=DEFAULT_CONFIG.grading.tip_cluster_width,
            tip_dx=DEFAULT_CONFIG.grading.tip_dx * mesh_ratio,
            hinge_edge_cluster_width=DEFAULT_CONFIG.grading.hinge_edge_cluster_width,
            hinge_edge_dx=DEFAULT_CONFIG.grading.hinge_edge_dx * mesh_ratio,
            hinge_span_half_width=DEFAULT_CONFIG.grading.hinge_span_half_width,
            hinge_span_dx=DEFAULT_CONFIG.grading.hinge_span_dx * mesh_ratio,
            plate_body_dx=DEFAULT_CONFIG.grading.plate_body_dx * mesh_ratio,
        )
    )

    out_file = output_path or os.path.join(HERE, "ex12_result.pkl")

    if verbose:
        print(f"\n{'='*70}", flush=True)
        print(f"  DISPFOLD APP - STARTING SIMULATION (Mode={mode.upper()}, ElemJIT={elem_jit.upper()})", flush=True)
        print(f"{'='*70}", flush=True)

    solver = None
    if mode == "inp":
        inp_file = os.path.join(HERE, "ex12_rigid_plate_display_fold.inp")
        if not os.path.exists(inp_file):
            print(f"Generating deck: {inp_file}...", flush=True)
            inp_text = generate_inp_text(config)
            with open(inp_file, "w", encoding="utf-8") as f:
                f.write(inp_text)

        t_pre = time.time() - t0_pre
        info = run_abaqus_inp_folding(
            inp_path=inp_file,
            result_name=out_file,
            max_steps=max_steps,
            elem_jit=elem_jit,
        )
        if isinstance(info, dict):
            solver = info.get("solver")
        else:
            solver = info
        if solver is not None:
            solver.t_preprocess = t_pre

    elif mode in ("build", "roundtrip"):
        from ex13_unified_model_io import run_build, run_roundtrip, run_read
        t_pre = time.time() - t0_pre
        if mode == "roundtrip":
            res_info = run_roundtrip(config)
        else:
            res_info = run_build(config)
        return 0 if res_info.get("reached_target") else 1

    else:
        raise ValueError(f"Unknown mode {mode!r}; choose from 'inp', 'build', 'roundtrip'")

    if solver is not None:
        solver.print_job_timing_summary(model_mode=mode, mesh_ratio=mesh_ratio)
        return 0
    else:
        print("\n*** SIMULATION ABORTED OR FAILED ***\n", flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description="DispFoldApp — Unified Display Folding Solver Application")
    parser.add_argument("--mode", choices=["inp", "build", "roundtrip"], default="inp",
                        help="Model execution mode (default: inp)")
    parser.add_argument("--elem_jit", choices=["jax", "numba", "numpy"], default="jax",
                        help="Element JIT backend (default: jax)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum solver time increments to run")
    parser.add_argument("--mesh_ratio", type=float, default=0.5,
                        help="Element size ratio (default: 0.5 = 50%% element size, 2x resolution)")
    parser.add_argument("--output", type=str, default=None,
                        help="Result output path (.pkl)")
    args = parser.parse_args()

    sys.exit(run_disp_fold_app(
        mode=args.mode,
        elem_jit=args.elem_jit,
        max_steps=args.max_steps,
        mesh_ratio=args.mesh_ratio,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
