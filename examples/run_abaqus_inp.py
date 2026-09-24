"""
run_abaqus_inp.py
=================
Example launcher script to run an Abaqus .inp deck file through WHT_DispFoldSolver,
print diagnostic pre-flight info, and export VTKHDF (.vtkhdf) binary field data.

Usage:
    python examples/run_abaqus_inp.py [path/to/deck.inp]
"""

import sys
import os

from dispsolver.cli.run_inp import run_inp_file


def main():
    if len(sys.argv) > 1:
        inp_path = sys.argv[1]
    else:
        # Default fallback to sample verification deck
        inp_path = os.path.join("tests", "fixtures", "verif_manual", "viscocylinder_cpe4i.inp")
        print(f"No .inp path provided. Using benchmark sample: {inp_path}")

    if not os.path.exists(inp_path):
        print(f"Error: Target .inp file '{inp_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"\n🚀 Running Abaqus .inp Solver Runner on: {inp_path}\n")

    summary = run_inp_file(
        inp_path=inp_path,
        dim="auto",
        elem_jit="numba",
        export_format="vtkhdf",
        verbose=True,
    )

    print("\nSummary Results:")
    print(f"  Converged      : {summary['converged']}")
    print(f"  Wall Time      : {summary['wall_time_seconds']:.2f} s")
    print(f"  Exported Files : {summary['exported_files']}")


if __name__ == "__main__":
    main()
