"""
ex13_unified_model_io.py
=========================
Unifies the folding-solve entry points around a single downstream solve
loop (reused from `ex12_abaqus_inp_plate_fold.run_abaqus_inp_folding`),
selectable via `--mode`:

    read       : read the standard, on-disk .inp deck
                 (examples/ex12_rigid_plate_display_fold.inp, as produced
                 by gen_ex12_inp.py).
    roundtrip  : call gen_ex12_inp.generate() to build the .inp text
                 in-memory, write it to a temp file, then run the exact
                 same read+solve path -- proves the generator's output is
                 self-consistent without needing a separate .inp writer.

A third mode ("build": construct the model directly as Python objects,
no .inp text at all, mirroring ex12_rigid_plate_display_fold_corotational.py)
is intentionally NOT implemented here yet -- it would require reproducing
the current 14-layer PET-PSA/Q4_VISCO_SIMO layup a second time as raw
Python objects, which is deferred (see AGENTS.md).
"""

import argparse
import os
import tempfile

from gen_ex12_inp import generate as generate_inp_text
from ex12_abaqus_inp_plate_fold import run_abaqus_inp_folding


def run_read():
    """Read the standard on-disk .inp deck and solve."""
    inp_path = os.path.join(os.path.dirname(__file__), "ex12_rigid_plate_display_fold.inp")
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"{inp_path} not found. Run gen_ex12_inp.py first.")
    return run_abaqus_inp_folding(
        inp_path=inp_path,
        before_png_name="ex13_read_before_folding_shape.png",
        after_png_name="ex13_read_final_folding_shape.png",
    )


def run_roundtrip():
    """Build the .inp text in-memory via gen_ex12_inp.generate(), write it
    to a temp file, then read+solve it through the same path as `read`.
    """
    inp_text = generate_inp_text()
    fd, tmp_path = tempfile.mkstemp(suffix=".inp", prefix="ex13_roundtrip_")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(inp_text)
        return run_abaqus_inp_folding(
            inp_path=tmp_path,
            before_png_name="ex13_roundtrip_before_folding_shape.png",
            after_png_name="ex13_roundtrip_final_folding_shape.png",
        )
    finally:
        os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["read", "roundtrip"], default="read",
        help="Model source: 'read' (existing .inp) or 'roundtrip' "
             "(generate() in-memory -> temp .inp -> read+solve).",
    )
    args = parser.parse_args()

    if args.mode == "read":
        info = run_read()
    else:
        info = run_roundtrip()

    print("=" * 100)
    print(f" EX13 [{args.mode}] SUMMARY: nodes={info['n_nodes']}  elements={info['n_elements']}  "
          f"rbe2_constraints={info['n_rbe2_constraints']}  penalty_constraints={info['n_penalty_constraints']}  "
          f"reached_target={info['reached_target']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
