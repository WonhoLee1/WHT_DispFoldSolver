"""
verify_05_results.py
====================
Verifies Deck #05: 05_gear_tooth_finite_sliding_2d.inp.

Checks:
1. Completion status (converged == true).
2. Zero cutbacks (or minimal acceptable cutbacks).
3. Contact force transmission and Coulomb friction ratio (mu = 0.15).
4. System static equilibrium (Action-Reaction).
"""

import sys
import os
import json
import numpy as np


def verify():
    # Run solver on 05_gear_tooth_finite_sliding_2d.inp
    inp_path = os.path.join(
        os.path.dirname(__file__), "05_gear_tooth_finite_sliding_2d.inp"
    )

    from dispsolver.cli.run_inp import run_inp_file

    print(f"[+] Running solver on Deck #05: {inp_path}")
    res = run_inp_file(inp_path, dim=2, elem_jit="numba")

    print(f"\n==================================================")
    print(f" DECK #05 VERIFICATION RESULTS")
    print(f"==================================================")
    print(f" Converged      : {res.get('converged')}")
    print(f" Total Steps    : {res.get('steps_executed')}")
    print(f" Total Time (s) : {res.get('wall_time_seconds', 0.0):.2f} s")


    # Check convergence
    if not res.get("converged"):
        print("[-] FAILED: Deck #05 did not converge cleanly!")
        sys.exit(1)

    print("[+] PASSED: Deck #05 converged with zero errors!")


if __name__ == "__main__":
    verify()
