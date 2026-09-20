"""2-case r-gate validation: CPE6M vs Opt1 with new accept reporting."""
import sys, json
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from benchmark_element.benchmark_pure_moment_rollup import run_rollup_2d

out = {}
for label, pet, psa in [("2D-CPE6M (CPE6M+CPE4H)", "CPE6M", "CPE4H"),
                        ("2D-Opt1 (CPE4I+CPE4H)", "CPE4I", "CPE4H")]:
    print(f"[RUN] {label}", flush=True)
    res = run_rollup_2d(pet, psa, n_substeps=25, label="")
    keys = ["max_theta_deg", "completed_180", "final_M_root", "final_M_tip",
            "final_U_strain_mJ", "total_iters", "worst_r_accept_N", "accept_branches",
            "global_force_pct", "global_moment_pct", "balance_ok", "circularity_rmse_pct"]
    print(json.dumps({k: res.get(k) for k in keys}, indent=1), flush=True)
    out[label] = res

with open(repo_root / "benchmark_element" / "results" / "benchmark_pure_moment_rollup_rtest.json", "w") as f:
    json.dump(out, f, indent=2)
print("[SAVED] rtest JSON", flush=True)
