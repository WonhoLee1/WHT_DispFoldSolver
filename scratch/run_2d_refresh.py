"""Refresh remaining 2D rows (CR/Opt2/Base) and merge all 5 into main JSON."""
import sys, json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from benchmark_element.benchmark_pure_moment_rollup import run_rollup_2d

cases = [("2D-CR   (CPE4_CR+CPE4_CR)", "CPE4_CR", "CPE4_CR"),
         ("2D-Opt2 (CPE4R+CPE4H)", "CPE4R", "CPE4H"),
         ("2D-Base (CPE4+CPE4)", "CPE4", "CPE4")]
fresh = {}
for label, pet, psa in cases:
    print(f"[RUN] {label}", flush=True)
    res = run_rollup_2d(pet, psa, n_substeps=25, label="")
    fresh[label] = res
    print(f"  M_root={res['final_M_root']} M_tip={res['final_M_tip']} U={res['final_U_strain_mJ']} "
          f"worst_r={res['worst_r_accept_N']:.3f} branches={res['accept_branches']} "
          f"G={res['global_force_pct']:.2f}/{res['global_moment_pct']:.2f} bal={res['balance_ok']}", flush=True)

main_p = repo_root / "benchmark_element" / "results" / "benchmark_pure_moment_rollup.json"
rtest_p = repo_root / "benchmark_element" / "results" / "benchmark_pure_moment_rollup_rtest.json"
main = json.load(open(main_p))
rtest = json.load(open(rtest_p))
main["2D"] = {**fresh, **rtest}
main.setdefault("meta", {})["note_2d_accept"] = (
    "2D rows carry worst_r_accept_N + accept_branches (DynamicSolver2D._record_acceptance, "
    "read-only port of 3D §12). 2D-CPE6M: worst_r 93.4N + Global M 90.4% FAIL -> accuracy "
    "NON-CONVERGED despite completed_180 (dev_log §16).")
json.dump(main, open(main_p, "w"), indent=2)
print("[SAVED] main JSON 2D refreshed", flush=True)
