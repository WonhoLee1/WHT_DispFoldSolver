"""
run_all.py
===========
Orchestrator: runs all verification benchmarks, saves structured
results to verification/results/, and generates a markdown report.

Usage
-----
    python -m verification.run_all
    python -m verification.run_all --no-jit-warmup   # skip JAX warm-up
    python -m verification.run_all --benchmark patch_test_element

Outputs (in verification/results/)
----------------------------------
    verification_report.md   — human-readable summary
    results.json             — structured data
    results.csv              — flat table
"""

from __future__ import annotations

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict

import numpy as np

# Ensure JAX float64
import jax
jax.config.update("jax_enable_x64", True)

from .benchmarks import ALL_BENCHMARKS, run_benchmark, run_all_benchmarks


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _warmup_jax():
    """Pre-compile JAX kernels to avoid measuring JIT compilation time."""
    print("Warming up JAX JIT compilation...", flush=True)
    from .element_backends import (
        _get_jax_q4_fn, _get_jax_up_fn, _get_jax_eas_fn,
        _get_jax_visco_fs_fn, _get_jax_simo_fs_fn,
    )
    import jax.numpy as jnp
    coords = jnp.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    u_zero = jnp.zeros(8)
    _get_jax_q4_fn(1000.0, 0.3)(coords, u_zero)
    _get_jax_up_fn(1000.0, 0.3)(coords, u_zero, {'E': 1000.0, 'nu': 0.3})
    alpha_zero = jnp.zeros(4)
    state5 = jnp.broadcast_to(jnp.array([1., 0., 0., 1., 0.]), (4, 5))
    _get_jax_eas_fn(1000.0, 0.3)(coords, u_zero, alpha_zero, state5)
    state4 = jnp.zeros((4, 8))
    fn_visco = _get_jax_visco_fs_fn(1000.0, 0.3)
    fn_visco(coords, u_zero, state4, 1.0)
    fn_simo, kappa, bparams, g_i, tau_i, g_inf = _get_jax_simo_fs_fn(1000.0, 0.3)
    state6 = jnp.zeros((4, 12))
    fn_simo(coords, u_zero, state6, kappa, bparams, g_i, tau_i, g_inf, 1.0, 1.0)
    print("JAX warm-up complete.", flush=True)


def _generate_markdown_report(results: List[dict], runtime_s: float) -> str:
    """Generate a markdown verification report."""
    n_pass = sum(1 for r in results if r.get('passed', False))
    n_fail = len(results) - n_pass
    n_err  = sum(1 for r in results if 'error' in r)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"# FEM Verification Report")
    lines.append("")
    lines.append(f"**Generated**: {timestamp}")
    lines.append(f"**Runtime**: {runtime_s:.1f}s")
    lines.append(f"**Results**: {n_pass} passed, {n_fail} failed"
                 + (f" ({n_err} errors)" if n_err else ""))
    lines.append(f"**Codebase**: dispsolver (JAX-based 2D Plane Strain FEM)")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| # | Benchmark | Category | Theory | Tolerance | Status |")
    lines.append("|---|-----------|----------|--------|-----------|--------|")
    for i, r in enumerate(results):
        name = r.get('name', '?')
        cat = r.get('category', '?')
        th = r.get('theory_value', None)
        tol = r.get('tolerance_pct', None)
        status = "PASS" if r.get('passed') else ("ERROR" if 'error' in r else "FAIL")
        th_str = f"{th:.4e}" if th is not None and not np.isnan(th) else "—"
        tol_str = f"{tol:.2f}%" if tol is not None else "—"
        lines.append(f"| {i+1} | {name} | {cat} | {th_str} | {tol_str} | {status} |")
    lines.append("")

    # Backend comparison table
    lines.append("## Backend Comparison")
    lines.append("")
    all_backends = set()
    for r in results:
        for bk in r.get('backends', {}):
            all_backends.add(bk)
    all_backends = sorted(all_backends)

    header = "| Benchmark | " + " | ".join(all_backends) + " |"
    sep = "|---|" + "|".join(["---"] * len(all_backends)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        name = r.get('name', '?')
        row = [name]
        for bk in all_backends:
            bk_data = r.get('backends', {}).get(bk, {})
            if 'error' in bk_data:
                row.append("ERROR")
            elif 'value' in bk_data:
                err = bk_data.get('error_pct', float('inf'))
                val = bk_data.get('value', float('nan'))
                cell = f"{val:.4e} ({err:.3f}%)"
                row.append(cell)
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")
    for r in results:
        name = r.get('name', '?')
        passed = r.get('passed', False)
        status_icon = "PASS" if passed else "FAIL"
        lines.append(f"### {name} — {status_icon}")
        lines.append("")
        if 'error' in r:
            lines.append(f"**Error**: {r['error']}")
            lines.append("")
            continue

        cat = r.get('category', '?')
        th = r.get('theory_value', None)
        tol = r.get('tolerance_pct', None)
        unit = r.get('unit', '')
        details = r.get('details', '')

        lines.append(f"- **Category**: {cat}")
        if th is not None and not np.isnan(th):
            lines.append(f"- **Theory**: {th:.6e} {unit}")
        if tol is not None:
            lines.append(f"- **Tolerance**: {tol:.2f}%")
        if details:
            lines.append(f"- **Details**: {details}")
        lines.append("")
        lines.append("| Backend | Value | Error (%) | Iter | Passed |")
        lines.append("|---------|-------|-----------|------|--------|")
        for bk, bk_res in r.get('backends', {}).items():
            val = bk_res.get('value', float('nan'))
            err = bk_res.get('error_pct', float('nan'))
            n_iter = bk_res.get('n_iter', '—')
            bk_pass = bk_res.get('passed', False)
            bk_status = "PASS" if bk_pass else "FAIL"
            val_str = f"{val:.6e}" if not np.isnan(val) else "N/A"
            err_str = f"{err:.4f}" if not np.isnan(err) else "N/A"
            lines.append(f"| {bk} | {val_str} | {err_str} | {n_iter} | {bk_status} |")
        lines.append("")

    # Backend notes
    lines.append("## Backend Notes")
    lines.append("")
    lines.append("### Available Backends")
    lines.append("")
    lines.append("- **NumPy Q4 B-bar**: Pure NumPy 2×2 Gauss quadrature with B-bar SRI "
                 "(dispsolver.element.q4). Baseline element formulation.")
    lines.append("- **NumPy Q4 EAS-4**: Enhanced Assumed Strain with 4 internal modes, "
                 "statically condensed (dispsolver.element.q4_eas). Eliminates bending locking.")
    lines.append("- **JAX Q4 B-bar**: JAX autodiff via NeoHookean material, "
                 "vmap-vectorized (dispsolver.solver.dynamic_jax). "
                 "At small strain NeoHookean ≡ linear elastic.")
    lines.append("- **JAX Q4 EAS-4**: JAX EAS with J2 plasticity "
                 "(dispsolver.element.q4_eas_jax). J2 yield set to 1e12 (never yields) "
                 "so response is linear elastic. Production EAS path for PET layers.")
    lines.append("- **JAX Q1P0 Hybrid**: Q1P0 mixed formulation (u-p) with static condensation, "
                 "energy-based JAX autodiff (dispsolver.element.q4_up_jax). "
                 "For near-incompressible materials.")
    lines.append("- **JAX Visco Hybrid F-bar**: Finite-strain Green-Lagrange F-bar viscoelastic "
                 "element (dispsolver.element.q4_visco_hybrid_fs_jax). Prony g_i set to [0] for "
                 "pure elastic verification. This is the ex03 PSA element path.")
    lines.append("- **JAX Simo Visco F-bar**: Finite-strain Flory-split Simo viscoelastic element "
                 "with pluggable hyperelastic base (dispsolver.element.q4_visco_simo_fs_jax). "
                 "Prony g_i set to [0] for pure NeoHookean verification. The more rigorous "
                 "formulation for large-rotation viscoelastic analysis.")
    lines.append("")
    lines.append("### Numba Note")
    lines.append("")
    lines.append("The dispsolver codebase does **not** use Numba. "
                 "Only NumPy and JAX backends are available. "
                 "If a Numba backend is added in the future, register it in "
                 "`verification/element_backends.py` with the same interface.")
    lines.append("")

    # Theory reference
    lines.append("## Theory Reference")
    lines.append("")
    lines.append("All formulas are for **plane strain** (ε_zz = 0):")
    lines.append("")
    lines.append("| Quantity | Formula |")
    lines.append("|----------|---------|")
    lines.append("| Plane strain D | E/((1+ν)(1-2ν)) · [[1-ν, ν, 0], [ν, 1-ν, 0], [0, 0, (1-2ν)/2]] |")
    lines.append("| Plane strain modulus | E* = E/(1-ν²) |")
    lines.append("| Beam I (per unit depth) | I = H³/12 |")
    lines.append("| Cantilever (EB) | δ = PL³/(3·E*·I) |")
    lines.append("| 3-pt bending (EB) | δ = PL³/(48·E*·I) |")
    lines.append("| 4-pt bending (EB) | δ = Pa(3L²-4a²)/(24·E*·I) |")
    lines.append("| Uniaxial (plane strain) | σ_xx = E/(1-ν²) · ε_xx (free lateral) |")
    lines.append("| Hydrostatic (plane strain) | σ = c · ε_vol/2, c = E/((1+ν)(1-2ν)) |")
    lines.append("")
    lines.append("Timoshenko shear correction (κ_s = 5/6 for rectangular cross-section) "
                 "is included for beam benchmarks.")
    lines.append("")

    # Rules reminder
    lines.append("## Post-Change Verification Rules")
    lines.append("")
    lines.append("See `verification/RULES.md` for the full rules. Key rule:")
    lines.append("")
    lines.append("> **After any change to the solver (`dispsolver/solver/`) or element "
                 "(`dispsolver/element/`) code, run `python -m verification.run_all` "
                 "and ensure all benchmarks PASS before committing.**")
    lines.append("")

    return "\n".join(lines)


def _save_results(results: List[dict], runtime_s: float):
    """Save results to JSON, CSV, and Markdown."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # JSON
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'runtime_seconds': runtime_s,
            'n_benchmarks': len(results),
            'n_passed': sum(1 for r in results if r.get('passed', False)),
            'results': results,
        }, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'benchmark', 'category', 'backend', 'value', 'theory',
            'error_pct', 'tolerance_pct', 'n_iter', 'passed', 'unit',
        ])
        for r in results:
            name = r.get('name', '?')
            cat = r.get('category', '?')
            th = r.get('theory_value', '')
            tol = r.get('tolerance_pct', '')
            unit = r.get('unit', '')
            overall_pass = r.get('passed', False)
            for bk, bk_res in r.get('backends', {}).items():
                writer.writerow([
                    name, cat, bk,
                    bk_res.get('value', ''),
                    th, bk_res.get('error_pct', ''),
                    tol, bk_res.get('n_iter', ''),
                    bk_res.get('passed', False),
                    unit,
                ])
    print(f"Saved CSV: {csv_path}")

    # Markdown report
    md_path = os.path.join(RESULTS_DIR, "verification_report.md")
    report = _generate_markdown_report(results, runtime_s)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved report: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run FEM verification benchmarks")
    parser.add_argument('--benchmark', '-b', type=str, default=None,
                        help="Run only the named benchmark")
    parser.add_argument('--no-jit-warmup', action='store_true',
                        help="Skip JAX JIT warm-up (first benchmark will be slower)")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Suppress per-benchmark progress output")
    args = parser.parse_args()

    if not args.no_jit_warmup:
        _warmup_jax()

    t0 = time.time()

    if args.benchmark:
        print(f"\nRunning single benchmark: {args.benchmark}")
        result = run_benchmark(args.benchmark)
        results = [result]
        status = "PASS" if result.get('passed') else "FAIL"
        print(f"  => {status}")
        for bk, bk_res in result.get('backends', {}).items():
            print(f"     {bk:25s}: {bk_res.get('value', 'N/A'):>14.6e}  "
                  f"err={bk_res.get('error_pct', 'N/A'):>8.4f}%")
    else:
        results = run_all_benchmarks(verbose=not args.quiet)

    runtime = time.time() - t0

    _save_results(results, runtime)

    # Print summary
    n_pass = sum(1 for r in results if r.get('passed', False))
    n_total = len(results)
    print(f"\n{'='*70}")
    print(f"  VERIFICATION COMPLETE: {n_pass}/{n_total} passed in {runtime:.1f}s")
    print(f"{'='*70}")

    if n_pass < n_total:
        print("\nFailed benchmarks:")
        for r in results:
            if not r.get('passed', False):
                print(f"  - {r.get('name', '?')}: {r.get('details', r.get('error', ''))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
