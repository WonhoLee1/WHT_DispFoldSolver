"""
run_all_nafems.py
=================
Unified Master CLI for NAFEMS Verification Suite:
  - Linear Elastic Benchmarks: LE1 to LE11 (21 Elements: 10 2D + 11 3D)
  - Nonlinear Benchmarks: NL1 to NL7 (Elastica, Snap-Through, J2 Plasticity, Necking)
  - Publication-quality Figure Generation

Usage:
  python benchmark_NAFEMS/run_all_nafems.py --mode all
  python benchmark_NAFEMS/run_all_nafems.py --mode le
  python benchmark_NAFEMS/run_all_nafems.py --mode nl
  python benchmark_NAFEMS/run_all_nafems.py --mode plot
"""

from __future__ import annotations
import os
import sys
import argparse
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark_NAFEMS.run_nafems_le import run_all_nafems_le_suite
from benchmark_NAFEMS.run_nafems_nl import run_all_nafems_nl_suite
from benchmark_NAFEMS.figures_nafems import plot_nafems_le_summary, plot_nafems_nl_summary


def main():
    parser = argparse.ArgumentParser(description="Unified Master Runner for NAFEMS LE1-LE11 & NL1-NL7 Benchmarks")
    parser.add_argument(
        "--mode",
        choices=["all", "le", "nl", "plot"],
        default="all",
        help="Execution mode: 'all' (run LE + NL + plot), 'le', 'nl', or 'plot'."
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    base_dir = os.path.dirname(__file__)
    res_dir = os.path.join(base_dir, "results")
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    res_le = os.path.join(res_dir, "results_le.json")
    res_nl = os.path.join(res_dir, "results_nl.json")
    fig_le = os.path.join(fig_dir, "nafems_le_elements_comparison.png")
    fig_nl = os.path.join(fig_dir, "nafems_nl_benchmarks_comparison.png")

    if args.mode in ("all", "le"):
        print("\n" + "="*80)
        print(" [1/2] EXECUTING NAFEMS LINEAR ELASTIC BENCHMARK SUITE (LE1 to LE11)...")
        print("="*80)
        run_all_nafems_le_suite()

    if args.mode in ("all", "nl"):
        print("\n" + "="*80)
        print(" [2/2] EXECUTING NAFEMS PROPOSED NONLINEAR BENCHMARK SUITE (NL1 to NL7)...")
        print("="*80)
        run_all_nafems_nl_suite()

    if args.mode in ("all", "plot", "le", "nl"):
        print("\n" + "="*80)
        print(" [PLOTTING] GENERATING PUBLICATION-QUALITY COMPARISON CHARTS...")
        print("="*80)
        if os.path.exists(res_le):
            plot_nafems_le_summary(res_le, fig_le)
        if os.path.exists(res_nl):
            plot_nafems_nl_summary(res_nl, fig_nl)

    elapsed = time.perf_counter() - t_start
    print("\n" + "="*80)
    print(f" [COMPLETE] NAFEMS Benchmarking Suite finished in {elapsed:.2f} seconds.")
    print(f" Figures saved to: {fig_dir}")
    print(f" Results saved to: {res_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
