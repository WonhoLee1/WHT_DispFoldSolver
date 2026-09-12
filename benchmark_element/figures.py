"""
figures.py
==========
PNG figure generation for benchmark_3d_mechanics.py's markdown report.
Kept in its own module so plotting concerns (matplotlib backend, colors,
layout) don't clutter the benchmark/report-writing logic.

All three functions take the same patch_res / bending_res / vol_res dicts
run_all_mechanics_benchmarks() already builds, plus (for the patch figure)
the official-Abaqus-patch result dict, and write one PNG each to
benchmark_element/figures/.
"""

from typing import Any, Dict, Optional
import os

import matplotlib
matplotlib.use("Agg")  # headless -- this runs from a script/CI, not a GUI session
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

_PATCH_THRESHOLD = 1e-6
_BENDING_THRESHOLD = 0.85
_VOL_THRESHOLD = 0.60


def _ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_patch_test_errors(
    elements: list,
    patch_res: Dict[str, Dict[str, Any]],
    abq_patch_res: Dict[str, Dict[str, Any]],
    out_path: Optional[str] = None,
) -> str:
    """Grouped log-scale bar chart: in-house Irons patch modes (tension /
    compression / shear_xy / shear_yz) plus the official Abaqus Verification
    Manual patch test, per element. Lower is better; 1e-6 is the pass bar
    every column here uses.
    """
    _ensure_dir()
    out_path = out_path or os.path.join(FIGURES_DIR, "patch_test_errors.png")

    modes = ["tension", "compression", "shear_xy", "shear_yz"]
    mode_labels = ["Tension", "Compression", "Shear XY", "Shear YZ"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    abq_color = "#DD8452"

    n_elem = len(elements)
    n_series = len(modes) + 1
    x = np.arange(n_elem)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(max(10, n_elem * 1.1), 6))

    for i, (mode, label, color) in enumerate(zip(modes, mode_labels, colors)):
        vals = []
        for e in elements:
            err = patch_res.get(e, {}).get(mode, {}).get("max_error", np.nan)
            vals.append(max(err, 1e-20) if np.isfinite(err) else np.nan)
        offset = (i - n_series / 2.0 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=label, color=color)

    abq_vals = []
    for e in elements:
        err = abq_patch_res.get(e, {}).get("max_error", np.nan)
        abq_vals.append(max(err, 1e-20) if np.isfinite(err) else np.nan)
    offset = (len(modes) - n_series / 2.0 + 0.5) * width
    ax.bar(x + offset, abq_vals, width=width, label="Abaqus VM patch (official)", color=abq_color, hatch="//")

    ax.axhline(_PATCH_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"pass threshold ({_PATCH_THRESHOLD:.0e})")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(elements, rotation=30, ha="right")
    ax.set_ylabel("max interior-node displacement error (log scale, mm)")
    ax.set_title(
        "3D Solid Element Patch Test Errors -- lower is better\n"
        "In-house Irons patch (tension/compression/shear) vs. official Abaqus\n"
        "Verification Manual patch test (reference_abaqus_docs/3dpatch.txt)"
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _verdict_color(ratio: float, threshold: float) -> str:
    if not np.isfinite(ratio):
        return "#999999"
    if ratio > threshold:
        return "#55A868"  # green -- passes this benchmark's own bar
    if ratio > threshold * 0.5:
        return "#DD8452"  # orange -- moderate
    return "#C44E52"  # red -- locked


def plot_bending_locking_ratio(
    elements: list,
    bending_res: Dict[str, Dict[str, Any]],
    out_path: Optional[str] = None,
) -> str:
    """Bar chart of computed/exact Timoshenko tip-deflection ratio at
    L/h=10. ratio=1.0 is exact agreement with the closed-form beam
    solution; 0.85 is THIS BENCHMARK SUITE's own LOCKING-FREE threshold
    (run_cantilever_bending_test's verdict logic), not a number Abaqus
    itself publishes -- labeled as such so it isn't mistaken for an
    external reference.
    """
    _ensure_dir()
    out_path = out_path or os.path.join(FIGURES_DIR, "bending_locking_ratio.png")

    ratios = [bending_res.get(e, {}).get("ratio", np.nan) for e in elements]
    colors = [_verdict_color(r, _BENDING_THRESHOLD) for r in ratios]

    fig, ax = plt.subplots(figsize=(max(8, len(elements) * 0.9), 5))
    x = np.arange(len(elements))
    bars = ax.bar(x, [r if np.isfinite(r) else 0.0 for r in ratios], color=colors)
    for xi, r in zip(x, ratios):
        if np.isfinite(r):
            ax.text(xi, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(_BENDING_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"this suite's LOCKING-FREE bar ({_BENDING_THRESHOLD})")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="exact Timoshenko (ratio = 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(elements, rotation=30, ha="right")
    ax.set_ylabel("computed / exact Timoshenko tip deflection")
    ax.set_ylim(0, max(1.15, np.nanmax([r for r in ratios if np.isfinite(r)] or [1.0]) + 0.1))
    ax.set_title(
        "Cantilever Bending Shear-Locking Ratio at L/h=10\n"
        "1.0 = exact Timoshenko beam solution; bar/color threshold is this\n"
        "suite's own convention, not an Abaqus-published number"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_volumetric_locking_ratio(
    elements: list,
    vol_res: Dict[str, Dict[str, Any]],
    out_path: Optional[str] = None,
) -> str:
    """Bar chart of computed/reference tip-deflection ratio at nu=0.49999.
    0.60 is this benchmark suite's own INCOMPRESSIBLE-OK threshold
    (run_volumetric_locking_test's verdict logic), not an Abaqus-published
    number.
    """
    _ensure_dir()
    out_path = out_path or os.path.join(FIGURES_DIR, "volumetric_locking_ratio.png")

    ratios = [vol_res.get(e, {}).get("ratio_to_ref", np.nan) for e in elements]
    colors = [_verdict_color(r, _VOL_THRESHOLD) for r in ratios]

    fig, ax = plt.subplots(figsize=(max(8, len(elements) * 0.9), 5))
    x = np.arange(len(elements))
    ax.bar(x, [r if np.isfinite(r) else 0.0 for r in ratios], color=colors)
    for xi, r in zip(x, ratios):
        if np.isfinite(r):
            ax.text(xi, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(_VOL_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"this suite's INCOMPRESSIBLE-OK bar ({_VOL_THRESHOLD})")
    ax.set_xticks(x)
    ax.set_xticklabels(elements, rotation=30, ha="right")
    ax.set_ylabel("computed / nu=0.3 reference tip deflection")
    ax.set_title(
        "Volumetric (Incompressibility) Locking Ratio at nu=0.49999\n"
        "Higher = less volumetric locking; bar is this suite's own\n"
        "convention, not an Abaqus-published number"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def generate_all_figures(
    elements: list,
    patch_res: Dict[str, Dict[str, Any]],
    abq_patch_res: Dict[str, Dict[str, Any]],
    bending_res: Dict[str, Dict[str, Any]],
    vol_res: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Generate all three figures, return {name: absolute_path}."""
    return {
        "patch_test_errors": plot_patch_test_errors(elements, patch_res, abq_patch_res),
        "bending_locking_ratio": plot_bending_locking_ratio(elements, bending_res),
        "volumetric_locking_ratio": plot_volumetric_locking_ratio(elements, vol_res),
    }
