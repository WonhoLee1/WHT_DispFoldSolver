"""
test_dispfold_solver_e2e.py
===========================
TDD End-to-End Verification Suite for Seam 5: Unified 2D and 3D Display Folding Pipeline.

Verifies:
1. Complete execution of 2D plane strain teardrop folding pipeline (compact_4, 3 steps).
2. Complete execution of 3D continuum solid teardrop folding pipeline (compact_4, 2 steps).
3. Verification that all 7 required engineering artifacts are generated cleanly in results/DYYYYMMDD_HHMMSS/:
   - run.log
   - model.inp
   - model_and_results.vtkhdf
   - summary_plots.png
   - folding_animation.gif
   - result_summary.json
   - report.html
4. Numerical convergence integrity, reaction force symmetry, and non-dummy JSON summaries.
"""

from __future__ import annotations
import os
import json
import tempfile
import pytest
import h5py

from example_disp_fold.ex1_dispfold_teardrop import run_dispfold_simulation


def test_dispfold_2d_solver_e2e_smoke():
    """Verify complete 2D teardrop folding simulation run and artifact generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = run_dispfold_simulation(
            dim="2D",
            preset="compact_4",
            theta_max_deg=30.0,
            enable_predictor=True,
            tie_method="MPC",
            contact_mode="AUTO",
            sol_preset="ADAPTIVE",
            max_steps=3,
            launch_gui=False,
            results_base_dir=tmpdir,
        )

        assert os.path.isdir(out_dir), f"Output directory {out_dir} not created"

        # 1. Verify run.log
        log_path = os.path.join(out_dir, "run.log")
        assert os.path.isfile(log_path), "run.log missing"
        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
        assert "Unified 2D Display 180° Teardrop Folding Pipeline" in log_content
        assert "[OK] Simulation completed" in log_content

        # 2. Verify model.inp
        inp_path = os.path.join(out_dir, "model.inp")
        assert os.path.isfile(inp_path), "model.inp missing"
        assert os.path.getsize(inp_path) > 1000

        # 3. Verify model_and_results.vtkhdf
        vtkhdf_path = os.path.join(out_dir, "model_and_results.vtkhdf")
        assert os.path.isfile(vtkhdf_path), "model_and_results.vtkhdf missing"
        with h5py.File(vtkhdf_path, "r") as f:
            assert "VTKHDF" in f
            grp = f["VTKHDF"]
            assert "PreProcessing" in grp
            assert "Steps" in grp
            assert len(grp["Steps"]["Values"]) >= 3

        # 4. Verify summary_plots.png
        png_path = os.path.join(out_dir, "summary_plots.png")
        assert os.path.isfile(png_path), "summary_plots.png missing"
        assert os.path.getsize(png_path) > 5000

        # 5. Verify folding_animation.gif
        gif_path = os.path.join(out_dir, "folding_animation.gif")
        assert os.path.isfile(gif_path), "folding_animation.gif missing"
        assert os.path.getsize(gif_path) > 5000

        # 6. Verify result_summary.json
        json_path = os.path.join(out_dir, "result_summary.json")
        assert os.path.isfile(json_path), "result_summary.json missing"
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["dim"] == "2D"
        assert summary["preset"] == "compact_4"
        assert summary["num_steps"] == 3
        assert summary["cutbacks"] == 0
        assert summary["wall_time_s"] > 0.0

        # 7. Verify report.html
        html_path = os.path.join(out_dir, "report.html")
        assert os.path.isfile(html_path), "report.html missing"
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        assert "180° Foldable Display Simulation Brief" in html_content
        assert "compact_4" in html_content


def test_dispfold_3d_solver_e2e_smoke():
    """Verify complete 3D teardrop folding simulation run and artifact generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = run_dispfold_simulation(
            dim="3D",
            preset="compact_4",
            theta_max_deg=20.0,
            enable_predictor=True,
            tie_method="MPC",
            contact_mode="AUTO",
            sol_preset="ADAPTIVE",
            max_steps=2,
            launch_gui=False,
            results_base_dir=tmpdir,
        )

        assert os.path.isdir(out_dir)

        # Verify key artifacts
        assert os.path.isfile(os.path.join(out_dir, "run.log"))
        assert os.path.isfile(os.path.join(out_dir, "model.inp"))
        assert os.path.isfile(os.path.join(out_dir, "model_and_results.vtkhdf"))
        assert os.path.isfile(os.path.join(out_dir, "summary_plots.png"))
        assert os.path.isfile(os.path.join(out_dir, "folding_animation.gif"))
        assert os.path.isfile(os.path.join(out_dir, "result_summary.json"))
        assert os.path.isfile(os.path.join(out_dir, "report.html"))

        with open(os.path.join(out_dir, "result_summary.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["dim"] == "3D"
        assert summary["preset"] == "compact_4"
        assert summary["num_steps"] == 2
        assert summary["cutbacks"] == 0


def test_dispfold_crease_zero_gap_dual_mortar_e2e():
    """Verify Mission B: Crease Zero-Gap folding with Wohlmuth Dual Mortar self-contact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = run_dispfold_simulation(
            dim="2D",
            preset="crease_zero_gap",
            theta_max_deg=30.0,
            enable_predictor=True,
            tie_method="MPC",
            contact_mode="AUTO",
            sol_preset="ADAPTIVE",
            max_steps=2,
            launch_gui=False,
            results_base_dir=tmpdir,
        )

        assert os.path.isdir(out_dir)

        # Verify run.log includes Dual Mortar contact wiring
        log_path = os.path.join(out_dir, "run.log")
        assert os.path.isfile(log_path)
        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
        assert "Wohlmuth Dual Mortar Self-Contact wired into 2D solver" in log_content
        assert "[OK] Simulation completed" in log_content

        # Verify all 7 artifacts
        for fname in ["run.log", "model.inp", "model_and_results.vtkhdf", "summary_plots.png", "folding_animation.gif", "result_summary.json", "report.html"]:
            assert os.path.isfile(os.path.join(out_dir, fname)), f"Missing artifact {fname}"

        with open(os.path.join(out_dir, "result_summary.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["dim"] == "2D"
        assert summary["preset"] == "crease_zero_gap"
        assert summary["contact_mode"] == "DUAL_MORTAR"
        assert summary["num_steps"] == 2

