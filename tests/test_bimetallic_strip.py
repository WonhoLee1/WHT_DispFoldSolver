"""
test_bimetallic_strip.py
========================
Automated regression test for Timoshenko Bimetallic Strip thermal bending.
Tests both 2D CPE4I and 3D SOLID_SHELL elements against Timoshenko (1925) exact theory.
"""

import pytest
from benchmark_element.benchmark_bimetallic_strip import run_benchmark_2d_bimetallic, run_benchmark_3d_bimetallic


def test_timoshenko_bimetallic_strip_2d():
    """Verify 2D Plane Strain CPE4I thermal bending matches Timoshenko (1925) within 0.5%."""
    passed, err_pct = run_benchmark_2d_bimetallic()
    assert passed, f"2D Bimetallic strip benchmark failed with error {err_pct:.3f}%"
    assert err_pct < 0.5, f"Expected error < 0.5%, got {err_pct:.3f}%"


def test_timoshenko_bimetallic_strip_3d():
    """Verify 3D SOLID_SHELL thermal bending matches Timoshenko (1925) within 0.5%."""
    passed, err_pct = run_benchmark_3d_bimetallic()
    assert passed, f"3D Bimetallic strip benchmark failed with error {err_pct:.3f}%"
    assert err_pct < 0.5, f"Expected error < 0.5%, got {err_pct:.3f}%"

