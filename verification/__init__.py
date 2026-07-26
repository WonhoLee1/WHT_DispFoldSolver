"""
verification
============
Element and solver verification suite for the dispsolver FEM code.

Benchmark catalog
-----------------
1.  Patch test (constant strain reproduction)      — element + solver level
2.  3-point bending                                — solver level
3.  4-point bending                                — solver level
4.  Cantilever bending                             — solver level
5.  Uniaxial tension                               — solver + element level
6.  Uniaxial compression                           — solver + element level
7.  Volumetric (hydrostatic) compression           — element + solver level
8.  Volumetric (hydrostatic) tension               — element + solver level

Each benchmark is run on every available backend (NumPy sequential, JAX vmap)
and the results are compared against closed-form analytical solutions.

Usage
-----
    python -m verification.run_all            # run everything, generate report
    python -m verification.run_all --fast     # skip JAX JIT warm-up heavy cases

Outputs are written to ``verification/results/``:
    - ``verification_report.md``  — human-readable summary
    - ``results.json``            — machine-readable structured data
    - ``results.csv``             — flat table for spreadsheets
"""

from .theory import (
    plane_strain_D,
    plane_strain_modulus,
    beam_I,
    cantilever_tip_deflection,
    three_point_bending_deflection,
    four_point_bending_deflection,
    uniaxial_stress_strain,
    volumetric_pressure_volume_change,
)

__all__ = [
    "plane_strain_D",
    "plane_strain_modulus",
    "beam_I",
    "cantilever_tip_deflection",
    "three_point_bending_deflection",
    "four_point_bending_deflection",
    "uniaxial_stress_strain",
    "volumetric_pressure_volume_change",
]
