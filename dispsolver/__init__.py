"""
dispsolver
==========
JAX-based 2D Plane Strain Implicit Dynamic FEM Solver.

Specialized for display panel folding simulation with:
- Large deformation (Total Lagrangian kinematics)
- Hyperelastic (Neo-Hookean / YEOH / Arruda-Boyce) + Viscoelastic (Prony + WLF)
- Elasto-plastic (J2 table-based)
- Q4 (SRI B-bar) and T3 (F-bar) elements
- RBE2 hinge / Tie constraints (Lagrange Multiplier)
- Newmark-β implicit dynamic integration
"""

__version__ = "0.1.0"

try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass
