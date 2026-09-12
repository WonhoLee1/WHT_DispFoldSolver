# Element Benchmark Comparison Matrix — 3D Solid Elements

| Element Type | Formulation | Geometry / Kinematics | Cook's Membrane ($\nu=0.49995$) | Cantilever Elastica ($P=269.35\text{ N}$) | Shear Locking | Volumetric Locking | Error vs Analytical |
|---|---|---|---|---|---|---|---|
| **`C3D8`** | Standard 8-node Hexa | Small Strain / TL | 100% Error (0.0001 mm) | 37.1% Error | Severe | Severe | High |
| **`C3D8_FBAR_TL`** | Multiplicative F-bar | Total Lagrangian | Blows up without EAS | **8.03 m (<0.1%)** | Moderate | **Eliminated** | **<0.1% (Elastica)** |
| **`C3D8I_TL`** | 9-mode EAS + F-bar | Total Lagrangian | **23.96 mm (<0.5%)** | **8.03 m (<0.1%)** | **Eliminated** | **Eliminated** | **<0.1%** |
| **`C3D4_ANP_TL`** | 4-node Average Nodal Pressure | Total Lagrangian | Stable | ~7.80 m | None (Linear) | **Eliminated** | <3.0% |

### Key Takeaways
1. **`C3D8I_TL`** is the premier 8-node hexahedral element formulation, combining 9-mode EAS (eliminating bending shear locking) and F-bar multiplicative volumetric split (eliminating incompressibility locking near $\nu \to 0.5$).
2. **Newton Engine & Solver**: Evaluating `||r_free||` on unconstrained DOFs and utilizing Best-Step Safeguard Line Search ensures monotonic non-linear convergence across extreme strain levels.
