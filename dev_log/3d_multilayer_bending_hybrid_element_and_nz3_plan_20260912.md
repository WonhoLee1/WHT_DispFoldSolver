# Implementation Plan: 3D Multi-Layer Bending with Hybrid Hex Elements (`C3D8H`) and $Z$-Direction Mesh Refinement (`nz = 3`)

## 1. Goal Description

In the 3D multi-layer thin bar folding simulation (`ex15`), severe element distortion and element inversion (crushing/pinching of the soft PSA layer into negative volume) were observed at large bending angles (as shown in the user-provided image). 

```
  Top PET Layer (J2 Plasticity)      ═════════════════════════
  Middle PSA Layer (Hyperelastic)    ░░░░░░░░░░░░░░░░░░░░░░░░░  <-- Crushed / Inverted
  Bottom PET Layer (J2 Plasticity)   ═════════════════════════
```

This plan addresses the root causes by:
1. **Developing a 3D Co-Rotational Hybrid Hexahedral Element (`C3D8H` / `C3D8_HYBRID`)**:
   - Formulates a mixed $u-p$ / volume-averaged pressure ($\bar{B}_{\text{vol}}$) hybrid element specifically for nearly incompressible hyperelastic materials ($\nu \to 0.5$, $K \gg \mu$).
   - Completely eliminates volumetric locking, artificial hydrostatic pressure blow-up, and element collapse.
2. **Fixing the Neo-Hookean Hyperelastic Parameter Mismatch**:
   - Corrects `DynamicSolver3D` and `numba_materials.py` where $C_{10}, D_1$ were incorrectly mapped to $E, \nu$ ($\nu=0.01$, virtually zero bulk modulus).
   - Enforces true hyperelastic properties: $\mu = 2 C_{10}$, $K = \frac{2}{D_1}$ ($\nu \approx 0.4997$).
3. **Enabling Heterogeneous Multi-Element Assembly in `DynamicSolver3D`**:
   - Allows PET layers to use `C3D8_CR` (Co-rotational J2 Plasticity) and PSA layers to use `C3D8H` (Co-rotational Hybrid Hyperelastic) in the same global solver.
4. **Refining the $Z$-Direction Mesh (`nz = 3`) & Kinematics**:
   - Increases the width division to 3 elements in $Z$ (`nz = 3`), resolving out-of-plane aspect ratios and Poisson deformation without diagonal face wrinkling.
   - Refines kinematic wing rotation and draw-in boundary conditions.

---

## 2. Root Cause Analysis

| Factor | Current State | Root Cause & Failure Mechanism | Fix in this Plan |
|---|---|---|---|
| **Material Mismatch** | `mat_psa = {"C10": 0.1, "D1": 0.01}` | `DynamicSolver3D` passes $C_{10}$ to $E$ and $D_1$ to $\nu$. Result: $\nu = 0.01, K = 0.034\text{ MPa}$! PSA had zero bulk stiffness and collapsed to zero volume. | Map $C_{10}, D_1 \implies \mu=0.2\text{ MPa}, K=200\text{ MPa}, \nu \approx 0.4997$. |
| **Volumetric Locking** | Standard displacement formulation | Near incompressibility ($\nu \approx 0.4997$) locks all 8 Gauss points in standard Hex8 elements, leading to numerical divergence or severe artificial shearing. | Implement `C3D8H` with volume-averaged constant pressure $p_0 = K \bar{\varepsilon}_{\text{vol}}$. |
| **Element Dispatch** | Single element type for whole mesh | `DynamicSolver3D` checked only `first_elem.elem_type`. Could not assign `C3D8_CR` to PET and `C3D8H` to PSA simultaneously. | Implement per-element / per-PID element kernel dispatch in `DynamicSolver3D`. |
| **$Z$-Direction Discretization** | `nz = 1` | Single element spanning width creates non-coplanar face warping and diagonal creases in visualization. | Mesh with `nz = 3` across width $W=1.0\text{ mm}$. |

---

## 3. User Review Required

> [!IMPORTANT]
> **Boundary Conditions with $nz = 3$**:
> When using 3 elements in the $Z$-direction, we will fix $U_z = 0$ on the symmetry plane ($z = W/2 = 0.5\text{ mm}$) to prevent rigid body drift in $Z$ while allowing natural Poisson lateral contraction/expansion ($-\nu \varepsilon_{xx}$) across the width. Alternatively, if strict plane-strain is desired, $U_z=0$ can be applied to all nodes. The plan defaults to plane-strain ($U_z=0$ on all nodes) for direct comparison with 2D folding, with an option to enable 3D Poisson free-surface expansion.

---

## 4. Proposed Changes

### Component 1: 3D Constitutive Law & Hyperelastic Parameter Mapping
#### [MODIFY] [`dispsolver/material3d/numba_materials.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/material3d/numba_materials.py)
- Correct `MAT_HYPERELASTIC_NEOHOOKEAN`:
  - Input `props`: `[mu, K, C10, D1]`
  - Compute deviatoric shear stress $S_{\text{dev}} = 2\mu \text{dev}(E)$ and volumetric pressure $S_{\text{vol}} = K \text{tr}(E) I$.
  - Tangent matrix:
    $$C_{\text{mat}} = K (I \otimes I) + 2\mu \left(I_{\text{sym}} - \frac{1}{3} I \otimes I\right)$$

---

### Component 2: 3D Co-Rotational Hybrid Hexahedral Element (`C3D8H`)
#### [NEW] [`dispsolver/element3d/c3d8_hybrid_numba.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element3d/c3d8_hybrid_numba.py)
- 8-Node Co-Rotational Hybrid Element formulation:
  1. **Rigid Body Rotation**: $R_{\text{elem}}$ computed from deformed natural edge triads.
  2. **Local Kinematics**: $u_{\text{local}} = R_{\text{elem}}^T (x - x_c) - (X - X_c)$.
  3. **Volume Dilatation & Mixed Pressure**:
     $$\bar{B}_{\text{vol}} = \frac{1}{V_0} \int_{\Omega_0} B_{\text{vol}}(\xi, \eta, \zeta) dV_0$$
     $$\bar{\varepsilon}_{\text{vol}} = \bar{B}_{\text{vol}} u_{\text{local}}, \quad p = K \bar{\varepsilon}_{\text{vol}}$$
  4. **Deviatoric Strain**: $\varepsilon_{\text{dev}}(\xi) = B_{\text{dev}}(\xi) u_{\text{local}}$.
  5. **Internal Force**:
     $$f_{\text{local}} = \int_{\Omega_0} B_{\text{dev}}^T (2\mu \varepsilon_{\text{dev}}) dV_0 + V_0 \bar{B}_{\text{vol}}^T p$$
  6. **Stiffness Matrix**:
     $$K_{\text{local}} = \int_{\Omega_0} B_{\text{dev}}^T (2\mu I_{\text{dev}}) B_{\text{dev}} dV_0 + V_0 K \bar{B}_{\text{vol}}^T \bar{B}_{\text{vol}}$$
     $$K_{\text{global}} = \mathbf{T}^T K_{\text{local}} \mathbf{T}, \quad f_{\text{global}} = \mathbf{T}^T f_{\text{local}}$$
- Provides parallel Numba assembly kernel `assemble_mesh_c3d8_hybrid_numba`.

---

### Component 3: Heterogeneous Multi-Element Assembly in 3D Solver
#### [MODIFY] [`dispsolver/solver3d/dynamic3d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/solver3d/dynamic3d.py)
- In `_setup_numba_topology()`:
  - Calculate $\mu = 2 C_{10}$ and $K = 2 / D_1$ for Neo-Hookean entries in `elem_props`.
  - Classify elements into type groups:
    - Group 1: `C3D8_CR` / `C3D8` (PET layers)
    - Group 2: `C3D8H` / `C3D8_HYBRID` (PSA layers)
    - Group 3: `C3D8I` / `C3D8_EAS`
- In `assemble_system()`:
  - If elements have mixed types, assemble each group with its respective kernel and scatter into the pre-allocated CSR matrix, eliminating the single-type restriction.

---

### Component 4: 4-Layer Thin Bar Bending Demo Refinement
#### [MODIFY] [`examples/ex15_multilayer_thin_bar_fold.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/examples/ex15_multilayer_thin_bar_fold.py)
- Set `nz = 3` (3 elements in $Z$-direction, width $W = 1.0\text{ mm}$, $dz = 0.333\text{ mm}$).
- Assign `C3D8_CR` to PET layers (`pid=0`) and `C3D8H` to PSA layers (`pid=1`).
- Set accurate Neo-Hookean parameters for PSA ($C_{10}=0.1\text{ MPa}, D_1=0.01\text{ MPa}^{-1} \implies \mu=0.2, K=200$).
- Apply kinematically consistent wing rotations with pivot offset to prevent end-face shear pinching.
- Re-run solve and save `examples/ex15_result.pkl`.

#### [MODIFY] [`examples/ex15_3d_viewer.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/examples/ex15_3d_viewer.py)
- Render updated deformed shapes and generate animations showing the smooth, un-inverted $nz=3$ folding deformation.

---

## 5. Verification Plan

### Automated Tests
```powershell
# 1. Unit test for 3D Hybrid Element: Locking-free at nu -> 0.4999, rank=18, symmetry
pytest tests/test_c3d8_hybrid.py -v

# 2. Existing 3D test suite regression
pytest tests/test_3d_corotational.py tests/test_3d_locking_free.py tests/test_3d_numba.py -v

# 3. Verification benchmarks suite
python -m verification.run_all
```

### Manual Verification & Visual Inspection
1. Run `python -u examples/ex15_multilayer_thin_bar_fold.py` and confirm:
   - Zero element inversions throughout the 180° folding process.
   - Smooth convergence without cutbacks.
2. Run `python -u examples/ex15_3d_viewer.py`:
   - Inspect generated PNGs (`ex15_3d_deformed_isometric.png`, `ex15_3d_deformed_side.png`) and GIF animation.
   - Verify that the PSA layer maintains positive thickness, shears cleanly, and does NOT invert or form V-shaped wrinkles.
