# Technical Report: Interface Mesh Coupling Between CPE6M (Quadratic Triangle) and CPE4H (Linear Hybrid Quad)

**Date**: 2026-09-13  
**Author**: Antigravity AI Engineering Assistant  
**Topic**: Conformal Node Matching, Kinematic Incompatibility, Surface Tie Coupling, and Order-Matching Strategies  
**Repository**: `WHT_DispFoldSolver`

---

## 1. Context & The User's Architectural Proposal

When configuring a multi-layer flexible display model consisting of:
- **Substrate / Cover Layers**: Quadratic modified triangles (**`CPE6M`**, 6 nodes, quadratic $P_2$ displacement).
- **Adhesive Interlayers (PSA, OCA)**: Linear hybrid quadrilaterals (**`CPE4H`**, 4 nodes, linear $P_1$ displacement with independent pressure $p$).

A fundamental geometrical and kinematic question arises at the shared interface:
> *"If `CPE6M` is adopted, its bottom edge contains a midside node between the corner vertices. To match node positions with `CPE4H` via direct conformal node sharing, should the PSA layer be discretized with twice as many elements horizontally?"*

This report analyzes the mechanical consequences of this 2:1 conformal refinement, identifies the underlying kinematic incompatibility issue, and compares the three primary engineering alternatives used in commercial CAE.

---

## 2. Analysis of the 2:1 Conformal Mesh Proposal

```
  [ Substrate Layer : CPE6M (Quadratic) ]    (1) ─────────── (M) ─────────── (2)   <-- Quadratic displacement curve (P2)
                                              │               │               │
  [ PSA Layer : CPE4H (Bilinear) ]           (A) ───── (B) ──┴── (C) ───── (D)   <-- Piecewise linear segments (P1)
                                              [  PSA Elem 1  ]    [  PSA Elem 2  ]
```

### 2.1 Geometrical Alignment
By bisecting each PSA element horizontally ($h_{\text{PSA}} = L_{\text{CPE6M}} / 2$), every node on the `CPE6M` boundary finds a coincident counterpart on the `CPE4H` boundary:
- Vertex 1 pairs with Node A.
- Midside Node M pairs with the shared interface Node B/C.
- Vertex 2 pairs with Node D.

This allows direct node-to-node connectivity in the finite element topology table without introducing Lagrange multipliers or penalty spring matrices.

### 2.2 The Kinematic Incompatibility Defect ($P_2$ vs $P_1$)
While displacement continuity is strictly enforced at the discrete node locations ($x_1, x_M, x_2$), it is **violated in the inter-nodal intervals**:
- The boundary displacement of `CPE6M` follows a quadratic parabola:
  $$u_{\text{CPE6M}}(x) = N_1(x) u_1 + N_2(x) u_2 + N_M(x) u_M \in P_2$$
- The boundary displacement of the two `CPE4H` elements consists of two piecewise straight lines:
  $$u_{\text{CPE4H}}(x) = \begin{cases} \tilde{N}_A(x) u_A + \tilde{N}_B(x) u_B \in P_1 & (x_1 \le x \le x_M) \\ \tilde{N}_C(x) u_C + \tilde{N}_D(x) u_D \in P_1 & (x_M \le x \le x_2) \end{cases}$$

**Mechanical Consequence**:
A finite displacement gap or penetration $\Delta u(x) = u_{\text{CPE6M}}(x) - u_{\text{CPE4H}}(x) \neq 0$ opens between adjacent nodes during bending. This introduces artificial localized traction oscillations and parasitic shear concentrations along the adhesive bondline.

### 2.3 Aspect Ratio Degradation of Thin Adhesive Layers
In foldable displays, PSA layers are typically $25 \sim 50\,\mu\text{m}$ thick while spanning tens of millimeters. Subdividing the horizontal element size by $2\times$ doubles the total number of PSA elements and alters the local aspect ratio ($AR = \Delta x / \Delta y$). While beneficial for spatial resolution, it increases the global system bandwidth and linear solve time.

---

## 3. The Three Robust Engineering Alternatives

To resolve the interface challenge cleanly, commercial CAE solvers (Abaqus, Ansys, LS-DYNA) offer three battle-tested strategies:

### Alternative 1: Non-Conformal Surface-to-Surface Tie (Abaqus Standard) ⭐⭐⭐
* **Method**:
  - Keep `CPE6M` on the substrate and `CPE4H` on the PSA, but mesh each layer independently at its optimal resolution (1:1 element count).
  - Couple the interface using the solver's `SurfaceTie` constraint (as implemented in `dispsolver/constraint/surface_tie.py` or master-slave multipoint constraints).
* **Kinematics**:
  $$\mathbf{u}_{\text{slave}}(\xi) = \sum_{j=1}^3 N_j^{\text{master}}(\xi) \mathbf{u}_j^{\text{master}}$$
* **Advantage**: Eliminates parasitic stress spikes via variational work-equivalent mortar blending; eliminates the need for manual 2:1 mesh splitting.

### Alternative 2: Polynomial Order Matching (Unify with Quadratic Elements) ⭐⭐
* **Option 2A (`CPE6M` throughout)**:
  - Discretize both the PET substrate and the PSA adhesive layer with `CPE6M`.
  - Because `CPE6M` incorporates volumetric $B$-bar projection and positive contact weighting, it handles hyperelastic incompressibility ($\nu \approx 0.499$) effectively.
  - The entire stack becomes a conformal triangular mesh with identical $P_2$ kinematics across all interfaces.
* **Option 2B (Substrate `CPE6M` + PSA `CPE8H`)**:
  - Use 8-node serendipity quadratic hybrid quadrilaterals (`CPE8H`) for the PSA layer.
  - `CPE8H` features top and bottom midside nodes, yielding a 1:1 conformal match with `CPE6M`'s quadratic boundary without mesh splitting.

### Alternative 3: Linear Quadrilateral Unification (`CPE4I` + `CPE4H`) ⭐⭐⭐
* **Method**:
  - Substrate layers: **`CPE4I`** (4-node Enhanced Assumed Strain Quad).
  - Adhesive layers: **`CPE4H`** (4-node Hybrid $u-p$ Quad).
* **Advantage**:
  - Both elements share the exact same 4-node topology and linear boundary kinematics.
  - Mesh alignment is naturally 1:1 with zero midside node complications.
  - Assembly throughput is maximized, and linear system size is minimized.

---

## 4. Summary Matrix & Recommendation

| Strategy | Mesh Topology | Coupling Mechanism | Kinematic Consistency | Computational Cost | Recommended Use Case |
|:---|:---|:---:|:---:|:---:|:---|
| **2:1 Conformal Bisection** | PET: `CPE6M`, PSA: $2\times$ `CPE4H` | Direct Node Sharing | ⚠️ Approximate ($P_2 \neq P_1$) | Medium (+2x PSA DOFs) | Simple models without contact constraints |
| **Alternative 1 (Surface Tie)** | PET: `CPE6M`, PSA: `CPE4H` | `SurfaceTie` (MPC) | ✅ Variational Blending | Low to Medium | Complex models with large contact deformations |
| **Alternative 2 (Order Matched)** | PET: `CPE6M`, PSA: `CPE6M` / `CPE8H` | Conformal $P_2$ Sharing | ✅ 100% Exact ($P_2 = P_2$) | High (quadratic DOFs) | Extreme localized stress precision requirements |
| **Alternative 3 (Linear Quad)** | PET: `CPE4I`, PSA: `CPE4H` | Conformal $P_1$ Sharing | ✅ 100% Exact ($P_1 = P_1$) | **Lowest (Fastest)** | **Global folding kinematics & fast iteration** |

### Final Engineering Recommendation
1. For **production folding simulation** where simulation throughput and simplicity are paramount, **Alternative 3 (`CPE4I` + `CPE4H`)** provides the cleanest, fastest, and most robust workflow.
2. If severe multi-body contact with the hinge plates necessitates **`CPE6M`** on the structural layers, couple it to **`CPE4H` via `SurfaceTie` (Alternative 1)** rather than forcing a 2:1 geometric bisection.
