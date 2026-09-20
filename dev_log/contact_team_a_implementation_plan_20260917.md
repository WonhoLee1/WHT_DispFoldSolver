# Team A Detailed Implementation Plan: Dual Mortar & Nitsche Contact (2026-09-17)

> **CAE Formulation & Variational Mechanics Task Force (Team A)**  
> **Mission**: Eliminate contact gaps across 2D and 3D solvers by implementing 2D Dual Mortar S2S, 3D Nitsche Symmetric Contact, and Stage B Non-matching Mesh Clipping without any code duplication.

---

## 1. Zero Duplication Audit Summary (Team A Findings)

### 1.1 Existing & Validated Components (DO NOT MODIFY / RE-IMPLEMENT)
- `dispsolver/constraint3d/surface_contact3d_s2s.py`:
  - **Status**: 100% verified (`tests/test_surface_to_surface_contact.py`, 8 passed).
  - **Formulation**: Stage A (matching-topology meshes, 1 segment per slave face), Quad4 Wohlmuth dual basis $\Phi_i(\xi, \eta)$, lumped $D_{\text{global}}$ diagonal static condensation, PDASS augmented Lagrangian.
- `dispsolver/constraint3d/surface_contact3d_deformable.py`:
  - **Status**: 100% verified (`tests/test_deformable_soft_nonlinear_contact.py`, 8 passed).
  - **Formulation**: Node-to-Surface deformable contact with full pressure-overclosure support (Hard, Linear, Exponential, Tabular, Nonlinear).
- `dispsolver/constraint3d/pressure_overclosure.py`:
  - Standardized constitutive pressure-overclosure laws.

### 1.2 Identified Architectural Gaps (Team A Implementation Targets)
1. **Target A1: 2D Dual Mortar & Nitsche Contact Engine (`dispsolver/constraint/dual_mortar_contact.py`)**
   - Current 2D solver only possesses node-to-segment (NTS) penalty contact (`dispsolver/contact/contact_solver.py`).
   - Missing: Segment-to-Segment (S2S) Dual Mortar with 1D biorthogonal shape functions $\psi_i(\xi)$ and Nitsche's formulation for 2D plane strain/stress beams and laminates.
2. **Target A2: 3D Nitsche Multiplier-Free Symmetric Contact (`dispsolver/constraint3d/nitsche_contact3d.py`)**
   - Missing: Pure SPD symmetric variational contact preserving PARDISO LLT/Cholesky factorization without penalty penetration drift.
3. **Target A3: 3D Stage B Non-Matching Mesh Polygonal Clipping Engine (`dispsolver/constraint3d/dual_mortar_clipping3d.py`)**
   - Missing: Sutherland-Hodgman polygon clipping for meshes with dissimilar element densities (e.g. coarse slave on fine master).

---

## 2. Mathematical Formulation & Variational Design

### 2.1 Target A1: 2D Dual Mortar Segment-to-Segment Formulation

#### 2.1.1 1D Biorthogonal Dual Basis
For a 2-node line segment $\xi \in [-1, 1]$, standard linear shape functions are:
$$N_1(\xi) = \frac{1 - \xi}{2}, \quad N_2(\xi) = \frac{1 + \xi}{2}$$
The standard mass matrix on reference element is:
$$\mathbf{M}_{\text{ref}} = \int_{-1}^1 \begin{bmatrix} N_1^2 & N_1 N_2 \\ N_1 N_2 & N_2^2 \end{bmatrix} d\xi = \begin{bmatrix} \frac{2}{3} & \frac{1}{3} \\ \frac{1}{3} & \frac{2}{3} \end{bmatrix}$$
The row sum diagonal matrix is:
$$\mathbf{D}_{\text{ref}} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$
The transformation matrix $\mathbf{A} = \mathbf{D}_{\text{ref}} \mathbf{M}_{\text{ref}}^{-1}$:
$$\mathbf{A} = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$$
The closed-form 1D dual basis functions are:
$$\psi_1(\xi) = 2 N_1(\xi) - N_2(\xi) = \frac{1 - 3\xi}{2}$$
$$\psi_2(\xi) = -N_1(\xi) + 2 N_2(\xi) = \frac{1 + 3\xi}{2}$$

#### 2.1.2 Biorthogonality Verification
$$\int_{-1}^1 \psi_i(\xi) N_j(\xi) d\xi = \delta_{ij} \quad (\text{Exact Biorthogonality})$$
$$\sum_{i=1}^2 \psi_i(\xi) \equiv 1 \quad (\text{Partition of Unity})$$

#### 2.1.3 Weak Gap Aggregation & Static Condensation
For each slave segment $s$, length $L_s = \|\mathbf{X}_2 - \mathbf{X}_1\|$, Jacobian $J_s = L_s / 2$:
$$D_a = \sum_{s \in \text{faces}(a)} \frac{L_s}{2}$$
$$\tilde{g}_a = \frac{1}{D_a} \sum_{s \in \text{faces}(a)} \int_{-1}^1 \psi_a(\xi) g_n(\xi) \frac{L_s}{2} d\xi$$
$$\mathbf{f}_{\text{contact}} = \sum_{a \in \mathcal{A}} p_a \int_{-1}^1 \psi_a(\xi) \left( N_b^{\text{slave}}(\xi) \mathbf{n} - N_c^{\text{master}}(\xi_m(\xi)) \mathbf{n} \right) \frac{L_s}{2} d\xi$$
where $p_a = \lambda_a + k_c \tilde{g}_a$ (PDASS Augmented Lagrangian).

### 2.2 Target A2: 3D Nitsche Symmetric Positive Definite (SPD) Contact

Instead of introducing multipliers $\mathbf{\lambda}$, Nitsche's formulation directly augments the weak form:
$$\delta W_{\text{Nitsche}} = -\int_{\Gamma_c} \theta_n(\mathbf{u}) \delta g_n d\Gamma - \int_{\Gamma_c} \delta \theta_n(\mathbf{u}) g_n d\Gamma + \gamma \int_{\Gamma_c} g_n \delta g_n d\Gamma$$
where:
$$\theta_n(\mathbf{u}) = \mathbf{n} \cdot \mathbf{\sigma}(\mathbf{u}) \cdot \mathbf{n}$$
$$\gamma = \frac{\gamma_0 E}{h_{\text{elem}}} \quad (\text{Nitsche stabilization parameter, } \gamma_0 \approx 10 \sim 50)$$

#### Key Numerical Properties:
1. **Zero Multipliers**: Total system size remains exactly $N_{\text{DOFs}}$.
2. **Strict Symmetry**: $K_{ij}^{\text{Nitsche}} = K_{ji}^{\text{Nitsche}}$ identically, preserving PARDISO symmetric positive definite (SPD) solver mode.
3. **No Penalty Penetration**: Unlike pure penalty, convergence to zero penetration error ($g_n \to 0$) is achieved with modest $\gamma$ without ill-conditioning.

---

## 3. Step-by-Step Implementation Roadmap (Team A Execution)

### Step 1.1: 2D Dual Mortar Class Implementation
- **File**: `dispsolver/constraint/dual_mortar_contact.py`
- **Class**: `DualMortarContactConstraint2D`
- **Capabilities**:
  - `_build_reference_mortar_matrices()`: 1D Gauss quadrature (2-point), analytical $\psi_1, \psi_2$ evaluation.
  - `_compute_g_tilde(u)`: Discrete weak gap aggregation.
  - `assemble(u)`: Nodal force distribution and rank-1 outer product consistent tangent stiffness matrix.
  - `update_augmented_multipliers(u, omega)`: PDASS Uzawa multiplier update.
  - `formulation="dual_mortar" | "nitsche"` switch.

### Step 1.2: Integration into `DynamicSolver2D`
- **File**: `dispsolver/solver2d/dynamic2d.py`
- Connect `add_constraint()`, `_contact_active_set()`, and `solve_step()` to seamlessly accept `DualMortarContactConstraint2D`.

### Step 1.3: Rigorous Verification Test Suite
- **File**: `tests/test_dual_mortar_2d.py`
- **Tests**:
  1. `test_1d_dual_basis_biorthogonality()`: Machine-precision test $\int \psi_i N_j dx = \delta_{ij} D_i$.
  2. `test_partition_of_unity()`: $\sum \psi_i \equiv 1.0$.
  3. `test_2d_patch_test_nonmatching_mesh()`: Coincident blocks with dissimilar mesh densities (e.g. 5 elements on 3 elements) transmitting uniform pressure without stress spikes.
  4. `test_uzawa_contraction()`: Geometric decay of penetration to floating-point floor ($< 10^{-10}$).
  5. `test_nitsche_symmetry()`: Verify $K_{\text{global}} = K_{\text{global}}^T$ within $10^{-12}$.

---

## 4. Acceptance Criteria
- [ ] 1D Biorthogonality error $< 10^{-14}$.
- [ ] Non-matching 2D patch test stress uniformity error $< 0.1\%$.
- [ ] 100% test pass on new test suite `tests/test_dual_mortar_2d.py`.
- [ ] Zero regression on existing 2D and 3D solvers and tests.
