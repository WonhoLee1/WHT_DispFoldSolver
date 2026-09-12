# Implementation Plan: Commercial-Grade 3D Solid Element Controls (`SectionControls`), Distortion Control, & Anti-Inversion Safeguards

## 1. Goal Description

In non-linear finite element analysis (FEA) involving large deformations, thin-sheet bending, or soft layers (e.g., PSA, rubbers, foams), elements can experience localized severe compression or shearing. Even with high bulk modulus, extreme non-linear trial steps or localized pinching can lead to **element inversion** ($\det(F) \le 0$), negative Jacobian, or mesh entanglement.

Commercial solvers like **Abaqus** provide sophisticated theoretical and numerical options via `*SECTION CONTROLS` (e.g., `DISTORTION CONTROL`, `HOURGLASS CONTROL`, `VISCOUS DAMPING`, `SECOND ORDER OBJECTIVITY`) to monitor and prevent element inversion and crushing.

This plan designs and implements:
1. **Abaqus CAE-Compatible `SectionControls` Architecture**:
   - `model.SectionControls(name, distortionControl=ON/OFF, lengthRatio=0.1, hourglassControl=ENHANCED, viscousDamping=..., antiInversionBarrier=ON/OFF)`
   - Seamless linkage to `SolidSection(..., controls='...')` and element instances.
2. **Element-Level Distortion Control & Anti-Inversion Barrier (Numba Kernels)**:
   - When an element's minimum characteristic length or volume ratio drops below threshold $r = \text{lengthRatio}$ (default 0.1):
     An elastic restoring barrier force and stiffness ($f_{\text{dist}}, K_{\text{dist}}$) are activated, placing an energetic barrier that prevents $\det(F) \to 0$ or negative volume.
   - Internal element viscous damping ($\beta K \dot{u}$) to suppress high-frequency local wrinkles.
3. **Solver-Level Line-Search Inversion Guard**:
   - Prevents Newton-Raphson trial steps from landing in an inverted state by dynamically back-tracking step size $\alpha$.

```
     Normal Deformation (J > 0.1)           Critical Zone (J < 0.1)
   ────────────────────────────────     ──────────────────────────────────
   Pure constitutive stress & tangent   Constitutive + Distortion Barrier Force
                                        Steep energetic repulsion prevents J <= 0
```

---

## 2. User Review Required

> [!IMPORTANT]
> **Default Option Configuration**:
> - `distortionControl`: `ON` (default `lengthRatio = 0.1`, matching Abaqus). When an element's thickness/volume compresses below 10%, a stabilizing barrier smoothly repels further collapse.
> - `antiInversionBarrier`: `ON` (activates logarithmic potential $W_{\text{vol}} \sim \frac{K}{2}(\ln J)^2 \to +\infty$ as $J \to 0^+$).
> - `viscousDamping`: Default `0.0` (can be configured by user, e.g., `0.01` to damp out sudden snap-through or shear oscillations).
> - All options can be explicitly turned `OFF` or customized per Section.

---

## 3. Mathematical Formulations

### 1) Distortion Control Barrier (Abaqus Formulation)
Monitors characteristic volume ratio $J = \det(F)$ or minimum edge length ratio $r = L_{\min} / L_0$.
When $J < J_{\text{crit}}$ (where $J_{\text{crit}} = \text{length\_ratio}$, default 0.1):
$$\Psi_{\text{dist}}(J) = \frac{1}{2} k_{\text{dist}} \left( \frac{J_{\text{crit}} - J}{J} \right)^2 \quad (J < J_{\text{crit}})$$
The resulting anti-inversion restoring stress and tangent are:
$$p_{\text{dist}} = - \frac{\partial \Psi_{\text{dist}}}{\partial J} = k_{\text{dist}} \frac{J_{\text{crit}} (J_{\text{crit}} - J)}{J^3}$$
$$C_{\text{dist}} = \frac{\partial^2 \Psi_{\text{dist}}}{\partial J^2} = k_{\text{dist}} \frac{J_{\text{crit}} (2 J_{\text{crit}} - J)}{J^4}$$
- As $J \to J_{\text{crit}}$: $p_{\text{dist}} \to 0$ (smooth activation, $C^1$ continuous).
- As $J \to 0^+$: $p_{\text{dist}} \to +\infty$ (infinite restoring force, completely preventing inversion).

### 2) Element-Level Viscous Damping
Suppresses high-frequency localized mesh oscillation during severe shear:
$$f_{\text{damp}} = \beta \cdot K_{\text{elem}} \cdot \dot{u}_{\text{elem}} = \frac{\beta}{\Delta t} K_{\text{elem}} \Delta u_{\text{elem}}$$
$$K_{\text{damp}} = \frac{\beta}{\Delta t} K_{\text{elem}}$$

### 3) Solver Line-Search Inversion Guard
During Newton line-search $\alpha \in (0, 1]$:
If $\min_{e} \det(F_e(u + \alpha \Delta u)) \le J_{\min\_safe}$ (e.g. $0.02$):
Line search automatically halves $\alpha \leftarrow 0.5 \alpha$ until the trial state is strictly positive-volume, preventing singular tangents and diverged return mappings.

---

## 4. Proposed Changes

### Component 1: CAE Model Architecture (`dispsolver/model/`)
#### [MODIFY] [`dispsolver/model/section.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/model/section.py)
- Define `SectionControls` dataclass:
  ```python
  @dataclass
  class SectionControls:
      name: str
      distortion_control: bool = True
      length_ratio: float = 0.1
      hourglass_control: str = "ENHANCED" # "ENHANCED", "STIFFNESS", "VISCOUS"
      viscous_damping: float = 0.0
      anti_inversion_barrier: bool = True
      min_det_f: float = 0.05
  ```
- Update `SolidSection` to support `controls: Optional[Union[str, SectionControls]] = None`.

#### [MODIFY] [`dispsolver/model/model.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/model/model.py)
- Add `model.SectionControls(name, ...)` method to register controls in `self.section_controls`.
- Expose `SectionControls` in `dispsolver/model/__init__.py`.

#### [MODIFY] [`dispsolver/model/assembly.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/model/assembly.py)
- In `build_solver_system()`, extract section controls per element and compile them into `sys.elem_controls` (DOD array of control flags and coefficients).

---

### Component 2: 3D Element Numba Kernels (`dispsolver/element3d/`)
#### [MODIFY] [`dispsolver/element3d/c3d8_hybrid_numba.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element3d/c3d8_hybrid_numba.py)
- Incorporate distortion control barrier and element damping into `compute_c3d8_hybrid_element_umat_numba`:
  - If `distortion_control` is active and $J < J_{\text{crit}}$:
    Inject $p_{\text{dist}}$ and $C_{\text{dist}}$ into volumetric dilatation response.
  - If `viscous_damping > 0`:
    Add stiffness-proportional damping force.
- Update `assemble_mesh_c3d8_hybrid_numba` to accept `elem_controls`.

#### [MODIFY] [`dispsolver/element3d/c3d8_corotational_numba.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element3d/c3d8_corotational_numba.py)
- Add equivalent distortion control barrier and damping for standard/plastic solid elements.

---

### Component 3: 3D Dynamic Solver (`dispsolver/solver3d/dynamic3d.py`)
#### [MODIFY] [`dispsolver/solver3d/dynamic3d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/solver3d/dynamic3d.py)
- In `_setup_numba_topology()`:
  - Allocate and populate `self.elem_controls = np.zeros((n_elems, 8), dtype=np.float64)`.
    - `col 0`: `distortion_control` (1.0 = ON, 0.0 = OFF)
    - `col 1`: `length_ratio` (e.g. 0.1)
    - `col 2`: `viscous_damping` (e.g. 0.0)
    - `col 3`: `anti_inversion_barrier` (1.0 = ON, 0.0 = OFF)
    - `col 4`: `min_det_f` (e.g. 0.05)
- In `assemble_system()`:
  - Pass `self.elem_controls` to Numba assembly kernels.
- In `solve_step()` line search:
  - Add `check_inversion_safeguard(u_trial)`: back-track if any element volume ratio drops below `min_det_f`.

---

## 5. Verification Plan

### Automated Tests
```powershell
# 1. Unit tests for SectionControls API and options toggling
pytest tests/test_section_controls.py -v

# 2. Extreme compression test: Element compressed to 95% strain (thickness -> 0.05)
# Verifies distortion barrier activates and prevents negative volume
pytest tests/test_c3d8_distortion_control.py -v

# 3. Existing 3D and hybrid test suite regression
pytest tests/test_c3d8_hybrid.py tests/test_3d_corotational.py tests/test_3d_locking_free.py -v

# 4. Full solver verification suite
python -m verification.run_all
```

### Manual Verification
1. Run a benchmark with severe compression and compare:
   - Case A (`distortionControl=OFF`): Under excessive load, element inverts and solver fails with `detF <= 0`.
   - Case B (`distortionControl=ON`): Distortion barrier activates, preventing inversion and allowing solver to smoothly converge or cut back without crash.
