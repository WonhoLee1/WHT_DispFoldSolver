# Numerical Lessons Learned: Timestep Decay & Abaqus Convergence Parity (2026-10-10)

## 1. Background & Problem Statement

In the 90-degree flexible display folding simulation (`examples/ex12a_abaqus_inp_plate_fold.py`),
the solver experienced a severe slowdown where the adaptive time increment stalled down to:
$$dt \approx 0.00022\text{ s} \quad (\approx 1/45 \text{ of target } dt_{\max} = 0.01\text{ s})$$
At this rate, completing the $t = 1.0\text{ s}$ (180° combined face-to-face closure) required several thousand steps, causing massive runtime overhead.

---

## 2. Root Cause Analysis: The Numerical Death Spiral

Detailed inspection of the iteration logs in Steps 60–62 revealed a three-stage numerical death spiral:

### 2.1 The Nanometer Displacement Trap ($C_n \Delta u_{\max}$)
- Standard displacement correction acceptance requires:
  $$c_{\max} \le C_n \Delta u_{\max} \quad (C_n = 0.01)$$
- When $dt$ drops (e.g. $dt = 0.00028\text{ s}$), the incremental displacement $\Delta u_{\max}$ shrinks proportionally to $\approx 18\text{ µm}$.
- Consequently, the displacement correction tolerance collapsed to:
  $$\text{tol}_c = 0.01 \times 18\text{ µm} = 0.18\text{ µm} = 180\text{ nm}$$
- Due to machine precision limitations, nonlinear hyperelastic/viscoelastic PSA layers, and kinematic multi-layer shear, the actual correction $c_{\max}$ hovered around $0.33 \sim 0.60\text{ µm}$.
- **Crucially, the force residual at this stage was already $R_{\max} = 8.51 \times 10^{-5}\text{ N}$ ($0.085\text{ mN}$)** — physically in near-perfect equilibrium!
- However, failing the $180\text{ nm}$ displacement criterion prevented increment acceptance.

### 2.2 False-Positive Stall Detector Triggering
- At near-zero residual ($R_{\max} \approx 10^{-4}\text{ N}$), residual ratio oscillations around $1.0$ naturally occur due to floating-point rounding.
- The stall detector was blindly checking:
  $$\text{conv\_rate} > 0.995 \text{ for } 4+ \text{ iterations} \implies \text{CUTBACK}$$
- It lacked a guard verifying whether the absolute force residual was already tiny.
- As a result, the solver treated successful near-zero equilibrium as a "convergence stall" and cut $dt$ by $4\times$.

### 2.3 The Vicious Death Spiral
$$\text{Cutback} \longrightarrow dt \downarrow \longrightarrow \Delta u_{\max} \downarrow \longrightarrow C_n \Delta u_{\max} \downarrow \text{ (into sub-nanometer)} \longrightarrow \text{Impossible Convergence} \longrightarrow \text{Recutback}$$

---

## 3. Engineering Solution: Abaqus Standard Parity

To eliminate this trap while preserving strict physical rigor (Anti-Mocking Protocol), we implemented official Abaqus/Standard criteria ([Abaqus Theory Manual §2.2.1] & [Analysis User's Guide §7.2.2]):

### 3.1 Direct Equilibrium Acceptance (Theory Manual §2.2.1)
When the maximum residual force is tiny relative to the characteristic flux:
$$R_{\max} \le 10^{-4} \tilde{q} \quad \text{and} \quad c_{\max} < 0.01\text{ mm}$$
The increment is directly accepted as converged because physical equilibrium is achieved within engineering tolerance.

### 3.2 Small Correction Floor ($1\text{ µm}$)
When residual force satisfies the standard tolerance ($R_{\max} \le R_n \tilde{q}$):
$$c_{\max} \le 1.0 \times 10^{-3}\text{ mm} \quad (1\text{ µm})$$
The step is accepted, preventing the threshold from shrinking indefinitely into sub-nanometer regimes during small increments.

### 3.3 Residual-Guarded Stall Detector
Stall counter is only incremented if the residual is actually unresolved:
```python
if (res_norm_0 is not None and conv_rate > 0.995
        and res_ratio > max(self.rtol * 100.0, 0.05)
        and max_R_val > R_n_eff * q_avg):
    stall_count += 1
else:
    stall_count = 0
```

### 3.4 Adaptive Time Stepping Acceleration (`AdaptiveDtController`)
- Allowed up to $1.5\times$ growth factor when Newton converges in $\le 2$ iterations.
- Established clean bounds: `dt_min = 1e-4s`, `dt_init = 0.005s`, `dt_max = 0.01s`.

---

## 4. Verification Results

In the final production run (`task-3602`):
- **Full 90° Folding Reached:** Completed all 101 increments to $t = 1.0000\text{ s}$ ($\theta = \pm 90.00^\circ$, $180^\circ$ face-to-face full close).
- **Wall-Clock Time:** **593.89 s (9 min 53 s)** — dramatic speedup compared to prior multi-hour stall.
- **Timestep Stability:** Maintained maximum limit $dt = 0.01000\text{ s}$ across the entire run.
- **Cutbacks:** **Zero cutbacks (0 / 101 steps)**.
- **Mesh Quality:** Zero element inversion throughout the entire simulation (`n_inverted == 0`).
- **Interlayer Mechanics (Multi-layer Shear):** Total staircase slip at $x = \pm 40\text{ mm}$ was **$+515.90\text{ µm}$**, with **$510.31\text{ µm}$ (98.9%)** carried entirely by the compliant PSA layers, accurately capturing realistic foldable display mechanics.
- **Surface Tie:** Kinematic MPC tie maintained **0.0000 mm gap** with zero penetration and zero separation drift.

---

## 5. Local Peak DOF Flip-Flop Limit Cycle & Solution

### 5.1 Phenomenon at Step 27 ($t \approx 0.2525\text{ s}$, $\theta \approx 9.6^\circ$)
In Step 27, Newton-Raphson entered a 2-cycle periodic limit cycle oscillation:
- Odd iterations ($k = 9, 11, 13, \dots$): $\Delta u_{\text{incr}} \approx 0.6562\text{ mm}$, $R_{\max} = 31.6\text{ N}$ (at surface node `N5095`)
- Even iterations ($k = 10, 12, 14, \dots$): $\Delta u_{\text{incr}} \approx 0.6575\text{ mm}$, $R_{\max} = 31.6\text{ N}$ (at `N5095`)
- The correction vector was jumping back and forth by $\pm 0.162\text{ mm}$ at peak node `N2466(UY)`.

### 5.2 Why Global Damping Failed
The pre-existing damping checked global dot-product:
$$\frac{\mathbf{du}_k \cdot \mathbf{du}_{k-1}}{\|\mathbf{du}_k\| \|\mathbf{du}_{k-1}\|} < -0.6$$
Because the oscillation was confined to a localized cluster of surface and subsurface nodes around $x = -5.75\text{ mm}$ while the remaining 12,100 DOFs moved forwards, the global inner product remained close to zero, masking the severe local see-saw motion.

### 5.3 Abaqus-Style Peak DOF Flip-Flop Damping
Added localized peak DOF detection:
```python
_max_dof = np.argmax(np.abs(du))
_local_flip = (du[_max_dof] * self._prev_du[_max_dof] < 0) and (
    abs(abs(du[_max_dof]) - abs(self._prev_du[_max_dof])) / (abs(du[_max_dof]) + 1e-15) < 0.25
)
if _global_flip or _local_flip:
    du = 0.5 * du
    du_ext = 0.5 * du_ext
    dlam = 0.5 * dlam
```
By scaling `du` by $0.5$ when a local sign reversal is detected, the trial state lands directly on the midpoint of the limit cycle where the residual gradient changes sign, collapsing the flip-flop and enabling quadratic convergence into the true equilibrium state.

