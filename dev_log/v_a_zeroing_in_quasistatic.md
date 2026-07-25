# v/a Zeroing Fix in Quasistatic Mode

## Problem

`ex03_v4_rbe2_element.py` (prescribed-rotation drive, TARGET_ANGLE=30°, ~2121 Q4 elements) converged only to θ≈-12.02° before failing with NaN/Inf residuals. After the first failure, cutbacks continued infinitely — even dt=1e-6 (essentially 0° delta-theta) produced a 7.99e+03 force residual at root node N60(UX).

## Root Cause

In `quasistatic` mode (`static_mode=True`, α=0.0), the Newmark update formula still executes at every converged step:

```
a_k = (u_k - u_n - dt·v_n) / (β·dt²) − ((1−2β)/2β)·a_n
v   = v_n + dt·((1−γ)·a_n + γ·a_k)
```

With β=0.25, γ=0.5, and consecutive tiny-dt cutbacks (dt≈1e-4), this accumulates unphysical velocity/acceleration:

- After ~100 steps with varying dt, `v → ±1000` (clip limit), `a → ±10000` (clip limit)
- The Newmark predictor `u_k = u_n + dt·v_n + ½dt²·a_n` then perturbs u_k by ~0.4 mm per step
- At θ≈12°, the wing-root Q4 elements are already near their Jacobian singularity limit
- This predictor perturbation pushes elements past the distortion threshold → NaN/Inf in assembly

The infinite cutback loop occurred because even with dt→1e-6, the predictor perturbation `dt·v_n ≈ 1e-6 × 1000 = 0.001 mm` was enough to trigger element failure at the critical root nodes.

## Fix

**File**: `dispsolver/solver/dynamic.py` lines 1669–1676

In `quasistatic` mode, zero out v and a after each converged step instead of running the Newmark update:

```python
if self.static_mode:
    self.a = np.zeros_like(a_k)
    self.v = np.zeros_like(v_n)
    self.a_extra = np.zeros_like(a_ext_k)
    self.v_extra = np.zeros_like(v_ext_n)
else:
    self.a = np.clip(a_k.copy(), -1.0e4, 1.0e4)
    self.v = np.clip(v_n + dt * ((1.0 - gamma) * a_n + gamma * a_k), -1.0e3, 1.0e3)
```

## Result

| Metric | Before | After |
|--------|--------|-------|
| Max convergence angle | -12.02° | **-13.05°** |
| Infinite cutback loop | Yes (dt→1e-6 still fails) | No (clean failure with iteration output) |
| Test suite | 140/140 pass | **140/140 pass** (no regressions) |

The convergence limit increased by ~1°, and the pathological infinite cutback behavior is eliminated.

## Remaining Issue

The fundamental barrier at ~13° is **Q4 element distortion at the wing root**, not solver numerics. The prescribed-rotation drive imposes `u_s = (R(θ)-I)·d₀` as displacement BCs on ALL wing nodes, forcing elements along a rigid kinematic path incompatible with Q4 bilinear shape functions at large rotations.

### Verified: RBE2 Element is Correct for Large Deformations

The RBE2 element uses exact rotation matrices (`cos θ`, `sin θ`) with Total Lagrangian formulation:
- Constraint: `g = u_s - u_m - (R(θ)-I)·d₀` (no small-angle approximation)
- Local Newton solves `g=0` for θ using exact dR/dθ Jacobian
- Geometric stiffness: `K_θθ += Σ λ·∂²g/∂θ²` (second-derivative consistent tangent)
- `d_n` parameter in `_build_tangent` is dead code (unused) — harmless, docstring only

The RBE2 element is **disabled** (`rbe2_elements=[]`) in ex03_v4 because it conflicts with prescribed DOFs (penalty K + BC K=1 on same DOFs → ill-conditioned global system). The failure at -13° is purely from prescribed-displacement-driven Q4 distortion.

## Next Steps (Recommendation)

1. **Switch to θ BC drive** (like v3-1): Add extra DOF for hinge rotation θ, prescribe θ via BC, let RBE2 constraint naturally propagate rotation to wing nodes. This gives elements freedom to find a compliant deformation path.
2. **Riks arc-length continuation**: For passing through limit points (requires significant solver changes).
3. **Mesh refinement at hinge**: Graded mesh near root to reduce element distortion per degree of rotation.
4. **Accept 13° limit** for the prescribed-rotation drive and document it as a formulation limitation.
