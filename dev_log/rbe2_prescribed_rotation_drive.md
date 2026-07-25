# RBE2 U-bend Folding: Prescribed-Rotation Drive (2026-07-01)

## TL;DR

- **Prescribed-rotation drive works** and converges cleanly: 171 Q4 elements, 20 steps, **3.99s** (0.2s/step), 2 Newton iters/step, max|u| grows 0.79 → 10.5 over a -90° fold.
- **RBE2 element is NOT used** in the prescribed-rotation drive path. The prescribed wing-tip BCs already enforce the rigid-rotation constraint; adding the RBE2 element puts a high penalty K on the same DOFs and causes the global system to be ill-conditioned (solver hangs in `solve_step`).
- **Force-driven (torque) drive remains BLOCKED** by the Abaqus DOF elimination + pinned-master moment-transfer issue (see below).

## Drive Mechanism Comparison

| Drive | Mechanism | RBE2 element | Convergence | Status |
|-------|-----------|--------------|--------------|--------|
| Prescribed rotation (θ prescribed → u_s = (R(θ)−I)·d₀) | Master pinned, wing-tip slave DOFs prescribed each step | NOT used (conflicts with prescribed DOFs) | ✅ Converges | **Practical** |
| Force/torque (f on wing tips, master pinned) | RBE2 element enforces constraint, master absorbs force | Required | ❌ Max\|u\|=0 | Blocked |

## Key Finding: RBE2 Element Conflicts With Prescribed DOFs

Symptom (quick test with RBE2 + prescribed BCs):
- Mesh: 6×3 = 10 Q4 elements, 18 nodes
- Solver hangs in `solve_step` (no response in 120s+)

Root cause: The RBE2 element's penalty K (auto-scaled as `100·E·h`, e.g. 4e5 for the realistic mesh) is assembled at all element DOFs (master + slaves). When the wing tips are also prescribed (`K[slave,slave]=1` via BC elimination), the system has **conflicting constraints**:
- Prescribed BC: `K[slave,slave]=1`, displacement fixed at rigid-rotation offset
- RBE2 element: `K[slave,slave]=penalty·C^T_s C_s` (large), tries to enforce the same constraint
- PARDISO factors this ill-conditioned system slowly or not at all.

**Resolution**: For prescribed-rotation drive, set `rbe2_elements=[]`. The prescribed BCs alone are sufficient because they *are* the exact rigid-rotation field at the wing tips. The Q4 mesh resists the imposed deformation; the RBE2 element would only add redundant stiffness.

## Force-Driven (Torque) Drive: Root Cause (Carried Over)

Symptom (original ex03_v4 with force on wing tips, master pinned):
- All steps converge in 1 iter with `max|u| = 0.00mm` — the system is at trivial equilibrium.

Root cause: Abaqus/Standard-style RBE2 DOF elimination (in `_assemble` and `f_ext_mapped`):
1. `f_ext[slave]` is redirected to `f_ext[master]` — wing-tip loads are sent to the hinge centre.
2. The wing-tip slave DOFs become dummy equations (`K=1`, no coupling).
3. With the master pinned, the redirected force is absorbed by the BC; no moment is transmitted to the Q4 mesh, so `u=0` everywhere.
4. The RBE2 element's T-projection (`f_m = T^T f_e`, `K_m = T^T K_e T`) adds the penalty to master DOFs only, which are also pinned — the contribution is wiped by BC elimination.

**Previous fix attempt (Option A — conditional DOF elimination)** was tried on 2026-07-01 and **reverted** due to severe performance regression:
- With master pinned: skip the DOF mapping, assemble RBE2 element at all DOFs (no T-projection), skip force redirection.
- Result: physically correct (moment was transmitted), but first step took **62s → 300s+** (PARDISO factorises a system with 80 extra stiff DOFs at penalty=4e5; the BC elimination then zeroed the master-side coupling, leaving the slaves decoupled from the master → ill-conditioned).
- The correct fix would require either: (a) restructuring the solver so the RBE2 element is assembled *after* the BC elimination, or (b) implementing a "slave-only" RBE2 variant. Both are larger refactors and were deferred.

## Implementation: `examples/ex03_v4_rbe2_element.py`

```python
# Wing-tip nodes for prescribed-rotation drive
tip_bc_dofs, tip_d0 = precompute_tip_offsets(master_xy, coords, wing_nodes)

def theta_offsets(theta):
    c, s = np.cos(theta), np.sin(theta)
    RmI = np.array([[c-1, -s], [s, c-1]])
    return (RmI @ tip_d0.T).T.reshape(-1)

while solver.time < T_TOTAL:
    theta = -TARGET_ANGLE * factor(t)
    bc_dofs, bc_vals = build_full_bcs(theta)   # master + centre-line + wing tips
    solver.set_prescribed_dofs(bc_dofs, bc_vals)
    solver.solve_step(dt)
    factor = rotation_ampl(solver.time + dt)
```

The example file `examples/ex03_v4_rbe2_element.py` has been updated to this drive. The `rbe2_elements=[]` line is annotated with the rationale.

## Verification

| Test | Mesh | Steps | Time | Result |
|------|------|-------|------|--------|
| `_quick_no_rbe2_test.py` | 6×3 = 10 elements | 10 | 2.85s | ✅ max\|u\| 0 → 5.5 |
| `_quick_medium_prescribed_test.py` | 20×10 = 171 elements | 20 | 3.99s | ✅ max\|u\| 0.79 → 10.5 |
| `ex03_v4_rbe2_element.py` (full) | ~2121 elements | partial | 1.85s/step after warm-up | ⚠️ first step slow (~96s for PARDISO symbol+factor), subsequent steps fast |

The full example's first step is dominated by PARDISO symbolic + numerical factorisation for a ~4000-DOF system with multiple materials and Q4_EAS + Q4_VISCO_SIMO layers. Once factorised, subsequent steps reuse the factorisation and run in ~1.85s each (per the output before the 120s kill).

## Force-Driven (Torque) Drive: Future Work

A proper force-driven drive would require one of:

1. **Solver restructure**: Assemble the RBE2 element *after* the BC elimination in `solve_step`, so its penalty K is not zeroed out by the BC row/column zeroing. This requires moving the RBE2 assembly from the Newton-loop into a post-`solve_step` step (or computing the KKT matrix differently for the RBE2 part).
2. **Slave-only RBE2 variant**: A modified element that only has slave DOFs (no master) when the master is pinned. The condensation over θ is computed without the master component, and only the slave portion of `K_e` is assembled. This is the same idea as Option A but with a correctly-conditioned `K_e`.
3. **Master-free formulation**: Use a penalty or augmented-Lagrangian method that does not require a pinned master at all (e.g., the KKT v3-1 path with a free master, then prescribe the rotation differently).

All three are larger changes and are deferred until a concrete use case demands force-driven folding.

## References

- `dispsolver/element/rbe2.py` — NumPy RBE2 element
- `dispsolver/element/rbe2_jax.py` — JAX RBE2 element (same math, JIT)
- `dispsolver/solver/dynamic.py:2624-2646` — Abaqus DOF elimination in `_assemble`
- `dispsolver/solver/dynamic.py:995-1002` — `f_ext_mapped` force redirection
- `dispsolver/solver/dynamic.py:1106+` — RBE2 element assembly in `solve_step` Newton loop
- `examples/ex03_v4_rbe2_element.py` — prescribed-rotation drive example
- `.omo/plans/rbe2-element-refactor.md` — full plan history (Task 6, 8)

## Status

- **Test suite**: 140/140 passing
- **ex03_v4 prescribed-rotation drive**: converges (small/medium meshes verified; full mesh slow only on first factorisation)
- **ex03_v4 force-driven (torque) drive**: BLOCKED — pending solver restructure or slave-only RBE2 variant
