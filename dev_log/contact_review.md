# Self-Contact Penalty Smoothing — Review

## Implementation Summary

Two contact implementations exist:

### 1. `constraint/contact_jax.py` — `PenaltyContactConstraint`
- Node-node penalty based on Euclidean distance d = ‖x_i − x_j‖
- Quadratic penalty: Π = ½·k·(d − d₀)² when d < d₀
- JAX autodiff for force and tangent: removes manual normal-derivative derivation
- Spatial hash grid (dict-based) with Morton-style keys for O(N) broad-phase
- Used for **self-contact** in the hinge zone

### 2. `contact/contact_solver.py` — `ContactPair`
- Full node-to-segment (NTS) with master-slave surface pairing
- Closest-point projection with log(cosh) regularisation:
  Π = ε·δ·ln(cosh(−g_N/δ)) when g_N < 0
- Log(cosh) gives smooth contact force onset (C² continuous)
- Spatial hash grid for broad-phase
- Initial-configuration wrong-side culling

## Assessment

### Strengths
- **JAX autodiff** eliminates manual tangent derivation (critical near d→0 where n = (x_j−x_i)/d diverges)
- **Spatial hash** avoids O(N²) brute-force — essential for large contact node sets
- **log(cosh) regularisation** in ContactPair provides C²-smooth contact — well-suited for Newton-Raphson
- **Wrong-side culling** in ContactPair prevents back-face penetration

### Weaknesses
1. **`PenaltyContactConstraint` uses bare quadratic penalty** — C¹ discontinuity at activation (gap=0). The jump in contact force gradient degrades Newton convergence when contact activates/disactivates each iteration.
2. **`ContactPair` master stiffness only** (lines 355-359): slave-to-master coupling assembled only to slave DOFs. For self-contact where both sides deform, this is asymmetric/unbalanced.
3. **No augmented Lagrangian**: pure penalty requires k→∞ for zero penetration. With k=1e8 the penetration at 1kN contact force is ~1e-5 mm — acceptable for the folding problem.
4. **No friction**: sticking/sliding tangential contact not modelled. The folding hinge primarily involves normal contact, so this is acceptable.

### Recommendation for Display Folding Problem
The current `PenaltyContactConstraint` with d₀=0.2 and k_contact=1e8 is adequate for:
- Hinge-zone self-contact where opposite faces press together
- Normal-force-only contact (no sliding along interface)
- Penalty stiffness tuned to give <1e-5 mm penetration

**If Newton convergence issues arise at contact activation**, switch from quadratic penalty to the log(cosh) regularisation used in `ContactPair` — this would involve changing `_compute_local_contact` from:
```python
jnp.where(gap < 0.0, 0.5 * k_contact * gap ** 2, 0.0)
```
to:
```python
r = gap / delta
return jnp.where(gap < 0.0, k_contact * delta * jnp.log(jnp.cosh(r)), 0.0)
```
with δ ≈ 0.01·d₀ as the regularisation length.
