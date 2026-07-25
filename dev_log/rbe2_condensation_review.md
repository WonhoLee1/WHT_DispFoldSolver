# RBE2 Saddle-Point → Master-Slave Condensation — Review

## Current Approach: KKT Saddle-Point (Lagrange Multiplier)

The RBE2 hinge constraint is enforced via Lagrange multipliers in a KKT system:

```
[ K_eff    C^T     ] [ u   ]   [ R_u ]
[ C        0       ] [ lam ] = [ R_lam ]
```

Where:
- C is the linearised constraint Jacobian (2·n_slaves equations)
- θ is an extra primal DOF in `u_extra` coupled via C_ext
- Lagrange multipliers λ enforce the rigid-rotation constraint exactly

n_total = n_dofs + n_extra + n_lambdas = 2·n_nodes + 2·n_slaves + 2·n_RBE2_constraints

### Advantages (current)
1. **Exact constraint**: Lagrange Multiplier enforces g=0 to machine precision
2. **Consistent tangent**: geometric stiffness K_θθ from `extra_geometric_stiffness` maintains quadratic convergence
3. **Constraint reaction**: λ are the constraint forces at the hinge — useful for design
4. **Reusable infrastructure**: KKT assembly works for RBE2, Tie, and future constraints

### Disadvantages
1. **Larger system**: +2·n_slaves λ DOFs and +1 θ per RBE2 (e.g., ex03 with 1700 slave nodes → +3400 λ equations)
2. **KKT conditioning**: saddle-point systems are indefinite — PARDISO mtype=-2 needs iterative refinement
3. **LM zero diagonal**: the (0,0) block in the constraint rows requires η·I regularisation to prevent singular pivots
4. **Line search complexity**: KKT residual is not a valid merit function, so Armijo backtracking cannot be used

## Alternative: Master-Slave Condensation (Elimination)

For the RBE2 hinge, the constraint is linearised at each Newton step:
```
u_s = G · u_master  (where u_master = [u_m; θ])
```
This could be applied as a transformation during assembly:

```
K_condensed = G^T · K_eff · G
f_condensed = G^T · f_eff
```

n_condensed = n_dofs − 2·n_RBE2_constraints + n_extra (much smaller)

### Practical Challenges

1. **Nonlinear θ coupling**: R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]] makes G depend on θ, which changes each Newton iteration. The condensation would need to be recomputed every assembly.

2. **Sparse structure**: G is a large (n_dofs + n_extra, n_dofs − 2·n_slaves + n_extra) scatter matrix. G^T·K_eff·G requires 2 sparse matrix products per assembly — potentially more expensive than the KKT solve for large slave sets.

3. **Element-by-element incompatibility**: Master-slave condensation works naturally at the global stiffness level, not per-element. The element kernels output (8,8) element stiffnesses that are assembled into a global K. The condensation operates on the assembled K_eff, which requires a separate assembly pass or a two-level approach.

4. **Line search**: Without λ, the residual norm ‖R_mech‖ is a valid merit function. Simple Armijo backtracking would work.

### When Condensation Beats Saddle-Point

| Aspect | KKT Saddle-Point | Master-Slave Condensation |
|--------|-----------------|--------------------------|
| System size | n_dofs + n_extra + 2·n_slaves | n_dofs − 2·n_slaves + n_extra |
| Matrix structure | Indefinite (saddle-point) | SPD (positive definite) |
| PARDISO mtype | −2 (symmetric indefinite) | 2 (SPD) — no refinement needed |
| Solver time | ~O(n²·n_slaves) refinement | ~O((n−n_slaves)²) direct |
| Merit function | None (KKT residual invalid) | ‖R_mech‖ works |
| Implementation | ~200 lines in solver | Significant refactor needed |
| Reaction forces | λ directly available | Must recover via r = −K_eff·u − f_int |

### Recommendation for ex03

**Keep the KKT saddle-point** for the current problem. Reasons:

1. The PARDISO solver already handles mtype=-2 with ~0.053s per solve (from profiling) — this is only 1.2% of wall time, so optimizing the KKT system provides negligible benefit.

2. The assembly bottleneck (97.7%) is independent of the constraint formulation — condensation would not reduce the 0.84s per assembly call.

3. RBE2 reaction forces (λ) are diagnostically useful: they show the hinge moment during folding, which cannot be directly recovered without λ.

**Revisit condensation if:**
- The number of slave nodes grows to 10,000+ (n_slaves → 20,000 λ equations)
- PARDISO time exceeds 10% of wall time
- A direct solver (UMFPACK) replaces PARDISO and cannot handle indefinite systems
