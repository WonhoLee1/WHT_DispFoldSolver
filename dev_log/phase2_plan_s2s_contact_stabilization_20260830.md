# Phase2 Plan — S2S Contact, Numba/GPU, AMG, EAS-7 (deferred)

Branch: feat/solver-element-abaqus-surpass-20260830
Date: 2026-08-30
Scope: Phase0 (P0) + Phase1 (P1) committed. Phase2 heavy features deferred to next PR after verification.

## Deferred items
- S2S mortar tie with re-projection and Augmented Lagrangian outer loop wiring in DynamicSolver._solve_step_impl outer iteration
- Numba full coverage for q4_visco_simo_fs and q4_eas (currently JAX only for some paths)
- AMG/iterative fallback for large (>20k DOF) PARDISO fallback
- EAS-7 / ANS for thin PET Poisson and transverse shear

## Rationale
Phase0/1 already achieve grading, Q4_EAS, lnJ volumetric, 2-term Prony, q_avg stabil, Generalized-alpha, equilibration cache. These are low-risk and smoke-tested (6/6). Phase2 requires contact patch test, non-matching mesh verification, and performance benchmarking under `verification/RULES.md` mandatory contact entry — not yet in rules. Implement in gated milestones M2 after verification 12 + ex12 90° full run confirms Phase1 not regressive.

## Next PR checklist
- Add `verification/benchmarks.contact_patch` for S2S
- Add `dispsolver/contact/` to RULES.md mandatory list
- Run `python -m verification.run_all` full (not smoke) + `DispFoldApp --mode build --max-steps 101` for U-shape and `check_interlayer_shear.py --cached` for slip 81µm regression
