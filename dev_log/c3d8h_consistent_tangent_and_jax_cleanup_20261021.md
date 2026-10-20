# C3D8H consistent tangent, C3D8R JAX removal, JIT warmup (2026-10-21)

All of this is on master inside agy's commit 6456909 (it swept up the uncommitted shared tree); this note is the record
that the commit message lacks.

## Four failing tests, four causes
| test | cause | change |
|---|---|---|
| `test_3d_c3d8r` x2 | `c3d8r_jax.py` had no `alpha_hg`, no `compute_orthogonal_hourglass_vectors`, unimplemented abstract methods; used only by `__init__` and tests | deleted; tests moved to the Numba kernel (hourglass orthogonality through the kernel, STIFFNESS vs ENHANCED differ only by hourglass modes) |
| `test_section_controls::..distortion_control_barrier` | kernel returns error 1 for `J<=0.05` since 1221807; the test used J=-0.895 | test uses J=0.052 (barrier band) plus a test for the error floor |
| `test_c3d8_hybrid::..heterogeneous_assembly` | C3D8H tangent was `R K_loc R^T` (frozen frame); `dR/du` dropped. 98% off along a rocking mode of a soft layer, line search stuck, 35 iterations. Exposed by 0c5e42c's relative acceptance | consistent tangent in `c3d8_hybrid_numba.py` (`dR/du` by central difference of the frame, rest analytic); 35 -> 4 iterations |

## Measurements
* Tangent vs finite difference, rocking direction, sheared cube: 98% (old) -> 0.4% at gamma=0.02; remaining error is proportional to the amplitude
  (zero state 3e-8) = the Hencky -> Green-Lagrange interface term, known.
* Random directions never showed it (<=2.4e-2 old and new); `tests/test_c3d8h_consistent_tangent.py` fails 4/5 on the old kernel.
* Cache version bumped v6 -> v7 (callers in `cr_wrapper_3d.py` are cache=True, AGENTS.md 4.19).

## Not done
* `c3d8_corotational_numba.py` (C3D8_CR, PET) and `cr_wrapper_3d.py` keep the frozen-frame tangent (random-direction error 4%; stiff layers are insensitive).
* 7 JAX 3D classes remain (EAS, FBAR, C3D4, C3D4_ANP, C3D10, C3D10M, C3D6); used by ex14, verification/, tests.
* `test_element_nlgeom_contract[CPE4R]` fails alone ("NLGEOM=NO f(10u)/f(u)=10.0135") at dca9fff and d7a319c; it passed in the last full run. Order dependence not investigated.
* `jit_manifest.json` not yet recorded: `python -m dispsolver.warmup --record`, then `--record-run -- <script>` for the real analyses.

## JIT warmup
`dispsolver/warmup.py`: module discovery of Numba dispatchers (333 found), compile-without-running from the manifest, `--status`, `--record`,
`--record-run`, `--if-needed`, `--clean`, `--jobs`. `_jit_cache.py` stamps a warm state (CPU, Python, Numba, NumPy, manifest hash) and prints a one-time hint when cold.
Cold first step of a two-element 3D solve: 128 s; warm: 0.7 s.
