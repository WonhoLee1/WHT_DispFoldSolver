# Tier 2 — root-causing the Numba/JAX backend drift, and the B4 FD-step tail (2026-09-12)

Closes AC2 and AC3 of `.omc/plans/2d_element_open_items_20260912.md`.
Context: AGENTS.md §4.16's "Open, not fixed" list; §4.14/§4.15 for the
defect class.

Three C10 entries and one C11 entry stood in
`tests/element_contract/test_contract.py::_XFAIL` as *characterized*
defects. All four are now fixed at the root and the characterizations
promoted to real passing tests.

---

## 1. The three C10 backend-drift defects

All three were the same sentence, once more: **"which deformation
gradient does the B-operator differentiate?"** — the F4/F6/B1/B2/B3
class. Each had a `dynamic.py` dispatch comment asserting an agreement
number that had never been measured on that element; C10 measured the
real one.

| Element | Claimed in comment | C10 measured | Root cause | After |
|---|---|---|---|---|
| `Q4_COROTATIONAL_SRI` (production PET/GLASS) | "force to 1e-13" | 1.54e-03 | Two **different co-rotational frames**. JAX (`q4_corotational_jax.compute_element_rotation`, shared by every JAX co-rotational kernel) takes the ξ-edge pair `v12+v43` alone; the Numba copy averaged both edge rotations through their vector mean. | **4.67e-14** |
| `CPE4I` (production PSA) | "force error <= 2e-5" | 1.61e-03 | Numba mirror **never ported onto finding B2**: it built `B_L` from the COMPATIBLE gradient (dropping the α-dependent half of `dE_inc/du`) and contracted `f_a` against the un-pushed `S_v`. | **1.70e-15** |
| `Q4_VISCO_SIMO` | — | 1.75e-03 | `B_L` built from `Fbar` instead of the real `F`. In the F-bar method the modified gradient enters the **constitutive call only**; the virtual strain is that of the real motion. | **2.00e-15** |

Two of these are worth keeping as method notes:

**The SRI frame difference is invisible on every other element in the
library.** The co-rotational frame cancels exactly out of a
full-integration Green–Lagrange element (`E = (F_l^T F_l − I)/2` with
`F_l = R^T F` is `R`-independent — finding F1's "bitwise TL" result), so
either frame choice gives the same answer there. It is visible on SRI
*only* because SRI samples the shear off-diagonal of `dU/dX` in fixed
Cartesian components, which is not frame-invariant
(`q4_sri_jax._F_sri`'s warning). Measured on C10's tilted+distorted
probe: `theta_jax = -0.014852°` vs `theta_numba = -0.681885°`. Feeding
the two-edge frame's local displacement into the JAX SRI core reproduces
the Numba kernel to **1.09e-15** — i.e. the frame was the whole of it.
Everything else in the two kernels (quadrature, SRI gradient, `B_L`, J2
return map) already agreed to 2e-16.

**`B_L(Fbar)` is not the gradient of any potential.** Building `B_L` from
the modified gradient differentiates a quantity while holding its own
`sqrt(J0/J)` factor fixed. That is neither the standard F-bar residual
nor a consistent variation. The resulting element tangent being
unsymmetric is a *documented property* of correctly-implemented F-bar
(de Souza Neto et al. 1996), not a defect to be engineered away.
Rebuilding the JAX residual with `B_L(Fbar)` reproduces the old Numba
kernel to 1.29e-15, which is how the term was isolated.

## 2. `Q4_VISCO_SIMO` was silently Total Lagrangian — a live §4.8-class defect

Found while fixing the above, and **the most consequential item in this
tier**. `q4_visco_hybrid_simo_numba` had no `F_n` parameter at all and
was handed `elem_coords` with the TOTAL displacement — while its
dispatch branch in `dynamic.py` is **not** gated on
`enable_new_numba_elements` and `elem_jit` defaults to `"numba"`, so it
is reachable by default.

With `nlgeom=True` it therefore ran Total Lagrangian while
`_use_ul_for(pid)` returned `True`, `element_large_deformation_report()`
printed "UL (rotated reference)", and the JAX branch three lines below
ran genuine UL. AGENTS.md §4.8's silent-fallthrough class, on a
reachable-by-default path.

Consequence worth recording: **`verification/abaqus_benchmarks/cook_membrane.py`'s
`Q4_VISCO_SIMO` `numba` rows were never comparable with its `jax` rows** —
different formulations, not a backend check. This is the same shape as
AGENTS.md §4.16's finding that `nlgeo_cantilever.py`'s `numba` rows do
not run Numba at all.

Fixed by giving the kernel a full UL path mirroring the three branches
above: `F_n` per Gauss point, `F = F_inc @ F_n[gp]`, the F4
work-conjugacy push-forward, `F0 = F0i @ mean(F_n)` for the F-bar
dilatation, and the JAX kernel's exact `J_ratio` clamps. F5 discipline
observed: `_ul_F_n` is **not** written in the assembly branch (it runs
every Newton iteration); the commit-time resync stays the sole
authoritative write.

## 3. B4's FD step — the remaining sibling, and a contract-masking lesson

`q4_visco_hybrid_up_numba.py` (CPE4H) carried B4's fixed **absolute**
`h = 1e-6`. C11 measured the `h/L` signature exactly: **2.61e-06 /
1.31e-06 / 6.54e-07 / 3.27e-07** as the element grows ×1/×2/×4/×8 —
error exactly inverse in element size, ratio **8.0** across a factor-8
sweep. Replaced with the per-column Dennis & Schnabel (1983) §5.4 step
`h_j = sqrt(eps) * max(|u_j|, L_elem)`; now **4.11e-08 flat, ratio 1.00**.
`q4_visco_eas_numba` and `q4_visco_hybrid_reduced_numba` fixed in the
same pass (→ 1.01 and by-inspection respectively).

**Method note worth keeping — a contract can be masked by another
contract's failure.** `CPE4I`'s C11 row was *flat* before the C10 fix and
grew the ratio-8.0 signature (6.63e-06 … 8.28e-07) the moment its force
agreed with JAX. C11 measures `|K_numba − K_jax|`, so a large
force-formulation disagreement was swamping the FD term and making a
real B4 defect look absent. **A flat C11 row is only evidence once C10
passes.**

Note on `alpha` sweeps in `q4_visco_eas_numba`: `alpha` is
*dimensionless* (added straight into the deformation gradient), so it has
no mesh-size dependence to correct — only the same sqrt(eps)-relative
conditioning, via `max(|alpha_j|, 1)`.

---

## 4. Additional finding — NOT fixed here, separate follow-up

**`det(F_n)` sign handling diverges between the two lowerings' guards.**
The F4 push-forward needs `det(F_n)`. The two backends guard it
differently:

- `q4_visco_simo_fs_jax.py:269` — `jnp.maximum(jnp.abs(det), 1e-30)`,
  which **discards the sign**.
- `q4_visco_hybrid_simo_numba.py` (and the other Numba kernels) —
  `if abs(detFn) < 1e-30: detFn = 1e-30`, which **preserves** it.

For `det(F_n) < 0` the two give push-forwards of opposite sign. C10
passes at 2e-15 because its probes never invert, so this is invisible to
the contract suite as it stands — which is precisely the failure mode the
comment directly above the guard warns about ("divergent guards are how
two lowerings drift apart under exactly the distortion the guard exists
for").

**Assessment: the Numba side is right and the JAX side is the defect.**
An inverted `F_n` can be a genuine (if pathological) state; throwing away
its sign silently corrupts the push-forward for exactly that state rather
than failing on it. Deliberately **not** folded into this commit — it
changes a production JAX kernel (`q4_visco_simo_fs_jax` is the PSA
fallback and Cook's-membrane reference) and deserves its own A/B rather
than riding along with a backend-agreement fix. A C10 probe driven to
`det(F_n) < 0` would pin it.

---

## Verification

- `tests/element_contract/` — 92 passed, 15 skipped, 17 xfailed, **0
  failed, 0 xpass** (614 s). The four promoted entries pass as real
  tests; no remaining `_XFAIL` entry turned green unnoticed.
- Full `pytest tests/`, `python -m verification.run_all`, and the
  `examples/ex12_abaqus_inp_plate_fold.py` 90°/side end-to-end re-run:
  see the commit message for the numbers.

**Scope note.** All four gated Numba branches
(`enable_new_numba_elements`, which nothing sets) are dead on the
production `ex12` path, and `ex12` runs PET/GLASS and PSA through JAX.
The `Q4_VISCO_SIMO` branch of §2 is the one exception — ungated and
reachable by default — but `ex12` does not select that element. So `ex12`
is expected to reproduce **bit-identically**, and that prediction is
itself part of the verification below.
