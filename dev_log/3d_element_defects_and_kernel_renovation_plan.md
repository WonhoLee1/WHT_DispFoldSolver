# 3D element defects — verification of claimed fixes + renovation plan (2026-09-12)

Handoff document for the session that owns `dispsolver/element3d/`,
`dispsolver/solver3d/`, `dynamic3d.py`, `material3d/`, `mesh3d/`. Written
by the 2D session after independently re-verifying (by reading the actual
code, not by trusting the report) a set of fixes that session reported
applying in response to `dev_log/3d_element_defect_audit_20260912.md`
(the read-only audit — numeric root-cause findings in §0-4, reviewer
findings in §5, Abaqus-theory verification in §6).

**Method note, since it matters here**: every verdict below comes from
directly reading the current file content (`grep`/`Read`), not from
re-reading the fix report. This is the same discipline this whole
project's 2D session has used throughout (AGENTS.md §4.9: a report that a
defect was fixed is not itself evidence — read the code).

---

## 1. Verification of the 5 claimed fixes

| # | Claim | Verified? | Evidence |
|---|---|---|---|
| 1 | Silent fallback removed from `assemble_system` | **✅ Confirmed fixed** | `dynamic3d.py:367-371`: `except Exception as e: ...traceback.print_exc()... print("CRITICAL: Numba assembly fastpath failed! Aborting...")... raise e`. Line 373: the linear-geometry branch now `raise NotImplementedError(...)` instead of running a Python fallback loop. No silent formulation swap remains here. |
| 2 | Silent dense-solve fallback removed from `pardiso_spsolve` sites | **✅ Confirmed fixed** | `dynamic3d.py:440` and `:632`: both now `raise RuntimeError("Pardiso linear solver failed (matrix is likely singular). Check boundary conditions or rigid body modes.") from e`. No `toarray()` / `1e-4*eye` dense fallback remains. |
| 3 | Element classification made strict, unmapped types blocked | **⚠️ Half-fixed — the misrouting bug is closed, but this is a stopgap, not the underlying fix** | `dynamic3d.py:156-170`: classification is now a strict `elif et in [...]` chain (`C3D8I`/`C3D8_EAS`→0, `C3D8_CR`/`C3D8_COROTATIONAL`/`C3D8_FBAR_CR`→1, `C3D8H`/`C3D8_HYBRID`/`HYBRID`→2, `C3D8`/`C3D8_FBAR`→3), and anything else raises `NotImplementedError`. **The error message itself admits the gap**: `"Element type '{et}' is not yet supported in the Numba fast-path. Note: C3D4, C3D10, and C3D10M implementations are currently incomplete placeholders."` The silent misroute-to-wrong-8-node-kernel bug (§0 of the audit) is genuinely closed — that's real, keep it — but C3D4/C3D10/C3D10M are not wired into the fast path at all. Loud failure instead of silent wrong physics is a real improvement; it is not tet-element support. |
| 4 | `C3D4_ANP` → `C3D4`, `C3D10M` → `C3D10` renamed throughout, honestly | **❌ Not found in the code** | `grep -rn "C3D4_ANP\|C3D10M" dispsolver/ verification/` (2026-09-12, after the fix report) still returns: `dispsolver/element3d/c3d4_anp_numba.py` (filename, docstring "C3D4_ANP Tet4 elements", function docstrings unchanged), `dispsolver/element3d/c3d10m_numba.py` (same), `dispsolver/mesh3d/mesh3d.py:28` (`elem_type` comment still lists `"C3D10M"`), `verification/element_benchmark_comparison.py:51` (`"C3D10M"` still used as a live dispatch key string), `verification/abaqus_3d_benchmark_suite.py:4`, `verification/generate_rollup_figure.py:103/113`. If a rename happened, it is not reflected in the current tree — re-check whether the change was made and reverted, made in a different branch/stash, or not actually executed. |
| 5 | Critical-defect warning attached to `c3d8_eas_tl_numba.py` | **✅ Confirmed attached — but it is a warning label, not a fix of the underlying defect** | File header now has a boxed `🚨 CRITICAL THEORETICAL DEFECT WARNING (2026-09-12 Audit) 🚨` block citing TG §3.2.5 correctly and saying "Do NOT use for production runs!" This documents the audit's §6 finding (TL-accumulated incompatible modes) precisely. **It does not touch the separate, more fundamental §2.1 finding**: the enhanced-strain parameters are computed and condensed but never fed back into stress/`f_int` at all (JAX twin bit-identical to plain compatible C3D8 at 2.8e-17) — the element currently does nothing an EAS element is for, independent of TL-vs-UL. A warning comment is a legitimate stopgap (prevents accidental production use) but is not the fix. |

**Bottom line**: 2 of 5 are real fixes. 1 of 5 (classification) genuinely closes a live bug but explicitly leaves the underlying gap (tet support) as a documented placeholder — that's honest, just incomplete. 1 of 5 (the rename) does not appear to have happened. 1 of 5 (the EAS warning) is a label, correctly worded, on a defect that is still unfixed underneath.

---

## 2. Full outstanding-defect list (from the audit, cross-referenced against what's actually fixed above)

Carried forward from `dev_log/3d_element_defect_audit_20260912.md`, with status updated:

| Defect | Audit ref | Status after this round |
|---|---|---|
| `assemble_system` silent formulation swap on exception | §5.1 (HIGH) | **Fixed** (see #1 above) |
| `pardiso_spsolve` silent dense-singular fallback (×2) | §5.2 (MEDIUM) | **Fixed** (see #2 above) |
| Element misrouted to wrong kernel on unmapped type | §2.6 / §5.3 | **Fixed** — now blocked, not misrouted (see #3 above) |
| C3D4/C3D10/C3D10M have no real fast-path implementation | §7 (not deeply probed — dead code) | **Still open** — now at least fails loudly instead of silently misrouting. This is the actual remaining work if these element types are needed. |
| `C3D8_EAS`/C3D8I: enhanced-strain condensation computed but never fed into stress/f_int (f_int bit-identical to plain C3D8, 2.8e-17) | §2.1 | **Still open** — only a warning comment added, the mechanism itself is untouched |
| `C3D8_EAS`'s condensed tangent 10.4% off its own FD Jacobian | §2.1 | **Still open** |
| `C3D8_EAS` TL-accumulated-total-F is the Abaqus TG §3.2.5 "fatal flaw" formulation | §6 | **Documented via warning header** — reformulation to UL/incremental-ΔF not yet done |
| 3D J2 plasticity material tangent never updates post-yield (53.5% error, shear ~10x too stiff) | §2.2 | **Still open**, not addressed by any of the 5 claims |
| Corotational C3D8 tangent omits ∂R/∂u_elem entirely (99.5% error vs FD Jacobian; residual itself is correct/objective) | §2.3 | **Still open** |
| `C3D8H` hybrid element crashes on every call (undefined `gp_idx`) and, once patched for testing, bypasses `material_dispatch_3d` entirely (hardcodes linear elasticity) | §2.4 | **Still open** — not addressed |
| F-bar C3D8 omits its own volumetric-consistency tangent term (20.5% error; residual objective to ~1e-14) | §2.5 | **Still open** |
| A *third*, previously-unnoticed element-classification table (in `_initialize_elements`'s slow path) maps both `"C3D8H"` and `"C3D8_FBAR"` to the same `Hexa8FbarElement` — formulation can differ depending on which code path (fast/slow) happens to run | §5.3 | **Status of the slow-path table specifically: not directly re-checked this round — verify separately, the fast-path table fix (#3 above) does not by itself guarantee the slow path was touched.** |
| `C3D4_ANP` claims Bonet & Burton (1998) average-nodal-pressure, implements none of it (plain 1-point linear tet) | §6 | **Still open** — claimed renamed, not found renamed (see #4 above); the underlying "name doesn't match implementation" problem is therefore also still open either way (rename OR implement, neither has landed) |
| `C3D10M` claims Abaqus's actual modified-tet-with-hourglass-control element, implements a plain unmodified quadratic C3D10 | §6 | **Still open**, same as above |
| `C3D8_FBAR`, `C3D8_CR`/`C3D8_COROTATIONAL` are not real Abaqus element names (Abaqus has no such names — F-bar is a technique inside real elements, not a name; there is no corotational continuum element in Abaqus) | §6 | **Not addressed by any of the 5 claims** — same "invented name wearing a real-looking label" pattern the 2D session already resolved for its own `CPE4S`/SRI case (see `dispsolver/element/q4_sri_jax.py`'s honest-naming docstring for the pattern to follow) |
| `MAT_VISCOELASTIC_PRONY` declared but has no dispatch branch (`numba_materials.py`, returns `error_flag=2`) | interface-alignment doc §, re-confirmed in audit §5.4 | **Still open** |

---

## 3. Renovation plan (priority order, for the 3D-owning session to execute)

This mirrors the 2D session's own approach to the equivalent situation
(AGENTS.md §4.14-§4.16): root-cause with a real measurement before
touching code, fix the mechanism rather than the symptom, verify with a
number, and don't claim a fix without re-reading the code afterward.

### Tier A — close what's still only half-done from this round
1. **Actually execute the `C3D4_ANP`→`C3D4`, `C3D10M`→`C3D10` rename** (or, if on reflection the real techniques should be implemented instead of renamed — see Tier C — decide explicitly and record the decision, don't leave the name claiming something the code doesn't do). If renaming: every file/function docstring, every `elem_type` string compared anywhere (`mesh3d.py`, `dynamic3d.py`'s classification table, every `verification/*.py` script), not just one location.
2. **Verify the *slow-path* element-classification table separately** — confirm whether `"C3D8H"`/`"C3D8_FBAR"` colliding into `Hexa8FbarElement` (§5.3 of the audit) was touched by this round's classification fix or is still live. If still live, the same Tier-A rigor as #1 in `dynamic3d.py` applies there too.
3. **`C3D8_FBAR` / `C3D8_CR` / `C3D8_COROTATIONAL`**: these are not real Abaqus names (§6). Decide now, same as the 2D precedent: keep as honestly-named in-house elements (rename away from anything that reads as an Abaqus code) or retire. Don't leave them presenting as Abaqus elements.

### Tier B — the two "name is real but content is missing" defects that are load-bearing for correctness
4. **`C3D8_EAS`/C3D8I**: this needs two things, not one — (a) actually feed the condensed enhanced-strain contribution back into stress/`f_int` (currently a no-op), and (b) per the TG §3.2.5 finding already in the warning header, do it in an incremental/ΔF frame rebuilt each increment, not accumulated into the total F. Doing (a) without (b) would just resurrect the exact "fatal flaw" formulation Abaqus's own theory guide warns against — the 2D session hit this precise trap (see `dev_log/eas_frame_consistency_benchmarks_20260910.md` and `AGENTS.md` §4.15/§4.16 for the analogous 2D fix, and the config-ledger pattern in `dispsolver/element/kinematics/frame.py` if a shared architecture is wanted — see also `dev_log/plan_2d_3d_interface_alignment_20260912.md` for what that convergence could look like).
5. **`C3D4_ANP` and `C3D10M`**: once Tier A's naming decision is made, if the decision is "implement the real technique" rather than "rename honestly": `C3D4_ANP` needs Bonet & Burton (1998)'s actual average-nodal-pressure algorithm (nodal volume-weighted pressure averaging, not a plain 1-point tet); `C3D10M` needs Abaqus's actual modified-tetrahedron hourglass control, not a plain 4-point quadratic C3D10 relabeled.

### Tier C — the remaining quantified defects, none touched by this round
6. 3D J2 plasticity tangent (post-yield update missing, 53.5% error) — root-cause is in `material3d/numba_materials.py`'s tangent computation, same defect family as the 2D session's own J2-tangent bug (`dispsolver/material/plastic.py`'s `_VOIGT_PAIRS` index-swap, AGENTS.md §4.16) — check for a similarly simple index/derivative-update bug before assuming a larger rewrite is needed.
7. Corotational C3D8 tangent (`∂R/∂u_elem` omitted, 99.5% error vs FD Jacobian) — residual is correct, so this only costs Newton convergence rate; still worth fixing since large-rotation folding is exactly where slow convergence bites hardest.
8. `C3D8H` — fix the `gp_idx` crash and route it through `material_dispatch_3d` for real (currently hardcodes linear elasticity once patched).
9. F-bar C3D8 volumetric-consistency tangent term (20.5% error, residual clean) — lowest severity of the open items, but still a real tangent-correctness gap.
10. `MAT_VISCOELASTIC_PRONY` — either implement its dispatch branch or remove the declaration that claims it exists.

### Verification standard to hold every fix to
Per this session's own repeated lesson (AGENTS.md §4.9, §4.14-§4.16): a
converged answer or a passing test is not proof a fix is real. For each
item above, verify with an actual number — a finite-difference Jacobian
check on a rotated/distorted reference, a before/after force or tangent
comparison, or (for the naming items) a `grep` that returns nothing
afterward — the same standard this document itself was written against.

---

## 4. Cross-references
- `dev_log/3d_element_defect_audit_20260912.md` — the original numeric audit (§0-4), reviewer pass (§5), Abaqus theory verification (§6).
- `dev_log/plan_2d_3d_interface_alignment_20260912.md` — architecture comparison and future convergence sketch between the 2D config-ledger pattern and 3D's element/material interfaces.
- `AGENTS.md` §4.14-§4.16 — the 2D session's own analogous defect class (frame/config mixing, partial fixes that break an assumed symmetry, absolute FD-step scaling) and the verification methodology that found it, offered here as prior art since several of the open 3D items are the same defect class in a different codebase region.
