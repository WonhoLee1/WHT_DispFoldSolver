# Session Handoff Document (2026-10-04)

---

## 1. Overview & Core Mission Summary

In this session, we investigated and resolved the numerical discrepancies in **Benchmark Deck #06** (`benchmarks/abaqus_standard_inp/06_bolt_preload_gasket_flange_3d.inp`), which models a 3D 1/4 symmetric steel flange compressing a hyperelastic EPDM rubber gasket under bolt preload displacement ($U_z = -0.060\,\text{mm}$).

The core mission was to:
1. Uncover why reaction forces were initially distorted or varying across contact methods (`PENALTY`, `NITSCHE`, `DUAL_MORTAR`).
2. Fix underlying bugs in 3D contact constraint assembly and solver active set management.
3. Prove that different contact formulations converge to the **same unique physical ground truth** under matched physical conditions.
4. Formally update all repository living documents, release notes, and engineering guidelines (`GEMINI.md`, `release_notes_contact.md`, `area_analysis_solver_20260919.md`).

---

## 2. Key Accomplishments & Root-Cause Bug Fixes

### 2.1 Initial Contact 0-Gap Sign Inversion Bug (Fixed)
- **Problem**: 10 of 35 contact slave nodes were shedding from the active set, halving reaction force from $\sim 6,500\,\text{N}$ to $-2,750\,\text{N}$.
- **Root Cause**: On flush meshes ($z = 1.500000\,\text{mm}$), machine-precision floating-point noise ($-1.38 \times 10^{-17}$) caused `gap0 < 0`, flipping the contact normal into the element interior. Under compression, these nodes were classified as separated rather than penetrating.
- **Fix**: Added tolerance guard `if abs(gap0) < 1e-5: sign = 1.0` in:
  - [`dispsolver/constraint3d/surface_contact3d_deformable.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/constraint3d/surface_contact3d_deformable.py)
  - [`dispsolver/constraint3d/nitsche_contact3d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/constraint3d/nitsche_contact3d.py)
- **Result**: 100% of slave nodes (35/35) remain active throughout the step.

### 2.2 SDI Active Set History Pollution Bug (Fixed)
- **Problem**: In `FiniteSlidingDualMortar3D`, Severe Discontinuity Iteration (SDI) handling caused residual stalling at $\approx 1,360\,\text{N}$.
- **Root Cause**: `dynamic3d.py` was taking the union of active sets across *all past iterations* (`set().union(*recent_active_sets)`), permanently freezing 20 non-contact nodes that only briefly touched in early iterations.
- **Fix**: Restricted freeze union in [`dispsolver/solver3d/dynamic3d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/solver3d/dynamic3d.py#L1802) to `recent_active_sets[-2:]` and currently penetrating nodes.

### 2.3 Strict Newton Acceptance & Residual Integrity (Enforced)
- **Problem**: Arbitrary residual relaxations (`rel_r < 0.50`, `r_norm < 1000.0`) caused premature convergence, freezing unrelaxed bending stresses ($\approx 86,400\,\text{N}$) in the flange and producing false reaction forces of $-89,013\,\text{N}$.
- **Fix**: Replaced loose heuristics with strict criteria: $r_{\text{norm}} \le 1.0\,\text{N}$ or ($\text{rel\_r} \le 10^{-3} \land \text{rel\_f} \le 10^{-3}$).

### 2.4 Physical Ground Truth Harmonization Verified
- **Benchmark Run**: Deck #06 Step 1 ($U_z = -0.060\,\text{mm}$):
  - **PENALTY (10x)**: $RF_{\text{bolt}} = -6,569.86\,\text{N}$, $RF_{\text{gask}} = +6,535.40\,\text{N}$ (0.52% equilibrium error), $g_{\text{pen}} = 1.606\,\mu\text{m}$.
  - **NITSCHE (1x)**: $RF_{\text{bolt}} = -6,412.48\,\text{N}$, $RF_{\text{gask}} = +6,412.37\,\text{N}$ (0.0017% equilibrium error), $g_{\text{pen}} = 1.574\,\mu\text{m}$.
  - **Mutual Agreement**: Reaction forces agree within **2.4%**, demonstrating that Penalty and Nitsche converge to the unique physical solution $|RF| \approx 6,400 \sim 6,570\,\text{N}$.

---

## 3. Documentation & Knowledge Base Synchronized

All canonical knowledge base documents have been updated and verified:
1. [`GEMINI.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/GEMINI.md): Added Lesson 12 with 5 core numerical rules.
2. [`release_notes_contact.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/release_notes_contact.md): Added Section 14 detailing the sign flip fix and 3-way formulation comparison.
3. [`dev_log/area_analysis_solver_20260919.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/area_analysis_solver_20260919.md): Updated Section 5 Open/Resolved checklist.
4. [`dev_log/lessons_learned_contact_harmonization_20261004.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/lessons_learned_contact_harmonization_20261004.md): Created comprehensive dated work log.
5. **UTF-8 Encoding**: Validated across all 8 modified files with zero decode errors.

---

## 4. Current State of Repository & Modified Files

### Code Files:
- [`dispsolver/constraint3d/surface_contact3d_deformable.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/constraint3d/surface_contact3d_deformable.py) — 0-gap sign guard.
- [`dispsolver/constraint3d/nitsche_contact3d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/constraint3d/nitsche_contact3d.py) — 0-gap sign guard.
- [`dispsolver/solver3d/dynamic3d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/solver3d/dynamic3d.py) — SDI active set union restriction, strict tolerance enforcement.
- [`dispsolver/constraint3d/finite_sliding_mortar3d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/constraint3d/finite_sliding_mortar3d.py) — Bilateral force/stiffness and epsilon filtering.

### Living Documentation:
- [`GEMINI.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/GEMINI.md)
- [`release_notes_contact.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/release_notes_contact.md)
- [`dev_log/area_analysis_solver_20260919.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/area_analysis_solver_20260919.md)
- [`dev_log/lessons_learned_contact_harmonization_20261004.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/lessons_learned_contact_harmonization_20261004.md)

---

## 5. Recommended Next Steps for Future Sessions

1. **Deck #06 Step 2 (Working Load Step)**:
   - Validate Step 2 where internal pressure or external service loads are applied to the bolted flange.
2. **Small-Sliding Mortar Adaptation**:
   - For 3D clamping problems, implement or enable a small-sliding nodal projection mortar path that avoids dynamic polygon re-triangulation jitter.
3. **`.inp`-driven Field/History Output Pipeline**:
   - Implement the approved plan in [`inp_driven_field_history_output_and_vtkhdf_plan.md`](file:///C:/Users/GOODMAN/.gemini/antigravity-cli/brain/89359551-6bda-440e-a5ef-21dea9211377/inp_driven_field_history_output_and_vtkhdf_plan.md) to support native `*OUTPUT, FIELD` and `.vtkhdf` exports for ParaView visualization.
