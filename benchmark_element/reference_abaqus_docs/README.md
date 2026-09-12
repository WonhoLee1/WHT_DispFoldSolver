# reference_abaqus_docs/ — provenance

Plain-text extracts of real Abaqus documentation (Theory Guide, Verification
Manual, Benchmarks Manual), fetched 2026-09-13 from the user's local Abaqus
documentation server (`http://desktop-whl:4040/English/<path>`) at the
user's direction, using the bare-path URL form (the `?show=` query-param
form serves a JavaScript SPA shell with no usable content — use
`/English/<SIMACAE...RefMap path>` directly, e.g.
`http://desktop-whl:4040/English/SIMACAEVERRefMap/simaver-c-3dpatch.htm`).

HTML tags stripped with a plain regex (`re.sub(r'<[^>]+>', ' ', html)`),
not rendered/interpreted — read as reference text, not as authoritative
markup. Equations lost their LaTeX/MathML formatting in the process
(rendered as flattened Unicode-ish text); where a formula matters, verify
against the live doc server or the official manual rather than trusting
this extract's exact symbol placement.

| File | Source page | Why it's here |
|---|---|---|
| `3dpatch.txt` | Verification Manual, "Patch test for three-dimensional solid elements" | The REAL Abaqus patch test every C3D* element is verified against: E=1e6, nu=0.25, exact affine displacement BC, exact reference stress/strain for 3 steps (linear, NLGEOM+pressure, post-load perturbation). `benchmark_3d_mechanics.py::run_abaqus_official_patch_test` reproduces Step 1 of this exactly — see that function's docstring for the precise numbers and what is/isn't checked. |
| `elempatcheseig.txt` | Verification Manual, "Eigenvalue extraction for unconstrained patches of elements" | Confirms the same patch geometry is reused for rigid-body-mode counting; cross-referenced by `benchmark_3d_elements.py`'s rank test. |
| `incompatible.txt` | Theory Guide, "Continuum elements with incompatible modes" | C3D8I/C3D8IH formulation. Contains the same "fatal flaw... once elements become distorted due to deformation" language AGENTS.md §4.14/§4.16 already cites (TG §3.2.5) — corroborates, from the primary source, why an additive-F incompatible-mode element degrades under large distortion. |
| `hybridincompress.txt` | Theory Guide, "Hybrid incompressible solid element formulation" | C3D8H formulation: independent pressure field via Lagrange multiplier, `rho=1e-9` regularization constant. Relevant to why `C3D8H` measures `VOL-LOCKED` in our benchmark if this mechanism isn't actually wired into the assembled tangent. |
| `solidform.txt`, `solids.txt` | Theory Guide, general solid element formulation | Hughes-Winget incremental-objectivity algorithm (ΔR via ΔW, central-difference strain) Abaqus uses for large-rotation increments; explicit warning that kinematic-hardening plasticity is inaccurate above 20-30% strain. Background reference for any future large-strain plasticity case in this suite. |
| `solidisoquadhex.txt` | Theory Guide, isoparametric quadrilaterals/hexahedra | Confirms Flanagan-Belytschko "uniform strain formulation" (average strain over the element volume, not Gauss-point strain) is what makes `C3D8R` pass the patch test and stay accurate when skewed -- directly explains why `C3D8R` is our only fully-clean row in `dev_log/benchmark_3d_mechanics_20260913.md`. Also warns hourglass control "can break down in strongly nonlinear problems" -- caveat for any future large-strain `C3D8R` case. |
| `tritetwedge.txt` | Theory Guide, triangles/tetrahedra/wedges | First-order tet/wedge (`C3D4`/`C3D6`) are single-integration-point constant-stress elements with no reduced-vs-full-integration distinction -- explains why they pass patch tests trivially but lock in bending (no mechanism available to *not* lock, unlike the hex family). |
| `basicelemover.txt` | Verification Manual overview | Context for how Abaqus structures its own element verification suite -- the model `benchmark_element/` follows. |
| `singleelemeig.txt` | Verification Manual, "Eigenvalue extraction for single unconstrained elements" (Table 10) | **Actionable, sourced numbers**: exact rigid-body-mode count + first-nonzero eigenvalue per element -- `C3D8`=6 modes/4.186e7, `C3D8H`=6/4.186e7, `C3D8I`=6/4.186e7, `C3D8R`=6/1.184e6, `C3D4`=6/3.623e9, `C3D6`=7(6 rigid)/3.846e8, `C3D10`=6/4.500e9, `C3D10M`=6/7.486e7. The mode COUNT is geometry/material-independent and directly assertable against `benchmark_3d_elements.py`'s existing rank test; the eigenvalue itself needs Abaqus's exact single-element mesh/material (not in this extract, only in the un-fetched `.inp`) so can't be matched exactly yet. |
| `uelmat.txt` | Subroutines Reference, UEL/UMAT interface | Not directly used yet; fetched for future reference if a user-element-style interface is ever considered. |
| `bmk_thickcompcyl.txt` | Benchmarks Manual, thick composite cylinder | **Real implementable benchmark, not yet built.** Full closed-form target: Ri=60mm, Ro=140mm, 8 orthotropic layers `[0,90]4`, two exact material sets (E1/E2/E3, G12/G23/G13, nu12/nu23/nu13), P=50MPa internal pressure, plane strain. Exact normalized reference values given: ū_r(inside)=1.4410, ū_r(outside)=0.1476, σ̄θ(inside)=5.7060, σ̄θ(outside)=0.0103, plus per-element-type error tables (C3D8, C3D20R at several refinements). Requires this codebase's 3D path to support layered/orthotropic composite materials -- **confirmed still missing 2026-09-13** (checked `dispsolver/material3d/numba_materials.py`'s dispatch, isotropic-only). See `dev_log/static_analysis_benchmark_design_20260913.md` for the full classification pass this went through alongside 12 other static-analysis benchmarks. |
| `bmk_neckingtensilebar.txt` | Benchmarks Manual, necking of a tensile bar | Background-only, not implementable as a pass/fail benchmark: only gives qualitative figure descriptions and Gurson (porous-plasticity) material parameters, no closed-form target displacement/stress. Relevant only if `dispsolver` ever adds a Gurson/porous-plasticity material. Reconfirmed 2026-09-13, see `dev_log/static_analysis_benchmark_design_20260913.md`. |
| `bmk_concreteslump.txt` | Benchmarks Manual, concrete slump test | Background-only -- needs a concrete/Drucker-Prager-cap material this codebase doesn't have, and even then only figure/qualitative results exist (no closed-form number). Reconfirmed 2026-09-13. |
| `bmk_rad_stretch.txt` | Benchmarks Manual, radial stretching of a cylinder | **The one READY candidate found across all 13 static-analysis pages (2026-09-13 classification pass, `dev_log/static_analysis_benchmark_design_20260913.md`).** Uses `CAX4R`/`CAX4`/`CAX8`/`CCL12`/`CCL24` (axisymmetric/cylindrical, not this folder's C3D8/C3D4 family) but the physics -- a hollow cylinder under a uniform outer-radius displacement -- is dimension-independent and gives a genuine closed-form, radius-dependent stress field (σ_rr, σ_θθ given symbolically with λ, μ and three constants C1/C2/C3, all in the doc). No contact, no composite material, no hyperelasticity needed -- isotropic linear elastic, E=2e11, nu=0.3. Design doc above has the full BC scheme and a proposed 3D-solid-wedge equivalent model plus function signatures; not yet implemented. |
| `bmk_rubberdisk.txt` | Benchmarks Manual, pressurized rubber disc | Uses real 3D solids in places (`C3D8IH`, `C3D10MH`) but needs genuine Mooney-Rivlin/Ogden hyperelasticity -- this codebase's 3D "NeoHookean" material was directly verified 2026-09-13 to be bit-identical to linear elasticity (not real finite-strain hyperelasticity at all, see `dev_log/solve_step_false_convergence_20260913.md`). Even with real hyperelasticity, results here are figure-only (pressure-vs-displacement curve vs. Oden 1972), no clean tabulated number to check against. |
| `bmk_elasticsheet.txt` | Benchmarks Manual, elastic sheet with hole | `CPS8R` -- **2D plane stress**, not a 3D solid problem at all; belongs to the 2D suite's territory if pursued, and needs real hyperelasticity there too (Mooney-Rivlin/Biderman/Ogden/Marlow, none implemented). Figure-only results (load vs. nominal strain). |
| `bmk_hertzcontact.txt` | Benchmarks Manual, Hertz contact problem | Real closed-form Hertz theory (Timoshenko & Goodier 1951) exists, but only meaningful through an actual contact solve -- `dispsolver/constraint3d/` has no contact/rigid-surface module (only `rbe3_distributing3d.py`, `surface_tie3d.py`), confirmed 2026-09-13. |
| `bmk_pipecrushing.txt` | Benchmarks Manual, crushing of a pipe | `CPE8H`/`CPE6H`/`CPE4I` (2D plane strain) + `SC8R` shell + rigid contact -- 2D plane-strain physics (this folder's territory is 3D solids only) and needs contact regardless. Experimental-curve reference only (Peech et al. 1977), not a closed-form number. |
| `bmk_beamgap.txt` | Benchmarks Manual, beam/gap example | `B23` beam elements + switching gap/contact conditions -- has a genuinely exact tabulated reference (Table 1, per-increment gap-force values) but no solid-element equivalent; beam+contact-switching physics is outside this folder's 3D-solid scope. |
| `bmk_anisoplate.txt` | Benchmarks Manual, anisotropic layered plate | `S3R`/`SC6R`/`S9R5`/`STRI65` shells only, no solid element used at all. Has exact tabulated reference (Table 1, disp/moment vs. Spilker et al. 1976) but is shell-only physics -- not applicable to `benchmark_element/`'s 3D solid family. |
| `bmk_compositeshells.txt` | Benchmarks Manual, composite shells in cylindrical bending | Mostly `S8R`/`S4R`/`SC8R` shells; ONE `C3D20R` composite-solid variant exists but needs layered/orthotropic material (not implemented, see `bmk_thickcompcyl.txt` row) and only has figure-based results (Pagano 1969 elasticity solution is plotted, not tabulated) for that solid case. |
| `bmk_unifcollapsepipe.txt` | Benchmarks Manual, uniform collapse of straight/curved pipes | `ELBOW31B` pipe + `S8R5` shell, no solid elements. Figure-only (moment-curvature curves; Brazier 1927 analytical is a curve, not a single number). Not applicable, and the collapse itself is inherently unstable (would need Riks even if it were solid-element-based). |
| `bmk_roofsnapthrough.txt` | Benchmarks Manual, snap-through of a shallow cylindrical roof | `S4R5`/`S4R`/`S3R`/`SC6R`/`SC8R` shells only. Classic Riks/arc-length demonstration -- `dispsolver/solver3d/dynamic3d.py` has no Riks/arc-length capability (confirmed 2026-09-13, `grep` for it returns nothing), and it's shell-only physics regardless. Figure-only results (load-displacement curves vs. several published solutions, no single number). |
| `bmk_caxasaxa2pointbending.txt` (here), plus `cantilevercaxasaxa`/`torsholcyl` (in `scratch/abaqus_bmk_docs/`) | Benchmarks Manual | **Do NOT use as a reference for C3D8/C3D4 results.** These describe Abaqus's CAXA/SAXA/CGAX axisymmetric-with-twist element families, which this codebase does not implement -- see `dev_log/benchmark_element_folder_setup_20260913.md`'s note on the prior, unsourced `abaqus_official_benchmarks_report_20260910.md`, which made exactly this element-family mismatch mistake. |
| `ctc_contactoverview.txt` | Interaction Guide, "About Contact Interactions" | Top-level map of contact pairs vs. general contact vs. contact elements; source for `dev_log/3d_contact_implementation_design_20260913.md`'s scope recommendation (contact pairs, not general contact, for this codebase). |
| `ctc_contactpair_std.txt`, `ctc_generalcontact_std.txt` | Interaction Guide, "About Contact Pairs/General Contact in Abaqus/Standard" | Full contact-pair and general-contact definitions; background for the design doc above, not directly quoted line-by-line there. |
| `ctc_contactpairform_std.txt` | Interaction Guide, "Contact Formulations in Abaqus/Standard" | Node-to-surface vs. surface-to-surface, small-sliding vs. finite-sliding, anchor-point definition -- directly informs the design doc's Phase 1 formulation choice (small-sliding, node-to-surface). |
| `ctc_contactconstraints_std.txt` | Interaction Guide, "Contact Constraint Enforcement Methods in Abaqus/Standard" | Penalty (linear/nonlinear) vs. augmented Lagrange vs. direct/Lagrange multiplier, with Abaqus's own default-method-by-context table and default stiffness multipliers (10x for penalty, 1000x for augmented Lagrange). Directly informs the design doc's enforcement-method recommendation (penalty first, augmented Lagrange as Phase 2). |
| `ctc_normalinteraction.txt` | Interaction Guide, "Contact Pressure-Overclosure Relationships" | Hard vs. softened (linear/exponential/tabular) contact -- informs the design doc's "hard contact only for Phase 1" recommendation. |
| `ctc_friction.txt` | Interaction Guide, "Frictional Behavior" | Coulomb friction, penalty/Lagrange enforcement in the tangent plane -- background for the design doc's Phase 3 (friction deferred past Phase 1/2). |
| `ctc_contactmechanical.txt`, `ctc_contactdamping.txt`, `ctc_contactcontrols_std.txt` | Interaction Guide | Overview tying normal+friction together; viscous contact damping (same concept as this project's own `AbaqusViscousStabilization`, just per-contact-pair); numerical controls for chattering/overconstraint -- all cited in the design doc's §8 robustness discussion. |
| `ctc_genlcontinit_std.txt`, `ctc_adjustsurfaces_std.txt` | Interaction Guide, contact initialization | Strain-free adjustment vs. interference-fit conventions for initial overclosure -- informs the design doc's initialization recommendation. |
| `ctc_surfoverview.txt`, `ctc_deformablesurf.txt`, `ctc_nodebasedsurf.txt`, `ctc_rigidsurf.txt` | Model Guide, surface definitions | Element-based deformable / node-based / analytical rigid surface types -- informs the design doc's surface-representation section (this codebase already has an element-based-deformable-surface equivalent in `surface_tie3d.py`; only an analytical rigid plane is missing for the Phase 1 target). |

See `dev_log/3d_contact_implementation_design_20260913.md` for the full design this material was gathered for -- a deferred, no-timeline plan, not scheduled work.

## Refreshing these

The local doc server may not always be reachable (different machine, VPN,
server down). If you need to re-fetch or add a page:

```python
import urllib.request, re, html
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    raw = urllib.request.urlopen(req, timeout=20).read().decode('utf-8', errors='ignore')
    text = re.sub(r'<script.*?</script>', ' ', raw, flags=re.S | re.I)
    text = re.sub(r'<style.*?</style>', ' ', text, flags=re.S | re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    return re.sub(r'[ \t]+', ' ', re.sub(r'\n\s*\n+', '\n', text)).strip()

# bare path, NOT the ?show= form:
fetch('http://desktop-whl:4040/English/SIMACAEVERRefMap/simaver-c-3dpatch.htm')
```

Write output as UTF-8 to a file rather than printing to a Windows console —
these pages contain non-cp949-encodable characters (`\xa0` etc.) that crash
a plain `print()` on this machine's default console codepage.
