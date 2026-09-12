# Static Stress/Displacement Analysis Benchmark Suite — Design (not implemented)

Source: Abaqus Benchmarks Manual, "Static stress/displacement analysis"
section (`simabmk-m-StaticStressdisplacementAnalysis-sb.htm`), 13 pages
fetched 2026-09-13 to `benchmark_element/reference_abaqus_docs/`. This
document classifies all 13 against this codebase's 3D solid element
family (`benchmark_element/`, C3D8/C3D8I/C3D8_FBAR/C3D8_CR/C3D8H/C3D8R/
C3D4/C3D4_ANP/C3D10/C3D10M/C3D6, dispatched through
`dispsolver/solver3d/dynamic3d.py`'s `DynamicSolver3D`) and designs the
one that is actually implementable now. **No code was written for this
document — design only, per explicit instruction.**

Checked against this codebase's real constraints before classifying
anything READY:
- `dispsolver/constraint3d/` has only `rbe3_distributing3d.py` and
  `surface_tie3d.py` — **no contact/rigid-surface capability at all**.
- `dispsolver/solver3d/dynamic3d.py` has **no Riks/arc-length method** —
  `grep -i "riks\|arc.length"` returns nothing.
- This codebase's only 3D "hyperelastic" material
  (`MAT_HYPERELASTIC_NEOHOOKEAN` in `dispsolver/material3d/numba_materials.py`)
  was directly verified this session (2026-09-13, see
  `dev_log/solve_step_false_convergence_20260913.md`'s discussion) to be
  bit-identical to the linear-elastic material at the same (E, nu) — it
  is NOT real finite-strain hyperelasticity (no J-dependent volumetric
  energy, no Mooney-Rivlin/Ogden/Marlow forms). Any benchmark requiring
  genuine rubber elasticity is out of reach until that's built.
- No composite/layered/orthotropic material path exists for 3D solids
  (checked `dispsolver/material3d/numba_materials.py`'s dispatch table —
  isotropic-only props).

## 1. Classification of all 13

| # | Benchmark | Real element types used | Reference data | Verdict |
|---|---|---|---|---|
| 1 | Beam/gap example | `B23` beam + gap (contact-switch) elements | Exact tabulated gap-force table (Table 1) | **NOT-APPLICABLE** — beam+switching-contact physics, no solid-element family here |
| 2 | Anisotropic layered plate | `S3R`/`SC6R`/`S9R5`/`STRI65` shells | Exact table (disp/moment vs analytical, Spilker et al. 1976) | **NOT-APPLICABLE** — shell-only, no solid model used at all |
| 3 | Composite shells, cylindrical bending | `S8R`/`S4R`/`SC8R` shells + one `C3D20R` composite-solid case | Figure-based only for the solid case (no single clean number); Pagano 1969 exact elasticity solution exists but only plotted, not tabulated | **NEEDS-NEW-CAPABILITY**: composite/layered material (3D solids have none) |
| 4 | Thick composite cylinder, internal pressure | `C3D8`/`C3DI`/`C3D20R` — **real 3D solids** | **Exact tabulated**: Table 1/2, analytical ū_r(i)=1.4410, ū_r(o)=0.1476, σ̄θ(i)=5.7060, σ̄θ(o)=0.0103, per-element-type % error rows | **NEEDS-NEW-CAPABILITY**: composite/layered material — already flagged 2026-09-13 (`reference_abaqus_docs/README.md`), confirmed still true, not re-derived. Otherwise the single best-instrumented candidate in the whole list — revisit first if/when layered-material support is ever added. |
| 5 | Uniform collapse of straight/curved pipes | `ELBOW31B` pipe + `S8R5` shell | Figure-based only (moment-curvature curves), Brazier 1927 analytical is a curve not a number | **NOT-APPLICABLE** — pipe/shell, plus inherently unstable collapse (would need Riks too) |
| 6 | Snap-through of shallow cylindrical roof | `S4R5`/`S4R`/`S3R`/`SC6R`/`SC8R` shells | Figure-based curves only (no clean number); needs Riks for the actual snap | **NOT-APPLICABLE** (shell) + **NEEDS-NEW-CAPABILITY** (arc-length/Riks) if it were ever solid |
| 7 | Pressurized rubber disc | `CAX8H`/`C3D8IH`/`C3D10MH`/etc, real 3D solids in places | Figure-only (pressure vs. center-displacement curve vs. Oden 1972/Hughes & Carnoy 1981) — no tabulated number to check against without digitizing a plot | **NEEDS-NEW-CAPABILITY**: real hyperelastic material (Mooney-Rivlin/Ogden) — moot anyway since there's no clean number even if the material existed |
| 8 | Elastic sheet with hole, uniaxial stretch | `CPS8R` — **plane stress, 2D**, not a 3D solid at all | Figure-based (load vs. nominal strain curve vs. Oden/Treloar) | **NOT-APPLICABLE** — 2D plane-stress problem; belongs to the 2D suite's territory (`tests/element_contract/`) if pursued at all, and even there needs real hyperelasticity |
| 9 | Necking of a round tensile bar | 3D solids, but requires Gurson porous-plasticity material | Figure-based (no closed-form) | **NEEDS-NEW-CAPABILITY**: Gurson material — carried forward from the earlier 2026-09-13 pass, re-confirmed, not re-derived |
| 10 | Concrete slump test | Specialized concrete/Drucker-Prager-cap material | Figure/qualitative only | **NOT-APPLICABLE** — carried forward from the earlier pass (background-only, no usable number even with the right material) |
| 11 | Hertz contact problem | 2D/3D solids, but the whole point is contact | Closed-form Hertz theory (Timoshenko & Goodier 1951) exists, but only meaningful through a real contact solve | **NEEDS-NEW-CAPABILITY**: contact (no contact module exists in `dispsolver/constraint3d/`) |
| 12 | Crushing of a pipe | `CPE8H`/`CPE6H`/`CPE4I` (2D plane strain, hybrid) + `SC8R` shell + rigid contact | Experimental curve (Peech et al. 1977), figure-only | **NOT-APPLICABLE** to this 3D-solid folder (2D plane-strain physics — the 2D suite's territory) **and** needs contact regardless |
| 13 | Radial stretching of a cylinder | `CAX4R`/`CAX4`/`CAX8` axisymmetric + `CCL12`/`CCL24` cylindrical | **Exact closed-form** stress field (Lamé-type, given symbolically: σ_rr, σ_θθ, σ_zz=0, σ_rz=0, with λ,μ,C1,C2,C3 all given) | **READY** — pure linear elasticity, isotropic, no contact, no special material. The physics (thick hollow cylinder, radial displacement BC, plane-strain-like axial constraint) is dimension-independent — reproducible with 3D solid elements even though the original uses axisymmetric/cylindrical ones (same reasoning already used for the Fung torsion-of-hollow-cylinder analytical solution earlier this session). |

**Summary**: 1 READY (`rad_stretch`), 5 NEEDS-NEW-CAPABILITY (composite
material x2, real hyperelastic material x2 combined with figure-only
data, Gurson material x1, contact x2 — some benchmarks stack more than
one missing capability), 6 NOT-APPLICABLE (shell/beam/pipe/2D-only
physics with no reasonable 3D-solid equivalent, or genuinely no usable
number).

## 2. Design: `rad_stretch` — Radial Stretching of a Hollow Cylinder

### Problem, exactly as published

- Inner radius `Ri = 4.0`, outer radius `Ro = 6.0`, height `H = 2.0`
  (consistent units, per the source doc — not physical units).
- Material: linear elastic, isotropic, `E = 2.0e11`, `nu = 0.3`
  (density `1000` given but irrelevant for a static solve).
- BCs: inner surface (`r = Ri`) constrained in the radial direction only
  (free axially and circumferentially); base (`z = 0`) constrained
  axially (`uz = 0`); outer surface (`r = Ro`) given a **uniform radial
  displacement** `U0 = 0.2`.
- Original uses axisymmetric (`CAX4`/`CAX4R`/`CAX8`) or cylindrical
  (`CCL12`/`CCL24`) elements over the full or a sector domain. This
  design instead builds a **3D solid wedge** (a thin angular slice, e.g.
  5-10 degrees, one element through the angle — the problem has no
  theta-dependence at all, so a thin wedge with symmetric BCs on its two
  flat cut faces is an exact 3D-solid equivalent of the axisymmetric
  model) meshed with `C3D8`/`C3D8R`/etc in (r, z).

### Exact reference solution (reproduce these symbols exactly, don't
### re-derive — source: `reference_abaqus_docs/bmk_rad_stretch.txt`)

```
lam = (E*nu) / ((1+nu)*(1-2*nu))
mu  = E / (2*(1+nu))

C1 = -(2*lam/(lam+2*mu)) * (U0*Ro/(Ro**2 - Ri**2))
C2 = U0*Ro / (Ro**2 - Ri**2)
C3 = -U0*Ro*Ri**2 / (Ro**2 - Ri**2)

sigma_rr(r)     = (2*lam + 2*mu)*C2 + lam*C1 - 2*mu*C3/r**2
sigma_thetatheta(r) = (2*lam + 2*mu)*C2 + lam*C1 + 2*mu*C3/r**2
sigma_zz = 0
sigma_rz = 0
```
(`r` in `[Ri, Ro]`.) This is a genuine closed-form field, not a single
scalar — the benchmark should sample it at several radii, not just one
point, which is a stronger check than `run_abaqus_official_patch_test`'s
single affine field (that one only needed uniform stress/strain
everywhere; this one is a real radius-dependent field, exercising the
element's ability to resolve a stress GRADIENT, which the current
`benchmark_element` suite has never specifically tested).

### What's still a gap even for this READY case

This codebase's 3D solver has **no per-Gauss-point Cauchy-stress-recovery
API** (already noted in `run_abaqus_official_patch_test`'s own docstring,
still true). Two options, in order of preference:
1. Check **displacement only** against the closed-form radial
   displacement `u_r(r) = C2*r + C3/r` (derivable from the same
   constants above via `eps_rr = du_r/dr`, `eps_thetatheta = u_r/r`, and
   the isotropic stress-strain relations — not yet derived here, do this
   before implementing) at several radii — same rigor tier as the
   existing patch test's displacement-only check.
2. If stress comparison is wanted, this is the natural forcing function
   to finally add a real stress-recovery helper to `benchmark_element/`
   (a per-element Cauchy-stress post-processor from `F`/material law,
   analogous to `mechanics_patches.compute_per_element_dilatation` added
   this session for the ANP checkerboard investigation) — worth doing
   here since it would benefit every other benchmark in this suite too,
   not just this one.

### Proposed API (mesh generator + benchmark function, NOT implemented)

`benchmark_element/mechanics_patches.py`:
```python
def make_hollow_cylinder_wedge_mesh(
    elem_type: str,
    Ri: float = 4.0, Ro: float = 6.0, H: float = 2.0,
    n_r: int = 10, n_z: int = 2,
    wedge_angle_deg: float = 10.0,
) -> Tuple[Mesh3D, Dict[str, List[int]]]:
    """(r, theta, z) structured grid, one element through theta, mapped to
    (x, y, z). Returns mesh + faces = {'inner': [...], 'outer': [...],
    'base': [...], 'theta0': [...], 'thetamax': [...]} for BC application
    (inner: radial-only constraint via a local cylindrical transform or an
    equation constraint since fix_dof works in global x/y — needs
    checking whether DynamicSolver3D supports a rotated-DOF BC or whether
    the inner-radius constraint has to be approximated with the wedge's
    flat faces already fixing the relevant global directions; this needs
    resolving during implementation, not assumed here)."""
```

`benchmark_element/benchmark_3d_mechanics.py`:
```python
def run_radial_stretch_benchmark(elem_type: str) -> Dict[str, Any]:
    """Reproduce the Abaqus Benchmarks Manual 'Radial stretching of a
    cylinder' (reference_abaqus_docs/bmk_rad_stretch.txt) with a 3D solid
    wedge model. E=2e11, nu=0.3, Ri=4.0, Ro=6.0, H=2.0, U0=0.2 (uniform
    radial displacement at r=Ro, base fixed axially, inner radius
    radially constrained). Checks computed radial displacement at several
    r against the exact u_r(r) = C2*r + C3/r closed form -- NOT full
    Cauchy stress (no stress-recovery API yet, see design doc note)."""
```

### Why this is worth doing despite being a small addition

It's the first benchmark in this suite that checks a **spatially-varying
closed-form field** rather than a single uniform value (patch test) or a
single scalar ratio (bending/volumetric tests) — a genuinely different,
complementary kind of evidence, and it needs no new solver capability at
all (unlike every other candidate on this list).

## 3. Not pursued further right now

The 5 NEEDS-NEW-CAPABILITY items are real, valuable future work but each
requires new `dispsolver` capability (composite material, real
hyperelasticity, Gurson plasticity, or contact) before a benchmark can be
designed around them — implementing any of those is out of scope for a
benchmark-design task and belongs with whoever owns that part of
`dispsolver`. The 6 NOT-APPLICABLE items are real Abaqus benchmarks but
test element families (`beam`/`shell`/`pipe`/2D-plane-stress) this
folder's 3D-solid scope was never meant to cover.
