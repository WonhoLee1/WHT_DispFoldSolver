# Plan: Abaqus-spirit 2D element architecture refactor (2026-09-11)

**Status: DESIGN DOCUMENT FOR USER REVIEW. Nothing implemented. No code changed.**

Extends and partially revises
`dev_log/plan_abaqus_element_consolidation_20260908.md` (the master
consolidation plan) under its standing directive:

> **"영구 예외 없음. 전면 재검토할 것임. abaqus 요소로 1:1 개발할 것임."**
> (No permanent exceptions. Full comprehensive re-review. Every element
> developed 1:1 as a real Abaqus element.)

It also incorporates `dev_log/eas_frame_consistency_benchmarks_20260910.md`
(B1–B4) and AGENTS.md §4.14/§4.15.

**What this document adds** that the consolidation plan does not have:
1. A **verified** description of Abaqus's actual continuum-element
   geometric-nonlinearity architecture, per-claim sourced (§1). Three of
   the working assumptions this project has been carrying — including two
   written into current code comments — turn out to be **wrong**.
2. The finding that **Abaqus's own Theory Guide documents our F4/F6/B1–B3
   defect class and rejects the formulation we use** (§1.5). Abaqus's
   answer is architectural, not a corrected pull-back. This is the single
   strongest argument for the refactor and it is quoted, not inferred.
3. A **disagreement with the brief's proposed target architecture** (§3.1):
   a per-GP polar-decomposition + objective-rate `FrameHandler` is the
   wrong layer to build, because it serves *rate-form* materials and this
   repo has none. What we actually need is a configuration ledger. This
   is the biggest open question in the document (§7).

---

## 0. Reading order / TL;DR

- **§1** — what Abaqus actually does, with confidence tags. Read this first;
  it invalidates several current code comments.
- **§2** — current-state inventory, every file, KEEP / REFACTOR / DELETE.
- **§3** — target architecture, and where I disagree with the brief.
- **§4** — migration sequence, 7 stages, each with its gate and blast radius.
- **§5** — risk register.
- **§6** — explicit non-goals.
- **§7** — the open question the user needs to settle before Stage 2.

**One-paragraph summary.** This codebase's element-type strings encode
*three* independent axes at once — kinematics (full/incompatible/hybrid/
reduced), large-rotation device (whole-element co-rotational / TL / UL),
and **material family** — producing ~20 monolithic kernels where Abaqus has
6 element names and one material interface. The material axis is the one
that should never have existed: `dynamic.py`'s own alias table admits it
("the same bare Abaqus name can't alias to both"). The large-rotation axis
should not exist either: Abaqus has one geometric-nonlinearity switch
(NLGEOM) and every solid element is formulated in the current
configuration. The proposal is to collapse to **6 Abaqus-named element
formulations × any material via the existing Phase-A `response_fn`
interface × 2 backends**, with the *configuration* of every kinematic
quantity made explicit and non-optional, since all six defects found since
2026-09-08 (F4, F6, B1, B2, B3, and B4's cousin) were "which
configuration is this quantity in" errors.

---

## 1. What Abaqus actually does — verified reference

Sources are the public Abaqus 2016 documentation mirrors
(`ceae-server.colorado.edu/v2016/books/{stm,usb,sub}/`) and the 6.11 mirror
at `abaqusdocs.eait.uq.edu.au`. Confidence tags: **V(doc)** = fetched page,
section number and verbatim quote available; **V(lit)** = established in
cited literature; **INF** = my inference, not stated anywhere found;
**UNVERIFIED** = could not confirm.

### 1.1 NLGEOM and the (non-)choice of TL vs UL

| Claim | Conf. | Source |
|---|---|---|
| NLGEOM=NO → "elements … are formulated in the reference (original) configuration, using original nodal coordinates." | **V(doc)** | AUG §6.1.2 |
| NLGEOM=YES → "**most elements are formulated in the current configuration using current nodal positions**." | **V(doc)** | AUG §6.1.2 |
| Once on, it cannot be turned off in a later step. | **V(doc)** | AUG §6.1.2 |
| There is **no** user-exposed TL/UL selector for solids. `*SECTION CONTROLS` in Abaqus/Standard exposes only hourglass control, C3D10HS distortion control, hourglass scale factors, element deletion. | **V(doc)** (by exhaustive enumeration) | AUG §27.1.4 |
| The phrase "Updated Lagrangian" **does not appear** in the fetched Theory Guide. Abaqus's own wording is "formulated in the current configuration". | **UNVERIFIED** (absence) | — |

> **Consequence.** §9 of the consolidation plan ("TL/UL is not a solver-level
> switch") is **confirmed and should be strengthened**: in Abaqus it is not
> a switch at *any* level. It is not exposed to the user and it is not a
> per-element choice either — every solid element is current-configuration
> under NLGEOM. The only lever is **which element you choose**. Our
> `_use_ul_for(pid)` returning `False` for `Q4_COROTATIONAL_SRI` is a
> correct safety net *for an element Abaqus does not have*; it should not
> be the long-term architecture.

Note Abaqus's own hedge, "**most** elements": it does not enumerate the
exceptions there. Do not write "always UL for every element" as fact.

### 1.2 Rigid-body rotation removal is per-INTEGRATION-POINT, from the INCREMENTAL F

| Claim | Conf. | Source |
|---|---|---|
| "For most element types in Abaqus/Standard we approach this problem by first using the polar decomposition **in the increment** to define the change in the average material rotation over the increment, ΔR, from the total deformation **in the increment**, ΔF." | **V(doc)** | TG §1.4.3 |
| "The increment … of local motion **at the material calculation point** is defined by the incremental deformation gradient ΔF … The polar decomposition of ΔF is ΔF = ΔR·ΔU where **ΔR is the average rigid body rotation at the material point**." | **V(doc)** | TG §1.5.4 |
| All vectors/tensors of material state are rotated `a → ΔR·a`, `A → ΔR·A·ΔRᵀ`, *then* passed to the constitutive routine. | **V(doc)** | TG §1.4.3 |
| Strain increment handed to constitutive routines is **Δε = ln(ΔU)** — the logarithmic strain of the incremental stretch. | **V(doc)** | TG §1.4.3 |
| Objective rate for **Abaqus/Standard, Solid (Continuum), all materials = JAUMANN**. Green–Naghdi is for *structural* elements and for Explicit VUMAT/viscoelastic/brittle-cracking. | **V(doc)** | TG §1.5.3, Table 1.5.3–1 |
| "**For hyperelastic materials a total formulation is used; hence, the concept of an objective rate is not relevant for the constitutive law.**" | **V(doc)** | TG §1.5.3 |
| Jaumann (spin-based) and the ΔR-from-ΔF polar algorithm are consistent: for an infinitesimal increment ΔF = I + L·dt, the polar rotation is I + skew(L)·dt = the spin. Green–Naghdi is defined against R from the **total** F. | **INF** | — |
| No element-level rotation extraction for continuum elements appears anywhere in the fetched Theory Guide. The only documented element-level co-rotational formulation is **Abaqus/Explicit small-strain shells** (S3RS/S4RS/S4RSW, "Belytschko et al. (1984, 1992)"). | **V(doc)** | TG §3.6.6 |

> **Correction to a standing project assumption.** The brief, and prior
> session notes, assumed **Green–Naghdi**. For Abaqus/Standard solids it is
> **Jaumann**. This matters if we ever build the rate-form layer (§3.4).

> **Consequence for `Q4_COROTATIONAL`.** A whole-element T8 rotation
> extracted from edge vectors has **no Abaqus continuum counterpart**. It is
> an ANSYS/Belytschko structural-element device. F1 already proved it is a
> numerical no-op (bitwise-equal to TL); §1.2 establishes it is also not
> Abaqus. Both reasons point the same way: delete.

### 1.3 The UMAT contract — one interface, both material forms

| Claim | Conf. | Source |
|---|---|---|
| "**In finite-strain problems the stress tensor has already been rotated to account for rigid body motion in the increment before UMAT is called, so that only the corotational part of the stress integration should be done in UMAT.**" | **V(doc)** | SUB §1.1.44, STRESS |
| DROT = "increment of rigid body rotation of the basis system in which the components of stress (STRESS) and strain (STRAN) are stored … **stress and strain components are already rotated by this amount before UMAT is called**." Passed as **identity** for small-displacement analysis and for shells / local orientations. | **V(doc)** | SUB §1.1.44, DROT |
| STRAN components "**have been rotated … and are approximations to logarithmic strain**." | **V(doc)** | SUB §1.1.44 |
| DFGRD1 is the **total** F, "computed with respect to the initial configuration". Available for solids, membranes, finite-strain shells; **not** beams or small-strain shells. | **V(doc)** | SUB §1.1.44 |
| "**In the former case [large elastic strains], total-form constitutive equations relating the Cauchy stress to the deformation gradient are commonly used; in the latter case [pressure-dependent plasticity], rate-form constitutive laws are generally used.**" — and DDSDDE has **two documented exact definitions**, chosen by the material author. | **V(doc)** | SUB §1.1.44 |
| Every argument is supplied unconditionally; the material chooses which to read. The element does not branch on material type. | **INF** (strong — from the fixed argument list; the manual never says it in those words) | — |
| USER SDVs are **NOT** rotated by Abaqus: "any vector-valued or tensor-valued state variables **must be rotated** … The rotation increment matrix, DROT, is provided for this purpose." | **V(doc)** | SUB §1.1.44, STATEV |
| Built-in material state **IS** rotated by Abaqus (`T → ΔR·T·ΔRᵀ` plus constitutive change). For isotropic materials in continuum elements "**the global, spatial, system is used—the material basis is fixed in time**." | **V(doc)** | TG §1.5.4 |

> **This is the load-bearing architectural fact for the whole refactor.**
> Abaqus achieves "one element × any material" by supplying *both*
> representations at every call — the rotated rate-form triple
> (STRESS/STRAN/DSTRAN + DROT) **and** the total-form pair
> (DFGRD0/DFGRD1) — and letting the material pick. The element is
> material-blind because the interface is redundant by design, not because
> the two forms were unified.

### 1.4 CPE4 — the F̄ technique, and a concrete 1:1 defect in our pilot

| Claim | Conf. | Source |
|---|---|---|
| "For fully integrated first-order isoparametric elements … **the actual volume changes at the Gauss points are replaced by the average volume change of the element. This is also known as the selectively reduced-integration technique … or as the B̄ technique** … see Nagtegaal et al. (1974)." | **V(doc)** | TG §3.2.4 |
| Finite-strain form: "**Define the modified deformation gradient F̄ = (J̄/J)^(1/n)·F** … J is the Jacobian at the Gauss point; and **J̄ is the average Jacobian over the element**." | **V(doc)** | TG §3.2.4 |
| "**a modified deformation gradient F̄ is passed into user subroutine UMAT.**" | **V(doc)** | SUB §1.1.44 |
| For 2D elements "J̄ and J are the change in **area**", and only the in-plane terms are modified (out-of-plane uses average strain for axisym/GPS). | **V(doc)** | TG §3.2.4 |
| CPE4 still shear-locks: "**Fully integrated first-order elements should not be used in cases where 'shear locking' can occur** … The incompatible mode elements … should be used for such applications." | **V(doc)** | TG §3.2.4, AUG §28.1.1 |
| The **centroid**-sampled dilatation is a *different* device — an optional Abaqus/**Explicit** C3D8R "centroidal strain formulation", explicitly described as "**less accurate when the elements are skewed**". Identical to the uniform-strain approach only for parallelogram/parallelepiped shapes. | **V(doc)** | TG §3.2.4 |

> **CONCRETE DEFECT, found by this review.** `dispsolver/element/cpe4_jax.py`
> — the one element already built to the target architecture, and the
> proposed Stage-1 pilot — uses the **centroid** Jacobian for its F̄:
> `_cpe4_internal_force` computes `J0` from `_grads(0.0, 0.0, coords)` and
> its docstring states "the dilatational part evaluated at the element
> centroid … Abaqus's own selectively reduced (volumetric) integration".
> **That is not what Abaqus does.** Abaqus uses the element-**average**
> J̄ = ∫J dV⁰/V⁰. The two coincide only for parallelogram elements —
> which every unit test in `tests/test_cpe4_element.py` that isn't the
> irregular-quad patch test uses. This is the §4.14 lesson recurring
> verbatim: a device that is exactly correct on the test geometry and
> wrong on the production geometry. **The graded `ex12` display mesh is
> not parallelogram** at the transition columns. Fixing this is Stage 1's
> first task and is independently verifiable.
> *(Note the same centroid-vs-average question applies to
> `q4_visco_simo_fs_jax.py`'s F-bar and `q4_hybrid_jax.py`'s J_bar —
> the latter's docstring says `J_bar = sum(J_k w_k)/sum(w_k)`, which IS
> the Abaqus average. Audit each in Stage 1.)*

### 1.5 CPE4I — Abaqus documents our defect class and rejects our formulation

This is the most important paragraph in the whole reference, quoted in full
from **TG §3.2.5, "Continuum elements with incompatible modes"** — **V(doc)**:

> "Since we want to use a formulation that can be used for any material
> model, **we want to express the incompatible modes as a modification of
> the deformation gradient F**. The most obvious approach is to add the
> incompatible modes to the deformation gradient: [F = F_c + F_enh].
> **This approach has been used successfully by Simo and Armero.**
> Elements formulated on this basis satisfy the large-strain patch test
> … **However, once the elements become distorted due to deformation, the
> patch test will no longer be satisfied in an incremental sense; that is,
> subsequent homogeneous deformations will not be represented exactly.
> This turns out to be a fatal flaw in the formulation for problems
> involving large distortions in compression.**"

And what Abaqus does instead — **V(doc)**, same section:

> "we write the **total deformation gradient as the product of a series of
> incremental deformation gradients** … **Principal incompatible modes are
> then added to the incremental deformation gradient** … **[T₀] is the
> parametric transformation at the center of the element in the state at
> the start of the increment** … **[J₀] is the Jacobian at the centroid at
> the start of the increment**. Note that [J] is evaluated based on the
> deformation caused by the displacement degrees of freedom only and does
> not include the volume change due to the incompatible modes."

> "**Bilinear volumetric terms are added to the principal terms in a
> multiplicative way.**"

> "**For a finite-strain increment we use the midincrement approach
> proposed by Hughes and Winget.**"

> "The **geometrically linear** incompatible mode formulation used in
> Abaqus is related to … **Simo and Rifai (1990)** … **The nonlinear
> formulation is based on work by Simo and Armero (1992).**"

**Read what that says.** Abaqus names the exact formulation this codebase
uses (additive enhancement of F, Simo–Armero), states that it degrades
"once the elements become distorted due to deformation" — i.e. exactly when
the reference configuration accumulates deformation across increments,
which is exactly the condition under which F6 was 195× wrong and B1's
Jacobian was 97.6% wrong — and calls it "**a fatal flaw**". Abaqus's fix is
**architectural**: rebuild `T₀`/`J₀` at the element centroid **at the start
of every increment**, so the reference never accumulates distortion at all.

That structurally removes the failure mode instead of correcting a
pull-back. Our five fixes since 2026-09-08 (F4 push-forward ×4 files, F6
transpose ×5 copies, B1 `_push_forward`, B2/B3 enhanced-`B_L` +
single-push-forward) are all corrections *within* the formulation Abaqus
documents as flawed. They are measurably correct now — B1 4.6e-10, B2/B3
machine-precision symmetry, both benchmarks passing — but they are patches
to a structure that generates this defect class repeatedly. **Six
occurrences in three sessions is the evidence.**

Other CPE4I facts — all **V(doc)**:

| Fact | Source |
|---|---|
| Internal DOF counts: **4 for CPS4I; 5 for CPE4I/CAX4I/CPEG4I; 13 for C3D8I**. The 5th 2D mode is a bilinear volumetric term ("For two-dimensional elements a single term is added"). | TG §3.2.5, AUG §28.1.3 |
| "The incompatible mode elements use **full integration** and, thus, have no hourglass modes"; "it is not necessary to use selectively reduced integration". | TG §3.2.5, AUG §28.1.1 |
| Condensed at element level ("eliminated immediately"). **The condensation algebra is NOT given in the docs.** | TG §3.2.5 / **UNVERIFIED** |
| Whether α persists across increments: **INF** (incremental and re-referenced each increment, from the ΔF-based construction). Not stated. | — |
| Performance is shape-sensitive: "almost as well as second-order elements … **if the elements have an approximately rectangular shape. The performance is considerably less if the elements have a parallelogram shape. For trapezoidal element shapes the performance is not much better than … regular displacement elements.**" | TG §3.2.5 |
| "**Incompatible mode elements should be used with caution in applications involving large compressive strains. Convergence may be slow at times, and inaccuracies may accumulate in hyperelastic applications.**" | AUG §28.1.1 |
| No documented statement about CPE4I under large **rotation** specifically (only large strain/distortion). | **UNVERIFIED** |

> The "caution … large compressive strains … convergence may be slow"
> warning is Abaqus documenting, from the vendor side, the same phenomenon
> §11 of the consolidation plan chased as the `_ALPHA_MAX` saturation /
> element-local Newton divergence. Worth recording: the line-search remedy
> adopted in §11 remains right; this just says the underlying sensitivity
> is inherent to CPE4I, not a bug we introduced.

### 1.6 CPE4H / CPE4IH — hybrid pressure

| Claim | Conf. | Source |
|---|---|---|
| "**We remove this singular behavior … by treating the pressure stress as an independently interpolated basic solution variable** … with this coupling implemented by a **Lagrange multiplier**." | **V(doc)** | TG §3.2.3 |
| Hyperelastic form: an independent J̄ replaces J in the strain-energy potential, with **J̄ = J imposed by a Lagrange multiplier p**, p = ∂U/∂J̄. **Not** a `(1/D₁)(J−1)²` penalty — that expression is the *material* potential's volumetric part, evaluated on the independent J̄ in a hybrid element. | **V(doc)** | TG §4.6.1 |
| "λ will be interpolated over each element so that **the constraint is satisfied in an integrated (average) sense**." Internal virtual work on the **current** volume; linearization taken by pulling back to V⁰ then pushing forward. | **V(doc)** | TG §3.2.3 |
| **CPE4H = constant pressure, ONE additional variable. CPE4IH = "hybrid with LINEAR pressure", THREE pressure variables.** "Abaqus uses constant J̄ in most first-order elements". | **V(doc)** | AUG §28.1.3, TG §4.6.1 |
| Sources **conflict** on CPE4IH's incompatible-mode count: TG §3.2.5 says "In the hybrid elements the additional incompatible modes used to prevent locking are not included" (⇒ 4); AUG §28.1.3 says CPE4I *and CPE4IH* have five. Unresolved. | **CONFLICT** | — |
| With a UMAT, Abaqus by default replaces the UMAT's pressure with the Lagrange multiplier's — "**suitable for … an incremental formulation … but is not consistent with a total formulation that is commonly used for hyperelastic materials**", for which an opt-in total hybrid formulation exists. | **V(doc)** | SUB §1.1.44 |

> **Second concrete 1:1 gap.** Our `CPE4IH` uses an **element-constant**
> pressure (`_VISCO_HYBRID_TYPES` comment: "element-constant pressure
> unknown condensed alongside the enhanced modes"). Abaqus's CPE4IH is
> **linear** pressure, 3 variables. Record it; decide in Stage 3 whether to
> implement linear pressure or rename ours honestly.

### 1.7 CPE4R and the element-name taxonomy

| Claim | Conf. | Source |
|---|---|---|
| Abaqus/**Standard** default hourglass control for first-order reduced-integration solids is **total stiffness**, not "relax stiffness" (an Explicit default) and not enhanced (opt-in in Standard). | **V(doc)** | AUG §27.1.4 |
| The base method is "the artificial stiffness method … given in **Flanagan and Belytschko (1981)**". Reduced-integration first-order strain is the **uniform strain** (average over element volume), not the centroidal Gauss point. | **V(doc)** | TG §3.2.4 |
| "**When using first-order, reduced-integration elements in a simulation where bending deformation will occur, use at least four elements through the thickness.**" | **V(doc)** | GSA §4.7 |
| **All six of {CPE4, CPE4R, CPE4H, CPE4RH, CPE4I, CPE4IH} exist.** Complete plane-strain list also has CPE3(H), CPE6(H/M/MH), CPE8(H/R/RH). | **V(doc)** | AUG §28.1.3 |
| **There is no "CPE4S"** — no Abaqus element applies selective reduced integration to the *shear* term. SRI in Abaqus is volumetric-only. | **V(doc)** (exhaustive enumeration + the positive statement in TG §3.2.4) | AUG §28.1.3 |
| **F̄ is not an element name.** It is "the F̄ technique" *inside* CPE4/CPS4/CAX4/C3D8. An element named `Q4_FBAR` / `Q4_VISCO_SIMO` has no Abaqus counterpart; the Abaqus-equivalent name is simply **CPE4**. | **V(doc)** | TG §3.2.4, SUB §1.1.44 |
| **No CPE4IT / CPE4IHT** — incompatible modes are not offered in the coupled temperature-displacement plane-strain family. | **V(doc)** (absence from the complete list) | AUG §28.1.3 |

> §7b of the consolidation plan asked whether `Q4`'s B-bar disqualifies it
> from being CPE4; §7c retracted that and said B-bar *is* Abaqus's default.
> **§7c is confirmed correct** by TG §3.2.4 verbatim — with the one
> refinement of §1.4 above (average, not centroid).

### 1.8 The layering, stated once

Putting §1.1–§1.7 together, Abaqus's continuum stack is:

```
[ELEMENT — kinematics only, in the CURRENT configuration under NLGEOM]
    shape functions, B-operator, quadrature rule,
    + at most ONE locking device, which IS the element's identity:
        CPE4   : F̄ from the element-AVERAGE J̄            (volumetric)
        CPE4I  : incompatible modes added to ΔF,
                 T₀/J₀ rebuilt at the centroid EACH increment  (shear)
        CPE4H  : independent element-constant pressure,
                 Lagrange multiplier, integrated-average sense (volumetric)
        CPE4IH : both, with LINEAR pressure (3 vars)
        CPE4R  : 1-point uniform strain + Flanagan–Belytschko hourglass
        CPE4RH : CPE4R + constant pressure
    -> hands the material F̄ (or F), and nothing else element-specific
                    |
[FRAME — per INTEGRATION POINT, from the INCREMENTAL ΔF]
    ΔF = ΔR·ΔU  (polar);  Δε = ln(ΔU)
    rotate built-in state and stress by ΔR;  Jaumann rate (Standard solids)
    *** exists to serve RATE-FORM materials. For hyperelastic/total-form
        materials Abaqus states the objective rate "is not relevant". ***
                    |
[MATERIAL — one interface (UMAT), redundant by design]
    receives BOTH: {rotated STRESS, STRAN, DSTRAN, DROT}
                AND {DFGRD0, DFGRD1}
    reads whichever it needs; returns Cauchy stress + DDSDDE.
    The element never branches on which.
```

---

## 2. Current-state inventory

Every file in `dispsolver/element/`. "LR mechanism" = how large rotation is
actually handled. Verdicts: **KEEP** / **REFACTOR** (move onto the new
layer) / **REFORMULATE** (formulation itself must change, not just its
plumbing) / **DELETE** / **OUT-OF-SCOPE**.

### 2.1 Elements that are, or should become, real Abaqus elements

| File(s) | Type string(s) | Abaqus 1:1? | LR mechanism | Matches Abaqus? | Verdict |
|---|---|---|---|---|---|
| `cpe4_jax.py` | *(none — unwired)* | **CPE4** | UL via `F_n`, F4 push-forward built in, material-agnostic `response_fn` | **Partly.** Structure ✅. **F̄ uses the CENTROID J₀, Abaqus uses the element AVERAGE J̄** (§1.4) | **REFACTOR** — pilot. Fix F̄ first. |
| `q4.py`/`q4_jax.py`/`q4_numba.py` | `Q4` | **CPE4** (small-strain only) | none (small-strain B-bar) | Formulation ✅ (§7c confirmed), but no finite strain, no UL, one hard-wired material | **REFACTOR** → becomes the small-strain path of CPE4, or DELETE once CPE4 covers it. Still needed today for the BC-driven rigid plates. |
| `q4_eas.py` / `q4_eas_jax.py` / `q4_eas_numba.py` | `Q4_EAS` | **CPE4I** | UL via `F_n`; enhancement on **total** F (Simo–Armero additive) | **NO** — §1.5: Abaqus names this form and rejects it. Currently correct post-B1 but structurally the wrong form. Also `q4_eas.py` still carries a flagged config-*n*-weight/config-0-strain residue (2026-09-10 §6, dead path). | **REFORMULATE** to ΔF-incremental. Merge the 3 backends into one kernel + 2 lowerings. |
| `q4_visco_eas_jax.py` | `CPE4I`, `CPE4H`, `CPE4IH`, `Q4_VISCO_EAS` | **CPE4I/H/IH** | UL via `F_n`; post-B2/B3 single-push-forward, enhanced `B_L` | **NO** (same as above) + **CPE4IH pressure is element-CONSTANT; Abaqus's is LINEAR** (§1.6) | **REFORMULATE.** This is production PSA — highest care. |
| `q4_visco_eas_numba.py` | (gated off) | CPE4I/H/IH | UL, but **still on the pre-B2 formulation** | **NO** — already off-formulation vs its JAX sibling, and unreachable (`enable_new_numba_elements` defaults False) | **DELETE**, regenerate as a lowering of the single kernel (§3.5). Do not port B2 into a divergent copy. |
| `q4_visco_hybrid_up_numba.py` | `CPE4H` | **CPE4H** | UL via `F_n`, F4-fixed | Pressure constant ✅ matches CPE4H | **REFACTOR** → lowering. |
| `q4_visco_hybrid_reduced_jax.py` / `_numba.py` | `CPE4RH` | **CPE4RH** | UL via `F_n`, F4-fixed | Plausible; **hourglass control not verified against Abaqus's "total stiffness" default** (§1.7) | **REFACTOR** + add the hourglass-scheme audit to its gate. |
| `q4_reduced_jax.py` | `Q4_COROTATIONAL_REDUCED` (never dispatched) | **CPE4R** | J2, centroid + F–B hourglass | Honest CPE4R. Uses the **centroid** Gauss point; Abaqus's CPE4R uses the **uniform (average) strain** (§1.7) — same centroid-vs-average question as §1.4 | **REFACTOR** (rename `CPE4R`, wire it, fix centroid→uniform) — but keep unwired for *this* mesh; the 4-elements-through-thickness rule (§1.7) is why, and Abaqus documents the same limit. |
| `t3.py` | `T3`/`TRIA3` | **CPE3** | none, small-strain by docstring | Real name exists | **KEEP**, rename to `CPE3`. Out of the critical path. |

### 2.2 Invented devices with no Abaqus counterpart

| File(s) | Type string(s) | LR mechanism | Verdict |
|---|---|---|---|
| `q4_sri_jax.py`, `q4_sri_numba.py`, `q4_corotational_sri_j2_numba.py` | `Q4_SRI`, `Q4_COROTATIONAL_SRI` (+ aliases `CPE4S`, `CPE4S_COR`) | Whole-element T8 co-rotational; **shear sampled as the raw Cartesian off-diagonal of dU/dX — not frame-invariant** (measured 3.6e-2…1.6e-1 isotropy error) | **DELETE.** No Abaqus element applies SRI to shear (§1.7). This is production PET/GLASS — see Stage 4, the highest-risk stage. The `CPE4S`/`CPE4SH` aliases are **actively harmful**: they present an invented device under an Abaqus-shaped name in `model_builder.py`'s `.inp` parser. Remove the aliases early (Stage 0) even if the element survives to Stage 4. |
| `q4_sri_hybrid_jax.py`, `q4_sri_hybrid_numba.py` | `Q4_HYBRID_SRI`, `Q4_COROTATIONAL_HYBRID_SRI` (`CPE4SH`) | as above + Q1P0 | **DELETE** with the above. |
| `q4_corotational_jax.py` | `Q4_COROTATIONAL` (`CPE4_COR`) | Whole-element T8 from edge vectors | **DELETE.** F1: provably a bitwise no-op vs TL. §1.2: no Abaqus continuum counterpart at all (Abaqus's only element-level CR is Explicit small-strain shells). Two independent reasons. *Caveat: it is the reference row in `nlgeo_cantilever.py`'s benchmark table (−83.10% / −49.56%, the "CPS4 too stiff" control). Keep that control by pointing it at the new **CPE4** instead — which is what it was proxying for anyway.* |
| `q4_corotational_eas.py`, `q4_corotational_eas_jax.py` | `Q4_COROTATIONAL_EAS` (`CPE4I_COR`) | T8 wrapper around `q4_eas_jax` | **DELETE.** It is `q4_eas_jax` with a proven-no-op wrapper. Its only reason to exist is that `CPE4I` was taken by the viscoelastic kernel — the material-in-the-name defect, stated outright in `_ELEMENT_TYPE_ALIASES`'s own comment. Removing the material axis removes this file. |
| `q4_hybrid_jax.py`, `q4_hybrid_numba.py` | `Q4_HYBRID`, `Q4_COROTATIONAL_HYBRID`, `Q4_COROTATIONAL_HYBRID_EAS` (`CPE4H_COR`, `CPE4IH_COR`) | T8 + Q1P0 | **REFACTOR-or-DELETE, pending re-verification.** Its `J_bar = Σ J_k w_k / Σ w_k` *is* Abaqus's average (§1.4) — better than `cpe4_jax.py` on that point. But a 2026-07-30 dev_log found its CR-hybrid-EAS kernel was "a functional duplicate of CR-EAS … zero actual pressure/hybrid content". **Re-check that specific function in Stage 0** before deciding. |
| `q4_visco_simo_fs_jax.py`, `q4_visco_hybrid_simo_numba.py` | `Q4_VISCO_SIMO`, `Q4_UP` | UL via `F_n`, F4-fixed; F-bar mean dilatation | **REFACTOR → this IS `CPE4`.** F-bar is not an element name (§1.7); an F̄ full-integration quad *is* CPE4. Merging it with `cpe4_jax.py` is the single largest de-duplication available. It is also the current production PSA fallback and Cook's-membrane reference — so merge onto it carefully, don't delete under it. |
| `q4_up_jax.py` | `Q4_UP` (plain) | TL Green–Lagrange, no `F_n` | **DELETE.** Confirmed unused in `fold_model_config.py`; superseded by real `CPE4H`. |
| `q4_visco_hybrid_fs_jax.py` | `Q4_UP` + `LinearViscoelastic` | TL, no UL, no Numba | **DELETE.** Legacy; `LinearViscoelastic` itself superseded. Only old `ex03`/`ex06`–`ex08` reference it. |
| `q4_visco_hybrid.py`, `q4_visco_hybrid_jax.py` | *(no dispatch string — dead)* | small-strain | **DELETE.** |

### 2.3 Infrastructure / out of scope

| File | Verdict |
|---|---|
| `q4_plastic_numba.py` | Shared J2 stress+tangent kernel, not an element. **KEEP**; note it still carries B4's fixed `h = 1e-6` FD step (2026-09-10 §7) — same latent mesh-refinement bug, unmeasured. Fix opportunistically. |
| `shell_2d_jax.py` | B21/B22 territory, different element category. **OUT-OF-SCOPE.** |
| `rbe2.py`, `rbe2_jax.py` | Constraint, not a continuum element; already superseded by `RigidBodyPart` (AGENTS.md §4.10). **OUT-OF-SCOPE.** |

### 2.4 Two live latent defects found while inventorying

Neither is caused by this plan; both should be fixed in Stage 0.

1. **`"CPE4"` is declared but not dispatched.** `_ELEMENT_LARGE_DEF` (dynamic.py:406)
   declares `"CPE4": supports_ul=True, frame_invariant=True`, and
   `element_large_deformation_report()` will happily print
   `pid N: CPE4  UL (rotated reference)`. But `grep '"CPE4"'` finds **exactly
   one occurrence in the whole file** — that table entry. There is no
   dispatch branch. A model built with `element_type="CPE4"` falls through
   to the sequential `_element_contributions()` → plain **small-strain
   B-bar Q4**, silently, while the report claims Updated Lagrangian. This
   is precisely the AGENTS.md §4.8 silent-fallthrough class, and the
   `Q4_COROTATIONAL_REDUCED` comment at dynamic.py:~478 shows the author
   *knew* about this trap for a different name and guarded that one.
   **Fix: make `_ELEMENT_LARGE_DEF` membership and dispatch membership
   assert-consistent at construction.** That assert is cheap and would have
   caught this.
2. **Element dispatch is split across two files-worth of distance.** The
   element-type string is consumed both in the **constructor**
   (dynamic.py ~1200–1470, where `_coro_vmap_cache` / `_visco_simo_vmap_cache`
   build and cache the per-pid jitted closures) and in **assembly**
   (~3596–4110, ~15 branches). Changing one element means editing both,
   consistently, or getting a silent mismatch. 43 `element_type` references
   total. Any migration stage must treat these as one unit.

---

## 3. Target architecture

### 3.1 Where I disagree with the brief — the "FrameHandler" is the wrong layer

The brief proposes:

> a per-GP `FrameHandler` (polar decomp + objective rate) that sits between
> a UL element kernel and the material `response_fn`

**I recommend against building this now, and the reason is in §1.3.**

Abaqus's per-GP polar-decomposition + Jaumann-rate machinery exists to make
**rate-form** constitutive laws objective. Abaqus says so directly:
"Objective rates are relevant only for rate form constitutive equations
(e.g., elastoplasticity). **For hyperelastic materials a total formulation
is used; hence, the concept of an objective rate is not relevant for the
constitutive law**" (TG §1.5.3).

**Every material in this repository is total-form.**
`plastic_jax.pk2_voigt_jax` is finite-strain *multiplicative* J2 — it takes
the total F, does a spectral return map, returns PK2. F2 already proved it
is exactly rotation-equivariant including its history state
(`F_p_inv_new = F⁻¹F_e_new` is literally unchanged under `F → QF`).
`_simo_pk2` likewise takes total F and returns PK2. Both are the
DFGRD1-consuming, total-form branch of the UMAT contract. Neither needs
STRAN, DSTRAN, DROT, or an objective rate — and Abaqus would not give them
one either.

Building a Jaumann-rate objective-stress-update layer would therefore be
**adding a device that neither our materials nor Abaqus's treatment of our
materials requires**, while the thing that actually keeps breaking —
work-conjugacy and reference-configuration bookkeeping for a *total-form*
stress evaluated under a *current-configuration* element — is a problem
Abaqus mostly sidesteps (its incompatible modes never accumulate a rotated
reference, §1.5) rather than one it solved with a frame handler.

**What to build instead: a configuration ledger.** All six defects (F4, F6,
B1, B2, B3, and the `q4_eas.py` residue) were the same sentence: *"this
quantity is referred to configuration X and I contracted it with something
referred to configuration Y."* The fix is to make that impossible to write,
not to add a rotation layer.

### 3.2 Module structure

```
dispsolver/element/
  kinematics/
    frame.py          # Config ledger:  Config enum {REF0, REF_N, CURRENT},
                      # GPKinematics record, push_forward()/pull_back()
                      # operators. NO objective rate. NO polar decomposition.
    cpe4.py           # CPE4    kinematics: 2x2 Gauss + F-bar (AVERAGE J-bar)
    cpe4i.py          # CPE4I   kinematics: incompatible modes on Delta-F,
                      #         T0/J0 rebuilt at centroid each increment
    cpe4h.py          # CPE4H   kinematics: constant pressure, LM
    cpe4ih.py         # CPE4IH  = cpe4i + linear pressure (3 vars)
    cpe4r.py          # CPE4R   uniform strain + F-B hourglass
    cpe4rh.py         # CPE4RH  = cpe4r + constant pressure
    condense.py       # ONE static-condensation routine for alpha and p,
                      # with the K_au = K_ua^T / K_qu = K_uq^T symmetry
                      # ASSERTED at build time in debug mode, not commented.
  backends/
    jax_lowering.py   # jax.jit(jax.vmap(kernel)), cache keyed per Sec 3.5
    numba_lowering.py # @njit mirror, MUST pass the JAX-agreement contract
  legacy/             # everything in Sec 2.2 marked DELETE, moved here
                      # unchanged in Stage 0, deleted in Stage 6
dispsolver/material/
  response_interface.py   # unchanged (Phase A) -- already the right shape
  rate_form_adapter.py    # NOT BUILT. Designed for, see Sec 3.4.
```

### 3.3 The configuration ledger — the concrete anti-defect device

```python
class Config(Enum):
    REF0    = 0   # original, undeformed
    REF_N   = 1   # last converged  (the "current" config, UL)
    CURRENT = 2   # trial iterate

@dataclass(frozen=True)
class GPKinematics:
    """Every field names the configuration it is referred to.
    Constructed once per Gauss point by the element; consumed by the
    assembly helper. No element may build f_int by hand."""
    F_total:  Array      # (2,2)  ALWAYS referred to REF0  (what the material gets)
    F_inc:    Array      # (2,2)  CURRENT relative to REF_N
    F_n:      Array      # (2,2)  REF_N relative to REF0
    B_L:      Array      # (3,8)  built from F_inc  -> conjugate on REF_N
    weight:   float      # detJ on REF_N  -> dV_n
    weight_config: Config = Config.REF_N
```

and exactly one place that contracts them:

```python
def gp_internal_force(kin: GPKinematics, S_voigt_ref0, C_voigt_ref0=None):
    """The ONLY function permitted to contract a stress with a B-operator.

    S_voigt_ref0 is the material's PK2, referred to REF0 by definition of
    the response interface. kin.B_L and kin.weight live on REF_N. The
    push-forward is applied HERE, unconditionally, once:
        S_n = F_n S F_n^T / det(F_n)
        C_n = T^T C T   / det(F_n)      (T = Voigt operator of E -> F_n^T E F_n)
    Exactly the identity when F_n = I, so TL is bit-identical.
    """
```

Three properties this buys, each mapping to a defect that actually shipped:

- **F4/B1 cannot recur.** The push-forward is not something an element
  author remembers; it is the only path from stress to force.
- **B2/B3 cannot recur.** A partial push-forward is unrepresentable — a
  mixed element gets `f_u` and `f_a` from the *same* `gp_internal_force`
  call on the *same* `GPKinematics`, so they are gradients of one
  potential by construction, and `K_au = K_ua^T` follows rather than being
  assumed in a comment.
- **F6 becomes a single-site risk.** `J0^{-T}` lives in one
  `cpe4i.py::enhanced_modes`, not five copies across JAX/NumPy/Numba.

**The design rule to write into the module docstring:** *no element kernel
may reference a coordinate array, a Jacobian, or a stress without going
through a `GPKinematics` field whose name states its configuration.*

### 3.4 The rate-form adapter — designed for, not built

If a genuinely rate-form material is ever added (a small-strain UMAT-style
J2, a hypoelastic law, an anisotropic model with a rotating material
basis), that is when §1.2/§1.3's machinery becomes necessary and the right
place for it is `material/rate_form_adapter.py`: a wrapper that turns a
rate-form law into something satisfying the total-form `response_fn`
signature by doing the polar decomposition of `F_inc`, `Δε = ln(ΔU)`,
`ΔR`-rotation of stress and tensor state, and the Jaumann update — i.e.
Abaqus's own §1.4.3 algorithm, at the material side of the interface where
it belongs. **Elements stay unaware.** Nothing in Stages 1–6 requires it.
Note when building it: **Jaumann for Standard solids, not Green–Naghdi**
(§1.2) — the opposite of what this project has been assuming.

### 3.5 Backends without combinatorial explosion

Today's cost is *N_kinematics × N_materials × N_backends* source files.
Target is *N_kinematics* source files + 2 lowerings.

- **One kernel per element**, written against `GPKinematics` and
  `response_fn`. JAX and Numba are **lowerings of the same kernel**, not
  independent implementations. `q4_visco_eas_numba.py` drifting off B2's
  formulation while gated off (2026-09-10 §7, "Divergence deliberately
  left in place") is the failure mode this rule exists to prevent.
- **JIT cache key**: extend the existing `_visco_simo_vmap_cache` pattern
  (dynamic.py:1359, which already keys on material params so pids sharing a
  material share one compile, per §4.5) to
  `(element_name, id(response_fn), n_state, n_elem_shape, backend)`.
  `response_fn` is already a `static_argnames` argument and
  `make_visco_response(base)` already caches per base so the same object
  comes back — that contract is in place and must be preserved.
  Compiles = (#element formulations in the model) × (#distinct materials in
  the model), typically 2–3 × 2 for `ex12`. Not N².
- **Numba agreement is a contract test, not an aspiration.** An element's
  Numba lowering ships only if it matches JAX to ~1e-13 on the tilted +
  distorted UL probe (`scratch/tier1_numba_jax_tilted.py` already does
  exactly this, and measured 2.71e-15 post-§11). If it can't, it doesn't
  ship — no gated-off divergent copies.

### 3.6 What happens to `_ELEMENT_LARGE_DEF` / `_use_ul_for`

**Keep them, but demote and simplify.** Once every element is
frame-covariant by construction, `frame_invariant` is true for all of them
and `supports_ul` collapses to "is this element in the new library".

- Rename the concept to match Abaqus: `supports_nlgeom`.
- Keep the **fail-closed** default (unknown type ⇒ refuse). That property
  is the reason F6 could not survive a second time and is worth more than
  the table's current content.
- **Add the missing assert** from §2.4.1: every key in the table must have
  a dispatch branch and vice versa, checked at `DynamicSolver.__init__`.
- Keep `element_large_deformation_report()` — it is the right diagnostic
  and it becomes trustworthy once the assert exists.

### 3.7 Naming — one axis, not three

| Today | Target |
|---|---|
| `CPE4I` (means: incompatible modes **+ viscoelastic**) | `CPE4I` + `material=ViscoelasticMaterial(...)` |
| `Q4_COROTATIONAL_EAS` / `CPE4I_COR` (means: incompatible modes **+ J2**) | `CPE4I` + `material=J2Plasticity(...)` |
| `Q4_VISCO_SIMO`, `Q4_UP`, `Q4` | `CPE4` + the material |
| `Q4_EAS` | `CPE4I` + the material |
| `Q4_SRI`, `CPE4S`, `Q4_COROTATIONAL_SRI` | **no target** — invented, delete (Stage 4) |
| `Q4_COROTATIONAL`, `CPE4_COR` | **no target** — delete |

Every retired string stays in `_ELEMENT_TYPE_ALIASES` mapping to its real
Abaqus name **except** `CPE4S`/`CPE4SH`/`CPE4S_COR`/`CPE4SH_COR`, which must
be **removed, not aliased** — they assert an Abaqus element that does not
exist (§1.7), including inside `model_builder.py`'s `.inp` parser. A deck
containing `CPE4S` should error, not silently resolve.

---

## 4. Migration sequence

Every stage's gate includes the **element contract suite** (§4.1) plus that
stage's own numbers. No stage may proceed on "tests pass" alone — AGENTS.md
§4.9 applies throughout.

### 4.1 The element contract suite (Stage 0 deliverable)

One parametrized harness, `tests/element_contract/`, applied to **every**
dispatched element type. Consolidates T1–T5 (consolidation plan §1) with
§4.15's new standing test:

| # | Test | Catches | Exists today? |
|---|---|---|---|
| C1 | Rigid-rotation canary, zero strain | gross errors | yes (scattered) |
| C2 | **Strain-then-rotate objectivity** (T1) at real free-span AR | F3 / SRI class | **no** |
| C3 | **UL ≡ TL consistency** (T2) | F4 class | yes, `test_ul_tl_consistency.py` (4 elements) — **extend to all** |
| C4 | Mid-step `F_n` idempotence (T3) | F5 | yes, `test_ul_fn_idempotence.py` |
| C5 | Constant-stress patch test on **irregular** quads | basic correctness | partly |
| C6 | **Rotated-reference AR sweep**, bending ratio vs 1.000 | F6 | ad hoc in `scratch/` — **promote** |
| C7 | **Global-axis isotropy** (whole problem rotated: reference AND displacement) | SRI/EAS class | yes for CPE4 only — **extend** |
| C8 | **`K_e` IS the FD Jacobian of the element's own residual, on a ROTATED reference** | B1 | **no** — §4.15's standing test |
| C9 | **Condensation symmetries** `K_au = K_ua^T`, `K_qu = K_uq^T`, asserted not commented | B2, B3 | **no** |
| C10 | **JAX ≡ Numba** on the tilted+distorted UL probe | backend drift | `scratch/` only — **promote** |
| C11 | **FD step is mesh-invariant**: `‖dK‖/‖K‖` flat across a factor-8 element-size range | B4 | **no** |

C2, C6, C7, C8, C9, C10, C11 are the seven that do not exist as permanent
tests. **Five of the six shipped defects would have been caught by C8 alone.**

### Stage 0 — instrumentation and characterization (no formulation change)

1. Build `tests/element_contract/`; run it against **every currently
   dispatched element type** and record the results as a committed
   baseline, `xfail`-marking the known-bad (SRI's C2/C7, `q4_eas.py`'s
   flagged residue). *This is the "what did I break" oracle for every
   later stage.*
2. Fix §2.4.1: assert `_ELEMENT_LARGE_DEF` keys ≡ dispatch branches.
3. Remove the `CPE4S`/`CPE4SH` aliases from `dynamic.py` **and**
   `model_builder.py` (§3.7). Small, contained, and stops the invented
   device from being presented as Abaqus in `.inp` decks.
4. Re-verify `q4_hybrid_jax.py`'s CR-hybrid-EAS kernel against the
   2026-07-30 "zero actual pressure content" finding (§2.2).
5. Move §2.2's DELETE candidates into `dispsolver/element/legacy/`
   **unchanged**, imports updated. No behavior change; makes the remaining
   surface legible.

**Blast radius**: dynamic.py — the alias dict, plus one assert in
`__init__`. No dispatch branch touched. No formulation touched.
**Gate**: full `pytest tests/`, `verification.run_all`, both Abaqus
benchmarks — all bit-identical to the documented baseline
(284 passed / 3 failed; 12/14; the §5 Cook table).

### Stage 1 — PILOT: CPE4 end-to-end on the new layer

The pilot is chosen as **CPE4, not CPE4I**, deliberately: `cpe4_jax.py` is
currently **unwired** (grep-confirmed: referenced only by
`tests/test_cpe4_element.py`), so Stage 1 is **purely additive** and cannot
regress anything, while still exercising the whole layer — config ledger,
material-agnostic dispatch, JIT cache key, both backends.

1. Fix `cpe4_jax.py`'s F̄: element-**average** J̄, not centroid J₀ (§1.4).
   Audit `q4_visco_simo_fs_jax.py` and `q4_hybrid_jax.py` for the same.
2. Introduce `kinematics/frame.py` and re-express `cpe4_jax.py` through
   `GPKinematics` + `gp_internal_force`.
3. Add **one** dispatch branch for `"CPE4"` in `_assemble_multi_material_batch`,
   plus its constructor-side JIT cache entry (§2.4.2 — both, or neither).
4. Numba lowering + C10.

**Gate**:
- Contract suite C1–C11 on CPE4, both materials (J2 and viscoelastic),
  both backends.
- `verification/abaqus_benchmarks/nlgeo_cantilever.py` with CPE4 must
  reproduce the documented **CPS4-is-too-stiff** control: ≈ −83% at nx=10,
  ≈ −50% at nx=20 (today's `Q4_COROTATIONAL` rows, 1.3697 / 4.0889 — which
  §2.2 says CPE4 should inherit as the control).
- `cook_membrane.py` with CPE4 must land in the `Q4_VISCO_SIMO` band
  (6.2053 / 6.7063 / 6.8648 / 6.9181 at n=4/8/16/32) — because
  `Q4_VISCO_SIMO` **is** CPE4 (§2.2), so this is a merge-equivalence check,
  not a new measurement.
- **Not** an `ex12` gate. Nothing production runs on CPE4 yet.

**Blast radius**: additive. 1 new dispatch branch, 1 new cache entry, 0
existing branches modified.

### Stage 2 — CPE4I reformulated to Abaqus's ΔF-incremental form

**This is the intellectually hard stage and the one to argue about (§7).**
It is a *reformulation*, not a port: §1.5 documents that Abaqus rejects the
form we use, and the fix is to add the enhancement to **ΔF** with `T₀`/`J₀`
rebuilt at the element centroid each increment.

1. Implement `kinematics/cpe4i.py` per §1.5. Both the 4 principal modes and
   the 5th bilinear volumetric mode (CPE4I has **5**, §1.5 — check whether
   ours has 4 or 5; if 4, that is a third 1:1 gap).
2. Run it **side by side** with the existing (B1/B2-corrected) kernel — do
   not replace. Both dispatchable, `CPE4I` and `CPE4I_LEGACY`.
3. Compare on both benchmarks *and* on ex12's PSA layers in isolation.

**Gate — numbers that already exist and must not get worse**:
- `nlgeo_cantilever`: Q4_EAS today is **7.7799 (−4.04%) at nx=10** and
  **8.0702 (−0.46%) at nx=20**, 0 cutbacks. New CPE4I must match or beat.
- `cook_membrane`: CPE4I today is **6.3224 / 6.6983 / 6.8478 / 6.9068** at
  n=4/8/16/32, 0 cutbacks. Must match or beat, and must retain the
  documented ordering (less locked than CPE4 at the coarse mesh).
- Contract C8 and C9 at machine precision on a rotated reference.
- **Abaqus's own claimed advantage must be visible**: the ΔF form should
  beat the total-F form on a *large-compressive-distortion* case, which is
  the specific failure §1.5 names. If we cannot construct a case where the
  new form wins, we have not reproduced Abaqus's reason and should say so
  rather than assert the reformulation was worthwhile.
- Only then: `ex12` A/B, PSA on new vs legacy CPE4I, comparing the fold
  shape, the §3.0.1 interlayer-slip table, cutback count and wall time.

**Blast radius**: adds a branch; modifies none until the A/B passes. The
`CPE4I`↔`CPE4I_LEGACY` swap at the end is a one-line config change in
`fold_model_config.py` (`psa_element_type`).

### Stage 3 — CPE4H / CPE4IH / CPE4RH

Same pattern. Additionally resolve:
- **CPE4IH pressure must become LINEAR (3 vars)** or be honestly renamed
  (§1.6). Decide explicitly; do not leave it constant-under-the-Abaqus-name.
- The TG-§3.2.5-vs-AUG-§28.1.3 conflict on CPE4IH's incompatible-mode count
  (§1.6) — pick one, document the choice and the conflict.
- CPE4RH's hourglass control vs Abaqus's documented Standard default
  ("total stiffness", §1.7).

**Gate**: contract suite + `cook_membrane` (the near-incompressible case
these elements exist for) + `test_cpe4h_hybrid.py`/`test_cpe4rh_patch.py`.

### Stage 4 — PET/GLASS off `Q4_COROTATIONAL_SRI` (Family B) — HIGHEST RISK

This is the stage that can break the production 90°/side result. It is
deliberately **fourth**, after CPE4 and CPE4I are proven, because CPE4I is
the standing candidate and it must be trustworthy before PET/GLASS moves
onto it.

De-risking, in order:
1. **Comparison before commitment.** Run `ex12_abaqus_inp_plate_fold.py`
   with `pet_element_type` / `glass_element_type` = `CPE4I` against the
   current `Q4_COROTATIONAL_SRI`, same mesh, same everything. Compare:
   final angle reached, cutback count, `n_inverted`, the U-shape/kink
   verdict, the interlayer-slip table (§3.0.1), wall time.
2. **The AR sweep is the crux.** AGENTS.md §4.1 measured COROT at 20.9× /
   80× / 316× artificial bending stiffness at AR 7.5 / 15 / 30 —
   *axis-aligned only*, which §4.14 notes is exactly the measurement that
   cannot see a frame-covariance defect. Redo it as contract C6 (rotated
   reference) for SRI **and** CPE4I, at the three real `gen_ex12_inp.py`
   column widths. Expect CPE4I ≈ 1.000 everywhere post-F6 and SRI to
   degrade off-axis. **If that is not what the measurement shows, stop and
   re-plan** — it would mean the F3 result (SRI objective to 1e-14 under
   its co-rotational wrapper) is doing more work than the isotropy
   measurement suggests.
3. Only after both: flip `fold_model_config.py`'s defaults, keep
   `Q4_COROTATIONAL_SRI` dispatchable under a deliberately non-Abaqus name
   (`X_SRI_LEGACY`) for one release, then delete in Stage 6.

**Gate**: the full AGENTS.md §1.0 success criteria — 90°/side, 180°
combined, `n_inverted == 0`, smooth U with no kink, plates parallel — plus
cutbacks and wall time no worse than the documented 101 steps / 0 cutbacks
/ ~85 s.

### Stage 5 — CPE4R, CPE3, and the deletions

Rename `q4_reduced_jax.py` → `CPE4R`, fix centroid → uniform strain (§1.7),
wire it, keep it *unused* for this mesh with the 4-elements-through-thickness
reason documented. Rename `t3.py` → `CPE3`. Delete everything in
`element/legacy/` whose references are now zero — confirm with a repo-wide
grep including `tests/`, `examples/`, `scratch/`, `verification/`.

### Stage 6 — dispatch collapse in `dynamic.py`

Only now. ~15 assembly branches + ~6 constructor cache builders collapse to
**6 element formulations × 1 generic material path**. `_ELEMENT_LARGE_DEF`
simplifies per §3.6. This is the stage that pays back the effort, and it is
last because it is the one that cannot be done incrementally.

**Gate**: full `pytest tests/`, `verification.run_all`, both Abaqus
benchmarks, and a full `ex12` + `ex13` production re-run with
`sanity_report()` on (AGENTS.md §4.9 — convergence is not evidence).

---

## 5. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **The production 90°/side `ex12` fold breaks.** | Medium | **Critical** — it is the project's reference result | Staged so nothing production-facing changes before Stage 2's A/B and Stage 4's gate. Both stages keep the old element dispatchable and flip only a config default. Full §1.0 criteria as the gate, not "it converged". |
| R2 | **Stage 2 is a reformulation of the production PSA element.** Current numbers may have been implicitly tuned around the total-F form (e.g. `_ALPHA_MAX`'s history, the §11 line-search parameters). | Medium | High | Side-by-side dispatch, never in-place. Gate on numbers that already exist (§5 Cook table, cantilever table). Accept "no improvement demonstrable" as a legitimate outcome that stops Stage 2 (§7). |
| R3 | **PET/GLASS on CPE4I under-performs SRI in the fold.** §4.1's AR measurement says COROT-SRI locks 80–316×, but that was measured on the *co-rotational* device, which F3 showed is objective to 1e-14 in its own frame. The two measurements are not obviously reconcilable. | Medium | High | Stage 4 step 2 makes this an explicit measurement with a documented stop condition, before any default flips. |
| R4 | **Numba/JAX divergence re-emerges.** Already happened once (`q4_visco_eas_numba.py`, off-formulation and gated off rather than fixed). | High if unguarded | Medium | C10 is a *contract*: a lowering that fails it does not ship. No `enable_new_numba_elements`-style gated divergence. |
| R5 | **JIT compile-time cliff.** `cpe4_jax.py` builds its tangent with full-element `jax.jacobian` — AGENTS.md §4.4 documents 480 s and never finishing for a plasticity-coupled element. | Medium | Medium | Keep the dual entry points (`compute_cpe4` autodiff / `compute_cpe4_material_tangent`) as a first-class part of the layer, not a CPE4 quirk: J2 always takes the material-tangent path. Budget a compile-time measurement into every stage's gate. |
| R6 | **`_ul_F_n` / state plumbing regressions.** The F5 class: state written at the wrong point in the Newton loop. | Low (fixed, and C4 covers it) | High (silent) | C4 in the contract suite, run every stage. |
| R7 | **Refactor scope creep into `dynamic.py`'s hot path.** 43 `element_type` references, split constructor/assembly (§2.4.2), and the §4.8 early-return trap live in the same function. | High | High | Stage 6 is deliberately last and is the *only* stage permitted to modify existing branches. Stages 1–3 are additive-only. Re-read AGENTS.md §4.8's warning before touching `_assemble`. |
| R8 | **Abaqus facts I could not verify become load-bearing.** Specifically: the CPE4I α-persistence question (INF), the CPE4IH mode-count conflict, and the exact condensation algebra (UNVERIFIED). | Medium | Medium | Each is flagged in §1 with its tag. Where a design decision depends on one, it must be justified on numerical grounds (a benchmark) rather than on the Abaqus claim. |
| R9 | **Performance regression from the generic layer** (a Python-level `GPKinematics` dataclass inside a hot loop). | Medium | Medium | The record exists at *trace* time only — under `jax.jit`/`@njit` it is fully unrolled and costs nothing at runtime. Verify with a timing probe in Stage 1's gate, not by assumption. |
| R10 | **Two files' worth of hidden dispatch** (§2.4.2) means a stage lands half-applied. | Medium | High (silent, §4.8 class) | Stage 0's assert (`_ELEMENT_LARGE_DEF` keys ≡ dispatch branches) is the tripwire. Extend it to the constructor cache map. |

---

## 6. Explicitly out of scope

- **All 3D.** `dispsolver/element3d/`, `dispsolver/solver3d/`,
  `dynamic3d.py`, `tests/test_3d_*.py`, `verification/abaqus_3d_*`. A
  different agent owns that work; commits `96f8adb` and `a4276a7` landed
  there concurrently. Nothing in this plan touches those paths. (Note the
  known-failing `test_3d_numba.py::test_3d_dod_multimaterial_assembly` is
  theirs, per 2026-09-10 §8.)
- **Beams and shells** — `shell_2d_jax.py` (B21/B22 territory). Different
  element category; Abaqus's treatment there is genuinely different (§1.2:
  Green–Naghdi, co-rotating material basis, DROT = I).
- **Constraints** — `rbe2.py`, `rbe2_jax.py`, `rbe2_condensed.py`,
  `surface_tie.py`. Not continuum elements.
- **Materials.** The Phase-A `response_interface.py` contract is correct and
  stays as-is. The only material-side addition contemplated
  (`rate_form_adapter.py`, §3.4) is explicitly **not built** by this plan.
- **Renames without structural change.** Aliasing `Q4_EAS → CPE4I` while
  leaving two kernels because one is "the J2 one" is not progress; it is
  what §7 of the consolidation plan already did and what the CORRECTION
  block rejected. Every rename in §3.7 is downstream of an actual merge.
- **`_ALPHA_MAX` / EAS stabilization.** Settled in consolidation plan §11
  (clamp removed, line search adopted). Stage 2 must carry that forward,
  not revisit it.
- **The solver's own machinery** — `dt_controller.py`, convergence criteria
  (AGENTS.md §4.7's reverted 3-tier rewrite), stabilization, PARDISO
  caching. Untouched.

---

## 7. The open question the user must settle — Stage 2's premise

Everything up to Stage 1 is, I believe, uncontroversial: the config ledger,
the material axis removal, the deletions, the contract suite. **Stage 2 is
not.**

**The question: should CPE4I be reformulated to Abaqus's ΔF-incremental
form, or should the corrected total-F form be kept?**

*For reformulating* (my recommendation, weakly held):
- §1.5 is Abaqus stating, in its own Theory Guide, that our formulation has
  "a fatal flaw" once elements distort — the exact condition under which we
  have now found six defects in three sessions (F4, F6, B1, B2, B3, plus
  `q4_eas.py`'s open residue). The defect *class* keeps recurring because
  the structure generates it.
- "abaqus 요소로 1:1 개발" is unambiguous about which form CPE4I is. Keeping
  the total-F form is precisely a "permanent exception".
- Abaqus's form structurally removes the accumulating-rotated-reference
  problem rather than correcting for it, which retires the whole `F_n`
  push-forward bookkeeping burden for this element.

*Against* (the honest counter-argument):
- The current form is, **as of 2026-09-10, measurably correct**: B1 4.6e-10
  FD-Jacobian agreement on a 39°-rotated reference, B2/B3 symmetry at
  machine precision, and both official Abaqus benchmarks passing at every
  mesh density (−0.46% on the cantilever at nx=20; within 3.5% on Cook's
  membrane throughout). We would be reformulating a component that
  currently passes its acceptance tests.
- It is the **production PSA element**. R2 is real.
- Abaqus's stated failure mode is "large distortions **in compression**".
  A folding display's hinge zone is a large-*rotation*, moderate-strain
  problem, and §1.5 has **no documented statement about CPE4I under large
  rotation specifically** (UNVERIFIED, §1.5). It is possible the flaw
  Abaqus names does not bind for our load case at all.

**My proposed resolution, which is why Stage 2's gate is written the way it
is:** treat "the ΔF form beats the total-F form on a large-compressive-
distortion case" as a **falsifiable prerequisite**. Build the new form, and
if we cannot construct a case where it measurably wins, we report that and
**stop** — keeping the corrected total-F form, documented as a *measured*
deviation from Abaqus with the evidence attached, rather than as an
unexamined exception. That is a different thing from a "permanent
exception": it is a deviation with a number behind it.

**Biggest single risk, separately:** R3 — the unreconciled tension between
AGENTS.md §4.1's "COROT locks 80–316× at production aspect ratios"
(axis-aligned measurement) and F3's "co-rotational SRI is objective to
1e-14" (rotated measurement). Those two facts point in opposite directions
about whether PET/GLASS is currently well-served, and Stage 4's entire
justification rests on resolving them. Contract C6 run on both elements at
the three real column widths is the measurement that decides it, and it
should arguably be run in **Stage 0**, before any of this is committed to —
it is a half-day of work and it could reorder the whole plan.

---

## 8. Appendix — provenance of the Abaqus claims

All §1 quotes are verbatim from these pages, fetched 2026-09-11:

- TG §1.4.3 Rate of deformation and strain increment — `ceae-server.colorado.edu/v2016/books/stm/ch01s04ath06.html`
- TG §1.5.3 Stress rates (Table 1.5.3–1) — `.../stm/ch01s05ath10.html`
- TG §1.5.4 State storage — `.../stm/ch01s05ath11.html`
- TG §3.2.3 Hybrid incompressible solid element formulation — `.../stm/ch03s02ath61.html`
- TG §3.2.4 Solid isoparametric quadrilaterals and hexahedra — `.../stm/ch03s02ath62.html`
- TG §3.2.5 Continuum elements with incompatible modes — `.../stm/ch03s02ath63.html`
- TG §3.6.6 Small-strain shell elements in Abaqus/Explicit — `.../stm/ch03s06ath84.html`
- TG §4.6.1 Hyperelastic material behavior — `.../stm/ch04s06ath123.html`
- SUB §1.1.44 UMAT — `ceae-server.colorado.edu/v2016/books/sub/ch01s01asb44.html`
- AUG §6.1.2 General and linear perturbation procedures — `abaqusdocs.eait.uq.edu.au/v6.11/books/usb/pt03ch06s01aus44.html`
- AUG §26.7.1 User-defined mechanical material behavior — `.../v6.11/books/usb/pt05ch25s07abm68.html`
- AUG §27.1.4 Section controls — `ceae-server.colorado.edu/v2016/books/usb/pt06ch27s01aus115.html`
- AUG §28.1.1 Solid (continuum) elements — `.../usb/pt06ch28s01alm01.html`
- AUG §28.1.3 Two-dimensional solid element library — `.../usb/pt06ch28s01ael02.html`
- GSA §4.7 Summary — `solar.colorado.edu/v2016/books/gsa/ch04s07.html`

The WUSTL v6.6 mirror returns 403; `docs.software.vt.edu` and `help.3ds.com`
are behind TLS/auth. Where the Colorado v2016 and UQ v6.11 mirrors were
cross-checked the text was identical. Equations are rendered as images on
these mirrors and are therefore paraphrased, never quoted.
