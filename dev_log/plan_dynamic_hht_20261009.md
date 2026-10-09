# Plan: Abaqus/Standard *DYNAMIC (HHT) end to end -- 2026-10-09

User decision (2026-10-09): "HHT를 지원해야지" -- implicit dynamics is supported, not refused.

## Findings (verified in code, 2026-10-09)

HHT exists in both modern solvers (`solver2d/dynamic2d.py` ~641, `solver3d/dynamic3d.py` ~1808,
presets in `solver/time_integrator_presets.py`), but a deck never reaches it and the branch itself
has defects:

| # | Defect | Where |
|---|---|---|
| D1 | `*DYNAMIC, APPLICATION= / ALPHA=` are undeclared params -> gap -> the whole block is dropped; the step stays STATIC with Step defaults (ex12 deck) | `io/keywords/steps.py` `Dynamic.Params` |
| D2 | `Model.create_solver2d/3d` always build `time_integrator="quasistatic"` (constructor default); `solve(step)` never looks at `step.procedure` -> every deck *DYNAMIC runs quasi-statically | `model/model.py` ~1039/~1148, `solve()` 2D ~931 / 3D ~2460 |
| D3 | Lumped mass is `I * rho_e / n_nodes` -- element volume/area (x plane thickness) missing. Hidden by the unit-cube test (`tests/test_rayleigh_damping.py`) | 2D ~665, 3D ~1833 |
| D4 | HHT weights f_int but not f_ext: residual must be `f_ext_hht - M a - f_int_hht` with `f_ext_hht = (1+a) f_ext_{n+1} - a f_ext_n` | 2D ~693, 3D ~1874 |
| D5 | Initial acceleration never computed (a = 0 at step start). Abaqus default (INITIAL omitted) computes it at the start of every *DYNAMIC step: `M a0 = f_ext(t0) - f_int(u0)` on free DOFs | both |
| D6 | Material without *DENSITY silently gets steel 7.85e-9 in dynamics; Abaqus requires a density for a dynamic step | 3D ~636, 2D ~156 |
| D7 | `resolve_time_integrator` unknown name silently becomes Newmark | presets |

## Abaqus reference (Keywords Guide *DYNAMIC; Analysis Guide "Implicit dynamic analysis", Table 1)

- APPLICATION default: TRANSIENT FIDELITY without contact, MODERATE DISSIPATION with contact.
  - TRANSIENT FIDELITY -> TIME INTEGRATOR=HHT-TF, IMPACT=AVERAGE TIME, INCREMENTATION=CONSERVATIVE
  - MODERATE DISSIPATION -> HHT-MD, IMPACT=NO, INCREMENTATION=AGGRESSIVE
  - QUASI-STATIC -> BWE, IMPACT=NO, INCREMENTATION=AGGRESSIVE, default amplitude RAMP
- HHT-TF: alpha=-0.05, beta=0.275625, gamma=0.55. HHT-MD: alpha=-0.41421, beta=0.5, gamma=0.91421.
- ALPHA given alone: beta=(1-alpha)^2/4, gamma=1/2-alpha. Ranges: -1/2<=alpha<=0, beta>0, gamma>=1/2.
- Data line: dt_init, time period (no defaults), dt_min (0 -> min(dt_init, 1e-5*T)),
  dt_max (0 -> T/100 TF, T/10 MD, T QUASI-STATIC).
- INITIAL=NO: initial accelerations zero (first dynamic step) / carried over (previous dynamic step).

## Scope

1. Reader (`steps.py`): declare APPLICATION, TIME INTEGRATOR, ALPHA, BETA, GAMMA, IMPACT,
   INCREMENTATION, INITIAL, HALFINC SCALE FACTOR, NOHAF (HAFTOL/DIRECT already control).
   Supported: HHT-TF, HHT-MD. BWE / APPLICATION=QUASI-STATIC: refuse (gap) -- not implemented.
   IMPACT / INCREMENTATION / HALFINC / HAFTOL / NOHAF: solution technique -> `control=True`.
   Validate ranges (InputError). Store on the step a resolved integrator
   (`step.time_integration = {"name", "alpha", "beta", "gamma", "initial_acceleration": bool}`),
   *STATIC sets `{"name": "quasistatic"}`. dt_max default follows APPLICATION when given.
   *DYNAMIC, EXPLICIT stays refused.
2. Solvers (2D + 3D): `solve(step)` applies `step.time_integration` when present (like NLGEOM);
   steps without it (Python API) keep the solver's constructor setting -- non-breaking.
   Fix D3 (row-sum lumped mass = rho * integral(N_a dV), plane-strain thickness from section,
   default 1), D4, D5 (skip when initial_acceleration False), D6 (deck dynamic step: refuse a
   material with no density, naming it). D7: raise on unknown preset name.
3. Legacy adapter: forward alpha/beta/gamma to `solver_config` (ex12 keeps its own config).
4. Verification (each must fail before the fix where applicable):
   - D3: non-unit element (e.g. 2x3x5 bar, and a 2D quad of area != 1): total lumped mass == rho*V.
   - Axial bar free vibration: period vs analytic within discretisation tolerance, Newmark (alpha=0)
     energy conserved; HHT alpha<0 amplitude decay matches the HHT spectral radius for omega*dt.
   - D4: SDOF under a linearly ramped force vs closed form.
   - D5: step-applied load: a0 == M^-1 f on free DOFs.
   - Deck level: a small *DYNAMIC deck through the reader/executor runs dynamic (inertia visible),
     APPLICATION/ALPHA reach the solver; ex12 deck's step reads as DYNAMIC with alpha=-0.22.
   - Corpus: the 17 Standard *DYNAMIC decks re-baselined with reasons; `tests/inp_corpus` green.
   - Existing: test_rayleigh_damping, test_unified_time_integrator_presets, test_dynamic3d_emcs_option,
     element_contract, step_plan/bind/reader tests.

Out of scope: legacy `solver/dynamic.py` (ex12's own HHT presets), BWE, Rayleigh *DAMPING keyword
wiring (check: if a deck's *DAMPING is currently dropped silently, report it).
