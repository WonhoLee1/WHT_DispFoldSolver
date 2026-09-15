# Precise design: contact active-set CHATTERING stabilization (2026-09-15)

**Status: design only. No `dispsolver/` code changed in this session.**
Explicit project-owner request: "접촉 stab. 기능도 함께 고려하자. 접촉의
치터링을 막는 stab. 기능을 의미한다" (also consider contact stabilization —
meaning a stabilization feature that prevents contact chattering). This is
a *different* failure mode from the one-time activation discontinuity SDI
already handles (`dispsolver/solver3d/dynamic3d.py`'s `solve_step()`,
L589-825) — see §0 for why SDI does not answer this task, then §1-§5 for
the actual mechanism.

Read in full this session, line numbers below are exact against these
reads: `dispsolver/constraint3d/surface_contact3d.py` (479 lines),
`dispsolver/constraint3d/surface_contact3d_deformable.py` (295 lines),
`dispsolver/constraint3d/pressure_overclosure.py` (172 lines),
`dispsolver/solver3d/dynamic3d.py` L560-860 (`_contact_active_set`,
`solve_step`, `solve_step_augmented`'s docstring),
`benchmark_element/reference_abaqus_docs/ctc_contactdamping.txt` (full
text), `dev_log/contact_pdass_precise_design_20260915.md` (full),
`dev_log/contact_development_report_20260915.md` (full),
`dev_log/contact_surface_to_surface_precise_design_20260915.md` (full, for
style/rigor and to confirm §5's compatibility claim about that document's
own per-node aggregation).

---

## 0. Why SDI does not answer this task, and why `c_stab` does not either

**SDI** (`dynamic3d.py` L700-711): `is_sdi = current_active_set !=
prev_active_set`, computed once per Newton iteration (`raw_iter`) by
comparing two consecutive `frozenset`s from `_contact_active_set()`
(L573-587, itself a union of every constraint's `get_active_set(u)`). When
`is_sdi` is true, that ONE iteration is exempted from the normal
convergence check and charged to a separate `max_sdi_iters` budget instead
of `max_iters` (docstring, L626-654). This is the correct, already-working
treatment for *a single flip*: it stops the solver from demanding residual
decrease across an iteration where the residual is genuinely discontinuous.

**It does nothing to stop the flip from recurring.** If a node's `x_i :=
p_i + k_contact*penetration_i(u)` (the exact quantity computed at
`surface_contact3d.py` L302/L100, before the `f_mag <= 0.0` clamp) sits
within numerical noise of zero across several consecutive iterates — which
is exactly what happens when the surrounding elastic stiffness and
`k_contact` combine to put the equilibrium gap near zero — then every one
of those iterations independently re-evaluates `get_active_set()` fresh
against the *current* trial `u`, and the set can flip ON/OFF/ON/OFF
indefinitely. Each flip is correctly SDI-exempted, but SDI provides **no
memory across flips**: it does not know or care that this is the fourth
flip of the same node, not the first. A sustained oscillation therefore
burns through `max_sdi_iters` one exemption at a time and either exhausts
the budget (reported non-convergence) or — worse — happens to land on
`equil_iters >= max_iters` by accident while `u` is still oscillating,
reporting a "converged" state that is not actually the settled contact
configuration. Confirms the task brief's framing precisely: SDI is the
right handler for a one-time event and the wrong (silent) handler for a
repeating one.

**`c_stab`** (`surface_contact3d.py` L125-157, L323-343, L406-407): a
pseudo-velocity damping force `f_damp_mag = -c_stab * k_contact *
v_pseudo`, `v_pseudo = gap - self._prev_gap[nid]` measured **between
consecutive `assemble()` calls** — and `assemble()` is called multiple
times per Newton iteration against *trial* states that may never be
accepted (once for the official residual, once for the line-search probe,
up to 6 times for Armijo halvings — the exact multiplicity
`contact_pdass_precise_design_20260915.md` §2.1 already catalogued for a
different reason). Its own docstring (L144-156) is explicit that it exists
to get "the local Newton iteration through the residual discontinuity at
first activation" — i.e. it is a *modified-Newton-style numerical aid for
convergence through one activation event*, the same problem class SDI
solves by a different route, not a debounce on the active-SET label
itself. Concretely: `c_stab` adds a force term while a node **is already
classified active** (it does not run at all inside the `continue`
branch); it has no mechanism that changes *whether* `get_active_set()`
reports a node active, so a node oscillating across the `f_mag<=0`
boundary flips its membership in `current_active_set` exactly as often
with `c_stab>0` as with `c_stab=0`. Also, per its own docstring's "known
limitation," `self._prev_gap` is not rolled back on a rejected trial —
already flagged there as acceptable for smoothing force magnitude, but it
would be a real defect if repurposed as chattering memory (see §2's
explicit non-reuse of this dict).

**Conclusion, stated as the task requires:** a genuinely different
mechanism is needed — one with memory that spans exactly the *accepted*
iterate sequence (not per-`assemble()`-call trials) and that gates the
active/inactive **label**, not just the force magnitude. §1-§2 derive it.

---

## 1. Precise characterization and programmatic detection

### 1.1 What a chattering sequence looks like in this codebase's own variables

Define, for slave node `nid`, the raw NCP argument already computed today
(`surface_contact3d.py` L302, `surface_contact3d_deformable.py` L203):

    x_i(u) := p_i + k_contact * penetration_i(u)      p_i = self._lam[nid]  (0 when AL is off)

`nid ∈ current_active_set` iff `x_i(u) > 0` (today's exact rule,
`get_active_set()` L245-251 / L176-180). **Newton-level chattering**: for
a fixed load increment, the sequence of booleans `b^(1), b^(2), ..., b^(K)`
(`b^(k) := nid in current_active_set` at `raw_iter=k` within one
`solve_step()` call) contains 2 or more sign changes, i.e. does not settle
to a constant value before `equil_iters` reaches `max_iters`.
**Increment-level chattering**: across consecutive *converged* increments
`t_1 < t_2 < ... `, the sequence of each increment's *final* `b` for the
same node flips sign more than once even though the prescribed load/BC
ramp is monotonic and smooth — i.e. the physical configuration is changing
smoothly but the discrete active-set label is not.

### 1.2 Programmatic detection — concrete, not vague

Both scales are captured by ONE rolling window per node, because the
natural sampling instant for "the accepted iterate" already exists at
`dynamic3d.py` L704 — `current_active_set = self._contact_active_set(u_k)`
— and this line runs exactly once per `raw_iter`, on `u_k`, which at that
point in the loop is *always* the last **accepted** iterate carried into
this iteration (never a discarded line-search trial: `u_k` is only ever
reassigned at the four commit points L745, L764, L802, and the initial
L666 copy of `self.u`). This is the single existing call site the sibling
docs already treat as authoritative for active-set state
(`contact_surface_to_surface_precise_design_20260915.md` §3's closing
paragraph makes the identical observation about `_contact_active_set`
being value-based and internals-agnostic).

Add, per constraint object, a bounded rolling window and a monotonic
transition counter, updated once per call to a new method
`update_gate_state(u)` (§2) invoked at that same L704 call site — **not**
inside `assemble()`, for the identical reason `update_augmented_multipliers()`
must not be called from `assemble()`
(`contact_pdass_precise_design_20260915.md` §2.1, restated in §2.3 below):

```python
# per node, in the constraint's __init__:
self._gate_history: Dict[int, collections.deque] = {
    nid: collections.deque(maxlen=self.chatter_window) for nid in self.slave_node_ids
}

# inside update_gate_state(u), after computing the new gate value g_i (sec 2):
hist = self._gate_history[nid]
hist.append(g_i)
flips = sum(1 for k in range(1, len(hist)) if hist[k] != hist[k - 1])
```

`flips >= self.chatter_flip_threshold` (default 3, over a default window
`chatter_window=6`) is the concrete, checkable chattering signal — both a
Newton-level run (many appends within one `solve_step()` call) and an
increment-level run (one append per `solve_step()` call, since the deque
persists on the constraint object across calls) are the exact same
counter; no separate bookkeeping is needed to distinguish the two scales,
because both are literally "did this node's gate flip repeatedly across
the last `chatter_window` samples taken at accepted iterates," regardless
of whether those samples span iterations or increments.

---

## 2. The remedy: a per-node Schmitt-trigger (hysteresis) gate on the active-set LABEL, with adaptive band widening on detected chattering

### 2.1 Why hysteresis, and why it must gate the label, not smooth the force

A dead-band / hysteresis switch (Schmitt trigger) is the standard
relay-chattering remedy for any system that makes a discrete decision from
a continuously-varying, noisy signal sitting near the decision boundary —
the general mechanism `c_stab` is not (§0). This project already has a
close conceptual precedent for "two different thresholds for the same
switch" in Abaqus's own nonlinear penalty law
(`pressure_overclosure.py::NonlinearPenaltyLaw`, breakpoints `e`/`d`) —
but that ramps *force magnitude* smoothly once active; it does not touch
*when* the label itself turns on/off, and a `NonlinearPenaltyLaw`/soft-law
contact still switches its `p>0` boundary at a single sharp point
(`hp<=0.0: return 0.0,0.0`, e.g. `pressure_overclosure.py` L65/L94/L135).
**Scope decision, stated explicitly**: this design therefore targets
exactly the sharp linear switch — `HardLaw` and `augmented_lagrange=True`
contact, where `x_i(u)` is the literal argument of the label decision —
because the graduated laws already reduce label-flip frequency by
construction (a smoothly ramping force rarely produces a sharp
`p<=0 ↔ p>0` relabeling at the *same* rate a hard switch does). If a future
benchmark shows soft/nonlinear-law contact chattering too, the identical
mechanism below applies unchanged (it only reads `x_i`/the law's own
raw pre-clamp argument), but it is not built for that case now, matching
§5's staging discipline.

**Why gating the label is not enough by itself, and the fix must also
touch the tangent that Newton actually sees**: relabeling
`current_active_set` without changing what `assemble()` puts into `K_g`/
`f_int` would only hide the oscillation from SDI's bookkeeping while the
real cause — the tangent stiffness `k_diag` discontinuously dropping from
`k_contact` to `0` exactly at the boundary Newton keeps re-crossing —
would still drive `u` back and forth. So the gate must also control what
`assemble()` assembles, which is the point of §2.2.

### 2.2 The exact formulas, in this codebase's variables

New constructor parameters (both classes; default OFF, matching
`c_stab`'s and `augmented_lagrange`'s own "off-by-default new mechanism"
convention, `AGENTS.md` §4.13):

```python
chatter_stabilization: bool = False
hysteresis_band: float = 0.0        # h0, FORCE units (same units as k_contact*penetration, i.e. the
                                     # same units as x_i / p_i / "total_normal_force" in the stats dict --
                                     # NOT a gap/length quantity), caller-supplied like k_contact itself
                                     # ("callers should pass a value derived from the actual material/
                                     # mesh stiffness scale of their problem", surface_contact3d.py L62-69)
chatter_window: int = 6
chatter_flip_threshold: int = 3
chatter_escalation_growth: float = 4.0
chatter_escalation_max: float = 16.0
chatter_escalation_decay: float = 0.5
```

Per-node persistent state (allocated in `__init__`, alongside `self._lam`):

```python
self._gate_active: Dict[int, bool] = {nid: False for nid in slave_node_ids}   # persistent Schmitt state g_i
self._escalation:  Dict[int, float] = {nid: 1.0 for nid in slave_node_ids}    # band multiplier, sec2.4
self._gate_history: Dict[int, deque] = {nid: deque(maxlen=chatter_window) for nid in slave_node_ids}
```

(For `DeformableSurfaceContactConstraint3D`, build these from `self.pairs`
after `_build_pairs()`, exactly the same reasoning
`contact_pdass_precise_design_20260915.md` §3 item 1 already gives for
that class's own `self._lam`.)

**New method `update_gate_state(u)`** (mutating; called exactly once per
`raw_iter` from `dynamic3d.py`, §3 — never from `assemble()`):

```python
def update_gate_state(self, u: np.ndarray) -> None:
    if not self.chatter_stabilization:
        return
    for nid in self.slave_node_ids:                 # or `for nid, *_ in self.pairs:` in the deformable class
        gap, ... = <this class's own current gap computation, unchanged>
        penetration = -gap
        x_i = self._lam[nid] + self.k_contact * penetration      # exactly assemble()'s own f_mag, pre-clamp
        h_i = self.hysteresis_band * self._escalation[nid]
        g = self._gate_active[nid]
        if not g and x_i > +h_i:
            g = True
        elif g and x_i < -h_i:
            g = False
        # else: inside the dead band -- no change (this IS the debounce)
        self._gate_active[nid] = g

        hist = self._gate_history[nid]
        hist.append(g)
        flips = sum(1 for k in range(1, len(hist)) if hist[k] != hist[k - 1])
        if flips >= self.chatter_flip_threshold:
            self._escalation[nid] = min(self._escalation[nid] * self.chatter_escalation_growth,
                                         self.chatter_escalation_max)
        else:
            self._escalation[nid] = max(1.0, self._escalation[nid] * self.chatter_escalation_decay)
```

**`get_active_set(u)`** — add one branch at the top (both classes),
existing instantaneous behavior otherwise completely unchanged:

```python
def get_active_set(self, u):
    if self.chatter_stabilization:
        return frozenset(nid for nid in self.slave_node_ids if self._gate_active[nid])
    # ... existing code, byte-for-byte unchanged ...
```

**`assemble(u)`** — the active branch (`surface_contact3d.py` L290-317,
`surface_contact3d_deformable.py` L197-208) gets one guard inserted before
today's `if f_mag <= 0.0: continue`:

```python
if self.chatter_stabilization:
    if not self._gate_active[s_nid]:
        continue                                    # fully inactive: zero force, zero stiffness, same as today
    f_mag_raw = self._lam[s_nid] + self.k_contact * penetration     # == x_i above
    f_mag = max(0.0, f_mag_raw)         # physical force cannot pull; bounded by construction, see sec2.3
    k_diag = self.k_contact             # ENGAGED throughout the dead band -- this, not the label change
                                         # alone, is what removes the oscillation-perpetuating mechanism
else:
    # ... today's exact existing code (law.evaluate / AL branch), byte-for-byte unchanged ...
```

Everything downstream of `f_mag`/`k_diag` (the sign-fixed `f_node =
-f_mag*self.normal`, the 3x3 diagonal stiffness scatter, the 5-node
stencil and rank pattern in the deformable class) is **completely
unchanged** — exactly the same "the stencil doesn't know or care which
branch produced the scalars" property
`contact_pdass_precise_design_20260915.md` §3 item 3 already established
for the AL branch applies here unchanged.

### 2.3 Why the clamped `f_mag=0` inside the dead band is a bounded, honest approximation, not a hidden defect

While `g_i` is True and `x_i` has dipped below 0 but not yet below `-h_i`
(the dead band's lower half), `f_mag_raw < 0` is clamped to exactly `0`.
This is not unbounded: by construction the gate only stays ON for `x_i >
-h_i`, so the clamped discrepancy `|f_mag - f_mag_raw|` is bounded by
`h_i` — a quantity the caller explicitly chose as "how much mislabeling
error is acceptable to buy chattering suppression," the same kind of
explicit user-facing tradeoff `NonlinearPenaltyLaw`'s `e`/`d` breakpoints
already are. At a genuinely converged increment this discrepancy vanishes
in the same sense `c_stab`'s docstring already claims for its own damping
term ("contact stabilization should not materially change the converged
answer") — because a converged state has no rolling gate history left to
decay through; `x_i` sits wherever the true equilibrium places it, and the
gate has already settled by the time equilibrium is reached (§4's
falsifiable test measures this directly, not by assumption).

### 2.4 Escalation (adaptive band widening) — Stage 2, described here for completeness, staged separately in §5

If a node still flips `>= chatter_flip_threshold` times within the last
`chatter_window` accepted samples even with the *default* `h_i =
hysteresis_band`, `self._escalation[nid]` grows geometrically
(`chatter_escalation_growth`, capped at `chatter_escalation_max`),
widening that SPECIFIC node's dead band on subsequent calls. Once a node
stops flipping, `self._escalation[nid]` decays back toward 1 (never below
it), so a node that settles reverts to the small default band rather than
carrying a permanently oversized one. This is a local, per-node adaptive
response — exactly the "escalating response" remedy class named in the
task brief — and is orthogonal to which node it applies to: a
well-behaved neighbor's `h_i` is untouched.

---

## 3. Exact wiring into `dynamic3d.py`

One new private helper, mirroring `_contact_active_set()`'s own style
(L573-587):

```python
def _update_contact_chatter_state(self, u: np.ndarray) -> None:
    for c in self.constraints:
        if hasattr(c, "update_gate_state"):
            c.update_gate_state(u)
```

One call site, inserted immediately **before** L704's existing
`current_active_set = self._contact_active_set(u_k)`:

```python
self._update_contact_chatter_state(u_k)          # NEW -- updates self._gate_active on the accepted iterate
current_active_set = self._contact_active_set(u_k)   # unchanged -- now reads the just-updated gate when
                                                        # chatter_stabilization=True, or the old instantaneous
                                                        # rule when it's False (byte-identical old behavior)
is_sdi = current_active_set != prev_active_set    # unchanged
```

No other line in `solve_step()`/`solve_step_augmented()` changes. Ordering
within one `raw_iter` is therefore, precisely: (1) assemble at `u_k`
(L683) — uses the CURRENT `self._gate_active`/`self._lam`, both left over
from the previous `raw_iter`'s commit; (2) gate update (new, reads this
iteration's `u_k`, writes `self._gate_active`); (3) SDI check (reads the
just-updated gate via `get_active_set`); (4) later in the same `raw_iter`,
once a trial is accepted, `update_augmented_multipliers()` runs at its
four existing commit points (L748, L768, L806-808, and implicitly not at
the two fast-exit branches L716-720/L732-736 — an existing asymmetry in
the AL wiring, not something this design introduces or needs to fix) and
writes `self._lam` for the *next* `raw_iter`'s step (1). This is a single,
well-defined, non-circular data dependency: the gate this iteration reads
whatever `self._lam` was at the end of the *previous* iteration, and the
gate this iteration produces is what `assemble()` will use as this
iteration's active/inactive decision on the *next* pass through the loop
only via `_contact_active_set`'s bookkeeping — `assemble()` itself
(step 1, L683) is called ONE line before the gate update, so within a
single `raw_iter` the force/stiffness `assemble()` computes at step (1)
and the gate SDI reads at step (3) can momentarily disagree by exactly one
iteration's lag on a genuine transition. This lag is intentional and
harmless: it is the same one-iteration lag `is_sdi`'s own detection
already has relative to `assemble()`'s force computation (SDI, too, only
finds out about a flip after the fact, on the very next comparison) — no
new class of staleness is introduced.

---

## 4. Compatibility with PDASS/AL and SDI — restated precisely, not just claimed

**With SDI**: unmodified. `_contact_active_set()`, `is_sdi`'s comparison,
and the `equil_iters`/`max_sdi_iters` budget split are all byte-identical.
The only effect is that `current_active_set` changes less often when
`chatter_stabilization=True`, so `is_sdi` fires less often for the SAME
physical run — fewer SDI exemptions consumed on spurious flips, leaving
the budget for genuine transitions. This is a strict reduction in SDI
*load*, not a change to SDI's own logic, matching the task's "does not
fight or duplicate" requirement directly.

**With PDASS/AL** (`contact_pdass_precise_design_20260915.md`): `x_i =
self._lam[nid] + k_contact*penetration_i(u)` is **the exact same quantity**
PDASS's `update_augmented_multipliers()` already computes as
`p_target = max(0, lam_old + k_contact*(-gap))` (its own L464/L288).
`update_gate_state()` does not duplicate that computation's *purpose*
(driving `self._lam` toward the true contact pressure) — it only reads
`self._lam` (never writes it) to decide the label. The two mutating
methods (`update_gate_state`, `update_augmented_multipliers`) are called
at different points in the same `raw_iter` (§3) and never race: neither
writes a field the other reads within the same call, only across
iterations, and always in the same fixed order. When `augmented_lagrange`
is `False` (plain HardLaw contact), `self._lam[nid]` stays permanently 0
and `x_i` reduces to `k_contact*penetration_i(u)` — the mechanism is fully
defined and useful for plain penalty contact too, not only the AL path.

**With `c_stab`**: orthogonal and composable, not a replacement (§0
already established they solve different problems). A constraint can
legally have both `c_stab>0` and `chatter_stabilization=True` — `c_stab`'s
pseudo-velocity term still only executes inside the `if not
self._gate_active[s_nid]: continue` survivors, damping residual force
noise *while* a node stays gated-active across sub-iterations of one
activation event; `chatter_stabilization` decides whether that gate is
open at all across the run's whole iterate/increment sequence.

**With `DeformableSurfaceContactConstraint3D`'s modified-Newton normal**:
no interaction, for the same reason `contact_pdass_precise_design_20260915.md`
§3's closing paragraph already gives for PDASS's own AL branch — the gate
only changes which scalar `(f_mag, k_diag)` pair gets computed and
whether it's computed at all; it says nothing about how `gap_i(u)`'s own
`u`-dependence is differentiated (frozen normal or not). The two
simplifications stack unchanged.

**With `contact_surface_to_surface_precise_design_20260915.md`'s
segment-to-segment design**: that document's §2.4/§3 already reduce a
node's segment-summed contribution to one scalar `g_tilde[nid]` *before*
any activation decision is made, and its own §3 states
`get_active_set()`'s per-node decision needs zero changes to compose with
`_contact_active_set()`/SDI. This design's gate plugs in at exactly that
same single-scalar point (`x_i` substitutes `g_tilde[nid]` in for
`penetration_i(u)` unchanged) — no interaction to derive beyond noting the
plug-in point is identical.

---

## 5. Verification plan — falsifiable, and staged/deferred scope

### 5.1 Constructing a case that chatters without the fix

Take the existing single-C3D8-element fixture already used by
`tests/test_augmented_lagrangian_contact.py`'s
`_build_cube_contact(...)`. Instead of a fixed prescribed displacement,
choose a prescribed displacement such that the *analytic* series
equilibrium penetration (`u_elastic_stiffness`, `k_contact` known in
closed form for one element, same 1D-spring re-derivation
`surface_contact3d.py`'s own history comment already uses) lands within
`1e-4 * (typical element-scale displacement)` of exactly `gap=0` — i.e.
deliberately construct the boundary-straddling case, not a comfortably
separated or comfortably penetrating one. Run `solve_step()` once with
`debug_sdi=True` and `chatter_stabilization=False` (today's behavior);
count `is_sdi=True` occurrences from the printed `[SDI]` lines (already
existing instrumentation, L708-710 — no new logging needed for this
baseline measurement).

**Falsifiable prediction 1 (baseline exists)**: the boundary-straddling
case shows `>= 3` `is_sdi=True` occurrences for the same node within a
single `solve_step()` call's iteration budget (i.e. it genuinely
chatters, not merely SDI-exempts once) — if it does not, the fixture needs
to be tuned closer to the true degenerate equilibrium before the
comparison below means anything.

### 5.2 The comparison

Re-run the identical fixture with `chatter_stabilization=True`,
`hysteresis_band` set to roughly `0.01 * k_contact * (typical element
edge length)` (a small fraction of the force scale, matching the "small,
bounded tradeoff" framing of §2.3), default `chatter_window`/
`chatter_flip_threshold`.

**Falsifiable prediction 2**: the same node's `is_sdi=True` count over the
same run drops to `<= 1` (the single genuine settling transition, or 0 if
it settles fully inside the band and never crosses `+h_i`), and the run
converges (`equil_iters` reaches the normal convergence branch) within a
comparable or smaller total `raw_iter` count than the un-stabilized run's
`max_raw_iters` exhaustion. **Prediction 3**: the final converged
`max_penetration`/nodal force-balance residual (same metric
`tests/test_augmented_lagrangian_contact.py` already checks to `~1e-10`)
differs from the un-stabilized-but-converged case (constructed at a
nearby, non-degenerate load level where both methods converge cleanly) by
no more than `hysteresis_band` in force units — i.e. §2.3's bounded-error
claim is measured, not assumed.

**Negative-result sign to watch for**: if prediction 2 fails (chattering
persists even with the gate on), the specific things to check first, in
order: (a) `update_gate_state()` is actually being called before
`_contact_active_set()` at the new L704-adjacent site, not after (an
off-by-one on the call order would make the gate perpetually one iteration
stale in the wrong direction); (b) `hysteresis_band` is too small relative
to the actual force-scale noise in this fixture (check the escalation
counter, §2.4 — if `self._escalation[nid]` is pinned at
`chatter_escalation_max` and still flipping, the static band is
fundamentally too small for this case's noise amplitude, not a logic bug).

### 5.3 Second fixture: increment-level chattering

Ramp a prescribed BC across ~20 increments such that the equilibrium gap
crosses zero smoothly somewhere around increment 10-12 (a monotonic ramp
through the boundary-straddling configuration of §5.1, rather than sitting
there for one static solve). Record the *converged* active-set membership
of the boundary node once per increment. **Falsifiable prediction 4**:
without the gate, this per-increment sequence shows `>= 2` sign changes
across increments 8-15 despite the monotonic ramp (the increment-level
chattering signature); with the gate, it shows exactly 1 (the genuine
crossing) or is delayed/advanced by at most one increment relative to the
un-stabilized case's *first* crossing (a small, bounded phase lag from the
dead band — not a large discrepancy).

### 5.4 Staged/deferred scope, stated honestly

**Stage 1 (buildable now, the minimal complete mechanism)**: static
per-constraint `hysteresis_band` (uniform across all nodes of one
constraint, not per-node), `chatter_stabilization` opt-in flag,
`self._gate_active` persistent dict, `update_gate_state()`, the one
`assemble()`/`get_active_set()` guard each, and the one `dynamic3d.py`
call site (§3). Ported to `SurfaceContactConstraint3D` first (simpler
class, direct dot-product gap), then near-verbatim to
`DeformableSurfaceContactConstraint3D` exactly as
`contact_pdass_precise_design_20260915.md` §3 already did for the AL
branch. **No rolling-window flip counter or escalation in Stage 1** — a
uniform static band is the standard, simplest form of hysteresis
stabilization and should be measured against §5.1-§5.3 before deciding
whether per-node adaptivity is actually needed.

**Stage 2 (escalation, §2.4) — build only if Stage 1 measurably fails
§5's predictions on some node while succeeding on most.** The rolling
`deque`/flip-counter/escalation-factor bookkeeping is real added
complexity (state that must be maintained correctly per node, decayed
correctly, capped correctly) and this project's own standing discipline
(`AGENTS.md`, and this session's sibling docs' own staging sections) is to
not build it speculatively. If Stage 1's single global band already
suppresses chattering to `<=1` flip on every constructed test case, Stage
2 should not be built at all — flag it as available, not default to
implementing it.

**Deferred entirely, not flagged as near-term work:**
- Per-node (rather than per-constraint) *default* `hysteresis_band` sizing
  from local element stiffness — today's design requires one caller-supplied
  scalar per constraint, matching `k_contact`'s own existing "caller
  supplies, no silent auto-guess" convention; an auto-derived default
  (analogous to `k_contact`'s own `k_ref` auto-scaling mentioned in
  `dev_log/contact_development_report_20260915.md` §4) is a real,
  separable follow-on, not attempted here.
- Load-step-size-adaptive banding (shrinking `hysteresis_band` as `dt`
  shrinks, so a fine-stepped ramp gets a tighter band than a coarse one) —
  a genuine Abaqus-grade refinement, not built without a concrete case
  showing the fixed band is wrong at both step-size extremes
  simultaneously.
- Tangential/frictional chattering — both contact classes are explicitly
  frictionless today (module docstrings); there is no tangential
  stick-slip switch to stabilize yet.
- Any change to `contact_surface_to_surface_precise_design_20260915.md`'s
  segment/quadrature machinery — §4 above already establishes the plug-in
  point is unchanged; no new design content needed there.

---

## 6. Summary of files touched if implemented (none touched this session)

- `dispsolver/constraint3d/surface_contact3d.py`: add `chatter_stabilization`/
  `hysteresis_band`/`chatter_window`/`chatter_flip_threshold`/
  `chatter_escalation_growth`/`chatter_escalation_max`/`chatter_escalation_decay`
  constructor params; `self._gate_active`/`self._escalation`/
  `self._gate_history` state; `update_gate_state()`; one guard each in
  `get_active_set()` and `assemble()`.
- `dispsolver/constraint3d/surface_contact3d_deformable.py`: identical
  near-verbatim port, substituting `self.pairs`/`_current_gap_and_normal()`
  for the rigid sibling's node list/direct dot product.
- `dispsolver/solver3d/dynamic3d.py`: one new private method
  `_update_contact_chatter_state()`; one call inserted immediately before
  the existing L704 `current_active_set = self._contact_active_set(u_k)`
  line in `solve_step()`. No signature changes to any public method.
- New test fixtures per §5.1/§5.3 (not written this session).
