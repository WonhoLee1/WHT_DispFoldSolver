"""
surface_contact3d.py
=====================
Penalty-based 3D Node-to-Analytical-Rigid-Plane Contact Constraint.

Phase 1 of dev_log/3d_contact_implementation_design_20260913.md §9:
frictionless, hard, small-sliding, node-to-surface, penalty contact
between deformable slave nodes and a single flat analytical rigid plane.

Adapted from dispsolver/constraint3d/surface_tie3d.py's projection/penalty
machinery, with the differences that make this CONTACT rather than a
bonded TIE (see the design doc §0 for the full comparison):
  - Gap is measured along the plane's NORMAL only, not the tie's full 3D
    gap vector -- contact must stay free to slide tangentially.
  - The constraint is ONE-SIDED: force is exactly zero for positive gap
    (separated), and only nonzero once gap turns negative (penetrating).
    This is a genuine non-smoothness (a gap>=0, pressure>=0,
    gap*pressure=0 complementarity condition) a bonded tie never has to
    handle. The activation check is re-evaluated fresh on every
    assemble() call (the cheapest correct active-set update, since
    solve_step already re-assembles every Newton iteration anyway) --
    NOT frozen across iterations.
  - The anchor projection (which point on the plane each slave node maps
    to, and the plane's fixed normal) is computed ONCE at construction and
    never re-solved -- this is what makes it SMALL-SLIDING, not
    finite-sliding. Do not add a reproject_deformed() method to this
    class without re-reading design doc §3 first -- adding one would
    silently turn this into finite-sliding contact, a different (also
    valid, but not Phase-1) formulation choice.
  - The main/master side (the rigid plane) has no DOFs, so unlike the
    tie's 5-node (1 slave + 4 master-vertex) stiffness pattern, contact
    here only ever touches the SLAVE node's own 3x3 diagonal block -- an
    analytical rigid plane can't deform, so there's nothing on the
    master side to distribute force onto or to differentiate the
    stiffness with respect to.
"""

from typing import Any, List, Tuple, Dict, Optional
import numpy as np

from dispsolver.constraint3d.pressure_overclosure import HardLaw


class SurfaceContactConstraint3D:
    """Penalty-based node-to-analytical-rigid-plane frictionless contact.

    Parameters
    ----------
    slave_node_ids : list of int
        Node IDs of the deformable body's candidate contact surface.
    rigid_plane_point : (3,) ndarray
        A point on the rigid plane.
    rigid_plane_normal : (3,) ndarray
        UNIT normal of the rigid plane, pointing away from the deformable
        body (i.e. toward the side the body must stay on; a slave node
        has penetrated once its signed distance along this normal goes
        negative).
    nid_to_idx : dict
        Mapping from global node ID to solver node index (0-indexed).
    coords : ndarray of shape (n_nodes, 3)
        Initial (reference) 3D coordinates of all nodes in the mesh.
    penalty_stiffness : float
        Penalty stiffness k_contact [force/length]. Abaqus's own default
        linear-penalty scaling is "10 times a representative underlying
        element stiffness" (ctc_contactconstraints_std.txt) -- callers
        should pass a value derived from the actual material/mesh
        stiffness scale of their problem; the ContactPair CAE-level
        resolver's own default (1e8) is a generic fallback, not tuned for
        any specific problem.
    strain_free_adjust : bool
        If True (design doc §7, matches Tie.adjust's semantics), any
        slave node that starts already penetrating the plane at
        construction time has its effective initial gap silently reset to
        exactly zero (a one-time geometry-consistent nudge, no resulting
        stress) rather than being treated as a real interference fit.
    name : str
        Constraint identifier.
    """

    def __init__(
        self,
        slave_node_ids: List[int],
        rigid_plane_point: np.ndarray,
        rigid_plane_normal: np.ndarray,
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        penalty_stiffness: float = 1.0e8,
        strain_free_adjust: bool = True,
        stabilization_coefficient: float = 0.0,
        augmented_lagrange: bool = False,
        law: Optional[Any] = None,
        name: str = "SURFACE_CONTACT_3D",
    ):
        self.slave_node_ids = list(slave_node_ids)
        self.point = np.asarray(rigid_plane_point, dtype=np.float64)
        self.normal = np.asarray(rigid_plane_normal, dtype=np.float64)
        n_norm = np.linalg.norm(self.normal)
        if n_norm < 1e-12:
            raise ValueError(f"SurfaceContactConstraint3D '{name}': normal vector length cannot be zero.")
        self.normal = self.normal / n_norm
        self.nid_to_idx = nid_to_idx
        self.coords = coords
        # k_contact is kept as a raw scalar (not routed through `law`)
        # specifically for the augmented-Lagrangian raw linear term below --
        # design doc sec B.4: augmented Lagrangian is restricted to HARD
        # contact only, so its math needs the UNCLAMPED k*penetration value
        # (lam + k*penetration, clamped at zero only after adding lam), not
        # a law object that already clamps to zero for h<=0 on its own.
        self.k_contact = float(penalty_stiffness)
        # Pressure-overclosure law seam (design doc sec B.6): every non-
        # augmented evaluation goes through `self.law.evaluate(h) -> (p,
        # dp/dh)`. Defaults to a HardLaw built from `penalty_stiffness`,
        # reproducing the pre-existing hard-contact-only behavior bit-for-
        # bit when no explicit law is given -- this parameter is additive,
        # not a breaking change for any existing caller.
        if augmented_lagrange and law is not None and not isinstance(law, HardLaw):
            raise ValueError(
                f"SurfaceContactConstraint3D '{name}': augmented_lagrange=True requires "
                "a HardLaw (or the default) -- Abaqus restricts the augmented Lagrange "
                "method to hard pressure-overclosure relationships only "
                "(dev_log/contact_abaqus_grade_design_20260915.md sec B.4). "
                f"Got law={type(law).__name__}."
            )
        self.law = law if law is not None else HardLaw(self.k_contact)
        # Activation stabilization (dev_log/hertz_contact_benchmark_20260913.md,
        # design doc's sec8 point 2 "contact damping" -- re-prioritized after
        # that benchmark found Newton reliably diverging the instant a NEW
        # slave node crosses into penetration, even on Phase 1's own
        # "simple" rigid-plane geometry). c_stab * k_contact scales a
        # pseudo-velocity term (see assemble()) added ONLY while a node is
        # active, opposing further approach the same way a physical dashpot
        # would -- purely a numerical aid to get the local Newton iteration
        # through the residual discontinuity at first activation, not a
        # real physical damping force (this is a quasi-static problem, no
        # real time/velocity/inertia). Default 0.0 (off) -- opt-in, matches
        # this project's "no permanent exceptions... but off-by-default new
        # mechanisms" pattern (AGENTS.md sec4.13's skip_fully_prescribed).
        self.c_stab = float(stabilization_coefficient)
        self.name = name
        # Previous-ITERATION gap per slave node, for the pseudo-velocity
        # term. Deliberately per-Newton-iteration, not per-load-step: the
        # failure this targets happens WITHIN a single increment's Newton
        # loop (the residual grows iteration-to-iteration right after
        # activation), not between converged steps -- so the "velocity"
        # that needs damping is the iteration-to-iteration gap-closing
        # rate, analogous to Abaqus's own automatic contact stabilization
        # for establishing new contact within an increment
        # (ctc_contactcontrols_std.txt), not ctc_contactdamping.txt's
        # physical relative-velocity damping (that one is for genuine
        # dynamic/inertial problems). Known limitation: a rejected/
        # cutback Newton trial does not roll this back (the solver reverts
        # u but has no hook to tell this constraint to revert its history)
        # -- acceptable for a first cut since this term only stabilizes
        # convergence, per Abaqus's own documented expectation that
        # contact stabilization should not materially change the
        # converged answer; not verified quantitatively here.
        self._prev_gap: Dict[int, float] = {}

        # Augmented Lagrange (design doc sec4, Phase 2: "reuses the same
        # penalty assembly, wrapped in an outer augmentation loop around
        # solve_step ... converge with penalty, check penetration against
        # a tolerance, 'augment' the effective contact pressure and
        # re-solve, repeat" -- ctc_contactconstraints_std.txt's own
        # description). self._lam is the persistent per-node contact-
        # pressure estimate (scalar, normal direction only -- Phase 2 is
        # still frictionless, unlike surface_tie3d.py's dead _lam_dict
        # scaffold, which is a 3-vector because a TIE constrains all 3
        # directions; contact here only ever constrains the normal one).
        # FROZEN during solve_step's own Newton iterations (that is what
        # makes this "augmented" rather than a true mixed/indefinite
        # Lagrange-multiplier system -- no new DOFs, no solver-interface
        # change, exactly the penalty-assembly reuse the design doc
        # calls for); only DynamicSolver3D.solve_step_augmented()'s OUTER
        # loop updates it, via update_augmented_multipliers(). Default
        # 0.0 for every node -- with augmented_lagrange=False (default),
        # this dict is allocated but never read, so behavior is bit-
        # identical to plain penalty (opt-in, matches this project's
        # off-by-default-new-mechanism pattern).
        # **FIXED, 2026-09-13** (dev_log/hertz_contact_benchmark_20260913.md
        # follow-up). Was: max_penetration grew GEOMETRICALLY cycle over
        # cycle (ratio ~1.422, constant, across 7 cycles), independent of
        # under-relaxation (omega 0.1-1.0 all diverged). Root cause: this
        # class's assemble() added the contact force into f_int_global with
        # the WRONG sign -- see the f_node fix and its comment in
        # assemble() below for the full derivation. That bug also silently
        # affected plain-penalty (augmented_lagrange=False) contact: it
        # converged to a state satisfying only the loose relative-residual
        # tolerance, not genuine force balance (measured nodal residual
        # ~1e3 pre-fix vs ~1e-10 post-fix on the same single-element case).
        # Post-fix, the single-element case CONVERGES: max_penetration
        # shrinks geometrically with ratio ~0.771/cycle (0.077 -> 0.059 ->
        # ... -> 0.0074 over 10 cycles), matching the first-principles 1D
        # spring re-derivation this docstring used to say was contradicted
        # by observation -- that derivation was correct; the code wasn't.
        # See tests/test_augmented_lagrangian_contact.py for the updated,
        # now-passing (not xfail) convergence assertion.
        self.augmented_lagrange = bool(augmented_lagrange)
        self._lam: Dict[int, float] = {nid: 0.0 for nid in self.slave_node_ids}

        # Strain-free initial-overclosure adjustment (design doc §7): a
        # per-node additive offset so that gap0 - offset == 0 for any node
        # that started penetrating. Zero for every node that started
        # already separated or exactly touching.
        self._gap_offset: Dict[int, float] = {}
        for s_nid in self.slave_node_ids:
            s_idx = self.nid_to_idx[s_nid]
            x0 = self.coords[s_idx]
            gap0 = float(np.dot(x0 - self.point, self.normal))
            if strain_free_adjust and gap0 < 0.0:
                self._gap_offset[s_nid] = gap0
            else:
                self._gap_offset[s_nid] = 0.0

        # NOTE: deliberately NO reproject_deformed() method -- see module
        # docstring. The plane/normal are fixed for the constraint's whole
        # life (small-sliding); only the per-node gap is recomputed live
        # inside assemble().

    def get_active_set(self, u: np.ndarray) -> frozenset:
        """Pure query (no side effects, does NOT touch self._prev_gap): the
        set of slave node ids currently penetrating (gap<0) at this u.
        Used by the solver's severe-discontinuity-iteration (SDI) handling
        (dev_log/hertz_contact_benchmark_20260913.md's teammate-research
        follow-up -- Abaqus exempts any iteration where a contact
        constraint's active status flips from the normal residual-decrease
        /convergence check, rather than trying to converge THROUGH the
        resulting residual discontinuity, per
        https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/gss/ch11s03.html).
        Deliberately separate from assemble() so the solver can call this
        cheaply (no COO triplet construction) once per iteration to detect
        activation events before deciding how to treat that iteration.
        """
        active = set()
        for s_nid in self.slave_node_ids:
            s_idx = self.nid_to_idx[s_nid]
            xs = self.coords[s_idx] + u[3 * s_idx: 3 * s_idx + 3]
            gap = float(np.dot(xs - self.point, self.normal)) - self._gap_offset[s_nid]
            if self.augmented_lagrange:
                # Augmented activation is on the AUGMENTED pressure, not
                # the raw gap -- a node with lam>0 from a previous outer
                # cycle can still be "active" (p>0) even at gap>=0 (a
                # small separation the multiplier hasn't relaxed away
                # yet). This is the standard Alart-Curnier/Uzawa
                # complementarity definition, not a Phase-1 simplification.
                p = self._lam[s_nid] + self.k_contact * (-gap)
                if p > 0.0:
                    active.add(s_nid)
            else:
                p, _ = self.law.evaluate(-gap)
                if p > 0.0:
                    active.add(s_nid)
        return frozenset(active)

    def assemble(self, u: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]]:
        """Assemble 3D penalty contact force and stiffness.

        Parameters
        ----------
        u : np.ndarray
            (3 * n_nodes,) Global 3D displacement vector.

        Returns
        -------
        f_contact : np.ndarray
            (3 * n_nodes,) Global internal contact penalty force vector.
        (rows, cols, data) : COO triplet for the stiffness contribution.
        stats : dict
            Diagnostics: max_penetration (0 if nothing active), n_active
            (how many of the candidate slave nodes are currently
            penetrating), and total_normal_force (see
            AnalyticalRigidSurface's docstring for why this substitutes
            for a solved reference-node reaction in Phase 1).
        """
        n_dof = len(u)
        f_contact = np.zeros(n_dof, dtype=np.float64)
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        max_penetration = 0.0
        n_active = 0
        total_normal_force_mag = 0.0

        for s_nid in self.slave_node_ids:
            s_idx = self.nid_to_idx[s_nid]
            xs = self.coords[s_idx] + u[3 * s_idx: 3 * s_idx + 3]

            gap = float(np.dot(xs - self.point, self.normal)) - self._gap_offset[s_nid]

            if self.augmented_lagrange:
                # p = max(0, lam + k*(-gap)) -- the standard augmented
                # Lagrangian / Alart-Curnier projection for a unilateral
                # (one-sided) constraint. lam is FROZEN here (only the
                # outer loop, update_augmented_multipliers(), changes it),
                # so within this Newton solve p is just a shifted penalty
                # law -- same d(p)/d(gap)=k_contact when active, hence the
                # SAME tangent stiffness as plain penalty below. What
                # changes across outer cycles is which gap value makes
                # p=0 (the effective "free length" shifts by lam/k), which
                # is exactly what drives true penetration toward zero as
                # lam converges to the real contact pressure.
                f_mag = self._lam[s_nid] + self.k_contact * (-gap)
                if f_mag <= 0.0:
                    continue
                penetration = -gap  # for stats only; may be negative (separated) once lam>0
                k_diag = self.k_contact
            else:
                # Pressure-overclosure law seam (design doc sec B.6):
                # `self.law` handles the one-sided activation itself
                # (returns (0,0) once inactive), covering HardLaw's old
                # `gap>=0: continue` branch exactly when c0=0, plus
                # nonlinear-penalty and every soft law through the same
                # interface.
                penetration = -gap
                f_mag, k_diag = self.law.evaluate(penetration)
                if f_mag <= 0.0:
                    continue

            n_active += 1
            if penetration > max_penetration:
                max_penetration = penetration

            if self.c_stab > 0.0:
                # Pseudo-velocity: gap change since the last assemble()
                # call (previous Newton iteration), NOT since the last
                # converged step -- see __init__ docstring. Undefined on
                # the very first call a node is active (no prior gap on
                # record), so v_pseudo=0 there (no damping on activation's
                # own first appearance, only on subsequent iterations
                # while the local Newton is still working through it).
                prev = self._prev_gap.get(s_nid)
                v_pseudo = (gap - prev) if prev is not None else 0.0
                # v_pseudo < 0 means still approaching (gap shrinking
                # further); resist that with an outward force, same sign
                # convention as the elastic penalty term above.
                f_damp_mag = -self.c_stab * self.k_contact * v_pseudo
                f_mag += f_damp_mag
                # d(f_damp)/d(u_s) = -c_stab*k_contact * d(gap)/d(u_s)
                #                  = -c_stab*k_contact * normal
                # (gap is linear in u_s along normal; prev is fixed for
                # this call), stacking onto the same diagonal block as the
                # elastic term below.
                k_diag += self.c_stab * self.k_contact

            total_normal_force_mag += f_mag
            # SIGN, 2026-09-13 (root-caused after the augmented-Lagrangian
            # outer loop was found to diverge geometrically -- see
            # update_augmented_multipliers()'s docstring history and
            # dev_log/hertz_contact_benchmark_20260913.md): f_mag is the
            # PHYSICAL contact pressure (>=0, pushes the slave node away
            # from the plane along +normal). But this function's return
            # value is accumulated directly into the SAME f_int_global
            # array as every element's own internal-force vector
            # (dynamic3d.py's assemble_system: `f_int_global += f_c`), and
            # that array uses the FEM "internal force" sign convention
            # f_int = -F_physical (confirmed against this element's own
            # tangent: d(f_int_elastic)/du > 0 while the physical restoring
            # force it represents decreases with increasing displacement --
            # i.e. f_int is the NEGATIVE of the force the body actually
            # exerts). surface_tie3d.py's f_slave = +k_tie*gap_vec looks
            # like it uses the opposite (direct, unnegated) convention, but
            # a bonded tie's physical force points OPPOSITE its own gap
            # vector (a two-sided spring pulls the slave back toward the
            # master), so that built-in sign flip already cancels the
            # -F_physical negation -- contact has no such cancellation
            # (its physical push points the SAME way its own gap-derived
            # force law does), so it needs an EXPLICIT negation here that
            # the tie gets "for free". Omitting it (the original bug) makes
            # the solved residual equation read
            # `f_int_elastic = +F_contact_physical` instead of the correct
            # `f_int_elastic = -F_contact_physical`, i.e. it solves for
            # F_cube_physical == F_contact_physical (same sign) instead of
            # the true equilibrium F_cube_physical == -F_contact_physical
            # (opposite, forces balancing). Verified: for this element,
            # d(u_z)/d(lam) was measured as -1/(K_elastic-k_contact)
            # (negative -- MORE augmented pressure produced MORE
            # penetration) before this fix, and the correct
            # +1/(K_elastic+k_contact) (positive, contracting Uzawa
            # iteration, matching standard theory) after it, on the exact
            # single-C3D8-element reproduction in
            # tests/test_augmented_lagrangian_contact.py. Also fixes Phase
            # 1 (plain penalty): before this fix, solve_step() was reaching
            # a "converged" state (by the loose relative-residual
            # tolerance) that was NOT the true elastic-vs-contact force
            # balance (measured nodal residual ~1e3, not ~0); after it,
            # the same case converges to residual ~1e-10.
            f_node = -f_mag * self.normal

            # Stiffness: d(f_node)/d(u_s) = k_diag * outer(normal, normal)
            # (only the slave node's own 3x3 block -- the rigid plane has
            # no DOFs to couple against). NOTE: this tangent sign is
            # UNCHANGED by the f_node fix above -- d(-f_mag*normal)/du_s
            # = -(-k_contact*normal)*normal = +k_contact*normal(x)normal,
            # i.e. the two negations (the f_node fix and f_mag's own
            # -k_contact*gap derivative) cancel, so k_diag stays positive
            # exactly as originally coded. Verified by finite difference
            # against this exact coded tangent (match to 1e-6).
            f_contact[3 * s_idx: 3 * s_idx + 3] += f_node
            for d1 in range(3):
                for d2 in range(3):
                    k_val = k_diag * self.normal[d1] * self.normal[d2]
                    rows.append(3 * s_idx + d1)
                    cols.append(3 * s_idx + d2)
                    data.append(k_val)

            if self.c_stab > 0.0:
                self._prev_gap[s_nid] = gap

        stats = {
            "max_penetration": max_penetration,
            "n_active": n_active,
            "n_candidates": len(self.slave_node_ids),
            # Reaction force magnitude the rigid plane would report through
            # a reference node, by Newton's third law (design doc's
            # AnalyticalRigidSurface docstring) -- the plane pushes back on
            # the body with this same magnitude, opposite sign, along
            # +normal.
            "total_normal_force": total_normal_force_mag,
        }

        return (
            f_contact,
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(data, dtype=np.float64)),
            stats,
        )

    def update_augmented_multipliers(self, u: np.ndarray, omega: float = 1.0) -> Dict[str, float]:
        """Outer augmented-Lagrangian cycle: called by
        DynamicSolver3D.solve_step_augmented() AFTER solve_step() has
        converged (equilibrium at the current, frozen self._lam), never
        during the inner Newton iterations themselves. Recomputes each
        node's augmented pressure p = max(0, lam + k*(-gap)) at the
        just-converged u and adopts it as the new lam -- this is exactly
        the Uzawa/augmented-Lagrangian update
        (ctc_contactconstraints_std.txt's "augment the effective contact
        pressure and re-solve"): each outer cycle's lam absorbs the
        previous cycle's leftover penalty-compliance penetration, so
        repeated cycles drive the TRUE penetration toward zero regardless
        of how small k_contact is, unlike plain penalty where penetration
        is permanently proportional to 1/k_contact.

        No-op (returns zeros) if augmented_lagrange=False -- callers
        should check that flag before bothering to call this in a loop.

        Returns
        -------
        dict with max_penetration (largest genuine physical penetration,
        i.e. -gap where gap<0, among nodes with p>0 -- the quantity the
        outer loop's convergence tolerance should be checked against) and
        max_lambda_change (largest |new_lam - old_lam|, a secondary
        diagnostic: once this stops shrinking, further outer cycles won't
        help either).
        """
        if not self.augmented_lagrange:
            return {"max_penetration": 0.0, "max_lambda_change": 0.0}

        max_penetration = 0.0
        max_lambda_change = 0.0
        for s_nid in self.slave_node_ids:
            s_idx = self.nid_to_idx[s_nid]
            xs = self.coords[s_idx] + u[3 * s_idx: 3 * s_idx + 3]
            gap = float(np.dot(xs - self.point, self.normal)) - self._gap_offset[s_nid]
            old_lam = self._lam[s_nid]
            p_target = max(0.0, old_lam + self.k_contact * (-gap))
            # Under-relaxed update (omega<1): a full (omega=1) Uzawa step
            # using rho=k_contact overshot in initial testing (measured
            # 2026-09-13: penetration and active-set size both GREW cycle
            # over cycle, at a FIXED prescribed BC, instead of converging
            # -- dev_log/hertz_contact_benchmark_20260913.md's Phase 2
            # section). Damping the step toward p_target is the standard
            # Uzawa remedy for an augmentation parameter that's too large
            # relative to the true constraint-direction stiffness.
            p = old_lam + omega * (p_target - old_lam)
            self._lam[s_nid] = p
            max_lambda_change = max(max_lambda_change, abs(p - old_lam))
            if p > 0.0 and -gap > max_penetration:
                max_penetration = -gap
        return {"max_penetration": max_penetration, "max_lambda_change": max_lambda_change}
