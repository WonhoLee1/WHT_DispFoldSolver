"""Interaction definitions for CAE model hierarchy (Abaqus parity).

Mirrors real Abaqus's own separation of the "Interaction" module (contact:
ContactProperty, ContactPair/SurfaceToSurfaceContactStd) from the plain
kinematic "Constraint" module (Tie, Coupling, MPC — see constraint.py) even
though both eventually reduce to equations assembled into the same global
system. Field names deliberately echo constraint.py's Tie class (master/
slave, adjust) so contact reads as a sibling of Tie, not an unrelated
one-off.

Phase 1 scope only (dev_log/3d_contact_implementation_design_20260913.md
§9): frictionless, hard, small-sliding, node-to-surface, penalty contact
between one deformable body and one analytical rigid plane. Deliberately
NOT implemented here: friction, finite-sliding, deformable-vs-deformable,
augmented Lagrange, softened (linear/exponential/tabular) contact — see
that design doc for the full phased plan and the reasoning behind every
choice below.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, Any, Sequence, Dict
import warnings
import numpy as np


@dataclass
class ContactProperty:
    """Interaction property (*SURFACE INTERACTION in Abaqus).

    normal_behavior: "HARD" | "SOFT_LINEAR" | "SOFT_EXPONENTIAL" |
    "SOFT_TABULAR" (dev_log/contact_abaqus_grade_design_20260915.md Part B).
    Deformable-vs-deformable contact, finite-sliding, and friction remain
    out of scope (design doc's own Part A/B are both still node-to-
    analytical-rigid-plane only) -- friction=None means frictionless; a
    nonzero value is accepted but has no runtime effect (no tangential-
    plane enforcement path exists in SurfaceContactConstraint3D at all).

    --- Soft contact: scale-based auto-derivation (design doc sec B.7) ---
    The PRIMARY way to configure a soft law is `softness_scale` +
    `allowable_penetration` -- both auto-derive from the SAME material/
    geometry-based reference stiffness (`k_ref`) hard contact already
    uses, so a caller never has to type an absolute pressure/stiffness
    number they have no principled basis for choosing:
      - `softness_scale` (dimensionless, default 1.0): 1.0 reproduces
        hard contact's own auto-stiffness EXACTLY; <1.0 proportionally
        softer, >1.0 proportionally stiffer, at every point on the curve.
      - `allowable_penetration` (length, default None -> 1% of the
        contact surface's characteristic element length): how far the
        surface can penetrate before the reaction approaches hard-
        contact-like stiffness (SOFT_EXPONENTIAL's ramp length; a slack
        tolerance for SOFT_LINEAR, which has no independent ramp to
        control).
    The absolute fields below (`linear_stiffness`, `exponential_c0`,
    `exponential_p0`, `tabular_overclosure`/`tabular_pressure`) remain
    available as an explicit opt-out override for a caller who already
    knows the number they want -- setting any of them takes precedence
    over `softness_scale`/`allowable_penetration` for that law (a warning
    is issued if both are set non-default at once, matching this
    project's "explicit warning over silent precedence" convention).

    --- Penalty hardening (design doc Part A) ---
    `penalty_form="NONLINEAR"` switches HARD contact from a single linear
    penalty slope to Abaqus's 4-region nonlinear penalty (sec A.2) --
    small initial stiffness at activation, ramping to a stiff final
    stiffness only after real penetration builds; this is the concrete,
    measured fix for the Hertz-benchmark activation-divergence finding
    in dev_log/hertz_contact_benchmark_20260913.md.
    `stiffness_scale_factor` (default 1.0) is a final multiplier applied
    after auto/override resolution, mirroring Abaqus's own
    `*CONTACT CONTROLS, STIFFNESS SCALE FACTOR`.
    """
    name: str
    normal_behavior: str = "HARD"
    friction: Optional[float] = None

    # --- scale-based auto-derivation, the DEFAULT path (sec B.7) ---
    softness_scale: float = 1.0
    allowable_penetration: Optional[float] = None

    # --- explicit absolute opt-out overrides (take precedence over the
    #     scale knobs above when set) ---
    linear_stiffness: Optional[float] = None
    clearance_at_zero_pressure: float = 0.0
    exponential_c0: Optional[float] = None
    exponential_p0: Optional[float] = None
    tabular_overclosure: Optional[Sequence[float]] = None
    tabular_pressure: Optional[Sequence[float]] = None

    # --- penalty hardening (Part A) ---
    penalty_form: str = "LINEAR"          # "LINEAR" | "NONLINEAR"
    stiffness_scale_factor: float = 1.0


@dataclass
class AnalyticalRigidSurface:
    """A rigid contact surface defined by a geometric primitive rather than
    a mesh (*RIGID SURFACE / analytical rigid body in Abaqus).

    Phase 1 only supports a flat plane (design doc §2: "a flat analytical
    rigid plane is the only rigid-surface primitive that needs to exist
    for Phase 1 -- do not build a general rigid-surface library
    speculatively"). point/normal define the plane; normal points AWAY
    from the deformable body (i.e. toward the side the body must stay on).

    reference_node is accepted for API compatibility with a future moving
    rigid body (mirrors RigidBody.ref_point in constraint.py) but Phase 1's
    plane is always geometrically fixed -- no reference-node kinematics are
    implemented. Reaction force is instead read directly off the contact
    constraint's own assemble() stats (the sum of penalty forces applied
    to the slave nodes, negated by Newton's third law -- see
    SurfaceContactConstraint3D.assemble()'s "total_normal_force" entry),
    which is the honest Phase-1-scale equivalent of reading a reference
    node's reaction force without needing to actually solve for one.
    """
    name: str
    point: Sequence[float]
    normal: Sequence[float]
    reference_node: Optional[int] = None

    def unit_normal(self) -> np.ndarray:
        n = np.asarray(self.normal, dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError(f"AnalyticalRigidSurface '{self.name}': normal vector length cannot be zero.")
        return n / norm


def _resolve_node_ids(region: Union[str, Any], model: Any = None) -> list:
    """Resolve a master/slave region reference into a concrete list of node
    IDs. Mirrors constraint.py's RigidBody.get_slave_node_ids resolver
    pattern exactly: accepts a GeneralSet/Surface (anything with a
    get_nodes() method), a raw list/tuple/set/ndarray of node IDs, or a
    string name to look up via model.get_set(...).
    """
    if region is None:
        return []
    if hasattr(region, "get_nodes"):
        return list(region.get_nodes(include_elements=True))
    if isinstance(region, (list, tuple, set, np.ndarray)):
        return list(region)
    if isinstance(region, str) and model is not None and hasattr(model, "get_set"):
        s = model.get_set(region)
        if s is not None and hasattr(s, "get_nodes"):
            return list(s.get_nodes(include_elements=True))
    return []


# First-order hex (C3D8-family) face -> local node index map, matching
# dispsolver/model/set.py's ElementFace docstring convention exactly
# (0=Bottom(0,1,2,3), 1=Top(4,5,6,7), 2=Front(0,1,5,4), 3=Right(1,2,6,5),
# 4=Back(2,3,7,6), 5=Left(3,0,4,7)). Deformable-vs-deformable contact's
# master surface is scoped to first-order hex meshes for now, matching
# every other contact/tie example in this codebase (AGENTS.md sec4.16's
# own note that second-order-face ambiguity doesn't apply here yet).
_HEX_FACE_LOCAL_NODES = {
    0: (0, 1, 2, 3), 1: (4, 5, 6, 7), 2: (0, 1, 5, 4),
    3: (1, 2, 6, 5), 4: (2, 3, 7, 6), 5: (3, 0, 4, 7),
}


def _resolve_master_faces(region: Union[str, Any], mesh: Any, model: Any = None) -> list:
    """Resolve a deformable-vs-deformable ContactPair's `master` into a
    concrete list of Quad4 face node-ID tuples, mirroring
    `_resolve_node_ids`'s own resolver pattern: accepts a raw list of
    4-tuples already, a `dispsolver.model.set.Surface` (anything with a
    get_faces() method), or a string name looked up via model.get_set(...).
    """
    if region is None:
        return []
    if isinstance(region, (list, tuple)) and region and isinstance(region[0], (list, tuple)):
        return [tuple(f) for f in region]

    faces = None
    if hasattr(region, "get_faces"):
        faces = region.get_faces(exterior_only=True)
    elif isinstance(region, str) and model is not None and hasattr(model, "get_set"):
        s = model.get_set(region)
        if s is not None and hasattr(s, "get_faces"):
            faces = s.get_faces(exterior_only=True)
    if not faces:
        return []

    result = []
    for ef in faces:
        elem = mesh.elements[ef.element_id]
        local = _HEX_FACE_LOCAL_NODES.get(ef.face_id)
        if local is None:
            continue
        result.append(tuple(elem.node_ids[i] for i in local))
    return result


@dataclass
class ContactPair:
    """A single contact interaction between a main and a secondary surface
    (*CONTACT PAIR in Abaqus). Modern CAE terminology: 'main' and 'secondary'
    (with full backward compatibility for legacy 'master' and 'slave').
    """
    name: str
    interaction_property: ContactProperty
    # main/master: an AnalyticalRigidSurface (rigid-plane contact, the
    # original Phase 1 scope), OR -- for deformable-vs-deformable contact
    # (design doc sec9 Phase 2) -- anything _resolve_master_faces accepts:
    # a Surface (dispsolver.model.set), a raw list of Quad4 face node-ID
    # tuples, or a string set name.
    main: Optional[Union[str, "AnalyticalRigidSurface", Any]] = None
    secondary: Optional[Union[str, Any]] = None
    master: Optional[Union[str, "AnalyticalRigidSurface", Any]] = None
    slave: Optional[Union[str, Any]] = None
    sliding: str = "SMALL"
    constraint_enforcement: str = "PENALTY"
    adjust: bool = True

    def __post_init__(self):
        if self.main is None and self.master is not None:
            self.main = self.master
        elif self.master is None and self.main is not None:
            self.master = self.main

        if self.secondary is None and self.slave is not None:
            self.secondary = self.slave
        elif self.slave is None and self.secondary is not None:
            self.slave = self.secondary

        if self.main is None or self.secondary is None:
            raise ValueError("ContactPair requires both 'main' (or 'master') and 'secondary' (or 'slave').")

    def build_runtime_constraint(
        self,
        mesh: Any,
        nid_to_idx: Dict[int, int],
        model: Any = None,
        penalty_stiffness: Optional[float] = None,
        stabilization_coefficient: float = 0.0,
        materials: Optional[Dict[int, Any]] = None,
    ) -> Any:
        """Resolve this CAE-level contact definition into a runtime
        SurfaceContactConstraint3D, ready to append to
        DynamicSolver3D.constraints. Mirrors RigidBody.get_slave_node_ids's
        resolver pattern: this is (per this session's own review) the
        FIRST real CAE-object-to-runtime-constraint resolver in this
        codebase -- SurfaceTieConstraint3D today is still always built
        directly from raw node/face arrays in example scripts, with no
        equivalent Tie-object resolver, so don't assume this pattern is
        already established elsewhere; it's being established here.
        """
        if self.sliding != "SMALL":
            raise NotImplementedError(
                f"ContactPair '{self.name}': sliding='{self.sliding}' not implemented -- "
                "Phase 1 only supports small-sliding contact (design doc §3). "
                "Do not silently fall back to small-sliding for a different request."
            )
        if self.constraint_enforcement not in ("PENALTY", "AUGMENTED_LAGRANGE"):
            raise NotImplementedError(
                f"ContactPair '{self.name}': constraint_enforcement='{self.constraint_enforcement}' "
                "not implemented -- only 'PENALTY' (Phase 1, design doc §4) and "
                "'AUGMENTED_LAGRANGE' (Phase 2, design doc §4) are supported."
            )
        is_rigid_master = isinstance(self.master, AnalyticalRigidSurface)
        if not is_rigid_master:
            # Deformable-vs-deformable (design doc sec9 Phase 2), first
            # cut: HARD+LINEAR penalty only, via DeformableSurfaceContactConstraint3D.
            # Nonlinear penalty, soft laws, and augmented Lagrangian are
            # not yet wired for a moving master surface -- the law-
            # resolution logic below is already generic (built from
            # `self.master` only at the very end), so extending this is a
            # smaller follow-on than the initial mechanism, not a rewrite.
            if self.constraint_enforcement != "PENALTY":
                raise NotImplementedError(
                    f"ContactPair '{self.name}': deformable-vs-deformable contact only supports "
                    f"constraint_enforcement='PENALTY' -- got '{self.constraint_enforcement}' "
                    "(augmented Lagrangian's Uzawa update is not yet generalized to a moving, "
                    "shape-function-weighted master stencil)."
                )

        prop = self.interaction_property
        if not is_rigid_master and (prop.normal_behavior != "HARD" or prop.penalty_form != "LINEAR"):
            raise NotImplementedError(
                f"ContactPair '{self.name}': deformable-vs-deformable contact only supports "
                f"normal_behavior='HARD' with penalty_form='LINEAR' for now -- got "
                f"normal_behavior='{prop.normal_behavior}', penalty_form='{prop.penalty_form}' "
                "(nonlinear penalty and soft laws are not yet wired for a moving master surface)."
            )
        valid_behaviors = ("HARD", "SOFT_LINEAR", "SOFT_EXPONENTIAL", "SOFT_TABULAR")
        if prop.normal_behavior not in valid_behaviors:
            raise NotImplementedError(
                f"ContactPair '{self.name}': normal_behavior='{prop.normal_behavior}' not implemented -- "
                f"supported values are {valid_behaviors} "
                "(dev_log/contact_abaqus_grade_design_20260915.md Part B)."
            )
        # sec B.4: Abaqus restricts augmented Lagrangian to HARD contact
        # only ("The augmented Lagrange method ... applies only to hard
        # pressure-overclosure relationships" -- ctc_contactconstraints_std.txt).
        if self.constraint_enforcement == "AUGMENTED_LAGRANGE" and prop.normal_behavior != "HARD":
            raise NotImplementedError(
                f"ContactPair '{self.name}': augmented Lagrangian enforcement only applies to "
                f"HARD contact (Abaqus restriction, design doc sec B.4) -- got normal_behavior="
                f"'{prop.normal_behavior}'. Use constraint_enforcement='PENALTY' for soft laws "
                "(Abaqus's own direct method is the only enforcement for softened contact)."
            )
        if self.constraint_enforcement == "AUGMENTED_LAGRANGE" and prop.penalty_form == "NONLINEAR":
            raise NotImplementedError(
                f"ContactPair '{self.name}': augmented Lagrangian with penalty_form='NONLINEAR' "
                "is not implemented -- the augmented-Lagrangian raw term "
                "(SurfaceContactConstraint3D's `lam + k_contact*(-gap)`) assumes a single linear "
                "stiffness. Use penalty_form='LINEAR' (the default) with AUGMENTED_LAGRANGE."
            )
        if prop.penalty_form not in ("LINEAR", "NONLINEAR"):
            raise NotImplementedError(
                f"ContactPair '{self.name}': penalty_form='{prop.penalty_form}' not implemented -- "
                "only 'LINEAR' and 'NONLINEAR' (design doc sec A.2) are supported."
            )

        slave_node_ids = _resolve_node_ids(self.slave, model=model)
        if len(slave_node_ids) == 0:
            raise ValueError(f"ContactPair '{self.name}': slave region resolved to zero nodes.")

        from dispsolver.constraint3d.surface_contact3d import SurfaceContactConstraint3D
        from dispsolver.constraint3d.contact_stiffness import (
            representative_element_stiffness,
            representative_element_length,
        )
        from dispsolver.constraint3d.pressure_overclosure import (
            HardLaw, NonlinearPenaltyLaw, LinearSoftLaw, ExponentialSoftLaw, TabularSoftLaw,
        )

        coords = mesh.nodes_array() if hasattr(mesh, "nodes_array") else mesh.nodes_array

        # sec A.1/B.7.1: the ONE auto-derived reference stiffness `k_ref`,
        # shared by hard contact's own 10x/1000x defaults, nonlinear
        # penalty's Ki/Kf, and every soft law's softness_scale resolution
        # below -- computed once, not per law.
        k_ref = representative_element_stiffness(mesh, materials, slave_node_ids)
        if k_ref is None:
            # Documented last resort (sec A.1): only when the constraint is
            # built with no mesh/materials context to estimate from at all.
            k_ref = 1.0e7
        L_char = representative_element_length(mesh, slave_node_ids)
        if L_char is None:
            L_char = 1.0
        k_default_hard = 10.0 * k_ref

        def _resolve_delta() -> float:
            return prop.allowable_penetration if prop.allowable_penetration is not None else 0.01 * L_char

        def _warn_if_absolute_and_scale_both_set(scale_default_ok: bool):
            if not scale_default_ok:
                warnings.warn(
                    f"ContactPair '{self.name}': both an absolute pressure-overclosure field and a "
                    "non-default softness_scale/allowable_penetration were set -- the absolute "
                    "field takes precedence (design doc sec B.7.5); the scale knobs are ignored "
                    "for this law."
                )

        law: Any
        k_hard_for_augmented = k_ref  # placeholder for NONLINEAR form; overwritten below for LINEAR

        if prop.normal_behavior == "HARD":
            if penalty_stiffness is not None:
                k_hard = float(penalty_stiffness)
            else:
                k_hard = k_default_hard
            k_hard *= prop.stiffness_scale_factor
            if k_hard > 1000.0 * k_ref:
                # sec A.5: Abaqus's own ceiling past which plain penalty/
                # augmented-Lagrange enforcement is considered ill-
                # conditioned enough to need Lagrange multipliers instead.
                # This codebase has no Lagrange-multiplier contact path --
                # warn, don't silently proceed or raise (a user may have a
                # legitimate reason).
                warnings.warn(
                    f"ContactPair '{self.name}': resolved penalty stiffness ({k_hard:.4g}) exceeds "
                    f"1000x the representative element stiffness ({k_ref:.4g}) -- Abaqus's own "
                    "guidance treats this as the point past which penalty/augmented-Lagrange "
                    "enforcement becomes ill-conditioned (design doc sec A.5). Consider a "
                    "smaller stiffness_scale_factor or penalty_stiffness."
                )
            k_hard_for_augmented = k_hard
            if prop.penalty_form == "NONLINEAR":
                k_i = k_ref * prop.stiffness_scale_factor
                k_f = 100.0 * k_ref * prop.stiffness_scale_factor
                e = 0.01 * L_char
                d = 0.03 * L_char
                law = NonlinearPenaltyLaw(k_i=k_i, k_f=k_f, e=e, d=d, c0=prop.clearance_at_zero_pressure)
                k_hard_for_augmented = k_i
            else:
                law = HardLaw(k_contact=k_hard, c0=prop.clearance_at_zero_pressure)

        elif prop.normal_behavior == "SOFT_LINEAR":
            if prop.linear_stiffness is not None:
                _warn_if_absolute_and_scale_both_set(
                    prop.softness_scale == 1.0 and prop.allowable_penetration is None
                )
                k = float(prop.linear_stiffness)
            else:
                k = prop.softness_scale * k_default_hard
            c0 = prop.allowable_penetration if prop.allowable_penetration is not None else prop.clearance_at_zero_pressure
            law = LinearSoftLaw(k=k * prop.stiffness_scale_factor, c0=c0)

        elif prop.normal_behavior == "SOFT_EXPONENTIAL":
            if prop.exponential_c0 is not None and prop.exponential_p0 is not None:
                _warn_if_absolute_and_scale_both_set(
                    prop.softness_scale == 1.0 and prop.allowable_penetration is None
                )
                c0 = float(prop.exponential_c0)
                p0 = float(prop.exponential_p0)
            else:
                delta = _resolve_delta()
                K_target = prop.softness_scale * k_default_hard
                c0 = delta
                p0 = 0.9 * K_target * delta / np.log(10.0)
            law = ExponentialSoftLaw(c0=c0, p0=p0 * prop.stiffness_scale_factor)

        else:  # SOFT_TABULAR
            if prop.tabular_overclosure is not None and prop.tabular_pressure is not None:
                _warn_if_absolute_and_scale_both_set(
                    prop.softness_scale == 1.0 and prop.allowable_penetration is None
                )
                h_pts = list(prop.tabular_overclosure)
                p_pts = list(prop.tabular_pressure)
            else:
                # sec B.7.4: auto-populate a starting table by sampling the
                # same exponential curve the SOFT_EXPONENTIAL auto-scale
                # would produce -- reuses B.7.3's formula rather than
                # inventing a third independent derivation.
                delta = _resolve_delta()
                K_target = prop.softness_scale * k_default_hard
                c0 = delta
                p0 = 0.9 * K_target * delta / np.log(10.0)
                sample_law = ExponentialSoftLaw(c0=c0, p0=p0 * prop.stiffness_scale_factor)
                # Ascending h (overclosure), from activation start (-c0)
                # through touching (h=0) out to deep penetration (3*delta)
                # -- TabularSoftLaw requires strictly increasing h_pts.
                h_pts = list(np.linspace(-c0, 3.0 * delta, 6))
                p_pts = [sample_law.evaluate(h)[0] for h in h_pts]
            law = TabularSoftLaw(h_pts=h_pts, p_pts=p_pts)

        if not is_rigid_master:
            from dispsolver.constraint3d.surface_contact3d_deformable import DeformableSurfaceContactConstraint3D

            master_faces = _resolve_master_faces(self.master, mesh, model=model)
            if len(master_faces) == 0:
                raise ValueError(f"ContactPair '{self.name}': master region resolved to zero faces.")
            return DeformableSurfaceContactConstraint3D(
                slave_node_ids=slave_node_ids,
                master_faces=master_faces,
                nid_to_idx=nid_to_idx,
                coords=coords,
                penalty_stiffness=float(k_hard_for_augmented),
                law=law,
                name=self.name,
            )

        return SurfaceContactConstraint3D(
            slave_node_ids=slave_node_ids,
            rigid_plane_point=np.asarray(self.master.point, dtype=np.float64),
            rigid_plane_normal=self.master.unit_normal(),
            nid_to_idx=nid_to_idx,
            coords=coords,
            penalty_stiffness=float(k_hard_for_augmented),
            strain_free_adjust=self.adjust,
            stabilization_coefficient=float(stabilization_coefficient),
            augmented_lagrange=(self.constraint_enforcement == "AUGMENTED_LAGRANGE"),
            law=law,
            name=self.name,
        )
