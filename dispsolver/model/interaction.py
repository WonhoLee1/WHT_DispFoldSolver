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
import numpy as np


@dataclass
class ContactProperty:
    """Interaction property (*SURFACE INTERACTION in Abaqus).

    Phase 1 only implements normal_behavior="HARD" (zero pressure for
    positive clearance, penalty-approximated non-penetration for zero/
    negative clearance) -- see design doc §5. "SOFT_LINEAR"/
    "SOFT_EXPONENTIAL" are named here as placeholders for a future phase,
    not implemented (constructing one with those values will not raise,
    but SurfaceContactConstraint3D only knows how to build a HARD-contact
    runtime constraint right now -- see ContactPair.build_runtime_constraint).

    friction=None means frictionless (design doc §6: deferred past Phase 1
    AND Phase 2). A nonzero friction coefficient here is accepted as a
    value but has NO runtime effect yet -- there is no tangential-plane
    enforcement code path in SurfaceContactConstraint3D at all.
    """
    name: str
    normal_behavior: str = "HARD"
    friction: Optional[float] = None


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


@dataclass
class ContactPair:
    """A single contact interaction between a main and a secondary surface
    (*CONTACT PAIR in Abaqus). Modern CAE terminology: 'main' and 'secondary'
    (with full backward compatibility for legacy 'master' and 'slave').
    """
    name: str
    interaction_property: ContactProperty
    main: Optional[Union[str, "AnalyticalRigidSurface"]] = None
    secondary: Optional[Union[str, Any]] = None
    master: Optional[Union[str, "AnalyticalRigidSurface"]] = None
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
        if not isinstance(self.master, AnalyticalRigidSurface):
            raise NotImplementedError(
                f"ContactPair '{self.name}': master must be an AnalyticalRigidSurface for Phase 1 -- "
                "deformable-vs-deformable contact is Phase 2 (design doc §9), not implemented."
            )
        if self.interaction_property.normal_behavior != "HARD":
            raise NotImplementedError(
                f"ContactPair '{self.name}': normal_behavior="
                f"'{self.interaction_property.normal_behavior}' not implemented -- "
                "Phase 1 only supports hard contact (design doc §5)."
            )

        slave_node_ids = _resolve_node_ids(self.slave, model=model)
        if len(slave_node_ids) == 0:
            raise ValueError(f"ContactPair '{self.name}': slave region resolved to zero nodes.")

        from dispsolver.constraint3d.surface_contact3d import SurfaceContactConstraint3D

        coords = mesh.nodes_array() if hasattr(mesh, "nodes_array") else mesh.nodes_array
        default_k = penalty_stiffness
        if default_k is None:
            # Abaqus's own default linear-penalty scaling is "10 times a
            # representative underlying element stiffness"
            # (ctc_contactconstraints_std.txt). We don't have a clean
            # single "representative element stiffness" scalar exposed by
            # DynamicSolver3D today, so fall back to a large fixed value
            # documented here rather than silently guessing per-model --
            # callers with a real material stiffness scale should pass
            # penalty_stiffness explicitly (see SurfaceContactConstraint3D's
            # own docstring for how this is used).
            default_k = 1.0e8

        return SurfaceContactConstraint3D(
            slave_node_ids=slave_node_ids,
            rigid_plane_point=np.asarray(self.master.point, dtype=np.float64),
            rigid_plane_normal=self.master.unit_normal(),
            nid_to_idx=nid_to_idx,
            coords=coords,
            penalty_stiffness=float(default_k),
            strain_free_adjust=self.adjust,
            stabilization_coefficient=float(stabilization_coefficient),
            augmented_lagrange=(self.constraint_enforcement == "AUGMENTED_LAGRANGE"),
            name=self.name,
        )
