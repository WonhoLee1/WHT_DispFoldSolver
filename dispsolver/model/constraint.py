"""Constraint definitions for commercial CAE model hierarchy (Abaqus parity).

Provides:
- RigidBody: Reference Point (RP) based kinematic rigid body constraint.
- Tie: Surface-to-surface or node-to-surface tie constraint.
- Coupling: KinematicCoupling (RBE2) and DistributingCoupling (RBE3).
- MPC: Multi-Point Constraint (PIN, BEAM, LINK, EQUATION).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Sequence
import numpy as np


@dataclass
class Constraint:
    """Base class for all constraints in CAE model hierarchy."""
    name: str


@dataclass
class AssemblableConstraint(Constraint):
    """Base class for constraints that assemble stiffness and forces directly (Interface Segregation)."""
    def assemble(self, u_vec: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], bool]:
        """Compute the internal force vector, stiffness matrix (COO format), and error flag."""
        raise NotImplementedError("assemble must be implemented by subclasses.")
        
    def reproject_deformed(self, u_vec: np.ndarray) -> None:
        """Optional hook to update internal geometric state using deformed coordinates."""
        pass


@dataclass
class RigidBody(Constraint):
    """Rigid body constraint (*RIGID BODY in Abaqus).
    
    Couples an entire set of nodes/elements to a Reference Point (RP).
    The motion of all slaved nodes is governed purely by the rigid translation
    and rotation of the RP:
        u_s = u_RP + theta_RP x (X_s - X_RP)
    """
    ref_point: Union[str, int, Sequence[float], Any]  # Node ID, RP set name, coords, or GeneralSet
    tie_nodes: Optional[Union[str, Any]] = None       # Slaved node set name or GeneralSet
    pin_nodes: Optional[Union[str, Any]] = None       # Pin node set (translational coupling only)
    body_elements: Optional[Union[str, Any]] = None   # Elements belonging to rigid body
    is_analytic: bool = False

    def get_slave_node_ids(self, model: Any = None) -> List[int]:
        """Resolve slaved node IDs from tie_nodes, pin_nodes, or body_elements."""
        nodes: List[int] = []
        if self.tie_nodes is not None:
            if hasattr(self.tie_nodes, "get_nodes"):
                nodes.extend(self.tie_nodes.get_nodes(include_elements=True))
            elif isinstance(self.tie_nodes, (list, tuple, np.ndarray, set)):
                nodes.extend(list(self.tie_nodes))
            elif isinstance(self.tie_nodes, str) and model is not None:
                # Try finding set in model/assembly
                s = model.get_set(self.tie_nodes) if hasattr(model, "get_set") else None
                if s is not None and hasattr(s, "get_nodes"):
                    nodes.extend(s.get_nodes(include_elements=True))
        if self.pin_nodes is not None:
            if hasattr(self.pin_nodes, "get_nodes"):
                nodes.extend(self.pin_nodes.get_nodes(include_elements=True))
            elif isinstance(self.pin_nodes, (list, tuple, np.ndarray, set)):
                nodes.extend(list(self.pin_nodes))
        if self.body_elements is not None:
            if hasattr(self.body_elements, "get_nodes"):
                nodes.extend(self.body_elements.get_nodes(include_elements=True))
            elif isinstance(self.body_elements, (list, tuple, np.ndarray, set)):
                nodes.extend(list(self.body_elements))
            elif isinstance(self.body_elements, str) and model is not None:
                s = model.get_set(self.body_elements) if hasattr(model, "get_set") else None
                if s is not None and hasattr(s, "get_nodes"):
                    nodes.extend(s.get_nodes(include_elements=True))
        return sorted(list(set(int(n) for n in nodes)))


@dataclass
class Tie(Constraint):
    """Surface-to-surface or node-to-surface tie constraint (*TIE in Abaqus).
    
    Fuses two surfaces/sets together so that relative displacement is zero.
    Follows modern Abaqus/CAE standard terminology: 'main' and 'secondary'
    (with full backward compatibility for legacy 'master' and 'slave').
    """
    main: Optional[Union[str, Any]] = None
    secondary: Optional[Union[str, Any]] = None
    master: Optional[Union[str, Any]] = None
    slave: Optional[Union[str, Any]] = None
    position_tolerance: Optional[float] = None
    adjust: bool = True
    tie_rotations: bool = False
    constraint_enforcement: str = "SURFACE_TO_SURFACE"  # "NODE_TO_SURFACE" or "SURFACE_TO_SURFACE"

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
            raise ValueError("Tie constraint requires both 'main' (or 'master') and 'secondary' (or 'slave').")


@dataclass
class Coupling(AssemblableConstraint):
    """Coupling constraint (*COUPLING in Abaqus).
    
    Couples a group of slave nodes to a Reference Point (RP).
    - KINEMATIC: All coupled nodes follow the RP as a rigid body (RBE2).
    - DISTRIBUTING: Coupled nodes follow the weighted average motion of the RP
      without artificial stiffening of the coupled surface (RBE3).
    """
    ref_point: Union[str, int, Sequence[float], Any]
    surface: Union[str, Any]  # Coupled surface/node set
    coupling_type: str = "KINEMATIC"  # "KINEMATIC" or "DISTRIBUTING"

    def assemble(self, u_vec: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], bool]:
        raise NotImplementedError(f"Assembly for {self.coupling_type} Coupling is not yet implemented.")
    influence_radius: Optional[float] = None
    u1: bool = True
    u2: bool = True
    u3: bool = True
    ur1: bool = True
    ur2: bool = True
    ur3: bool = True
    weighting_method: str = "UNIFORM"  # "UNIFORM", "LINEAR", "QUADRATIC"


@dataclass
class KinematicCoupling(Coupling):
    """Convenience class for Kinematic Coupling (rigid coupling, RBE2)."""
    coupling_type: str = "KINEMATIC"


@dataclass
class DistributingCoupling(Coupling):
    """Convenience class for Distributing Coupling (weighted average coupling, RBE3)."""
    coupling_type: str = "DISTRIBUTING"


@dataclass
class EquationConstraint(AssemblableConstraint):
    """Linear equation constraint (*EQUATION in Abaqus).

    Enforces sum(coeff_i * u_dof_i) = constant via penalty method.
    Terms are stored as part-local (node_id, dof_0based, coeff) triples;
    global DOF indices resolve at build time through bind(), since the
    flattened node mapping only exists after assembly.
    """
    terms: List[Tuple[int, int, float]] = field(default_factory=list)
    constant: float = 0.0
    penalty_stiffness: float = 1e6
    _dofs: List[int] = field(default_factory=list, repr=False)
    _coeffs: List[float] = field(default_factory=list, repr=False)
    _bound: bool = field(default=False, repr=False)

    def bind(self, nid_to_idx, global_nsets, node_local_to_global, dim: int) -> bool:
        """Resolve part-local node ids to global DOF indices. Returns False when unresolvable."""
        dofs: List[int] = []
        coeffs: List[float] = []
        for ref, dof, coeff in self.terms:
            if isinstance(ref, str):
                arr = global_nsets.get(ref)
                if arr is None:
                    for key, val in global_nsets.items():
                        if key.split(".")[-1] == ref:
                            arr = val
                            break
                if arr is None:
                    return False
                for idx in [int(i) for i in np.asarray(arr).ravel().tolist()]:
                    if dof < 0 or dof >= dim:
                        return False
                    dofs.append(int(idx) * dim + int(dof))
                    coeffs.append(float(coeff))
            else:
                matches = [g for (inst, nid), g in node_local_to_global.items()
                           if int(nid) == int(ref)]
                if len(matches) != 1:
                    return False
                gid = matches[0]
                idx = nid_to_idx.get(gid)
                if idx is None or dof < 0 or dof >= dim:
                    return False
                dofs.append(int(idx) * dim + int(dof))
                coeffs.append(float(coeff))
        if not dofs:
            return False
        self._dofs = dofs
        self._coeffs = coeffs
        self._bound = True
        return True

    def assemble(self, u_vec: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], bool]:
        """Penalty assembly of the linear equation residual."""
        n_dof = len(u_vec)
        f_c = np.zeros(n_dof, dtype=np.float64)
        if not self._bound or not self._dofs:
            return f_c, (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
                         np.zeros(0, dtype=np.float64)), False
        k = float(self.penalty_stiffness)
        r = float(sum(c * u_vec[d] for d, c in zip(self._dofs, self._coeffs)) - self.constant)
        for d, c in zip(self._dofs, self._coeffs):
            f_c[d] += k * r * c
        n = len(self._dofs)
        rows = np.zeros(n * n, dtype=np.int32)
        cols = np.zeros(n * n, dtype=np.int32)
        data = np.zeros(n * n, dtype=np.float64)
        pos = 0
        for i in range(n):
            for j in range(n):
                rows[pos] = self._dofs[i]
                cols[pos] = self._dofs[j]
                data[pos] = k * self._coeffs[i] * self._coeffs[j]
                pos += 1
        return f_c, (rows, cols, data), False


@dataclass
class MPC(Constraint):
    """Multi-Point Constraint (*MPC in Abaqus).
    
    Enforces kinematic relations between discrete nodes.
    Supported types:
    - PIN: Spherical hinge (relative displacement=0, relative rotation free).
    - BEAM: Rigid beam connection (displacements and rotations matched).
    - LINK: Pin-jointed rod (distance between two nodes remains constant).
    - SLIDER: Node constrained to slide along a line/surface.
    - EQUATION: Linear algebraic multi-freedom constraint sum(A_i * u_i) = C.
    """
    mpc_type: str  # "PIN", "BEAM", "LINK", "SLIDER", "EQUATION"
    first_point: Union[str, int, Any]
    second_point: Optional[Union[str, int, Any]] = None
    dofs: Optional[List[int]] = None
    coefficients: Optional[List[float]] = None
    constant: float = 0.0
