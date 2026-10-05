"""Constraint definitions for commercial CAE model hierarchy (Abaqus parity).

Provides:
- RigidBody: Reference Point (RP) based kinematic rigid body constraint.
- Tie: Surface-to-surface or node-to-surface tie constraint.
- Coupling: KinematicCoupling (RBE2) and DistributingCoupling (RBE3).
- MPC: Multi-Point Constraint (PIN, BEAM, LINK, EQUATION).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Sequence, Tuple
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
    ref_point: Union[str, int, Sequence[float], Any] = None
    surface: Union[str, Any] = None  # Coupled surface/node set
    coupling_type: str = "KINEMATIC"  # "KINEMATIC" or "DISTRIBUTING"
    influence_radius: Optional[float] = None
    u1: bool = True
    u2: bool = True
    u3: bool = True
    ur1: bool = True
    ur2: bool = True
    ur3: bool = True
    weighting_method: str = "UNIFORM"  # "UNIFORM", "LINEAR", "QUADRATIC"
    _rp_idx: Optional[int] = field(default=None, repr=False)
    _slave_indices: List[int] = field(default_factory=list, repr=False)
    _dim: int = field(default=2, repr=False)
    _bound: bool = field(default=False, repr=False)

    def bind(self, nid_to_idx: dict, global_nsets: dict, node_local_to_global: dict, dim: int,
             global_elsets: dict = None, elem_conn_0based: np.ndarray = None, sys: Any = None) -> bool:
        """Resolve the reference node and the coupled node set through
        ``model.regions`` (the BC/load name rules). Raises RegionError for a
        reference that does not resolve or is ambiguous; returns False only
        when the coupled set is empty."""
        from dispsolver.model.regions import describe, make_view, resolve_node_region, resolve_single_node
        if sys is None:
            sys = make_view(nid_to_idx, global_nsets, node_local_to_global, global_elsets)
        w = describe(self, f"constraint {self.name}")
        ref_str = str(self.ref_point).strip() if self.ref_point is not None else ""
        if not ref_str:
            return False
        rp_idx = resolve_single_node(sys, ref_str, what=f"{w} reference node")

        slave_indices = []
        surf_str = str(self.surface).strip() if self.surface is not None else ""
        if surf_str:
            arr = resolve_node_region(sys, surf_str, what=f"{w} surface")
            for raw_id in np.asarray(arr).ravel().tolist():
                s_idx = int(raw_id)
                if s_idx != rp_idx and s_idx not in slave_indices:
                    slave_indices.append(s_idx)

        if len(slave_indices) == 0:
            self._bound = False
            return False

        self._rp_idx = rp_idx
        self._slave_indices = slave_indices
        self._dim = dim
        self._bound = True
        return True

    def assemble(self, u_vec: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], bool]:
        """Penalty assembly of kinematic (RBE2) or distributing (RBE3) coupling."""
        n_dof = len(u_vec)
        f_c = np.zeros(n_dof, dtype=np.float64)
        if not getattr(self, "_bound", False) or not hasattr(self, "_rp_idx") or self._rp_idx is None:
            return f_c, (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)), False

        k = 1e7
        dim = self._dim
        rp_idx = self._rp_idx
        slave_indices = self._slave_indices
        n_s = len(slave_indices)
        if n_s == 0:
            return f_c, (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)), False

        rows_l, cols_l, data_l = [], [], []

        if str(self.coupling_type).upper() == "KINEMATIC":
            for s_idx in slave_indices:
                for d in range(dim):
                    dof_rp = rp_idx * dim + d
                    dof_s = s_idx * dim + d
                    if dof_s >= n_dof or dof_rp >= n_dof:
                        continue
                    
                    diff = u_vec[dof_s] - u_vec[dof_rp]
                    f_c[dof_s] += k * diff
                    f_c[dof_rp] -= k * diff

                    for r, c, val in [
                        (dof_s, dof_s, k),
                        (dof_s, dof_rp, -k),
                        (dof_rp, dof_s, -k),
                        (dof_rp, dof_rp, k)
                    ]:
                        rows_l.append(r)
                        cols_l.append(c)
                        data_l.append(val)

        else:
            w = 1.0 / n_s
            for d in range(dim):
                dof_rp = rp_idx * dim + d
                dofs = [s_idx * dim + d for s_idx in slave_indices if (s_idx * dim + d) < n_dof] + [dof_rp]
                if dof_rp >= n_dof:
                    continue
                coeffs = [w] * (len(dofs) - 1) + [-1.0]

                r = sum(c_i * u_vec[d_i] for d_i, c_i in zip(dofs, coeffs))
                for d_i, c_i in zip(dofs, coeffs):
                    f_c[d_i] += k * r * c_i

                for i, d_i in enumerate(dofs):
                    for j, d_j in enumerate(dofs):
                        rows_l.append(d_i)
                        cols_l.append(d_j)
                        data_l.append(k * coeffs[i] * coeffs[j])

        return f_c, (np.array(rows_l, dtype=np.int32), np.array(cols_l, dtype=np.int32), np.array(data_l, dtype=np.float64)), False


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
    penalty_stiffness: float = 1e9
    pair_index: Optional[int] = None
    _dofs: List[int] = field(default_factory=list, repr=False)
    _coeffs: List[float] = field(default_factory=list, repr=False)
    _bound: bool = field(default=False, repr=False)

    def bind(self, nid_to_idx, global_nsets, node_local_to_global, dim: int,
             global_elsets: dict = None, elem_conn_0based: np.ndarray = None, sys: Any = None) -> bool:
        """Resolve each term's node or node set through ``model.regions``.
        A set term contributes every node of the set, or -- when ``pair_index``
        is set (an equation generated from node sets, one per set position) --
        only the pair_index-th node in the set's stored order. Raises
        RegionError on an unresolved / ambiguous reference."""
        from dispsolver.model.regions import describe, make_view, resolve_node_region
        if sys is None:
            sys = make_view(nid_to_idx, global_nsets, node_local_to_global, global_elsets)
        w = describe(self, f"equation {self.name}")
        dofs: List[int] = []
        coeffs: List[float] = []
        for ref, dof, coeff in self.terms:
            if dof < 0 or dof >= dim:
                return False
            idxs = [int(i) for i in resolve_node_region(sys, str(ref), what=f"{w} term {ref!r}").tolist()]
            if self.pair_index is not None and len(idxs) > 1:
                if self.pair_index >= len(idxs):
                    return False
                idxs = [idxs[self.pair_index]]
            for idx in idxs:
                dofs.append(idx * dim + int(dof))
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
