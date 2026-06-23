"""Penalty NTS (node-to-segment) contact solver with JAX autodiff.

Architecture
------------
SpatialHashGrid    broad-phase segment query (O(N) average)
closest_point      JAX-compatible slave-node → master-segment projection
contact_energy     log(cosh) exponential regularization (JAX autodiff)
ContactPair        main class assembled into DynamicSolver as a penalty constraint

Usage
-----
pair = ContactPair(slave_surface, master_surface, mesh, eps=1e6, delta=0.01)
solver = DynamicSolver(..., penalty_constraints=[pair])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse as sps

from .contact_surface import ContactSurface


# =========================================================================
# Spatial Hash Grid (broad-phase collision detection)
# =========================================================================


@dataclass
class Segment:
    """A 2D line segment for master surface contact."""

    node1: int            # global node ID
    node2: int            # global node ID
    X1: np.ndarray        # (2,) initial coordinates of node 1
    X2: np.ndarray        # (2,) initial coordinates of node 2
    normal: np.ndarray    # (2,) outward unit normal


class SpatialHashGrid:
    """Dict-based spatial hash grid for O(N) broad-phase segment queries.

    Parameters
    ----------
    cell_size : float
        Grid cell side length (typically element size or similar).
    """

    def __init__(self, cell_size: float):
        self.cell_size = float(cell_size)
        self._grid: Dict[Tuple[int, int], List[int]] = {}
        self._segments: List[Segment] = []

    def build(self, segments: List[Segment]) -> None:
        """Register segments in the hash grid."""
        self._segments = segments
        self._grid.clear()
        inv = 1.0 / self.cell_size
        for idx, seg in enumerate(segments):
            # AABB of the segment
            x_min = min(seg.X1[0], seg.X2[0])
            x_max = max(seg.X1[0], seg.X2[0])
            y_min = min(seg.X1[1], seg.X2[1])
            y_max = max(seg.X1[1], seg.X2[1])
            c_min = int(np.floor(x_min * inv))
            c_max = int(np.floor(x_max * inv))
            r_min = int(np.floor(y_min * inv))
            r_max = int(np.floor(y_max * inv))
            for c in range(c_min, c_max + 1):
                for r in range(r_min, r_max + 1):
                    key = (c, r)
                    if key not in self._grid:
                        self._grid[key] = []
                    self._grid[key].append(idx)

    def query(self, point: np.ndarray) -> List[Segment]:
        """Return segments in cells around *point*."""
        inv = 1.0 / self.cell_size
        cx = int(np.floor(point[0] * inv))
        cy = int(np.floor(point[1] * inv))
        result: List[Segment] = []
        seen: Set[int] = set()
        for dcx in (-1, 0, 1):
            for dcy in (-1, 0, 1):
                key = (cx + dcx, cy + dcy)
                if key not in self._grid:
                    continue
                for idx in self._grid[key]:
                    if idx not in seen:
                        seen.add(idx)
                        result.append(self._segments[idx])
        return result


# =========================================================================
# JAX autodiff: closest point projection and contact energy
# =========================================================================


def _closest_point_on_segment_jax(
    p: jnp.ndarray, x1: jnp.ndarray, x2: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute closest-point projection of point *p* onto segment (x1, x2).

    All arguments and returns are JAX arrays (2,) each.

    Returns
    -------
    xi     : scalar natural coordinate on segment (0 = x1, 1 = x2)
    x_proj : (2,) projected point coordinates
    n      : (2,) outward unit normal (pointing from master toward slave)
    """
    edge = x2 - x1
    L = jnp.sqrt(jnp.dot(edge, edge) + 1e-30)
    t = edge / L
    n = jnp.array([-t[1], t[0]])
    xi = jnp.dot(p - x1, t) / L
    xi_clamped = jnp.clip(xi, 0.0, 1.0)
    x_proj = x1 + xi_clamped * L * t
    return xi_clamped, x_proj, n


def _nts_contact_energy(
    x_slave_cur: jnp.ndarray,
    X1: jnp.ndarray,
    X2: jnp.ndarray,
    n: jnp.ndarray,
    eps: float,
    delta: float,
) -> jnp.ndarray:
    """Contact potential for a slave node against a master segment.

    Parameters
    ----------
    x_slave_cur : (2,) current position of the slave node.
    X1, X2      : (2,) current positions of master segment endpoints.
    n           : (2,) outward unit normal of the master segment.
    eps         : penalty stiffness.
    delta       : regularization length.

    Returns
    -------
    energy : scalar contact potential (0 if gap > 0).
    """
    _, x_proj, _ = _closest_point_on_segment_jax(x_slave_cur, X1, X2)
    g_N = jnp.dot(x_slave_cur - x_proj, n)
    r = -g_N / delta
    return jnp.where(g_N < 0.0, eps * delta * jnp.log(jnp.cosh(r)), 0.0)


# JIT-compiled gradient and hessian of the contact energy (w.r.t. slave position)
_contact_energy_grad = jax.jit(jax.grad(_nts_contact_energy, argnums=0))
_contact_energy_hess = jax.jit(jax.hessian(_nts_contact_energy, argnums=0))


# =========================================================================
# ContactPair — main constraint class
# =========================================================================


class ContactPair:
    """Penalty NTS contact between a slave surface and a master surface.

    Implements the ``apply_penalty(u, f_int, K_T)`` interface expected by
    ``DynamicSolver.penalty_constraints``.

    Parameters
    ----------
    slave_surface : ContactSurface
        Slave surface (nodes projected onto master segments).
    master_surface : ContactSurface
        Master surface (line segments).
    mesh : Mesh
        Mesh object for node coordinate lookup.
    eps : float
        Penalty stiffness (force / length^2).
    delta : float
        Regularisation length for log(cosh) contact law.
        Default: automatic (1% of mean master segment length).
    """

    def __init__(
        self,
        slave_surface: ContactSurface,
        master_surface: ContactSurface,
        mesh,
        eps: float = 1e6,
        delta: Optional[float] = None,
    ):
        self.slave_surface = slave_surface
        self.master_surface = master_surface
        self.mesh = mesh
        self.eps = float(eps)

        # Node coordinate lookup
        self._coords = mesh.nodes_array()
        self._nid_to_idx = mesh.node_id_to_index()

        # Slave node global indices
        self._slave_indices = np.array(
            [self._nid_to_idx[nid] for nid in sorted(slave_surface.node_ids)],
            dtype=np.int32,
        )
        # Slave node IDs in sorted order (for consistent assembly)
        self._slave_nids = sorted(slave_surface.node_ids)

        # Master segment definitions (initial configuration)
        self._master_segments: List[Segment] = []
        seg_lengths: List[float] = []
        for i, (n1, n2) in enumerate(master_surface.edges):
            X1 = self._coords[self._nid_to_idx[n1]]
            X2 = self._coords[self._nid_to_idx[n2]]
            n = master_surface.normals[i] if i < len(master_surface.normals) else (0.0, 0.0)
            seg = Segment(node1=n1, node2=n2, X1=X1.copy(), X2=X2.copy(), normal=np.array(n, dtype=np.float64))
            self._master_segments.append(seg)
            seg_lengths.append(float(np.linalg.norm(X2 - X1)))

        # Auto delta if not provided
        if delta is not None:
            self.delta = float(delta)
        elif seg_lengths:
            self.delta = np.mean(seg_lengths) * 0.01
        else:
            self.delta = 0.001

        # Spatial hash for broad-phase (cell size = 2 * max segment length)
        if seg_lengths:
            cell_size = max(seg_lengths) * 2.0
        else:
            cell_size = 1.0
        self._hash_grid = SpatialHashGrid(cell_size)

        # Tracking
        self.n_active = 0

    # ------------------------------------------------------------------
    # Penalty constraint interface
    # ------------------------------------------------------------------

    def apply_penalty(self, u: np.ndarray, f_int: np.ndarray, K_T) -> None:
        """Add NTS contact forces and stiffness contributions.

        Parameters
        ----------
        u    : (n_dofs,) current displacement vector
        f_int: (n_dofs,) internal force vector (modified in-place)
        K_T  : (n_dofs, n_dofs) LIL-format stiffness matrix (modified in-place)
        """
        # Current positions of master segments
        u_2d = u.reshape(-1, 2)
        current_segments: List[Segment] = []
        for seg in self._master_segments:
            idx1 = self._nid_to_idx[seg.node1]
            idx2 = self._nid_to_idx[seg.node2]
            X1 = seg.X1 + u_2d[idx1]
            X2 = seg.X2 + u_2d[idx2]
            current_segments.append(
                Segment(
                    node1=seg.node1,
                    node2=seg.node2,
                    X1=X1.copy(),
                    X2=X2.copy(),
                    normal=seg.normal.copy(),
                )
            )

        # Build spatial hash from current master segments
        self._hash_grid.build(current_segments)

        active_count = 0
        nid_to_idx = self._nid_to_idx

        for slave_nid in self._slave_nids:
            slv_idx = nid_to_idx[slave_nid]
            x_slave = self._coords[slv_idx] + u_2d[slv_idx]

            # Find closest master segment via spatial hash + brute-force
            candidates = self._hash_grid.query(x_slave)
            if not candidates:
                continue

            best_gap = 1e30
            best_seg: Optional[Segment] = None
            x_slave_init = self._coords[slv_idx]
            for seg in candidates:
                # Current positions
                idx1 = nid_to_idx[seg.node1]
                idx2 = nid_to_idx[seg.node2]
                X1 = seg.X1  # already current (built above)
                X2 = seg.X2

                # Undeformed gap check: skip segments where the slave starts on
                # the interior (back) side. Such segments should never be valid
                # contact targets.
                X1_init = self._coords[nid_to_idx[seg.node1]]
                X2_init = self._coords[nid_to_idx[seg.node2]]
                edge_init = X2_init - X1_init
                L_init = np.linalg.norm(edge_init)
                if L_init < 1e-30:
                    continue
                t_init = edge_init / L_init
                xi_init = np.dot(x_slave_init - X1_init, t_init) / L_init
                xi_init = np.clip(xi_init, 0.0, 1.0)
                x_proj_init = X1_init + xi_init * L_init * t_init
                g_N_init = np.dot(x_slave_init - x_proj_init, seg.normal)
                if g_N_init < -1e-12:
                    continue  # wrong side in initial config

                # Compute signed gap at current configuration
                edge = X2 - X1
                L = np.linalg.norm(edge)
                if L < 1e-30:
                    continue
                t = edge / L
                xi = np.dot(x_slave - X1, t) / L
                xi = np.clip(xi, 0.0, 1.0)
                x_proj = X1 + xi * L * t
                g_N = np.dot(x_slave - x_proj, seg.normal)

                if g_N < best_gap:
                    best_gap = g_N
                    best_seg = seg

            if best_seg is None or best_gap >= 0.0:
                continue  # no penetration

            # Penetration detected — apply JAX autodiff contact
            active_count += 1
            idx1 = nid_to_idx[best_seg.node1]
            idx2 = nid_to_idx[best_seg.node2]

            # JAX computation: pass current slave position, current master positions
            x_slave_init = jnp.array(self._coords[slv_idx])
            u_slave_jax = jnp.array(u_2d[slv_idx])
            x_slave_cur = x_slave_init + u_slave_jax
            X1_master = jnp.array(best_seg.X1)
            X2_master = jnp.array(best_seg.X2)
            n_master_jax = jnp.array(best_seg.normal)

            f_N = np.asarray(_contact_energy_grad(x_slave_cur, X1_master, X2_master, n_master_jax, self.eps, self.delta))
            K_N = np.asarray(_contact_energy_hess(x_slave_cur, X1_master, X2_master, n_master_jax, self.eps, self.delta))

            # Assemble slave node (dofs: slv_idx*2, slv_idx*2+1)
            dof_sx = slv_idx * 2
            dof_sy = slv_idx * 2 + 1

            f_int[dof_sx] -= f_N[0]
            f_int[dof_sy] -= f_N[1]

            # Slave-side stiffness (master assumed rigid / fixed)
            K_T[dof_sx, dof_sx] += K_N[0, 0]
            K_T[dof_sx, dof_sy] += K_N[0, 1]
            K_T[dof_sy, dof_sx] += K_N[1, 0]
            K_T[dof_sy, dof_sy] += K_N[1, 1]

        self.n_active = active_count
