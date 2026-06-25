"""
contact_jax.py
==============
JAX-accelerated penalty-based self-contact constraint for 2D plane strain.

Formulation
-----------
Penalty contact energy for a node pair (i, j):

    Π_contact(d) = {  ½·k·(d − d₀)²   if d < d₀   (active)
                   {  0                  otherwise  (inactive)

where d = ‖x_i − x_j‖ is the current Euclidean distance and d₀ is the
activation threshold (sum of contact layer half-thicknesses).

The penalty force and tangent stiffness are obtained by automatic
differentiation via JAX:

    f = dΠ/dx,    K_t = d²Π/dx²

This eliminates any error-prone manual derivation of the contact tangent
(which requires 2nd-order linearisation of the unit normal vector n =
(x_j − x_i)/d — a term that is infinite at d=0).

Contact search
--------------
A dict-based spatial hash grid (cell size = d₀) reduces candidate pairs
from O(N²) to O(N·m) where m ≈ 9–25 (neighbouring cells in 2D).
The hash function: h(cell_ix, cell_iy) = (ix·p₁ − iy·p₂) % table_size,
with Morton-hashed keys for efficiency.

Self-contact is critical for the display folding problem because:
1. The hinge zone closes onto itself (d → 0) during a 0→90° fold.
2. PET layers on opposite sides of the hinge would interpenetrate
   without contact enforcement — not physically correct.
3. The contact activation also stabilises the tangent during the
   compression buckling phase (divergence at t≈0.06s in the baseline
   solver is in the hinge compression zone where contact just activates).

References
----------
- Wriggers, P. (2006) "Computational Contact Mechanics", 2nd ed., Springer.
  Ch. 3: contact kinematics, penalty regularisation.
- Zienkiewicz, O.C. & Taylor, R.L. (2013) "The Finite Element Method",
  7th ed., Vol. 2, Ch. 12: contact problems.
- Belytschko, T. et al. (2013) "Nonlinear Finite Elements for Continua
  and Structures", 2nd ed., Ch. 10: contact-impact.
- Teschner, M. et al. (2003) "Optimized Spatial Hashing for Collision
  Detection" — the grid hashing approach used here.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Set, Tuple
from ..mesh import Mesh


@jax.jit
def _compute_local_contact(u_loc, X_loc, k_contact, d_0):
    """Penalty contact energy for a node pair.

    Parameters
    ----------
    u_loc     : (4,) [ux_i, uy_i, ux_j, uy_j]
    X_loc     : (4,) [X_i, Y_i, X_j, Y_j]
    k_contact : float  penalty stiffness
    d_0       : float  activation distance threshold
    """
    x_i = X_loc[:2] + u_loc[:2]
    x_j = X_loc[2:] + u_loc[2:]
    d = jnp.sqrt(jnp.sum((x_i - x_j) ** 2) + 1e-20)
    gap = d - d_0
    return jnp.where(gap < 0.0, 0.5 * k_contact * gap ** 2, 0.0)


_contact_val_and_grad = jax.jit(jax.value_and_grad(_compute_local_contact))
_contact_hessian      = jax.jit(jax.hessian(_compute_local_contact))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_grid(positions: np.ndarray, cell_size: float):
    """Build dict-based spatial hash grid.

    Parameters
    ----------
    positions : (N, 2) current positions of contact nodes (local ordering)
    cell_size : float  grid cell side length

    Returns
    -------
    grid : dict[(cx,cy) -> list[local_index]]
    cells: (N, 2) int  cell coordinates per node
    """
    inv = 1.0 / cell_size
    cells = np.floor(positions * inv).astype(np.int64)
    grid: dict = {}
    for li in range(len(positions)):
        key = (int(cells[li, 0]), int(cells[li, 1]))
        if key not in grid:
            grid[key] = []
        grid[key].append(li)
    return grid, cells


def _spatial_pairs(
    positions: np.ndarray,
    contact_indices: np.ndarray,
    search_dist: float,
) -> List[Tuple[int, int]]:
    """Return global-index pairs whose current distance is < search_dist.

    Uses spatial grid; complexity O(N · avg_neighbors).
    """
    grid, cells = _make_grid(positions, search_dist)
    pairs: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    for li in range(len(contact_indices)):
        cx, cy = int(cells[li, 0]), int(cells[li, 1])
        gi = int(contact_indices[li])
        for dcx in (-1, 0, 1):
            for dcy in (-1, 0, 1):
                nb = (cx + dcx, cy + dcy)
                if nb not in grid:
                    continue
                for lj in grid[nb]:
                    if lj <= li:
                        continue
                    gj = int(contact_indices[lj])
                    key = (li, lj)
                    if key in seen:
                        continue
                    seen.add(key)
                    d = float(np.linalg.norm(positions[li] - positions[lj]))
                    if d < search_dist:
                        pairs.append((gi, gj))
    return pairs


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PenaltyContactConstraint:
    """Penalty-based self-contact or node-to-node contact constraint.

    Pair discovery uses a Spatial Hash Grid (O(N) per call) rather than
    exhaustive O(N²) brute-force, enabling large contact node sets.

    Parameters
    ----------
    mesh          : Mesh
    contact_nodes : list[int]  node IDs to monitor
    k_contact     : float      penalty stiffness [force/length²]
    d_0           : float      contact activation distance [length]
    """

    def __init__(
        self,
        mesh: Mesh,
        contact_nodes: list,
        k_contact: float = 1e8,
        d_0: float = 0.2,
        depth: float = 1.0,
    ):
        self.mesh = mesh
        self.contact_nodes = contact_nodes
        # Contact force = pressure * area, and area ∝ out-of-plane depth, so the
        # penalty stiffness scales with the section depth to stay consistent with
        # depth-scaled solid elements.
        self.depth = float(depth)
        self.k_contact = float(k_contact) * self.depth
        self.d_0 = float(d_0)

        self.nid_to_idx = mesh.node_id_to_index()
        self.coords = mesh.nodes_array()

        # Global mesh-index array for the contact nodes (local ordering)
        self.contact_indices = np.array(
            [self.nid_to_idx[nid] for nid in contact_nodes], dtype=np.int32
        )

        # Build excluded pair set: pairs initially closer than 1.5*d_0
        # (these are mesh-adjacent nodes that should never be in contact)
        ref_pos = self.coords[self.contact_indices]
        initially_close = _spatial_pairs(ref_pos, self.contact_indices, 1.5 * d_0)
        self._excluded: Set[Tuple[int, int]] = set(
            (min(a, b), max(a, b)) for a, b in initially_close
        )

        self.n_active = 0

        print(
            f"PenaltyContactConstraint: {len(contact_nodes)} contact nodes, "
            f"{len(self._excluded)} excluded pairs, "
            f"spatial grid search active (d_0={d_0})"
        )

    # ------------------------------------------------------------------
    # BaseConstraint interface (no Lagrange multipliers)
    # ------------------------------------------------------------------

    def n_multipliers(self) -> int:
        return 0

    def n_extra_primal(self) -> int:
        return 0

    # ------------------------------------------------------------------
    # Core penalty application
    # ------------------------------------------------------------------

    def apply_penalty(self, u: np.ndarray, f_int: np.ndarray, K_T) -> None:
        """Add contact forces and stiffness contributions to global system."""
        d_0 = self.d_0
        k = self.k_contact

        u_2d = u.reshape(-1, 2)
        x_curr = self.coords + u_2d

        # Current positions of contact nodes (local ordering)
        pos_contact = x_curr[self.contact_indices]

        # Discover active pairs via spatial grid
        candidate_pairs = _spatial_pairs(pos_contact, self.contact_indices, d_0)

        # Filter excluded pairs (mesh-adjacent)
        active_pairs = [
            (gi, gj) for gi, gj in candidate_pairs
            if (min(gi, gj), max(gi, gj)) not in self._excluded
        ]
        self.n_active = len(active_pairs)

        for gi, gj in active_pairs:
            X_i = self.coords[gi]
            X_j = self.coords[gj]

            dof_ix, dof_iy = gi * 2, gi * 2 + 1
            dof_jx, dof_jy = gj * 2, gj * 2 + 1

            u_loc = np.array([u[dof_ix], u[dof_iy], u[dof_jx], u[dof_jy]], dtype=np.float64)
            X_loc = np.array([X_i[0], X_i[1], X_j[0], X_j[1]], dtype=np.float64)

            _, grad = _contact_val_and_grad(u_loc, X_loc, k, d_0)
            hess    = _contact_hessian(u_loc, X_loc, k, d_0)

            grad = np.asarray(grad)
            hess = np.asarray(hess)

            f_int[dof_ix] += grad[0]
            f_int[dof_iy] += grad[1]
            f_int[dof_jx] += grad[2]
            f_int[dof_jy] += grad[3]

            dofs = [dof_ix, dof_iy, dof_jx, dof_jy]
            for r in range(4):
                for c in range(4):
                    K_T[dofs[r], dofs[c]] += hess[r, c]
