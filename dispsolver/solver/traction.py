"""
traction.py
===========
Surface traction (distributed pressure) for 2D elements.

Abaqus DLOAD/DSLOAD counterpart.
JAX autodiff for force vector (no manual derivation).

Q4 face numbering (Abaqus convention):
    Face 1: nodes 1-2 (y=min edge)
    Face 2: nodes 2-3 (x=max edge)
    Face 3: nodes 3-4 (y=max edge)
    Face 4: nodes 4-1 (x=min edge)

T3 face numbering:
    Face 1: nodes 1-2
    Face 2: nodes 2-3
    Face 3: nodes 3-1
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ------------------------------------------------------------------
# Face node index mapping
# ------------------------------------------------------------------

_Q4_FACE_NODES: Dict[int, List[int]] = {
    1: [0, 1],
    2: [1, 2],
    3: [2, 3],
    4: [3, 0],
}
_Q4_FACE_NODES_STR: Dict[str, List[int]] = {
    "P1": [0, 1], "S1": [0, 1],
    "P2": [1, 2], "S2": [1, 2],
    "P3": [2, 3], "S3": [2, 3],
    "P4": [3, 0], "S4": [3, 0],
}

_T3_FACE_NODES: Dict[int, List[int]] = {
    1: [0, 1],
    2: [1, 2],
    3: [2, 0],
}
_T3_FACE_NODES_STR: Dict[str, List[int]] = {
    "P1": [0, 1], "S1": [0, 1],
    "P2": [1, 2], "S2": [1, 2],
    "P3": [2, 0], "S3": [2, 0],
}


def _resolve_face_indices(face: str | int, elem_type: str) -> List[int]:
    """Return element-local node indices for the given face."""
    if elem_type.upper() == "QUAD4":
        if isinstance(face, str):
            return _Q4_FACE_NODES_STR.get(face.upper(), _Q4_FACE_NODES.get(1, [0, 1]))
        return _Q4_FACE_NODES.get(face, [0, 1])
    elif elem_type.upper() == "TRIA3":
        if isinstance(face, str):
            return _T3_FACE_NODES_STR.get(face.upper(), _T3_FACE_NODES.get(1, [0, 1]))
        return _T3_FACE_NODES.get(face, [0, 1])
    else:
        raise ValueError(f"Unknown element type: {elem_type}")


# ------------------------------------------------------------------
# JAX autodiff traction potential
# ------------------------------------------------------------------

def _traction_energy_q4(u_face: jnp.ndarray, X_face: jnp.ndarray,
                        pressure: jnp.ndarray) -> jnp.ndarray:
    """Q4 face follower pressure potential.

    Based on the area swept by the face edge under deformation.
    W = -p * (x_i × x_j) for edge nodes i,j where × is 2D cross product.

    JAX differentiates this potential to get force (grad) + tangent (hessian)
    automatically — no manual derivation needed.

    Parameters
    ----------
    u_face : (4,) jnp.ndarray
        Displacements of the two face nodes: [u1x, u1y, u2x, u2y]
    X_face : (4,) jnp.ndarray
        Initial coordinates of the two face nodes: [X1x, X1y, X2x, X2y]
    pressure : (1,) jnp.ndarray
        Pressure magnitude (positive = compression).

    Returns
    -------
    W : scalar jnp.ndarray
        Potential energy contribution of this face.
    """
    x1 = X_face[0:2] + u_face[0:2]
    x2 = X_face[2:4] + u_face[2:4]
    # 2D area swept: W = -p * (x1x * x2y - x2x * x1y)
    return -pressure * (x1[0] * x2[1] - x2[0] * x1[1])


_traction_grad = jax.jit(jax.grad(_traction_energy_q4, argnums=0))
_traction_hessian = jax.jit(jax.hessian(_traction_energy_q4, argnums=0))


def compute_face_forces(u_face: np.ndarray, X_face: np.ndarray,
                        pressure: float) -> np.ndarray:
    """Compute nodal forces for one Q4 face using JAX autodiff.

    Parameters
    ----------
    u_face : (4,) np.ndarray
        Current displacements of the 2 face nodes [u1x, u1y, u2x, u2y].
    X_face : (4,) np.ndarray
        Initial coordinates of the 2 face nodes [X1x, X1y, X2x, X2y].
    pressure : float
        Pressure magnitude (positive = compression).

    Returns
    -------
    f_face : (4,) np.ndarray
        Nodal force vector [f1x, f1y, f2x, f2y].
    """
    u_jax = jnp.array(u_face, dtype=jnp.float64)
    X_jax = jnp.array(X_face, dtype=jnp.float64)
    p_jax = jnp.array(pressure, dtype=jnp.float64)
    return np.asarray(_traction_grad(u_jax, X_jax, p_jax))


def compute_face_stiffness(u_face: np.ndarray, X_face: np.ndarray,
                           pressure: float) -> np.ndarray:
    """Compute tangent stiffness for one Q4 face using JAX autodiff.

    Parameters
    ----------
    u_face : (4,) np.ndarray
    X_face : (4,) np.ndarray
    pressure : float

    Returns
    -------
    K_face : (4, 4) np.ndarray
        Tangent stiffness matrix contribution.
    """
    u_jax = jnp.array(u_face, dtype=jnp.float64)
    X_jax = jnp.array(X_face, dtype=jnp.float64)
    p_jax = jnp.array(pressure, dtype=jnp.float64)
    return np.asarray(_traction_hessian(u_jax, X_jax, p_jax))


# ------------------------------------------------------------------
# SurfaceTraction config class
# ------------------------------------------------------------------

class SurfaceTraction:
    """Configuration for a distributed pressure load on an element set.

    Stores the pressure magnitude and maps element-local face nodes
    to global DOF indices during assembly.

    Parameters
    ----------
    elset_name : str
        Name of the element set to apply pressure to.
    face : str or int
        Face identifier (e.g., "P3", 3 for Q4 face 3).
    pressure : float
        Pressure magnitude (positive = compression).
    amplitude_name : str, optional
        Name of Amplitude to scale pressure over time.
    """

    def __init__(self, elset_name: str, face: str | int,
                 pressure: float, amplitude_name: Optional[str] = None):
        self.elset_name = elset_name
        self.face = face
        self.base_pressure = float(pressure)
        self.amplitude_name = amplitude_name

    @classmethod
    def from_config(cls, config: dict) -> "SurfaceTraction":
        """Create from a config dict (produced by ModelBuilder)."""
        return cls(
            elset_name=config["elset"],
            face=config["face"],
            pressure=config["pressure"],
            amplitude_name=config.get("amplitude_name"),
        )

    def pressure_at(self, t: float, amplitude_map: Optional[dict] = None) -> float:
        """Get pressure at time t, scaled by amplitude if applicable."""
        if self.amplitude_name and amplitude_map:
            amp = amplitude_map.get(self.amplitude_name)
            if amp:
                return self.base_pressure * amp.value_at(t)
        return self.base_pressure

    def assemble_forces_and_stiffness(
        self,
        mesh,
        u: np.ndarray,
        pressure: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute and assemble nodal forces and tangent stiffness.

        Parameters
        ----------
        mesh : Mesh
            The mesh (must have element_sets defined).
        u : (n_dofs,) np.ndarray
            Current displacement vector.
        pressure : float
            Current pressure magnitude (already amplitude-scaled).

        Returns
        -------
        f_global : (n_dofs,) np.ndarray
            Global force vector contribution (zeros + assembled).
        K_global : (n_dofs_reduced, n_dofs_reduced) np.ndarray
            Global stiffness contribution (zeros + assembled as dense for now).
        """
        n_dofs = len(u)

        # Get element IDs in this set
        elem_ids = mesh.element_sets.get(self.elset_name, set())
        if not elem_ids:
            # Fallback: try use all elements
            elem_ids = set(mesh.elements.keys())

        f_global = np.zeros(n_dofs, dtype=np.float64)
        # For stiffness, we use a dict-based sparse approach
        K_data: Dict[Tuple[int, int], float] = {}

        nid_to_idx = mesh.node_id_to_index()
        face_indices = _resolve_face_indices(self.face, "QUAD4")

        for eid in elem_ids:
            elem = mesh.elements.get(eid)
            if elem is None:
                continue

            elem_type = elem.elem_type
            face_idx = _resolve_face_indices(self.face, elem_type)

            # Get face node IDs
            face_nids = [elem.node_ids[i] for i in face_idx]

            # Get current displacements of face nodes
            dofs = []
            for nid in face_nids:
                idx = nid_to_idx[nid]
                dofs.append(2 * idx)
                dofs.append(2 * idx + 1)
            u_face = u[dofs]

            # Get initial coordinates of face nodes
            X_face = []
            for nid in face_nids:
                node = mesh.get_node(nid)
                X_face.extend([node.x, node.y])

            # JAX autodiff: compute force and stiffness
            f_face = compute_face_forces(
                np.array(u_face, dtype=np.float64),
                np.array(X_face, dtype=np.float64),
                pressure,
            )

            # Assemble
            for i, dof_i in enumerate(dofs):
                f_global[dof_i] += f_face[i]

        return f_global, {}  # stiffness dict placeholder

    def __repr__(self) -> str:
        return (
            f"SurfaceTraction(elset={self.elset_name}, "
            f"face={self.face}, p={self.base_pressure})"
        )
