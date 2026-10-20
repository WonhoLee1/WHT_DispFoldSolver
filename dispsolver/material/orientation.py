"""Orthotropic stiffness from engineering constants, and its rotation into a material orientation.

Voigt order is the one every solver kernel uses: xx yy zz xy yz xz with ENGINEERING shear strain, so the
shear moduli sit at (3,3)=G12, (4,4)=G23, (5,5)=G13.

``frame`` is the 3x3 direction-cosine matrix of ``*ORIENTATION``: its rows are the local axes 1, 2, 3 in
global coordinates. A vector with local components x_l has global components ``frame.T @ x_l``.
"""
from __future__ import annotations

import numpy as np

_V = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def orthotropic_stiffness(ec) -> np.ndarray:
    """6x6 stiffness of ``(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)`` (inverse of the compliance)."""
    E1, E2, E3, nu12, nu13, nu23, G12, G13, G23 = (float(x) for x in ec)
    nu21, nu31, nu32 = nu12 * E2 / E1, nu13 * E3 / E1, nu23 * E3 / E2
    S = np.zeros((6, 6), dtype=np.float64)
    S[0, 0] = 1.0 / E1;    S[0, 1] = -nu21 / E2; S[0, 2] = -nu31 / E3
    S[1, 0] = -nu12 / E1; S[1, 1] = 1.0 / E2;   S[1, 2] = -nu32 / E3
    S[2, 0] = -nu13 / E1; S[2, 1] = -nu23 / E2; S[2, 2] = 1.0 / E3
    S[3, 3] = 1.0 / G12
    S[4, 4] = 1.0 / G23
    S[5, 5] = 1.0 / G13
    return np.linalg.inv(S)


def rotate_stiffness(C6: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """``C'`` of the material whose local axes are the rows of ``frame``: the fourth-order tensor rotation
    ``C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs`` with ``R = frame.T`` (Voigt with engineering shear strain maps
    onto the tensor components without extra factors)."""
    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_V):
        for J, (k, l) in enumerate(_V):
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    C4[a, b, c, d] = C6[I, J]
    R = np.asarray(frame, dtype=np.float64).T
    G4 = np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, C4)
    return np.array([[G4[i, j, k, l] for (k, l) in _V] for (i, j) in _V])


def rotate_frame(frame: np.ndarray, axis: int, angle_deg: float) -> np.ndarray:
    """The frame turned by ``angle_deg`` about its own local axis ``axis`` (1, 2 or 3), right-hand rule."""
    ax = int(axis) - 1
    if ax not in (0, 1, 2):
        raise ValueError(f"local direction must be 1, 2 or 3, got {axis}")
    t = np.radians(float(angle_deg))
    c, s = np.cos(t), np.sin(t)
    a, b = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[ax]
    # new_a = c*old_a + s*old_b ; new_b = -s*old_a + c*old_b   (cyclic order ax -> a -> b)
    f = np.array(frame, dtype=np.float64)
    out = f.copy()
    out[a] = c * f[a] + s * f[b]
    out[b] = -s * f[a] + c * f[b]
    return out


def oriented_material(mat_obj, ori, dim: int, where: str):
    """The material a section with ``ORIENTATION=`` uses: for ENGINEERING CONSTANTS a copy carrying the
    stiffness rotated into the orientation (as ``C_mat``, which every 3D solid kernel reads); any other
    material is returned unchanged (its response does not depend on a direction).

    Raises NotImplementedError for what is not represented: a frame that varies by element (cylindrical,
    spherical, node-defined) and 2D, where the orthotropic stiffness itself does not exist yet."""
    import dataclasses
    extra = getattr(mat_obj, "extra_params", None) or {}
    if "engineering_constants" not in extra:
        return mat_obj
    name = getattr(mat_obj, "name", None)
    if dim != 3:
        raise NotImplementedError(f"{where}: material {name!r}: orthotropic ENGINEERING CONSTANTS are not "
                                  f"implemented in {dim}D")
    frame = getattr(ori, "frame", None)
    if frame is None:
        raise NotImplementedError(
            f"{where}: *ORIENTATION {ori.name} (SYSTEM={ori.system}, DEFINITION={ori.definition}): only a "
            f"RECTANGULAR orientation given by COORDINATES is implemented; the other systems give every element "
            f"its own axes, which the 3D kernels do not take per element")
    C = rotate_stiffness(orthotropic_stiffness(extra["engineering_constants"]), frame)
    new_extra = {k: v for k, v in extra.items() if k not in ("engineering_constants", "elastic_type")}
    new_extra["C_mat"] = C
    return dataclasses.replace(mat_obj, extra_params=new_extra)
