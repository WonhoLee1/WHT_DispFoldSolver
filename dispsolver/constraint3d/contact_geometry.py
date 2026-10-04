"""
contact_geometry.py
===================
Shared contact-geometry rules for the 3D contact constraints: which
configuration the contact geometry lives in, and closest-point search for
finite sliding.

Abaqus/Standard semantics (Analysis User's Guide, "Contact formulations in
Abaqus/Standard", local copy benchmark_element/reference_abaqus_docs/
ctc_contactpairform_std.txt; *CONTACT PAIR keyword page ctc_contactpair_std.txt):

* *CONTACT PAIR without SMALL SLIDING is FINITE sliding: the closest master
  point is tracked in the current configuration. Finite sliding is allowed
  in a geometrically linear analysis too ("load transfer paths and contact
  direction are updated").
* SMALL SLIDING, NLGEOM=YES: arbitrarily large rotations, but a secondary node
  keeps interacting with the same local area of the master surface (anchor
  chosen in the initial configuration); the local plane rotates with the
  master surface.
* SMALL SLIDING, NLGEOM=NO: "reduces to an infinitesimal-sliding and rotation
  approach" -- relative AND absolute motions are small, so the contact
  geometry (pairing and normal) stays in the reference configuration and the
  gap is linear in u.

`contact_mode(sliding, nlgeom)` maps the two settings to one of
"finite", "small", "linear"; the contact classes read it in bind_solver().
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dispsolver.constraint3d.surface_tie3d import _project_point_to_quad4


def normalize_sliding(sliding: Optional[str]) -> str:
    s = "SMALL" if sliding is None else str(sliding).upper().replace("-", "_").replace(" ", "_")
    if s in ("SMALL", "SMALL_SLIDING"):
        return "SMALL"
    if s in ("FINITE", "FINITE_SLIDING"):
        return "FINITE"
    raise ValueError(f"sliding={sliding!r}: expected 'SMALL' or 'FINITE'")


def contact_mode(sliding: str, nlgeom: bool) -> str:
    """'finite' (re-track every iteration), 'small' (frozen pairing, current
    normal) or 'linear' (frozen pairing, reference normal: gap linear in u)."""
    if normalize_sliding(sliding) == "FINITE":
        return "finite"
    return "small" if nlgeom else "linear"


def _quad_normal(q: Sequence[np.ndarray], xi: float, eta: float) -> np.ndarray:
    dxi = 0.25 * (-(1 - eta) * q[0] + (1 - eta) * q[1] + (1 + eta) * q[2] - (1 + eta) * q[3])
    deta = 0.25 * (-(1 - xi) * q[0] - (1 + xi) * q[1] + (1 + xi) * q[2] + (1 - xi) * q[3])
    n = np.cross(dxi, deta)
    nn = float(np.linalg.norm(n))
    return n / nn if nn > 0.0 else n


def face_signs(master_faces: Iterable[Tuple[int, ...]], master_interior: Dict[tuple, np.ndarray],
               coords: np.ndarray, nid_to_idx: Dict[int, int]) -> Dict[tuple, float]:
    """+1/-1 per master face so that sign * raw normal points OUT of the master
    body. Determined once from the owning element's interior point in the
    reference configuration; it is a property of the face's node ordering
    relative to its element, so it survives any (non-inverting) deformation
    and stays valid when finite sliding moves a secondary point to that face."""
    out: Dict[tuple, float] = {}
    for f in master_faces:
        interior = master_interior.get(tuple(f))
        if interior is None:
            continue
        q = [coords[nid_to_idx[n]] for n in f]
        c = 0.25 * (q[0] + q[1] + q[2] + q[3])
        out[tuple(f)] = 1.0 if float(np.dot(_quad_normal(q, 0.0, 0.0), c - interior)) >= 0.0 else -1.0
    return out


class MasterFaceLocator:
    """Closest master-face point for many query points in the CURRENT
    configuration (finite sliding). Candidate faces are pre-selected by
    centroid distance (n_candidates nearest), then projected exactly."""

    def __init__(self, master_faces: Sequence[Tuple[int, ...]], nid_to_idx: Dict[int, int],
                 coords: np.ndarray, n_candidates: int = 8):
        self.master_faces = [tuple(f) for f in master_faces]
        self.idx = np.array([[nid_to_idx[n] for n in f] for f in self.master_faces], dtype=np.int64)
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n_candidates = int(max(1, min(n_candidates, len(self.master_faces))))

    def current_nodes(self, u: np.ndarray) -> np.ndarray:
        """(n_faces, 4, 3) current vertex positions."""
        x = self.coords + np.asarray(u, dtype=np.float64).reshape(-1, 3)
        return x[self.idx]

    def nearest(self, points: np.ndarray, u: np.ndarray, faces_x: Optional[np.ndarray] = None
                ) -> List[Tuple[Tuple[int, ...], float, float, float]]:
        """[(face, xi, eta, distance)] for each point (current coordinates)."""
        if faces_x is None:
            faces_x = self.current_nodes(u)
        cent = faces_x.mean(axis=1)
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        d2 = ((pts[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        k = self.n_candidates
        cand = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < d2.shape[1] else np.tile(np.arange(d2.shape[1]), (len(pts), 1))
        out = []
        for p, cs in zip(pts, cand):
            best = None
            for c in cs:
                q = faces_x[c]
                xi, eta, dist, _ = _project_point_to_quad4(p, q[0], q[1], q[2], q[3])
                inside = max(abs(xi), abs(eta)) <= 1.0 + 1e-9
                key = (0 if inside else 1, dist)
                if best is None or key < best[0]:
                    best = (key, self.master_faces[c], float(xi), float(eta), float(dist))
            out.append(best[1:])
        return out
