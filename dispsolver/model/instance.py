"""Part Instance definition within Assembly."""

from __future__ import annotations
from typing import Sequence, Optional, Dict, List
import numpy as np

from dispsolver.model.part import Part
from dispsolver.model.base import Transform3D


class Instance:
    """Positioned instance of a Part in the Assembly global space."""

    def __init__(self, name: str, part: Part, dependent: bool = True):
        self.name = str(name)
        self.part = part
        self.dependent = bool(dependent)
        self.transform = Transform3D()

    def translate(self, vector: Sequence[float]) -> Instance:
        """Translate this instance in global coordinates."""
        self.transform.translation += np.array(vector, dtype=np.float64)
        return self

    def rotate(
        self,
        axis: Sequence[float],
        angle_deg: float,
        center: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> Instance:
        """Rotate this instance about an axis through a center point."""
        self.transform.rotation_axis = np.array(axis, dtype=np.float64)
        self.transform.rotation_angle_rad += np.radians(angle_deg)
        self.transform.rotation_center = np.array(center, dtype=np.float64)
        return self

    def get_world_coordinates(self, local_nids: Optional[Sequence[int]] = None) -> np.ndarray:
        """Get transformed world coordinates for specified local nodes (or all nodes)."""
        if local_nids is None:
            nids = sorted(self.part.nodes.keys())
        else:
            nids = list(local_nids)
            
        raw = np.array([self.part.nodes[nid] for nid in nids], dtype=np.float64)
        dim = raw.shape[1] if raw.size else 3
        shifted = raw + np.asarray(self.transform.translation[:dim], dtype=np.float64)
        rot_only = self.transform.copy()
        rot_only.translation = np.zeros_like(rot_only.translation)
        if (rot_only.rotation_axis is not None
                and abs(rot_only.rotation_angle_rad) > 1e-15
                and float(np.linalg.norm(rot_only.rotation_axis)) < 1e-12):
            rot_only.rotation_axis = None
        return rot_only.apply(shifted)

    def get_exterior_faces(self) -> List[Tuple[int, ...]]:
        """Exterior face node-ID tuples of this instance's part."""
        return self.part.get_exterior_faces()

    @property
    def exterior(self) -> Any:
        """Exterior boundary surface of this instance."""
        surf = self.part.exterior
        surf.instance_name = self.name
        return surf

