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
        return self.transform.apply(raw)
