"""Base utility classes and transformation mathematics for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np


class Transform3D:
    """Rigid body 3D affine transformation (translation and axis-angle rotation)."""

    def __init__(
        self,
        translation: Sequence[float] = (0.0, 0.0, 0.0),
        rotation_axis: Optional[Sequence[float]] = None,
        rotation_angle_deg: float = 0.0,
        rotation_center: Sequence[float] = (0.0, 0.0, 0.0)
    ):
        self.translation = np.array(translation, dtype=np.float64)
        self.rotation_axis = (
            np.array(rotation_axis, dtype=np.float64) if rotation_axis is not None else None
        )
        self.rotation_angle_rad = np.radians(rotation_angle_deg)
        self.rotation_center = np.array(rotation_center, dtype=np.float64)

    def copy(self) -> Transform3D:
        t = Transform3D(
            translation=self.translation.copy(),
            rotation_axis=self.rotation_axis.copy() if self.rotation_axis is not None else None,
            rotation_angle_deg=np.degrees(self.rotation_angle_rad),
            rotation_center=self.rotation_center.copy()
        )
        return t

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply rigid transformation to an (N, dim) coordinate array.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2) or (N, 3) coordinate array

        Returns
        -------
        np.ndarray
            Transformed coordinate array with same shape
        """
        coords_arr = np.asarray(coords, dtype=np.float64)
        if len(coords_arr) == 0:
            return coords_arr.copy()

        dim = coords_arr.shape[1]
        out = coords_arr.copy()

        # Apply rotation if specified
        if self.rotation_axis is not None and abs(self.rotation_angle_rad) > 1e-15:
            c = np.cos(self.rotation_angle_rad)
            s = np.sin(self.rotation_angle_rad)

            if dim == 2:
                # 2D in-plane rotation about rotation_center
                cx, cy = self.rotation_center[0], self.rotation_center[1]
                shifted_x = out[:, 0] - cx
                shifted_y = out[:, 1] - cy
                out[:, 0] = shifted_x * c - shifted_y * s + cx
                out[:, 1] = shifted_x * s + shifted_y * c + cy
            else:
                # 3D Rodrigues rotation formula
                axis = self.rotation_axis / np.linalg.norm(self.rotation_axis)
                center = self.rotation_center[:3]
                shifted = out - center
                cross_prod = np.cross(axis, shifted)
                dot_prod = np.sum(shifted * axis, axis=1, keepdims=True)
                out = shifted * c + cross_prod * s + axis * dot_prod * (1.0 - c) + center

        # Apply translation
        out += self.translation[:dim]
        return out
