"""FieldOutput, FieldValue, and FieldLocation data containers for Abaqus parity."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence, Any, Union
import numpy as np

from dispsolver.output.variables import FieldPosition, FieldType, FIELD_VARIABLE_COMPONENTS


@dataclass
class FieldValue:
    """Abaqus-compatible individual value within a FieldOutput.
    
    Attributes
    ----------
    node_label : Optional[int]
        Node ID if nodal field.
    element_label : Optional[int]
        Element ID if element/integration point field.
    integration_point : Optional[int]
        1-based integration point number (0 for centroid).
    data : np.ndarray
        Raw numerical data (scalar, vector, or Voigt tensor components).
    mises : Optional[float]
        Von Mises stress if stress tensor.
    pressure : Optional[float]
        Equivalent hydrostatic pressure (-1/3 trace) if stress tensor.
    magnitude : Optional[float]
        Vector magnitude if vector field.
    """
    node_label: Optional[int] = None
    element_label: Optional[int] = None
    integration_point: Optional[int] = None
    data: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    mises: Optional[float] = None
    pressure: Optional[float] = None
    max_principal: Optional[float] = None
    mid_principal: Optional[float] = None
    min_principal: Optional[float] = None
    magnitude: Optional[float] = None


@dataclass
class FieldLocation:
    """Location metadata for FieldOutput."""
    position: FieldPosition = FieldPosition.NODES
    section_point: Optional[int] = None


class FieldOutput:
    """Abaqus-compatible FieldOutput container for a converged frame.
    
    Provides both Abaqus-style object iteration (.values[i]) and modern
    NumPy-vectorized batch array access (.data, .get_mises(), .get_component()).
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        field_type: Union[str, FieldType] = FieldType.VECTOR,
        position: Union[str, FieldPosition] = FieldPosition.NODES,
        data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        integration_points: Optional[np.ndarray] = None,
        component_labels: Optional[Tuple[str, ...]] = None,
    ):
        self.name = str(name)
        self.description = str(description)
        self.type = FieldType(field_type) if isinstance(field_type, str) else field_type
        self.position = FieldPosition(position) if isinstance(position, str) else position
        
        # Batch NumPy storage
        self.data: np.ndarray = np.asarray(data if data is not None else np.zeros((0, 0), dtype=np.float64))
        self.labels: np.ndarray = np.asarray(labels if labels is not None else np.zeros(0, dtype=np.int32))
        self.integration_points: Optional[np.ndarray] = integration_points
        
        # Component labels
        if component_labels is not None:
            self.component_labels = tuple(component_labels)
        else:
            self.component_labels = FIELD_VARIABLE_COMPONENTS.get(self.name, ())
            
        self.locations: List[FieldLocation] = [FieldLocation(position=self.position)]
        self._cached_values: Optional[List[FieldValue]] = None

    @property
    def values(self) -> List[FieldValue]:
        """Abaqus-compatible sequence of individual FieldValue objects (lazy generated)."""
        if self._cached_values is not None:
            return self._cached_values

        n_items = len(self.data)
        values_list: List[FieldValue] = []
        is_stress = (self.name == "S")
        is_vec = (self.type == FieldType.VECTOR)
        is_node = (self.position == FieldPosition.NODES)

        for i in range(n_items):
            d = self.data[i]
            lbl = int(self.labels[i]) if i < len(self.labels) else i + 1
            ipt = int(self.integration_points[i]) if self.integration_points is not None and i < len(self.integration_points) else 0

            mises_val = None
            press_val = None
            max_p_val = None
            mid_p_val = None
            min_p_val = None
            mag_val = None

            if is_stress and len(d) >= 3:
                mises_val = float(self._compute_single_mises(d))
                press_val = float(-np.mean(d[:3]))
                p_max, p_mid, p_min = self._compute_single_principal(d)
                max_p_val = float(p_max)
                mid_p_val = float(p_mid)
                min_p_val = float(p_min)
            elif is_vec:
                mag_val = float(np.linalg.norm(d))

            val = FieldValue(
                node_label=lbl if is_node else None,
                element_label=lbl if not is_node else None,
                integration_point=ipt if not is_node else None,
                data=d,
                mises=mises_val,
                pressure=press_val,
                max_principal=max_p_val,
                mid_principal=mid_p_val,
                min_principal=min_p_val,
                magnitude=mag_val
            )
            values_list.append(val)

        self._cached_values = values_list
        return self._cached_values

    # ─── NUMPY VECTORIZED MODERN ACCESSORS ─────────────────────────────

    def get_magnitude(self) -> np.ndarray:
        """Compute vector magnitude across all rows."""
        if self.data.ndim == 1:
            return np.abs(self.data)
        return np.linalg.norm(self.data, axis=1)

    def get_component(self, comp_name: str) -> np.ndarray:
        """Extract a single component 1D array by component name (e.g. 'U1', 'S11', 'MISES', 'MAXP')."""
        comp_upper = comp_name.upper()
        if comp_upper == "MISES" and self.name == "S":
            return self.get_mises()
        if comp_upper in ["MAXP", "MAX_PRINCIPAL"] and self.name == "S":
            p_max, _, _ = self.get_principal()
            return p_max
        if comp_upper in ["MIDP", "MID_PRINCIPAL"] and self.name == "S":
            _, p_mid, _ = self.get_principal()
            return p_mid
        if comp_upper in ["MINP", "MIN_PRINCIPAL"] and self.name == "S":
            _, _, p_min = self.get_principal()
            return p_min
        if comp_upper in ["PRESS", "PRESSURE"] and self.name == "S":
            return self.get_pressure()
        if comp_upper in ["MAG", "MAGNITUDE"] or (comp_upper == self.name and self.type == FieldType.VECTOR):
            return self.get_magnitude()

        if comp_upper in self.component_labels:
            idx = self.component_labels.index(comp_upper)
            if idx < self.data.shape[1]:
                return self.data[:, idx]

        raise KeyError(f"Component '{comp_name}' not found in {self.name} components: {self.component_labels}")

    def get_pressure(self) -> np.ndarray:
        """Compute equivalent hydrostatic pressure (-1/3 trace)."""
        if self.name != "S" or self.data.shape[1] < 3:
            raise ValueError("Pressure is only defined for 3D/plane-strain stress tensors.")
        mean_stress = np.mean(self.data[:, :3], axis=1)
        return -mean_stress

    def get_mises(self) -> np.ndarray:
        """Compute Von Mises equivalent stress array in a single vectorized pass."""
        if self.name != "S":
            raise ValueError("Von Mises is only defined for Stress tensor field ('S').")
        d = self.data
        if d.shape[1] >= 6:
            # Full 3D Voigt: [S11, S22, S33, S12, S13, S23]
            s11, s22, s33 = d[:, 0], d[:, 1], d[:, 2]
            s12, s13, s23 = d[:, 3], d[:, 4], d[:, 5]
            mises_sq = 0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) + 3.0 * (s12**2 + s13**2 + s23**2)
            return np.sqrt(np.maximum(mises_sq, 0.0))
        elif d.shape[1] == 4:
            # 2D Plane strain: [S11, S22, S33, S12]
            s11, s22, s33, s12 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
            mises_sq = 0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) + 3.0 * s12**2
            return np.sqrt(np.maximum(mises_sq, 0.0))
        elif d.shape[1] == 3:
            # 2D Plane stress: [S11, S22, S12] (S33=0)
            s11, s22, s12 = d[:, 0], d[:, 1], d[:, 2]
            mises_sq = s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2
            return np.sqrt(np.maximum(mises_sq, 0.0))
        else:
            raise ValueError(f"Unsupported stress tensor shape: {d.shape}")

    def get_principal(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (Max, Mid, Min) principal stresses for each row.
        
        Supports 3D full Voigt (>=6), 2D plane strain (4 components [S11, S22, S33, S12]),
        and 2D plane stress (3 components [S11, S22, S12]).
        """
        if self.name != "S":
            raise NotImplementedError("Principal stress calculation requires stress tensor 'S'.")
        n = len(self.data)
        p_max = np.zeros(n, dtype=np.float64)
        p_mid = np.zeros(n, dtype=np.float64)
        p_min = np.zeros(n, dtype=np.float64)

        for i in range(n):
            p1, p2, p3 = self._compute_single_principal(self.data[i])
            p_max[i] = p1
            p_mid[i] = p2
            p_min[i] = p3

        return p_max, p_mid, p_min

    def get_subset(
        self,
        region_labels: Optional[Sequence[int]] = None,
        position: Optional[Union[str, FieldPosition]] = None
    ) -> FieldOutput:
        """Filter field output to a specific subset of node/element IDs."""
        if region_labels is None:
            return self

        labels_set = set(region_labels)
        mask = np.array([lbl in labels_set for lbl in self.labels], dtype=bool)

        sub_data = self.data[mask]
        sub_labels = self.labels[mask]
        sub_ipts = self.integration_points[mask] if self.integration_points is not None else None

        pos = position if position is not None else self.position
        return FieldOutput(
            name=self.name,
            description=f"{self.description} (Subset)",
            field_type=self.type,
            position=pos,
            data=sub_data,
            labels=sub_labels,
            integration_points=sub_ipts,
            component_labels=self.component_labels
        )

    @staticmethod
    def _compute_single_mises(s: np.ndarray) -> float:
        if len(s) >= 6:
            m_sq = 0.5 * ((s[0] - s[1])**2 + (s[1] - s[2])**2 + (s[2] - s[0])**2) + 3.0 * (s[3]**2 + s[4]**2 + s[5]**2)
            return float(np.sqrt(max(m_sq, 0.0)))
        elif len(s) == 4:
            m_sq = 0.5 * ((s[0] - s[1])**2 + (s[1] - s[2])**2 + (s[2] - s[0])**2) + 3.0 * s[3]**2
            return float(np.sqrt(max(m_sq, 0.0)))
        elif len(s) == 3:
            m_sq = s[0]**2 - s[0] * s[1] + s[1]**2 + 3.0 * s[2]**2
            return float(np.sqrt(max(m_sq, 0.0)))
        return 0.0

    @staticmethod
    def _compute_single_principal(s: np.ndarray) -> Tuple[float, float, float]:
        """Compute (p_max, p_mid, p_min) in descending order for a single stress state."""
        if len(s) >= 6:
            tensor = np.array([
                [s[0], s[3], s[4]],
                [s[3], s[1], s[5]],
                [s[4], s[5], s[2]]
            ], dtype=np.float64)
            eigvals = np.linalg.eigvalsh(tensor)
            return float(eigvals[2]), float(eigvals[1]), float(eigvals[0])
        elif len(s) == 4:
            s11, s22, s33, s12 = s[0], s[1], s[2], s[3]
            r = np.sqrt(0.25 * (s11 - s22)**2 + s12**2)
            c = 0.5 * (s11 + s22)
            p_in1 = c + r
            p_in2 = c - r
            vals = sorted([float(p_in1), float(p_in2), float(s33)], reverse=True)
            return vals[0], vals[1], vals[2]
        elif len(s) == 3:
            s11, s22, s12 = s[0], s[1], s[2]
            r = np.sqrt(0.25 * (s11 - s22)**2 + s12**2)
            c = 0.5 * (s11 + s22)
            p_in1 = c + r
            p_in2 = c - r
            vals = sorted([float(p_in1), float(p_in2), 0.0], reverse=True)
            return vals[0], vals[1], vals[2]
        return 0.0, 0.0, 0.0

