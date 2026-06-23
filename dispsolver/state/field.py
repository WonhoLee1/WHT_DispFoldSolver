"""
field.py
========
Temperature field and internal variable storage for time-dependent materials.

TemperatureField
----------------
Stores nodal or element temperatures as scalar state variables.
Used by the WLF time-temperature superposition to scale Prony relaxation times.

InternalVarStore
----------------
Pytree-compatible storage for per-element, per-integration-point internal
variables (e.g., Prony history terms h_i, plastic strain eps_p).
Fixed-size arrays for JAX jit compatibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np


class TemperatureField:
    """Scalar temperature field for WLF time-temperature superposition.

    Temperatures can be specified per node (interpolated to elements)
    or per element (constant within each element).
    All temperatures in Celsius or Kelvin (consistent units with WLF params).

    Parameters
    ----------
    nodal_temperatures : dict[int, float], optional
        Node ID → temperature mapping.
    element_temperatures : dict[int, float], optional
        Element ID → temperature mapping (overrides nodal if both set).
    default_temp : float
        Fallback temperature (default 20.0 °C).
    """

    def __init__(
        self,
        nodal_temperatures: Optional[Dict[int, float]] = None,
        element_temperatures: Optional[Dict[int, float]] = None,
        default_temp: float = 20.0,
    ):
        self.nodal: Dict[int, float] = dict(nodal_temperatures or {})
        self.elemental: Dict[int, float] = dict(element_temperatures or {})
        self.default = default_temp

    def get_node_temperature(self, nid: int) -> float:
        """Temperature at a single node."""
        return self.nodal.get(nid, self.default)

    def get_element_temperature(self, eid: int) -> float:
        """Temperature for an element (element value or average of corners)."""
        if eid in self.elemental:
            return self.elemental[eid]
        return self.default

    def node_temperature_array(
        self, node_ids: List[int], nid_to_idx: Dict[int, int]
    ) -> np.ndarray:
        """Return (N,) temperature array sorted by node_id order."""
        arr = np.full(len(node_ids), self.default, dtype=np.float64)
        for nid, idx in nid_to_idx.items():
            if nid in self.nodal:
                arr[idx] = self.nodal[nid]
        return arr

    def element_temperature_array(
        self, elem_ids: List[int]
    ) -> np.ndarray:
        """Return (n_elem,) temperature per element."""
        return np.array(
            [self.elemental.get(eid, self.default) for eid in elem_ids],
            dtype=np.float64,
        )

    def __repr__(self) -> str:
        n_nodal = len(self.nodal)
        n_elem = len(self.elemental)
        return (
            f"TemperatureField(default={self.default}, "
            f"{n_nodal} nodal, {n_elem} elemental)"
        )


class InternalVarStore:
    """Fixed-size pytree-compatible storage for element internal variables.

    Each element type has a known number of integration points and each
    integration point has a known number of internal variables.

    Naming convention: visco_h_0, visco_h_1, ... for Prony history terms,
    plas_eps_p for equivalent plastic strain.

    Arrays are shaped (n_elem, n_gp, n_vars) for jit-compatible indexing.

    Parameters
    ----------
    n_elem : int
        Number of elements.
    n_gp : int
        Number of integration points per element.
    n_vars : int
        Number of internal variable components per GP.
        For viscoelastic: n_vars = 3 * M (S_voigt per Prony term).
    var_names : list of str, optional
        Names for each var component.
    """

    def __init__(
        self,
        n_elem: int,
        n_gp: int,
        n_vars: int,
        var_names: Optional[List[str]] = None,
    ):
        self.n_elem = n_elem
        self.n_gp = n_gp
        self.n_vars = n_vars
        self.var_names = var_names or [f"v{i}" for i in range(n_vars)]
        self._data = np.zeros((n_elem, n_gp, n_vars), dtype=np.float64)

    @property
    def data(self) -> np.ndarray:
        """Raw (n_elem, n_gp, n_vars) array."""
        return self._data

    @data.setter
    def data(self, arr: np.ndarray):
        self._data = np.asarray(arr, dtype=np.float64)

    def get_elem(self, elem_idx: int) -> np.ndarray:
        """Return (n_gp, n_vars) for one element."""
        return self._data[elem_idx]

    def set_elem(self, elem_idx: int, values: np.ndarray):
        self._data[elem_idx] = values

    def copy(self) -> InternalVarStore:
        """Deep copy."""
        store = InternalVarStore(self.n_elem, self.n_gp, self.n_vars, self.var_names)
        store._data = self._data.copy()
        return store

    def __repr__(self) -> str:
        return (
            f"InternalVarStore({self.n_elem} elem × {self.n_gp} gp × "
            f"{self.n_vars} vars)"
        )
