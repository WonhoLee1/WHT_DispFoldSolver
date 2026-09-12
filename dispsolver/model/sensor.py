"""Sensor and Iteration Hook infrastructure for real-time monitoring and feedback control (Abaqus UAMP / DISP / Sensor subroutine interface)."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
import numpy as np


class ControlAction(Enum):
    """Action returned by IterationHook to dynamically control solver execution."""
    CONTINUE = "CONTINUE"          # Proceed normally with Newton-Raphson iteration
    ADAPT_DT = "ADAPT_DT"          # Request adaptive time step adjustment
    RELAX_CRITERIA = "RELAX_CRITERIA"  # Temporarily relax residual tolerance
    ABORT_STEP = "ABORT_STEP"      # Abort current step/increment and rollback
    UPDATE_BC = "UPDATE_BC"        # Trigger immediate re-evaluation of boundary conditions


@dataclass
class Sensor:
    """Monitors real-time physical quantities across nodes, elements, or sets during solve.
    
    Variables supported:
        'U' : Displacement magnitude or component (variable_sub='1'|'2'|'3'|'MAG')
        'RF': Reaction force magnitude or component
        'S' : Mises stress or principal stress
        'E' : Strain magnitude
        'J_min': Minimum determinant of Jacobian across elements
        'GAP': Distance / closure gap between two sets
    """
    name: str
    entity_type: str = "node"   # "node", "element", "set", "model"
    entity_id: Any = None      # Node ID, Element ID, Set Name, or List of IDs
    variable: str = "U"        # "U", "RF", "S", "E", "J_min", "GAP"
    comp: str = "MAG"          # "1", "2", "3", "MAG", "MISES", "MIN"

    def evaluate(self, solver: Any, sys: Optional[Any] = None) -> float:
        """Extract current value of monitored variable from solver state."""
        if self.variable == "J_min":
            if hasattr(solver, "_check_mesh_quality"):
                mq = solver._check_mesh_quality(getattr(solver, "u", None))
                return float(mq.get("min_det_j", 1.0))
            return 1.0

        u = getattr(solver, "u", None)
        if u is None:
            return 0.0

        if self.variable == "U":
            dim = 3 if len(u) % 3 == 0 else 2
            u_nodes = u.reshape(-1, dim)
            if self.entity_type == "node" and isinstance(self.entity_id, int):
                idx = self.entity_id
                if sys is not None and hasattr(sys, "nid_to_idx") and self.entity_id in sys.nid_to_idx:
                    idx = sys.nid_to_idx[self.entity_id]
                if idx < len(u_nodes):
                    u_vec = u_nodes[idx]
                    if self.comp == "1":
                        return float(u_vec[0])
                    elif self.comp == "2":
                        return float(u_vec[1])
                    elif self.comp == "3" and dim > 2:
                        return float(u_vec[2])
                    return float(np.linalg.norm(u_vec))
            elif self.entity_type == "set" and isinstance(self.entity_id, str) and sys is not None:
                nset = sys.global_nsets.get(self.entity_id, [])
                if len(nset) > 0:
                    set_u = u_nodes[nset]
                    if self.comp == "1":
                        return float(np.mean(set_u[:, 0]))
                    elif self.comp == "2":
                        return float(np.mean(set_u[:, 1]))
                    elif self.comp == "3" and dim > 2:
                        return float(np.mean(set_u[:, 2]))
                    return float(np.mean(np.linalg.norm(set_u, axis=1)))

        return 0.0


class SensorManager:
    """Manages collection of sensors and evaluates them concurrently."""

    def __init__(self, sensors: Optional[List[Sensor]] = None):
        self.sensors: Dict[str, Sensor] = {}
        if sensors:
            for s in sensors:
                self.sensors[s.name] = s

    def add_sensor(self, sensor: Sensor) -> Sensor:
        self.sensors[sensor.name] = sensor
        return sensor

    def evaluate_all(self, solver: Any, sys: Optional[Any] = None) -> Dict[str, float]:
        """Evaluate all registered sensors and return name-value dictionary."""
        res = {}
        for name, sensor in self.sensors.items():
            res[name] = sensor.evaluate(solver, sys)
        return res


@dataclass
class IterationHook:
    """User subroutine hook executed at each Newton-Raphson iteration."""
    name: str
    callback: Callable[[int, float, Any, Dict[str, float]], ControlAction]

    def execute(self, iter_count: int, step_time: float, solver: Any, sensor_values: Dict[str, float]) -> ControlAction:
        """Execute user hook callback and return ControlAction decision."""
        if self.callback is not None:
            return self.callback(iter_count, step_time, solver, sensor_values)
        return ControlAction.CONTINUE
