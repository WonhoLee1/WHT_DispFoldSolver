"""FieldOutputRequest, HistoryOutputRequest, and timing management for Abaqus parity."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union, Callable, Any
import numpy as np

from dispsolver.output.variables import FieldPosition


@dataclass
class FieldOutputRequest:
    """Abaqus-compatible Field Output Request.
    
    Parameters
    ----------
    name : str
        Identifier name (e.g. 'F-Output-1').
    create_step_name : str
        Name of the step where this request is defined.
    variables : Sequence[str]
        List of variable names to output (e.g. ['U', 'S', 'LE', 'PEEQ', 'RF']).
    frequency : Optional[int]
        Output every N increments (default: 1).
    num_intervals : Optional[int]
        Evenly space output over N intervals in step time period (Abaqus parity).
    time_interval : Optional[float]
        Output every fixed time interval delta_t.
    time_points : Optional[Sequence[float]]
        Explicit list of time points at which output is requested.
    exact_time_points : bool
        If True, solver's adaptive time stepper will cut dt to hit time points exactly.
    position : Union[str, FieldPosition]
        Evaluation position ('INTEGRATION_POINTS', 'CENTROID', 'NODES', 'ELEMENT_NODES').
    region : Optional[str]
        Set name (e.g. 'ELSET-1' or 'NSET-1') or None for whole model.
    modes : str
        'ALL' or 'LAST_INCREMENT'.
    condition : Optional[Callable[[Any], bool]]
        Custom event trigger callback returning True when high-rate output is wanted.
    """
    name: str
    create_step_name: str = "Step-1"
    variables: List[str] = field(default_factory=lambda: ["U", "RF", "S", "LE", "PEEQ"])
    frequency: Optional[int] = 1
    num_intervals: Optional[int] = None
    time_interval: Optional[float] = None
    time_points: Optional[List[float]] = None
    exact_time_points: bool = True
    position: Union[str, FieldPosition] = FieldPosition.INTEGRATION_POINTS
    region: Optional[str] = None
    modes: str = "ALL"
    condition: Optional[Callable[[Any], bool]] = None

    def __post_init__(self) -> None:
        if isinstance(self.position, str):
            self.position = FieldPosition(self.position.upper())
        self.modes = self.modes.upper()
        if self.time_points is not None:
            self.time_points = sorted(list(self.time_points))
        
        # Normalize variable names and expand aliases / PRESELECT
        normalized: List[str] = []
        alias_map = {
            "VONMISES": "MISES",
            "VON_MISES": "MISES",
            "MAX_PRINCIPAL": "MAXP",
            "MIN_PRINCIPAL": "MINP",
            "MID_PRINCIPAL": "MIDP",
            "PRESSURE": "PRESS",
        }
        for v in self.variables:
            v_up = str(v).upper()
            if v_up == "PRESELECT":
                normalized.extend(["U", "RF", "S", "LE", "PEEQ", "CPRESS", "COPEN", "CSTATUS", "CSHEAR"])
            else:
                normalized.append(alias_map.get(v_up, v_up))
        self.variables = list(dict.fromkeys(normalized))


@dataclass
class HistoryOutputRequest:
    """Abaqus-compatible History Output Request.
    
    Parameters
    ----------
    name : str
        Identifier name (e.g. 'H-Output-1').
    create_step_name : str
        Name of the step where this request is defined.
    variables : Sequence[str]
        Energy, nodal, or section variables (e.g. ['ALLIE', 'ALLKE', 'ALLSE', 'ALLWK', 'ETOTAL']).
    frequency : Optional[int]
        Output every N increments (default: 1).
    num_intervals : Optional[int]
        Evenly space output over N intervals in step time period.
    time_interval : Optional[float]
        Output every fixed delta_t.
    time_points : Optional[Sequence[float]]
        Explicit list of time points at which output is requested.
    region : Optional[str]
        Node set, element set, or None ('WholeModel').
    section_name : Optional[str]
        Section name if monitoring section forces/moments (SOF, SOM).
    """
    name: str
    create_step_name: str = "Step-1"
    variables: List[str] = field(default_factory=lambda: ["ALLIE", "ALLSE", "ALLPD", "ALLVD", "ALLKE", "ALLWK", "ETOTAL"])
    frequency: Optional[int] = 1
    num_intervals: Optional[int] = None
    time_interval: Optional[float] = None
    time_points: Optional[List[float]] = None
    region: Optional[str] = None
    section_name: Optional[str] = None

    def __post_init__(self) -> None:
        normalized: List[str] = []
        for v in self.variables:
            v_up = str(v).upper()
            if v_up == "PRESELECT":
                normalized.extend(["ALLIE", "ALLSE", "ALLPD", "ALLVD", "ALLKE", "ALLWK", "ALLAE", "ALLSD", "ALLFD", "ETOTAL"])
            else:
                normalized.append(v_up)
        self.variables = list(dict.fromkeys(normalized))



class OutputTimingController:
    """Controls whether field/history output should be sampled at a given time and increment."""

    def __init__(
        self,
        request: Union[FieldOutputRequest, HistoryOutputRequest],
        step_time_period: float = 1.0
    ):
        self.request = request
        self.time_period = float(step_time_period)
        self.last_output_time: float = -1e9
        self.last_output_inc: int = -1
        self._target_times: Optional[np.ndarray] = None

        # Build target times if num_intervals is set
        time_points = getattr(self.request, "time_points", None)
        if self.request.num_intervals is not None and self.request.num_intervals > 0:
            self._target_times = np.linspace(0.0, self.time_period, self.request.num_intervals + 1)
        elif time_points is not None and len(time_points) > 0:
            self._target_times = np.array(time_points, dtype=np.float64)

    def next_exact_time(self, step_time: float, eps: float = 1e-9) -> Optional[float]:
        """Next time the solver must land on exactly (Abaqus ``*OUTPUT, TIME MARKS=YES``, the
        documented default): the next explicit time point / interval boundary after `step_time`.
        None when the request does not ask for exact times or has no time-based trigger."""
        if not getattr(self.request, "exact_time_points", True):
            return None
        cands = []
        if self._target_times is not None:
            later = self._target_times[self._target_times > step_time + eps]
            if later.size:
                cands.append(float(later.min()))
        ti = getattr(self.request, "time_interval", None)
        if ti is not None and ti > 0.0:
            k = np.floor((step_time + eps) / ti) + 1.0
            cands.append(float(k * ti))
        cands = [c for c in cands if c <= self.time_period + eps]
        return min(cands) if cands else None

    def should_output(
        self,
        step_time: float,
        inc: int,
        is_last_step: bool = False,
        solver: Optional[Any] = None
    ) -> bool:
        """Evaluate whether to trigger output at current state."""
        req = self.request

        # 1. Custom event condition check
        if getattr(req, "condition", None) is not None and solver is not None:
            if req.condition(solver):
                return True

        # 2. LAST_INCREMENT mode check
        if getattr(req, "modes", "ALL") == "LAST_INCREMENT":
            return is_last_step

        # 3. Always output at t=0 (inc=0) and final converged step
        if inc == 0 or is_last_step:
            return True

        # 4. Target times matching (num_intervals or explicit time_points)
        if self._target_times is not None:
            # Check if we crossed or closely matched any target time since last_output_time
            matched = np.any((self._target_times > self.last_output_time) & (self._target_times <= step_time + 1e-9))
            if matched:
                return True

        # 5. Fixed time_interval matching
        if req.time_interval is not None and req.time_interval > 0.0:
            if step_time - self.last_output_time >= req.time_interval - 1e-9:
                return True

        # 6. Frequency-based increment matching (fallback/default)
        if req.frequency is not None and req.frequency > 0:
            if inc % req.frequency == 0:
                return True

        return False

    def mark_recorded(self, step_time: float, inc: int) -> None:
        """Mark output recorded at this time and increment."""
        self.last_output_time = step_time
        self.last_output_inc = inc
