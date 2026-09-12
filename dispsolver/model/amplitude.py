"""Amplitude definitions for Abaqus-like time-dependent loading and user-defined kinematic functions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any
import numpy as np


@dataclass
class Amplitude:
    """Base class for time-dependent amplitude curves (*AMPLITUDE in Abaqus)."""
    name: str

    def evaluate(self, t: float, step_time: float = 0.0, total_time: float = 0.0) -> float:
        """Evaluate amplitude scale factor at time t."""
        return 1.0


@dataclass
class TabularAmplitude(Amplitude):
    """Tabular amplitude with linear interpolation between time-value pairs (*AMPLITUDE, TYPE=TABULAR)."""
    data: List[Tuple[float, float]] = field(default_factory=list)  # [(time, value), ...]

    def evaluate(self, t: float, step_time: float = 0.0, total_time: float = 0.0) -> float:
        if not self.data:
            return 1.0
        times = [pt[0] for pt in self.data]
        values = [pt[1] for pt in self.data]
        return float(np.interp(t, times, values))


@dataclass
class SmoothStepAmplitude(Amplitude):
    """Smooth step amplitude curve using C² continuous polynomial interpolation (*AMPLITUDE, TYPE=SMOOTH STEP).
    
    Abaqus SmoothStep polynomial:
        s(tau) = 10 * tau^3 - 15 * tau^4 + 6 * tau^5
    where tau = (t - t_i) / (t_{i+1} - t_i).
    Produces zero first and second derivatives at endpoints, preventing initial/final accelerations.
    """
    data: List[Tuple[float, float]] = field(default_factory=list)  # [(time, value), ...]

    def evaluate(self, t: float, step_time: float = 0.0, total_time: float = 0.0) -> float:
        if not self.data:
            return 1.0
        if len(self.data) == 1:
            return float(self.data[0][1])

        times = [pt[0] for pt in self.data]
        values = [pt[1] for pt in self.data]

        if t <= times[0]:
            return float(values[0])
        if t >= times[-1]:
            return float(values[-1])

        # Find interval [times[i], times[i+1]]
        idx = 0
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                idx = i
                break

        t0, t1 = times[idx], times[idx + 1]
        v0, v1 = values[idx], values[idx + 1]

        if abs(t1 - t0) < 1e-14:
            return float(v1)

        tau = (t - t0) / (t1 - t0)
        s_tau = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        return float(v0 + (v1 - v0) * s_tau)


@dataclass
class UserFunctionAmplitude(Amplitude):
    """User-defined function amplitude supporting Python callable or Numba @njit C-kernel function."""
    py_func: Optional[Callable[[float, float, float], float]] = None
    numba_func: Optional[Any] = None

    def evaluate(self, t: float, step_time: float = 0.0, total_time: float = 0.0) -> float:
        if self.numba_func is not None:
            return float(self.numba_func(t, step_time, total_time))
        if self.py_func is not None:
            return float(self.py_func(t, step_time, total_time))
        return 1.0
