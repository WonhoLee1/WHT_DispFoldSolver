"""
amplitude.py
============
Time-dependent load/BC amplitude curves — equivalent to Abaqus *AMPLITUDE.

An Amplitude maps analysis time -> scalar multiplier. A boundary condition or
load magnitude at time t is  base_value * amplitude(t).

Interpolation methods
---------------------
- "linear" (default, Abaqus RAMP): piecewise-linear between table points.
- "smooth" (Abaqus SMOOTH STEP): 5th-order polynomial with zero first/second
  derivatives at each point — smooth acceleration, avoids inertia spikes.
- "step": piecewise-constant (value held until the next time point).

Outside the table range the endpoint values are held constant.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class Amplitude:
    """Tabular time-amplitude curve with selectable interpolation."""

    def __init__(self, times: Sequence[float], values: Sequence[float], method: str = "linear"):
        t = np.asarray(times, dtype=np.float64).ravel()
        v = np.asarray(values, dtype=np.float64).ravel()
        if t.shape != v.shape:
            raise ValueError("times and values must have the same length")
        if t.size < 2:
            raise ValueError("Amplitude needs at least two table points")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if method not in ("linear", "smooth", "step"):
            raise ValueError(f"unknown method '{method}'")
        self.t = t
        self.v = v
        self.method = method

    def __call__(self, time: float) -> float:
        t, v = self.t, self.v
        if time <= t[0]:
            return float(v[0])
        if time >= t[-1]:
            return float(v[-1])

        # interval index i such that t[i] <= time < t[i+1]
        i = int(np.searchsorted(t, time, side="right") - 1)
        t0, t1 = t[i], t[i + 1]
        v0, v1 = v[i], v[i + 1]
        xi = (time - t0) / (t1 - t0)

        if self.method == "step":
            return float(v0)
        if self.method == "smooth":
            f = xi ** 3 * (10.0 - 15.0 * xi + 6.0 * xi ** 2)
        else:  # linear
            f = xi
        return float(v0 + (v1 - v0) * f)

    def __repr__(self) -> str:
        return (f"Amplitude({self.method}, t=[{self.t[0]:.3g}..{self.t[-1]:.3g}], "
                f"{self.t.size} pts)")
