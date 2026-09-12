"""
dt_controller.py
================
Physics-informed adaptive time-stepping controller.
"""

import numpy as np


class AdaptiveDtController:
    """Adaptive time step controller based on iteration count and convergence rate.

    Parameters
    ----------
    dt_init : float
        Initial time step size.
    dt_min : float
        Minimum allowable time step size.
    dt_max : float
        Maximum allowable time step size.
    target_iters : int
        Target number of Newton iterations per step (default 5).
    """

    def __init__(
        self,
        dt_init: float = 0.01,
        dt_min: float = 1e-6,
        dt_max: float = 0.1,
        target_iters: int = 20,
    ):
        self.dt = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.target_iters = target_iters

    def update(self, n_iter: int, converged: bool, conv_rate: float = 1.0) -> float:
        """Update and return the next time step size dt.

        Parameters
        ----------
        n_iter : int
            Number of iterations taken in the current step.
        converged : bool
            True if the step converged, False if cutback required.
        conv_rate : float
            Ratio of final residual to previous residual.

        Returns
        -------
        dt_next : float
            Adjusted time step size.
        """
        if not converged:
            if conv_rate > 1.5:
                # Severe divergence -> sharp reduction
                self.dt *= 0.25
            elif conv_rate > 1.0:
                # Slow divergence -> moderate reduction
                self.dt *= 0.4
            else:
                # Stall -> standard cutback
                self.dt *= 0.5
        else:
            # Step succeeded -> scale based on target iteration count
            ratio = self.target_iters / max(n_iter, 1)
            growth = min(max(ratio, 0.85), 1.5)
            self.dt *= growth

        self.dt = float(np.clip(self.dt, self.dt_min, self.dt_max))
        return self.dt

    def notify_success(self, n_iter: int) -> float:
        """Notify successful convergence and update dt."""
        return self.update(n_iter, True)

    def notify_cutback(self, conv_rate: float = 1.0) -> bool:
        """Notify cutback/divergence and reduce dt. Returns False if dt drops below dt_min."""
        prev_dt = self.dt
        self.update(1, False, conv_rate=conv_rate)
        return self.dt > self.dt_min
