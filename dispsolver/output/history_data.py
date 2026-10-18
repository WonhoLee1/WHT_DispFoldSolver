"""HistoryOutput, HistoryRegion, and EnergyTracker containers for Abaqus parity."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence, Any, Union
import numpy as np


class HistoryOutput:
    """Abaqus-compatible History Output container for a scalar time-history curve."""

    def __init__(
        self,
        name: str,
        description: str = "",
        times: Optional[Sequence[float]] = None,
        data: Optional[Sequence[float]] = None
    ):
        self.name = str(name)
        self.description = str(description)
        self._times: List[float] = list(times) if times is not None else []
        self._data: List[float] = list(data) if data is not None else []

    def append(self, time: float, value: float) -> None:
        """Add a (time, value) sample point."""
        self._times.append(float(time))
        self._data.append(float(value))

    @property
    def times(self) -> np.ndarray:
        """Return 1D numpy array of time points."""
        return np.asarray(self._times, dtype=np.float64)

    @property
    def values(self) -> np.ndarray:
        """Return 1D numpy array of values."""
        return np.asarray(self._data, dtype=np.float64)

    @property
    def data(self) -> Tuple[Tuple[float, float], ...]:
        """Abaqus-compatible sequence of (time, value) tuples."""
        return tuple(zip(self._times, self._data))

    def __len__(self) -> int:
        return len(self._times)


class HistoryRegion:
    """Abaqus-compatible HistoryRegion container (e.g. 'WholeModel', 'Node SET-1')."""

    def __init__(self, name: str, description: str = ""):
        self.name = str(name)
        self.description = str(description)
        self.historyOutputs: Dict[str, HistoryOutput] = {}

    def get_or_create(self, var_name: str, description: str = "") -> HistoryOutput:
        """Retrieve existing HistoryOutput or create new one."""
        if var_name not in self.historyOutputs:
            self.historyOutputs[var_name] = HistoryOutput(name=var_name, description=description)
        return self.historyOutputs[var_name]

    def record(self, var_name: str, time: float, value: float) -> None:
        """Record a single scalar point for variable at time."""
        ho = self.get_or_create(var_name)
        ho.append(time, value)

    def to_dataframe(self):
        """Convert all history outputs in this region to a Pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for HistoryRegion.to_dataframe().")

        if not self.historyOutputs:
            return pd.DataFrame()

        # Find reference times from first non-empty history output
        ref_ho = next(iter(self.historyOutputs.values()))
        times = ref_ho.times

        data_dict = {"Time": times}
        for name, ho in self.historyOutputs.items():
            vals = ho.values
            if len(vals) == len(times):
                data_dict[name] = vals
            else:
                # Interpolate if sampling points differ
                data_dict[name] = np.interp(times, ho.times, vals)

        return pd.DataFrame(data_dict)

    def plot(self, variables: Optional[Sequence[str]] = None, title: Optional[str] = None, **kwargs):
        """Plot time-history curves using matplotlib."""
        import matplotlib.pyplot as plt
        try:
            import koreanize_matplotlib
        except ImportError:
            pass

        df = self.to_dataframe()
        if df.empty:
            print(f"[Warning] HistoryRegion '{self.name}' has no data to plot.")
            return None

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 4.5)), dpi=kwargs.get("dpi", 120))
        cols = [c for c in df.columns if c != "Time"]
        if variables is not None:
            cols = [c for c in cols if c in variables]

        for c in cols:
            ax.plot(df["Time"], df[c], label=c, lw=1.8)

        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.set_title(title if title else f"History Outputs - {self.name}", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        return fig, ax


class EnergyTracker:
    """Tracks and accumulates whole-model energy quantities across increments."""

    def __init__(self):
        self.allie: float = 0.0   # Total internal energy
        self.allse: float = 0.0   # Strain energy (elastic)
        self.allpd: float = 0.0   # Plastic dissipation
        self.allvd: float = 0.0   # Viscous dissipation
        self.allke: float = 0.0   # Kinetic energy
        self.allwk: float = 0.0   # External work
        self.allae: float = 0.0   # Artificial strain energy (hourglass / numerical damping)
        self.allsd: float = 0.0   # Stabilization dissipation energy
        self.allfd: float = 0.0   # Frictional dissipation energy

    def update_step(
        self,
        dt: float,
        solver: Any,
        du: np.ndarray,
        f_int: np.ndarray,
        f_ext: Optional[np.ndarray] = None
    ) -> None:
        """Update energy components for current converged increment."""
        # Incremental internal work dW_int = 0.5 * (f_int_old + f_int_new) . du
        # For simplicity in quasi-static: dW_int approx f_int . du
        dW_int = float(np.dot(f_int, du))
        self.allie += max(dW_int, 0.0)

        # Recoverable strain energy approximation from elastic stiffness:
        # If solver has strain_energy attribute or computation:
        if hasattr(solver, "compute_strain_energy"):
            try:
                self.allse = float(solver.compute_strain_energy())
            except Exception:
                self.allse = self.allie - self.allpd
        else:
            self.allse = self.allie - self.allpd

        # External work dW_ext = f_ext . du
        if f_ext is not None:
            self.allwk += float(np.dot(f_ext, du))
        else:
            self.allwk = self.allie  # in equilibrium W_ext = W_int

        # Query solver/elements for artificial strain energy (ALLAE) if exposed
        if hasattr(solver, "energy_allae"):
            self.allae = float(solver.energy_allae)
        elif hasattr(solver, "allae"):
            self.allae = float(solver.allae)

        # Query solver for stabilization dissipation energy (ALLSD) if exposed
        if hasattr(solver, "energy_allsd"):
            self.allsd = float(solver.energy_allsd)
        elif hasattr(solver, "allsd"):
            self.allsd = float(solver.allsd)

        # Query solver/constraints for frictional dissipation energy (ALLFD) if exposed
        if hasattr(solver, "energy_allfd"):
            self.allfd = float(solver.energy_allfd)
        elif hasattr(solver, "allfd"):
            self.allfd = float(solver.allfd)
        elif hasattr(solver, "constraints"):
            fd_total = 0.0
            for c in solver.constraints:
                if hasattr(c, "frictional_dissipation"):
                    fd_total += float(c.frictional_dissipation)
                elif hasattr(c, "energy_allfd"):
                    fd_total += float(c.energy_allfd)
            if fd_total > 0.0:
                self.allfd = fd_total

    @property
    def etotal(self) -> float:
        """Energy balance error: ALLIE + ALLKE - ALLWK."""
        return (self.allie + self.allke) - self.allwk

