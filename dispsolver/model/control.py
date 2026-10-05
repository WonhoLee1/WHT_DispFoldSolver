"""Abaqus-compatible Solution Controls and Active Tolerance Adaptation Engine.

Implements:
1. FieldControl: Abaqus *CONTROLS, PARAMETERS=FIELD (R_n, C_n, q_0, c_0, R_p, C_a)
2. TimeIncrementationControl: Abaqus *CONTROLS, PARAMETERS=TIME INCREMENTATION (I_0, I_R, I_P, I_C, I_L, I_G, D_G, D_C)
3. LineSearchControl: Abaqus *CONTROLS, PARAMETERS=LINE SEARCH (N_ls, s_ls, f_ls, eta_ls)
4. ActiveToleranceEngine: Dynamic active tolerance adaptation to prevent cutbacks and speed up convergence
5. SolutionControl: Unified container with 4 presets: STANDARD, RELAXED, CONCEPTUAL, ADAPTIVE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union


@dataclass
class FieldControl:
    """Abaqus Field Convergence Controls (*CONTROLS, PARAMETERS=FIELD).

    Parameters
    ----------
    R_n : float
        Convergence criterion for the ratio of the largest residual to the corresponding
        average flux norm. Default = 0.005 (0.5%).
    C_n : float
        Convergence criterion for the ratio of the largest solution correction to the largest
        corresponding incremental solution. Default = 0.01 (1.0%).
    q_0 : Optional[float]
        Initial value of the time-average flux. Default = None.
    c_0 : Optional[float]
        User-defined initial value of the average solution correction. Default = None.
    R_p : float
        Alternative residual criterion tolerance when solution correction is small (C_a).
        Default = 0.02 (2.0%).
    C_a : float
        Alternative correction criterion tolerance. When max_correction <= C_a * max_increment,
        the increment is accepted if max_residual <= R_p * q_avg. Default = 0.001 (0.1%).
    """
    R_n: float = 0.005
    C_n: float = 0.01
    q_0: Optional[float] = None
    c_0: Optional[float] = None
    R_p: float = 0.02
    C_a: float = 0.001

    def to_tuple(self) -> Tuple[Any, ...]:
        return (self.R_n, self.C_n, self.q_0, self.c_0, self.R_p, self.C_a)

    @classmethod
    def from_tuple(cls, values: Tuple[Any, ...]) -> FieldControl:
        args = list(values) + [None] * (6 - len(values))
        return cls(
            R_n=float(args[0]) if args[0] is not None else 0.005,
            C_n=float(args[1]) if args[1] is not None else 0.01,
            q_0=float(args[2]) if args[2] is not None else None,
            c_0=float(args[3]) if args[3] is not None else None,
            R_p=float(args[4]) if args[4] is not None else 0.02,
            C_a=float(args[5]) if args[5] is not None else 0.001,
        )


@dataclass
class TimeIncrementationControl:
    """Abaqus Time Incrementation Controls (*CONTROLS, PARAMETERS=TIME INCREMENTATION).

    Parameters
    ----------
    I_0 : int
        Equilibrium iteration limit after which time increment is cut back. Default = 16.
    I_R : int
        Number of consecutive iterations before checking convergence rate. Default = 16.
    I_P : int
        Iteration number after which alternative convergence check is attempted. Default = 9.
    I_C : int
        Maximum number of cutbacks allowed in an increment. Default = 25.
    I_L : int
        Number of consecutive successful increments required before increasing dt. Default = 2.
    I_G : int
        Number of iterations below which dt can be increased. Default = 4.
    D_G : float
        Factor by which time increment size is multiplied when expanding dt. Default = 1.25.
    D_C : float
        Factor by which time increment size is multiplied during a cutback. Default = 0.25.
    """
    I_0: int = 16
    I_R: int = 16
    I_P: int = 9
    I_C: int = 25
    I_L: int = 2
    I_G: int = 4
    D_G: float = 1.25
    D_C: float = 0.25

    def to_tuple(self) -> Tuple[Any, ...]:
        return (self.I_0, self.I_R, self.I_P, self.I_C, self.I_L, self.I_G, self.D_G, self.D_C)

    @classmethod
    def from_tuple(cls, values: Tuple[Any, ...]) -> TimeIncrementationControl:
        args = list(values) + [None] * (8 - len(values))
        return cls(
            I_0=int(args[0]) if args[0] is not None else 16,
            I_R=int(args[1]) if args[1] is not None else 16,
            I_P=int(args[2]) if args[2] is not None else 9,
            I_C=int(args[3]) if args[3] is not None else 25,
            I_L=int(args[4]) if args[4] is not None else 2,
            I_G=int(args[5]) if args[5] is not None else 4,
            D_G=float(args[6]) if args[6] is not None else 1.25,
            D_C=float(args[7]) if args[7] is not None else 0.25,
        )


@dataclass
class LineSearchControl:
    """Abaqus Line Search Controls (*CONTROLS, PARAMETERS=LINE SEARCH).

    Parameters
    ----------
    N_ls : int
        Maximum number of line search iterations. Default = 4.
    s_ls : float
        Line search scale factor / tolerance. Default = 0.001.
    f_ls : float
        Line search upper energy bound. Default = 1.01.
    eta_ls : float
        Line search residual tolerance. Default = 0.25.
    """
    N_ls: int = 4
    s_ls: float = 0.001
    f_ls: float = 1.01
    eta_ls: float = 0.25

    def to_tuple(self) -> Tuple[Any, ...]:
        return (self.N_ls, self.s_ls, self.f_ls, self.eta_ls)

    @classmethod
    def from_tuple(cls, values: Tuple[Any, ...]) -> LineSearchControl:
        args = list(values) + [None] * (4 - len(values))
        return cls(
            N_ls=int(args[0]) if args[0] is not None else 4,
            s_ls=float(args[1]) if args[1] is not None else 0.001,
            f_ls=float(args[2]) if args[2] is not None else 1.01,
            eta_ls=float(args[3]) if args[3] is not None else 0.25,
        )


class ActiveToleranceEngine:
    """Engine that dynamically adapts convergence tolerances to accelerate solve speed.

    When adaptive convergence is enabled:
    1. During normal smooth progression, standard strict tolerances (R_n=0.005, C_n=0.01)
       are maintained to preserve physical fidelity and prevent state variable drift.
    2. Abaqus Alternative Convergence (C_a=0.001, R_p=0.02) is fully leveraged to truncate
       redundant 3~5 tail iterations per increment when displacement correction is already minuscule.
    3. If local severe stalling or cutback occurs, tolerances are intelligently and temporarily
       relaxed up to max_relax_factor, enabling the solver to pass through the bottleneck without
       wasting multiple 16-iteration cutback cycles.
    4. Upon successful convergence of the increment, tolerances immediately reset to base standards.
    """

    def __init__(self, R_n_base: float = 0.005, C_n_base: float = 0.01, max_relax_factor: float = 5.0):
        self.R_n_base = float(R_n_base)
        self.C_n_base = float(C_n_base)
        self.max_relax_factor = float(max_relax_factor)
        self.is_active = False

    def compute_tolerances(
        self,
        cutback_count: int = 0,
        iter_count: int = 0,
        is_stalling: bool = False
    ) -> Tuple[float, float, float, float]:
        """Compute effective tolerances (R_n, C_n, R_p, C_a) for current iteration attempt.

        Returns
        -------
        Tuple[float, float, float, float]
            (R_n_eff, C_n_eff, R_p_eff, C_a_eff)
        """
        if not self.is_active:
            return (self.R_n_base, self.C_n_base, self.R_n_base * 4.0, self.C_n_base * 0.1)

        relax = 1.0
        if cutback_count > 0:
            relax += 1.0 * cutback_count
        elif is_stalling and iter_count >= 6:
            relax += 0.5 * (iter_count - 5)

        relax = min(relax, self.max_relax_factor)

        R_n_eff = self.R_n_base * relax
        C_n_eff = self.C_n_base * relax
        R_p_eff = max(R_n_eff * 2.0, 0.02)
        C_a_eff = max(C_n_eff * 0.1, 0.001)

        return (R_n_eff, C_n_eff, R_p_eff, C_a_eff)


class SolutionControl:
    """Top-level Abaqus Solution Control container with 4-tier Presets and API Parity.

    Presets:
    --------
    1. 'STANDARD': Default Abaqus/Standard (R_n=0.005, C_n=0.01, I_0=16, I_C=25)
    2. 'RELAXED': Discontinuous / Contact Chattering (R_n=0.02, C_n=0.05, I_0=16, I_C=25)
    3. 'CONCEPTUAL': Ultra-fast Coarse Screening (R_n=0.05, C_n=0.10, I_0=25, I_C=30)
    4. 'ADAPTIVE': High-speed Dynamic Adaptation (Standard strict base + Alternative Checks + Dynamic Relaxation)
    """

    def __init__(
        self,
        name: str = "DefaultSolutionControl",
        preset: str = "STANDARD",
        adaptiveConvergence: bool = False,
        maxRelaxFactor: float = 5.0,
        **kwargs
    ):
        self.name = str(name)
        self.preset = str(preset).upper()
        self.adaptive_convergence = bool(adaptiveConvergence)
        self.max_relax_factor = float(maxRelaxFactor)

        self.field = FieldControl()
        self.time_incrementation = TimeIncrementationControl()
        self.line_search = LineSearchControl()

        self.tolerance_engine = ActiveToleranceEngine(
            R_n_base=self.field.R_n,
            C_n_base=self.field.C_n,
            max_relax_factor=self.max_relax_factor
        )

        self.apply_preset(self.preset)
        if kwargs:
            self.setValues(**kwargs)

    def apply_preset(self, preset_name: str) -> None:
        """Apply one of the 4 standard solution control presets."""
        p = str(preset_name).upper()
        self.preset = p

        if p == "STANDARD":
            self.field = FieldControl(R_n=0.005, C_n=0.01, R_p=0.02, C_a=0.001)
            self.time_incrementation = TimeIncrementationControl(I_0=16, I_C=25, I_L=2, D_G=1.25, D_C=0.25)
            self.adaptive_convergence = False
        elif p == "RELAXED" or p == "ROUGH":
            self.field = FieldControl(R_n=0.02, C_n=0.05, R_p=0.05, C_a=0.005)
            self.time_incrementation = TimeIncrementationControl(I_0=16, I_C=25, I_L=2, D_G=1.25, D_C=0.25)
            self.adaptive_convergence = False
        elif p == "CONCEPTUAL":
            self.field = FieldControl(R_n=0.05, C_n=0.10, R_p=0.10, C_a=0.01)
            self.time_incrementation = TimeIncrementationControl(I_0=25, I_C=30, I_L=1, D_G=1.50, D_C=0.50)
            self.adaptive_convergence = False
        elif p == "ADAPTIVE":
            self.field = FieldControl(R_n=0.005, C_n=0.01, R_p=0.02, C_a=0.001)
            self.time_incrementation = TimeIncrementationControl(I_0=16, I_C=25, I_L=2, D_G=1.25, D_C=0.25)
            self.adaptive_convergence = True
        else:
            raise ValueError(f"Unknown solution control preset '{preset_name}'. Supported: 'STANDARD', 'RELAXED', 'CONCEPTUAL', 'ADAPTIVE'.")

        self.tolerance_engine.R_n_base = self.field.R_n
        self.tolerance_engine.C_n_base = self.field.C_n
        self.tolerance_engine.max_relax_factor = self.max_relax_factor
        self.tolerance_engine.is_active = self.adaptive_convergence

    def setValues(self, **kwargs) -> None:
        """Abaqus Python API compatible parameter configuration.

        Accepts:
        - preset: 'STANDARD' | 'RELAXED' | 'CONCEPTUAL' | 'ADAPTIVE'
        - adaptiveConvergence: bool
        - maxRelaxFactor: float
        - displacementField: Tuple or FieldControl
        - timeIncrementation: Tuple or TimeIncrementationControl
        - lineSearch: Tuple or LineSearchControl
        - R_n, C_n, R_p, C_a: Individual field tolerances
        - I_0, I_C, I_L, D_G, D_C: Individual time incrementation parameters
        """
        if "preset" in kwargs:
            self.apply_preset(kwargs.pop("preset"))

        if "adaptiveConvergence" in kwargs:
            self.adaptive_convergence = bool(kwargs.pop("adaptiveConvergence"))
            self.tolerance_engine.is_active = self.adaptive_convergence

        if "maxRelaxFactor" in kwargs:
            self.max_relax_factor = float(kwargs.pop("maxRelaxFactor"))
            self.tolerance_engine.max_relax_factor = self.max_relax_factor

        # Abaqus tuple styles
        if "displacementField" in kwargs:
            val = kwargs.pop("displacementField")
            if isinstance(val, (tuple, list)):
                self.field = FieldControl.from_tuple(tuple(val))
            elif isinstance(val, FieldControl):
                self.field = val

        if "timeIncrementation" in kwargs:
            val = kwargs.pop("timeIncrementation")
            if isinstance(val, (tuple, list)):
                self.time_incrementation = TimeIncrementationControl.from_tuple(tuple(val))
            elif isinstance(val, TimeIncrementationControl):
                self.time_incrementation = val

        if "lineSearch" in kwargs:
            val = kwargs.pop("lineSearch")
            if isinstance(val, (tuple, list)):
                self.line_search = LineSearchControl.from_tuple(tuple(val))
            elif isinstance(val, LineSearchControl):
                self.line_search = val

        # Individual field parameters
        for attr in ("R_n", "C_n", "q_0", "c_0", "R_p", "C_a"):
            if attr in kwargs:
                v = kwargs.pop(attr)
                setattr(self.field, attr, float(v) if v is not None else None)

        # Individual time incrementation parameters
        for attr in ("I_0", "I_R", "I_P", "I_C", "I_L", "I_G"):
            if attr in kwargs:
                setattr(self.time_incrementation, attr, int(kwargs.pop(attr)))
        for attr in ("D_G", "D_C"):
            if attr in kwargs:
                setattr(self.time_incrementation, attr, float(kwargs.pop(attr)))

        # Update engine base values
        self.tolerance_engine.R_n_base = self.field.R_n
        self.tolerance_engine.C_n_base = self.field.C_n

    def get_effective_tolerances(
        self,
        cutback_count: int = 0,
        iter_count: int = 0,
        is_stalling: bool = False
    ) -> Tuple[float, float, float, float]:
        """Get (R_n, C_n, R_p, C_a) for current iteration attempt."""
        return self.tolerance_engine.compute_tolerances(
            cutback_count=cutback_count,
            iter_count=iter_count,
            is_stalling=is_stalling
        )

    def to_inp_snippet(self) -> str:
        """Generate Abaqus input deck keywords corresponding to these controls."""
        lines = []
        # Time incrementation
        t = self.time_incrementation
        lines.append("*CONTROLS, PARAMETERS=TIME INCREMENTATION")
        lines.append(f"  {t.I_0}, {t.I_R}, {t.I_P}, {t.I_C}, {t.I_L}, {t.I_G}, {t.D_G}, {t.D_C}")

        # Field controls
        f = self.field
        conv_check = "RELAXED" if (self.preset in ["RELAXED", "CONCEPTUAL"]) else "STANDARD"
        lines.append(f"*CONTROLS, PARAMETERS=FIELD, CONVERGENCE CHECK={conv_check}")
        q_0_str = f"{f.q_0}" if f.q_0 is not None else ""
        c_0_str = f"{f.c_0}" if f.c_0 is not None else ""
        lines.append(f"  {f.R_n}, {f.C_n}, {q_0_str}, {c_0_str}, {f.R_p}, {f.C_a}")

        # Line search
        ls = self.line_search
        lines.append("*CONTROLS, PARAMETERS=LINE SEARCH")
        lines.append(f"  {ls.N_ls}, {ls.s_ls}, {ls.f_ls}, {ls.eta_ls}")

        return "\n".join(lines)
