"""
abaqus_config.py
================
Abaqus-style FEA solver configuration engine for display folding simulations.

Maps solver tuning parameters directly to Abaqus/Standard keyword specifications:
  - *STEP, NLGEOM=YES
  - *DYNAMIC, APPLICATION=MODERATE DISSIPATION, ALPHA=... (or *STATIC)
  - *CONTROLS, PARAMETERS=TIME INCREMENTATION (I_0, I_R, I_P, I_C, I_L, ...)
  - *CONTROLS, PARAMETERS=FIELD (R_n, C_n, rtol, atol, max_disp_corr)
  - *SECTION CONTROLS, DISTORTION CONTROL=YES, LENGTH RATIO=...
  - *STABILIZE, FACTOR=...

Provides bidirectional Abaqus .inp snippet export/import, JSON/YAML serialization,
dot-notation key path manipulation (e.g. "time_incrementation.I_C"), and clean
conversion to DynamicSolver initialization keyword arguments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple


@dataclass
class AbaqusStepConfig:
    """Abaqus *STEP analysis step configuration.

    Parameters
    ----------
    procedure : str
        Analysis procedure type ("DYNAMIC", "STATIC", or "VISCO").
    application : str
        Dynamic application preset ("MODERATE DISSIPATION", "QUASI-STATIC", "TRANSIENT FIDELITY").
    nlgeom : bool
        Geometric nonlinearity flag (*STEP, NLGEOM=YES).
    t_total : float
        Total time period for the step (seconds).
    dt_init : float
        Initial time increment size (seconds).
    dt_min : float
        Minimum allowed time increment size (seconds).
    dt_max : float
        Maximum allowed time increment size (seconds).
    """
    procedure: str = "DYNAMIC"
    application: str = "MODERATE DISSIPATION"
    nlgeom: bool = True
    t_total: float = 1.0
    dt_init: float = 0.0025
    dt_min: float = 1e-7
    dt_max: float = 0.1


@dataclass
class AbaqusTimeIncrementationControls:
    """Abaqus *CONTROLS, PARAMETERS=TIME INCREMENTATION parameters.

    Parameters
    ----------
    I_0 : int
        Equilibrium iteration limit before cutback (default in Abaqus is 4; increased to 10 for difficult nonlinear solves).
    I_R : int
        Iteration count threshold for severe discontinuity check (default 16).
    I_P : int
        Max severe discontinuity iterations per increment (default 9).
    I_C : int
        Max cutback attempts allowed per increment (default in Abaqus is 5; set to 20 for complex contact/folding).
    I_L : int
        Consecutive low-iteration increments required before increasing dt (default 4).
    I_G : int
        Max iterations in consecutive increments before decreasing dt (default 10).
    dt_growth_factor : float
        Multiplier for increasing dt after consecutive easy increments (default 1.5).
    dt_cutback_factor : float
        Multiplier for reducing dt on cutback (default 0.25).
    """
    I_0: int = 10
    I_R: int = 16
    I_P: int = 9
    I_C: int = 20
    I_L: int = 4
    I_G: int = 10
    dt_growth_factor: float = 1.5
    dt_cutback_factor: float = 0.25


@dataclass
class AbaqusFieldControls:
    """Abaqus *CONTROLS, PARAMETERS=FIELD convergence parameters.

    Parameters
    ----------
    R_n : float
        Convergence tolerance ratio for residual forces (default 0.005 = 0.5% average force).
    C_n : float
        Convergence tolerance ratio for displacement correction (default 0.01 = 1.0% max displacement increment).
    rtol : float
        Relative energy / KKT residual tolerance for solver accept condition (default 1e-4).
    atol : float
        Absolute displacement correction tolerance (default 1e-6 mm).
    max_disp_corr : Optional[float]
        Absolute cap on single-iteration trial step displacement correction (mm). Default 0.03 mm (1 PSA layer).
    """
    R_n: float = 0.005
    C_n: float = 0.01
    rtol: float = 1e-4
    atol: float = 1e-6
    max_disp_corr: Optional[float] = 0.03


@dataclass
class AbaqusDynamicControls:
    """Abaqus *DYNAMIC numerical integration and damping parameters.

    Parameters
    ----------
    integration_mode : str
        Integration mode preset ("moderate-4", "moderate-3", "generalized-0.8", "quasistatic", etc.).
    alpha : float
        HHT-alpha numerical damping parameter (-0.05 light, -0.22 moderate-4, -0.30 heavy).
    beta : Optional[float]
        Newmark beta parameter. If None, auto-calculated as (1 - alpha)^2 / 4.
    gamma : Optional[float]
        Newmark gamma parameter. If None, auto-calculated as 0.5 - alpha.
    galpha_rho_inf : Optional[float]
        OptiStruct PARAM, GALPHA spectral radius (0.0 to 1.0). If set, overrides HHT-alpha with Generalized-alpha.
    """
    integration_mode: str = "moderate-4"
    alpha: float = -0.22
    beta: Optional[float] = None
    gamma: Optional[float] = None
    galpha_rho_inf: Optional[float] = None

    def __post_init__(self):
        if self.beta is None:
            self.beta = (1.0 - self.alpha) ** 2 / 4.0
        if self.gamma is None:
            self.gamma = 0.5 - self.alpha


@dataclass
class AbaqusSectionControls:
    """Abaqus *SECTION CONTROLS element distortion control parameters.

    Parameters
    ----------
    distortion_control : bool
        Enable distortion control for soft/viscoelastic elements (*SECTION CONTROLS, DISTORTION CONTROL=YES).
    distortion_j_crit : float
        Critical element volume ratio threshold (J_crit). Default 0.20.
    """
    distortion_control: bool = True
    distortion_j_crit: float = 0.20


@dataclass
class AbaqusStabilizeControls:
    """Abaqus *STABILIZE automatic viscous damping parameters.

    Parameters
    ----------
    enabled : bool
        Enable automatic stabilization damping (*STABILIZE).
    factor : float
        Damping factor (dissipated energy fraction). Default 0.0002 (0.02%).
    """
    enabled: bool = False
    factor: float = 0.0002


@dataclass
class AbaqusSolverConfig:
    """Unified Abaqus-style FEA solver configuration container.

    Attributes
    ----------
    step : AbaqusStepConfig
        Step & procedure time bounds.
    time_incrementation : AbaqusTimeIncrementationControls
        Iteration limit and cutback parameters (*CONTROLS, PARAMETERS=TIME INCREMENTATION).
    field_controls : AbaqusFieldControls
        Convergence tolerance parameters (*CONTROLS, PARAMETERS=FIELD).
    dynamic_controls : AbaqusDynamicControls
        Numerical dissipation parameters (*DYNAMIC).
    section_controls : AbaqusSectionControls
        Element distortion control (*SECTION CONTROLS).
    stabilize : AbaqusStabilizeControls
        Viscous stabilization parameters (*STABILIZE).
    pardiso_mtype : str
        Linear solver matrix type ("auto", "spd", "indefinite", "nonsymmetric").
    pardiso_phase_reuse : bool
        Reuse symbolic factorization phase across Newton iterations.
    pet_element_type : str
        Element formulation for PET layers (default "Q4_COROTATIONAL_EAS").
    psa_element_type : str
        Element formulation for PSA layers (default "Q4_VISCO_SIMO").
    glass_element_type : str
        Element formulation for Glass layers (default "Q4_COROTATIONAL_EAS").
    theta_penalty_k : float
        Penalty stiffness for prescribed hinge rotation constraints (default 1e8).
    ul_mode : bool
        Updated Lagrangian kinematic frame mode (default False).
    """
    step: AbaqusStepConfig = field(default_factory=AbaqusStepConfig)
    time_incrementation: AbaqusTimeIncrementationControls = field(default_factory=AbaqusTimeIncrementationControls)
    field_controls: AbaqusFieldControls = field(default_factory=AbaqusFieldControls)
    dynamic_controls: AbaqusDynamicControls = field(default_factory=AbaqusDynamicControls)
    section_controls: AbaqusSectionControls = field(default_factory=AbaqusSectionControls)
    stabilize: AbaqusStabilizeControls = field(default_factory=AbaqusStabilizeControls)

    pardiso_mtype: str = "auto"
    pardiso_phase_reuse: bool = True

    pet_element_type: str = "Q4_COROTATIONAL_EAS"
    psa_element_type: str = "Q4_VISCO_SIMO"
    glass_element_type: str = "Q4_COROTATIONAL_EAS"

    theta_penalty_k: float = 1e8
    ul_mode: bool = False

    # ------------------------------------------------------------------
    # Serialization & Deserialization (Dict, JSON, YAML)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration instance to a recursive Python dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AbaqusSolverConfig:
        """Create configuration instance from a Python dict (handles partial dicts)."""
        d = dict(d or {})

        step = AbaqusStepConfig(**d.pop("step", {})) if "step" in d and isinstance(d["step"], dict) else AbaqusStepConfig()
        time_inc = AbaqusTimeIncrementationControls(**d.pop("time_incrementation", {})) if "time_incrementation" in d and isinstance(d["time_incrementation"], dict) else AbaqusTimeIncrementationControls()
        field_ctrl = AbaqusFieldControls(**d.pop("field_controls", {})) if "field_controls" in d and isinstance(d["field_controls"], dict) else AbaqusFieldControls()
        dyn_ctrl = AbaqusDynamicControls(**d.pop("dynamic_controls", {})) if "dynamic_controls" in d and isinstance(d["dynamic_controls"], dict) else AbaqusDynamicControls()
        sec_ctrl = AbaqusSectionControls(**d.pop("section_controls", {})) if "section_controls" in d and isinstance(d["section_controls"], dict) else AbaqusSectionControls()
        stab_ctrl = AbaqusStabilizeControls(**d.pop("stabilize", {})) if "stabilize" in d and isinstance(d["stabilize"], dict) else AbaqusStabilizeControls()

        # Handle legacy flat parameter overrides (e.g. max_cutbacks, max_iter, tol, rtol, atol)
        if "max_iter" in d:
            time_inc.I_0 = int(d.pop("max_iter"))
        if "max_cutbacks" in d:
            time_inc.I_C = int(d.pop("max_cutbacks"))
        if "tol" in d:
            field_ctrl.R_n = float(d.pop("tol"))
        if "rtol" in d:
            field_ctrl.rtol = float(d.pop("rtol"))
        if "atol" in d:
            field_ctrl.atol = float(d.pop("atol"))
        if "max_displacement_corr" in d:
            field_ctrl.max_disp_corr = d.pop("max_displacement_corr")
        if "integration_mode" in d:
            dyn_ctrl.integration_mode = str(d.pop("integration_mode"))
        if "alpha" in d:
            dyn_ctrl.alpha = float(d.pop("alpha"))
        if "distortion_control" in d:
            sec_ctrl.distortion_control = bool(d.pop("distortion_control"))
        if "distortion_j_crit" in d:
            sec_ctrl.distortion_j_crit = float(d.pop("distortion_j_crit"))

        return cls(
            step=step,
            time_incrementation=time_inc,
            field_controls=field_ctrl,
            dynamic_controls=dyn_ctrl,
            section_controls=sec_ctrl,
            stabilize=stab_ctrl,
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        )

    def to_json(self, path_or_file: Optional[Union[str, Path]] = None, indent: int = 2) -> Optional[str]:
        """Export configuration to JSON format."""
        data_str = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path_or_file is not None:
            p = Path(path_or_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(data_str)
            return None
        return data_str

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> AbaqusSolverConfig:
        """Import configuration from JSON file or JSON string."""
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(path_or_str))
        return cls.from_dict(data)

    def to_yaml(self, path_or_file: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Export configuration to YAML format (falls back to JSON if PyYAML is not installed)."""
        try:
            import yaml
            data_str = yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        except ImportError:
            data_str = self.to_json(indent=2)

        if path_or_file is not None:
            p = Path(path_or_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(data_str)
            return None
        return data_str

    @classmethod
    def from_yaml(cls, path_or_str: Union[str, Path]) -> AbaqusSolverConfig:
        """Import configuration from YAML file or YAML string."""
        try:
            import yaml
            p = Path(path_or_str)
            if p.exists() and p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                data = yaml.safe_load(str(path_or_str))
            return cls.from_dict(data)
        except ImportError:
            return cls.from_json(path_or_str)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to path, inferring file format (.json, .yaml, .yml) from extension."""
        p = Path(path)
        if p.suffix.lower() in (".yaml", ".yml"):
            self.to_yaml(p)
        else:
            self.to_json(p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> AbaqusSolverConfig:
        """Load configuration from path, inferring file format (.json, .yaml, .yml) from extension."""
        p = Path(path)
        if p.suffix.lower() in (".yaml", ".yml"):
            return cls.from_yaml(p)
        return cls.from_json(p)

    # ------------------------------------------------------------------
    # Dot-Notation Key Path Manipulation
    # ------------------------------------------------------------------
    def set_path(self, key_path: str, value: Any) -> None:
        """Set nested attribute using dot-notation string (e.g. 'time_incrementation.I_C')."""
        parts = key_path.split(".")
        target = self
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise AttributeError(f"Invalid config key path: '{key_path}' (missing sub-object '{part}')")
            target = getattr(target, part)
        attr_name = parts[-1]
        if not hasattr(target, attr_name):
            raise AttributeError(f"Invalid config attribute: '{attr_name}' in key path '{key_path}'")

        # Cast primitive types appropriately if necessary
        curr_val = getattr(target, attr_name)
        if curr_val is not None:
            if isinstance(curr_val, bool):
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                else:
                    value = bool(value)
            elif isinstance(curr_val, int):
                value = int(value)
            elif isinstance(curr_val, float):
                value = float(value)
        setattr(target, attr_name, value)

    def get_path(self, key_path: str) -> Any:
        """Get nested attribute using dot-notation string (e.g. 'time_incrementation.I_C')."""
        parts = key_path.split(".")
        target = self
        for part in parts:
            if not hasattr(target, part):
                raise AttributeError(f"Invalid config key path: '{key_path}' (missing '{part}')")
            target = getattr(target, part)
        return target

    # ------------------------------------------------------------------
    # DynamicSolver & Abaqus INP Interoperability
    # ------------------------------------------------------------------
    def to_solver_kwargs(self) -> Dict[str, Any]:
        """Convert configuration to keyword arguments compatible with DynamicSolver.__init__."""
        return {
            "mode": self.dynamic_controls.integration_mode,
            "tol": self.field_controls.R_n,
            "rtol": self.field_controls.rtol,
            "atol": self.field_controls.atol,
            "max_iter": self.time_incrementation.I_0,
            "max_cutbacks": self.time_incrementation.I_C,
            "ul_mode": self.ul_mode,
            "alpha": self.dynamic_controls.alpha,
            "pardiso_mtype": self.pardiso_mtype,
            "pardiso_phase_reuse": self.pardiso_phase_reuse,
            "max_displacement_corr": self.field_controls.max_disp_corr,
            "distortion_control": self.section_controls.distortion_control,
            "distortion_j_crit": self.section_controls.distortion_j_crit,
        }

    def to_inp_snippet(self) -> str:
        """Generate valid Abaqus .inp keyword text block."""
        lines = []
        nlgeom_str = ", NLGEOM=YES" if self.step.nlgeom else ""
        lines.append(f"*STEP{nlgeom_str}")

        if self.step.procedure.upper() == "DYNAMIC":
            lines.append(f"*DYNAMIC, APPLICATION={self.step.application}, ALPHA={self.dynamic_controls.alpha:.4f}")
        else:
            lines.append("*STATIC")

        lines.append(f"  {self.step.dt_init}, {self.step.t_total}, {self.step.dt_min}, {self.step.dt_max}")
        lines.append("**")
        lines.append("*CONTROLS, PARAMETERS=TIME INCREMENTATION")
        lines.append(
            f"  {self.time_incrementation.I_0}, {self.time_incrementation.I_R}, "
            f"{self.time_incrementation.I_P}, {self.time_incrementation.I_C}, "
            f"{self.time_incrementation.I_L}, {self.time_incrementation.I_G}, "
            f"{self.time_incrementation.dt_growth_factor}, {self.time_incrementation.dt_cutback_factor}"
        )
        lines.append("**")
        lines.append("*CONTROLS, PARAMETERS=FIELD")
        lines.append(f"  {self.field_controls.R_n}, {self.field_controls.C_n}, {self.field_controls.rtol}, {self.field_controls.atol}")

        if self.section_controls.distortion_control:
            lines.append("**")
            lines.append(f"*SECTION CONTROLS, NAME=PSA_DISTORTION_CTRL, DISTORTION CONTROL=YES, LENGTH RATIO={self.section_controls.distortion_j_crit:.2f}")

        if self.stabilize.enabled:
            lines.append("**")
            lines.append(f"*STABILIZE, FACTOR={self.stabilize.factor:.6f}")

        return "\n".join(lines)

    @classmethod
    def from_inp_snippet(cls, text: str) -> AbaqusSolverConfig:
        """Parse an Abaqus .inp keyword text snippet into an AbaqusSolverConfig object."""
        cfg = cls()
        lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("**")]

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            upper_line = line.upper()

            if upper_line.startswith("*STEP"):
                cfg.step.nlgeom = "NLGEOM=YES" in upper_line
            elif upper_line.startswith("*DYNAMIC"):
                cfg.step.procedure = "DYNAMIC"
                app_match = re.search(r"APPLICATION\s*=\s*([^,]+)", upper_line)
                if app_match:
                    cfg.step.application = app_match.group(1).strip()
                alpha_match = re.search(r"ALPHA\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", upper_line)
                if alpha_match:
                    cfg.dynamic_controls.alpha = float(alpha_match.group(1))

                # Next line contains time incrementation numbers
                if idx + 1 < len(lines) and not lines[idx + 1].startswith("*"):
                    idx += 1
                    tokens = [t.strip() for t in lines[idx].split(",")]
                    if len(tokens) >= 4:
                        cfg.step.dt_init = float(tokens[0])
                        cfg.step.t_total = float(tokens[1])
                        cfg.step.dt_min = float(tokens[2])
                        cfg.step.dt_max = float(tokens[3])
            elif upper_line.startswith("*STATIC"):
                cfg.step.procedure = "STATIC"
                if idx + 1 < len(lines) and not lines[idx + 1].startswith("*"):
                    idx += 1
                    tokens = [t.strip() for t in lines[idx].split(",")]
                    if len(tokens) >= 4:
                        cfg.step.dt_init = float(tokens[0])
                        cfg.step.t_total = float(tokens[1])
                        cfg.step.dt_min = float(tokens[2])
                        cfg.step.dt_max = float(tokens[3])
            elif upper_line.startswith("*CONTROLS, PARAMETERS=TIME INCREMENTATION"):
                if idx + 1 < len(lines) and not lines[idx + 1].startswith("*"):
                    idx += 1
                    tokens = [t.strip() for t in lines[idx].split(",")]
                    if len(tokens) >= 1: cfg.time_incrementation.I_0 = int(tokens[0])
                    if len(tokens) >= 2: cfg.time_incrementation.I_R = int(tokens[1])
                    if len(tokens) >= 3: cfg.time_incrementation.I_P = int(tokens[2])
                    if len(tokens) >= 4: cfg.time_incrementation.I_C = int(tokens[3])
                    if len(tokens) >= 5: cfg.time_incrementation.I_L = int(tokens[4])
                    if len(tokens) >= 6: cfg.time_incrementation.I_G = int(tokens[5])
                    if len(tokens) >= 7: cfg.time_incrementation.dt_growth_factor = float(tokens[6])
                    if len(tokens) >= 8: cfg.time_incrementation.dt_cutback_factor = float(tokens[7])
            elif upper_line.startswith("*CONTROLS, PARAMETERS=FIELD"):
                if idx + 1 < len(lines) and not lines[idx + 1].startswith("*"):
                    idx += 1
                    tokens = [t.strip() for t in lines[idx].split(",")]
                    if len(tokens) >= 1: cfg.field_controls.R_n = float(tokens[0])
                    if len(tokens) >= 2: cfg.field_controls.C_n = float(tokens[1])
                    if len(tokens) >= 3: cfg.field_controls.rtol = float(tokens[2])
                    if len(tokens) >= 4: cfg.field_controls.atol = float(tokens[3])
            elif upper_line.startswith("*SECTION CONTROLS"):
                cfg.section_controls.distortion_control = "DISTORTION CONTROL=YES" in upper_line
                lr_match = re.search(r"LENGTH RATIO\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", upper_line)
                if lr_match:
                    cfg.section_controls.distortion_j_crit = float(lr_match.group(1))
            elif upper_line.startswith("*STABILIZE"):
                cfg.stabilize.enabled = True
                f_match = re.search(r"FACTOR\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", upper_line)
                if f_match:
                    cfg.stabilize.factor = float(f_match.group(1))

            idx += 1

        return cfg
