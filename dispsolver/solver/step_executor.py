"""Commercial Multi-Step CAE Execution & Branching Step Tree Engine.

Features:
- StateCheckpoint: Deep state snapshot of solver state (u, v, a, SDVs, t).
- Branching Tree Execution: Execute serial steps or branched alternatives (e.g. Step 1 -> Branch A vs Branch B).
- Entity Lifecycle Management: CREATED, PROPAGATED, MODIFIED, DEACTIVATED, REACTIVATED.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import copy
import os
import pickle
import numpy as np

from dispsolver.model.step import Step, EntityStatus, StepStateEntry
from dispsolver.model.model import Model


@dataclass
class StateCheckpoint:
    """Complete physical and numerical state snapshot of a solver at a step boundary."""
    step_name: str
    t_accum: float
    u: np.ndarray
    v: np.ndarray
    a: np.ndarray
    elem_sdvs: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def restore_to_solver(self, solver: Any) -> None:
        """Restore checkpointed state back into the solver instance."""
        solver.u = self.u.copy()
        solver.v = self.v.copy()
        solver.a = self.a.copy()
        if self.elem_sdvs is not None and hasattr(solver, "elem_sdvs"):
            solver.elem_sdvs = self.elem_sdvs.copy()

    @property
    def time_accumulated(self) -> float:
        """Alias for t_accum."""
        return self.t_accum


class MultiStepExecutor:
    """Executes multi-step CAE analysis trees with state checkpoints and branching."""

    def __init__(self, model: Model, verbose: bool = True, dim: Optional[int] = None):
        self.model = model
        self.verbose = verbose
        self.dim = dim
        self.checkpoints: Dict[str, StateCheckpoint] = {}
        self.results_by_step: Dict[str, Dict[str, Any]] = {}
        self.solver: Optional[Any] = None
        self.sys: Optional[Any] = None

    def initialize_solver(self, solver_factory=None, notices=None, dim=None) -> Tuple[Any, Any]:
        """Build global solver system and instantiate the solver.

        solver_factory: zero-argument factory returning (solver, sys).
        Defaults to model.create_solver (auto-detects 2D vs 3D).
        notices: optional list receiving solve-side warning strings.
        """
        use_dim = dim if dim is not None else self.dim
        if solver_factory is None:
            self.solver, self.sys = self.model.create_solver(notices=notices, dim=use_dim)
        else:
            self.solver, self.sys = solver_factory()
        return self.solver, self.sys

    def capture_checkpoint(self, step_name: str, t_accum: float) -> StateCheckpoint:
        """Capture current solver state into a named checkpoint."""
        if self.solver is None:
            raise RuntimeError("Solver is not initialized.")
        
        sdvs = getattr(self.solver, "elem_sdvs", None)
        ckpt = StateCheckpoint(
            step_name=str(step_name),
            t_accum=float(t_accum),
            u=self.solver.u.copy(),
            v=self.solver.v.copy(),
            a=self.solver.a.copy(),
            elem_sdvs=sdvs.copy() if sdvs is not None else None
        )
        self.checkpoints[step_name] = ckpt
        return ckpt

    def restore_checkpoint(self, step_name: str) -> StateCheckpoint:
        """Restore solver state from an existing checkpoint."""
        if step_name not in self.checkpoints:
            raise KeyError(f"Checkpoint for step '{step_name}' not found.")
        ckpt = self.checkpoints[step_name]
        ckpt.restore_to_solver(self.solver)
        return ckpt

    def execute_step(
        self,
        step: Step,
        from_parent_checkpoint: bool = True,
        output_mgr: Optional[Any] = None,
    ) -> bool:
        """Execute a single analysis step, restoring parent checkpoint if requested."""
        if self.solver is None:
            self.initialize_solver()

        # If this step has a parent and checkpoint exists, restore parent state
        if from_parent_checkpoint and step.parent_step is not None:
            p_name = step.parent_step.name
            if p_name in self.checkpoints:
                if self.verbose:
                    print(f"[*] Restoring checkpoint from parent step '{p_name}'...", flush=True)
                self.restore_checkpoint(p_name)

        # Ensure step links back to model for amplitude resolution
        step.parent_model = self.model

        if self.verbose:
            print(f"\n{'=' * 100}", flush=True)
            print(f"  EXECUTING STEP: '{step.name}' (Procedure: {step.procedure})", flush=True)
            print(f"{'=' * 100}", flush=True)

        success = self.solver.solve(step=step, sys=self.sys, output_mgr=output_mgr, verbose=self.verbose)
        if success:
            t_parent = 0.0
            if step.parent_step is not None and step.parent_step.name in self.checkpoints:
                t_parent = self.checkpoints[step.parent_step.name].t_accum
            t_end = t_parent + float(getattr(step, "time_period", 1.0))
            ckpt = self.capture_checkpoint(step.name, t_end)
            self.results_by_step[step.name] = {
                "u": self.solver.u.copy(),
                "checkpoint": ckpt,
                "success": True
            }
        else:
            self.results_by_step[step.name] = {
                "success": False
            }

        return success

    def execute_branch(self, step_names: List[str]) -> bool:
        """Execute a sequence of steps forming a single branch (e.g. ['Step-1', 'Step-2A'])."""
        for sname in step_names:
            if sname not in self.model.steps:
                raise KeyError(f"Step '{sname}' not found in model.")
            step = self.model.steps[sname]
            ok = self.execute_step(step, from_parent_checkpoint=True)
            if not ok:
                if self.verbose:
                    print(f"[-] Branch execution failed at step '{sname}'.", flush=True)
                return False
        return True

    def run_step_tree(self, save_dir: Optional[str] = None, verbose: Optional[bool] = None) -> Dict[str, Any]:
        """Execute all steps defined in the model step tree."""
        if verbose is not None:
            self.verbose = verbose
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        for step_name, step in self.model.steps.items():
            ok = self.execute_step(step, from_parent_checkpoint=True)
            if ok and save_dir is not None:
                p_path = os.path.join(save_dir, f"result_branch_{step_name}.pkl")
                with open(p_path, "wb") as f:
                    pickle.dump(self.results_by_step[step_name], f)
        return self.results_by_step

