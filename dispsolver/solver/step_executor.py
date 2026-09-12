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


class MultiStepExecutor:
    """Executes multi-step CAE analysis trees with state checkpoints and branching."""

    def __init__(self, model: Model, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.checkpoints: Dict[str, StateCheckpoint] = {}
        self.results_by_step: Dict[str, Dict[str, Any]] = {}
        self.solver: Optional[Any] = None
        self.sys: Optional[Any] = None

    def initialize_solver(self) -> Tuple[Any, Any]:
        """Build global solver system and instantiate DynamicSolver3D."""
        self.solver, self.sys = self.model.create_solver3d()
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

    def execute_step(self, step: Step, from_parent_checkpoint: bool = True) -> bool:
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
            print(f"\n=======================================================", flush=True)
            print(f"  EXECUTING STEP: '{step.name}' (Procedure: {step.procedure})", flush=True)
            print(f"=======================================================", flush=True)

        success = self.solver.solve(step=step, sys=self.sys, verbose=self.verbose)
        
        if success:
            # Capture step completion checkpoint
            t_end = getattr(step, "time_period", 1.0)
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
