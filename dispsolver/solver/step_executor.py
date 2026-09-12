"""Multi-Step Analysis & Branching Tree Execution Engine for CAE Models."""

from __future__ import annotations
import os
import copy
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence, Any, Union
import numpy as np

from dispsolver.model.model import Model
from dispsolver.model.step import Step, InitialStep, EntityStatus, DisplacementBC, ConcentratedLoad, SurfaceTieInteraction, PredefinedField


@dataclass
class StateCheckpoint:
    """Captured state snapshot of the solver at the completion of an Analysis Step."""
    step_name: str
    time_accumulated: float
    u: np.ndarray
    v: np.ndarray
    a: np.ndarray
    sdvs: Dict[int, Any]
    mesh_coords: np.ndarray
    history_u: List[np.ndarray] = field(default_factory=list)


class MultiStepExecutor:
    """Execution engine for Multi-Step CAE models supporting entity lifecycle (active/inactive/modified)
    and Branching Step Tree Execution (State Checkpoint & Restore).
    """

    def __init__(self, model: Model):
        self.model: Model = model
        self.checkpoints: Dict[str, StateCheckpoint] = {}
        self.branch_results: Dict[str, Dict[str, Any]] = {}

    def _resolve_bc_dofs(self, bc: DisplacementBC, sys: Any) -> Dict[int, float]:
        """Resolve DisplacementBC onto global DOF index -> prescribed value mapping."""
        resolved = {}
        region_name = bc.region
        parts = region_name.split(".")
        nset_name = parts[-1]

        node_indices = None
        if region_name in sys.global_nsets:
            node_indices = sys.global_nsets[region_name]
        elif nset_name in sys.global_nsets:
            node_indices = sys.global_nsets[nset_name]

        if node_indices is not None:
            for idx in node_indices:
                if bc.u1 is not None:
                    resolved[idx * sys.dim + 0] = float(bc.u1)
                if bc.u2 is not None:
                    resolved[idx * sys.dim + 1] = float(bc.u2)
                if bc.u3 is not None and sys.dim == 3:
                    resolved[idx * sys.dim + 2] = float(bc.u3)

        return resolved

    def _apply_predefined_fields(self, solver: Any, sys: Any) -> None:
        """Apply Predefined Fields from InitialStep (Initial Velocities, Stresses, SDVs)."""
        initial_step = self.model.initial_step
        for pf_name, pf in initial_step.predefined_fields.items():
            if pf.field_type.upper() == "VELOCITY":
                # Values: [v1, v2, v3]
                vals = np.array(pf.values, dtype=np.float64)
                if pf.region.upper() == "ALL":
                    for gnid in range(1, solver.num_nodes + 1):
                        idx = gnid - 1
                        dim = min(len(vals), sys.dim)
                        solver.v[idx * sys.dim : idx * sys.dim + dim] = vals[:dim]

            elif pf.field_type.upper() in ["STRESS", "SDV"]:
                # Initialize material state variables
                pass

    def run_step_tree(self, save_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """Execute all Analysis Steps in the Model hierarchy following step tree branching.
        
        Parameters
        ----------
        save_dir : str, optional
            If provided, saves each branch result file into save_dir/branch_{step_name}.pkl
        verbose : bool
            Print step progress and branching logs
        """
        sys = self.model.build_solver_system()
        mesh3d = sys.to_mesh3d()
        
        from dispsolver.solver3d.dynamic3d import DynamicSolver3D
        solver = DynamicSolver3D(mesh=mesh3d, materials=sys.materials_by_pid)

        # Apply Initial Step Predefined Fields
        self._apply_predefined_fields(solver, sys)

        # Initial Checkpoint (t=0)
        self.checkpoints["Initial"] = StateCheckpoint(
            step_name="Initial",
            time_accumulated=0.0,
            u=solver.u.copy(),
            v=solver.v.copy(),
            a=solver.a.copy(),
            sdvs=copy.deepcopy(solver.element_states),
            mesh_coords=mesh3d.coords.copy(),
            history_u=[solver.u.copy()]
        )

        # Sort steps by tree hierarchy (skip "Initial")
        analysis_steps = [st for st in self.model.steps.values() if st.name != "Initial"]

        for step in analysis_steps:
            parent_name = step.parent_step.name if step.parent_step else "Initial"
            if verbose:
                print("=" * 70)
                print(f"  Executing Step: '{step.name}' (Procedure: {step.procedure}, Parent: '{parent_name}')")
                print("=" * 70)

            # Restore State from Parent Checkpoint (Branching Rollback / Propagation)
            parent_ckpt = self.checkpoints.get(parent_name, self.checkpoints["Initial"])
            solver.u = parent_ckpt.u.copy()
            solver.v = parent_ckpt.v.copy()
            solver.a = parent_ckpt.a.copy()
            solver.element_states = copy.deepcopy(parent_ckpt.sdvs)
            accumulated_time = parent_ckpt.time_accumulated
            step_history = list(parent_ckpt.history_u)

            # Configure Active Boundary Conditions for this Step
            solver.fixed_dofs.clear()
            active_bcs = step.get_active_bcs()
            for bc_name, bc in active_bcs.items():
                resolved_dofs = self._resolve_bc_dofs(bc, sys)
                for dof_idx, val in resolved_dofs.items():
                    solver.fixed_dofs[dof_idx] = val

            if verbose:
                print(f"  [Active Entities] BCs: {list(active_bcs.keys())}, Fixed DOFs: {len(solver.fixed_dofs)}")
                print(f"  [Step Time Period] {step.time_period:.2f} s | dt_init={step.dt_init:.4f}")

            # Time integration loop for current step
            t_step = 0.0
            dt = step.dt_init
            inc = 0

            while t_step < step.time_period - 1e-12:
                inc += 1
                dt_actual = min(dt, step.time_period - t_step)
                
                # Evaluate amplitude curves if any
                for bc_name, bc in active_bcs.items():
                    if bc.amplitude and bc.amplitude in self.model.amplitudes:
                        amp = self.model.amplitudes[bc.amplitude]
                        scale = amp.evaluate(t_step / step.time_period)
                        resolved_dofs = self._resolve_bc_dofs(bc, sys)
                        for dof_idx, val in resolved_dofs.items():
                            solver.fixed_dofs[dof_idx] = val * scale

                conv, iters = solver.solve_step(dt=dt_actual, max_iters=25)
                if conv:
                    t_step += dt_actual
                    accumulated_time += dt_actual
                    step_history.append(solver.u.copy())
                    if verbose and inc % 5 == 0:
                        print(f"    Inc {inc:3d}: t_step={t_step:.3f}/{step.time_period:.2f}s, dt={dt_actual:.4f}s (Iters={iters})")
                    # Adjust dt (increase slightly if fast convergence)
                    if iters <= 4:
                        dt = min(step.dt_max, dt * 1.25)
                else:
                    if verbose:
                        print(f"    Inc {inc:3d}: Cutback triggered! dt={dt:.4f}s -> {dt * 0.5:.4f}s")
                    dt *= 0.5
                    if dt < step.dt_min:
                        print(f"  [ERROR] Step '{step.name}' failed to converge below dt_min={step.dt_min}")
                        break

            # Capture Checkpoint for this Step
            ckpt = StateCheckpoint(
                step_name=step.name,
                time_accumulated=accumulated_time,
                u=solver.u.copy(),
                v=solver.v.copy(),
                a=solver.a.copy(),
                sdvs=copy.deepcopy(solver.element_states),
                mesh_coords=mesh3d.coords.copy(),
                history_u=step_history
            )
            self.checkpoints[step.name] = ckpt

            # Record Branch Result
            result_data = {
                "step_name": step.name,
                "parent_step": parent_name,
                "accumulated_time": accumulated_time,
                "mesh": mesh3d,
                "displacement": solver.u.copy(),
                "history": step_history,
                "active_bcs": list(active_bcs.keys()),
                "active_loads": list(step.get_active_loads().keys())
            }
            self.branch_results[step.name] = result_data

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"result_branch_{step.name}.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(result_data, f)
                if verbose:
                    print(f"  [Saved Branch Result] {out_path}")

        return self.branch_results
