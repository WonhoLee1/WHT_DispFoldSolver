"""Step, Boundary Condition, Load, Interaction, and PredefinedField definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Sequence, Any, Union
import numpy as np


class EntityStatus(Enum):
    """Lifecycle status of a BC, Load, or Interaction within an Analysis Step."""
    CREATED = "CREATED"          # Newly created in this step
    PROPAGATED = "PROPAGATED"    # Inherited from parent step without change
    MODIFIED = "MODIFIED"        # Modified in this step (amplitude, magnitude, etc.)
    DEACTIVATED = "DEACTIVATED"  # Turned off (inactive) in this step
    REACTIVATED = "REACTIVATED"  # Re-enabled in this step after being deactivated


from dispsolver.model.amplitude import Amplitude, TabularAmplitude, SmoothStepAmplitude, UserFunctionAmplitude
from dispsolver.model.bc import BoundaryCondition, DisplacementBC, VelocityBC, UserFunctionBC
from dispsolver.model.sensor import Sensor, SensorManager, IterationHook, ControlAction




@dataclass
class PredefinedField:
    """Predefined field condition assigned in InitialStep (Initial Stress, Velocity, SDVs, etc.)."""
    name: str
    field_type: str  # "STRESS", "STRAIN", "VELOCITY", "TEMPERATURE", "SDV"
    region: str       # Set name or "ALL"
    values: Any       # Scalar, vector, or tensor values


@dataclass
class StepStateEntry:
    """Tracks an entity instance alongside its step lifecycle status and local modifications."""
    entity: Any
    status: EntityStatus = EntityStatus.CREATED
    modified_params: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Return True if entity is active in the current step."""
        return self.status in [EntityStatus.CREATED, EntityStatus.PROPAGATED, EntityStatus.MODIFIED, EntityStatus.REACTIVATED]


class Step:
    """Analysis Step definition (Static, Dynamic, Visco, etc.) with parent-child step tree hierarchy."""

    def __init__(
        self,
        name: str,
        procedure: str = "STATIC",
        time_period: float = 1.0,
        parent_step: Optional[Step] = None,
        nlgeom: bool = True,
        dt_init: float = 0.02,
        dt_min: float = 1e-5,
        dt_max: float = 0.05
    ):
        self.name = str(name)
        self.procedure = str(procedure).upper()
        self.time_period = float(time_period)
        self.nlgeom = bool(nlgeom)
        self.dt_init = float(dt_init)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        self.parent_step: Optional[Step] = parent_step
        self.child_steps: List[Step] = []

        if parent_step is not None:
            parent_step.child_steps.append(self)

        self.boundary_conditions: Dict[str, StepStateEntry] = {}
        self.loads: Dict[str, StepStateEntry] = {}
        self.interactions: Dict[str, StepStateEntry] = {}
        self.predefined_fields: Dict[str, PredefinedField] = {}
        self.sensors: Dict[str, Sensor] = {}
        self.iteration_hooks: Dict[str, IterationHook] = {}

        # If created from a parent step, propagate active entities from parent
        if parent_step is not None:
            self.propagate_from_parent(parent_step)

    def propagate_from_parent(self, parent: Step) -> None:
        """Propagate active BCs, Loads, and Interactions from parent step as PROPAGATED, and inactive as DEACTIVATED."""
        import copy
        for name, entry in parent.boundary_conditions.items():
            status = EntityStatus.PROPAGATED if entry.is_active() else EntityStatus.DEACTIVATED
            self.boundary_conditions[name] = StepStateEntry(
                entity=copy.copy(entry.entity),
                status=status,
                modified_params=dict(entry.modified_params)
            )

        for name, entry in parent.loads.items():
            status = EntityStatus.PROPAGATED if entry.is_active() else EntityStatus.DEACTIVATED
            self.loads[name] = StepStateEntry(
                entity=copy.copy(entry.entity),
                status=status,
                modified_params=dict(entry.modified_params)
            )

        for name, entry in parent.interactions.items():
            status = EntityStatus.PROPAGATED if entry.is_active() else EntityStatus.DEACTIVATED
            self.interactions[name] = StepStateEntry(
                entity=copy.copy(entry.entity),
                status=status,
                modified_params=dict(entry.modified_params)
            )

    # ─── BOUNDARY CONDITION LIFECYCLE ──────────────────────────────────

    def add_boundary_condition(self, bc: DisplacementBC) -> DisplacementBC:
        """Create and activate a new Boundary Condition in this Step."""
        self.boundary_conditions[bc.name] = StepStateEntry(entity=bc, status=EntityStatus.CREATED)
        return bc

    def deactivate_bc(self, bc_name: str) -> None:
        """Deactivate (turn OFF) a Boundary Condition in this Step."""
        if bc_name in self.boundary_conditions:
            self.boundary_conditions[bc_name].status = EntityStatus.DEACTIVATED

    def modify_bc(self, bc_name: str, **kwargs) -> None:
        """Modify parameters of an existing Boundary Condition in this Step."""
        if bc_name in self.boundary_conditions:
            entry = self.boundary_conditions[bc_name]
            entry.status = EntityStatus.MODIFIED
            entry.modified_params.update(kwargs)
            for k, v in kwargs.items():
                if hasattr(entry.entity, k):
                    setattr(entry.entity, k, v)

    def reactivate_bc(self, bc_name: str) -> None:
        """Re-enable a previously deactivated Boundary Condition in this Step."""
        if bc_name in self.boundary_conditions:
            self.boundary_conditions[bc_name].status = EntityStatus.REACTIVATED

    # ─── LOAD LIFECYCLE ────────────────────────────────────────────────

    def add_load(self, load: Any) -> Any:
        """Create and activate a new Load in this Step."""
        self.loads[load.name] = StepStateEntry(entity=load, status=EntityStatus.CREATED)
        return load

    def deactivate_load(self, load_name: str) -> None:
        """Deactivate a Load in this Step."""
        if load_name in self.loads:
            self.loads[load_name].status = EntityStatus.DEACTIVATED

    def modify_load(self, load_name: str, **kwargs) -> None:
        """Modify parameters of a Load in this Step."""
        if load_name in self.loads:
            entry = self.loads[load_name]
            entry.status = EntityStatus.MODIFIED
            entry.modified_params.update(kwargs)
            for k, v in kwargs.items():
                if hasattr(entry.entity, k):
                    setattr(entry.entity, k, v)

    def reactivate_load(self, load_name: str) -> None:
        """Re-enable a previously deactivated Load in this Step."""
        if load_name in self.loads:
            self.loads[load_name].status = EntityStatus.REACTIVATED

    # ─── INTERACTION LIFECYCLE ─────────────────────────────────────────

    def add_interaction(self, interaction: Any) -> Any:
        """Create and activate a new Interaction (e.g. SurfaceTie) in this Step."""
        self.interactions[interaction.name] = StepStateEntry(entity=interaction, status=EntityStatus.CREATED)
        return interaction

    def deactivate_interaction(self, interaction_name: str) -> None:
        """Deactivate an Interaction in this Step."""
        if interaction_name in self.interactions:
            self.interactions[interaction_name].status = EntityStatus.DEACTIVATED

    def modify_interaction(self, interaction_name: str, **kwargs) -> None:
        """Modify parameters of an Interaction in this Step."""
        if interaction_name in self.interactions:
            entry = self.interactions[interaction_name]
            entry.status = EntityStatus.MODIFIED
            entry.modified_params.update(kwargs)
            for k, v in kwargs.items():
                if hasattr(entry.entity, k):
                    setattr(entry.entity, k, v)

    def reactivate_interaction(self, interaction_name: str) -> None:
        """Re-enable a previously deactivated Interaction in this Step."""
        if interaction_name in self.interactions:
            self.interactions[interaction_name].status = EntityStatus.REACTIVATED

    # ─── PREDEFINED FIELDS (INITIAL STEP ONLY) ─────────────────────────

    def add_predefined_field(self, field: PredefinedField) -> PredefinedField:
        """Add a Predefined Field (Initial Stress, Velocity, SDVs, etc.)."""
        self.predefined_fields[field.name] = field
        return field

    def get_active_bcs(self) -> Dict[str, DisplacementBC]:
        """Return dict of active DisplacementBCs in this step."""
        return {name: entry.entity for name, entry in self.boundary_conditions.items() if entry.is_active()}

    def get_active_loads(self) -> Dict[str, Any]:
        """Return dict of active Loads in this step."""
        return {name: entry.entity for name, entry in self.loads.items() if entry.is_active()}

    def get_active_interactions(self) -> Dict[str, Any]:
        """Return dict of active Interactions in this step."""
        return {name: entry.entity for name, entry in self.interactions.items() if entry.is_active()}


class InitialStep(Step):
    """Initial Step (t=0) for defining base boundary conditions and PredefinedFields."""

    def __init__(self, name: str = "Initial"):
        super().__init__(name=name, procedure="INITIAL", time_period=0.0, parent_step=None)

