"""dispsolver.model: Commercial-Grade CAE Object Model Architecture.

Abaqus-compatible Model -> Part -> Section/Material/Set -> Assembly -> Instance -> Step hierarchy.
"""

from dispsolver.model.base import Transform3D
from dispsolver.model.set import NodeSet, ElementSet, Surface, ElementFace, SetScope, GeneralSet
from dispsolver.model.material import Material
from dispsolver.model.section import Section, SolidSection, ShellSection, SectionControls
from dispsolver.model.part import Part, SectionAssignment
from dispsolver.model.instance import Instance
from dispsolver.model.assembly import Assembly, FlattenedSolverSystem
from dispsolver.model.step import (
    Step,
    InitialStep,
    PredefinedField,
    EntityStatus,
    StepStateEntry,
    DisplacementBC,
)
from dispsolver.model.constraint import (
    Constraint,
    RigidBody,
    Tie,
    Coupling,
    KinematicCoupling,
    DistributingCoupling,
    MPC,
)
from dispsolver.model.load import (
    Load,
    ConcentratedForce,
    Pressure,
    Gravity,
    BodyForce,
)
from dispsolver.model.model import Model

__all__ = [
    "Transform3D",
    "NodeSet",
    "ElementSet",
    "Surface",
    "ElementFace",
    "SetScope",
    "GeneralSet",
    "Material",
    "Section",
    "SolidSection",
    "ShellSection",
    "SectionControls",
    "Part",
    "SectionAssignment",
    "Instance",
    "Assembly",
    "FlattenedSolverSystem",
    "Step",
    "InitialStep",
    "PredefinedField",
    "EntityStatus",
    "StepStateEntry",
    "DisplacementBC",
    "Constraint",
    "RigidBody",
    "Tie",
    "Coupling",
    "KinematicCoupling",
    "DistributingCoupling",
    "MPC",
    "Load",
    "ConcentratedForce",
    "Pressure",
    "Gravity",
    "BodyForce",
    "Model",
]
