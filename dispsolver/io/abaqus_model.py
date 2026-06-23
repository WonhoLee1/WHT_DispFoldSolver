"""Pure dataclass definitions for Abaqus .inp parser intermediate model.

No business logic — data containers only. No imports from dispsolver.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class AbaqusNode:
    """Abaqus *NODE definition."""
    id: int
    x: float
    y: float


@dataclass
class AbaqusElement:
    """Abaqus *ELEMENT definition."""
    eid: int
    etype: str  # e.g., "CPE4", "CPE3", "CPE4R"
    node_ids: List[int]


@dataclass
class AbaqusMaterial:
    """Generic Abaqus material definition.

    Supports two modes:
    1. Legacy flat: mat_type="ELASTIC"/"PLASTIC"/etc. with params dict.
    2. COMPOSITE (via *MATERIAL): properties accumulated into dedicated fields.
    """
    name: str
    mat_type: str = "COMPOSITE"  # "COMPOSITE" or legacy ("ELASTIC", "PLASTIC", ...)
    params: dict = field(default_factory=dict)  # legacy flat parameters
    # Composite fields (populated when *MATERIAL scoping is used)
    elastic: Optional[Tuple[float, float]] = None       # (E, nu)
    plastic_table: Optional[List[Tuple[float, float]]] = None  # [(sigma_y, eps_p), ...]
    hyperelastic: Optional[dict] = None                  # {type, C10, D1, ...}
    viscoelastic_prony: Optional[List[Tuple[float, float, float]]] = None  # [(g_i, k_i, tau_i), ...]
    density: Optional[float] = None
    trs: Optional[dict] = None                           # WLF or user TTS parameters


@dataclass
class AbaqusSection:
    """Abaqus *SOLID SECTION definition."""
    name: str
    material: str
    elset: str
    thickness: float = 1.0


@dataclass
class AbaqusBoundary:
    """Abaqus *BOUNDARY definition (1-based DOF)."""
    nset: str
    dof1: int
    dof2: int
    value: float = 0.0
    amplitude: Optional[str] = None


@dataclass
class AbaqusLoad:
    """Abaqus *CLOAD definition."""
    name: str
    nset_or_elset: str
    magnitude: float
    amplitude: Optional[str] = None


@dataclass
class AbaqusStep:
    """Abaqus *STEP definition."""
    name: str
    procedure: str  # "STATIC" or "DYNAMIC"
    bounds: Optional[dict] = None  # time stepping parameters
    loads: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class AbaqusMpc:
    """Abaqus *MPC definition."""
    mpc_type: str  # e.g., "BEAM", "TIE", "RBE2"
    nodes: List[int]
    dofs: Optional[List[int]] = None


@dataclass
class AbaqusTie:
    """Abaqus *TIE definition."""
    slave: str
    master: str
    position_tolerance: float = 0.0
    slave_surfaces: List[str] = field(default_factory=list)
    master_surfaces: List[str] = field(default_factory=list)


@dataclass
class AbaqusAmplitude:
    """Abaqus *AMPLITUDE definition."""
    name: str
    time_type: str = "STEP"  # "STEP", "FREQUENCY", "TRANSIENT"
    data: List[Tuple[float, float]] = field(default_factory=list)
    smooth: bool = False


@dataclass
class AbaqusDload:
    """Abaqus *DLOAD / *DSLOAD definition."""
    elset: str
    face: str  # Abaqus face identifier (e.g. "P3", "S1", "P1")
    magnitude: float


@dataclass
class AbaqusSurface:
    """Abaqus *SURFACE definition."""
    name: str
    surface_type: str  # "ELEMENT" or "NODE"
    definitions: List[Tuple[str, str]]  # [(elset_or_instance, "SPOS"|"SNEG"), ...]


@dataclass
class AbaqusContactPair:
    """Abaqus *CONTACT PAIR definition."""
    name: str
    slave_surface: str
    master_surface: str
    interaction: str = ""


@dataclass
class AbaqusContactProperty:
    """Abaqus *SURFACE INTERACTION definition."""
    name: str
    penalty_stiffness: float = 1e6
    friction_coeff: float = 0.0


@dataclass
class AbaqusRigidBody:
    """Abaqus *RIGID BODY definition.

    Binds all nodes in a set to move rigidly with a reference node.
    """
    ref_node: int           # reference node that controls the rigid body motion
    node_ids: List[int]     # nodes in this rigid body
    name: str = ""          # optional name


@dataclass
class AbaqusPart:
    """Abaqus *PART definition (for Part/Assembly flattening)."""
    name: str
    nodes: List[AbaqusNode] = field(default_factory=list)
    elements: List[AbaqusElement] = field(default_factory=list)
    nsets: dict = field(default_factory=dict)
    elsets: dict = field(default_factory=dict)
    materials: dict = field(default_factory=dict)
    sections: dict = field(default_factory=dict)


@dataclass
class AbaqusInstance:
    """Abaqus *INSTANCE definition."""
    part_name: str
    instance_name: str
    translation: Optional[Tuple[float, float]] = None
    rotation: Optional[Tuple[float, float]] = None  # (angle_deg, origin_x, origin_y) — rotation about origin


@dataclass
class AbaqusModel:
    """Top-level container for the entire Abaqus model definition."""
    nodes: List[AbaqusNode] = field(default_factory=list)
    elements: List[AbaqusElement] = field(default_factory=list)
    nsets: dict = field(default_factory=dict)      # name -> node ids
    elsets: dict = field(default_factory=dict)      # name -> element ids
    materials: dict = field(default_factory=dict)   # name -> AbaqusMaterial
    sections: dict = field(default_factory=dict)    # name -> AbaqusSection
    boundaries: List[AbaqusBoundary] = field(default_factory=list)
    loads: List[AbaqusLoad] = field(default_factory=list)
    steps: List[AbaqusStep] = field(default_factory=list)
    mpcs: List[AbaqusMpc] = field(default_factory=list)
    ties: List[AbaqusTie] = field(default_factory=list)
    amplitudes: dict = field(default_factory=dict)  # name -> AbaqusAmplitude
    dloads: List[AbaqusDload] = field(default_factory=list)
    surfaces: dict = field(default_factory=dict)    # name -> AbaqusSurface
    contact_pairs: List[AbaqusContactPair] = field(default_factory=list)
    contact_properties: dict = field(default_factory=dict)  # name -> AbaqusContactProperty
    rigid_bodies: List[AbaqusRigidBody] = field(default_factory=list)
    parts: dict = field(default_factory=dict)       # name -> AbaqusPart
    instances: List[AbaqusInstance] = field(default_factory=list)
    assemblies: dict = field(default_factory=dict)  # name -> assembly dict
    density: Optional[float] = None
