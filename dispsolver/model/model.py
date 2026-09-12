"""Top-level Model container for CAE model hierarchy."""

from __future__ import annotations
from typing import Dict, Optional, Any, Union

from dispsolver.model.part import Part
from dispsolver.model.material import Material
from dispsolver.model.section import Section, SolidSection, ShellSection, SectionControls
from dispsolver.model.assembly import Assembly, FlattenedSolverSystem
from dispsolver.model.step import Step, InitialStep, DisplacementBC, ConcentratedLoad, SurfaceTieInteraction, PredefinedField, Amplitude


class Model:
    """Top-level CAE Model container corresponding to Abaqus mdb.models['Model-1']."""

    def __init__(self, name: str = "Model-1", dim: int = 3):
        self.name = str(name)
        self.dim = int(dim)
        self.parts: Dict[str, Part] = {}
        self.materials: Dict[str, Material] = {}
        self.sections: Dict[str, Section] = {}
        self.section_controls: Dict[str, SectionControls] = {}
        self.amplitudes: Dict[str, Amplitude] = {}
        
        self.initial_step: InitialStep = InitialStep(name="Initial")
        self.steps: Dict[str, Step] = {"Initial": self.initial_step}
        self.root_assembly: Assembly = Assembly(name="rootAssembly", dim=self.dim)

    # ─── FACTORY & REGISTRATION METHODS ────────────────────────────────

    def Part(self, name: str, dim: Optional[int] = None) -> Part:
        """Create and register a new Part in this Model."""
        d = dim if dim is not None else self.dim
        p = Part(name=str(name), dim=d)
        self.parts[str(name)] = p
        return p

    def Material(self, name: str, mat_type: str = "ELASTIC") -> Material:
        """Create and register a new Material in this Model."""
        m = Material(name=str(name), mat_type=mat_type)
        self.materials[str(name)] = m
        return m

    def SectionControls(
        self,
        name: str,
        distortion_control: bool = True,
        length_ratio: float = 0.1,
        hourglass_control: str = "ENHANCED",
        viscous_damping: float = 0.0,
        anti_inversion_barrier: bool = True,
        min_det_f: float = 0.02,
        **kwargs
    ) -> SectionControls:
        """Create and register a SectionControls definition in this Model (*SECTION CONTROLS in Abaqus)."""
        if "distortionControl" in kwargs:
            distortion_control = (kwargs["distortionControl"] in [True, "ON", "YES", 1])
        if "lengthRatio" in kwargs:
            length_ratio = float(kwargs["lengthRatio"])
        if "hourglassControl" in kwargs:
            hourglass_control = str(kwargs["hourglassControl"])
        if "viscousDamping" in kwargs:
            viscous_damping = float(kwargs["viscousDamping"])
        if "antiInversionBarrier" in kwargs:
            anti_inversion_barrier = (kwargs["antiInversionBarrier"] in [True, "ON", "YES", 1])
        if "minDetF" in kwargs:
            min_det_f = float(kwargs["minDetF"])

        ctrl = SectionControls(
            name=str(name),
            distortion_control=distortion_control,
            length_ratio=length_ratio,
            hourglass_control=hourglass_control,
            viscous_damping=viscous_damping,
            anti_inversion_barrier=anti_inversion_barrier,
            min_det_f=min_det_f
        )
        self.section_controls[str(name)] = ctrl
        return ctrl

    def SolidSection(
        self,
        name: str,
        material: str,
        thickness: float = 1.0,
        controls: Optional[Union[str, SectionControls]] = None
    ) -> SolidSection:
        """Create and register a SolidSection in this Model."""
        sec = SolidSection(
            name=str(name),
            material_name=str(material),
            thickness=float(thickness),
            controls=controls
        )
        self.sections[str(name)] = sec
        return sec

    def ShellSection(self, name: str, material: str, thickness: float = 1.0, num_int_pts: int = 5) -> ShellSection:
        """Create and register a ShellSection in this Model."""
        sec = ShellSection(
            name=str(name),
            material_name=str(material),
            thickness=float(thickness),
            num_int_pts=int(num_int_pts)
        )
        self.sections[str(name)] = sec
        return sec

    def Step(
        self,
        name: str,
        previous: str = "Initial",
        procedure: str = "STATIC",
        time_period: float = 1.0,
        dt_init: float = 0.02,
        dt_min: float = 1e-5,
        dt_max: float = 0.05
    ) -> Step:
        """Create and register an analysis Step in this Model, inheriting from previous step."""
        parent = self.steps.get(str(previous), self.initial_step)
        st = Step(
            name=str(name),
            procedure=procedure,
            time_period=time_period,
            parent_step=parent,
            dt_init=dt_init,
            dt_min=dt_min,
            dt_max=dt_max
        )
        self.steps[str(name)] = st
        return st

    def create_step_branch(
        self,
        branch_name: str,
        parent_step_name: str,
        procedure: str = "STATIC",
        time_period: float = 1.0,
        dt_init: float = 0.02,
        dt_min: float = 1e-5,
        dt_max: float = 0.05
    ) -> Step:
        """Create an explicit branch Step off parent_step_name for scenario testing."""
        return self.Step(
            name=branch_name,
            previous=parent_step_name,
            procedure=procedure,
            time_period=time_period,
            dt_init=dt_init,
            dt_min=dt_min,
            dt_max=dt_max
        )

    def PredefinedField(self, name: str, field_type: str, region: str, values: Any) -> PredefinedField:
        """Create and assign a Predefined Field (Initial Stress, Velocity, SDVs) to InitialStep."""
        pf = PredefinedField(name=str(name), field_type=field_type, region=str(region), values=values)
        self.initial_step.add_predefined_field(pf)
        return pf

    def Amplitude(self, name: str, data: Optional[list] = None, smooth: bool = True) -> Amplitude:
        """Create and register an Amplitude in this Model."""
        amp = Amplitude(name=str(name), data=data if data is not None else [], smooth=smooth)
        self.amplitudes[str(name)] = amp
        return amp

    def add_part(self, part: Part) -> Part:
        self.parts[part.name] = part
        return part

    def add_material(self, material: Material) -> Material:
        self.materials[material.name] = material
        return material

    def add_section(self, section: Section) -> Section:
        self.sections[section.name] = section
        return section

    def add_step(self, step: Step) -> Step:
        self.steps[step.name] = step
        return step

    # ─── SOLVER INTEGRATION ENGINE ──────────────────────────────────────

    def build_solver_system(self) -> FlattenedSolverSystem:
        """Flatten root assembly into global solver structures."""
        return self.root_assembly.build_solver_system(self)

    def create_solver3d(self, step_name: Optional[str] = None):
        """Create and configure a production DynamicSolver3D instance from this Model."""
        from dispsolver.solver3d.dynamic3d import DynamicSolver3D
        sys = self.build_solver_system()
        mesh3d = sys.to_mesh3d()
        
        solver = DynamicSolver3D(mesh=mesh3d, materials=sys.materials_by_pid)
        
        # Inject precomputed sparse topology for zero-overhead Numba assembly
        if len(sys.rows_topo) > 0:
            solver.rows_topo = sys.rows_topo
            solver.cols_topo = sys.cols_topo
            
        # If a step is provided, apply its boundary conditions
        if step_name is not None and step_name in self.steps:
            st = self.steps[step_name]
            for entry in st.boundary_conditions.values():
                if entry.is_active:
                    bc = entry.entity
                    if bc.region in sys.global_nsets:
                        global_node_indices = sys.global_nsets[bc.region]
                        inv_nid = {idx: g for g, idx in sys.nid_to_idx.items()}
                        for node_idx in global_node_indices:
                            gid = inv_nid[node_idx]
                            if bc.u1 is not None:
                                solver.fix_dof(gid, 0, float(bc.u1))
                            if bc.u2 is not None:
                                solver.fix_dof(gid, 1, float(bc.u2))
                            if bc.u3 is not None:
                                solver.fix_dof(gid, 2, float(bc.u3))
                            
        return solver, sys
