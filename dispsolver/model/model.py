"""Top-level Model container for CAE model hierarchy."""

from __future__ import annotations
from typing import Dict, Optional, Any, Union

from dispsolver.model.part import Part
from dispsolver.model.material import Material
from dispsolver.model.section import Section, SolidSection, ShellSection
from dispsolver.model.assembly import Assembly, FlattenedSolverSystem
from dispsolver.model.step import Step, DisplacementBC, Amplitude


class Model:
    """Top-level CAE Model container corresponding to Abaqus mdb.models['Model-1']."""

    def __init__(self, name: str = "Model-1", dim: int = 3):
        self.name = str(name)
        self.dim = int(dim)
        self.parts: Dict[str, Part] = {}
        self.materials: Dict[str, Material] = {}
        self.sections: Dict[str, Section] = {}
        self.amplitudes: Dict[str, Amplitude] = {}
        self.steps: Dict[str, Step] = {}
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

    def SolidSection(self, name: str, material: str, thickness: float = 1.0) -> SolidSection:
        """Create and register a SolidSection in this Model."""
        sec = SolidSection(name=str(name), material_name=str(material), thickness=float(thickness))
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

    def Step(self, name: str, procedure: str = "STATIC", time_period: float = 1.0) -> Step:
        """Create and register an analysis Step in this Model."""
        st = Step(name=str(name), procedure=procedure, time_period=time_period)
        self.steps[str(name)] = st
        return st

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
            for bc in st.boundary_conditions.values():
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
