"""Top-level Model container for CAE model hierarchy."""

from __future__ import annotations
from typing import Dict, Optional, Any, Union

from dispsolver.model.part import Part
from dispsolver.model.material import Material
from dispsolver.model.section import Section, SolidSection, ShellSection, SectionControls
from dispsolver.model.assembly import Assembly, FlattenedSolverSystem
from dispsolver.model.step import Step, InitialStep, PredefinedField
from dispsolver.model.amplitude import Amplitude, TabularAmplitude, SmoothStepAmplitude, UserFunctionAmplitude
from dispsolver.model.bc import BoundaryCondition, DisplacementBC, VelocityBC, UserFunctionBC
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
        self.constraints: Dict[str, Constraint] = {}
        
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
        previous: Optional[Union[str, Step]] = "Initial",
        procedure: str = "STATIC",
        time_period: float = 1.0,
        dt_init: float = 0.02,
        dt_min: float = 1e-5,
        dt_max: float = 0.05,
        parent_step: Optional[Step] = None,
        **kwargs
    ) -> Step:
        """Create and register an analysis Step in this Model, inheriting from previous step."""
        parent = parent_step
        if parent is None:
            if isinstance(previous, Step):
                parent = previous
            elif isinstance(previous, str):
                parent = self.steps.get(previous, self.initial_step)
            else:
                parent = self.initial_step

        st = Step(
            name=str(name),
            procedure=procedure,
            time_period=time_period,
            parent_step=parent,
            dt_init=dt_init,
            dt_min=dt_min,
            dt_max=dt_max,
            **kwargs
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
        """Create and register a SmoothStep/Tabular Amplitude in this Model."""
        if smooth:
            amp = SmoothStepAmplitude(name=str(name), data=data if data is not None else [])
        else:
            amp = TabularAmplitude(name=str(name), data=data if data is not None else [])
        self.amplitudes[str(name)] = amp
        return amp

    def TabularAmplitude(self, name: str, data: List[Tuple[float, float]]) -> TabularAmplitude:
        """Create and register a TabularAmplitude (*AMPLITUDE, TYPE=TABULAR)."""
        amp = TabularAmplitude(name=str(name), data=data)
        self.amplitudes[str(name)] = amp
        return amp

    def SmoothStepAmplitude(self, name: str, data: List[Tuple[float, float]]) -> SmoothStepAmplitude:
        """Create and register a SmoothStepAmplitude (*AMPLITUDE, TYPE=SMOOTH STEP)."""
        amp = SmoothStepAmplitude(name=str(name), data=data)
        self.amplitudes[str(name)] = amp
        return amp

    def UserFunctionAmplitude(
        self,
        name: str,
        py_func: Optional[Callable[[float, float, float], float]] = None,
        numba_func: Optional[Any] = None
    ) -> UserFunctionAmplitude:
        """Create and register a UserFunctionAmplitude with Python or Numba @njit kernel."""
        amp = UserFunctionAmplitude(name=str(name), py_func=py_func, numba_func=numba_func)
        self.amplitudes[str(name)] = amp
        return amp

    def DisplacementBC(
        self,
        name: str,
        createStepName: str,
        region: str,
        u1: Optional[float] = None,
        u2: Optional[float] = None,
        u3: Optional[float] = None,
        amplitude: Optional[Union[str, Amplitude]] = None
    ) -> DisplacementBC:
        """Create and assign a DisplacementBC to a Step."""
        amp_name = amplitude.name if hasattr(amplitude, "name") else amplitude
        bc = DisplacementBC(
            name=str(name),
            region=str(region),
            u1=u1,
            u2=u2,
            u3=u3,
            amplitude=amp_name
        )
        step = self.steps.get(str(createStepName), self.initial_step)
        step.add_boundary_condition(bc)
        return bc

    def UserFunctionBC(
        self,
        name: str,
        createStepName: str,
        region: str,
        py_func: Optional[Callable[[np.ndarray, float, float], Any]] = None,
        numba_func: Optional[Any] = None
    ) -> UserFunctionBC:
        """Create and assign a UserFunctionBC with Python or Numba @njit kernel to a Step."""
        bc = UserFunctionBC(
            name=str(name),
            region=str(region),
            py_func=py_func,
            numba_func=numba_func
        )
        step = self.steps.get(str(createStepName), self.initial_step)
        step.add_boundary_condition(bc)
        return bc

    def Sensor(
        self,
        name: str,
        entity_type: str = "node",
        entity_id: Any = None,
        variable: str = "U",
        comp: str = "MAG"
    ) -> Sensor:
        """Create and return a Sensor instance."""
        from dispsolver.model.sensor import Sensor
        return Sensor(
            name=str(name),
            entity_type=entity_type,
            entity_id=entity_id,
            variable=variable,
            comp=comp
        )

    def IterationHook(
        self,
        name: str,
        callback: Callable[[int, float, Any, Dict[str, float]], Any]
    ) -> IterationHook:
        """Create and return an IterationHook instance."""
        from dispsolver.model.sensor import IterationHook
        return IterationHook(name=str(name), callback=callback)

    # ─── CONSTRAINT FACTORY METHODS ────────────────────────────────────

    def RigidBody(
        self,
        name: str,
        refPoint: Any = None,
        tieNset: Optional[Any] = None,
        pinNset: Optional[Any] = None,
        elset: Optional[Any] = None,
        **kwargs
    ) -> RigidBody:
        """Create and register a RigidBody constraint (*RIGID BODY in Abaqus)."""
        rp = refPoint if refPoint is not None else kwargs.get("ref_point")
        tn = tieNset if tieNset is not None else kwargs.get("tie_nodes")
        pn = pinNset if pinNset is not None else kwargs.get("pin_nodes")
        be = elset if elset is not None else kwargs.get("body_elements")
        
        rb = RigidBody(
            name=str(name),
            ref_point=rp,
            tie_nodes=tn,
            pin_nodes=pn,
            body_elements=be,
            is_analytic=bool(kwargs.get("is_analytic", False))
        )
        self.constraints[str(name)] = rb
        return rb

    def Tie(
        self,
        name: str,
        master: Any,
        slave: Any,
        positionTolerance: Optional[float] = None,
        adjust: bool = True,
        tieRotations: bool = False,
        **kwargs
    ) -> Tie:
        """Create and register a Tie constraint (*TIE in Abaqus)."""
        pos_tol = positionTolerance if positionTolerance is not None else kwargs.get("position_tolerance")
        tie_rot = tieRotations if "tieRotations" in kwargs else kwargs.get("tie_rotations", False)
        
        tie = Tie(
            name=str(name),
            master=master,
            slave=slave,
            position_tolerance=pos_tol,
            adjust=bool(adjust),
            tie_rotations=bool(tie_rot),
            constraint_enforcement=str(kwargs.get("constraint_enforcement", "SURFACE_TO_SURFACE"))
        )
        self.constraints[str(name)] = tie
        return tie

    def Coupling(
        self,
        name: str,
        refPoint: Any = None,
        surface: Any = None,
        couplingType: str = "KINEMATIC",
        influenceRadius: Optional[float] = None,
        u1: bool = True,
        u2: bool = True,
        u3: bool = True,
        ur1: bool = True,
        ur2: bool = True,
        ur3: bool = True,
        weightingMethod: str = "UNIFORM",
        **kwargs
    ) -> Coupling:
        """Create and register a Coupling constraint (*COUPLING in Abaqus)."""
        rp = refPoint if refPoint is not None else kwargs.get("ref_point")
        surf = surface if surface is not None else kwargs.get("surface")
        c_type = couplingType if "couplingType" in kwargs or couplingType != "KINEMATIC" else kwargs.get("coupling_type", "KINEMATIC")
        
        w_method = kwargs.get("weighting_method", weightingMethod)
        c = Coupling(
            name=str(name),
            ref_point=rp,
            surface=surf,
            coupling_type=str(c_type).upper(),
            influence_radius=influenceRadius if influenceRadius is not None else kwargs.get("influence_radius"),
            u1=bool(u1), u2=bool(u2), u3=bool(u3),
            ur1=bool(ur1), ur2=bool(ur2), ur3=bool(ur3),
            weighting_method=str(w_method).upper()
        )
        self.constraints[str(name)] = c
        return c

    def KinematicCoupling(self, name: str, refPoint: Any = None, surface: Any = None, **kwargs) -> KinematicCoupling:
        """Create and register a KinematicCoupling constraint (RBE2)."""
        kwargs["coupling_type"] = "KINEMATIC"
        c = self.Coupling(name=name, refPoint=refPoint, surface=surface, **kwargs)
        kc = KinematicCoupling(
            name=c.name, ref_point=c.ref_point, surface=c.surface,
            influence_radius=c.influence_radius, u1=c.u1, u2=c.u2, u3=c.u3,
            ur1=c.ur1, ur2=c.ur2, ur3=c.ur3, weighting_method=c.weighting_method
        )
        self.constraints[str(name)] = kc
        return kc

    def DistributingCoupling(self, name: str, refPoint: Any = None, surface: Any = None, **kwargs) -> DistributingCoupling:
        """Create and register a DistributingCoupling constraint (RBE3)."""
        kwargs["coupling_type"] = "DISTRIBUTING"
        c = self.Coupling(name=name, refPoint=refPoint, surface=surface, **kwargs)
        dc = DistributingCoupling(
            name=c.name, ref_point=c.ref_point, surface=c.surface,
            influence_radius=c.influence_radius, u1=c.u1, u2=c.u2, u3=c.u3,
            ur1=c.ur1, ur2=c.ur2, ur3=c.ur3, weighting_method=c.weighting_method
        )
        self.constraints[str(name)] = dc
        return dc

    def MPC(
        self,
        name: str,
        mpcType: str = "PIN",
        firstPoint: Any = None,
        secondPoint: Optional[Any] = None,
        **kwargs
    ) -> MPC:
        """Create and register a Multi-Point Constraint (*MPC in Abaqus)."""
        m_type = mpcType if "mpcType" in kwargs or mpcType != "PIN" else kwargs.get("mpc_type", "PIN")
        fp = firstPoint if firstPoint is not None else kwargs.get("first_point")
        sp = secondPoint if secondPoint is not None else kwargs.get("second_point")
        
        mpc = MPC(
            name=str(name),
            mpc_type=str(m_type).upper(),
            first_point=fp,
            second_point=sp,
            dofs=kwargs.get("dofs"),
            coefficients=kwargs.get("coefficients"),
            constant=float(kwargs.get("constant", 0.0))
        )
        self.constraints[str(name)] = mpc
        return mpc

    # ─── LOAD FACTORY METHODS ──────────────────────────────────────────

    def ConcentratedForce(
        self,
        name: str,
        createStepName: str = "Initial",
        region: Any = "",
        cf1: Optional[float] = None,
        cf2: Optional[float] = None,
        cf3: Optional[float] = None,
        cm1: Optional[float] = None,
        cm2: Optional[float] = None,
        cm3: Optional[float] = None,
        amplitude: Optional[Union[str, Amplitude]] = None,
        **kwargs
    ) -> ConcentratedForce:
        """Create and assign a ConcentratedForce to a Step (*CLOAD in Abaqus)."""
        amp_name = amplitude.name if hasattr(amplitude, "name") else amplitude
        load = ConcentratedForce(
            name=str(name),
            create_step_name=str(createStepName),
            region=str(region) if isinstance(region, str) else region,
            cf1=cf1, cf2=cf2, cf3=cf3,
            cm1=cm1, cm2=cm2, cm3=cm3,
            amplitude=amp_name
        )
        step = self.steps.get(str(createStepName), self.initial_step)
        step.add_load(load)
        return load

    def Pressure(
        self,
        name: str,
        createStepName: str = "Initial",
        region: Any = "",
        magnitude: float = 0.0,
        amplitude: Optional[Union[str, Amplitude]] = None,
        distribution: str = "UNIFORM",
        **kwargs
    ) -> Pressure:
        """Create and assign a surface Pressure load to a Step (*PRESSURE in Abaqus)."""
        amp_name = amplitude.name if hasattr(amplitude, "name") else amplitude
        load = Pressure(
            name=str(name),
            create_step_name=str(createStepName),
            region=str(region) if isinstance(region, str) else region,
            magnitude=float(magnitude),
            amplitude=amp_name,
            distribution=str(distribution).upper()
        )
        step = self.steps.get(str(createStepName), self.initial_step)
        step.add_load(load)
        return load

    def Gravity(
        self,
        name: str,
        createStepName: str = "Initial",
        comp1: float = 0.0,
        comp2: float = 0.0,
        comp3: float = 0.0,
        amplitude: Optional[Union[str, Amplitude]] = None,
        region: Optional[Any] = None,
        **kwargs
    ) -> Gravity:
        """Create and assign a Gravity body force load to a Step (*DLOAD, GRAV in Abaqus)."""
        amp_name = amplitude.name if hasattr(amplitude, "name") else amplitude
        load = Gravity(
            name=str(name),
            create_step_name=str(createStepName),
            comp1=float(comp1),
            comp2=float(comp2),
            comp3=float(comp3),
            amplitude=amp_name,
            region=region
        )
        step = self.steps.get(str(createStepName), self.initial_step)
        step.add_load(load)
        return load

    def BodyForce(
        self,
        name: str,
        createStepName: str = "Initial",
        region: Optional[Any] = None,
        b1: float = 0.0,
        b2: float = 0.0,
        b3: float = 0.0,
        amplitude: Optional[Union[str, Amplitude]] = None,
        **kwargs
    ) -> BodyForce:
        """Create and assign a volumetric BodyForce load to a Step."""
        amp_name = amplitude.name if hasattr(amplitude, "name") else amplitude
        load = BodyForce(
            name=str(name),
            create_step_name=str(createStepName),
            region=region,
            b1=float(b1), b2=float(b2), b3=float(b3),
            amplitude=amp_name
        )
        step = self.steps.get(str(createStepName), self.initial_step)
        step.add_load(load)
        return load

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
        
        # Note: DynamicSolver3D._setup_numba_topology() automatically builds the
        # correct heterogeneous kernel-grouped rows_topo and cols_topo.
        # Overwriting with sys.rows_topo corrupts the sparsity mapping for multi-material meshes.
            
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
                            u1_val = getattr(bc, "u1", None)
                            u2_val = getattr(bc, "u2", None)
                            u3_val = getattr(bc, "u3", None)
                            if u1_val is not None:
                                solver.fix_dof(gid, 0, float(u1_val))
                            if u2_val is not None:
                                solver.fix_dof(gid, 1, float(u2_val))
                            if u3_val is not None:
                                solver.fix_dof(gid, 2, float(u3_val))

        # 3. Apply Predefined Fields (Initial Stress/SDV)
        if len(sys.predefined_fields) > 0:
            inv_eid = {idx: g for g, idx in sys.elem_local_to_global.items()} if hasattr(sys, 'elem_local_to_global') else {}
            # Fallback if elem_local_to_global is not structured that way, just use idx -> gid
            # Wait, sys.elem_local_to_global is Dict[Tuple[str, int], int], where int is gid.
            # But we want 0-based index to gid. In sys.global_elsets, it gives 0-based indices.
            # So gid = idx + 1
            for pf in sys.predefined_fields:
                if pf.field_type.upper() == "STRESS":
                    el_indices = sys.global_elsets.get(pf.region, [])
                    for idx in el_indices:
                        gid = idx + 1
                        states = solver.element_states.get(gid)
                        if states:
                            # pf.values can be scalar, array, or callable
                            for q, state in enumerate(states):
                                # Determine stress vector
                                s_vec = np.zeros(6, dtype=np.float64)
                                if callable(pf.values):
                                    # We don't have natural coordinates handy here, so we approximate with element centroid
                                    # In a full implementation, we'd query the exact quadrature point coords
                                    # For now, approximate centroid:
                                    conn = sys.elem_conn_0based[idx]
                                    coords = sys.coords[conn]
                                    cx, cy, cz = np.mean(coords, axis=0)
                                    val = pf.values(cx, cy, cz)
                                    s_vec[:] = val
                                elif isinstance(pf.values, (list, np.ndarray, tuple)):
                                    s_vec[:] = pf.values
                                else:
                                    # scalar -> hydrostatic pressure
                                    p = float(pf.values)
                                    s_vec[0] = p
                                    s_vec[1] = p
                                    s_vec[2] = p
                                state.stress_initial[:] = s_vec
                            
                            # Also update the DOD fast path array if it exists
                            if hasattr(solver, "elem_stress_init") and solver.elem_stress_init is not None:
                                solver.elem_stress_init[idx, :, :] = s_vec
                            
        return solver, sys
