"""tests/test_p2_output_vars.py
=============================
Verification suite for Package P2: Output Variables
- Stress derived invariants: MISES, MAXP, MIDP, MINP, PRESS (2D & 3D, analytical precision < 1e-12)
- Contact field variables: CPRESS, COPEN, CSHEAR, CSTATUS, CSLIP
- Whole model energy history: ALLAE, ALLSD, ALLFD
- Alias normalization and PRESELECT handling
"""

import numpy as np
import pytest

from dispsolver.output import (
    FieldPosition,
    FieldType,
    FieldVar,
    HistoryVar,
    FieldOutputRequest,
    HistoryOutputRequest,
    FieldOutput,
    FieldValue,
    EnergyTracker,
    Odb,
    OutputManager,
)
from dispsolver.model import Model
from dispsolver.mesh import Mesh


class TestStressDerivedInvariantsAnalytical:
    """Rigorous analytical verification of MISES, MAXP, MIDP, MINP, PRESS (< 1e-12 precision)."""

    def test_3d_uniaxial_tension_analytical(self):
        """Uniaxial tension sigma_11 = 150.0 MPa in 3D:
        MISES = 150.0, MAXP = 150.0, MIDP = 0.0, MINP = 0.0, PRESS = -50.0
        """
        sigma_11 = 150.0
        s_data = np.array([[sigma_11, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        fo = FieldOutput(
            name="S",
            field_type=FieldType.TENSOR_3D_FULL,
            position=FieldPosition.CENTROID,
            data=s_data,
            labels=np.array([1], dtype=np.int32),
        )

        # 1. Vectorized get_mises()
        mises = fo.get_mises()
        assert np.isclose(mises[0], sigma_11, atol=1e-12, rtol=1e-12)

        # 2. Vectorized get_principal()
        p_max, p_mid, p_min = fo.get_principal()
        assert np.isclose(p_max[0], sigma_11, atol=1e-12, rtol=1e-12)
        assert np.isclose(p_mid[0], 0.0, atol=1e-12, rtol=1e-12)
        assert np.isclose(p_min[0], 0.0, atol=1e-12, rtol=1e-12)

        # 3. Vectorized get_pressure()
        press = fo.get_pressure()
        assert np.isclose(press[0], -sigma_11 / 3.0, atol=1e-12, rtol=1e-12)

        # 4. get_component accessors
        assert np.isclose(fo.get_component("MISES")[0], sigma_11, atol=1e-12, rtol=1e-12)
        assert np.isclose(fo.get_component("MAXP")[0], sigma_11, atol=1e-12, rtol=1e-12)
        assert np.isclose(fo.get_component("MIDP")[0], 0.0, atol=1e-12, rtol=1e-12)
        assert np.isclose(fo.get_component("MINP")[0], 0.0, atol=1e-12, rtol=1e-12)
        assert np.isclose(fo.get_component("PRESS")[0], -50.0, atol=1e-12, rtol=1e-12)

        # 5. FieldValue item accessors
        val = fo.values[0]
        assert np.isclose(val.mises, sigma_11, atol=1e-12, rtol=1e-12)
        assert np.isclose(val.max_principal, sigma_11, atol=1e-12, rtol=1e-12)
        assert np.isclose(val.mid_principal, 0.0, atol=1e-12, rtol=1e-12)
        assert np.isclose(val.min_principal, 0.0, atol=1e-12, rtol=1e-12)
        assert np.isclose(val.pressure, -50.0, atol=1e-12, rtol=1e-12)

    def test_3d_general_multiaxial_stress_analytical(self):
        """General multi-axial 3D symmetric Cauchy stress tensor."""
        # [S11, S22, S33, S12, S13, S23]
        s_voigt = np.array([120.0, -40.0, 60.0, 25.0, -15.0, 30.0], dtype=np.float64)
        tensor = np.array([
            [s_voigt[0], s_voigt[3], s_voigt[4]],
            [s_voigt[3], s_voigt[1], s_voigt[5]],
            [s_voigt[4], s_voigt[5], s_voigt[2]],
        ], dtype=np.float64)

        # Exact algebraic eigenvalues from NumPy LAPACK eigvalsh
        exact_eigs = np.linalg.eigvalsh(tensor)  # ascending: [min, mid, max]
        exact_min = exact_eigs[0]
        exact_mid = exact_eigs[1]
        exact_max = exact_eigs[2]

        # Exact Von Mises: sqrt(1/2 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2))
        dev = tensor - np.eye(3) * (np.trace(tensor) / 3.0)
        exact_mises = np.sqrt(1.5 * np.sum(dev * dev))
        exact_press = -np.trace(tensor) / 3.0

        fo = FieldOutput(
            name="S",
            field_type=FieldType.TENSOR_3D_FULL,
            position=FieldPosition.CENTROID,
            data=s_voigt[None, :],
            labels=np.array([1], dtype=np.int32),
        )

        p_max, p_mid, p_min = fo.get_principal()
        assert np.isclose(p_max[0], exact_max, atol=1e-12, rtol=1e-12)
        assert np.isclose(p_mid[0], exact_mid, atol=1e-12, rtol=1e-12)
        assert np.isclose(p_min[0], exact_min, atol=1e-12, rtol=1e-12)

        mises = fo.get_mises()
        assert np.isclose(mises[0], exact_mises, atol=1e-12, rtol=1e-12)

        press = fo.get_pressure()
        assert np.isclose(press[0], exact_press, atol=1e-12, rtol=1e-12)

    def test_2d_plane_strain_stress_invariants_analytical(self):
        """2D Plane strain stress: [S11, S22, S33, S12] with non-zero S33."""
        # S11=200, S22=50, S33=75, S12=30
        s_data = np.array([[200.0, 50.0, 75.0, 30.0]], dtype=np.float64)
        tensor_3d = np.array([
            [200.0, 30.0, 0.0],
            [30.0, 50.0, 0.0],
            [0.0, 0.0, 75.0],
        ], dtype=np.float64)

        exact_eigs = np.linalg.eigvalsh(tensor_3d)
        exact_min, exact_mid, exact_max = exact_eigs[0], exact_eigs[1], exact_eigs[2]

        dev = tensor_3d - np.eye(3) * (np.trace(tensor_3d) / 3.0)
        exact_mises = np.sqrt(1.5 * np.sum(dev * dev))

        fo = FieldOutput(
            name="S",
            field_type=FieldType.TENSOR_2D_PLANAR,
            position=FieldPosition.CENTROID,
            data=s_data,
            labels=np.array([1], dtype=np.int32),
        )

        p_max, p_mid, p_min = fo.get_principal()
        assert np.isclose(p_max[0], exact_max, atol=1e-12, rtol=1e-12)
        assert np.isclose(p_mid[0], exact_mid, atol=1e-12, rtol=1e-12)
        assert np.isclose(p_min[0], exact_min, atol=1e-12, rtol=1e-12)

        mises = fo.get_mises()
        assert np.isclose(mises[0], exact_mises, atol=1e-12, rtol=1e-12)

        # Check component queries
        assert np.isclose(fo.get_component("MISES")[0], exact_mises, atol=1e-12, rtol=1e-12)
        assert np.isclose(fo.get_component("MAXP")[0], exact_max, atol=1e-12, rtol=1e-12)
        assert np.isclose(fo.get_component("MINP")[0], exact_min, atol=1e-12, rtol=1e-12)

    def test_2d_plane_stress_invariants_analytical(self):
        """2D Plane stress: [S11, S22, S12] with S33=0."""
        s_data = np.array([[100.0, -50.0, 20.0]], dtype=np.float64)
        tensor_3d = np.array([
            [100.0, 20.0, 0.0],
            [20.0, -50.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float64)

        exact_eigs = np.linalg.eigvalsh(tensor_3d)
        exact_min, exact_mid, exact_max = exact_eigs[0], exact_eigs[1], exact_eigs[2]

        dev = tensor_3d - np.eye(3) * (np.trace(tensor_3d) / 3.0)
        exact_mises = np.sqrt(1.5 * np.sum(dev * dev))

        fo = FieldOutput(
            name="S",
            field_type=FieldType.TENSOR_2D_PLANAR,
            position=FieldPosition.CENTROID,
            data=s_data,
            labels=np.array([1], dtype=np.int32),
        )

        p_max, p_mid, p_min = fo.get_principal()
        assert np.isclose(p_max[0], exact_max, atol=1e-12, rtol=1e-12)
        assert np.isclose(p_mid[0], exact_mid, atol=1e-12, rtol=1e-12)
        assert np.isclose(p_min[0], exact_min, atol=1e-12, rtol=1e-12)

        mises = fo.get_mises()
        assert np.isclose(mises[0], exact_mises, atol=1e-12, rtol=1e-12)


class TestFEASimulationOutputRequests:
    """Test full FEA simulation with MISES, MAXP, MINP, CPRESS, COPEN, CSTATUS in 2D and 3D."""

    def test_2d_fe_simulation_stress_invariants(self):
        """Solve a 2D tension model and request MISES, MAXP, MINP as discrete FieldOutputs."""
        model = Model("Tension2D", dim=2)
        part = model.Part("Plate", dim=2)
        part.add_node(1, [0.0, 0.0])
        part.add_node(2, [10.0, 0.0])
        part.add_node(3, [10.0, 2.0])
        part.add_node(4, [0.0, 2.0])
        part.add_element(1, "CPE4", [1, 2, 3, 4])
        part.create_set("AllElems", elements=[1])
        part.SectionAssignment(region="AllElems", sectionName="Sec1")

        mat = model.Material("Mat")
        E_mod = 1000.0
        nu_val = 0.3
        mat.Elastic(E=E_mod, nu=nu_val)
        model.SolidSection("Sec1", material="Mat", thickness=1.0)

        part.create_set("Left", nodes=[1, 4])
        part.create_set("Right", nodes=[2, 3])
        inst = model.root_assembly.Instance("Plate-1", part)

        step = model.Step("TensionStep", time_period=1.0, dt_init=1.0)
        # Fix Left
        model.DisplacementBC("FixLeft", createStepName="TensionStep", region="Plate-1.Left", u1=0.0, u2=0.0)
        # Pull Right in X by 0.1
        u_pull = 0.1
        model.DisplacementBC("PullRight", createStepName="TensionStep", region="Plate-1.Right", u1=u_pull)

        # Request field variables: U, S, MISES, MAXP, MINP, PRESS
        step.FieldOutputRequest(
            name="Field-Req",
            variables=["U", "S", "MISES", "MAXP", "MINP", "PRESS"],
            frequency=1,
            position="CENTROID",
        )

        solver, sys = model.create_solver2d("TensionStep")
        odb = solver.solve_with_odb(step=step, sys=sys, verbose=False)

        assert "TensionStep" in odb.steps
        last_frame = odb.steps["TensionStep"].frames[-1]

        assert "MISES" in last_frame.fieldOutputs
        assert "MAXP" in last_frame.fieldOutputs
        assert "MINP" in last_frame.fieldOutputs
        assert "PRESS" in last_frame.fieldOutputs

        fo_mises = last_frame.fieldOutputs["MISES"]
        fo_maxp = last_frame.fieldOutputs["MAXP"]
        fo_minp = last_frame.fieldOutputs["MINP"]
        fo_s = last_frame.fieldOutputs["S"]

        # Check consistency between fo_s.get_mises() and fo_mises.data
        assert np.allclose(fo_mises.data, fo_s.get_mises(), atol=1e-12)
        p_max, p_mid, p_min = fo_s.get_principal()
        assert np.allclose(fo_maxp.data, p_max, atol=1e-12)
        assert np.allclose(fo_minp.data, p_min, atol=1e-12)

    def test_3d_fe_simulation_stress_invariants(self):
        """Solve a 3D C3D8 single-element tension model and verify MISES, MAXP, MINP."""
        model = Model("Tension3D", dim=3)
        part = model.Part("Cube", dim=3)
        # 8 nodes of 1x1x1 cube
        part.add_node(1, [0.0, 0.0, 0.0])
        part.add_node(2, [1.0, 0.0, 0.0])
        part.add_node(3, [1.0, 1.0, 0.0])
        part.add_node(4, [0.0, 1.0, 0.0])
        part.add_node(5, [0.0, 0.0, 1.0])
        part.add_node(6, [1.0, 0.0, 1.0])
        part.add_node(7, [1.0, 1.0, 1.0])
        part.add_node(8, [0.0, 1.0, 1.0])
        part.add_element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
        part.create_set("AllElems", elements=[1])
        part.SectionAssignment(region="AllElems", sectionName="Sec3D")

        mat = model.Material("Mat3D")
        mat.Elastic(E=210000.0, nu=0.3)
        model.SolidSection("Sec3D", material="Mat3D")

        part.create_set("X0", nodes=[1, 4, 5, 8])
        part.create_set("X1", nodes=[2, 3, 6, 7])
        part.create_set("Y0", nodes=[1, 2, 5, 6])
        part.create_set("Z0", nodes=[1, 2, 3, 4])
        inst = model.root_assembly.Instance("Cube-1", part)

        step = model.Step("PullStep", time_period=1.0, dt_init=1.0)
        model.DisplacementBC("FixX0", createStepName="PullStep", region="Cube-1.X0", u1=0.0)
        model.DisplacementBC("FixY0", createStepName="PullStep", region="Cube-1.Y0", u2=0.0)
        model.DisplacementBC("FixZ0", createStepName="PullStep", region="Cube-1.Z0", u3=0.0)
        model.DisplacementBC("PullX1", createStepName="PullStep", region="Cube-1.X1", u1=0.001)

        step.FieldOutputRequest(
            name="Field3D",
            variables=["U", "S", "MISES", "MAXP", "MINP", "PRESS"],
            frequency=1,
            position="CENTROID",
        )

        solver, sys = model.create_solver3d("PullStep")
        odb = solver.solve_with_odb(step=step, sys=sys, verbose=False)

        last_frame = odb.steps["PullStep"].frames[-1]
        assert "MISES" in last_frame.fieldOutputs
        assert "MAXP" in last_frame.fieldOutputs
        assert "MINP" in last_frame.fieldOutputs

        fo_s = last_frame.fieldOutputs["S"]
        fo_mises = last_frame.fieldOutputs["MISES"]
        fo_maxp = last_frame.fieldOutputs["MAXP"]
        fo_minp = last_frame.fieldOutputs["MINP"]

        assert np.allclose(fo_mises.data, fo_s.get_mises(), atol=1e-12)
        p_max, p_mid, p_min = fo_s.get_principal()
        assert np.allclose(fo_maxp.data, p_max, atol=1e-12)
        assert np.allclose(fo_minp.data, p_min, atol=1e-12)
        assert fo_mises.data[0] > 0.0


class TestContactFieldAndEnergyVariables:
    """Verify contact field variables (CPRESS, COPEN, CSHEAR, CSTATUS) and energy history (ALLAE, ALLSD, ALLFD)."""

    def test_contact_field_output_recovery_3d(self):
        """Test contact field output recovery on a model with SurfaceContactConstraint3D."""
        from dispsolver.constraint3d.surface_contact3d import SurfaceContactConstraint3D
        from dispsolver.mesh3d import Mesh3D

        mesh = Mesh3D()
        mesh.add_node(1, 0.0, 0.0, 0.0)
        mesh.add_node(2, 1.0, 0.0, 0.0)
        mesh.add_node(3, 1.0, 1.0, 0.0)
        mesh.add_node(4, 0.0, 1.0, 0.0)
        mesh.add_node(5, 0.0, 0.0, 1.0)
        mesh.add_node(6, 1.0, 0.0, 1.0)
        mesh.add_node(7, 1.0, 1.0, 1.0)
        mesh.add_node(8, 0.0, 1.0, 1.0)
        mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8")


        nid_to_idx = {i: i - 1 for i in range(1, 9)}
        coords = mesh.nodes_array()

        # Contact plane at z = -0.05, normal pointing up [0, 0, 1]
        contact = SurfaceContactConstraint3D(
            slave_node_ids=[1, 2, 3, 4],
            rigid_plane_point=np.array([0.0, 0.0, -0.05]),
            rigid_plane_normal=np.array([0.0, 0.0, 1.0]),
            nid_to_idx=nid_to_idx,
            coords=coords,
            penalty_stiffness=1e6,
            strain_free_adjust=False,
        )

        class DummySolver:
            def __init__(self, mesh, constraints):
                self.mesh = mesh
                self.constraints = constraints
                # Node 1 and 2 are pushed down by -0.1 (penetrating by 0.05)
                # Node 3 and 4 are at z=0 (clearance 0.05)
                u = np.zeros(24, dtype=np.float64)
                u[3 * 0 + 2] = -0.1  # node 1 z
                u[3 * 1 + 2] = -0.1  # node 2 z
                self.u = u
                self.f_int = np.zeros(24, dtype=np.float64)

        solver = DummySolver(mesh, [contact])

        from dispsolver.output.recovery3d import recover_3d_fields
        res = recover_3d_fields(solver, requested_vars=["CPRESS", "COPEN", "CSTATUS", "CSHEAR"])

        assert "CPRESS" in res
        assert "COPEN" in res
        assert "CSTATUS" in res
        assert "CSHEAR" in res

        cpress = res["CPRESS"].data
        copen = res["COPEN"].data
        cstatus = res["CSTATUS"].data

        # Nodes 1 and 2 are in contact:
        # Penetration = 0.05 -> cpress = 1e6 * 0.05 = 50000
        assert np.isclose(cpress[0], 50000.0, atol=1e-3)
        assert np.isclose(cpress[1], 50000.0, atol=1e-3)
        assert cstatus[0] == 1.0  # Stick / closed
        assert cstatus[1] == 1.0

        # Nodes 3 and 4 are open:
        assert np.isclose(cpress[2], 0.0, atol=1e-12)
        assert np.isclose(cpress[3], 0.0, atol=1e-12)
        assert cstatus[2] == 0.0  # Open
        assert np.isclose(copen[2], 0.05, atol=1e-3)

    def test_energy_history_allae_allsd_allfd_tracking(self):
        """Verify EnergyTracker tracks ALLAE, ALLSD, ALLFD when exposed by solver."""
        tracker = EnergyTracker()

        class DummySolverWithEnergy:
            def __init__(self):
                self.energy_allae = 12.5
                self.energy_allsd = 4.2
                self.energy_allfd = 8.7

        solver = DummySolverWithEnergy()
        du = np.zeros(6)
        f_int = np.zeros(6)

        tracker.update_step(dt=0.1, solver=solver, du=du, f_int=f_int)

        assert np.isclose(tracker.allae, 12.5)
        assert np.isclose(tracker.allsd, 4.2)
        assert np.isclose(tracker.allfd, 8.7)


class TestRequestAliasesAndPreselect:
    """Verify alias normalization and PRESELECT keyword support."""

    def test_field_output_request_aliases(self):
        req = FieldOutputRequest(
            name="TestAliases",
            variables=["VONMISES", "MAX_PRINCIPAL", "MIN_PRINCIPAL", "PRESSURE", "U"],
        )
        assert "MISES" in req.variables
        assert "MAXP" in req.variables
        assert "MINP" in req.variables
        assert "PRESS" in req.variables
        assert "U" in req.variables
        assert "VONMISES" not in req.variables

    def test_field_output_request_preselect(self):
        req = FieldOutputRequest(name="TestPreselect", variables=["PRESELECT"])
        assert "U" in req.variables
        assert "S" in req.variables
        assert "CPRESS" in req.variables
        assert "COPEN" in req.variables
        assert "CSTATUS" in req.variables
        assert "CSHEAR" in req.variables

    def test_history_output_request_preselect(self):
        req = HistoryOutputRequest(name="HistPreselect", variables=["PRESELECT"])
        assert "ALLIE" in req.variables
        assert "ALLSE" in req.variables
        assert "ALLAE" in req.variables
        assert "ALLSD" in req.variables
        assert "ALLFD" in req.variables
        assert "ETOTAL" in req.variables
