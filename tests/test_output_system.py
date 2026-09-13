"""tests/test_output_system.py
============================
Comprehensive test suite for Modern Field Output, History Output, and ODB System.
Verifies Abaqus API parity, NumPy vectorization, HDF5 persistence, and VTKHDF export.
"""

import os
import tempfile
import numpy as np
import pytest
import h5py as h5

from dispsolver.output import (
    FieldPosition,
    FieldType,
    FieldOutputRequest,
    HistoryOutputRequest,
    OutputTimingController,
    FieldOutput,
    FieldValue,
    HistoryOutput,
    HistoryRegion,
    EnergyTracker,
    Odb,
    open_odb,
    TransientVTKHDFWriter,
)
from dispsolver.model import Model
from dispsolver.mesh import Mesh, Node, Element


class TestOutputRequestsAndTiming:
    """Test timing controller logic for frequency, num_intervals, time_interval, time_points."""

    def test_frequency_timing(self):
        req = FieldOutputRequest(name="F-1", frequency=3)
        ctrl = OutputTimingController(req, step_time_period=1.0)

        # inc=0 should always output
        assert ctrl.should_output(step_time=0.0, inc=0) is True
        # inc=1, 2 should not
        assert ctrl.should_output(step_time=0.1, inc=1) is False
        assert ctrl.should_output(step_time=0.2, inc=2) is False
        # inc=3 should output
        assert ctrl.should_output(step_time=0.3, inc=3) is True

    def test_num_intervals_timing(self):
        # 5 intervals over 1.0s -> target times: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        req = FieldOutputRequest(name="F-Intervals", num_intervals=5, frequency=None)
        ctrl = OutputTimingController(req, step_time_period=1.0)

        assert ctrl.should_output(step_time=0.0, inc=0) is True
        ctrl.mark_recorded(0.0, 0)

        # 0.1s -> between 0.0 and 0.2 -> no target crossed
        assert ctrl.should_output(step_time=0.1, inc=1) is False

        # 0.21s -> crossed 0.2 target
        assert ctrl.should_output(step_time=0.21, inc=2) is True
        ctrl.mark_recorded(0.21, 2)

        # 0.3s -> no target crossed
        assert ctrl.should_output(step_time=0.3, inc=3) is False

        # 0.4s -> target 0.4 crossed
        assert ctrl.should_output(step_time=0.4, inc=4) is True

    def test_fixed_time_interval_timing(self):
        req = FieldOutputRequest(name="F-Dt", time_interval=0.25, frequency=None)
        ctrl = OutputTimingController(req, step_time_period=1.0)

        ctrl.mark_recorded(0.0, 0)
        assert ctrl.should_output(step_time=0.1, inc=1) is False
        assert ctrl.should_output(step_time=0.25, inc=2) is True
        ctrl.mark_recorded(0.25, 2)
        assert ctrl.should_output(step_time=0.3, inc=3) is False
        assert ctrl.should_output(step_time=0.51, inc=4) is True

    def test_explicit_time_points(self):
        req = FieldOutputRequest(name="F-Points", time_points=[0.15, 0.45, 0.9], frequency=None)
        ctrl = OutputTimingController(req, step_time_period=1.0)

        ctrl.mark_recorded(0.0, 0)
        assert ctrl.should_output(step_time=0.1, inc=1) is False
        assert ctrl.should_output(step_time=0.16, inc=2) is True
        ctrl.mark_recorded(0.16, 2)
        assert ctrl.should_output(step_time=0.3, inc=3) is False
        assert ctrl.should_output(step_time=0.45, inc=4) is True


class TestFieldDataVectorization:
    """Test FieldOutput batch NumPy vectorization and Abaqus .values compatibility."""

    def test_stress_mises_vectorization(self):
        # 3D Voigt: [S11, S22, S33, S12, S13, S23]
        s_data = np.array([
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],          # Uniaxial tension -> Mises = 100
            [0.0, 0.0, 0.0, 57.7350269, 0.0, 0.0],      # Pure shear -> Mises = sqrt(3)*57.735 ~ 100
            [100.0, 100.0, 100.0, 0.0, 0.0, 0.0],      # Hydrostatic -> Mises = 0
        ], dtype=np.float64)

        fo = FieldOutput(
            name="S",
            field_type=FieldType.TENSOR_3D_FULL,
            position=FieldPosition.CENTROID,
            data=s_data,
            labels=np.array([1, 2, 3])
        )

        # 1. Vectorized get_mises()
        mises = fo.get_mises()
        assert np.isclose(mises[0], 100.0, atol=1e-5)
        assert np.isclose(mises[1], 100.0, atol=1e-3)
        assert np.isclose(mises[2], 0.0, atol=1e-5)

        # 2. Vectorized get_pressure()
        press = fo.get_pressure()
        assert np.isclose(press[0], -100.0 / 3.0)
        assert np.isclose(press[2], -100.0)

        # 3. Component extraction
        s11 = fo.get_component("S11")
        assert np.allclose(s11, [100.0, 0.0, 100.0])

        # 4. Abaqus .values iteration compatibility
        vals = fo.values
        assert len(vals) == 3
        assert vals[0].element_label == 1
        assert np.isclose(vals[0].mises, 100.0, atol=1e-5)
        assert np.isclose(vals[2].mises, 0.0, atol=1e-5)

    def test_vector_magnitude_and_subsets(self):
        u_data = np.array([
            [3.0, 4.0, 0.0],
            [1.0, 2.0, 2.0],
            [0.0, 0.0, 5.0],
        ], dtype=np.float64)

        fo = FieldOutput(
            name="U",
            field_type=FieldType.VECTOR,
            position=FieldPosition.NODES,
            data=u_data,
            labels=np.array([10, 20, 30])
        )

        mag = fo.get_magnitude()
        assert np.allclose(mag, [5.0, 3.0, 5.0])

        # Subset filtering
        sub = fo.get_subset(region_labels=[20, 30])
        assert len(sub.data) == 2
        assert np.array_equal(sub.labels, [20, 30])
        assert np.allclose(sub.get_magnitude(), [3.0, 5.0])


class TestHistoryOutputAndPandas:
    """Test HistoryOutput recording and Pandas DataFrame integration."""

    def test_history_region_dataframe(self):
        hr = HistoryRegion(name="WholeModel")
        hr.record("ALLIE", 0.0, 0.0)
        hr.record("ALLIE", 0.5, 12.5)
        hr.record("ALLIE", 1.0, 50.0)

        hr.record("ALLKE", 0.0, 0.0)
        hr.record("ALLKE", 0.5, 1.2)
        hr.record("ALLKE", 1.0, 0.0)

        # Abaqus tuple sequence compatibility
        allie_ho = hr.historyOutputs["ALLIE"]
        assert allie_ho.data == ((0.0, 0.0), (0.5, 12.5), (1.0, 50.0))

        # Modern Pandas DataFrame conversion
        df = hr.to_dataframe()
        assert list(df.columns) == ["Time", "ALLIE", "ALLKE"]
        assert len(df) == 3
        assert df["ALLIE"].iloc[-1] == 50.0


class TestVTKHDFExportAndH5ODB:
    """Test VTKHDF export and HDF5 persistent storage/retrieval."""

    def test_vtkhdf_2d_and_3d(self):
        # Create a simple 2D Mesh with 2 Quad elements
        mesh = Mesh()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 1.0, 0.0)
        mesh.add_node(3, 1.0, 1.0)
        mesh.add_node(4, 0.0, 1.0)
        mesh.add_node(5, 2.0, 0.0)
        mesh.add_node(6, 2.0, 1.0)

        mesh.add_element(1, [1, 2, 3, 4], "QUAD4", pid=1)
        mesh.add_element(2, [2, 5, 6, 3], "QUAD4", pid=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            vtkhdf_path = os.path.join(tmpdir, "test_output.vtkhdf")
            writer = TransientVTKHDFWriter(filepath=vtkhdf_path, mesh=mesh)

            # Step 1 (t=0.0)
            u0 = np.zeros((6, 2), dtype=np.float32)
            writer.add_step(time_val=0.0, u=u0)

            # Step 2 (t=0.5)
            u1 = np.ones((6, 2), dtype=np.float32) * 0.1
            writer.add_step(
                time_val=0.5,
                u=u1,
                cell_data={"Mises": np.array([25.0, 30.0], dtype=np.float32)}
            )

            writer.close()

            # Verify standard VTKHDF file structure using h5py
            assert os.path.exists(vtkhdf_path)
            with h5.File(vtkhdf_path, "r") as f:
                assert "VTKHDF" in f
                g = f["VTKHDF"]
                assert g.attrs["Version"].tolist() == [2, 2]
                assert "Points" in g
                assert "PointData/Displacement" in g
                assert "CellData/Mises" in g
                assert "Steps" in g
                assert g["Steps"].attrs["NSteps"] == 2
                assert np.allclose(g["Steps/Values"][:], [0.0, 0.5])

    def test_odb_save_and_open(self):
        mesh = Mesh()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 1.0, 0.0)
        mesh.add_node(3, 1.0, 1.0)
        mesh.add_node(4, 0.0, 1.0)
        mesh.add_element(1, [1, 2, 3, 4], "QUAD4", pid=1)

        odb = Odb(name="TestJob", mesh=mesh)
        step = odb.get_or_create_step("Step-1", procedure="STATIC", time_period=1.0)

        # Add Frame
        u_arr = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.2], [0.0, 0.2]])
        fo_u = FieldOutput("U", field_type=FieldType.VECTOR, position=FieldPosition.NODES, data=u_arr, labels=np.array([1, 2, 3, 4]))
        fo_s = FieldOutput("S", field_type=FieldType.TENSOR_2D_PLANAR, position=FieldPosition.CENTROID, data=np.array([[100.0, 50.0, 15.0, 0.0]]), labels=np.array([1]))

        from dispsolver.output import OdbFrame
        fr = OdbFrame(frame_id=1, frame_value=0.5, description="Inc 1")
        fr.add_field_output(fo_u)
        fr.add_field_output(fo_s)
        step.add_frame(fr)

        # Add History
        hr = step.get_or_create_history_region("WholeModel")
        hr.record("ALLIE", 0.5, 42.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "job.h5odb")
            odb.save(h5_path)

            # Re-open and verify
            loaded = open_odb(h5_path)
            assert loaded.name == "TestJob"
            assert "Step-1" in loaded.steps
            l_step = loaded.steps["Step-1"]
            assert len(l_step.frames) == 1
            l_frame = l_step.frames[0]
            assert l_frame.frameValue == 0.5
            assert "U" in l_frame.fieldOutputs
            assert "S" in l_frame.fieldOutputs
            assert np.allclose(l_frame.fieldOutputs["U"].data, u_arr)
            assert l_step.historyRegions["WholeModel"].historyOutputs["ALLIE"].values[0] == 42.0


class TestModelAndSolverIntegration:
    """Test end-to-end integration: Model -> Step -> Output Requests -> solve_with_odb -> ODB."""

    def test_2d_solve_with_odb(self):
        # 1. Build a simple 2D cantilever Model
        model = Model("CantileverTest", dim=2)
        part = model.Part("Bar", dim=2)
        part.add_node(1, [0.0, 0.0])
        part.add_node(2, [10.0, 0.0])
        part.add_node(3, [10.0, 2.0])
        part.add_node(4, [0.0, 2.0])
        part.add_element(1, "CPE4", [1, 2, 3, 4])
        part.create_set("AllElems", elements=[1])
        part.SectionAssignment(region="AllElems", sectionName="Sec1")

        mat = model.Material("Steel")
        mat.Elastic(E=200000.0, nu=0.3)
        model.SolidSection("Sec1", material="Steel", thickness=1.0)

        part.create_set("FixedNodes", nodes=[1, 4])
        part.create_set("TipNode", nodes=[3])

        # Instance into assembly
        inst = model.root_assembly.Instance("Bar-1", part)

        # Step
        step = model.Step("BendStep", time_period=1.0, dt_init=0.5, dt_max=0.5)

        # Fixed left edge (nodes 1, 4)
        model.DisplacementBC("FixLeft", createStepName="BendStep", region="Bar-1.FixedNodes", u1=0.0, u2=0.0)

        # Prescribe vertical displacement on node 3
        model.DisplacementBC("PushTip", createStepName="BendStep", region="Bar-1.TipNode", u2=0.5)

        # Output requests
        step.FieldOutputRequest("F-Output-1", variables=["U", "S", "RF"], frequency=1, position="NODES")
        step.HistoryOutputRequest("H-Output-1", variables=["ALLIE", "ALLSE", "ALLWK"], frequency=1)

        # Solve with ODB
        solver, sys = model.create_solver2d("BendStep")
        odb = solver.solve_with_odb(step=step, sys=sys, verbose=False)

        assert isinstance(odb, Odb)
        assert "BendStep" in odb.steps
        odb_step = odb.steps["BendStep"]

        # Expect at least initial frame (t=0) + converged steps
        assert len(odb_step.frames) >= 2

        last_frame = odb_step.frames[-1]
        assert "U" in last_frame.fieldOutputs
        assert "S" in last_frame.fieldOutputs
        assert "RF" in last_frame.fieldOutputs

        # Verify tip displacement u2 reaches approximately 0.5
        u_data = last_frame.fieldOutputs["U"].data
        tip_idx = sys.global_nsets["Bar-1.TipNode"][0]
        assert np.isclose(u_data[tip_idx, 1], 0.5, atol=1e-3)

        # Verify energy tracking in History Output
        hr = odb_step.historyRegions["WholeModel"]
        assert "ALLIE" in hr.historyOutputs
        allie_vals = hr.historyOutputs["ALLIE"].values
        assert len(allie_vals) >= 2
        assert allie_vals[-1] > 0.0  # Positive strain energy produced by tip bending
