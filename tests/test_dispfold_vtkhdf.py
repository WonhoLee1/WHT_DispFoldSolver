"""
test_dispfold_vtkhdf.py
=======================
TDD Unit Verification Suite for Seam 4: Kitware VTKHDF v2.2 Database Exporter & Reader.

Verifies:
1. UnifiedVTKHDFWriter creates valid VTKHDF v2.2 HDF5 file hierarchy:
   - Version [2, 2], Type "UnstructuredGrid"
   - Static topology: Connectivity, Offsets, Types (VTK_QUAD=9, VTK_HEXAHEDRON=12)
   - PreProcessing/Sets hierarchy (NodeSets and ElementSets)
   - CellData: Part_ID, Layer_ID, Material_Type
   - PointData: Displacement, U
2. Streaming transient steps via add_transient_step:
   - Appends Displacement (U), Stress (S), Strain (LE), and Equivalent Plastic Strain (PEEQ)
3. Data integrity verification using UnifiedVTKHDFReader:
   - Lossless retrieval of transient field time series across all steps.
4. Support for both 2D plane strain and 3D continuum solid formulations.
"""

from __future__ import annotations
import os
import tempfile
import pytest
import numpy as np
import h5py

from example_disp_fold.model_spec import DisplayStackConfig
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.vtkhdf_unified_io import (
    UnifiedVTKHDFWriter,
    UnifiedVTKHDFReader,
    load_vtkhdf_summary,
    VTK_QUAD,
    VTK_HEXAHEDRON,
)


def test_vtkhdf_writer_structure_and_preprocessing_sets_2d():
    """Verify 2D VTKHDF v2.2 file generation, PreProcessing/Sets, CellData, and PointData."""
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_2d()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_display_2d.vtkhdf")
        writer = UnifiedVTKHDFWriter(out_path, mesh, meta, config)
        writer.close()

        assert os.path.exists(out_path)

        # Inspect raw HDF5 group hierarchy
        with h5py.File(out_path, "r") as f:
            assert "VTKHDF" in f
            grp = f["VTKHDF"]

            # 1. Version and Type attributes
            assert np.array_equal(grp.attrs["Version"], [2, 2])
            ascii_type = grp.attrs["Type"]
            type_str = ascii_type.decode("ascii") if isinstance(ascii_type, bytes) else str(ascii_type)
            assert type_str == "UnstructuredGrid"

            # 2. Static Topology
            assert "Connectivity" in grp
            assert "Offsets" in grp
            assert "Types" in grp
            types_arr = grp["Types"][:]
            assert len(types_arr) == len(mesh.elements)
            assert np.all(types_arr == VTK_QUAD)

            # 3. PreProcessing and PreProcessing/Sets
            assert "PreProcessing" in grp
            prep = grp["PreProcessing"]
            assert pytest.approx(float(prep.attrs["TotalThickness"]), rel=1e-5) == config.total_thickness
            assert str(prep.attrs["Preset"]) == "compact_4"
            assert str(prep.attrs["Dimension"]) == "2D"

            assert "Sets" in prep
            sets_grp = prep["Sets"]
            assert "NodeSets" in sets_grp
            assert "ElementSets" in sets_grp

            assert "disp_bottom_left_nids" in sets_grp["NodeSets"]
            assert "disp_bottom_right_nids" in sets_grp["NodeSets"]
            assert "PET_Bottom" in sets_grp["ElementSets"]
            assert "PSA_Mid" in sets_grp["ElementSets"]

            # 4. CellData (ParaView Native filtering)
            assert "CellData" in grp
            cd = grp["CellData"]
            assert "Part_ID" in cd
            assert "Layer_ID" in cd
            assert "Material_Type" in cd
            assert len(cd["Part_ID"]) == len(mesh.elements)
            assert len(cd["Layer_ID"]) == len(mesh.elements)
            assert len(cd["Material_Type"]) == len(mesh.elements)

            # 5. PointData
            assert "PointData" in grp
            pd = grp["PointData"]
            assert "Displacement" in pd
            assert "U" in pd


def test_vtkhdf_transient_streaming_and_reader_roundtrip_2d():
    """Verify transient field streaming (U, S, LE, PEEQ) and lossless reading via UnifiedVTKHDFReader."""
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_2d()

    n_nodes = len(mesh.nodes)
    n_cells = len(mesh.elements)
    coords = mesh.nodes_array()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_transient_2d.vtkhdf")
        writer = UnifiedVTKHDFWriter(out_path, mesh, meta, config)

        # Create 3 synthetic, deterministic time increment steps (no random numbers)
        times = [0.0, 0.5, 1.0]

        # Step 0: Initial rest state
        u0 = np.zeros((n_nodes, 2), dtype=np.float64)
        s0 = np.zeros(n_cells, dtype=np.float32)
        le0 = np.zeros(n_cells, dtype=np.float32)
        peeq0 = np.zeros(n_cells, dtype=np.float32)
        writer.add_transient_step(times[0], u0, s=s0, le=le0, peeq=peeq0)

        # Step 1: Intermediate elastic deformation
        u1 = np.zeros((n_nodes, 2), dtype=np.float64)
        u1[:, 0] = 0.05 * coords[:, 0] # 5% stretch along X
        u1[:, 1] = -0.02 * coords[:, 1]
        s1 = np.linspace(10.0, 150.0, n_cells, dtype=np.float32)
        le1 = np.linspace(0.001, 0.050, n_cells, dtype=np.float32)
        peeq1 = np.linspace(0.0, 0.010, n_cells, dtype=np.float32)
        writer.add_transient_step(times[1], u1, s=s1, le=le1, peeq=peeq1)

        # Step 2: Final large deformation state
        u2 = np.zeros((n_nodes, 2), dtype=np.float64)
        u2[:, 0] = 0.12 * coords[:, 0]
        u2[:, 1] = -0.05 * coords[:, 1]
        s2 = np.linspace(25.0, 320.0, n_cells, dtype=np.float32)
        le2 = np.linspace(0.005, 0.120, n_cells, dtype=np.float32)
        peeq2 = np.linspace(0.0, 0.035, n_cells, dtype=np.float32)
        writer.add_transient_step(times[2], u2, s=s2, le=le2, peeq=peeq2)

        writer.close()

        # Read back and verify via UnifiedVTKHDFReader
        with UnifiedVTKHDFReader(out_path) as reader:
            assert reader.n_steps == 3
            assert np.allclose(reader.times, times)
            assert reader.n_nodes == n_nodes
            assert reader.n_cells == n_cells

            # PreProcessing metadata & sets check
            meta_prep = reader.get_preprocessing_metadata()
            assert pytest.approx(meta_prep["TotalThickness"], rel=1e-5) == config.total_thickness
            assert meta_prep["Preset"] == "compact_4"

            sets_dict = reader.get_preprocessing_sets()
            assert "NodeSets" in sets_dict
            assert "ElementSets" in sets_dict
            assert "disp_bottom_left_nids" in sets_dict["NodeSets"]
            assert set(sets_dict["NodeSets"]["disp_bottom_left_nids"]) == set(meta["disp_bottom_left_nids"])

            # Check transient steps
            step0 = reader.read_transient_step(0)
            assert pytest.approx(step0["time"]) == 0.0
            assert np.allclose(step0["U"], 0.0)
            assert np.allclose(step0["S"], s0)
            assert np.allclose(step0["LE"], le0)
            assert np.allclose(step0["PEEQ"], peeq0)

            step1 = reader.read_transient_step(1)
            assert pytest.approx(step1["time"]) == 0.5
            assert np.allclose(step1["U"][:, :2], u1, atol=1e-6)
            assert np.allclose(step1["S"], s1, atol=1e-6)
            assert np.allclose(step1["LE"], le1, atol=1e-6)
            assert np.allclose(step1["PEEQ"], peeq1, atol=1e-6)

            step2 = reader.read_transient_step(2)
            assert pytest.approx(step2["time"]) == 1.0
            assert np.allclose(step2["U"][:, :2], u2, atol=1e-6)
            assert np.allclose(step2["S"], s2, atol=1e-6)
            assert np.allclose(step2["LE"], le2, atol=1e-6)
            assert np.allclose(step2["PEEQ"], peeq2, atol=1e-6)

            # Direct field retrieval check
            s_all = reader.get_cell_data("S")
            assert s_all.shape == (3 * n_cells,)
            u_all = reader.get_point_data("Displacement")
            assert u_all.shape == (3 * n_nodes, 3)

        # Summary check
        summary = load_vtkhdf_summary(out_path)
        assert summary["num_steps"] == 3
        assert summary["num_cells"] == n_cells
        assert summary["preset"] == "compact_4"


def test_vtkhdf_3d_hex_streaming_and_reader():
    """Verify 3D Hexahedral VTKHDF v2.2 streaming with tensor stress and strain fields."""
    config = DisplayStackConfig(preset="compact_4", dim="3D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_3d()

    n_nodes = len(mesh.nodes)
    n_cells = len(mesh.elements)
    coords = mesh.nodes_array()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_3d_hex.vtkhdf")
        writer = UnifiedVTKHDFWriter(out_path, mesh, meta, config)

        # 3D displacement vector field (n_nodes, 3)
        u_3d = np.zeros((n_nodes, 3), dtype=np.float64)
        u_3d[:, 0] = 0.02 * coords[:, 0]
        u_3d[:, 1] = 0.01 * coords[:, 1]
        u_3d[:, 2] = -0.01 * coords[:, 2]

        # Multi-component stress tensor (n_cells, 6): [S11, S22, S33, S12, S13, S23]
        s_3d = np.zeros((n_cells, 6), dtype=np.float32)
        s_3d[:, 0] = 120.0
        s_3d[:, 1] = 45.0
        s_3d[:, 2] = -30.0

        le_3d = np.zeros((n_cells, 6), dtype=np.float32)
        le_3d[:, 0] = 0.03

        peeq_3d = np.full(n_cells, 0.005, dtype=np.float32)

        writer.add_transient_step(0.25, u_3d, s=s_3d, le=le_3d, peeq=peeq_3d)
        writer.close()

        with UnifiedVTKHDFReader(out_path) as reader:
            assert reader.n_steps == 1
            assert reader.n_nodes == n_nodes
            assert reader.n_cells == n_cells
            assert np.all(reader.types == VTK_HEXAHEDRON)

            step0 = reader.read_transient_step(0)
            assert pytest.approx(step0["time"]) == 0.25
            assert np.allclose(step0["U"], u_3d, atol=1e-6)
            assert step0["S"].shape == (n_cells, 6)
            assert np.allclose(step0["S"][:, 0], 120.0, atol=1e-5)
            assert np.allclose(step0["LE"][:, 0], 0.03, atol=1e-5)
            assert np.allclose(step0["PEEQ"], 0.005, atol=1e-5)
