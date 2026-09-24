"""
test_vtkhdf_exporter.py
========================
Verification suite for VTKHDF (.vtkhdf) field data exporter.
"""

import os
import tempfile
import numpy as np
import pytest
import h5py

from dispsolver.export.vtkhdf_exporter import export_vtkhdf, ABAQUS_TO_VTK_TYPE, VTK_QUAD, VTK_HEXAHEDRON


def test_export_vtkhdf_2d_mesh():
    """Verify VTKHDF file generation for 2D CPE4 quad mesh."""
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float64)

    elements = np.array([[1, 2, 3, 4]], dtype=np.int64)
    u = np.array([0.0, 0.0, 0.1, 0.0, 0.1, 0.2, 0.0, 0.2], dtype=np.float64)
    stresses = np.array([[100.0, 50.0, 10.0, 0.0]], dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_2d.vtkhdf")
        res_path = export_vtkhdf(
            out_path, nodes, elements, u, element_type="CPE4I", stresses=stresses
        )

        assert os.path.exists(res_path)
        assert res_path.endswith(".vtkhdf")

        # Read back with h5py and verify datasets
        with h5py.File(res_path, "r") as f:
            assert "VTKHDF" in f
            grp = f["VTKHDF"]
            assert np.array_equal(grp["Version"][:], [1, 0])
            assert grp.attrs["Type"] == "UnstructuredGrid"
            assert bytes(grp["Type"][:]).decode("ascii") == "UnstructuredGrid"
            assert grp["NumberOfPoints"][0] == 4
            assert grp["NumberOfCells"][0] == 1

            pts = grp["Points"][:]
            assert pts.shape == (4, 3)
            assert np.allclose(pts[:, 2], 0.0)

            disp = grp["PointData/Displacement"][:]
            assert disp.shape == (4, 3)
            assert np.allclose(disp[1, :2], [0.1, 0.0])

            vm = grp["CellData/VonMises"][:]
            assert len(vm) == 1
            assert vm[0] > 0.0


def test_export_vtkhdf_3d_mesh():
    """Verify VTKHDF file generation for 3D C3D8 solid mesh."""
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)

    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    u = np.zeros(24, dtype=np.float64)
    u[3] = 0.05  # x-displacement on node 1

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_3d.vtkhdf")
        res_path = export_vtkhdf(
            out_path, nodes, elements, u, element_type="C3D8"
        )

        assert os.path.exists(res_path)
        with h5py.File(res_path, "r") as f:
            grp = f["VTKHDF"]
            assert grp["Types"][0] == VTK_HEXAHEDRON
            disp = grp["PointData/Displacement"][:]
            assert disp[1, 0] == 0.05
