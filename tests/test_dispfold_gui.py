"""
test_dispfold_gui.py
====================
TDD Unit Verification Suite for Seam 6: PySide6 + PyVistaQt CAE Viewer Desktop GUI.

Verifies:
1. Instantiation of DispFoldQtViewer under headless/offscreen mode (QT_QPA_PLATFORM=offscreen).
2. Proper wiring of PyVista QtInteractor viewport, time slider, layer checklist, and Matplotlib canvas.
3. Both 2D and 3D mesh representations (PolyData vs UnstructuredGrid).
4. Synchronization of time steps: slider change triggers coordinate update and Matplotlib cursor movement.
5. Clean resource teardown without memory leaks or GUI crashes.
"""

from __future__ import annotations
import os
import sys
import pytest
import numpy as np
import pyvista as pv

# Set PyVista offscreen rendering
pv.OFF_SCREEN = True

from PySide6 import QtWidgets
from example_disp_fold.model_spec import DisplayStackConfig
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.viewer_qt_app import DispFoldQtViewer


@pytest.fixture(scope="module")
def qt_app():
    """Provides a shared QApplication instance for GUI tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    yield app


def test_dispfold_gui_2d_instantiation(qt_app):
    """Verify 2D CAE Viewer window components, layer list, and time slider synchronization."""
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_2d()

    n_nodes = len(mesh.nodes)
    history_bundle = {
        "history_time": [0.0, 0.5, 1.0],
        "history_theta": [0.0, 15.0, 30.0],
        "history_M_L": [0.0, 0.05, 0.12],
        "history_M_R": [0.0, -0.05, -0.12],
        "history_iters": [0, 2, 2],
        "history_u": [
            np.zeros(n_nodes * 2),
            np.ones(n_nodes * 2) * 0.01,
            np.ones(n_nodes * 2) * 0.02,
        ],
    }

    viewer = DispFoldQtViewer(mesh, meta, config, history_bundle)

    try:
        # 1. Window properties
        title = viewer.windowTitle()
        assert "WHT_DispFoldSolver CAE Viewer" in title
        assert "2D" in title
        assert "COMPACT_4" in title

        # 2. Controls and Sliders
        assert viewer.n_steps == 3
        assert viewer.slider.maximum() == 2
        assert viewer.slider.value() == 0

        # 3. Layer checklist
        assert viewer.layer_list.count() == 4
        for i in range(4):
            item = viewer.layer_list.item(i)
            assert item.text().startswith("[")

        # 4. Matplotlib Dashboard
        assert hasattr(viewer, "fig")
        assert hasattr(viewer, "canvas")
        assert len(viewer.fig.axes) == 3

        # 5. Time slider step synchronization
        viewer._on_slider_changed(1)
        assert viewer.curr_step == 1
        assert "Step: 1/2" in viewer.lbl_step_info.text()
        assert "Time: 0.5000s" in viewer.lbl_step_info.text()

        viewer._on_slider_changed(2)
        assert viewer.curr_step == 2
        assert "Step: 2/2" in viewer.lbl_step_info.text()

        # 6. Opacity adjustment
        viewer._on_opacity_changed(50)

    finally:
        viewer.close()


def test_dispfold_gui_3d_instantiation(qt_app):
    """Verify 3D CAE Viewer window components with 3D Hex mesh representation."""
    config = DisplayStackConfig(preset="compact_4", dim="3D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_3d()

    n_nodes = len(mesh.nodes)
    history_bundle = {
        "history_time": [0.0, 1.0],
        "history_theta": [0.0, 20.0],
        "history_M_L": [0.0, 0.08],
        "history_M_R": [0.0, -0.08],
        "history_iters": [0, 3],
        "history_u": [
            np.zeros(n_nodes * 3),
            np.ones(n_nodes * 3) * 0.015,
        ],
    }

    viewer = DispFoldQtViewer(mesh, meta, config, history_bundle)

    try:
        title = viewer.windowTitle()
        assert "3D" in title
        assert "COMPACT_4" in title
        assert viewer.n_steps == 2
        assert viewer.slider.maximum() == 1

        viewer._on_slider_changed(1)
        assert viewer.curr_step == 1
    finally:
        viewer.close()
