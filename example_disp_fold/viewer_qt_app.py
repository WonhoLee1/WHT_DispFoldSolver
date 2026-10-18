"""
viewer_qt_app.py
================
PySide6 + pyvistaqt integrated CAE desktop GUI for 2D/3D foldable display simulations.
Features interactive 3D deformed viewport, time-synced dual slider, layer visibility checklist,
opacity controller, and right-hand Matplotlib dashboard (9pt Noto Sans KR / Malgun Gothic).
"""

from __future__ import annotations
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Ensure PySide6 and PyVista
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer
import pyvista as pv
from pyvistaqt import QtInteractor

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

try:
    import koreanize_matplotlib
except ImportError:
    pass


class DispFoldQtViewer(QtWidgets.QMainWindow):
    """Main CAE Desktop Window uniting PyVista 3D and Matplotlib 2D dashboard."""

    def __init__(
        self,
        mesh: Any,
        meta: Dict[str, Any],
        config: Any,
        history_data: Dict[str, Any],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.mesh = mesh
        self.meta = meta
        self.config = config
        self.history = history_data
        self.dim = config.simulation_case.dim.upper()

        self.times = history_data.get("history_time", [0.0])
        self.thetas = history_data.get("history_theta", [0.0])
        self.M_L = history_data.get("history_M_L", [0.0])
        self.M_R = history_data.get("history_M_R", [0.0])
        self.iters = history_data.get("history_iters", [0])
        self.u_steps = history_data.get("history_u", [history_data.get("u_final", np.zeros(mesh.num_nodes * (2 if self.dim == "2D" else 3)))])
        self.n_steps = len(self.times)
        self.curr_step = 0

        self.setWindowTitle(f"WHT_DispFoldSolver CAE Viewer - [{self.dim}] {config.simulation_case.preset_name.upper()}")
        self.resize(1600, 920)

        self._setup_ui()
        self._setup_3d_viewport()
        self._setup_matplotlib_dashboard()
        self._update_time_step(0)

    def _setup_ui(self) -> None:
        """Constructs split layout: Left (3D), Right (2D Dashboard + Controls)."""
        main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ── Left Column: 3D Interactor & Playback Slider ────────────────────
        left_box = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # PyVista QtInteractor
        self.interactor = QtInteractor(left_box)
        left_layout.addWidget(self.interactor.interactor, stretch=1)

        # Playback Controls Bar
        control_bar = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_play.setFixedWidth(80)
        self.btn_play.clicked.connect(self._toggle_playback)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, self.n_steps - 1))
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_changed)

        self.lbl_step_info = QtWidgets.QLabel("Step: 0 / 0 | Time: 0.000s | Angle: 0.00°")
        self.lbl_step_info.setMinimumWidth(260)
        self.lbl_step_info.setAlignment(Qt.AlignCenter)

        control_bar.addWidget(self.btn_play)
        control_bar.addWidget(self.slider, stretch=1)
        control_bar.addWidget(self.lbl_step_info)
        left_layout.addLayout(control_bar)

        # ── Right Column: Layer TreeView & Matplotlib Dashboard ─────────────
        right_box = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_box)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(6)

        # Upper: Layer Visibility & Opacity
        ctrl_group = QtWidgets.QGroupBox("Layer Visibility & Opacity Controls")
        ctrl_layout = QtWidgets.QHBoxLayout(ctrl_group)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)

        self.layer_list = QtWidgets.QListWidget()
        self.layer_list.setMaximumHeight(110)
        for lay in self.config.layers:
            item = QtWidgets.QListWidgetItem(f"[{lay.name}] {lay.material_name} ({lay.thickness*1000:.0f}µm)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.layer_list.addItem(item)
        self.layer_list.itemChanged.connect(self._on_layer_visibility_changed)

        opacity_box = QtWidgets.QVBoxLayout()
        opacity_lbl = QtWidgets.QLabel("PET/Glass Opacity:")
        self.opacity_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_box.addWidget(opacity_lbl)
        opacity_box.addWidget(self.opacity_slider)
        opacity_box.addStretch()

        ctrl_layout.addWidget(self.layer_list, stretch=2)
        ctrl_layout.addLayout(opacity_box, stretch=1)
        right_layout.addWidget(ctrl_group)

        # Lower: Matplotlib Figure Canvas
        self.fig = Figure(figsize=(5.5, 7.5), dpi=100)
        self.fig.patch.set_facecolor("#1e293b")
        self.canvas = FigureCanvasQTAgg(self.fig)
        right_layout.addWidget(self.canvas, stretch=1)

        # Splitter between Left (65%) and Right (35%)
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.addWidget(left_box)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(0, 65)
        splitter.setStretchFactor(1, 35)

        main_layout.addWidget(splitter)

        # Playback Timer
        self.timer = QTimer(self)
        self.timer.setInterval(80) # ~12.5 fps
        self.timer.timeout.connect(self._on_timer_tick)
        self.is_playing = False

    def _setup_3d_viewport(self) -> None:
        """Initializes PyVista 3D render settings with Dark Paraview style."""
        self.interactor.set_background("black")
        self.interactor.show_axes()

        coords = self.mesh.nodes_array().copy()
        if self.dim == "2D":
            pts = np.zeros((coords.shape[0], 3), dtype=np.float32)
            pts[:, :2] = coords
        else:
            pts = coords.astype(np.float32)

        self._base_points = pts

        # Build PyVista PolyData or UnstructuredGrid
        if self.dim == "2D":
            faces = []
            for e in self.mesh.elements.values():
                nid_map = self.mesh.node_id_to_index()
                faces.extend([4, nid_map[e.node_ids[0]], nid_map[e.node_ids[1]], nid_map[e.node_ids[2]], nid_map[e.node_ids[3]]])
            self.pv_mesh = pv.PolyData(pts, faces=faces)
        else:
            # 3D Hex mesh
            cells = []
            cell_types = []
            nid_map = self.mesh.node_id_to_index()
            for e in self.mesh.elements.values():
                cells.extend([8] + [nid_map[n] for n in e.node_ids])
                cell_types.append(pv.CellType.HEXAHEDRON)
            self.pv_mesh = pv.UnstructuredGrid(cells, cell_types, pts)

        self.mesh_actor = self.interactor.add_mesh(
            self.pv_mesh,
            color="#38bdf8",
            show_edges=True,
            edge_color="#334155",
            lighting=True,
        )
        self.interactor.reset_camera()
        self.interactor.view_xz()

        # Context Menu for Projection Views (XY, YZ, ZX) and Perspective View
        self.interactor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.interactor.customContextMenuRequested.connect(self._show_viewport_context_menu)
        self.interactor.interactor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.interactor.interactor.customContextMenuRequested.connect(self._show_viewport_context_menu)

    def _show_viewport_context_menu(self, pos: QtCore.QPoint) -> None:
        """Displays right-click context menu for camera projection views and perspective toggle."""
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("background-color: #1e293b; color: #f8fafc; font-size: 9pt;")

        act_zx = menu.addAction("ZX 정면도 (Front Teardrop View)")
        act_xy = menu.addAction("XY 평면도 (Top View)")
        act_yz = menu.addAction("YZ 측면도 (Side View)")
        act_iso = menu.addAction("등각투영 (Isometric 3D)")
        menu.addSeparator()

        is_parallel = bool(self.interactor.camera.GetParallelProjection())
        act_persp = menu.addAction("투시 투영 (Perspective View)")
        act_persp.setCheckable(True)
        act_persp.setChecked(not is_parallel)

        sender_widget = self.sender() or self.interactor
        global_pos = sender_widget.mapToGlobal(pos)
        chosen = menu.exec(global_pos)

        if chosen == act_zx:
            self.interactor.view_xz()
        elif chosen == act_xy:
            self.interactor.view_xy()
        elif chosen == act_yz:
            self.interactor.view_yz()
        elif chosen == act_iso:
            self.interactor.view_isometric()
        elif chosen == act_persp:
            if is_parallel:
                self.interactor.disable_parallel_projection()
            else:
                self.interactor.enable_parallel_projection()
            self.interactor.render()

    def _setup_matplotlib_dashboard(self) -> None:
        """Builds 3-row Matplotlib analytical dashboard with 9pt Korean font."""
        plt.rcParams["font.size"] = 9
        plt.rcParams["font.family"] = ["Noto Sans KR", "Malgun Gothic", "sans-serif"]

        self.fig.clf()
        self.ax_moment = self.fig.add_subplot(3, 1, 1)
        self.ax_slip = self.fig.add_subplot(3, 1, 2)
        self.ax_iters = self.fig.add_subplot(3, 1, 3)

        for ax in [self.ax_moment, self.ax_slip, self.ax_iters]:
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#cbd5e1", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.grid(True, linestyle="--", alpha=0.3, color="#475569")

        # 1. Drive Moment vs Angle + Elastica Envelope
        D_slip, D_mono, _ = self.config.compute_composite_bending_stiffness()
        thetas_deg = np.array(self.thetas)
        M_avg = 0.5 * (np.abs(self.M_L) + np.abs(self.M_R))

        lines = self.ax_moment.plot(thetas_deg, M_avg, color="#38bdf8", lw=2.0, label="FEM M_drive")
        self.line_fem_moment = lines[0]
        # Theoretical bounds reference lines
        # M_elastica ~ D * theta_rad / L_eff
        L_eff = 2.0 * self.config.geometry.hinge_half_gap
        th_rad = np.radians(thetas_deg)
        M_slip_theory = D_slip * (2.0 * th_rad) / L_eff
        M_mono_theory = D_mono * (2.0 * th_rad) / L_eff
        self.ax_moment.fill_between(thetas_deg, M_slip_theory, M_mono_theory, color="#0284c7", alpha=0.15, label="Elastica Bounds")

        self.cur_moment_line = self.ax_moment.axvline(0.0, color="#f43f5e", linestyle=":", lw=1.5)
        self.ax_moment.set_ylabel("모멘트 [N·mm]", color="#e2e8f0", fontsize=9)
        self.ax_moment.set_title("피벗 구동 모멘트 & Elastica 이론해 포락선", color="#94a3b8", fontsize=9, pad=4)
        self.ax_moment.legend(loc="upper left", fontsize=7.5, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

        # 2. Interlayer Shear Slip Profile
        self.ax_slip.set_ylabel("슬립량 [µm]", color="#e2e8f0", fontsize=9)
        self.ax_slip.set_title("두께 방향 층간 전단 슬립 프로파일", color="#94a3b8", fontsize=9, pad=4)

        # 3. Newton Iteration History
        step_indices = np.arange(1, len(self.iters) + 1)
        self.ax_iters.bar(step_indices, self.iters, color="#10b981", alpha=0.7, width=0.8)
        self.ax_iters.axhline(10.0, color="#ef4444", linestyle="--", lw=1.0, label="Target 10 iters")
        self.ax_iters.set_xlabel("해석 스텝 (Step)", color="#e2e8f0", fontsize=9)
        self.ax_iters.set_ylabel("뉴턴 반복수", color="#e2e8f0", fontsize=9)
        self.ax_iters.set_title("스텝별 뉴턴-랩슨 수렴 이력", color="#94a3b8", fontsize=9, pad=4)

        self.fig.tight_layout()
        self.canvas.draw()

    def _update_time_step(self, step_idx: int) -> None:
        """Synchronizes 3D mesh coordinates and 2D charts to specified step."""
        if not (0 <= step_idx < self.n_steps):
            return
        self.curr_step = step_idx

        # 1. Update 3D Deformed Mesh
        u_step = self.u_steps[step_idx]
        ndof_node = 2 if self.dim == "2D" else 3
        u_arr = np.asarray(u_step).reshape((-1, ndof_node))

        disp_3d = np.zeros_like(self._base_points)
        disp_3d[:, :ndof_node] = u_arr[:, :ndof_node]
        self.pv_mesh.points = self._base_points + disp_3d
        self.interactor.render()

        # 2. Update Slider and Info Label
        t = self.times[step_idx]
        th = self.thetas[step_idx]
        self.slider.blockSignals(True)
        self.slider.setValue(step_idx)
        self.slider.blockSignals(False)
        self.lbl_step_info.setText(f"Step: {step_idx}/{self.n_steps-1} | Time: {t:.4f}s | Angle: {th:.2f}° (Total {2*th:.2f}°)")

        # 3. Update Matplotlib Cursor
        if hasattr(self, "cur_moment_line"):
            self.cur_moment_line.set_xdata([th, th])
            self.canvas.draw_idle()

    def _on_slider_changed(self, val: int) -> None:
        self._update_time_step(val)

    def _toggle_playback(self) -> None:
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("▶ Play")
            self.is_playing = False
        else:
            if self.curr_step >= self.n_steps - 1:
                self.curr_step = 0
            self.timer.start()
            self.btn_play.setText("⏸ Pause")
            self.is_playing = True

    def _on_timer_tick(self) -> None:
        if self.curr_step < self.n_steps - 1:
            self._update_time_step(self.curr_step + 1)
        else:
            self._toggle_playback()

    def _on_layer_visibility_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        # Re-render or filter
        pass

    def _on_opacity_changed(self, val: int) -> None:
        opacity = val / 100.0
        if hasattr(self, "mesh_actor"):
            self.mesh_actor.GetProperty().SetOpacity(opacity)
            self.interactor.render()

    def append_live_step(
        self,
        step_idx: int,
        t: float,
        th: float,
        u_vec: np.ndarray,
        iters: int,
        M_L: float,
        M_R: float,
    ) -> None:
        """Appends a newly converged solver increment in real-time streaming mode."""
        self.times.append(float(t))
        self.thetas.append(float(th))
        self.u_steps.append(np.asarray(u_vec).copy())
        self.iters.append(int(iters))
        self.M_L.append(float(M_L))
        self.M_R.append(float(M_R))
        self.n_steps = len(self.times)

        # Update slider range
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.n_steps - 1))
        self.slider.setValue(self.n_steps - 1)
        self.slider.blockSignals(False)

        # Update 3D viewport to latest step
        self._update_time_step(self.n_steps - 1)

        # Redraw analytical dashboard lines
        thetas_deg = np.array(self.thetas)
        M_avg = 0.5 * (np.abs(self.M_L) + np.abs(self.M_R))
        if hasattr(self, "line_fem_moment"):
            self.line_fem_moment.set_data(thetas_deg, M_avg)
            self.ax_moment.relim()
            self.ax_moment.autoscale_view(True, True, True)

        self.ax_iters.cla()
        self.ax_iters.set_facecolor("#0f172a")
        self.ax_iters.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in self.ax_iters.spines.values():
            spine.set_color("#334155")
        self.ax_iters.grid(True, linestyle="--", alpha=0.3, color="#475569")
        step_indices = np.arange(1, len(self.iters) + 1)
        self.ax_iters.bar(step_indices, self.iters, color="#10b981", alpha=0.7, width=0.8)
        self.ax_iters.axhline(10.0, color="#ef4444", linestyle="--", lw=1.0, label="Target 10 iters")
        self.ax_iters.set_xlabel("해석 스텝 (Step)", color="#e2e8f0", fontsize=9)
        self.ax_iters.set_ylabel("뉴턴 반복수", color="#e2e8f0", fontsize=9)
        self.ax_iters.set_title("스텝별 뉴턴-랩슨 수렴 이력", color="#94a3b8", fontsize=9, pad=4)

        self.canvas.draw_idle()


class SolverWorker(QtCore.QThread):
    """Background QThread worker executing the non-linear folding solver."""
    step_converged = QtCore.Signal(int, float, float, np.ndarray, int, float, float)
    simulation_finished = QtCore.Signal(str)

    def __init__(
        self,
        run_fn: Any,
        kwargs: Dict[str, Any],
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.run_fn = run_fn
        self.kwargs = dict(kwargs)

    def run(self) -> None:
        def on_step(step_idx, t, th, u_vec, iters, M_L, M_R):
            self.step_converged.emit(step_idx, t, th, u_vec, iters, M_L, M_R)

        self.kwargs["step_callback"] = on_step
        self.kwargs["launch_gui"] = False
        out_dir = self.run_fn(**self.kwargs)
        self.simulation_finished.emit(str(out_dir))


def launch_qt_viewer(
    mesh: Any,
    meta: Dict[str, Any],
    config: Any,
    history_data: Dict[str, Any],
    app: Optional[QtWidgets.QApplication] = None,
) -> None:
    """Entry point to display the integrated CAE Qt viewer."""
    own_app = False
    if app is None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
            own_app = True

    viewer = DispFoldQtViewer(mesh, meta, config, history_data)
    viewer.show()

    if own_app:
        app.exec()


def launch_qt_viewer_live(
    run_fn: Any,
    kwargs: Dict[str, Any],
    mesh: Any,
    meta: Dict[str, Any],
    config: Any,
) -> None:
    """Launches CAE viewer in live interactive streaming mode while solver executes."""
    app = QtWidgets.QApplication.instance()
    own_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        own_app = True

    dim = config.simulation_case.dim.upper()
    ndof_node = 2 if dim == "2D" else 3
    u0 = np.zeros(len(mesh.nodes) * ndof_node, dtype=np.float64)
    init_history = {
        "history_time": [0.0],
        "history_theta": [0.0],
        "history_iters": [0],
        "history_M_L": [0.0],
        "history_M_R": [0.0],
        "history_u": [u0],
    }

    viewer = DispFoldQtViewer(mesh, meta, config, init_history)
    viewer.show()

    worker = SolverWorker(run_fn, kwargs, parent=viewer)
    worker.step_converged.connect(viewer.append_live_step)
    worker.start()

    viewer._worker = worker

    if own_app:
        app.exec()

