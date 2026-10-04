"""odb.py
======
Abaqus-compatible Odb, OdbStep, and OdbFrame containers with modern Python extensions:
- NumPy vectorized batch arrays
- HDF5 binary persistent storage (.h5odb)
- VTKHDF export (.vtkhdf)
- One-click PyVista 3D/2D interactive viewer
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Sequence
import os
import numpy as np
import h5py as h5

from dispsolver.output.field_data import FieldOutput, FieldPosition, FieldType
from dispsolver.output.history_data import HistoryRegion, HistoryOutput
from dispsolver.output.vtkhdf_writer import TransientVTKHDFWriter


class OdbFrame:
    """Abaqus-compatible Frame container for a single converged increment."""

    def __init__(
        self,
        frame_id: int,
        frame_value: float,
        description: str = "",
        field_outputs: Optional[Dict[str, FieldOutput]] = None
    ):
        self.frameId: int = int(frame_id)
        self.frameValue: float = float(frame_value)
        self.description: str = str(description)
        self.fieldOutputs: Dict[str, FieldOutput] = dict(field_outputs) if field_outputs is not None else {}

    def add_field_output(self, field_output: FieldOutput) -> FieldOutput:
        """Register a FieldOutput into this Frame."""
        self.fieldOutputs[field_output.name] = field_output
        return field_output

    def __getitem__(self, field_name: str) -> FieldOutput:
        return self.fieldOutputs[field_name]


class OdbStep:
    """Abaqus-compatible Step container holding frames and history regions."""

    def __init__(self, name: str, procedure: str = "STATIC", time_period: float = 1.0):
        self.name: str = str(name)
        self.procedure: str = str(procedure)
        self.timePeriod: float = float(time_period)
        self.frames: List[OdbFrame] = []
        self._history_regions: Dict[str, HistoryRegion] = {}

    @property
    def historyRegions(self) -> Dict[str, HistoryRegion]:
        """Abaqus-compatible historyRegions dictionary."""
        return self._history_regions

    @property
    def history_regions(self) -> Dict[str, HistoryRegion]:
        return self._history_regions

    def get_or_create_history_region(self, name: str, description: str = "") -> HistoryRegion:
        """Get existing or create new HistoryRegion."""
        if name not in self._history_regions:
            self._history_regions[name] = HistoryRegion(name=name, description=description)
        return self._history_regions[name]

    def add_frame(self, frame: OdbFrame) -> OdbFrame:
        """Add a frame to this step."""
        self.frames.append(frame)
        return frame


class Odb:
    """Top-level Output Database (ODB) container."""

    def __init__(self, name: str = "Job-1", mesh: Optional[Any] = None):
        self.name: str = str(name)
        self.mesh: Optional[Any] = mesh
        self.steps: Dict[str, OdbStep] = {}

    def get_or_create_step(self, step_name: str, procedure: str = "STATIC", time_period: float = 1.0) -> OdbStep:
        """Get existing or create new OdbStep."""
        if step_name not in self.steps:
            self.steps[step_name] = OdbStep(name=step_name, procedure=procedure, time_period=time_period)
        return self.steps[step_name]

    # ─── EXPORT & PERSISTENCE ──────────────────────────────────────────

    def export_vtkhdf(self, filepath: str, step_name: Optional[str] = None) -> None:
        """Export all frames of a step to a standard .vtkhdf file for ParaView/PyVista."""
        if self.mesh is None:
            raise ValueError("Odb requires a Mesh reference to export VTKHDF.")

        st_name = step_name if step_name is not None else next(iter(self.steps.keys()))
        odb_step = self.steps[st_name]

        writer = TransientVTKHDFWriter(filepath=filepath, mesh=self.mesh)
        for frame in odb_step.frames:
            t = frame.frameValue
            u_field = frame.fieldOutputs.get("U")
            if u_field is None:
                continue

            point_data: Dict[str, np.ndarray] = {}
            cell_data: Dict[str, np.ndarray] = {}

            # Populate point data
            for fname, fo in frame.fieldOutputs.items():
                if fo.position == FieldPosition.NODES:
                    if fo.name == "S":
                        point_data["Mises"] = fo.get_mises()
                    elif fo.name != "U":
                        point_data[fname] = fo.data

            # Populate cell data
            for fname, fo in frame.fieldOutputs.items():
                if fo.position in [FieldPosition.CENTROID, FieldPosition.INTEGRATION_POINTS]:
                    if fo.name == "S":
                        cell_data["Mises"] = fo.get_mises()
                        cell_data["Pressure"] = fo.get_pressure()
                    elif fo.name == "PEEQ":
                        cell_data["PEEQ"] = fo.data
                    else:
                        cell_data[fname] = fo.data

            writer.add_step(
                time_val=t,
                u=u_field.data,
                point_data=point_data,
                cell_data=cell_data
            )

        writer.close()
        print(f"[Odb] Successfully exported {len(odb_step.frames)} frames to VTKHDF: '{filepath}'")

    def save(self, filepath: str) -> None:
        """Save complete ODB data to HDF5 binary file (.h5odb)."""
        with h5.File(filepath, "w") as f:
            f.attrs["odb_name"] = self.name
            
            # Save Mesh geometry if available
            if self.mesh is not None:
                g_mesh = f.create_group("Mesh")
                g_mesh.create_dataset("nodes", data=self.mesh.nodes_array())
                # connectivity
                conn_list = []
                for el in self.mesh.elements.values():
                    conn_list.append(el.node_ids)
                g_mesh.create_dataset("connectivity", data=np.array(conn_list, dtype=np.int32))

            # Save Steps
            g_steps = f.create_group("Steps")
            for sname, step in self.steps.items():
                gst = g_steps.create_group(sname)
                gst.attrs["procedure"] = step.procedure
                gst.attrs["time_period"] = step.timePeriod

                # Frames
                g_frames = gst.create_group("Frames")
                for f_idx, frame in enumerate(step.frames):
                    gfr = g_frames.create_group(f"Frame_{f_idx:05d}")
                    gfr.attrs["frameId"] = frame.frameId
                    gfr.attrs["frameValue"] = frame.frameValue
                    gfr.attrs["description"] = frame.description

                    g_fields = gfr.create_group("FieldOutputs")
                    for fo_name, fo in frame.fieldOutputs.items():
                        gfo = g_fields.create_group(fo_name)
                        gfo.attrs["field_type"] = fo.type.value
                        gfo.attrs["position"] = fo.position.value
                        gfo.attrs["description"] = fo.description
                        gfo.create_dataset("data", data=fo.data)
                        gfo.create_dataset("labels", data=fo.labels)

                # History Regions
                g_hist = gst.create_group("HistoryRegions")
                for hr_name, hr in step.history_regions.items():
                    ghr = g_hist.create_group(hr_name)
                    for ho_name, ho in hr.historyOutputs.items():
                        gho = ghr.create_group(ho_name)
                        gho.create_dataset("times", data=ho.times)
                        gho.create_dataset("values", data=ho.values)

        print(f"[Odb] Saved ODB to binary file: '{filepath}'")

    @classmethod
    def open_odb(cls, filepath: str) -> Odb:
        """Load ODB from HDF5 binary file (.h5odb)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ODB file not found: {filepath}")

        with h5.File(filepath, "r") as f:
            odb_name = f.attrs.get("odb_name", "Loaded-Job")
            odb = cls(name=str(odb_name))

            g_steps = f.get("Steps")
            if g_steps:
                for sname in g_steps.keys():
                    gst = g_steps[sname]
                    proc = gst.attrs.get("procedure", "STATIC")
                    t_per = gst.attrs.get("time_period", 1.0)
                    step = odb.get_or_create_step(sname, procedure=str(proc), time_period=float(t_per))

                    # Frames
                    g_frames = gst.get("Frames")
                    if g_frames:
                        for fr_key in sorted(g_frames.keys()):
                            gfr = g_frames[fr_key]
                            fid = gfr.attrs.get("frameId", 0)
                            fval = gfr.attrs.get("frameValue", 0.0)
                            fdesc = gfr.attrs.get("description", "")
                            frame = OdbFrame(frame_id=int(fid), frame_value=float(fval), description=str(fdesc))

                            g_fields = gfr.get("FieldOutputs")
                            if g_fields:
                                for fo_name in g_fields.keys():
                                    gfo = g_fields[fo_name]
                                    ftype = gfo.attrs.get("field_type", "VECTOR")
                                    pos = gfo.attrs.get("position", "NODES")
                                    d_data = np.array(gfo["data"])
                                    d_labels = np.array(gfo["labels"])
                                    fo = FieldOutput(
                                        name=fo_name,
                                        field_type=ftype,
                                        position=pos,
                                        data=d_data,
                                        labels=d_labels
                                    )
                                    frame.add_field_output(fo)

                            step.add_frame(frame)

                    # History
                    g_hist = gst.get("HistoryRegions")
                    if g_hist:
                        for hr_name in g_hist.keys():
                            ghr = g_hist[hr_name]
                            hr = step.get_or_create_history_region(hr_name)
                            for ho_name in ghr.keys():
                                gho = ghr[ho_name]
                                times = np.array(gho["times"])
                                vals = np.array(gho["values"])
                                ho = HistoryOutput(name=ho_name, times=times, data=vals)
                                hr.historyOutputs[ho_name] = ho

        return odb

    def view(
        self,
        step_name: Optional[str] = None,
        frame_idx: int = -1,
        field_name: str = "S",
        comp: str = "mises",
        **kwargs
    ) -> None:
        """Launch interactive PyVista 3D/2D viewer for the selected frame."""
        import pyvista as pv

        if self.mesh is None:
            raise ValueError("Interactive viewing requires a Mesh reference.")

        st_name = step_name if step_name is not None else next(iter(self.steps.keys()))
        step = self.steps[st_name]
        if not step.frames:
            print("[Warning] No frames available in step.")
            return

        frame = step.frames[frame_idx]
        u_field = frame.fieldOutputs.get("U")
        if u_field is None:
            print("[Warning] No displacement field found.")
            return

        # Prepare PyVista UnstructuredGrid
        dim = 2 if u_field.data.shape[1] == 2 else 3
        x0 = self.mesh.nodes_array()
        u = u_field.data

        pts = np.zeros((len(x0), 3), dtype=np.float32)
        pts[:, :dim] = (x0[:, :dim] + u[:, :dim]).astype(np.float32)

        # Cells
        cells_list = []
        cell_types = []
        nid_to_idx = self.mesh.node_id_to_index()
        for el in self.mesh.elements.values():
            n_el = len(el.node_ids)
            cells_list.append(n_el)
            for nid in el.node_ids:
                cells_list.append(nid_to_idx[nid])
            cell_types.append(pv.CellType.QUAD if n_el == 4 else pv.CellType.HEXAHEDRON)

        grid = pv.UnstructuredGrid(cells_list, cell_types, pts)

        # Scalar field
        target_field = frame.fieldOutputs.get(field_name)
        if target_field is not None:
            if target_field.name == "S":
                vals = target_field.get_mises() if comp.lower() == "mises" else target_field.get_component(comp)
            elif target_field.type == FieldType.VECTOR:
                vals = target_field.get_magnitude()
            else:
                vals = target_field.data

            if target_field.position == FieldPosition.NODES:
                grid.point_data[f"{field_name}_{comp}"] = vals
            else:
                grid.cell_data[f"{field_name}_{comp}"] = vals

        # Display with PyVista Standard theme
        plotter = pv.Plotter()
        plotter.set_background("black")
        plotter.add_mesh(grid, scalars=f"{field_name}_{comp}" if target_field else None, cmap="turbo", show_edges=True, edge_color="dimgray")
        plotter.add_axes()
        plotter.add_text(f"Step: {st_name}, Frame: {frame.frameId} (t={frame.frameValue:.3f}s)", font_size=12, color="white")
        plotter.show()
