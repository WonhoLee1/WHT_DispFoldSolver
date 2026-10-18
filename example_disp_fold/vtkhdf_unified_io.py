"""
vtkhdf_unified_io.py
====================
Kitware VTKHDF Version 2.2 Universal Pre- and Post-processing database exporter.
Compatible with ParaView 5.10+ and PyVista. Stores parts, sets, material metadata,
and streams dynamic time-series field data (Displacement, Stress, Slip) in real time.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import h5py as h5

# Standard VTK Cell Types
VTK_QUAD = 9
VTK_HEXAHEDRON = 12


class UnifiedVTKHDFDatabase:
    """Universal Pre/Post VTKHDF database writer supporting 2D and 3D display models."""

    def __init__(self, filepath: str, mesh: Any, meta: Dict[str, Any], config: Any) -> None:
        self.filepath = filepath
        self.mesh = mesh
        self.meta = meta
        self.config = config
        self.step_count = 0

        # Coordinates
        pts = mesh.nodes_array().copy()
        self.n_nodes = pts.shape[0]
        self._ref_points = np.zeros((self.n_nodes, 3), dtype=np.float32)
        if pts.shape[1] == 2:
            self._ref_points[:, :2] = pts
            self.dim = 2
            self.vtk_type = VTK_QUAD
        else:
            self._ref_points[:, :3] = pts[:, :3]
            self.dim = 3
            self.vtk_type = VTK_HEXAHEDRON

        # Topology
        nid_to_idx = mesh.node_id_to_index()
        connectivity_list = []
        offsets_list = [0]
        types_list = []

        layer_ids = []
        part_ids = []
        mat_types = []

        # Map elements
        layer_elem_sets = meta.get("layer_elem_sets", {})
        elem_to_layer = {}
        for lay_idx, (lay_name, eids) in enumerate(layer_elem_sets.items()):
            for eid in eids:
                elem_to_layer[eid] = lay_idx

        for elem in mesh.elements.values():
            n_elem_nodes = len(elem.node_ids)
            for nid in elem.node_ids:
                connectivity_list.append(nid_to_idx[nid])
            offsets_list.append(offsets_list[-1] + n_elem_nodes)
            types_list.append(self.vtk_type)

            # Metadata tags for ParaView filtering
            lay_id = elem_to_layer.get(getattr(elem, "eid", getattr(elem, "id", None)), -1)
            layer_ids.append(lay_id)

            pid = getattr(elem, "pid", 1)
            if pid == 4:
                # Plate
                part_id = 1
                mat_type = 4
            else:
                part_id = 0
                mat_type = pid
            part_ids.append(part_id)
            mat_types.append(mat_type)

        self._connectivity = np.array(connectivity_list, dtype=np.int32)
        self._offsets = np.array(offsets_list, dtype=np.int32)
        self._types = np.array(types_list, dtype=np.uint8)
        self.n_cells = len(types_list)

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self._hf = h5.File(self.filepath, "w")
        self._vtk_grp = self._hf.create_group("VTKHDF", track_order=True)
        self._vtk_grp.attrs["Version"] = np.array([2, 2], dtype=np.int32)
        ascii_type = b"UnstructuredGrid"
        self._vtk_grp.attrs.create("Type", ascii_type, dtype=h5.string_dtype("ascii", len(ascii_type)))

        # Static Topology
        self._vtk_grp.create_dataset("Connectivity", data=self._connectivity)
        self._vtk_grp.create_dataset("Offsets", data=self._offsets)
        self._vtk_grp.create_dataset("Types", data=self._types)

        # Dynamic offsets / lengths
        self._vtk_grp.create_dataset("NumberOfPoints", shape=(0,), maxshape=(None,), dtype=np.int32)
        self._vtk_grp.create_dataset("NumberOfCells", shape=(0,), maxshape=(None,), dtype=np.int32)
        self._vtk_grp.create_dataset("NumberOfConnectivityIds", shape=(0,), maxshape=(None,), dtype=np.int32)
        self._vtk_grp.create_dataset("Points", shape=(0, 3), maxshape=(None, 3), dtype=np.float32)

        # Preprocessing metadata group
        self._prep_grp = self._vtk_grp.create_group("PreProcessing", track_order=True)
        self._prep_grp.attrs["TotalThickness"] = float(config.total_thickness)
        self._prep_grp.attrs["Preset"] = str(config.simulation_case.preset_name)
        self._prep_grp.attrs["Dimension"] = str(config.simulation_case.dim)

        # PreProcessing/Sets group hierarchy
        sets_grp = self._prep_grp.create_group("Sets", track_order=True)
        nsets_subgrp = sets_grp.create_group("NodeSets", track_order=True)
        elsets_subgrp = sets_grp.create_group("ElementSets", track_order=True)

        # Direct NodeSets and ElementSets for backward compatibility
        nsets_grp = self._prep_grp.create_group("NodeSets", track_order=True)
        for k in ["disp_bottom_left_nids", "disp_bottom_right_nids", "plate_left_nodes", "plate_right_nodes"]:
            if k in meta:
                arr = np.array(meta[k], dtype=np.int32)
                nsets_grp.create_dataset(k, data=arr)
                nsets_subgrp.create_dataset(k, data=arr)

        elsets_grp = self._prep_grp.create_group("ElementSets", track_order=True)
        for lay_name, eids in layer_elem_sets.items():
            arr = np.array(eids, dtype=np.int32)
            elsets_grp.create_dataset(lay_name, data=arr)
            elsets_subgrp.create_dataset(lay_name, data=arr)

        # Static CellData for Native ParaView Filtering
        self._cd_grp = self._vtk_grp.create_group("CellData", track_order=True)
        self._cd_grp.create_dataset("Layer_ID", data=np.array(layer_ids, dtype=np.int32))
        self._cd_grp.create_dataset("Part_ID", data=np.array(part_ids, dtype=np.int32))
        self._cd_grp.create_dataset("Material_Type", data=np.array(mat_types, dtype=np.int32))

        # Dynamic PointData and Steps
        self._pd_grp = self._vtk_grp.create_group("PointData", track_order=True)
        self._pd_grp.create_dataset("Displacement", shape=(0, 3), maxshape=(None, 3), dtype=np.float32)
        self._pd_grp.create_dataset("U", shape=(0, 3), maxshape=(None, 3), dtype=np.float32)

        self._steps_grp = self._vtk_grp.create_group("Steps", track_order=True)
        self._steps_grp.create_dataset("Values", shape=(0,), maxshape=(None,), dtype=np.float64)
        self._steps_grp.create_dataset("PointOffsets", shape=(0,), maxshape=(None,), dtype=np.int64)
        self._steps_grp.create_dataset("CellOffsets", shape=(0,), maxshape=(None,), dtype=np.int64)
        self._steps_grp.create_dataset("ConnectivityIdOffsets", shape=(0,), maxshape=(None,), dtype=np.int64)
        self._steps_grp.create_dataset("PartOffsets", shape=(0,), maxshape=(None,), dtype=np.int64)

        self._pd_offsets_grp = self._steps_grp.create_group("PointDataOffsets")
        self._pd_offsets_grp.create_dataset("Displacement", shape=(0,), maxshape=(None,), dtype=np.int64)
        self._pd_offsets_grp.create_dataset("U", shape=(0,), maxshape=(None,), dtype=np.int64)

        self._cd_offsets_grp = self._steps_grp.create_group("CellDataOffsets")
        self._steps_grp.attrs["NSteps"] = np.int32(0)
        self._hf.flush()

    def add_transient_step(
        self,
        time_val: float,
        u: np.ndarray,
        s: Optional[np.ndarray] = None,
        le: Optional[np.ndarray] = None,
        peeq: Optional[np.ndarray] = None,
        step_name: Optional[str] = None,
    ) -> None:
        """Appends a converged time increment with transient field data to the VTKHDF database.

        Parameters
        ----------
        time_val : float
            Simulation time increment value.
        u : np.ndarray
            Nodal displacement field array.
        s : Optional[np.ndarray]
            Element stress field (e.g. Mises stress or stress tensor).
        le : Optional[np.ndarray]
            Element logarithmic strain field.
        peeq : Optional[np.ndarray]
            Element equivalent plastic strain field.
        step_name : Optional[str]
            Optional step name identifier.
        """
        u_arr = np.asarray(u, dtype=np.float64)
        ndof_node = 2 if self.dim == 2 else 3

        # Displacement array in 3D
        disp_3d = np.zeros((self.n_nodes, 3), dtype=np.float32)
        if u_arr.ndim == 1:
            u_reshaped = u_arr.reshape((self.n_nodes, ndof_node))
        else:
            u_reshaped = u_arr

        if self.dim == 2:
            disp_3d[:, :2] = u_reshaped[:, :2].astype(np.float32)
        else:
            disp_3d[:, :3] = u_reshaped[:, :3].astype(np.float32)

        # Deformed current points
        pts_curr = self._ref_points + disp_3d

        # Append to HDF5 Points
        n_prev_pts = self._vtk_grp["Points"].shape[0]
        self._vtk_grp["Points"].resize((n_prev_pts + self.n_nodes, 3))
        self._vtk_grp["Points"][n_prev_pts:] = pts_curr

        self._vtk_grp["NumberOfPoints"].resize((self.step_count + 1,))
        self._vtk_grp["NumberOfPoints"][self.step_count] = self.n_nodes

        self._vtk_grp["NumberOfCells"].resize((self.step_count + 1,))
        self._vtk_grp["NumberOfCells"][self.step_count] = self.n_cells

        self._vtk_grp["NumberOfConnectivityIds"].resize((self.step_count + 1,))
        self._vtk_grp["NumberOfConnectivityIds"][self.step_count] = self._connectivity.size

        # Append Displacement and U
        n_prev_disp = self._pd_grp["Displacement"].shape[0]
        self._pd_grp["Displacement"].resize((n_prev_disp + self.n_nodes, 3))
        self._pd_grp["Displacement"][n_prev_disp:] = disp_3d

        self._pd_grp["U"].resize((n_prev_disp + self.n_nodes, 3))
        self._pd_grp["U"][n_prev_disp:] = disp_3d

        # Append CellData transient fields (S, LE, PEEQ) if provided
        for field_name, field_val in [("S", s), ("LE", le), ("PEEQ", peeq)]:
            if field_val is not None:
                arr = np.asarray(field_val, dtype=np.float32)
                if field_name not in self._cd_grp:
                    if arr.ndim == 1:
                        self._cd_grp.create_dataset(field_name, shape=(0,), maxshape=(None,), dtype=np.float32)
                    else:
                        ncols = arr.shape[1]
                        self._cd_grp.create_dataset(field_name, shape=(0, ncols), maxshape=(None, ncols), dtype=np.float32)
                    self._cd_offsets_grp.create_dataset(field_name, shape=(0,), maxshape=(None,), dtype=np.int64)

                prev_size = self._cd_grp[field_name].shape[0]
                if arr.ndim == 1:
                    self._cd_grp[field_name].resize((prev_size + self.n_cells,))
                    self._cd_grp[field_name][prev_size:] = arr
                else:
                    ncols = arr.shape[1]
                    self._cd_grp[field_name].resize((prev_size + self.n_cells, ncols))
                    self._cd_grp[field_name][prev_size:] = arr

                self._cd_offsets_grp[field_name].resize((self.step_count + 1,))
                self._cd_offsets_grp[field_name][self.step_count] = prev_size

        # Update Steps
        self._steps_grp["Values"].resize((self.step_count + 1,))
        self._steps_grp["Values"][self.step_count] = float(time_val)

        self._steps_grp["PointOffsets"].resize((self.step_count + 1,))
        self._steps_grp["PointOffsets"][self.step_count] = n_prev_pts

        self._steps_grp["CellOffsets"].resize((self.step_count + 1,))
        self._steps_grp["CellOffsets"][self.step_count] = self.step_count * self.n_cells

        self._steps_grp["ConnectivityIdOffsets"].resize((self.step_count + 1,))
        self._steps_grp["ConnectivityIdOffsets"][self.step_count] = self.step_count * self._connectivity.size

        self._steps_grp["PartOffsets"].resize((self.step_count + 1,))
        self._steps_grp["PartOffsets"][self.step_count] = 0

        self._pd_offsets_grp["Displacement"].resize((self.step_count + 1,))
        self._pd_offsets_grp["Displacement"][self.step_count] = n_prev_disp

        self._pd_offsets_grp["U"].resize((self.step_count + 1,))
        self._pd_offsets_grp["U"][self.step_count] = n_prev_disp

        self.step_count += 1
        self._steps_grp.attrs["NSteps"] = np.int32(self.step_count)
        self._hf.flush()

    def add_step(self, time_val: float, u: np.ndarray, step_name: Optional[str] = None) -> None:
        """Compatibility wrapper forwarding to add_transient_step."""
        self.add_transient_step(time_val=time_val, u=u, step_name=step_name)

    def close(self) -> None:
        """Closes the HDF5 file handle cleanly."""
        if hasattr(self, "_hf") and self._hf:
            self._steps_grp.attrs["NSteps"] = np.int32(self.step_count)
            self._hf.flush()
            self._hf.close()
            self._hf = None


class UnifiedVTKHDFReader:
    """Reader for VTKHDF v2.2 files generated by UnifiedVTKHDFWriter."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._hf = h5.File(filepath, "r")
        if "VTKHDF" not in self._hf:
            raise ValueError(f"Not a valid VTKHDF file: 'VTKHDF' root group missing in {filepath}")
        self.vtk_grp = self._hf["VTKHDF"]
        self.version = tuple(int(x) for x in self.vtk_grp.attrs.get("Version", [2, 2]))
        self.vtk_type = self.vtk_grp.attrs.get("Type")

        self.n_cells = len(self.vtk_grp["Types"])
        self.connectivity = self.vtk_grp["Connectivity"][:]
        self.offsets = self.vtk_grp["Offsets"][:]
        self.types = self.vtk_grp["Types"][:]

        steps_grp = self.vtk_grp.get("Steps")
        if steps_grp:
            self.n_steps = int(steps_grp.attrs.get("NSteps", 0))
            self.times = list(steps_grp["Values"][:]) if self.n_steps > 0 else []
        else:
            self.n_steps = 0
            self.times = []

        if self.n_steps > 0 and "NumberOfPoints" in self.vtk_grp:
            self.n_nodes = int(self.vtk_grp["NumberOfPoints"][0])
        elif "Points" in self.vtk_grp:
            self.n_nodes = self.vtk_grp["Points"].shape[0]
        else:
            self.n_nodes = 0

    @property
    def cell_data_keys(self) -> List[str]:
        cd = self.vtk_grp.get("CellData")
        return list(cd.keys()) if cd else []

    @property
    def point_data_keys(self) -> List[str]:
        pd = self.vtk_grp.get("PointData")
        return list(pd.keys()) if pd else []

    def get_preprocessing_metadata(self) -> Dict[str, Any]:
        prep = self.vtk_grp.get("PreProcessing")
        if not prep:
            return {}
        return {
            "TotalThickness": float(prep.attrs.get("TotalThickness", 0.0)),
            "Preset": str(prep.attrs.get("Preset", "")),
            "Dimension": str(prep.attrs.get("Dimension", "")),
        }

    def get_preprocessing_sets(self) -> Dict[str, Dict[str, np.ndarray]]:
        res = {"NodeSets": {}, "ElementSets": {}}
        prep = self.vtk_grp.get("PreProcessing")
        if not prep:
            return res
        # Check Sets group
        sets_grp = prep.get("Sets")
        if sets_grp:
            if "NodeSets" in sets_grp:
                for k in sets_grp["NodeSets"].keys():
                    res["NodeSets"][k] = sets_grp["NodeSets"][k][:]
            if "ElementSets" in sets_grp:
                for k in sets_grp["ElementSets"].keys():
                    res["ElementSets"][k] = sets_grp["ElementSets"][k][:]
        if "NodeSets" in prep:
            for k in prep["NodeSets"].keys():
                res["NodeSets"][k] = prep["NodeSets"][k][:]
        if "ElementSets" in prep:
            for k in prep["ElementSets"].keys():
                res["ElementSets"][k] = prep["ElementSets"][k][:]
        return res

    def get_cell_data(self, name: str, step_idx: Optional[int] = None) -> np.ndarray:
        cd = self.vtk_grp["CellData"]
        if name not in cd:
            raise KeyError(f"CellData '{name}' not found. Available: {list(cd.keys())}")
        dset = cd[name]
        if step_idx is None:
            return dset[:]
        start = step_idx * self.n_cells
        end = start + self.n_cells
        return dset[start:end]

    def get_point_data(self, name: str, step_idx: Optional[int] = None) -> np.ndarray:
        pd = self.vtk_grp["PointData"]
        target_name = name
        if name == "U" and "U" not in pd and "Displacement" in pd:
            target_name = "Displacement"
        elif name == "Displacement" and "Displacement" not in pd and "U" in pd:
            target_name = "U"

        if target_name not in pd:
            raise KeyError(f"PointData '{name}' not found. Available: {list(pd.keys())}")
        dset = pd[target_name]
        if step_idx is None:
            return dset[:]
        start = step_idx * self.n_nodes
        end = start + self.n_nodes
        return dset[start:end]

    def read_transient_step(self, step_idx: int) -> Dict[str, Any]:
        if not (0 <= step_idx < self.n_steps):
            raise IndexError(f"step_idx {step_idx} out of range [0, {self.n_steps - 1}]")
        step_dict = {"time": self.times[step_idx]}
        if "U" in self.vtk_grp["PointData"]:
            step_dict["U"] = self.get_point_data("U", step_idx)
        elif "Displacement" in self.vtk_grp["PointData"]:
            step_dict["U"] = self.get_point_data("Displacement", step_idx)

        for field in ["S", "LE", "PEEQ"]:
            if field in self.vtk_grp["CellData"]:
                step_dict[field] = self.get_cell_data(field, step_idx)
        return step_dict

    def close(self) -> None:
        if hasattr(self, "_hf") and self._hf:
            self._hf.close()
            self._hf = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Unified VTKHDF Writer alias
UnifiedVTKHDFWriter = UnifiedVTKHDFDatabase


def load_vtkhdf_summary(filepath: str) -> Dict[str, Any]:
    """Inspects a VTKHDF file and returns structural metadata summary."""
    with h5.File(filepath, "r") as f:
        vtk_grp = f["VTKHDF"]
        version = list(vtk_grp.attrs.get("Version", []))
        n_cells = vtk_grp["Types"].shape[0]
        n_steps = int(vtk_grp["Steps"].attrs.get("NSteps", 0))
        times = list(vtk_grp["Steps"]["Values"][:]) if n_steps > 0 else []

        prep = vtk_grp.get("PreProcessing")
        preset = prep.attrs.get("Preset", "unknown") if prep else "unknown"

        return {
            "version": version,
            "num_cells": n_cells,
            "num_steps": n_steps,
            "times": times,
            "preset": preset,
        }
