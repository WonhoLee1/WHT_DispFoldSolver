"""vtkhdf_writer.py
================
Universal Transient VTKHDF exporter supporting both 2D and 3D solid unstructured grids.
Standard Kitware VTKHDF format (Version 2.2) compatible with ParaView 5.10+ and PyVista.

[Streaming Design]
OpenRadioss-to-vtkhdf 프로젝트의 add_or_create_dataset 패턴을 채용:
- __init__()에서 h5.File을 열고 정적 토폴로지(Connectivity, Offsets, Types)를 기록
- add_step() 호출 시마다 즉시 HDF5 resizable dataset에 append → 해석 중 실시간 결과 확인 가능
- close()는 h5.File만 닫음 (데이터 손실 없음)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
import h5py as h5

# Standard VTK Cell Types
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_QUADRATIC_TETRA = 24


class TransientVTKHDFWriter:
    """Universal exporter for transient FEA results to a single .vtkhdf file (streaming).

    해석 시작 시 파일을 열고 add_step() 호출마다 즉시 디스크에 기록합니다.
    ParaView에서 해석 도중에도 실시간으로 결과를 열람할 수 있습니다.

    Parameters
    ----------
    filepath : str
        출력 VTKHDF 파일 경로 (예: "result.vtkhdf").
    mesh : Any
        FEA 메쉬 객체. ``nodes_array()``, ``elements``, ``node_id_to_index()`` API 필요.

    Examples
    --------
    >>> writer = TransientVTKHDFWriter("result.vtkhdf", mesh)
    >>> for step in solver.steps:
    ...     writer.add_step(time_val=step.time, u=step.u)
    >>> writer.close()
    """

    def __init__(self, filepath: str, mesh: Any) -> None:
        self.filepath = filepath
        self.mesh = mesh
        self.step_count: int = 0

        # ── 1. 기본 좌표 및 차원 결정 ──────────────────────────────────────
        pts = mesh.nodes_array().copy()
        self.n_nodes: int = pts.shape[0]
        self._ref_points = np.zeros((self.n_nodes, 3), dtype=np.float32)
        if pts.shape[1] == 2:
            self._ref_points[:, :2] = pts
            self.dim: int = 2
        else:
            self._ref_points[:, :3] = pts[:, :3]
            self.dim = 3

        # ── 2. 정적 토폴로지 구성 ──────────────────────────────────────────
        nid_to_idx = mesh.node_id_to_index()
        connectivity_list: List[int] = []
        offsets_list: List[int] = [0]
        types_list: List[int] = []

        for elem in mesh.elements.values():
            n_nodes_elem = len(elem.node_ids)
            vtk_type = self._map_vtk_type(n_nodes_elem)
            types_list.append(vtk_type)
            for nid in elem.node_ids:
                connectivity_list.append(nid_to_idx[nid])
            offsets_list.append(offsets_list[-1] + n_nodes_elem)

        self._connectivity = np.array(connectivity_list, dtype=np.int32)
        self._offsets = np.array(offsets_list, dtype=np.int32)
        self._types = np.array(types_list, dtype=np.uint8)
        self.n_cells: int = len(types_list)
        self.connectivity_size: int = self._connectivity.size

        # ── 3. HDF5 파일 열기 및 정적 구조 기록 ────────────────────────────
        self._hf = h5.File(self.filepath, "w")
        self._vtk_grp = self._hf.create_group("VTKHDF", track_order=True)
        self._vtk_grp.attrs["Version"] = np.array([2, 2], dtype=np.int32)
        ascii_type = b"UnstructuredGrid"
        self._vtk_grp.attrs.create(
            "Type", ascii_type, dtype=h5.string_dtype("ascii", len(ascii_type))
        )

        # 정적 토폴로지 (한 번만 기록)
        self._vtk_grp.create_dataset("Connectivity", data=self._connectivity)
        self._vtk_grp.create_dataset("Offsets", data=self._offsets)
        self._vtk_grp.create_dataset("Types", data=self._types)

        # 동적 데이터셋 (step마다 append → maxshape None)
        self._vtk_grp.create_dataset(
            "NumberOfPoints", shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self._vtk_grp.create_dataset(
            "NumberOfCells", shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self._vtk_grp.create_dataset(
            "NumberOfConnectivityIds", shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self._vtk_grp.create_dataset(
            "Points", shape=(0, 3), maxshape=(None, 3), dtype=np.float32
        )

        # PointData / CellData / Steps 그룹
        self._pd_grp = self._vtk_grp.create_group("PointData", track_order=True)
        self._cd_grp = self._vtk_grp.create_group("CellData", track_order=True)
        self._steps_grp = self._vtk_grp.create_group("Steps", track_order=True)
        self._steps_grp.create_dataset(
            "Values", shape=(0,), maxshape=(None,), dtype=np.float64
        )
        self._steps_grp.create_dataset(
            "PointOffsets", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        self._steps_grp.create_dataset(
            "CellOffsets", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        self._steps_grp.create_dataset(
            "ConnectivityIdOffsets", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        self._steps_grp.create_dataset(
            "PartOffsets", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        self._pd_offsets_grp = self._steps_grp.create_group("PointDataOffsets")
        self._cd_offsets_grp = self._steps_grp.create_group("CellDataOffsets")

        # NSteps 속성은 close() 시 업데이트
        self._steps_grp.attrs["NSteps"] = np.int32(0)
        self._hf.flush()

    # ──────────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────────

    def add_step(
        self,
        time_val: float,
        u: np.ndarray,
        point_data: Optional[Dict[str, np.ndarray]] = None,
        cell_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """수렴된 단일 스텝 결과를 HDF5에 즉시 append 기록.

        Parameters
        ----------
        time_val : float
            현재 스텝 시간 (초 또는 의사 시간).
        u : np.ndarray
            절점 변위 배열. shape ``(n_dofs,)`` 또는 ``(n_nodes, dim)``.
        point_data : dict, optional
            절점 결과 필드 {이름: shape (n_nodes,) 또는 (n_nodes, k)}.
        cell_data : dict, optional
            요소 결과 필드 {이름: shape (n_cells,) 또는 (n_cells, k)}.
        """
        step_idx = self.step_count
        n = self.n_nodes
        nc = self.n_cells

        # ── 변형 좌표 계산 ─────────────────────────────────────────────────
        u_arr = np.asarray(u, dtype=np.float32).reshape((n, self.dim))
        u_3d = np.zeros((n, 3), dtype=np.float32)
        u_3d[:, :self.dim] = u_arr
        deformed_pts = self._ref_points + u_3d   # shape (n_nodes, 3)

        # ── Points append ──────────────────────────────────────────────────
        self._append_dataset(self._vtk_grp["Points"], deformed_pts)

        # ── NumberOf* append ───────────────────────────────────────────────
        self._append_scalar(self._vtk_grp["NumberOfPoints"], n)
        self._append_scalar(self._vtk_grp["NumberOfCells"], nc)
        self._append_scalar(self._vtk_grp["NumberOfConnectivityIds"], self.connectivity_size)

        # ── 기본 Displacement 필드 ─────────────────────────────────────────
        point_data = dict(point_data) if point_data is not None else {}
        point_data.setdefault("Displacement", u_3d)

        # ── PointData append ───────────────────────────────────────────────
        for key, arr in point_data.items():
            v = np.asarray(arr, dtype=np.float32)
            self._append_field(self._pd_grp, key, v, step_idx)

        # ── CellData append ────────────────────────────────────────────────
        if cell_data is not None:
            for key, arr in cell_data.items():
                v = np.asarray(arr, dtype=np.float32)
                self._append_field(self._cd_grp, key, v, step_idx)

        # ── Steps 인덱싱 append ────────────────────────────────────────────
        self._append_scalar(self._steps_grp["Values"], float(time_val))
        self._append_scalar(self._steps_grp["PointOffsets"], step_idx * n)
        self._append_scalar(self._steps_grp["CellOffsets"], np.int64(0))
        self._append_scalar(self._steps_grp["ConnectivityIdOffsets"], np.int64(0))
        self._append_scalar(self._steps_grp["PartOffsets"], np.int64(0))

        for key in point_data.keys():
            ds_name = key
            if ds_name not in self._pd_offsets_grp:
                self._pd_offsets_grp.create_dataset(
                    ds_name, shape=(0,), maxshape=(None,), dtype=np.int64
                )
            self._append_scalar(self._pd_offsets_grp[ds_name], step_idx * n)

        if cell_data is not None:
            for key in cell_data.keys():
                if key not in self._cd_offsets_grp:
                    self._cd_offsets_grp.create_dataset(
                        key, shape=(0,), maxshape=(None,), dtype=np.int64
                    )
                self._append_scalar(self._cd_offsets_grp[key], step_idx * nc)

        self.step_count += 1
        self._steps_grp.attrs["NSteps"] = np.int32(self.step_count)

        # 즉시 디스크에 플러시 → ParaView가 실시간으로 읽을 수 있음
        self._hf.flush()

    def close(self) -> None:
        """HDF5 파일 핸들을 닫습니다. 모든 데이터는 add_step()에서 이미 기록됨."""
        if not self._hf.id.valid:
            return
        self._steps_grp.attrs["NSteps"] = np.int32(self.step_count)
        self._hf.close()

    def __enter__(self) -> "TransientVTKHDFWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ──────────────────────────────────────────────────────────────────────
    # 내부 헬퍼 메서드
    # ──────────────────────────────────────────────────────────────────────

    def _map_vtk_type(self, n_nodes_elem: int) -> int:
        """요소 절점 수를 VTK Cell Type 정수로 변환.

        Parameters
        ----------
        n_nodes_elem : int
            요소의 절점 수.

        Returns
        -------
        int
            VTK Cell Type 상수.
        """
        if self.dim == 2:
            return VTK_QUAD if n_nodes_elem == 4 else VTK_TRIANGLE
        else:
            mapping = {8: VTK_HEXAHEDRON, 4: VTK_TETRA, 10: VTK_QUADRATIC_TETRA, 6: VTK_WEDGE}
            return mapping.get(n_nodes_elem, VTK_HEXAHEDRON)

    @staticmethod
    def _append_scalar(ds: h5.Dataset, value: Any) -> None:
        """HDF5 1-D resizable dataset에 스칼라 1개를 append.

        Parameters
        ----------
        ds : h5.Dataset
            maxshape=(None,) 로 생성된 HDF5 dataset.
        value : scalar
            추가할 값.
        """
        old_len = ds.shape[0]
        ds.resize(old_len + 1, axis=0)
        ds[old_len] = value

    @staticmethod
    def _append_dataset(ds: h5.Dataset, arr: np.ndarray) -> None:
        """HDF5 N-D resizable dataset에 배열을 axis-0 방향으로 append.

        Parameters
        ----------
        ds : h5.Dataset
            maxshape=(None, ...) 로 생성된 HDF5 dataset.
        arr : np.ndarray
            추가할 배열. shape[0]만큼 행 추가.
        """
        old_len = ds.shape[0]
        ds.resize(old_len + arr.shape[0], axis=0)
        ds[old_len:] = arr

    def _append_field(
        self, grp: h5.Group, key: str, arr: np.ndarray, step_idx: int
    ) -> None:
        """PointData / CellData 그룹에 필드 배열을 append.

        첫 번째 스텝에는 resizable dataset을 생성하고,
        이후 스텝에는 resize + append.

        Parameters
        ----------
        grp : h5.Group
            PointData 또는 CellData HDF5 그룹.
        key : str
            필드 이름.
        arr : np.ndarray
            저장할 배열. shape (n, k) 또는 (n,).
        step_idx : int
            현재 스텝 인덱스 (0-based).
        """
        if key not in grp:
            # 첫 등장 시 resizable dataset 생성
            if arr.ndim == 1:
                grp.create_dataset(
                    key, data=arr, maxshape=(None,), dtype=np.float32, chunks=True
                )
            else:
                k = arr.shape[1]
                grp.create_dataset(
                    key, data=arr, maxshape=(None, k), dtype=np.float32, chunks=True
                )
        else:
            self._append_dataset(grp[key], arr)
