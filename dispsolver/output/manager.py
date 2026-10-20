"""manager.py
==========
OutputManager는 FieldOutputRequest, HistoryOutputRequest, 실시간 VTKHDF 기록을
솔버 콜백과 연동하여 조율합니다.

[VTKHDF 연동]
- ``vtkhdf_path`` 인수를 지정하면 해석 중 수렴 스텝마다 즉시 .vtkhdf 파일에 append.
- ParaView에서 해석 도중 실시간으로 현재 결과를 열람 가능.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import warnings
import numpy as np

from dispsolver.output.variables import FieldPosition, FieldType
from dispsolver.output.requests import FieldOutputRequest, HistoryOutputRequest, OutputTimingController
from dispsolver.output.field_data import FieldOutput
from dispsolver.output.history_data import HistoryRegion, EnergyTracker
from dispsolver.output.odb import Odb, OdbStep, OdbFrame
from dispsolver.output.recovery2d import recover_2d_fields
from dispsolver.output.recovery3d import recover_3d_fields
from dispsolver.output.vtkhdf_writer import TransientVTKHDFWriter


class OutputManager:
    """Field / History Output 평가 및 실시간 VTKHDF 기록을 총괄하는 매니저.

    Parameters
    ----------
    odb : Odb
        결과 저장소 객체.
    step_name : str
        해석 스텝 이름 (예: "Step-1").
    time_period : float, optional
        스텝 전체 시간 (기본값 1.0).
    field_requests : list of FieldOutputRequest, optional
        Field Output 요청 목록.
    history_requests : list of HistoryOutputRequest, optional
        History Output 요청 목록.
    dim : int, optional
        공간 차원 (2 또는 3). 기본값 3.
    vtkhdf_path : str, optional
        실시간 스트리밍 저장 파일 경로.
        지정 시 매 Field Output 기록 시마다 즉시 .vtkhdf 파일에 append.
        지정하지 않으면 VTKHDF 저장 없음.
    """

    def __init__(
        self,
        odb: Odb,
        step_name: str,
        time_period: float = 1.0,
        field_requests: Optional[List[FieldOutputRequest]] = None,
        history_requests: Optional[List[HistoryOutputRequest]] = None,
        dim: int = 3,
        vtkhdf_path: Optional[str] = None,
        time_offset: float = 0.0,
    ) -> None:
        self.odb = odb
        self.step_name = step_name
        self.time_period = float(time_period)
        self.time_offset = float(time_offset)
        self.dim = int(dim)

        self.odb_step = self.odb.get_or_create_step(step_name=step_name, time_period=self.time_period)
        self.energy_tracker = EnergyTracker()

        # Field 요청 & 타이밍 컨트롤러
        self.field_controllers: List[Tuple[FieldOutputRequest, OutputTimingController]] = []
        if field_requests:
            for req in field_requests:
                ctrl = OutputTimingController(request=req, step_time_period=self.time_period)
                self.field_controllers.append((req, ctrl))

        # History 요청 & 타이밍 컨트롤러
        self.history_controllers: List[Tuple[HistoryOutputRequest, OutputTimingController]] = []
        if history_requests:
            for req in history_requests:
                ctrl = OutputTimingController(request=req, step_time_period=self.time_period)
                self.history_controllers.append((req, ctrl))

        self.current_frame_id: int = 0

        # VTKHDF 스트리밍 writer (지정된 경우에만 초기화)
        self._vtkhdf_writer: Optional[TransientVTKHDFWriter] = None
        self._vtkhdf_path: Optional[str] = vtkhdf_path
        # writer는 첫 번째 record_step()에서 lazy 초기화 (solver.mesh 필요)

    def next_exact_time(self, step_time: float) -> Optional[float]:
        """Earliest time any output request wants an increment to end on exactly, or None."""
        times = [t for _, c in (*self.field_controllers, *self.history_controllers)
                 if (t := c.next_exact_time(step_time)) is not None]
        return min(times) if times else None

    def switch_step(
        self,
        step_name: str,
        time_period: float = 1.0,
        field_requests: Optional[List[FieldOutputRequest]] = None,
        history_requests: Optional[List[HistoryOutputRequest]] = None,
        time_offset: Optional[float] = None,
    ) -> None:
        """Switch active step for subsequent recordings, preserving existing VTKHDF writer and accumulated states."""
        self.step_name = step_name
        self.time_period = float(time_period)
        if time_offset is not None:
            self.time_offset = float(time_offset)
        self.odb_step = self.odb.get_or_create_step(step_name=step_name, time_period=self.time_period)

        self.field_controllers = []
        if field_requests:
            for req in field_requests:
                ctrl = OutputTimingController(request=req, step_time_period=self.time_period)
                self.field_controllers.append((req, ctrl))

        self.history_controllers = []
        if history_requests:
            for req in history_requests:
                ctrl = OutputTimingController(request=req, step_time_period=self.time_period)
                self.history_controllers.append((req, ctrl))

    # ──────────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────────

    def record_step(
        self,
        step_time: float,
        inc: int,
        solver: Any,
        dt: float,
        du: np.ndarray,
        f_int: np.ndarray,
        f_ext: Optional[np.ndarray] = None,
        is_last_step: bool = False,
    ) -> None:
        """수렴된 스텝 상태를 활성 요청에 따라 기록.

        Parameters
        ----------
        step_time : float
            현재 스텝 누적 시간.
        inc : int
            현재 increment 번호 (0-based).
        solver : Any
            DynamicSolver2D 또는 DynamicSolver3D 인스턴스.
        dt : float
            이번 스텝의 시간 증분.
        du : np.ndarray
            이번 스텝의 절점 변위 증분.
        f_int : np.ndarray
            내력 벡터.
        f_ext : np.ndarray, optional
            외력 벡터.
        is_last_step : bool, optional
            마지막 스텝 여부.
        """
        # 1. 전체 에너지 업데이트
        self.energy_tracker.update_step(dt=dt, solver=solver, du=du, f_int=f_int, f_ext=f_ext)

        # 2. History Output 기록 체크
        for req, ctrl in self.history_controllers:
            if ctrl.should_output(step_time=step_time, inc=inc, is_last_step=is_last_step, solver=solver):
                ctrl.mark_recorded(step_time=step_time, inc=inc)
                self._record_history_request(req=req, step_time=step_time, solver=solver)

        # 3. Field Output 기록 체크
        triggered_field_reqs: List[FieldOutputRequest] = []
        for req, ctrl in self.field_controllers:
            if ctrl.should_output(step_time=step_time, inc=inc, is_last_step=is_last_step, solver=solver):
                ctrl.mark_recorded(step_time=step_time, inc=inc)
                triggered_field_reqs.append(req)

        if triggered_field_reqs:
            self._record_field_frame(
                step_time=step_time,
                solver=solver,
                requests=triggered_field_reqs,
            )

    def finalize(self) -> None:
        """해석 완료 후 VTKHDF writer 파일을 닫습니다.

        ``solve_with_odb()`` 에서 자동 호출됩니다. 직접 호출할 수도 있습니다.
        """
        if self._vtkhdf_writer is not None:
            self._vtkhdf_writer.close()
            self._vtkhdf_writer = None

    # ──────────────────────────────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_vtkhdf_writer(self, solver: Any) -> Optional[TransientVTKHDFWriter]:
        """VTKHDF writer를 lazy 초기화 (solver.mesh가 필요하므로 첫 호출 시 생성).

        Parameters
        ----------
        solver : Any
            메쉬 정보를 가진 솔버 인스턴스 (``solver.mesh`` 속성 필요).

        Returns
        -------
        TransientVTKHDFWriter or None
            vtkhdf_path가 지정되지 않은 경우 None.
        """
        if self._vtkhdf_path is None:
            return None
        if self._vtkhdf_writer is None:
            mesh = getattr(solver, "mesh", None)
            if mesh is None:
                return None
            self._vtkhdf_writer = TransientVTKHDFWriter(
                filepath=self._vtkhdf_path, mesh=mesh
            )
        return self._vtkhdf_writer

    def _record_field_frame(
        self,
        step_time: float,
        solver: Any,
        requests: List[FieldOutputRequest],
    ) -> None:
        """요청된 필드를 복원하고 OdbFrame 생성 + VTKHDF append.

        Parameters
        ----------
        step_time : float
            현재 스텝 시간.
        solver : Any
            솔버 인스턴스.
        requests : list of FieldOutputRequest
            이번 스텝에 트리거된 요청 목록.
        """
        # 전체 요청 변수 병합
        merged_vars: set = set()
        primary_pos = FieldPosition.INTEGRATION_POINTS
        for req in requests:
            merged_vars.update(req.variables)
            if req.position == FieldPosition.NODES:
                primary_pos = FieldPosition.NODES

        # 차원별 필드 복원
        if self.dim == 2:
            field_dict = recover_2d_fields(solver=solver, requested_vars=list(merged_vars), position=primary_pos)
        else:
            field_dict = recover_3d_fields(solver=solver, requested_vars=list(merged_vars), position=primary_pos)

        # OdbFrame 생성
        actual_total_time = float(step_time) + getattr(self, "time_offset", 0.0)
        frame = OdbFrame(
            frame_id=self.current_frame_id,
            frame_value=actual_total_time,
            description=f"Increment {self.current_frame_id}: Total Time = {actual_total_time:.4f}, Step Time = {step_time:.4f}",
            field_outputs=field_dict,
        )
        self.odb_step.add_frame(frame)

        # VTKHDF 실시간 append
        writer = self._ensure_vtkhdf_writer(solver)
        if writer is not None:
            u = getattr(solver, "u", None)
            if u is not None:
                # point_data / cell_data 구성
                point_data: Dict[str, np.ndarray] = {}
                cell_data: Dict[str, np.ndarray] = {}

                for var_name, fo in field_dict.items():
                    if fo is None:
                        continue
                    arr = None
                    if hasattr(fo, "bulkDataBlocks") and fo.bulkDataBlocks:
                        arr = fo.bulkDataBlocks[0].data
                    elif hasattr(fo, "data"):
                        arr = fo.data
                    if arr is None or getattr(arr, "size", 0) == 0:
                        continue

                    # 절점 결과 vs 요소 결과 분류
                    pos = getattr(fo, "position", None)
                    pos_val = pos.value if hasattr(pos, "value") else str(pos)
                    ftype = getattr(fo, "type", None)
                    ftype_val = ftype.value if hasattr(ftype, "value") else str(ftype)

                    if pos_val == "NODES":
                        point_data[var_name] = arr
                    elif pos_val in ("INTEGRATION_POINTS", "CENTROID"):
                        cell_data[var_name] = arr
                    else:
                        point_data[var_name] = arr

                writer.add_step(
                    time_val=actual_total_time,
                    u=u,
                    point_data=point_data if point_data else None,
                    cell_data=cell_data if cell_data else None,
                )

        self.current_frame_id += 1

    def _record_history_request(
        self,
        req: HistoryOutputRequest,
        step_time: float,
        solver: Any,
    ) -> None:
        """스칼라 History 이력을 HistoryRegion에 기록.

        Parameters
        ----------
        req : HistoryOutputRequest
            기록할 요청.
        step_time : float
            현재 스텝 시간.
        solver : Any
            솔버 인스턴스.
        """
        region_name = req.region if req.region is not None else "WholeModel"
        hr = self.odb_step.get_or_create_history_region(region_name)

        for var_name in req.variables:
            var_upper = var_name.upper()

            # 전체 모델 에너지 변수
            energy_map = {
                "ALLIE": self.energy_tracker.allie,
                "ALLSE": self.energy_tracker.allse,
                "ALLPD": self.energy_tracker.allpd,
                "ALLVD": self.energy_tracker.allvd,
                "ALLKE": self.energy_tracker.allke,
                "ALLWK": self.energy_tracker.allwk,
                "ALLAE": self.energy_tracker.allae,
                "ALLSD": self.energy_tracker.allsd,
                "ALLFD": self.energy_tracker.allfd,
                "ETOTAL": self.energy_tracker.etotal,
            }
            if var_upper in energy_map:
                hr.record(var_upper, step_time, energy_map[var_upper])
                continue

            # 절점 변수
            if var_upper.startswith("U") or var_upper.startswith("RF"):
                val = self._extract_nodal_history(solver, req.region, var_upper)
                if val is not None:
                    hr.record(var_upper, step_time, val)

    def _extract_nodal_history(
        self, solver: Any, region: Optional[str], var_name: str
    ) -> Optional[float]:
        """절점 변위/반력 성분을 솔버 벡터에서 추출.

        Parameters
        ----------
        solver : Any
            솔버 인스턴스 (``u`` 속성 필요).
        region : str or None
            절점 세트 이름 또는 None.
        var_name : str
            변수 이름 (예: "U1", "RF2").

        Returns
        -------
        float or None
            추출된 스칼라 값. 실패 시 None.
        """
        u = getattr(solver, "u", None)
        if u is None:
            return None

        u_nodes = u.reshape(-1, self.dim)
        comp_idx = 0
        if var_name.endswith("1"):
            comp_idx = 0
        elif var_name.endswith("2"):
            comp_idx = 1
        elif var_name.endswith("3") and self.dim > 2:
            comp_idx = 2

        if var_name.startswith("U"):
            target_idx = 0
            if isinstance(region, int):
                target_idx = region
            elif hasattr(solver, "mesh") and region in getattr(solver.mesh, "node_sets", {}):
                nids = solver.mesh.node_sets[region]
                if nids:
                    target_idx = solver.mesh.node_id_to_index()[nids[0]]
            elif region is not None:
                # Used to record node 0's value under this request's name.
                warned = self.__dict__.setdefault("_warned_history_regions", set())
                if region not in warned:
                    warned.add(region)
                    warnings.warn(
                        f"history output: node set {region!r} is not defined; "
                        f"no {var_name} data is recorded for it",
                        stacklevel=2)
                return None
            if target_idx < len(u_nodes):
                return float(u_nodes[target_idx, comp_idx])

        return None
