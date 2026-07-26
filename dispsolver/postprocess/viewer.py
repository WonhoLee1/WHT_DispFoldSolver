"""
dispsolver/postprocess/viewer.py — 2D FEA Postprocessing Viewer
================================================================

VTKHDF exporter를 대체하는 2D 전용 후처리 도구.
matplotlib + PySide6로 각 프레임의 해석 결과를 2D 컨투어 플롯으로 시각화.

요구사항: PySide6, matplotlib, numpy

사용법:
    from dispsolver.postprocess import PostprocessViewer
    viewer = PostprocessViewer(solver)
    viewer.show()

또는 CLI:
    python -m dispsolver.postprocess.viewer

아키텍처 개요:
    _ResultCache  →  solver state로부터 변위/변형률/응력 필드 계산 (데이터 레이어)
    _MplCanvas    →  matplotlib Figure를 PySide6에 임베드 (뷰 레이어)
    PostprocessViewer → PySide6 QMainWindow, 전체 UI 조립 (컨트롤러 레이어)

데이터 흐름:
    DynamicSolver → _ResultCache.compute_field() → FieldData
    FieldData.values → PatchCollection (element-wise fill)
    사용자 조작 (field combo, scale slider, PID 체크박스) → _update_plot() 재호출

필드 종류:
    - disp: 절점 변위 magnitude (mm)
    - strain_xx/yy/xy: Green-Lagrange 변형률 E (무차원)
    - strain_vm: von Mises 등가 변형률
    - stress_xx/yy/xy: Cauchy 응력 σ (MPa) — PK2에서 push-forward
    - stress_vm: von Mises 등가 응력 (MPa)
    - principal_*/principal_*_abs_max: 주응력/주변형률 및 절대값 최대
"""

from __future__ import annotations

import sys
import os
import numpy as np
from typing import Optional, List, Tuple, Dict, Callable
from dataclasses import dataclass, field

os.environ["PYTHONIOENCODING"] = "utf-8"

# ---------------------------------------------------------------------------
# Qt / matplotlib imports (graceful fallback if not installed)
# ---------------------------------------------------------------------------
# PySide6는 GUI 백엔드. matplotlib canvas를 QWidget으로 임베드하는 데 사용.
# 설치되지 않은 경우 ImportError를 나중에 throw하도록 None 처리.
try:
    from PySide6 import QtWidgets, QtCore, QtGui
    from PySide6.QtCore import Qt
except ImportError:
    QtWidgets = None
    QtCore = None
    QtGui = None
    Qt = None

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as MplCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as MplToolbar
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import matplotlib.tri as mtri
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon
except ImportError:
    MplCanvas = None
    MplToolbar = None
    plt = None
    Figure = None


# ===================================================================
# 데이터 계층 (Data Layer)
# ===================================================================
# _ResultCache는 DynamicSolver의 상태(u, state, conn, coords)를 읽어
# 사용자가 선택한 물리량(변위, 변형률, 응력)을 요소별/절점별 스칼라 필드로 계산한다.
#
# 계산 파이프라인 (각 요소의 중심 ξ=η=0 에서 평가):
#   1. 절점 변위 u_elem → shape function 미분 → 변위구배 ∇u
#   2. ∇u → F = I + ∇u (변형구배)
#   3. F → E = 0.5(F^T·F - I) (Green-Lagrange 변형률)
#   4. F + 재료 내부변수 state → PK2 응력 S → Cauchy σ = F·S·F^T / det(F)
#   5. E 또는 σ → 주값/등가값 계산
#
# ※ 모든 필드는 요소 중심 1점 적분으로 계산 (Gauss point 평균 대신).
#   정밀한 시각화가 필요하면 향후 2x2 Gauss point로 확장 가능.
# ===================================================================

@dataclass
class FieldData:
    """단일 스칼라 필드를 메시 위에 표현하기 위한 컨테이너.

    Attributes:
        name:     표시용 이름 (e.g. "Displacement", "Stress Von Mises")
        unit:     물리 단위 ("mm", "MPa", 또는 "" 무차원)
        per_node: True=절점 데이터 (n_nodes,), False=요소 데이터 (n_elem,)
        values:   각 절점 또는 요소의 스칼라값 배열
        vmin:     컬러맵 하한 (자동 계산)
        vmax:     컬러맵 상한 (자동 계산)
    """
    name: str
    unit: str = ""
    per_node: bool = True
    values: Optional[np.ndarray] = None
    vmin: float = 0.0
    vmax: float = 1.0


class _ResultCache:
    """DynamicSolver state로부터 필드 데이터를 계산하고 캐싱.

    동일한 solver state에 대해 compute_field()를 여러 번 호출해도
    내부 _cache 딕셔너리에 저장된 결과를 반환하므로 재계산 없음.
    (solver state가 변경되면 _cache를 비워야 함 — 향후 TimeHistory 지원 시 필요)

    Public Methods:
        compute_field(name: str) → FieldData
        available_fields() → List[str]
    """

    def __init__(self, solver):
        """
        Args:
            solver: DynamicSolver 인스턴스 (solve_step이 최소 1회 호출된 후).
        """
        self._solver = solver
        self._cache: Dict[str, np.ndarray] = {}

    def _grad_u_elem(self, coords: np.ndarray, u_elem: np.ndarray,
                     dN_dxi, dN_deta, invJ) -> np.ndarray:
        """사변형 요소의 중심에서 변위구배 ∇u 계산.

        Total Lagrangian 정식화에서 변위구배는 기준형상(X)에 대한 미분.
        ∇u = du/dX = Σ dN_i/dX * u_i

        Args:
            coords:  요소 절점 좌표 (4, 2) — 기준형상.
            u_elem:  요소 절점 변위 (8,) — [ux0, uy0, ux1, uy1, ...]
            dN_dxi:  자연좌표계 ξ 방향 shape function 미분 (4,)
            dN_deta: 자연좌표계 η 방향 shape function 미분 (4,)
            invJ:    Jacobian 역행렬 (2, 2) — 자연→물리좌표 변환.

        Returns:
            grad_u: (2, 2) 변위구배 텐서.
                [[ux_x, ux_y],   ← dux/dX, dux/dY
                 [uy_x, uy_y]]   ← duy/dX, duy/dY
        """
        # dN/dX = J^{-1} · dN/d(ξ,η)  — chain rule
        dN_dX = invJ @ np.vstack([dN_dxi, dN_deta])  # shape: (2, 4)
        # dN_dX[0, :] = dN/dX,  dN_dX[1, :] = dN/dY

        ux = u_elem[0::2]  # x-변위 성분 [ux0, ux1, ux2, ux3]
        uy = u_elem[1::2]  # y-변위 성분 [uy0, uy1, uy2, uy3]

        grad = np.zeros((2, 2), dtype=np.float64)
        grad[0, 0] = np.dot(dN_dX[0], ux)  # dux/dX = Σ dN_i/dX * ux_i
        grad[0, 1] = np.dot(dN_dX[1], ux)  # dux/dY = Σ dN_i/dY * ux_i
        grad[1, 0] = np.dot(dN_dX[0], uy)  # duy/dX = Σ dN_i/dX * uy_i
        grad[1, 1] = np.dot(dN_dX[1], uy)  # duy/dY = Σ dN_i/dY * uy_i
        return grad

    def _deformation_gradient(self, grad_u: np.ndarray) -> np.ndarray:
        """변형구배 텐서 F 계산 (Total Lagrangian).

        F = I + ∇u
        여기서 ∇u는 기준형상(X)에 대한 변위구배 (위 _grad_u_elem 참조).

        Args:
            grad_u: 변위구배 (2, 2).

        Returns:
            F: 변형구배 (2, 2). F = ∂x/∂X = I + ∂u/∂X.
        """
        return np.eye(2, dtype=np.float64) + grad_u

    def _green_lagrange_strain(self, F: np.ndarray) -> np.ndarray:
        """Green-Lagrange 변형률 E 계산.

        E = 0.5 * (F^T · F - I)

        Voigt 표기법으로 반환 (평면변형률):
            [E11, E22, 2*E12]
        여기서 2*E12 = γ_xy는 공학적 전단변형률.

        Args:
            F: 변형구배 (2, 2).

        Returns:
            E_v: (3,) Voigt 변형률 [E11, E22, γ_xy].
        """
        E = 0.5 * (F.T @ F - np.eye(2))
        return np.array([E[0, 0], E[1, 1], 2.0 * E[0, 1]])

    def _principal_2d(self, a11: float, a22: float, a12: float):
        """대칭 2x2 텐서의 주값 계산.

        텐서 [a11 a12; a12 a22]의 고유값을 해석적으로 계산.
        특성방정식: λ² - tr(A)·λ + det(A) = 0
        판별식: Δ = tr² - 4·det

        Args:
            a11, a22: 대각 성분.
            a12:      비대각 성분 (a12 = a21).

        Returns:
            (p1, p2): 두 주값. p1 ≥ p2 (큰 값, 작은 값 순서).
        """
        trace = a11 + a22
        det = a11 * a22 - a12 * a12
        disc = max(0.0, trace * trace - 4.0 * det)
        sqrt_disc = np.sqrt(disc)
        p1 = 0.5 * (trace + sqrt_disc)
        p2 = 0.5 * (trace - sqrt_disc)
        return p1, p2

    def _von_mises(self, s11: float, s22: float, s12: float) -> float:
        """von Mises 등가 응력 (평면변형률).

        σ_vm = sqrt(σ_xx² + σ_yy² - σ_xx·σ_yy + 3·σ_xy²)

        3D von Mises: sqrt( ( (σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)² ) / 2 )
        평면변형률(σ₃₃ = ν·(σ₁₁+σ₂₂) 또는 ε₃₃=0 조건):
        위 공식 사용.

        Args:
            s11: σ_xx (Cauchy).
            s22: σ_yy.
            s12: σ_xy.

        Returns:
            von Mises 등가 응력 (같은 단위).
        """
        return np.sqrt(s11*s11 + s22*s22 - s11*s22 + 3.0*s12*s12)

    def compute_field(self, name: str) -> FieldData:
        """solver state로부터 이름이 지정된 물리량 필드를 계산.

        캐시 확인 → 미계산 시 계산 → FieldData로 반환.
        계산은 모든 요소를 순회하며 요소 중심(ξ=η=0)에서 1점 평가.

        Args:
            name: 필드 이름.
                'disp'                    — 절점 변위 magnitude (mm)
                'strain_xx/yy/xy/vm'      — Green-Lagrange 변형률
                'stress_xx/yy/xy/vm'      — Cauchy 응력 (MPa)
                'principal_strain_1/2'    — 주변형률
                'principal_stress_1/2'    — 주응력
                'principal_strain/stress_abs_max'  — 절대값 최대 주값

        Returns:
            FieldData 인스턴스.
        """
        if name in self._cache:
            return self._cache[name]

        solver = self._solver
        n_elem = solver.n_elem
        conn = solver.conn             # (n_elem, 4) — 요소별 절점 인덱스 (0-based)
        coords_all = solver.coords     # (n_nodes, 2) — 절점 좌표 (기준형상)
        u_all = solver.u               # (n_dofs,) — 변위 벡터 [ux0, uy0, ux1, uy1, ...]
        state_all = solver.state       # (n_elem, n_gp, n_vars) — 요소별 내부변수 또는 None

        # --- 절점 데이터: 변위 magnitude ---
        if 'disp' == name:
            # 절점별 변위 벡터 magnitude: |u| = sqrt(ux² + uy²)
            ux = u_all[0::2]  # 짝수 인덱스 = x-변위
            uy = u_all[1::2]  # 홀수 인덱스 = y-변위
            vals = np.sqrt(ux**2 + uy**2)
            fd = FieldData('Displacement', 'mm', per_node=True, values=vals,
                           vmin=0.0, vmax=float(np.max(vals)))
            self._cache[name] = fd
            return fd

        # --- 요소 데이터: 변형률/응력 (요소 중심 1점 평가) ---
        # 요소 중심 (ξ=0, η=0)에서 shape function 미분값을 미리 계산.
        # QUAD4 요소에서 중심 = (0,0), shape function: N_i = 0.25*(1±ξ)*(1±η)
        # dN/dξ, dN/dη는 (ξ=0,η=0)에서 일정.
        from dispsolver.element.q4 import jacobian, shape_derivatives

        xi, eta = 0.0, 0.0  # 요소 중심
        dN_dxi, dN_deta = shape_derivatives(xi, eta)

        # 재료 모델 — PK2 응력 계산에 사용.
        # solver.material은 단일 재료 또는 dict의 첫 번째 재료.
        mat = solver.material

        elem_vals = np.zeros(n_elem, dtype=np.float64)

        for e in range(n_elem):
            # ------------------------------------------------------------
            # 요소별 데이터 추출
            # ------------------------------------------------------------
            nidx = conn[e]                # (4,) 절점 인덱스
            coords_e = coords_all[nidx]   # (4, 2) 요소 절점 좌표

            # 절점별 변위 → 요소 변위 벡터 (8,)
            ux_e = u_all[2 * nidx]               # (4,)
            uy_e = u_all[2 * nidx + 1]           # (4,)
            u_elem = np.zeros(8, dtype=np.float64)
            u_elem[0::2] = ux_e
            u_elem[1::2] = uy_e

            # Jacobian: 자연좌표(ξ,η)와 물리좌표(X,Y) 간 변환.
            # detJ > 0이어야 요소가 올바른 방향 (면적 > 0).
            try:
                J, detJ, invJ = jacobian(xi, eta, coords_e)
                if detJ <= 0:
                    continue  # 뒤집힌 요소는 스킵
            except Exception:
                continue

            # ------------------------------------------------------------
            # 변형구배 F = I + ∇u
            # ------------------------------------------------------------
            grad_u = self._grad_u_elem(coords_e, u_elem, dN_dxi, dN_deta, invJ)
            F = self._deformation_gradient(grad_u)

            # ------------------------------------------------------------
            # Green-Lagrange 변형률 E
            # ------------------------------------------------------------
            # E = 0.5*(F^T·F - I), Voigt: [E11, E22, 2*E12]
            E_v = self._green_lagrange_strain(F)
            e11, e22, e12_2 = E_v
            e12 = e12_2 / 2.0  # 공학적 전단 → 텐서 전단

            # ------------------------------------------------------------
            # Cauchy 응력 σ = F·S·F^T / det(F)
            # ------------------------------------------------------------
            # solver는 PK2 응력 S를 저장함 (참조형상 정의).
            # 물리적 응력 시각화를 위해 Cauchy 응력(현재형상)으로 변환:
            #   σ = (1/detF) * F * S * F^T
            # 이 변환을 "push-forward"라고 함.
            if 'stress' in name or 'principal_stress' in name:
                if state_all is not None and state_all.shape[0] > e:
                    # 첫 번째 Gauss point의 내부변수 사용
                    # (J2 소성: [Fp_inv_00, Fp_inv_01, Fp_inv_10, Fp_inv_11, eqps])
                    gp_state = state_all[e, 0, :]
                else:
                    # 기본 상태: 탄성 (Fp_inv = I, eqps = 0)
                    gp_state = np.array([1.0, 0.0, 0.0, 1.0, 0.0])

                try:
                    # PK2 응력 (Voigt) 계산: S = pk2_voigt(F, params, state)
                    # 반환값: (S_voigt, C_tangent, state_new) 또는 S_voigt만
                    S_res = mat.pk2_voigt(F, {}, gp_state)
                    S_v = S_res[0] if isinstance(S_res, tuple) else S_res
                    # S_v = [S11, S22, S12]

                    # PK2 Voigt → 2x2 텐서
                    S_tensor = np.array([[S_v[0], S_v[2]],
                                          [S_v[2], S_v[1]]])

                    detF = np.linalg.det(F)
                    if abs(detF) > 1e-15:
                        # Push-forward: σ = F·S·F^T / detF
                        sigma = F @ S_tensor @ F.T / detF
                        s11, s22 = sigma[0, 0], sigma[1, 1]
                        s12 = sigma[0, 1]
                    else:
                        s11 = s22 = s12 = 0.0
                except Exception:
                    # 재료 계산 실패 시 응력 0 처리
                    # (비수렴 요소, 초기 상태 등에서 발생 가능)
                    s11 = s22 = s12 = 0.0
            else:
                s11 = s22 = s12 = 0.0

            # ------------------------------------------------------------
            # 필드 선택 및 값 할당
            # ------------------------------------------------------------
            if name == 'strain_xx':
                elem_vals[e] = e11
            elif name == 'strain_yy':
                elem_vals[e] = e22
            elif name == 'strain_xy':
                elem_vals[e] = e12_2  # 공학적 전단변형률 γ_xy = 2*E12
            elif name == 'strain_vm':
                # von Mises 등가 변형률 (소변형 근사)
                ev = np.sqrt((e11-e22)**2 + e11**2 + e22**2 + 6*e12**2) / np.sqrt(2)
                elem_vals[e] = ev
            elif name == 'stress_xx':
                elem_vals[e] = s11
            elif name == 'stress_yy':
                elem_vals[e] = s22
            elif name == 'stress_xy':
                elem_vals[e] = s12
            elif name == 'stress_vm':
                elem_vals[e] = self._von_mises(s11, s22, s12)
            elif name == 'principal_strain_1':
                p1, p2 = self._principal_2d(e11, e22, e12)
                elem_vals[e] = p1
            elif name == 'principal_strain_2':
                p1, p2 = self._principal_2d(e11, e22, e12)
                elem_vals[e] = p2
            elif name == 'principal_stress_1':
                p1, p2 = self._principal_2d(s11, s22, s12)
                elem_vals[e] = p1
            elif name == 'principal_stress_2':
                p1, p2 = self._principal_2d(s11, s22, s12)
                elem_vals[e] = p2
            elif name == 'principal_strain_abs_max':
                # 두 주변형률 중 절대값이 큰 쪽
                p1, p2 = self._principal_2d(e11, e22, e12)
                elem_vals[e] = p1 if abs(p1) >= abs(p2) else p2
            elif name == 'principal_stress_abs_max':
                # 두 주응력 중 절대값이 큰 쪽
                p1, p2 = self._principal_2d(s11, s22, s12)
                elem_vals[e] = p1 if abs(p1) >= abs(p2) else p2
            else:
                raise ValueError(f"Unknown field: {name}")

        # ------------------------------------------------------------
        # FieldData 구성 — 유효값 범위로 vmin/vmax 설정
        # ------------------------------------------------------------
        valid = elem_vals[np.isfinite(elem_vals)]
        vmin = float(np.min(valid)) if len(valid) > 0 else 0.0
        vmax = float(np.max(valid)) if len(valid) > 0 else 1.0
        if abs(vmax - vmin) < 1e-15:
            vmax = vmin + 1.0  # 모든 값이 동일하면 폭 1 부여

        display_name = name.replace('_', ' ').title()
        unit = 'mm' if 'disp' in name else ('MPa' if 'stress' in name else '')
        fd = FieldData(display_name, unit, per_node=False,
                       values=elem_vals, vmin=vmin, vmax=vmax)
        self._cache[name] = fd
        return fd

    def available_fields(self) -> List[str]:
        """사용 가능한 모든 필드 이름 목록 반환."""
        return [
            'disp',
            'strain_xx', 'strain_yy', 'strain_xy', 'strain_vm',
            'stress_xx', 'stress_yy', 'stress_xy', 'stress_vm',
            'principal_strain_1', 'principal_strain_2',
            'principal_stress_1', 'principal_stress_2',
            'principal_strain_abs_max', 'principal_stress_abs_max',
        ]


# ===================================================================
# 뷰 계층 (View Layer) — matplotlib canvas 임베드
# ===================================================================
# FigureCanvasQTAgg는 matplotlib Figure를 Qt QWidget으로 래핑.
# NavigationToolbar2QT는 표준 matplotlib 툴바 (줌, 팬, 저장 등).
# ===================================================================

class _MplCanvas(MplCanvas):
    """matplotlib Figure를 PySide6 QWidget에 임베드하는 캔버스.

    Figure를 생성하고 add_subplot(111)로 단일 축을 추가.
    _update_plot()에서 이 ax를 clear()하고 다시 그림.
    """

    def __init__(self, parent=None):
        """
        Args:
            parent: Qt 부모 위젯 (선택사항).
        """
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.axes = self.fig.add_subplot(111)  # 단일 축
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


# ===================================================================
# 메인 뷰어 (PostprocessViewer)
# ===================================================================
# PySide6 QMainWindow 기반. 좌측에 matplotlib 캔버스 + 툴바,
# 우측에 컨트롤 패널 (필드 선택, 변형 배율, 메시 표시, PID 가시성).
#
# 컨트롤 패널 구성 (위→아래):
#   1. Result (QComboBox) — 표시할 물리량 선택
#   2. Deformation Scale (QSlider) — 변형 확대 배율 (0~10배)
#   3. Show Element Lines (QCheckBox) — 요소 경계선 표시 토글
#   4. Part / Layer (QCheckBox 그룹) — PID별 요소 가시성
#   5. Step Info (QGroupBox) — 현재 스텝 정보 표시
# ===================================================================

class PostprocessViewer(QtWidgets.QMainWindow):
    """PySide6 + matplotlib FEA 후처리 뷰어 메인 윈도우.

    사용법:
        solver = DynamicSolver(...)
        solver.solve_step(dt)
        viewer = PostprocessViewer(solver)
        viewer.show()

    Args:
        solver: DynamicSolver 인스턴스 (solve_step 호출 후).
        parent: Qt 부모 위젯 (선택사항).
    """

    # 내부 필드 키 → 콤보박스 표시 텍스트 매핑
    FIELD_LABELS: Dict[str, str] = {
        'disp': 'Displacement',
        'strain_xx': 'Strain XX',
        'strain_yy': 'Strain YY',
        'strain_xy': 'Strain XY',
        'strain_vm': 'Strain von Mises',
        'stress_xx': 'Stress XX',
        'stress_yy': 'Stress YY',
        'stress_xy': 'Stress XY',
        'stress_vm': 'Stress von Mises',
        'principal_strain_1': 'Principal Strain 1',
        'principal_strain_2': 'Principal Strain 2',
        'principal_stress_1': 'Principal Stress 1',
        'principal_stress_2': 'Principal Stress 2',
        'principal_strain_abs_max': 'Principal Strain (abs max)',
        'principal_stress_abs_max': 'Principal Stress (abs max)',
    }

    def __init__(self, solver, parent=None):
        super().__init__(parent)

        if QtWidgets is None:
            raise ImportError(
                "PySide6 is required. Install with: pip install pyside6"
            )

        # --- 데이터 ---
        self._solver = solver
        self._cache = _ResultCache(solver)
        self._current_field = 'disp'         # 초기 표시 필드: 변위
        self._deformed_scale: float = 1.0    # 변형 확대 배율
        self._show_mesh: bool = True         # 요소 경계선 표시 여부

        # PID (Part ID) 기반 요소 가시성.
        # mesh.elements의 모든 고유 pid를 수집하여 각각 체크박스 생성.
        self._pid_set: List[int] = sorted(set(
            solver.mesh.elements[eid].pid
            for eid in solver.mesh.elements
        ))
        self._pid_visible: Dict[int, bool] = {pid: True for pid in self._pid_set}

        # --- UI 구성 후 초기 플롯 ---
        self._init_ui()
        self._update_plot()

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------
    def _init_ui(self):
        """전체 UI 위젯을 조립.

        레이아웃 구조:
            QMainWindow
              └─ QHBoxLayout (central)
                   ├─ QVBoxLayout (canvas_container) [stretch=3]
                   │    ├─ _MplCanvas (matplotlib figure)
                   │    └─ MplToolbar (matplotlib 툴바)
                   └─ QWidget (panel) [fixed width=240]
                        ├─ Result (QComboBox)
                        ├─ Deformation Scale (QSlider)
                        ├─ Show Element Lines (QCheckBox)
                        ├─ Part / Layer (QCheckBox group)
                        └─ Step Info (QGroupBox)
        """
        self.setWindowTitle("FEA 2D Postprocessing Viewer")
        self.resize(1200, 800)

        # 중앙 위젯
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # --- 좌측: matplotlib 캔버스 + 툴바 ---
        canvas_container = QtWidgets.QVBoxLayout()
        self.canvas = _MplCanvas(self)
        canvas_container.addWidget(self.canvas)

        # matplotlib 기본 툴바 (줌, 팬, 홈, 저장 등)
        self.toolbar = MplToolbar(self.canvas, self)
        # 툴바 아이콘 크기를 기본의 70%로 축소 (compact UI)
        style = self.toolbar.style()
        icon_size = max(14, style.pixelMetric(QtWidgets.QStyle.PM_ToolBarIconSize))
        target_size = int(icon_size * 0.7)
        self.toolbar.setIconSize(QtCore.QSize(target_size, target_size))
        canvas_container.addWidget(self.toolbar)

        main_layout.addLayout(canvas_container, stretch=3)

        # --- 우측: 컨트롤 패널 ---
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(240)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setSpacing(4)

        # 1. 결과 필드 선택 (QComboBox)
        panel_layout.addWidget(QtWidgets.QLabel("<b>Result</b>"))
        self.field_combo = QtWidgets.QComboBox()
        for key, label in self.FIELD_LABELS.items():
            self.field_combo.addItem(label, key)  # 표시텍스트, 내부키
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        panel_layout.addWidget(self.field_combo)
        panel_layout.addSpacing(12)

        # 2. 변형 배율 슬라이더 (0~10x)
        panel_layout.addWidget(QtWidgets.QLabel("<b>Deformation Scale</b>"))
        scale_layout = QtWidgets.QHBoxLayout()
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.scale_slider.setRange(0, 100)       # 0~100 → 0.0~10.0x
        self.scale_slider.setValue(20)           # 기본 2.0x
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.scale_slider)
        self.scale_label = QtWidgets.QLabel("2.0x")
        self.scale_label.setFixedWidth(50)
        scale_layout.addWidget(self.scale_label)
        panel_layout.addLayout(scale_layout)
        panel_layout.addSpacing(12)

        # 3. 요소 경계선 표시 토글
        self.mesh_check = QtWidgets.QCheckBox("Show Element Lines")
        self.mesh_check.setChecked(True)
        self.mesh_check.stateChanged.connect(self._on_mesh_toggle)
        panel_layout.addWidget(self.mesh_check)
        panel_layout.addSpacing(12)

        # 4. PID별 요소 가시성 (Layer별 Show/Hide)
        # 각 pid에 대해 QCheckBox 생성. 체크 해제 시 해당 레이어의 요소가
        # 플롯에서 제외됨 (hide).
        panel_layout.addWidget(QtWidgets.QLabel("<b>Part / Layer</b>"))
        self.pid_checkboxes: Dict[int, QtWidgets.QCheckBox] = {}
        for pid in self._pid_set:
            cb = QtWidgets.QCheckBox(f"Layer {pid}")
            cb.setChecked(True)
            # lambda에서 pid를 기본값으로 캡처 (루프 변수 참조 방지)
            cb.stateChanged.connect(lambda state, p=pid: self._on_pid_toggle(p))
            panel_layout.addWidget(cb)
            self.pid_checkboxes[pid] = cb

        panel_layout.addStretch()

        # 5. 스텝 정보 표시
        info_group = QtWidgets.QGroupBox("Step Info")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        self.info_label = QtWidgets.QLabel("")
        info_layout.addWidget(self.info_label)
        panel_layout.addWidget(info_group)

        main_layout.addWidget(panel)

    # ------------------------------------------------------------------
    # Qt 슬롯 (사용자 조작에 반응)
    # ------------------------------------------------------------------
    def _on_field_changed(self, idx: int):
        """콤보박스에서 필드 변경 시 호출. 플롯 재생성."""
        self._current_field = self.field_combo.itemData(idx)
        self._update_plot()

    def _on_scale_changed(self, val: int):
        """변형 배율 슬라이더 변경 시 호출. 슬라이더값 0~100 → 0.0~10.0x."""
        self._deformed_scale = val * 0.1
        self.scale_label.setText(f"{self._deformed_scale:.1f}x")
        self._update_plot()

    def _on_mesh_toggle(self, state: int):
        """요소 경계선 표시 체크박스 변경 시 호출."""
        self._show_mesh = (state == QtCore.Qt.CheckState.Checked.value)
        self._update_plot()

    def _on_pid_toggle(self, pid: int):
        """Layer 체크박스 변경 시 호출. 해당 PID 요소 표시/숨김."""
        cb = self.pid_checkboxes.get(pid)
        self._pid_visible[pid] = cb.isChecked() if cb else True
        self._update_plot()

    # ------------------------------------------------------------------
    # 플로팅
    # ------------------------------------------------------------------
    def _update_plot(self):
        """matplotlib Figure를 현재 설정으로 다시 그림.

        플로팅 파이프라인:
          1. FieldData 계산 (또는 캐시에서 로드)
          2. 변형 후 절점 좌표 계산
          3. PatchCollection 생성 — 요소별 면색/경계선
          4. 컬러바 + 축 설정
          5. Step Info 업데이트
        """
        solver = self._solver
        ax = self.canvas.axes
        ax.clear()

        # 1. 필드 데이터 로드
        field = self._cache.compute_field(self._current_field)
        vals = field.values
        per_node = field.per_node

        # 2. 메시 데이터 추출
        conn = solver.conn            # (n_elem, 4) 절점 인덱스 (0-based)
        coords = solver.coords        # (n_nodes, 2) 기준 좌표
        u_all = solver.u              # (n_dofs,) 변위

        # 3. 변형 후 좌표 = 기준좌표 + 배율 × 변위
        if self._deformed_scale > 1e-6:
            ux = u_all[0::2]
            uy = u_all[1::2]
            coords_deformed = coords + self._deformed_scale * np.column_stack([ux, uy])
        else:
            coords_deformed = coords

        # 4. 요소별 PatchCollection 구성
        # 각 QUAD4 요소를 MplPolygon(4각형)으로 변환.
        # 면색 = 필드값, 경계선 = 검정 또는 없음.
        patches = []
        face_colors = []
        edge_colors = []
        linewidths = []

        for e in range(solver.n_elem):
            # PID 체크 — 숨김 레이어는 스킵
            eid = solver.elem_ids[e]
            elem_obj = solver.mesh.elements[eid]
            pid = elem_obj.pid if elem_obj.pid is not None else 0
            if not self._pid_visible.get(pid, True):
                continue

            nidx = conn[e]  # (4,) 절점 인덱스
            xy = coords_deformed[nidx]  # (4, 2) 변형 후 절점 좌표

            # 필드값: 절점 데이터면 요소 절점값 평균, 요소 데이터면 그대로
            if per_node:
                fc = float(np.mean(vals[nidx]))
            else:
                fc = float(vals[e])

            patches.append(MplPolygon(xy, closed=True))
            face_colors.append(fc)
            edge_colors.append('k' if self._show_mesh else 'none')
            linewidths.append(0.3 if self._show_mesh else 0.0)

        if not patches:
            ax.set_title("No visible elements")
            self.canvas.draw()
            return

        # 5. PatchCollection 생성 및 축에 추가
        pc = PatchCollection(patches, array=np.array(face_colors),
                             cmap='viridis', edgecolors=edge_colors,
                             linewidths=linewidths)
        ax.add_collection(pc)

        # 6. 컬러바 추가
        cb_label = field.name
        if field.unit:
            cb_label += f" [{field.unit}]"
        cbar = self.canvas.fig.colorbar(pc, ax=ax, label=cb_label)

        # 7. 축 범위 자동 설정 (equal aspect)
        all_xy = np.vstack([p.xy for p in patches])
        if len(all_xy) > 0:
            x_min, y_min = all_xy.min(axis=0)
            x_max, y_max = all_xy.max(axis=0)
            margin_x = max(1.0, (x_max - x_min) * 0.05)
            margin_y = max(1.0, (y_max - y_min) * 0.05)
            ax.set_xlim(x_min - margin_x, x_max + margin_x)
            ax.set_ylim(y_min - margin_y, y_max + margin_y)

        ax.set_aspect('equal')           # 종횡비 1:1
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title(f"{field.name}  (t = {solver.time:.4f} s)")

        # 8. Step Info 업데이트
        self.info_label.setText(
            f"Time: {solver.time:.4f} s\n"
            f"DOFs: {solver.n_dofs}\n"
            f"Elements: {solver.n_elem}\n"
            f"Field: {field.name}"
        )

        self.canvas.draw()

    # ------------------------------------------------------------------
    # 공개 API (프로그래매틱 제어)
    # ------------------------------------------------------------------
    def show(self):
        """뷰어 윈도우를 화면에 표시."""
        self.showNormal()
        self.raise_()

    def set_field(self, name: str):
        """표시할 필드를 이름으로 전환.

        Args:
            name: FIELD_LABELS의 키 (예: 'stress_vm', 'strain_xx').
        """
        if name in self.FIELD_LABELS:
            idx = self.field_combo.findData(name)
            if idx >= 0:
                self.field_combo.setCurrentIndex(idx)

    def set_deformed_scale(self, scale: float):
        """변형 확대 배율 설정.

        Args:
            scale: 확대 배율 (0 = 미변형, 1 = 실제, 10 = 10배).
        """
        self._deformed_scale = max(0.0, scale)
        self.scale_slider.setValue(int(scale * 10))
        self._update_plot()


# ===================================================================
# CLI 진입점
# ===================================================================
# launch_from_solver(solver)를 스크립트에서 호출하면 뷰어가 실행됨.
# QApplication은 한 프로세스에 하나만 존재해야 하므로,
# 기존 인스턴스가 있으면 재사용하고 없으면 새로 생성.
# ===================================================================

def launch_from_solver(solver):
    """solver 인스턴스로 뷰어를 실행 (스크립트에서 사용).

    Args:
        solver: DynamicSolver 인스턴스.

    Returns:
        QApplication.exec()의 종료 코드.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    viewer = PostprocessViewer(solver)
    viewer.show()
    return app.exec()


# ===================================================================
# 데모 (직접 실행 시)
# ===================================================================
# python -m dispsolver.postprocess.viewer 로 실행하면
# 2개 QUAD4 요소로 구성된 간단한 문제를 풀고 결과를 시각화.
# ===================================================================

if __name__ == "__main__":
    print("Building demo problem...")
    from dispsolver.mesh import Mesh
    from dispsolver.material import J2Plasticity
    from dispsolver.solver import DynamicSolver

    # 2개 QUAD4 요소, 2개 레이어 (PID=0, PID=1)
    m = Mesh()
    m.add_node(0, 0.0, 0.0)
    m.add_node(1, 10.0, 0.0)
    m.add_node(2, 10.0, 10.0)
    m.add_node(3, 0.0, 10.0)
    m.add_node(4, 0.0, 20.0)
    m.add_node(5, 10.0, 20.0)
    m.add_element(0, [0, 1, 2, 3], 'QUAD4', pid=0)
    m.add_element(1, [3, 2, 5, 4], 'QUAD4', pid=1)

    mat0 = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    mat1 = J2Plasticity(E=1000.0, nu=0.3, sigma_y0=40.0, H=100.0)

    solver = DynamicSolver(
        m, {0: mat0, 1: mat1}, rho=1000.0, material_params={},
        constraints=[], max_iter=10, tol=1e-3, verbose=False,
        element_type='Q4', mode='quasistatic',
    )

    nid = m.node_id_to_index()
    # 좌측 에지 고정 (node 0: Ux=Uy=0, node 3: Ux=0)
    solver.set_prescribed_dofs(
        [nid[0]*2, nid[0]*2+1, nid[3]*2], [0.0, 0.0, 0.0]
    )
    # 우측 상단 모서리에 하중 (node 5: Uy = -10 N)
    solver.apply_load([nid[5]*2+1], [-10.0])

    print("Solving...")
    n_iter = solver.solve_step(0.5)
    print(
        f"Converged in {n_iter+1} iterations" if n_iter >= 0
        else f"Not converged: {n_iter}"
    )

    sys.exit(launch_from_solver(solver))
