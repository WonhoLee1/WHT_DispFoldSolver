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
    FieldData.values → PolyCollection (element-wise fill, vectorized)
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
    from matplotlib.collections import PolyCollection
    import matplotlib as _mpl
    # 전체 UI 폰트 통일(Cascadia Code, 9pt) -- Qt 쪽은 PostprocessViewer.__init__
    # 에서 self.setFont()로 적용, matplotlib 쪽(축 라벨/제목/컬러바/틱)은
    # Figure 생성 전에 rcParams를 바꿔야 이후 만들어지는 Text 아티스트들이
    # 기본값으로 이 폰트를 집어간다.
    _mpl.rcParams['font.family'] = 'Cascadia Code'
    _mpl.rcParams['font.size'] = 9
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
        # solver.material은 단일 재료 또는 dict의 첫 번째 재료 -- 이건 폴백일
        # 뿐, 실제로는 각 요소의 pid로 solver.materials에서 찾는다. (버그 수정:
        # 예전엔 이 폴백을 모든 요소에 그대로 썼기 때문에, 다중 재료 모델(PET/
        # PSA/STEEL 등)에서 PSA/STEEL 요소도 PET 재료로 응력을 계산했다.)
        mat_by_pid = getattr(solver, "materials", None) or {}
        default_mat = solver.material
        # pid -> numeric params dict, only populated (and only needed) for
        # a file-backed ResultSolverAdapter -- ViscoelasticMaterial.pk2_voigt
        # needs its base material's params (mu/lambda_m/K, ...) passed in
        # each call; a live DynamicSolver's MaterialAdapter keeps that
        # separately (`self.params`), which this cache never touched
        # before because it always used solver.material's own state.
        params_by_pid = getattr(solver, "material_params", None) or {}

        elem_vals = np.zeros(n_elem, dtype=np.float64)

        for e in range(n_elem):
            # ------------------------------------------------------------
            # 요소별 데이터 추출
            # ------------------------------------------------------------
            nidx = conn[e]                # (4,) 절점 인덱스
            coords_e = coords_all[nidx]   # (4, 2) 요소 절점 좌표

            # 이 요소의 실제 pid에 해당하는 재료 선택 (버그 수정, 위 주석 참고).
            eid = solver.elem_ids[e]
            elem_pid = getattr(solver.mesh.elements[eid], "pid", None)
            mat = mat_by_pid.get(elem_pid, default_mat)

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
                # ViscoelasticMaterial(PSA 등)은 J2Plasticity와 pk2_voigt
                # 호출 규약이 전혀 다르다: h_prev가 (M+1,3,3) 텐서여야 하고
                # (J2는 5칸 flat), dt 인자가 필수(J2는 안 받음), params도
                # base 재료(mu/lambda_m/K)용이 필요(J2는 무시). 예전엔 모든
                # 요소가 pid=1(J2)의 solver.material로만 계산돼서 이 분기가
                # 실행될 일이 없었다 -- per-pid 재료 선택을 고치면서 PSA
                # 요소가 실제로 ViscoelasticMaterial.pk2_voigt를 타게 됐고,
                # 옛 호출 방식(3-args, flat state)은 TypeError로 죽어서
                # 조용히 응력 0으로 깔리고 있었다.
                is_visco = hasattr(mat, "base") and hasattr(mat, "M") and hasattr(mat, "g_i")
                mat_params = params_by_pid.get(elem_pid, {}) if is_visco else {}

                if is_visco:
                    from dispsolver.material.viscoelastic import _flat_batch_to_tensor_3d
                    n_vars = mat.n_internal_vars  # 6*(M+1), may be < padded row width
                    if state_all is not None and state_all.shape[0] > e:
                        flat = state_all[e, 0, :n_vars]
                    else:
                        flat = np.zeros(n_vars, dtype=np.float64)
                    gp_state = _flat_batch_to_tensor_3d(flat[None, :], mat.M)[0]
                elif state_all is not None and state_all.shape[0] > e:
                    # 첫 번째 Gauss point의 내부변수 사용
                    # (J2 소성: [Fp_inv_00, Fp_inv_01, Fp_inv_10, Fp_inv_11, eqps])
                    gp_state = state_all[e, 0, :]
                else:
                    # 기본 상태: 탄성 (Fp_inv = I, eqps = 0)
                    gp_state = np.array([1.0, 0.0, 0.0, 1.0, 0.0])

                try:
                    if is_visco:
                        # dt=0: 저장된 state는 이미 실제 해석에서 이 F로
                        # 수렴한 시점의 값이므로, 여기서 dt=0으로 재평가하면
                        # dS_dev(=S_dev_el-h_prev[M])가 0이 되어 Prony 항이
                        # 갱신되지 않고 정확히 그 순간의 평형응력이 재현된다
                        # (추가로 시간을 흘려보내는 근사가 아니라, 저장된
                        # 상태를 그대로 재조회하는 것).
                        S_res = mat.pk2_voigt(F, mat_params, gp_state, dt=0.0)
                    else:
                        S_res = mat.pk2_voigt(F, {}, gp_state)
                    # PK2 응력 (Voigt) 계산: S = pk2_voigt(F, params, state)
                    # 반환값: (S_voigt, C_tangent, state_new) 또는 S_voigt만
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
        # 메인 axes와 컬러바 axes 위치를 고정 rect로 직접 지정 -- 둘 다
        # add_subplot(111)/tight_layout()의 자동 배치에 맡기지 않는다.
        # fig.colorbar(sm, ax=self.axes)처럼 컬러바를 메인 axes 기준으로
        # 매번 새로 만들면 make_axes_gridspec()이 self.axes의 SubplotSpec
        # 자체를 새 gridspec으로 바꿔치기한다 (ax.set_position()으로는
        # 되돌릴 수 없는 상태) -- 그래서 재호출마다 self.axes가 그
        # 바꿔치기된(이미 줄어든) SubplotSpec을 기준으로 또 줄어들어 계속
        # 누적된다. 여기서는 두 axes의 rect를 __init__에서 한 번만 정해두고,
        # 매 redraw마다 cax=self.cax로 그 axes에만 그려서 self.axes의
        # SubplotSpec/position은 아예 건드리지 않는다.
        self.axes.set_position([0.09, 0.10, 0.72, 0.85])
        self.cax = self.fig.add_axes([0.86, 0.15, 0.03, 0.7])


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

    def __init__(self, solver, parent=None, result=None):
        """
        Args:
            solver: DynamicSolver (live) or ResultSolverAdapter (saved
                file, see `live_view.py`) -- duck-typed, no type check.
            result: the backing `Result`, if any. Only needed to enable
                the step slider (a live-solver session has exactly one
                state and no history to scrub through). Passed
                automatically by `launch_from_result`.
        """
        super().__init__(parent)

        if QtWidgets is None:
            raise ImportError(
                "PySide6 is required. Install with: pip install pyside6"
            )

        # 전체 UI 폰트 통일(Cascadia Code, 9pt) -- matplotlib 쪽은 모듈
        # 로드 시 rcParams로 이미 적용됨(위 import 블록 참고). QApplication
        # 레벨에 걸어두면 이 창의 모든 자식 위젯(라벨/체크박스/버튼/콤보/
        # 스핀박스)이 상속받는다.
        _font = QtGui.QFont("Cascadia Code", 9)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setFont(_font)
        self.setFont(_font)

        # --- 데이터 ---
        self._solver = solver
        self._cache = _ResultCache(solver)
        self._result = result   # None for a live-solver session
        self._cbar = None       # 현재 컬러바 (지속 아티스트, __init__ 이후 첫 _update_plot에서 1회 생성)
        self._current_field = 'disp'         # 초기 표시 필드: 변위
        self._deformed_scale: float = 1.0    # 변형 확대 배율
        self._show_mesh: bool = True         # 요소 경계선 표시 여부

        # 성능 캐시 -- overlay_n<=1(기본, 가장 흔한 조작)일 때 매 재조작마다
        # ax.clear()+PatchCollection 통째로 재생성하는 대신, PolyCollection/
        # 컬러바/힌지 마커를 한 번만 만들고 이후엔 배열/속성만 갱신한다.
        # (원인: 요소별 Polygon 객체 3480개를 Python for문으로 새로 만드는
        # 비용 + ax.clear() 후 컬러바/축 재구성 비용이 실제 병목이었음 --
        # matplotlib 자체 렌더링 비용이 아니었음.)
        self._pc = None                 # 지속 PolyCollection (overlay<=1 전용)
        self._hinge_line = None         # 지속 힌지 마커 Line2D
        self._fast_artists_valid = False  # overlay(N>1) 경로가 ax.clear()한 뒤엔 무효화
        self._pids_key = None           # 마지막으로 pid 배열을 계산한 solver의 id()
        self._pids_val = None           # 그 solver의 요소별 pid 배열 (캐시)
        self._panning = False           # 중간 마우스 버튼 드래그 팬 진행 중 여부

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

        # 항상 켜진 pyvista 스타일 줌/팬 -- 기존 NavigationToolbar2QT(홈/
        # 뒤로/앞으로/팬 버튼/줌 사각형/저장)는 그대로 두고 추가로 붙인다.
        # 버튼 충돌 없음: 툴바의 자체 pan/zoom 모드는 왼쪽 버튼 드래그만
        # 반응하고(toolbar.mode로 게이팅됨), 우리 핸들러는 스크롤/중간
        # 버튼만 본다.
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_pan_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_pan_motion)
        self.canvas.mpl_connect('button_release_event', self._on_pan_release)

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

        # 2b. 스텝 슬라이더 -- 저장된 Result로 열었고 스텝이 2개 이상일 때만
        # 표시. 라이브 solver 세션은 상태가 하나뿐이라 스크러빙할 이력이
        # 없으므로 이 위젯 자체를 만들지 않는다 (동작 변경 없음).
        self.step_slider = None
        if self._result is not None and self._result.n_steps > 1:
            panel_layout.addWidget(QtWidgets.QLabel("<b>Step</b>"))
            step_layout = QtWidgets.QHBoxLayout()
            self.step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.step_slider.setRange(0, self._result.n_steps - 1)
            self.step_slider.setValue(self._result.n_steps - 1)  # 기본: 마지막 스텝
            self.step_slider.valueChanged.connect(self._on_step_changed)
            step_layout.addWidget(self.step_slider)
            self.step_label = QtWidgets.QLabel(f"{self._result.n_steps}/{self._result.n_steps}")
            self.step_label.setFixedWidth(60)
            step_layout.addWidget(self.step_label)
            panel_layout.addLayout(step_layout)
            panel_layout.addSpacing(4)

            # 2c. 오버레이 스텝 수 -- N>1이면 0..현재 스텝 사이에서 균등
            # 간격으로 N개를 뽑아 색까지 전부 입혀서 겹쳐 그린다(변형 진행
            # 과정을 한 그림에서 보기 위한 용도). 슬라이더와 마찬가지로
            # Result가 있을 때만 노출.
            overlay_layout = QtWidgets.QHBoxLayout()
            overlay_layout.addWidget(QtWidgets.QLabel("Overlay Steps (N)"))
            self.overlay_spin = QtWidgets.QSpinBox()
            self.overlay_spin.setRange(1, self._result.n_steps)
            self.overlay_spin.setValue(1)
            self.overlay_spin.valueChanged.connect(lambda _v: self._update_plot())
            overlay_layout.addWidget(self.overlay_spin)
            panel_layout.addLayout(overlay_layout)
            panel_layout.addSpacing(12)
        else:
            self.overlay_spin = None

        # 3. 요소 경계선 표시 토글
        self.mesh_check = QtWidgets.QCheckBox("Show Element Lines")
        self.mesh_check.setChecked(True)
        self.mesh_check.stateChanged.connect(self._on_mesh_toggle)
        panel_layout.addWidget(self.mesh_check)
        panel_layout.addSpacing(12)

        # 3b. 뷰 리셋 버튼 -- 스크롤/팬으로 어디에 있든 현재 표시 요소
        # 전체가 보이도록 되돌린다.
        self.fit_view_btn = QtWidgets.QPushButton("Fit View")
        self.fit_view_btn.clicked.connect(self._on_fit_view_clicked)
        panel_layout.addWidget(self.fit_view_btn)
        panel_layout.addSpacing(12)

        # 4. PID별 요소 가시성 (Layer별 Show/Hide)
        # 각 pid에 대해 QCheckBox 생성. 체크 해제 시 해당 레이어의 요소가
        # 플롯에서 제외됨 (hide). 레이어가 많으면(예: 14 레이어 + 플레이트
        # 2개) 패널 세로 공간을 넘치므로, 스크롤 영역 안에 넣어 나머지
        # 패널(Step Info 등)이 밀려나지 않게 한다.
        panel_layout.addWidget(QtWidgets.QLabel("<b>Part / Layer</b>"))
        self.pid_checkboxes: Dict[int, QtWidgets.QCheckBox] = {}
        # pid -> *MATERIAL 이름 (예: "PET") -- Result로 열었고 이 메타가
        # 있을 때만; 없으면 그냥 "Layer {pid}"로 표시 (기존 동작 유지).
        pid_names = (self._result.meta.get("pid_names", {}) if self._result is not None else {})
        # pid -> 사람이 읽을 이름 오버라이드(예: "Plate Left") -- 힌지 강체
        # 플레이트처럼 "Layer N" 표기가 안 맞는 파트용. 먼저 확인하고, 없으면
        # 기존 pid_names/"Layer N" 로직으로 폴백 (일반 표시 레이어는 여기
        # 항목이 없어서 자연히 폴백됨).
        pid_part_names = (self._result.meta.get("pid_part_names", {}) if self._result is not None else {})

        layer_scroll = QtWidgets.QScrollArea()
        layer_scroll.setWidgetResizable(True)
        layer_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        layer_container = QtWidgets.QWidget()
        layer_layout = QtWidgets.QVBoxLayout(layer_container)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.setSpacing(2)
        for pid in self._pid_set:
            if str(pid) in pid_part_names:
                label = pid_part_names[str(pid)]
            elif str(pid) in pid_names:
                label = f"Layer {pid} ({pid_names[str(pid)]})"
            else:
                label = f"Layer {pid}"
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            # lambda에서 pid를 기본값으로 캡처 (루프 변수 참조 방지)
            cb.stateChanged.connect(lambda state, p=pid: self._on_pid_toggle(p))
            layer_layout.addWidget(cb)
            self.pid_checkboxes[pid] = cb
        layer_layout.addStretch()
        layer_scroll.setWidget(layer_container)
        panel_layout.addWidget(layer_scroll, stretch=1)

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

    def _on_step_changed(self, step: int):
        """스텝 슬라이더 변경 시 호출.

        `_ResultCache`는 자체적으로 무효화하지 않으므로(§118 주석 참고,
        "향후 TimeHistory 지원 시 필요"라 적혀 있던 그 지점), 스텝이 바뀔
        때마다 solver 어댑터와 캐시를 모두 다시 만든다 -- 라이브 solver
        세션에서는 이 슬라이더 자체가 존재하지 않으므로 호출될 일이 없다.
        """
        from .live_view import ResultSolverAdapter
        self._solver = ResultSolverAdapter(self._result, step=step)
        self._cache = _ResultCache(self._solver)
        self.step_label.setText(f"{step + 1}/{self._result.n_steps}")
        self._update_plot()

    def _on_pid_toggle(self, pid: int):
        """Layer 체크박스 변경 시 호출. 해당 PID 요소 표시/숨김."""
        cb = self.pid_checkboxes.get(pid)
        self._pid_visible[pid] = cb.isChecked() if cb else True
        self._update_plot()

    def _on_fit_view_clicked(self):
        """'Fit View' 버튼 -- 현재 표시 중인 요소 전체가 보이도록 뷰 리셋."""
        self._fit_view()
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        """마우스 휠 = 커서 위치 고정 줌 (pyvista 스타일).

        event.step > 0(휠 위) = 확대, < 0(휠 아래) = 축소. x/y 범위를
        동일 배율로 스케일 -- _fit_view()가 세워둔 "x span == y span"
        불변식을 계속 유지해야 정사각 렌더링이 깨지지 않는다
        (adjustable='box'로는 더 이상 이걸 대신 맞춰주지 않으므로,
        _configure_static_axes 참고).
        """
        ax = self.canvas.axes
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return  # 커서가 메인 axes 밖(컬러바 axes 포함)이면 무시

        factor = 0.9 ** event.step
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        new_x0 = xdata - (xdata - x0) * factor
        new_x1 = xdata + (x1 - xdata) * factor
        new_y0 = ydata - (ydata - y0) * factor
        new_y1 = ydata + (y1 - ydata) * factor
        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)
        self._view_ready = True
        self.canvas.draw_idle()

    def _on_pan_press(self, event):
        """중간 마우스 버튼 누름 -- 팬 시작점(픽셀, xlim/ylim) 기록."""
        ax = self.canvas.axes
        if event.button != 2 or event.inaxes is not ax or event.x is None:
            return
        self._panning = True
        self._pan_start_px = (event.x, event.y)
        self._pan_start_xlim = ax.get_xlim()
        self._pan_start_ylim = ax.get_ylim()

    def _on_pan_motion(self, event):
        """드래그 중 -- 시작점 대비 누적 픽셀 델타를 데이터 델타로 환산해
        xlim/ylim을 이동 (시작 시점 값 기준, 프레임마다 재적산 없음)."""
        if not self._panning:
            return
        from matplotlib.backend_bases import MouseButton
        if event.buttons is not None and MouseButton.MIDDLE not in event.buttons:
            self._panning = False
            return
        if event.x is None or event.y is None:
            return

        ax = self.canvas.axes
        bbox = ax.get_window_extent()  # 픽셀 단위 axes 사각형 (event.x/y와 동일 좌표계)
        x0, x1 = self._pan_start_xlim
        y0, y1 = self._pan_start_ylim
        dpx = event.x - self._pan_start_px[0]
        dpy = event.y - self._pan_start_px[1]
        dx = dpx * (x1 - x0) / bbox.width
        dy = dpy * (y1 - y0) / bbox.height
        ax.set_xlim(x0 - dx, x1 - dx)
        ax.set_ylim(y0 - dy, y1 - dy)
        self.canvas.draw_idle()

    def _on_pan_release(self, event):
        """중간 마우스 버튼 뗌 -- 팬 종료."""
        if event.button == 2:
            self._panning = False

    # ------------------------------------------------------------------
    # 플로팅
    # ------------------------------------------------------------------
    def _compute_frame_arrays(self, solver, cache):
        """한 스텝(solver/cache 쌍)에서 (field, verts, face_colors) 구성.

        전체 요소(pid 필터링 전) 기준으로 완전히 벡터화되어 있다 -- 이전
        구현은 `for e in range(solver.n_elem):` Python 루프에서 요소마다
        `MplPolygon(xy, closed=True)` 객체를 새로 만들었는데(3480개 요소면
        매 재조작마다 3480번), 이게 실측상 진짜 병목이었다(matplotlib
        렌더링 자체가 아니라). `coords_deformed[conn]` 팬시 인덱싱 한 번으로
        (n_elem,4,2) 정점 배열을, `vals[conn].mean(axis=1)` 한 번으로
        요소별 색상을 얻는다 -- Python 객체 생성이 전혀 없다.
        """
        field = cache.compute_field(self._current_field)
        vals = field.values
        per_node = field.per_node

        conn = solver.conn                 # (n_elem, 4) node-index array
        coords = solver.coords
        u_all = solver.u
        if self._deformed_scale > 1e-6:
            ux = u_all[0::2]
            uy = u_all[1::2]
            coords_deformed = coords + self._deformed_scale * np.column_stack([ux, uy])
        else:
            coords_deformed = coords

        verts = coords_deformed[conn]      # (n_elem, 4, 2), vectorized
        if per_node:
            face_colors = vals[conn].mean(axis=1)   # (n_elem,), vectorized
        else:
            face_colors = np.asarray(vals, dtype=float)

        return field, verts, face_colors

    def _get_elem_pids(self, solver):
        """요소별 pid 배열, solver 정체성(id()) 기준으로 캐싱.

        pid는 메시 토폴로지에 속한 값이라 필드/스케일이 바뀌어도 절대
        바뀌지 않는다 -- 같은 solver(같은 스텝)에 머무는 한(필드 콤보박스,
        Deformation Scale, Layer 체크박스, Show Element Lines 조작) 이
        dict-lookup 루프를 다시 돌 필요가 없다. 스텝이 바뀌면(`_on_step_changed`
        가 매번 새 `ResultSolverAdapter`를 만듦) id()가 달라지므로 자연히
        재계산된다.
        """
        key = id(solver)
        if self._pids_key != key:
            self._pids_key = key
            self._pids_val = np.array([
                (solver.mesh.elements[eid].pid if solver.mesh.elements[eid].pid is not None else 0)
                for eid in solver.elem_ids
            ], dtype=np.int64)
        return self._pids_val

    def _visible_mask(self, pids: np.ndarray) -> np.ndarray:
        """pid 배열 -> 가시성 boolean 배열, 벡터화(룩업 테이블 인덱싱).

        `self._pid_visible.get(pid, True)`를 요소마다 Python으로 부르는
        대신, pid가 조밀한 작은 정수(1..pid 개수)라는 점을 이용해 룩업
        배열 하나를 만들고 `lut[pids]`로 한 번에 인덱싱한다.
        """
        if pids.size == 0:
            return np.zeros(0, dtype=bool)
        max_pid = int(pids.max())
        lut = np.ones(max_pid + 1, dtype=bool)
        for pid, visible in self._pid_visible.items():
            if 0 <= pid <= max_pid:
                lut[pid] = visible
        return lut[pids]

    def _configure_static_axes(self, ax):
        """축 종횡비/라벨 등, 매번 안 바뀌는 설정 -- fast 경로는 지속 아티스트
        생성 시 1회, overlay 경로는 매 ax.clear() 직후(그래서 두 경로 모두
        이 수정이 자동 적용됨).

        adjustable='box'(기본값)를 쓴다 -- 이전엔 'datalim'이었는데,
        matplotlib은 매 draw마다 Axes.apply_aspect()를 부르고 'datalim'
        분기는 self.dataLim(PolyCollection.set_verts()가 매 프레임 갱신하는
        값) 기준으로 xlim/ylim(viewLim)을 직접 덮어쓴다 -- autoscale을
        꺼도 이 분기는 경고 로그만 남기고 그대로 강행된다(matplotlib
        3.10 axes/_base.py 확인). 그래서 스텝/레이어/필드를 바꿔서
        dataLim이 바뀔 때마다 뷰가 조용히 재조정됐다. 'box' 분기는
        반대로 우리가 관리하는 xlim/ylim 비율에 맞춰 배정된 고정 rect
        안에서 렌더링 박스의 위치/크기만 letterbox 조정 -- viewLim은
        아예 읽지도 쓰지도 않는다.

        set_box_aspect()는 여기서 호출하지 않는다: 호출하는 순간
        matplotlib이 adjustable을 강제로 'datalim'으로 되돌리고(공식
        동작), 그 뒤에 다시 adjustable='box'로 덮어써서 이겨도 'box'
        분기는 self._box_aspect를 아예 참조하지 않으므로 effect가 없다.
        정사각형 렌더링은 _fit_view()/_on_scroll()/_on_pan_motion()이
        항상 x/y span을 동일하게 유지하는 것으로 우리가 직접 보장한다.

        set_autoscale_on(False)는 'box' 분기 자체를 막진 못하지만(그건
        adjustable 전환으로 이미 막힘), 매 draw마다 도는 별도의
        autoscale_view() 경로를 추가로 차단하는 두 번째 안전장치다.
        """
        ax.set_aspect('equal', adjustable='box')
        ax.set_autoscale_on(False)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')

    def _fit_view(self, verts_vis: Optional[np.ndarray] = None):
        """뷰를 현재 표시 중인 요소 전체에 맞춰 리셋 -- 5% 마진, x/y span 동일화.

        matplotlib의 aspect 기계(adjustable='box')는 렌더링 박스 모양만
        맞출 뿐 xlim/ylim 자체를 정사각으로 맞춰주지 않으므로, 정사각 뷰를
        만드는 책임은 전부 여기 있다: 더 큰 쪽 span으로 두 축을 통일하고,
        각 축을 그 데이터의 중점에 맞춰 중앙정렬한다. 최초 렌더링과
        "Fit View" 버튼 양쪽에서 호출.

        Args:
            verts_vis: 이미 계산된 (n_visible_elem, 4, 2) 정점 배열이 있으면
                전달(최초 draw 시 재계산 방지). None이면(예: Fit View 버튼)
                현재 solver/cache/가시성 상태로부터 새로 계산한다.
        """
        if verts_vis is None:
            solver, cache = self._solver, self._cache
            _field, verts, _face_colors = self._compute_frame_arrays(solver, cache)
            pids = self._get_elem_pids(solver)
            mask = self._visible_mask(pids)
            verts_vis = verts[mask]

        if verts_vis.size == 0:
            return  # 표시할 요소가 없으면 현재 뷰를 그대로 둔다

        ax = self.canvas.axes
        v = verts_vis.reshape(-1, 2)
        x_min, y_min = v.min(axis=0)
        x_max, y_max = v.max(axis=0)
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        span = max(x_max - x_min, y_max - y_min, 1.0)
        half = span * 0.5 * 1.05
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        self._view_ready = True

    def _update_plot(self):
        """현재 설정으로 다시 그림 -- overlay_n<=1(기본/가장 흔한 조작)이면
        지속 아티스트 갱신(빠른 경로), N>1(겹쳐 보기, 드물게 사용)이면
        기존처럼 전체 재구성(느린 경로)."""
        overlay_n = self.overlay_spin.value() if self.overlay_spin is not None else 1
        if self._result is not None and overlay_n > 1:
            self._update_plot_overlay(overlay_n)
        else:
            self._update_plot_fast()

    def _update_plot_fast(self):
        """단일 프레임(overlay<=1) 전용 빠른 경로.

        `ax.clear()`도, 요소별 `Polygon`/`PatchCollection`/컬러바 재생성도
        하지 않는다 -- PolyCollection/컬러바/힌지 마커를 한 번만 만들어
        두고(`_fast_artists_valid`), 이후 재조작마다 배열/속성만 갱신한다
        (`set_verts`/`set_array`/`set_clim`/`set_edgecolor`). 뷰(줌/팬)도
        건드리지 않으므로 확대한 채로 Layer 체크박스/필드를 바꿔도 그대로
        유지된다. overlay(N>1) 경로가 직전에 ax.clear()를 했다면(모드
        전환) `_fast_artists_valid`가 꺼져 있어 여기서 다시 만든다.
        """
        ax = self.canvas.axes
        solver, cache = self._solver, self._cache
        field, verts, face_colors = self._compute_frame_arrays(solver, cache)
        pids = self._get_elem_pids(solver)
        mask = self._visible_mask(pids)

        verts_vis = verts[mask]
        colors_vis = face_colors[mask]

        if not self._fast_artists_valid or self._pc is None:
            ax.clear()
            self.canvas.cax.clear()
            import matplotlib as mpl
            self._pc = PolyCollection([], cmap=mpl.cm.get_cmap('viridis'))
            ax.add_collection(self._pc)
            self._configure_static_axes(ax)
            self._hinge_line = ax.plot([], [], marker='^', color='red', markersize=9,
                                        markeredgecolor='k', linestyle='none', zorder=5)[0]
            self._cbar = self.canvas.fig.colorbar(self._pc, cax=self.canvas.cax)
            self._view_ready = False
            self._fast_artists_valid = True

        if verts_vis.size == 0:
            self._pc.set_verts([])
            ax.set_title("No visible elements")
            self.canvas.draw_idle()
            return

        vmin, vmax = float(colors_vis.min()), float(colors_vis.max())
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5

        self._pc.set_verts(verts_vis)
        self._pc.set_array(colors_vis)
        self._pc.set_clim(vmin, vmax)
        self._pc.set_edgecolor('k' if self._show_mesh else 'none')
        self._pc.set_linewidth(0.3 if self._show_mesh else 0.0)

        cb_label = field.name + (f" [{field.unit}]" if field.unit else "")
        self._cbar.set_label(cb_label)
        self._cbar.update_normal(self._pc)

        # 힌지 피벗 마커 -- Result로 열었을 때만(라이브 solver 세션은 RBE2
        # master id 정보가 없음). 매번 새 아티스트를 만들지 않고 좌표만 갱신.
        if self._result is not None:
            master_ids = [int(nid) for nid in
                          (self._result.meta.get("theta_targets_rad") or {}).keys()]
            if master_ids:
                step = self.step_slider.value() if self.step_slider is not None else -1
                hinge_xy = self._result.deformed_points(step)[:, :2]
                idxs = [self._result.node_index(nid) for nid in master_ids]
                self._hinge_line.set_data(hinge_xy[idxs, 0], hinge_xy[idxs, 1])
            else:
                self._hinge_line.set_data([], [])
        else:
            self._hinge_line.set_data([], [])

        # 뷰(줌/팬) -- 이 캔버스의 첫 렌더링일 때만 _fit_view(), 이후로는
        # 절대 xlim/ylim을 건드리지 않아 확대 상태가 유지된다.
        if not self._view_ready:
            self._fit_view(verts_vis)

        ax.set_title(f"{field.name}  (t = {solver.time:.4f} s)")

        self.info_label.setText(
            f"Time: {solver.time:.4f} s\n"
            f"DOFs: {solver.n_dofs}\n"
            f"Elements: {solver.n_elem}\n"
            f"Field: {field.name}"
        )

        self.canvas.draw_idle()

    def _update_plot_overlay(self, overlay_n: int):
        """겹쳐 보기(Overlay Steps N>1) 전용 -- 드물게 쓰이는 모드라 기존처럼
        매번 전체(ax.clear() + 프레임별 PatchCollection) 재구성한다. 빠른
        경로의 지속 아티스트는 다음 단일-프레임 호출 때 다시 만들도록
        무효화해둔다."""
        self._fast_artists_valid = False

        ax = self.canvas.axes
        had_view = getattr(self, "_view_ready", False)
        prev_xlim = ax.get_xlim() if had_view else None
        prev_ylim = ax.get_ylim() if had_view else None
        ax.clear()
        self.canvas.cax.clear()
        self._cbar = None

        current_step = self.step_slider.value() if self.step_slider is not None else self._result.n_steps - 1
        step_indices = sorted(set(
            int(round(x)) for x in np.linspace(0, current_step, overlay_n)
        ))

        frames = []  # list of (field, verts, face_colors, pids, time)
        for step in step_indices:
            from .live_view import ResultSolverAdapter
            solver = ResultSolverAdapter(self._result, step=step)
            cache = _ResultCache(solver)
            field, verts, face_colors = self._compute_frame_arrays(solver, cache)
            pids = self._get_elem_pids(solver)
            mask = self._visible_mask(pids)
            frames.append((field, verts[mask], face_colors[mask], solver.time))

        if not any(v.size for _, v, _, _ in frames):
            ax.set_title("No visible elements")
            self.canvas.draw_idle()
            return

        all_vals = np.concatenate([fc for _, _, fc, _ in frames if fc.size])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        import matplotlib as mpl
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.cm.get_cmap('viridis')

        n_frames = len(frames)
        all_xy = []
        for i, (field, verts, face_colors, _t) in enumerate(frames):
            if not verts.size:
                continue
            alpha = 1.0 if n_frames == 1 else 0.25 + 0.75 * (i / (n_frames - 1))
            pc = PolyCollection(verts, cmap=cmap, norm=norm,
                                edgecolors=('k' if self._show_mesh else 'none'),
                                linewidths=(0.3 if self._show_mesh else 0.0),
                                alpha=alpha)
            pc.set_array(face_colors)
            ax.add_collection(pc)
            all_xy.append(verts.reshape(-1, 2))

        last_field = frames[-1][0]

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb_label = last_field.name
        if last_field.unit:
            cb_label += f" [{last_field.unit}]"
        self._cbar = self.canvas.fig.colorbar(sm, cax=self.canvas.cax, label=cb_label)

        if self._result is not None:
            master_ids = [int(nid) for nid in
                          (self._result.meta.get("theta_targets_rad") or {}).keys()]
            if master_ids:
                last_step = step_indices[-1]
                hinge_xy = self._result.deformed_points(last_step)[:, :2]
                idxs = [self._result.node_index(nid) for nid in master_ids]
                ax.plot(hinge_xy[idxs, 0], hinge_xy[idxs, 1], marker='^',
                       color='red', markersize=9, markeredgecolor='k',
                       linestyle='none', zorder=5)

        if prev_xlim is not None:
            ax.set_xlim(prev_xlim)
            ax.set_ylim(prev_ylim)
        elif all_xy:
            all_xy = np.vstack(all_xy)
            x_min, y_min = all_xy.min(axis=0)
            x_max, y_max = all_xy.max(axis=0)
            margin_x = max(1.0, (x_max - x_min) * 0.05)
            margin_y = max(1.0, (y_max - y_min) * 0.05)
            ax.set_xlim(x_min - margin_x, x_max + margin_x)
            ax.set_ylim(y_min - margin_y, y_max + margin_y)
        self._view_ready = True

        self._configure_static_axes(ax)
        title = f"{last_field.name}  (t = {frames[-1][3]:.4f} s)"
        title += f"  [overlay of {n_frames} steps]"
        ax.set_title(title)

        solver = self._solver
        self.info_label.setText(
            f"Time: {solver.time:.4f} s\n"
            f"DOFs: {solver.n_dofs}\n"
            f"Elements: {solver.n_elem}\n"
            f"Field: {last_field.name}"
        )

        self.canvas.draw_idle()

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


def launch_from_result(result_or_path, step: int = -1):
    """저장된 결과 파일(또는 이미 로드된 Result)로 뷰어를 실행.

    `.pkl` 파일 하나만 있으면 재해석 없이 바로 열 수 있음 -- 5분짜리 폴딩
    해석도 즉시 열림. `material_objects`/`state`가 저장돼 있으면(픽클
    백엔드로 저장한 결과) stress/principal-stress 필드도 그대로 동작; 없으면
    (예: .h5, 또는 이 기능 이전에 저장된 .pkl) disp/strain류만 동작.

    Args:
        result_or_path: `.pkl`/`.h5` 파일 경로, 또는 이미 `load_result()`한
            `Result` 인스턴스.
        step: 처음 열 때 표시할 스텝 (기본값 -1 = 마지막 스텝). 뷰어를 연
            후에는 Step 슬라이더로 아무 스텝이나 오갈 수 있음.

    Returns:
        QApplication.exec()의 종료 코드.
    """
    from .live_view import ResultSolverAdapter
    from .result_io import Result as _Result
    from .result_io import load_result

    result = (result_or_path if isinstance(result_or_path, _Result)
              else load_result(result_or_path))
    if not result.material_objects:
        print(f"[launch_from_result] warning: {result.path!r} has no recorded "
              f"material_objects (saved before this feature existed, or an "
              f".h5 file) -- stress/principal-stress fields will show as 0.")

    from .model_review import print_model_review_from_result
    print_model_review_from_result(result)

    adapter = ResultSolverAdapter(result, step=step)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    viewer = PostprocessViewer(adapter, result=result)
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
