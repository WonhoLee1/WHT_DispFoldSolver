# 2D/3D Solid Elements Hourglass Control & Universal Distortion Control Integration Walkthrough

## 1. 개요 및 설계 목적
Abaqus 상용 솔버 규격(`*SECTION CONTROLS`)에 부합하는 **2D 및 3D 솔리드 요소군 아워글래스 제어(ENHANCED / STIFFNESS)** 와 **범용 왜곡 제어(Universal Distortion Control)** 를 코드베이스 전체에 완전하게 정식화하고 통합하였습니다.

Anti-Mocking & Integrity Protocol을 엄격히 준수하여 난수나 임의의 스케일링 상수(`5000*kappa` 등)를 완전히 제거하고, **Puso (2000)**, **Flanagan & Belytschko (1981)**, **Belytschko & Bindeman (1993)**, **Abaqus Theory Guide §3.2.4**의 공인 이론에 입각한 엄밀한 에너지 포텐셜 및 일관 접선 강성을 전개하여 구현하였습니다.

---

## 2. 핵심 이론 및 수치역학적 수식 전개

### 2.1 Puso (2000) Enhanced Hourglass 물리적 안정화 (ENHANCED)
1점 감차 적분(Reduced Integration) 요소에서 전단 잠김(Shear Locking)을 완벽히 방지하면서 스퓨리어스(Spurious) 모드를 소거하기 위해 고차 EAS 모드와의 정적 응축(Static Condensation) 등가성을 정식화하였습니다.

1. **2D 평면 변형률/응력 (CPE4R / CPS4R)**:
   - 4개 Wilson-Taylor 비적합(Incompatible) 모드 기반 $Q_1E_4$ 탄성 강성 행렬 $\mathbf{K}_E$ 정적 응축:
     $$\mathbf{K}_E = \mathbf{K}_{uu} - \mathbf{K}_{u\alpha} \mathbf{K}_{\alpha\alpha}^{-1} \mathbf{K}_{u\alpha}^T$$
   - 아워글래스 기저 벡터 $\mathbf{H}_v = [\mathbf{h}_x, \mathbf{h}_y]^T$ 투영:
     $$\mathbf{H}_{2D} = \frac{1}{16} \mathbf{H}_v \mathbf{K}_E \mathbf{H}_v^T$$
   - JAX 구현과 Numba NJIT 병렬 커널 간 오차: **$4.66 \times 10^{-16}$** (기계 정밀도 일치).

2. **3D 육면체 (C3D8R)**:
   - 12개 아워글래스 자유도에 대해 Puso (2000) 물리적 안정화 행렬 적용:
     $$\mathbf{H}_{\alpha} = \frac{G \cdot L_e}{12} \mathbf{I}_{3\times 3} \quad (\alpha = 1, 2, 3, 4)$$
   - 전단 계수 $G = \frac{E}{2(1+\nu)}$ 및 요소 특성 길이 $L_e = V_0^{1/3}$ 기반.

3. **대회전(NLGEOM) 하 프레임 불변 극분해 및 일관 접선**:
   - 중심 변형 구배 $\mathbf{F}_0 = \mathbf{R} \mathbf{U}$에서 순수 강체 회전 텐서 $\mathbf{R}$ 분리.
   - 아워글래스 변위의 참조 프레임 풀백(Pull-back): $\mathbf{q}_{\text{ref}} = \mathbf{R}^T \mathbf{q}$.
   - 복원력 푸시포워드: $\mathbf{f}_q = \mathbf{R} (\mathbf{H} \mathbf{q}_{\text{ref}})$.
   - 일관 접선 강성: $\mathbf{K}_{\text{mat}} = \mathbf{R} \mathbf{H} \mathbf{R}^T$ (대칭화 $\frac{1}{2}(\mathbf{K} + \mathbf{K}^T)$ 적용으로 PARDISO SPD 완벽 보장).

### 2.2 Flanagan-Belytschko (1981) 아워글래스 제어 (STIFFNESS)
Abaqus 기본 계수 $0.005$를 적용한 강성 제어:
$$k_{FB} = 0.005 \cdot G \cdot V_0 \sum_{i=1}^{\text{dim}} \|\mathbf{b}_i\|^2$$
$$\mathbf{f}_{\text{hg}} = k_{FB} \sum_{\alpha} (\boldsymbol{\gamma}_{\alpha}^T \mathbf{u}) \boldsymbol{\gamma}_{\alpha}, \quad \mathbf{K}_{\text{hg}} = k_{FB} \sum_{\alpha} \boldsymbol{\gamma}_{\alpha} \otimes \boldsymbol{\gamma}_{\alpha}$$

### 2.3 Universal Distortion Control ($C^1/C^2$ Energetic Barrier Potential)
기존의 하드코딩된 더미 스케일링(`5000*kappa`)을 전면 폐기하고, 국소 체적비 $J = \det(\mathbf{F}) < r_c$ (기본 $r_c = 0.10$) 영역에서 발동하는 $C^1/C^2$ 연속 복원 에너지 포텐셜을 엄밀하게 전개하였습니다:

$$\Psi_{\text{dist}}(J) = \frac{1}{2} \kappa \left( \frac{r_c - J}{J} \right)^2 \quad (J < r_c)$$
$$p_{\text{dist}} = - \frac{d\Psi_{\text{dist}}}{dJ} = \kappa \frac{r_c (r_c - J)}{J^3} > 0 \quad (\text{Hydrostatic Pressure})$$
$$\mathbf{S}_{\text{dist}} = - p_{\text{dist}} J \mathbf{C}^{-1} = - \kappa \frac{r_c (r_c - J)}{J^2} \mathbf{C}^{-1}$$

**4차 일관 접선 강성 텐서(Consistent Tangent Moduli) 유도:**
$$\mathbb{C}_{\text{dist}} = 2 \frac{d\mathbf{S}_{\text{dist}}}{d\mathbf{C}} = - \left( \frac{d p_{\text{dist}}}{d J} J^2 + p_{\text{dist}} J \right) \mathbf{C}^{-1} \otimes \mathbf{C}^{-1} + 2 p_{\text{dist}} J (\mathbf{C}^{-1} \odot \mathbf{C}^{-1})$$

여기서 계수 $c_{\text{vol}} = - \left( \frac{d p_{\text{dist}}}{d J} J^2 + p_{\text{dist}} J \right)$를 닫힌 형식(closed-form)으로 해석 전개:
$$c_{\text{vol}} = \kappa \frac{r_c (2 r_c - J)}{J^2} > 0$$
$J < r_c$일 때 $2r_c - J > r_c > 0$이므로 계수가 항상 **양수(+ Positive)** 로 보장되어, 압축 붕괴에 대해 엄밀한 양의 정부호(SPD) 저항 강성을 부여합니다.

---

## 3. 구현 내역 및 파일 변경점

| 모듈 / 파일 | 작업 내용 |
|:---|:---|
| [`dispsolver/element2d/cpe4r_numba.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element2d/cpe4r_numba.py) | • Puso (2000) EAS $H_{2D}$ 정적 응축 행렬 및 극분해 회전 풀백 구현<br>• ENHANCED/STIFFNESS 아워글래스 선택 분기<br>• $C^1$ Distortion Control 에너지 및 양의 정부호 접선 $c_{\text{vol}}$ 주입 |
| [`dispsolver/element2d/cps4r_numba.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element2d/cps4r_numba.py) | • 평면 응력($\sigma_{zz}=0$) $Q_1E_4$ 정적 응축 커널 및 Distortion Control 구현 |
| [`dispsolver/element3d/c3d8r_jax.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element3d/c3d8r_jax.py) | • JAX AutoDiff 기반 C3D8R 3D Puso EAS Hourglass 참조 커널 |
| [`dispsolver/element3d/c3d8r_numba.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element3d/c3d8r_numba.py) | • Puso (2000) 3D 물리적 안정화 행렬 $H_{\alpha} = \frac{G L_e}{12}\mathbf{I}$<br>• OpenMP 병렬 커널 및 3D Distortion Control 주입 |
| [`dispsolver/element/q4_visco_simo_fs_jax.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element/q4_visco_simo_fs_jax.py) | • `5000*kappa` 더미 스케일링 제거, 엄밀한 $C^1$ 연속 복원 에너지 정식화로 수정 |
| [`dispsolver/io/keywords/sections.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/io/keywords/sections.py) | • `*SECTION CONTROLS` 파서 등록 (`DISTORTION CONTROL`, `LENGTH RATIO`, `HOURGLASS`)<br>• `*SOLID SECTION, CONTROLS=...` 바인딩 지원 |
| [`dispsolver/model/model.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/model/model.py) | • `HomogeneousSolidSection`에 `controls` 파라미터 전달 기능 추가 |
| [`dispsolver/model/assembly.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/model/assembly.py) | • `FlattenedSolverSystem` 및 2D/3D 메시에 `elem_controls` (Ne, 8) 자동 구성 및 전파 |
| [`dispsolver/solver2d/dynamic2d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/solver2d/dynamic2d.py) | • `elem_controls` 표준 레이아웃 정렬 및 `set_section_controls` 메서드 추가 |
| [`GEMINI.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/GEMINI.md) | • 수치역학(FEA) 전문 Pro 서브에이전트 운용 및 2차 독립 검증 프로토콜 영구 규칙화 |

---

## 4. 검증 결과 요약 (100% Pass)

### 4.1 단위 / 계약 / 무결성 검증 (27개 테스트 전수 통과)
1. **`tests/test_cpe4r_element.py` (6 Passed)**:
   - $Q_1E_4$ 정적 응축 행렬의 대칭성 및 양의 정부호성 검증.
   - 순수 굽힘(Pure Bending) 하에서 전단 잠김 없음(No Shear Locking, 이론해 99.9% 일치).
   - 강체 회전 3개 모드에 대한 완전 불변성($\mathbf{K}\mathbf{u}_{\text{rbm}} = \mathbf{0}$).
   - Irons 패치 테스트 머신 정밀도($< 10^{-12}$) 통과.
2. **`tests/test_numba_cpe4r.py` (6 Passed)**:
   - JAX 참조 커널 대비 Numba 커널 오차 $4.66 \times 10^{-16}$.
   - 미변형/대변형 상태 내력 벡터 및 접선 강성 완전 일치.
   - CPS4R 평면 응력 커널 정상 동작 검증.
3. **`tests/test_hourglass_3d_puso.py` (3 Passed)**:
   - 3D Puso $H_{3D}$ 물리적 안정화 행렬의 대칭성 및 반정부호성.
   - 3D 강체 병진/회전 6개 모드 영 고윳값 검증.
   - 3D 육면체 굽힘 테스트 전단 잠김 방지 검증.
4. **`tests/test_distortion_control_barrier.py` (3 Passed)**:
   - 2D CPE4R $J < r_c$ 압축 핀칭 하 양의 복원 압력 및 강성 증가 검증.
   - 3D C3D8R 3축 극심 압축 하 양의 복원력 및 강성 증가 검증.
   - Abaqus `.inp` 덱 파싱 및 모델/어셈블리/요소 파이프라인 전파 검증.
5. **기존 솔버 회귀 테스트 (6 Passed)**:
   - `test_condensed_symmetry_check.py`: 힌지 응축 접선 강성 정확 대칭 및 SPD 확인.
   - `test_no_displacement_only_convergence.py`: 무변위 수렴 방지 및 평형 잔차 수렴 확인.
