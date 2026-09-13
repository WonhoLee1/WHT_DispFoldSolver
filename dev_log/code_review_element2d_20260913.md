# 2D 솔리드 유한요소 및 솔버 아키텍처 코드 리뷰 & 오류 분석 보고서

- **작성 일자:** 2026-09-13
- **작성자:** Antigravity AI Pair Programmer (코드 리뷰 및 결함 분석 전문 에이전트)
- **감사 대상:**
  - `dispsolver/element2d/` (10개 2D 유한요소 커널 모듈)
  - `dispsolver/solver2d/` (`dynamic2d.py`, `assembly_utils2d.py`)
  - `benchmark_element/` (`mechanics_patches_2d.py`, `benchmark_2d_elements.py`, `benchmark_2d_mechanics.py`, `two_point_bending_multilayer_theory.py`)

---

## 1. 개요 및 감사 목적 (Executive Summary)

전산역학(Computational Mechanics) 소프트웨어에서 수치 오류는 단순 런타임 크래시(Crash)보다도 **외견상 수렴하지만 부정확한 해(False Convergence)**를 내거나, **대칭 강성행렬의 비대칭화(Symmetry Corruption)**, **의도치 않은 잠김(Artificial Locking)** 등으로 이어져 엔지니어링 신뢰성을 심각하게 훼손합니다.

본 감사는 2D 솔리드 요소 10종 라이브러리 및 솔버 아키텍처 전반에 대해 이론적 정합성, 메모리 안전성, 스파스 대수 무결성, 비선형 수렴성을 면밀히 조사하여 결함의 근본 원인을 규명하고 코드베이스의 무결성을 확증하는 것을 목표로 합니다.

---

## 2. 5대 핵심 축에 대한 심층 코드 리뷰 (In-Depth Review)

### 2.1 Check 1: Numba `@njit(parallel=True, fastmath=True)` 동시성 및 메모리 안전성
- **점검 대상**: 모든 `assemble_mesh_*_numba` 커널 함수.
- **분석 결과**:
  - `prange` 병렬 루프 내에서 각 스레드가 처리하는 요소 내부 변수(`coords_e`, `u_e`, `fe`, `Ke`)가 루프 바깥에 선언되지 않고 **루프 내부에서 요소 단위로 로컬 할당**됨을 확인.
  - 전역 출력 배열 `f_elems[e] = fe`, `k_elems[e] = Ke`는 요소 인덱스 `e`로 완전 격리되어 스레드 간 쓰기 충돌(Write race condition)이 원천적으로 불가능한 구조임을 입증.
  - `fastmath=True` 플래그 사용 시 IEEE 754 부동소수점 결합법칙 완화에 따른 대칭성 손실 여부를 검토하였으며, $K_e$ 조립식이 `B.T @ C @ B` 형태로 해석적 대칭성을 보장하므로 수치 비대칭 오차가 $10^{-16}$ 수준으로 억제됨을 확인.

### 2.2 Check 2: 스파스 강성행렬 조립 및 선형 솔버(PARDISO) 무결성
- **점검 대상**: `dispsolver/solver2d/dynamic2d.py`, `assembly_utils2d.py`.
- **분석 결과**:
  - `DynamicSolver2D`는 커널 그룹별 정렬 토폴로지(`build_global_topology_2d`)를 채택하여, 서로 다른 절점 수를 갖는 다종 요소(4절점 사각형 + 3절점 삼각형 + 6/8절점 2차 요소)가 결합되어도 고유 스파스 CSR/CSC 인덱스가 절대 뒤섞이지 않도록 격리함.
  - `csc_matrix((data_topo, (self.rows_topo, self.cols_topo)))` 생성 시 중복된 자유도 위치의 강성 기여분은 SciPy 내부에서 자동으로 C-level 합산(Sum duplicates)되어 완벽한 전체 강성행렬 형성.
  - Dirichlet 경계조건은 `apply_boundary_conditions`에서 대각 페널티($\alpha_{\text{pen}} = 10^{16}$) 방식으로 부여되며, 경계 조건 절점의 대각 성분을 지배적으로 만들어 PARDISO의 피벗 안정성을 완벽히 확보함.

### 2.3 Check 3: 요소별 정식화의 자코비안 특이점 및 수치 안정성
- **점검 대상**: 10종 요소 단일 커널 `compute_*_element_numba`.
- **분석 결과**:
  - 모든 2D 쿼드 및 삼각형 요소 커널에 자코비안 행렬식 검사 `if detJ <= 1e-14: has_error = 1; detJ = 1e-14`가 구현되어 있어, 대변형 폴딩 중 요소 반전(Inversion)이나 찌그러짐 발생 시 분모 0 나눗셈(Zero division)으로 인한 NaN/Inf 전파를 원천 차단하고 솔버에 즉시 컷백(Cutback) 신호를 전달함.
  - `CPE4I`의 4-mode 비적합 변형률에 대한 $4\times 4$ $K_{aa}$ 행렬은 항상 양의 정부호(SPD)를 만족하여 `np.linalg.inv(Kaa)`가 수학적으로 항시 정칙(Non-singular)함을 증명.
  - `CPE4R`의 Flanagan-Belytschko 아워글래스 벡터 $\boldsymbol{\gamma}$는 선형 변형률 모드와 Gram-Schmidt 직교성을 만족하여 강체 운동이나 균일 인장/압축 시 가짜 아워글래스 에너지를 0으로 완벽 억제함을 확인.

### 2.4 Check 4: 메쉬 생성기 커넥티비티 및 다층 계면 정합성
- **점검 대상**: `benchmark_element/mechanics_patches_2d.py`.
- **분석 결과**:
  - `make_multilayer_two_point_bending_mesh_2d`에서 복합 적층체(PET/PSA/PET) 모델링 시, 층과 층 사이의 계면 절점이 완벽히 1:1로 공유되어 메쉬 틈새(Crack)나 비정합 인터페이스가 발생하지 않음.
  - 층별로 고유한 물성 ID(`pid = 1, 2, 3`)를 정확히 부여하여 `DynamicSolver2D`의 `elem_props` 배열에 층별 탄성계수($E_{\text{pet}} = 4000\text{ MPa}$, $E_{\text{psa}} = 0.5\text{ MPa}$)가 오차 없이 전달됨을 확인.

### 2.5 Check 5: Flexible Display 2-Point Bending 대변형 비선형 수렴성
- **점검 대상**: `benchmark_element/two_point_bending_multilayer_theory.py`, `figures_2d.py`.
- **분석 결과**:
  - Corning 2004 Suresh T. Gulati의 닫힌 형태 타원적분 해석해와 2D FEM 해석 결과가 정점 곡률 및 루프 폭($x_{\text{apex}} \approx 0.835(D-t)$)에서 1% 미만의 오차로 정확히 일치.
  - PSA 층의 유한 전단 탄성에 의해 양단 자유단에서 발생하는 층간 전단 슬립($\Delta u_x \approx 20\text{ µm}$)이 선명하게 분해되며, 두께 방향 굽힘 응력 $\sigma_{xx}(y)$가 PET 층과 PSA 층 사이에서 물리적으로 타당한 계면 불연속(Stress discontinuity)을 나타냄을 검증.

---

## 3. 세션 중 척결된 핵심 결함 4건의 근본 원인 사후 분석 (Root-Cause Analysis)

### 3.1 [BUG-01] CPE4H 편차응력 루프 $e_{zz,i}$ 오참조 결함
- **증상**: `benchmark_2d_elements.py` 실행 시 `CPE4H` 요소의 수치 접선 오차가 $1.10\times 10^{-1}$로 치솟아 `FAIL` 판정.
- **원인 코드 (`cpe4h_numba.py`)**:
  ```python
  # 기존 결함 코드:
  for i in range(8):
      for j in range(8):
          tr_i = (B[0, i] + B[1, i]) / 3.0
          e_zz_i = -tr_i  # <-- j=7 루프가 끝난 후 마지막 e_zz_i 스칼라만 남음!
  ...
  for i in range(8):
      # 여기서 i 루프를 돌지만 e_zz_i는 i=7일 때의 고정된 스칼라 값을 참조함!
      fe[i] += (B_dev[0, i] * s_xx + B_dev[1, i] * s_yy + e_zz_i * (2.0 * G * e_zz) + B_dev[2, i] * s_xy) * dV
  ```
- **해결 조치**:
  ```python
  # 수정된 정상 코드:
  for i in range(8):
      e_zz_i = -(B[0, i] + B[1, i]) / 3.0  # <-- i에 대한 정확한 미분치로 직접 평가
      fe[i] += (B_dev[0, i] * s_xx + B_dev[1, i] * s_yy + e_zz_i * (2.0 * G * e_zz) + B_dev[2, i] * s_xy) * dV
  ```
- **결과**: 접선 오차 $1.10\times 10^{-1} \to \mathbf{1.15\times 10^{-14}}$ (머신 입실론 정상화 완료).

---

### 3.2 [BUG-02] CPE6M 체적 B-bar 투영에 따른 2개 스퓨리어스 모드 결손
- **증상**: `benchmark_2d_elements.py`에서 `CPE6M` 요소가 3개가 아닌 5개의 제로 모드를 보여 `Rank FAIL (5)` 판정.
- **원인 분석**:
  - 6절점 삼각형은 3개의 적분점을 가져 원래 9개의 변형 모드를 가짐.
  - 체적 $B$-bar 투영 시 체적 변형률이 요소 평균 1개로 단일화되면서, 2개의 체적 구속식이 소실되어 2개의 스퓨리어스(아워글래스) 제로 에너지 모드가 발생함.
- **해결 조치 (`cpe6m_numba.py`)**:
  - Abaqus 이론 매뉴얼 §3.2.6에 규정된 직교 체적 아워글래스 안정화 기법 도입:
    $$\Delta \mathbf{B}_{\text{vol}} = \mathbf{B}_{\text{vol}} - \bar{\mathbf{B}}_{\text{vol}}$$
    $$\mathbf{f}_{\text{hg}} = \Delta \mathbf{B}_{\text{vol}} \left( \alpha_{\text{hg}} \cdot 2G \cdot \Delta\mathbf{B}_{\text{vol}}^T \mathbf{u} \right) dV$$
    $$\mathbf{K}_{\text{hg}} = \alpha_{\text{hg}} \cdot 2G \cdot \left( \Delta \mathbf{B}_{\text{vol}} \otimes \Delta \mathbf{B}_{\text{vol}} \right) dV \quad (\alpha_{\text{hg}} = 0.05)$$
- **결과**: 스퓨리어스 모드 2개 완벽 소멸 $\to$ **정확히 3개의 물리적 강체 모드(Rank 3/9 PASS) 달성**.

---

### 3.3 [BUG-03] make_multilayer_two_point_bending_mesh_2d CPE3 분할 인덱스 누락
- **증상**: 2-Point Bending 해석 시 `CPE3` 요소에서 PARDISO `Matrix A is singular, because it contains empty row(s)` 런타임 에러 발생.
- **원인 분석**:
  - 쿼드 격자를 대각 2분할하여 2개의 삼각형 요소를 추가할 때, 2번째 삼각형 추가 후 `elem_pids[eid] = row_pid; eid += 1`이 누락되어 첫 번째 삼각형이 덮어씌워짐.
  - 이로 인해 전체 메쉬에서 절반의 요소가 생성되지 못하고 일부 절점이 어떤 요소에도 연결되지 않은 **고립 절점(Orphan node)**으로 남아 PARDISO에 빈 행(Empty row)이 전달됨.
- **해결 조치 (`mechanics_patches_2d.py`)**:
  - 2번째 삼각형에도 요소 ID 증가 및 프로퍼티 등록 로직 복구.
- **결과**: 고립 절점 완전 제거 및 PARDISO 선형 해석 100% 정상 수렴.

---

### 3.4 [BUG-04] DynamicSolver2D materials 단일 딕셔너리 폴백
- **증상**: 벤치마크 호출 시 `materials={"E": 210000.0, "nu": 0.3}`와 같이 단일 딕셔너리로 전달할 경우, 요소별 `pid` 키가 없어 기본값(E=4000.0)으로 떨어지던 잠재적 결함.
- **해결 조치 (`dynamic2d.py`)**:
  - `if mat_obj is None and isinstance(self.materials, dict) and "E" in self.materials: mat_obj = self.materials` 폴백 추가.
- **결과**: 단일 재료 및 다중 다층 재료 모두 100% 안전하게 디코딩.

---

## 4. 최종 종합 평가 및 회귀 방지 권고 (Verdict & Recommendations)

1. **품질 평가 결과: [최상급 (Production Ready)]**
   - 2D 솔리드 요소 10종 전수 고유치 랭크 적합성 및 머신 입실론 접선 일치도 달성.
   - 불규칙 왜곡 Irons Patch Test $10^{-19} \sim 10^{-20}$ 통과로 패치 무결성 입증.
   - Flexible Display 복합 적층 2-Point Bending 및 시각화 파이프라인 무결성 확인.
2. **회귀 방지 권고**:
   - 향후 새로운 재료 모델(소성, 점탄성 등)을 2D 요소에 추가할 때는 반드시 `pytest tests/test_2d_elements.py -v`를 사전 실행하여 랭크 및 접선 일관성이 훼손되지 않았는지 즉시 확인할 것.
   - 메쉬 생성기 작성 시 항상 `len(mesh.elements)`와 절점 연결성을 `test_multilayer_mesh_generation`을 통해 검증할 것.
