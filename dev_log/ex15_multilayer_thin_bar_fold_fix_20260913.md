# [Work Log] ex15 Multilayer Thin Bar Fold Convergence & Topology Fix (2026-09-13)

## 1. 개요
- **대상 문제**: `examples/ex15_multilayer_thin_bar_fold.py` 실행 시 미완료/발산 및 극심한 수렴 저하 발생.
- **추가 문의**: `Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.` Matplotlib 경고 발생.

---

## 2. 발견된 결함 및 근본 원인 분석

### 결함 1: 다종 요소(Heterogeneous) 토폴로지 매핑 뒤섞임 (Topology Scrambling)
- **위치**: `dispsolver/model/model.py`, `create_solver3d()`
- **내용**: 
  - `DynamicSolver3D`는 Numba 커널별로 요소를 그룹핑(`k=0, 1, 2...`, 예: PET `C3D8_CR` 전체 후 PSA `C3D8H` 전체)하여 어셈블리함.
  - 그러나 `model.py`에서 `solver.rows_topo = sys.rows_topo`로 무단 덮어쓰기를 수행함.
  - `sys.rows_topo`는 요소 순서(PET $\to$ PSA $\to$ PET $\to$ PSA)로 생성되어 있어, 강성 데이터의 행/열 좌표가 엉뚱한 요소의 자유도와 뒤섞임.
- **영향**:
  - 대각 강성이 정상 물리적 수치인 $1271.25\text{ N/mm}$에서 $0.9967\text{ N/mm}$로 1/1200 토막 붕괴.
  - 2 N의 작은 불평형력에도 변위 증분 $du$가 $5.49\text{ mm}$(두께 0.4mm의 약 14배)로 폭발하여 요소 반전 유발.
- **수정**: `create_solver3d()`에서 `solver.rows_topo = sys.rows_topo` 덮어쓰기 코드 제거. `DynamicSolver3D` 자체의 커널 그룹별 토폴로지 유지.

### 결함 2: Line Search의 요소 반전 감지 및 백트래킹 결함
- **위치**: `dispsolver/solver3d/dynamic3d.py`, `solve_step()` 및 `_solve_step_with_hooks()`
- **내용**: 
  - `best_u = u_k + du`로 초기화되어 있어, 5회의 백트래킹 시도가 요소 반전 등으로 모두 실패했을 때 반전된 최악의 스텝($u_k+du$)이 채택됨.
  - 다음 반복에서 잔차가 $10^{12}$로 폭발하고 비선형 발산.
- **수정**:
  - `best_u = u_k.copy()`로 안전하게 초기화.
  - 유효한(요소 반전이 없고 잔차가 개선된) 스텝이 없을 경우 폭발하는 스텝 대신 `return False, iter_count`로 즉각 Cutback 유도.

### 결함 3: 변위 제어(Dirichlet) 1번째 반복에서의 Line Search 정체(Stall)
- **위치**: `dispsolver/solver3d/dynamic3d.py`
- **내용**:
  - $t=0, u=0$에서 초기 잔차는 기계적 0($10^{-10}$).
  - 1번째 반복에서 강제 회전 변위를 주면 내부 변형으로 인해 잔차가 약 10 N으로 증가하는 것이 당연함.
  - Line Search가 "초기 잔차($10^{-10}$)보다 줄어들어야 한다"고 고집하여 정상적인 변위 부여를 거부하고 정체됨.
- **수정**:
  - 1번째 반복에서는 요소 반전이 없는 한 기구학적 Dirichlet 변위(Full step $s=1.0$)를 정상 수용하도록 처리.

### 결함 4: 물성 파싱 및 기본값 누락
- **위치**: `dispsolver/solver3d/dynamic3d.py`, `_setup_numba_topology()` 및 `dispsolver/model/material.py`
- **내용**:
  - `to_numba_props()`가 재료 속성을 읽지 않고 기본값($E=1.0$)을 내보내던 문제 수정.
  - `_setup_numba_topology()`가 `mat_type` 속성을 인식하지 못해 모든 요소를 강철($E=200,000\text{ MPa}$)로 간주하던 결함 수정.

### 결함 5: Matplotlib 경고
- **위치**: `examples/ex15_visualize_multilayer.py`
- **내용**: `ax.axis('equal')`과 `ax.set_ylim(-15, 15)` 동시 사용 충돌.
- **수정**: `ax.set_aspect('equal', adjustable='box')`로 변경 완료.

---

## 3. 검증 결과
- **단일 스텝 (Inc 1)**: 단 4 반복만에 잔차 $3 \times 10^{-7}$로 완벽한 이차 수렴(Quadratic convergence).
- **전체 폴딩 시뮬레이션**: 180° 전체 대변형 폴딩 순항 중.
