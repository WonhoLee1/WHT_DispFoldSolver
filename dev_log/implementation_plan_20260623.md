# JAX 기반 자가 접촉(Self-Contact) 및 적층 디스플레이 폴딩 해석 구현 계획 (갱신안)

디스플레이 적층 구조(PET 탄소성 및 PSA 점성 초탄성 번갈아 적층)와 회전 구동식 힌지 메커니즘을 지원하기 위해, 기존 기하 구조를 재구축하고 솔버의 멀티 머티리얼(Multi-material) 지원 및 JAX 기반 자가 접촉 구속조건을 구현합니다.

## User Review Required

> [!IMPORTANT]
> **멀티 머티리얼 점탄성 연동**: 기존 `DynamicSolver`는 단일 재료 모델만 염두에 두고 설계되어 있어 점탄성 모델(`ViscoelasticMaterial`) 연동 시 시간 간격(`dt`) 누락 및 상태 텐서 포맷 차이로 인해 런타임 오류가 발생하게 됩니다. 이를 해결하기 위해 솔버 내부의 조립 루프에 `dt` 파라미터 융단 전송 및 `Viscoelastic` 상태 변수 텐서 평탄화 로직을 Surgical하게 이식합니다.

---

## Proposed Changes

### 1. 솔버 다중 재료(Multi-Material) 및 점탄성 연동 확장

#### [MODIFY] [dynamic.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/solver/dynamic.py)
* **MaterialAdapter 개편**:
  - 점탄성 물성(`ViscoelasticMaterial`)인 경우 flat array 형식의 상태 변수 `(6 * (M+1),)`를 내부 텐서 형태 `(M+1, 3, 3)`로 복원하여 `pk2_voigt(F, params, h_prev, dt)`를 올바르게 호출하도록 수정.
  - 반환된 상태 텐서 `h_new`를 다시 flat array로 평탄화하여 솔버에 반환.
  - `tangent_voigt(F, params, dt)` 호출 시 `dt`를 올바르게 전달하도록 보완.
* **DynamicSolver 다중 재료 딕셔너리 지원**:
  - `material`과 `material_params` 인자가 딕셔너리(`dict[int, MaterialModel]`) 형태로 주어지면 `dict[int, MaterialAdapter]`로 감싸서 `self.materials`로 보관.
  - 각 요소의 `pid`에 매핑된 재료 어댑터를 조립 시 동적으로 호출.
  - 내부 변수 상태 벡터 `self.state` 배열의 3번째 축 크기를 `max(materials[pid].n_internal_vars)`로 할당하고, 각 요소 조립 시 재료별 `n_internal_vars` 크기만큼 슬라이싱하여 전달 및 업데이트.
* **조립 루프 `dt` 파라미터 전달**:
  - `_assemble(self, u, dt=None)` 및 `_element_contributions(..., dt)` 시그니처 수정.
  - `dt`가 주어지지 않는 호출(예: reaction forces)에는 `dt = 1e-14` (무한소 시간 증분) 등으로 폴백 제공.

---

### 2. JAX 기반 Penalty Self-Contact 구속조건 추가

#### [NEW] [contact_jax.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/constraint/contact_jax.py)
* **PenaltyContactConstraint 클래스 정의**:
  - 디스플레이 패널의 상부 표면 노드셋(`panel_top`)과 하부 표면 노드셋(`panel_bottom`) 등 자가 접촉 가능 노드셋을 감시.
  - 생성자에서 메쉬 위상 관계를 필터링하여 동일 요소 내 노드가 아닌 노드 쌍들을 접촉 후보군(Contact Pairs)으로 등록.
  - 매 NR 이터레이션마다 JAX 자동 미분을 통해, 활성화된 접촉 절점 쌍(현재 변형 후 거리 $d < d_0$)에 대한 반발력 에너지 $E = \frac{1}{2} k_{contact} \langle d_0 - d \rangle^2$의 로컬 그래디언트(4x1) 및 헤시안(4x4)을 구하고 글로벌 힘과 강성에 가산.

---

### 3. 디스플레이 적층 구조 및 구동식 힌지 메쉬 재구축

#### [MODIFY] [ex03_display_fold.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/examples/ex03_display_fold.py)
* **7층 적층 디스플레이 패널 메쉬 생성**:
  - 디스플레이 크기: $x \in [-40, 40]$ mm, 두께 $y \in [0.0, 0.35]$ mm.
  - y방향으로 7개 요소를 분할하여 7개 층을 구성:
    - Layer 0, 2, 4, 6 (PET): $E = 2000.0$ MPa, $\nu = 0.3$, Yield stress $\sigma_{y0} = 50.0$ MPa (탄소성 J2Plasticity) -> `pid = 0`
    - Layer 1, 3, 5 (PSA): $E = 10.0$ MPa, $\nu = 0.49$, Prony series $g_i=[0.8]$, $\tau_i=[1.0]$ (점성 초탄성 Viscoelastic) -> `pid = 1`
  - 힌지 변형 집중부($x \in [-15, 15]$)는 dx = 0.5 mm 수준으로 촘촘히 분할하고, 외곽 구동부($x < -15$, $x > 15$)는 상대적으로 성기게 분할.
* **구동식 힌지 메커니즘 설정**:
  - 가상의 왼쪽 힌지 노드(Hinge L: $x = -15, y = -5.0$ mm)와 오른쪽 힌지 노드(Hinge R: $x = 15, y = -5.0$ mm)를 메쉬에 추가.
  - Hinge L과 Hinge R의 x, y 변위는 0.0으로 구속 (`bc_dofs`에 구속 추가).
  - $x \le -40$ mm 영역의 하부 패널 절점들을 Hinge L 노드에 RBE2 강체 구속(`RBE2HingeConstraint`)으로 묶음.
  - $x \ge 40$ mm 영역의 하부 패널 절점들을 Hinge R 노드에 RBE2 강체 구속으로 묶음.
  - **회전 강제 구동**:
    - 시간 $t = 0 \to 1.0$ 동안 회전 각도를 $0 \to 90^\circ$ (약 1.5708 rad)로 점진적 인가.
    - Hinge L 추가 자유도(theta_L)에는 시계 방향 회전($-\theta$) 구동.
    - Hinge R 추가 자유도(theta_R)에는 반시계 방향 회전($+\theta$) 구동.

---

## Verification Plan

### Automated Tests
* **점탄성 솔버 연동 테스트**: [test_visco_solver.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/tests/test_visco_solver.py)
  - `ViscoelasticMaterial`과 단일 요소를 사용한 solver.solve_step()이 에러 없이 완벽히 구동하고 점탄성 완화 거동을 보여주는지 검증.
* **접촉 수치적 검증**: [test_contact.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/tests/test_contact.py)
  - `PenaltyContactConstraint`에 대한 Finite Difference Check를 통해 JAX 미분 강성 행렬의 정확성 확인.
* **기존 테스트 검증**: `pytest`를 실행하여 기존 86개 테스트에 회귀 버그가 발생하지 않는지 확인.

### Manual Verification
* **디스플레이 적층 폴딩 해석 실행**: `python examples/ex03_display_fold.py`
  - 180도 폴딩 각도($90^\circ$ 회전)까지 점진적으로 하중이 인가되어 수렴 완주하는지 모니터링.
  - 최종 변형 형상 및 응력을 ParaView(`.vtkhdf`)로 시각화하여 자가 접촉 확인.
