# Connector Elements — Slider/Hinge Joint with Local Coordinate System

## TL;DR

> **Quick Summary**: 로컬 좌표계 기반 Connector 요소군 구현 — 두 독립 바디 사이의 상대 운동(Hinge 회전, Slider 슬라이드)을 로컬 좌표계에서 정의하고 prescribed motion을 가할 수 있는 구속조건 시스템
>
> **Deliverables**:
> - `dispsolver/constraint/connector.py` — ConnectorConstraint ABC + LocalCoordinateSystem
> - `dispsolver/constraint/hinge_joint.py` — HingeJointConstraint (상대 회전 θ)
> - `dispsolver/constraint/slider_joint.py` — SliderJointConstraint (상대 변위 d)
> - 테스트 10개 이상
> - ex03에서의 활용 업데이트
>
> **Estimated Effort**: Medium (4~6일)
> **Parallel Execution**: YES - 3 waves

---

## Context

### Original Request
두 독립 바디가 힌지 또는 슬라이더로 연결된 구조에서, **로컬 좌표계**를 기준으로 한 **상대 운동**을 정의하고 prescribed motion을 가할 수 있는 Connector 요소 구현.

### Current Constraint System
| 구속 | 방식 | 설명 |
|------|------|------|
| `RBE2HingeConstraint` | Lagrange Multiplier | 단일 바디 강체 회전 (extra primal: θ) |
| `PenaltyHingeConstraint` | Penalty | node-to-node 변위 동기화 |
| `BaseConstraint` (ABC) | - | `assemble(u, u_extra)→(C_u, C_extra, g)` |

### Gap Analysis
- **로컬 좌표계 개념 없음** → 모든 구속이 전역 X/Y에서 정의됨
- **두 독립 바디 간 구속 없음** → 현재는 master-slave 단일 바디만
- **상대 운동 prescribed 기능 없음** → d(t) 또는 θ(t)를 직접 가할 수 없음

---

## Architecture

```
BaseConstraint (ABC)
└── ConnectorConstraint (ABC)          ← 로컬좌표계 + 상대운동 프레임워크
    ├── HingeJointConstraint           ← 상대 회전 θ (revolute)
    └── SliderJointConstraint          ← 상대 변위 d (prismatic)
```

### ConnectorConstraint 공통 설계

**두 노드 집합 간 상대 운동**을 로컬 좌표계에서 정의:

```
        LCS {origin, θ₀}
     Body A ●═══════════● Body B
              Δu_local

Δu_local = R(θ₀) @ (u_B_global - u_A_global)

Constraint equations (로컬 좌표계):
  자유 방향 (e₁):  Δu_local[0] = prescribed_d   (슬라이더)
                   Δu_local[1] = prescribed_θ   (힌지 — 회전각)
  구속 방향 (e₂):  Δu_local[1] = 0 (슬라이더)
                   u_hinge_A - u_hinge_B = 0 (힌지)
```

**Extra Primal DOF 추가**:
- Hinge: 1 (θ — 상대 회전각)
- Slider: 1 (d — 상대 변위)

**Lagrange Multiplier**:
- 각 연결노드쌍 × 2 (글로벌 DOF 기준 구속)

---

## Work Objectives

### Core Objective
로컬 좌표계 기반 ConnectorConstraint 프레임워크 구축 + HingeJoint/SliderJoint 구현

### Concrete Deliverables
- `dispsolver/constraint/lcs.py` — `LocalCoordinateSystem` (회전행렬, 변환)
- `dispsolver/constraint/connector.py` — `ConnectorConstraint` ABC
- `dispsolver/constraint/hinge_joint.py` — `HingeJointConstraint`
- `dispsolver/constraint/slider_joint.py` — `SliderJointConstraint`
- `tests/test_connector.py` — 10개 이상 단위 테스트
- `constraint/__init__.py` 업데이트

### Must Have
- `LocalCoordinateSystem(origin, theta)` — `R` 속성, `global_to_local()`, `local_to_global()`
- `HingeJointConstraint(mesh, node_set_A, node_set_B, lcs)` — 상대회전 θ, prescribed motion 가능
- `SliderJointConstraint(mesh, node_set_A, node_set_B, lcs)` — 상대변위 d, prescribed motion 가능
- 두 바디 각각 독립 변형 + 조인트에서만 연결
- prescribed d(t) 또는 θ(t) → u_extra 초기값으로 전달
- BaseConstraint assemble() 인터페이스 완전 호환

### Must NOT Have
- **3D 확장 미포함** — 2D 전용 (z축 회전만)
- **비선형 LCS 미포함** — follower force 효과는 향후 과제 (LCS = 초기 형상 기준)
- **마찰/댐퍼 미포함** — 순수 구속조건 only
- **Abaqus ConnectorSection 매핑 미포함** — `.inp` 파서 연동은 별도 플랜

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest, 95 tests)
- **Automated tests**: YES (TDD — test-first)
- **Framework**: pytest

### QA Policy
Agent-executed QA scenarios with evidence capture.

---

## Execution Strategy

```
Wave 1 (Foundation):
├── Task 1: LocalCoordinateSystem + 단위 테스트 [quick]
├── Task 2: ConnectorConstraint ABC + 기본 assemble 구조 [unspecified-high]
└── Task 3: 기존 constraint/__init__.py 정리 [quick]

Wave 2 (구현 — 병렬 가능):
├── Task 4: HingeJointConstraint — full assemble + prescribed θ [deep]
├── Task 5: SliderJointConstraint — full assemble + prescribed d [deep]
└── Task 6: DynamicSolver 연동 — u_extra prescribed motion 전달 [deep]

Wave 3 (검증):
├── Task 7: 단위 테스트 — HingeJoint (4개) [unspecified-high]
├── Task 8: 단위 테스트 — SliderJoint (4개) [unspecified-high]
└── Task 9: 통합 테스트 — 2-body hinge/slider 왕복검증 [unspecified-high]

Wave FINAL:
└── F1-F4: 계획/품질/QA/범위 검증
```

---

## TODOs

- [ ] 1. **LocalCoordinateSystem** — `dispsolver/constraint/lcs.py`

  **What to do**:
  ```python
  @dataclass
  class LocalCoordinateSystem:
      origin: np.ndarray  # (2,) 기준점
      theta: float        # 전역 X 기준 회전각 [rad]

      @property
      def R(self) -> np.ndarray:
          """회전행렬: u_local = R @ u_global
          e₁ = slide direction (cos θ, sin θ)
          e₂ = perpendicular  (-sin θ, cos θ)
          """
          ...

      def global_to_local(self, u_global: np.ndarray) -> np.ndarray: ...
      def local_to_global(self, u_local: np.ndarray) -> np.ndarray: ...
  ```
  - 2x2 rotation matrix R: [[cos θ, sin θ], [-sin θ, cos θ]]
  - origin은 향후 follower LCS를 위해 보관 (현재는 미사용)
  - numpy-only, no JAX (host-side coordinate transform)

  **Must NOT do**:
  - 시간 의존성 넣지 않음 (θ₀ = 상수)
  - JAX 변환 넣지 않음 (host-side operation)

  **Recommended Agent Profile**: `quick`

  **Parallelization**: Blocks 2, 4, 5 | Blocked By: None

  **Acceptance Criteria**:
  - [ ] `LCS(theta=30°) → R @ [1,0] = [0.866, -0.5]` (30° 회전 확인)
  - [ ] `LCS.local_to_global(LCS.global_to_local(u)) == u` (왕복 검증)

  **Commit**: YES — `feat(constraint): add LocalCoordinateSystem`

- [ ] 2. **ConnectorConstraint ABC** — `dispsolver/constraint/connector.py`

  **What to do**:
  - `ConnectorConstraint(BaseConstraint)` ABC:
    ```python
    class ConnectorConstraint(BaseConstraint):
        def __init__(self, mesh, node_set_A: List[int], node_set_B: List[int],
                     lcs: LocalCoordinateSystem):
            ...

        def n_multipliers(self) -> int:
            return 2 * len(self.node_pairs)  # 각 쌍 x, y 구속

        def n_extra_primal(self) -> int:
            return 1  # 상대운동 DOF (θ 또는 d)

        @abstractmethod
        def _relative_motion(self, u_A_local, u_B_local, u_extra) -> Tuple[float, float]:
            """로컬 좌표계에서의 상대 운동량 반환.
              return (Δu_free, Δu_fixed)"""
            pass

        def _local_stiffness_contribution(self, ...):
            """로컬 좌표계 강성 기여 — subclass에서 override 가능"""
            ...

        def assemble(self, u, u_extra):
            """BaseConstraint.assemble() 구현:
            1. body_A 노드, body_B 노드 각각 변위 수집
            2. 로컬 좌표계로 변환: u_local = R @ u_global
            3. 상대 운동 Δu_local 계산
            4. _relative_motion() 호출 → free/fixed 성분
            5. C 행렬 + g 벡터 조립 (글로벌 DOF 기준)
            """
            ...
    ```

  **Must NOT do**:
  - 구체적인 상대운동 로직 구현하지 않음 — subclass 책임
  - penalty 방식 사용하지 않음 — 순수 LM

  **Recommended Agent Profile**: `unspecified-high`

  **Parallelization**: Blocked By: 1 | Blocks: 4, 5

  **Acceptance Criteria**:
  - [ ] `ConnectorConstraint` 인스턴스 생성 후 `isinstance(c, BaseConstraint) == True`
  - [ ] `n_multipliers() == 2 * len(node_pairs)`
  - [ ] `n_extra_primal() == 1`

  **Commit**: YES — `feat(constraint): add ConnectorConstraint ABC`

- [ ] 3. **HingeJointConstraint** — `dispsolver/constraint/hinge_joint.py`

  **What to do**:
  - `HingeJointConstraint(ConnectorConstraint)`:
    - 두 바디가 힌지점에서 만남 (coincident node pairs)
    - 구속: `u_B_local - u_A_local = 0` (변위 일치, 같은 점에서)
    - 자유: `θ` — 상대 회전 (extra primal DOF)
    - prescribed θ(t) 지원:
      ```python
      def set_prescribed_theta(self, theta_func: Callable[[float], float]):
          """θ(t) 설정. solve_step(t)에서 호출되어 u_extra 초기값 결정."""
          self._prescribed_fn = theta_func

      def get_prescribed_theta(self, t: float) -> float:
          return self._prescribed_fn(t) if self._prescribed_fn else 0.0
      ```
    - assemble() 상세:
      - 각 노드쌍 (A_i, B_i)에 대해:
      - Δu_global = u_B - u_A
      - Δu_local = R @ Δu_global
      - C_u[eq, dof_A] = -R[0,:]  (e₁ 방향 자유 — θ로 대체)
      - C_u[eq, dof_B] = +R[0,:]
      - C_extra[eq, θ] = -1 (회전 구속)
      - C_u[eq+1, dof_A] = -R[1,:]  (e₂ 방향 구속)
      - C_u[eq+1, dof_B] = +R[1,:]
      - g = [prescribed_θ - Δu_local[0], 0 - Δu_local[1]]

  **Must NOT do**:
  - RBE2처럼 강체 회전 가정하지 않음 — 각 바디의 변형 자유도 완전 보존

  **Recommended Agent Profile**: `deep`

  **Parallelization**: Blocked By: 1, 2 | Blocks: 7

  **Acceptance Criteria**:
  - [ ] 두 독립 Q4 요소, 힌지로 연결 → θ prescribed 시, 각 요소가 독립적으로 변형
  - [ ] θ=0에서 두 바디 변위 일치 (hinge locking)
  - [ ] prescribed_θ(t)=π/2*t → 90° 회전 시 힌지점 위치 연속성 유지

  **QA Scenarios**:
  ```
  Scenario: Hinge — prescribed 회전
    Tool: Bash (python test)
    Steps:
      1. 두 개 Q4 요소, 힌지 연결
      2. θ(t)=π/2 * t prescribed
      3. solve_step(1.0) 실행
    Expected: 힌지점 두 노드 변위 차이 < 1e-10
    Evidence: .omo/evidence/task-3-hinge-prescribed.txt
  ```

  **Commit**: YES — `feat(constraint): add HingeJointConstraint with prescribed θ`

- [ ] 4. **SliderJointConstraint** — `dispsolver/constraint/slider_joint.py`

  **What to do**:
  - `SliderJointConstraint(ConnectorConstraint)`:
    - 두 바디가 슬라이드 축(e₁)을 따라 상대 운동
    - 구속: `Δu_local[1] = 0` (e₂ 방향 — 축 직각)
    - 자유: `Δu_local[0] = d` (e₁ 방향 — 슬라이드량, extra primal DOF)
    - prescribed d(t) 지원:
      ```python
      def set_prescribed_displacement(self, d_func: Callable[[float], float]):
          self._prescribed_fn = d_func
      ```
    - assemble() 상세:
      - 각 노드쌍 (A_i, B_i)에 대해:
      - Δu_global = u_B - u_A
      - Δu_local = R @ Δu_global
      - C_u[eq, dof_A] = -R[1,:]  (e₂ 방향만 구속)
      - C_u[eq, dof_B] = +R[1,:]
      - C_extra 없음 (e₁ 자유 — d는 extra primal로 관리)
      - g[eq] = 0 - Δu_local[1]
      - extra primal d: f_int[dof_d] += k_d * (Δu_local[0] - d)
      - K[dof_d, dof_d] += k_d
    - k_d: 약한 penalty로 d가 extra DOF로 따라가게 함

  **Must NOT do**:
  - DOF 1,2가 아닌 로컬 e₁, e₂ 기준 — 전역 X/Y 변환 주의

  **Recommended Agent Profile**: `deep`

  **Parallelization**: Blocked By: 1, 2 | Blocks: 8

  **Acceptance Criteria**:
  - [ ] 두 독립 Q4, θ₀=0° → e₁=X방향 슬라이드, e₂=Y방향 고정
  - [ ] d=0.5 prescribed → 슬라이드 방향으로만 0.5mm 상대 변위
  - [ ] θ₀=45° → 대각선 슬라이드 (X,Y 동시 변위)
  - [ ] 슬라이드 직각 방향 변위차 < 1e-10

  **Commit**: YES — `feat(constraint): add SliderJointConstraint with prescribed d`

- [ ] 5. **Solver 연동: prescribed motion 전달** — `dispsolver/solver/dynamic.py`

  **What to do**:
  - `DynamicSolver`에 ConnectorConstraint의 prescribed motion을 `u_extra` 초기값에 반영:
    ```python
    def solve_step(self, t, dt):
        # 각 connector constraint의 prescribed motion 조회
        for c in self.constraints:
            if isinstance(c, ConnectorConstraint):
                c.update_prescribed(t)
        # 초기 u_extra 설정 (prescribed 기반)
        ...
    ```
  - 수정 사항 최소화 — `BaseConstraint`에 `update_prescribed(t)` optional method 추가
    ```python
    class BaseConstraint(ABC):
        def update_prescribed(self, t: float):
            """Optional: prescribed motion 업데이트 (connector 전용)."""
            pass
    ```

  **Must NOT do**:
  - 기존 RBE2 동작 변경하지 않음 — update_prescribed는 optional
  - 솔버 구조 대규모 변경하지 않음

  **Recommended Agent Profile**: `deep`

  **Parallelization**: Blocked By: 3, 4 | Blocks: 7, 8, 9

  **Acceptance Criteria**:
  - [ ] `solver.solve_step(t=1.0, dt=1.0)`에서 HingeJoint θ(t) 호출 확인
  - [ ] 기존 RBE2 테스트 전부 PASS 유지

  **Commit**: YES (groups with 3, 4) — `feat(solver): add prescribed motion support for connector constraints`

- [ ] 6. **단위 테스트 — HingeJoint** — `tests/test_connector.py`

  **What to do**:
  - 최소 4개 테스트:
    1. `test_hinge_assemble_structure` — assemble() 호출 후 C matrix 크기/형상 확인
    2. `test_hinge_prescribed_theta` — θ prescribed → 힌지점 변위 연속성 검증
    3. `test_hinge_independent_deformation` — 각 바디 독립 변형 확인 (다른 재료)
    4. `test_hinge_zero_theta_locking` — θ=0 → rigid connection (변위차 < eps)

  **Recommended Agent Profile**: `unspecified-high`

  **Parallelization**: Blocked By: 3, 5 | Blocks: 9

  **Acceptance Criteria**: `pytest tests/test_connector.py::test_hinge* -v → 4/4 PASS`

  **Commit**: YES (groups with 8) — `test(constraint): add hinge joint tests`

- [ ] 7. **단위 테스트 — SliderJoint** — `tests/test_connector.py`

  **What to do**:
  - 최소 4개 테스트:
    1. `test_slider_assemble_structure` — assemble() C matrix 검증
    2. `test_slider_prescribed_d` — d prescribed → 슬라이드 방향 변위차 확인
    3. `test_slider_rotated_lcs` — θ₀=45° → 대각선 슬라이드 검증
    4. `test_slider_perp_locked` — 슬라이드 직각 방향 변위차 < 1e-10

  **Recommended Agent Profile**: `unspecified-high`

  **Parallelization**: Blocked By: 4, 5 | Blocks: 9

  **Acceptance Criteria**: `pytest tests/test_connector.py::test_slider* -v → 4/4 PASS`

  **Commit**: YES (groups with 6) — `test(constraint): add slider joint tests`

- [ ] 8. **통합 테스트 + 예제** — `tests/test_connector.py` + `examples/`

  **What to do**:
  - 2-body 통합 테스트:
    - 왼쪽 블록(Q4, 강성) + 오른쪽 블록(Q4, 연성) = HingeJoint
    - θ(t) prescribed → 힌지 회전 + 각 바디 독립 변형
    - 변위 콘투어 확인 (한쪽은 강체처럼, 다른 쪽은 크게 변형)
  - `ex03_display_fold.py` 업데이트 옵션:
    - RBE2 대신 HingeJoint 사용하는 variant 추가
  - 기존 95개 회귀 테스트 유지 확인

  **Recommended Agent Profile**: `unspecified-high`

  **Parallelization**: Blocked By: 6, 7 | Blocks: F1-F4

  **Acceptance Criteria**:
  - [ ] 2-body hinge 통합 테스트 PASS
  - [ ] `pytest tests/` — 100% (기존 + 신규)

  **Commit**: YES — `test(constraint): add 2-body integration test for connector elements`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify: Must Have 모두 구현, Must NOT Have 침범 없음, BaseConstraint 호환성 유지

- [ ] F2. **Code Quality Review** — `unspecified-high`
  `pytest tests/` 전부 PASS. Check: LCS가 numpy-only, JAX leakage 없음

- [ ] F3. **Real Manual QA** — `unspecified-high`
  모든 QA 시나리오 실행. 2-body hinge θ prescribed 테스트 시각 확인

- [ ] F4. **Scope Fidelity Check** — `deep`
  3D 확장/마찰/댐퍼/비선형 LCS 포함되지 않았는지 확인

---

## Commit Strategy

- **1**: `feat(constraint): add LocalCoordinateSystem`
- **2**: `feat(constraint): add ConnectorConstraint ABC`
- **3~5**: `feat(constraint): add HingeJointConstraint + SliderJointConstraint + solver prescribed motion`
- **6~8**: `test(constraint): add connector element tests + integration`

---

## Success Criteria

### Verification Commands
```bash
pytest tests/test_connector.py -v  # 8+ PASS
pytest tests/  # 103+ PASS (기존 95 + 신규 8)
```

### Final Checklist
- [ ] `LocalCoordinateSystem` — R, global_to_local, local_to_global 모두 정상
- [ ] `HingeJointConstraint` — θ prescribed + 각 바디 독립 변형
- [ ] `SliderJointConstraint` — d prescribed + 임의 LCS θ₀
- [ ] 기존 95개 테스트 회귀 없음
- [ ] BaseConstraint 호환성 유지
