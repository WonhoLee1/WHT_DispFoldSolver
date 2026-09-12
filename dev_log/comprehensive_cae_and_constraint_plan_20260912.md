# Commercial CAE Comprehensive Implementation Plan: Multi-Step Branching, SectionControls, Constraints & Loads (2026-09-12)

## 1. 개요 및 목적 (Overview & Goals)

본 계획은 이전 세션에서 도출된 핵심 구조적 요구사항들과 금일 논의된 구속/하중 조건을 유기적으로 통합하여, **완전한 상용 CAE(Abaqus / ANSYS / LS-DYNA) 급의 Python API 및 Numba 가속 솔버 아키텍처**를 완성하는 마스터 구현 로드맵입니다.

### 통합 핵심 영역 (Integrated Pillars):
1. **스마트 세트 및 기하학적 영역 선택 (`GeneralSet` & `findAt`)** [기구현 완료 & 27/27 PASS]
2. **구속 조건 및 강체/커플링 (`Constraint` 모듈)**:
   - `RigidBody`: 기준점(RP) 기반 완전 강체 구속 (Kinematic Master-Slave Condensation / Penalty)
   - `Tie`: 마스터-슬레이브 접합 (표면/절점 일체화)
   - `Coupling`: `KinematicCoupling` (RBE2 강체화) vs `DistributingCoupling` (RBE3 가중 평균, 변형 허용)
   - `MPC`: 다점 기구학 방정식 (`PIN`, `BEAM`, `LINK`, `EQUATION`)
3. **하중 시스템 (`Load` 모듈)**:
   - `ConcentratedForce` (`Cload`): 절점 집중 하중 및 모멘트 (진폭 `Amplitude` 연동)
   - `Pressure`: 2D 에지 및 3D 페이스 표면 압력 하중 ($\int N^T P \, dA$)
   - `Gravity` / `BodyForce`: 전역 가속도 및 밀도 기반 체적 중력 하중 ($\int \rho \mathbf{g} N \, dV$)
4. **멀티 스텝 및 가지치기 분기 해석 엔진 (`MultiStepExecutor`)**:
   - `InitialStep`, `StaticStep`, `DynamicStep`의 부모-자식 트리 구조 (`previous` 링크)
   - 엔티티 생명주기 관리: `CREATED`, `PROPAGATED`, `MODIFIED`, `DEACTIVATED`, `REACTIVATED`
   - `PredefinedField`: 초기 응력, 초기 변형률, 초기 속도, 초기 SDVs
   - `StateCheckpoint` 기반의 가지치기(Branching) 해석 (예: Step 1 성형 후 Branch A 스프링백, Branch B 압착 충격 분기)
5. **요소 제어 및 반전 방지 보호막 (`SectionControls` & Inversion Barrier)**:
   - `distortionControl` (길이비 $r < 0.1$ 시 에너지 장벽 복원력 가동)
   - `antiInversionBarrier` ($J \to 0^+$ 극한 발산 잠재력)
   - 뉴턴-랩슨 라인서치 Inversion Guard (체적 음수화 시 $\alpha$ 자동 스텝 축소)

```
                       Model ("Model-1")
 ┌─────────────────────────────┼─────────────────────────────┐
 │                             │                             │
Part (Local Coords)         Assembly                      Material / Section
├── Geometry & Mesh         ├── Instances (Transform3D)   ├── Elastic / Plastic
├── GeneralSets (Smart)     ├── Constraints               └── SectionControls
│   ├── BoundingBox / Sphere│   ├── RigidBody (RP ↔ Set)      ├── distortionControl
│   ├── Cylinder / Normal   │   ├── Tie (Master ↔ Slave)      ├── antiInversionBarrier
│   └── findAt (Proximity)  │   ├── Coupling (Kin/Dist)       └── viscousDamping
│                           │   └── MPC (PIN, BEAM, EQUATION)
│                           └── FlattenedSolverSystem (CSR)
│
└──────────────────────────────┬─────────────────────────────┘
                               │
                      Multi-Step Tree
          ┌────────────────────┴────────────────────┐
     InitialStep (t=0)                         Step-1 ("Folding")
     ├── PredefinedFields                      ├── BCs: [CREATED]
     │   (Stress, Velocity, SDV)               ├── Loads: [CREATED] (Cload, Pressure)
     └── Base Constraints                      └── StateCheckpoint-1 (u, v, a, SDV)
                                                            │
                               ┌────────────────────────────┴───────────────────────────┐
                      Branch A: Step-2A ("Springback")                        Branch B: Step-2B ("Impact")
                      ├── BCs: [MODIFIED / DEACTIVATED]                       ├── BCs: [PROPAGATED]
                      └── Loads: [DEACTIVATED]                                └── Loads: [MODIFIED (High Load)]
```

---

## 2. 세부 컴포넌트 설계 (Detailed Component Design)

### Phase 1: 구속 조건 모듈 (`dispsolver/model/constraint.py` & Numba 가속)

#### 1. `RigidBody` (강체 구속)
- **수학적 모델**:
  $$\mathbf{u}_s = \mathbf{u}_{RP} + \boldsymbol{\theta}_{RP} \times (\mathbf{X}_s - \mathbf{X}_{RP})$$
  - 3D 솔리드 요소의 경우 절점에 회전 DOF가 없으므로, RP(6 DOF)의 병진 및 회전이 슬레이브 절점의 3축 병진 변위($u_x, u_y, u_z$)로 정확히 사영됨.
- **클래스 정의**:
  ```python
  @dataclass
  class RigidBody(Constraint):
      name: str
      ref_point: Union[str, int, GeneralSet]  # Reference Point Node ID or Set
      tie_nodes: Union[str, GeneralSet]       # Slaved node region
      pin_nodes: Optional[Union[str, GeneralSet]] = None
      is_analytic: bool = False
  ```

#### 2. `Coupling` (KinematicCoupling & DistributingCoupling)
- **`KinematicCoupling`**:
  - 선택된 DOF 목록(`dofs=[1, 2, 3]`)에 대해 RP와 슬레이브 절점 간 강체 구속 체결.
- **`DistributingCoupling`**:
  - 가중치 계수 $w_i = \frac{1}{N}$ (또는 면적 비례 가중치)를 적용하여 하중 분산 및 평균 변위 추종:
    $$\mathbf{u}_{RP} = \sum_{i=1}^N w_i \mathbf{u}_i, \quad \mathbf{f}_i = w_i \mathbf{F}_{RP}$$
  - 슬레이브 영역 자체의 팽창, 수축, 굽힘 변형을 100% 보존.

#### 3. `MPC` (Multi-Point Constraint)
- **타입**: `PIN` (절점 간 상대 변위 0, 상대 회전 허용), `BEAM` (완전 일체화), `LINK` (거리 유지), `EQUATION` (임의 계수식 $\sum A_i u_i = 0$).

---

### Phase 2: 하중 시스템 (`dispsolver/model/load.py` & Numba 가속)

#### 1. `ConcentratedForce` (`Cload`)
- 절점 세트의 지정된 자유도(`dof=1, 2, 3`)에 크기 및 `Amplitude`를 부여하여 글로벌 우변 외력 벡터 $\mathbf{F}_{\text{ext}}(t)$에 직접 주입.
- Numba 가속 커널을 통해 매 이터레이션/인크리먼트마다 $O(1)$ 초고속 벡터 가산.

#### 2. `Pressure` (표면 압력 하중)
- 3D Hex8 요소의 외곽 쿼드 페이스(4개 절점) 또는 2D Quad4 요소의 외곽 선분(2개 절점)에 균일 또는 비균일 압력 $P$ 인가.
- 등가 절점 하중 계산:
  $$\mathbf{f}_e = \int_{-1}^1 \int_{-1}^1 \mathbf{N}^T P \, \mathbf{n} \, \det(J_s) \, d\xi d\eta$$
- `GeneralSet.get_faces(exterior_only=True)`와 직접 연동되어, 사용자가 면 번호를 몰라도 면 세트만 지정하면 자동 적분.

#### 3. `Gravity` / `BodyForce` (체적력)
- 재료 밀도 $\rho$와 전체 볼륨의 형상함수 적분을 통해 중력 가속도 $\mathbf{g} = (g_x, g_y, g_z)$에 따른 절점 하중 산출:
  $$\mathbf{f}_{\text{body}} = \int_V \rho \mathbf{N}^T \mathbf{g} \, dV$$

---

### Phase 3: 멀티 스텝 및 가지치기 엔진 (`StepState` & `MultiStepExecutor`)

#### 1. 엔티티 생명주기 제어 (Entity Lifecycle State Machine)
- 모든 BC, Load, Constraint는 스텝 전환 시 다음 상태 전이를 거침:
  - `CREATED`: 생성된 스텝에서 유효화.
  - `PROPAGATED`: 다음 스텝으로 자동 상속.
  - `MODIFIED`: 크기, 진폭(`amplitude`), 위치 등 수정.
  - `DEACTIVATED`: 해당 스텝에서 하중/구속 해제 (예: 프레스 해제, 스프링백).
  - `REACTIVATED`: 이전 스텝에서 꺼졌던 조건 재가동.

#### 2. `StateCheckpoint` & 브랜칭 엔진
- 스텝 완료 시 `StateCheckpoint` 객체 생성:
  - 변위(`u`), 속도(`v`), 가속도(`a`), 재료 내부 상태변수(`SDVs`), 시간(`t_accum`) 스냅샷 보존.
- 자식 스텝 분기 시 부모 체크포인트의 상태를 완전 롤백 복원하여 독립된 분기 해석(`Branch A`, `Branch B`) 병렬/순차 수행.

---

### Phase 4: 요소 제어 및 반전 방지 보호막 (`SectionControls` & Inversion Barrier)

#### 1. `SectionControls` 정의 및 할당
- `SolidSection(..., controls='Control-1')`
- `SectionControls(name='Control-1', distortionControl=True, lengthRatio=0.1, antiInversionBarrier=True)`

#### 2. Numba 요소 커널 내 반전 방지 에너지 장벽
- $J = \det(F)$가 한계치 $J_{\text{crit}} = 0.1$ 이하로 진입 시, 급격한 에너지 장벽 잠재력 $\Psi_{\text{dist}}(J) = \frac{1}{2} k_{\text{dist}} \left( \frac{J_{\text{crit}} - J}{J} \right)^2$이 발동하여 $J \le 0$으로의 붕괴를 물리적/수치적으로 원천 차단.

#### 3. 솔버 라인서치 Inversion Guard
- 뉴턴-랩슨 이터레이션의 Trial 변위 $u + \alpha \Delta u$ 계산 시, 최소 $\det(F)$가 $0.02$ 미만으로 떨어지면 라인서치 스텝 $\alpha$를 $0.5 \alpha$로 자동 후퇴(Back-tracking)시켜 역전된 상태로의 진입 방지.

---

## 3. 단계별 구현 작업 목록 (Implementation Task Checklist)

- [ ] **Task 1: Constraint 모듈 구현 (`dispsolver/model/constraint.py`)**
  - [ ] `Constraint` 베이스 클래스 및 `RigidBody`, `Tie`, `KinematicCoupling`, `DistributingCoupling`, `MPC` 정의
  - [ ] `Model` 클래스에 팩토리 메서드 연결 (`model.RigidBody()`, `model.Tie()`, `model.Coupling()`, `model.MPC()`)
  - [ ] 단위 테스트 작성 (`tests/test_cae_constraints.py`)

- [ ] **Task 2: Load 모듈 구현 (`dispsolver/model/load.py`)**
  - [ ] `Load` 베이스 클래스 및 `ConcentratedForce`, `Pressure`, `Gravity` 정의
  - [ ] 3D Hex 페이스 및 2D 에지 압력 적분 Numba 커널 구현
  - [ ] `Model` 클래스에 팩토리 메서드 연결 (`model.ConcentratedForce()`, `model.Pressure()`, `model.Gravity()`)
  - [ ] 단위 테스트 작성 (`tests/test_cae_loads.py`)

- [ ] **Task 3: SectionControls & 반전 방지 Numba 커널 구현**
  - [ ] `SectionControls` 클래스 생성 및 `SolidSection` 연동
  - [ ] `c3d8_corotational_numba.py` 및 `c3d8_hybrid_numba.py`에 `distortionControl` 장벽 항 주입
  - [ ] `DynamicSolver3D` 라인서치에 Inversion Guard 장착
  - [ ] 단위 테스트 작성 (`tests/test_section_controls.py`)

- [ ] **Task 4: MultiStep & 가지치기 분기 해석 엔진 구현 (`dispsolver/solver/step_executor.py`)**
  - [ ] `InitialStep` 및 `PredefinedField` 구현
  - [ ] BC/Load/Constraint 생명주기 관리 (`PROPAGATED`, `MODIFIED`, `DEACTIVATED`)
  - [ ] `StateCheckpoint` 저장 및 롤백 분기 실행 엔진 구현
  - [ ] 단위 테스트 작성 (`tests/test_multi_step_branching.py`)

- [ ] **Task 5: 종합 통합 검증 예제 제작 (`examples/ex17_cae_full_features_demo.py`)**
  - [ ] 스마트 세트, RP 기반 RigidBody/Coupling, Pressure 하중, Multi-Step 가지치기(성형 -> 스프링백 vs 추가 압착) 통합 구동
  - [ ] 전체 검증 벤치마크(`python -m verification.run_all`) 100% 무결성 확인

---

## 4. 검증 기준 (Success Criteria)

1. **단위 테스트 무결성**: 신규 작성되는 4개 단위 테스트 파일 모두 100% PASS.
2. **SOLID 및 Karpathy 원칙 준수**: 기존 검증 코드의 손상 없는 외연적 확장 및 회귀 방지.
3. **상용 Abaqus CAE 문법 100% 호환**:
   - `model.RigidBody(name=..., refPoint=..., tieNset=...)`
   - `model.Pressure(name=..., region=..., magnitude=...)`
   - `model.ConcentratedForce(name=..., region=..., dof=..., magnitude=..., amplitude=...)`
   - `model.create_step_branch(name=..., parent=...)`
