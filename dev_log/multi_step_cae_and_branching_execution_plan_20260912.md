# Implementation Plan: Commercial Multi-Step CAE Architecture & Branching Step Tree Execution Engine

## 1. 개요 (Overview)

상용 CAE 소프트웨어(Abaqus / ANSYS / LS-DYNA)의 표준 아키텍처를 도입하여 **멀티 스텝(Multi-Step) 해석 설정**, **스텝별 엔티티 생명주기 제어(활성화 / 비활성화 / 수정 / 계승)**, **초기 상태 부여(Predefined Field)**, 그리고 **가지치기 분기 해석(Branching Step Tree Execution Engine)**을 설계 및 구현합니다.

```
Model ("Model-1")
├── InitialStep ("Initial", t=0)
│   ├── PredefinedFields (Initial Stress, Strain, Velocity, SDVs)
│   └── Base BCs / Interactions
├── Step 1 ("Forming_Phase") ──> Checkpoint-1 (State Snapshot: u, v, a, SDVs, t)
│   ├── BCs: [BC-1: CREATED]
│   ├── Loads: [Load-1: CREATED]
│   └── Step 1 Solver Execution ──> Saved Result Branch 1
│
├── Step Tree Branching (가지치기 및 분기 실행)
│   ├── Branch A: Step 2-1 ("Unloading_CaseA") [Restores Checkpoint-1]
│   │   ├── BCs: [BC-1: MODIFIED], [BC-2: CREATED]
│   │   └── Loads: [Load-1: DEACTIVATED]
│   │       └── Step 3-1 ("Springback_A") ──> Result Branch 2-1
│   │
│   └── Branch B: Step 2-2 ("Crushing_CaseB")  [Restores Checkpoint-1]
│       ├── BCs: [BC-1: DEACTIVATED], [BC-3: CREATED]
│       └── Loads: [Load-1: MODIFIED (3x force)]
│           └── Step 3-2 ("Impact_B")     ──> Result Branch 2-2
```

---

## 2. 핵심 설계 개념 (Key Architectural Concepts)

### 2.1 스텝 구조 및 엔티티 생명주기 상태 (Entity Lifecycle Status)
Abaqus Python API의 스텝 상태 관리 체계를 반영하여, BC / Load / Interaction이 스텝을 진행하면서 가지는 5가지 상태를 관리합니다:
1. `CREATED`: 해당 스텝에서 새로 생성 및 활성화됨.
2. `PROPAGATED`: 직전 스텝의 설정(값, 진폭 등)을 그대로 유지함.
3. `MODIFIED`: 해당 스텝에서 크기, 진폭(Amplitude), 방향 등이 변경됨.
4. `DEACTIVATED`: 해당 스텝에서 비활성화(OFF)됨.
5. `REACTIVATED`: 이전 스텝에서 비활성화되었던 조건이 다시 활성화(ON)됨.

### 2.2 사전 정의된 필드 (Predefined Field)
- `InitialStep`에서 부여되는 초기 상태값:
  - `type`: `"STRESS"`, `"STRAIN"`, `"VELOCITY"`, `"TEMPERATURE"`, `"STATE_VARIABLE"`
  - 노드/요소 세트 영역별 초기 속도($v_0$), 초기 잔류 응력($\sigma_0$), 초기 상태 변수(SDVs) 할당.

### 2.3 가지치기 분기 해석 트리 (Branching Step Tree Execution Engine)
- **체크포인트 스냅샷 (State Checkpoint)**:
  - 스텝 완료 시점의 솔버 무결성 상태(`u`, `v`, `a`, `SDVs`, `mesh`, `t_accum`)를 메모리 및 디스크 스냅샷으로 저장.
- **가지치기 (Branching)**:
  - 직렬 순차 해석(`Step 1 -> Step 2 -> Step 3`) 뿐만 아니라, 특정 스텝의 체크포인트를 부모 노드로 삼아 **독립적인 수평 브랜치(`Step 2-1`, `Step 2-2`)로 분기 해석**.
  - 동일한 초기 전처리/성형 스텝(`Step 1`)을 매번 재계산하지 않고, 다양한 시험 시나리오(Load Cases / Parameter Sweep / Springback / Impact)를 병렬/순차 분기 해석으로 초고속 실행.

---

## 3. 세부 클래스 설계 및 변경 사항 (Proposed Changes)

### Component 1: `dispsolver/model/step.py`
#### [MODIFY] `step.py`
- `EntityStatus` Enum 정의 (`CREATED`, `PROPAGATED`, `MODIFIED`, `DEACTIVATED`, `REACTIVATED`).
- `StepStateEntry[T]`: 각 엔티티의 객체 레퍼런스, 파라미터 및 `EntityStatus` 상태 추적 객체.
- `PredefinedField` 클래스 추가 (초기 속도, 응력, SDV 부여).
- `Step` 클래스 확장:
  - `parent_step`: 부모 스텝 레퍼런스 (트리 구조 형성).
  - `child_steps`: 자식 스텝 목록.
  - `boundary_conditions`: `{bc_name: StepStateEntry[DisplacementBC]}`
  - `loads`: `{load_name: StepStateEntry[Load]}`
  - `interactions`: `{interaction_name: StepStateEntry[Interaction]}`
  - `deactivate_bc(bc_name)`, `modify_bc(bc_name, **kwargs)`, `reactivate_bc(bc_name)`
  - `deactivate_load(load_name)`, `modify_load(load_name, **kwargs)`

```python
class EntityStatus(Enum):
    CREATED = "CREATED"
    PROPAGATED = "PROPAGATED"
    MODIFIED = "MODIFIED"
    DEACTIVATED = "DEACTIVATED"
    REACTIVATED = "REACTIVATED"

@dataclass
class PredefinedField:
    name: str
    field_type: str  # "STRESS", "STRAIN", "VELOCITY", "TEMPERATURE", "SDV"
    region: str
    values: Any

@dataclass
class StepStateEntry:
    entity: Any
    status: EntityStatus = EntityStatus.CREATED
    modified_params: Dict[str, Any] = field(default_factory=dict)
```

---

### Component 2: `dispsolver/model/model.py`
#### [MODIFY] `model.py`
- `model.initial_step` (`InitialStep` 생성 및 관리).
- `model.Step(name, previous="Initial", procedure="STATIC", time_period=1.0)`:
  - `previous` 인자를 받아 부모 스텝에 연결하고 자식 스텝으로 자동 등록.
  - 직전 스텝으로부터 활성화된 BC / Load / Interaction 상태를 `PROPAGATED`로 자동 승계.
- `model.create_step_branch(branch_name, parent_step_name)`:
  - 명시적 가지치기(Branching) API 추가.

---

### Component 3: `dispsolver/solver/step_executor.py`
#### [NEW] `dispsolver/solver/step_executor.py`
- `StateCheckpoint`:
  - `step_name: str`
  - `time_accumulated: float`
  - `u: np.ndarray`, `v: np.ndarray`, `a: np.ndarray`
  - `sdvs: Dict[int, np.ndarray]`
  - `mesh_coords: np.ndarray`
- `MultiStepExecutor`:
  - `Model` 객체를 수용하여 스텝 트리를 위상 정렬(Topological Sort) 또는 브랜치별 순회.
  - 부모 스텝이 완료되면 `StateCheckpoint` 자동 저장.
  - 자식 브랜치 실행 시 부모의 `StateCheckpoint`를 롤백하여 복원 후 독립 해석 진행.
  - 스텝별로 활성화된 최신 BC/Load/Interaction 상태를 전역 DOF 행렬 및 RHS 하중 벡터로 조립.

---

### Component 4: 단위 테스트 `tests/test_multi_step_and_branching.py`
#### [NEW] `tests/test_multi_step_and_branching.py`
- 스텝 생성 및 BC/Load 생명주기(`CREATED`, `PROPAGATED`, `DEACTIVATED`, `MODIFIED`) 전환 검증.
- `PredefinedField` 초기 속도/응력 부여 검증.
- `MultiStepExecutor`의 롤백 및 가지치기(Branching Tree) 실행 검증 (Step 1 완료 후 Step 2-1과 Step 2-2 독립 분기 실행 및 결과 파일 분리 확인).

---

### Component 5: 예제 데모 `examples/ex17_multi_step_branching_demo.py`
#### [NEW] `examples/ex17_multi_step_branching_demo.py`
- 3D 디스플레이 굽힘 모델 시나리오:
  - `Initial`: 대칭 BC 및 초기 하중 조건.
  - `Step 1 (Folding_Phase)`: 힌지 90° 1차 굽힘 실행 -> Checkpoint-1 포획.
  - `Branch A (Step 2-1)`: 굽힘 상태에서 스프링백(Springback Unloading) 해제 해석.
  - `Branch B (Step 2-2)`: 굽힘 상태에서 국부 중앙부 압착(Local Impact/Crushing) 연속 과하중 해석.
- 두 브랜치 결과 시각화 및 비교 리포트 생성.

---

## 4. 검증 계획 (Verification Plan)

### 4.1 자동화 단위 테스트
```bash
pytest tests/test_multi_step_and_branching.py -v
pytest tests/test_cae_model_hierarchy.py tests/test_set_operations_and_findat.py -v
python -m verification.run_all
```
- 모든 스텝 생성, 생명주기 관리, Checkpoint 롤백, 가지치기 실행 검증 100% 통과 확인.

### 4.2 매뉴얼 검증 및 시각화
```bash
python -u examples/ex17_multi_step_branching_demo.py
```
- `Step 1`에서 `Step 2-1`(Springback)과 `Step 2-2`(Crushing)로 분기 해석이 정상 실행되고, 각각 독립적인 결과 스냅샷이 저장되었는지 터미널 리포트 및 그래프 확인.
