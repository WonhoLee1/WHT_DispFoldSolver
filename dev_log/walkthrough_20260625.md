# Walkthrough: `*RIGID BODY` 키워드 지원 — Abaqus `.inp` 파이프라인 확장 (2026-06-25)

Abaqus `.inp` 입력 파이프라인에 `*RIGID BODY` 키워드 파서와 모델 빌더를 추가하여, Abaqus/CAE에서 정의한 강체 바디를 `RBE2HingeConstraint`로 자동 변환합니다. 전체 **115개 단위 테스트 통과**.

---

## 1. 개요

`*RIGID BODY`는 Abaqus에서 특정 노드 집합(ELSET 또는 NSET으로 지정)을 하나의 참조노드(REF NODE)에 강체 결합하는 구속조건입니다. 기존 `*MPC, TYPE=RBE2`와 운동학이 동일하므로, `RBE2HingeConstraint`를 재사용합니다.

### `*RIGID BODY` 입력 예시

```
*RIGID BODY, ELSET=BLOCK_A, REF NODE=5     ← 요소 집합으로 지정
*RIGID BODY, NSET=MY_NODES, REF NODE=5      ← 노드 집합으로 지정
*RIGID BODY, NSET=ALL_NODES, REF NODE=5     ← NSET,GENERATE 범위 지정
```

---

## 2. 변경 파일 및 수정 내용

### 2.1. `dispsolver/io/abaqus_model.py` — 데이터모델 확장

**`AbaqusRigidBody`** dataclass 추가:

```python
@dataclass
class AbaqusRigidBody:
    ref_node: int            # 참조 노드 (마스터)
    node_ids: List[int]      # 구속 대상 노드 (슬레이브)
    name: str                # 식별자 (NSET/ELSET 이름)
```

**`AbaqusModel`**에 `rigid_bodies` 필드 추가:

```python
class AbaqusModel:
    ...
    rigid_bodies: List[AbaqusRigidBody] = field(default_factory=list)
```

### 2.2. `dispsolver/io/abaqus_lexer.py` — Lexer 버그 2건 수정

#### 2.2.1. `RE_PARAM`: 다중 단어 파라미터 이름 지원

`REF NODE=5`는 두 단어로 된 파라미터 이름. 기존 정규식 `(\w[\w.]*)`은 공백을 허용하지 않아 `REF`만 캡처되고 `NODE`는 무시됨.

**수정 전**:
```python
RE_PARAM = re.compile(r"(\w[\w.]*)\s*=\s*([^,]*)")
```

**수정 후**:
```python
RE_PARAM = re.compile(r"(\w[\w.]*(?:\s+[\w.]+)*)\s*=\s*([^,]*)")
```

`(?:\s+[\w.]+)*`로 공백 + 단어 조합을 0회 이상 허용 → `REF NODE`, `NEO HOOKE` 등 다중 단어 파라미터 정상 파싱.

#### 2.2.2. `_parse_keyword_params`: 플래그 인자(= 없는 키워드) 지원

`*NSET,NSET=MY_NODES,GENERATE`에서 `GENERATE`는 값이 없는 플래그 인자. 기존 코드는 `=`이 없으면 조용히 무시 → `_parse_nset`에서 `generate=yes`를 감지 못해, 데이터 `1,4,1`을 단순 콤마 제거(`141`)로 파싱하는 버그 발생.

**수정 전**:
```python
for part in param_text.split(","):
    m = RE_PARAM.match(part)           # = 없는 부분은 매치 실패 → 무시
    if m:
        params[m.group(1).lower()] = m.group(2).strip()
```

**수정 후**:
```python
for part in param_text.split(","):
    m = RE_PARAM.match(part)
    if m:
        params[m.group(1).lower()] = m.group(2).strip()
    else:
        bare = part.strip()
        if bare:
            params[bare.lower()] = "yes"    # 플래그 → "yes"로 저장
```

### 2.3. `dispsolver/io/abaqus_parser.py` — `_parse_rigid_body` 핸들러

`_parse_rigid_body` 핸들러 추가 및 `_handlers` 딕셔너리에 등록:

```python
self._handlers["RIGID BODY"] = self._parse_rigid_body
```

핸들러 로직:
1. `ELSET=` 또는 `NSET=` 파라미터에서 대상 노드/요소 집합 이름 획득
2. `REF NODE=`에서 참조 노드 ID 획득
3. `nset`이 있으면 `model.nsets[name]`에서 직접 노드 ID 리스트 조회
4. `elset`이 있으면 해당 요소들의 `node_ids`를 중복 제거하여 수집
5. `AbaqusRigidBody` 생성 → `model.rigid_bodies`에 추가

### 2.4. `dispsolver/io/model_builder.py` — 모델 빌더 구속조건 변환

**`_next_extra_primal_offset()`** 헬퍼 추가:

```python
def _next_extra_primal_offset(self) -> int:
    """Return the cumulative extra primal DOF offset for the next constraint."""
    return sum(c.n_extra_primal() for c in self._result.constraints)
```

`RBE2HingeConstraint`는 extra primal DOF로 회전각 `θ`를 가지므로, 복수의 강체 바디/MPC가 있을 때 각각 서로 다른 extra primal DOF index를 가져야 합니다. 위 헬퍼는 이미 빌드된 모든 constraint의 `n_extra_primal()` 합계를 반환하여 다음 constraint의 offset을 계산합니다.

**`_build_constraints()`에 강체 바디 루프 추가**:

```python
for rb in self._abq.rigid_bodies:
    constraint = RBE2HingeConstraint(
        mesh, master_id=rb.ref_node,
        slave_ids=rb.node_ids,
        extra_primal_offset=ext_offset,
    )
    self._result.constraints.append(constraint)
```

### 2.5. `tests/test_rbe2.py` — 테스트 3종 추가

| 테스트 | 검증 내용 |
|--------|-----------|
| `test_rigid_body_parser` | `*RIGID BODY,ELSET=BLOCK_A,REF NODE=5` → 4개 노드가 `RBE2HingeConstraint`로 변환, master=5, slave_ids=[1,2,3,4], offset=0 |
| `test_rigid_body_nset` | `*NSET,NSET=MY_NODES,GENERATE` + `*RIGID BODY,NSET=MY_NODES,REF NODE=5` → NSET+GENERATE 경로 검증 |
| `test_rigid_body_nset_generate` | 동일하나 노드 좌표가 모두 동일(강체 바디 검증에 영향 없음) → 다양한 입력 변형 커버 |

### 2.6. `dispsolver/constraint/rbe2.py` — 변경 없음

`RBE2HingeConstraint`는 기존 생성자 시그니처 `(mesh, master_id, slave_ids, extra_primal_offset)`를 그대로 사용합니다. 모델 빌더가 적절히 offset을 계산하여 전달하므로 변경 불필요.

---

## 3. 파이프라인 데이터 흐름

```
*.inp 파일
  ↓ [AbaqusLexer.tokenize()]
AbaqusKeywordBlock 리스트
  ↓ [AbaqusParser.parse()]
AbaqusModel (rigid_bodies 포함)
  ↓ [ModelBuilder.build()]
ModelBuilderResult.constraints: [RBE2HingeConstraint, ...]
  ↓ [DynamicSolver.__init__()]
해석 실행
```

---

## 4. 발견된 버그 및 해결 과정

### 버그 1: `RE_PARAM`이 `REF NODE`를 한 파라미터로 인식 못함

- **증상**: `*RIGID BODY,ELSET=BLOCK,REF NODE=6`에서 `ref node`가 `block.params`에 저장되지 않음 → `ref_node=0` → 핸들러가 `return` 처리
- **원인**: 정규식 `(\w[\w.]*)`이 첫 번째 단어(`REF`)만 캡처, 두 번째 단어(`NODE`)는 `=6`의 일부로 잘못 해석
- **해결**: `(?:\s+[\w.]+)*` 추가로 공백+단어 반복 허용

### 버그 2: `GENERATE` 같은 플래그 인자 무시

- **증상**: `*NSET,NSET=X,GENERATE\n1,4,1` → `ids=[141]` (1,4,1의 콤마를 제거하면 `"141"`이 됨)
- **원인**: `_parse_keyword_params`가 `=` 없는 인자를 조용히 스킵 → `_parse_nset`에서 `generate` 파라미터 부재 → 비-GENERATE 경로로 파싱
- **해결**: `=` 없는 인자는 `{key: "yes"}`로 저장

---

## 5. 기존 버그: MPC→RBE2 모델 빌더 키워드 인자 불일치

모델 빌더 `_build_constraints()`의 MPC→RBE2 경로(라인 529-531)는 키워드 인자 `master_node=`, `slave_nodes=`를 사용하지만, `RBE2HingeConstraint.__init__`은 위치 인자 `master_id`, `slave_ids`만 받습니다:

```python
# model_builder.py:529 — 키워드 인자 사용 (잘못됨)
constraint = RBE2HingeConstraint(
    master_node=mpc.nodes[0],       # RBE2HingeConstraint는 master_node 인자 없음
    slave_nodes=list(mpc.nodes[1:]), # slave_nodes 인자 없음
)

# 실제 생성자: 위치 인자만
RBE2HingeConstraint(mesh, master_id, slave_ids, extra_primal_offset)
```

이 경로는 실행 중 `TypeError`가 발생하지만, 현재 테스트에서 MPC(특히 RBE2 타입)를 사용하는 케이스가 없어 발견되지 않았습니다. MPC를 사용할 때는 `mesh` 인자도 누락되어 추가 수정이 필요합니다.

---

## 6. 커밋

```
78b99c8 docs: add 2026-06-24 walkthrough
c3ad938 ex03: drive hinge rotation BC via Amplitude curve
28efcff Section thickness, contact depth, and time-dependent loads (Amplitude)
4a27bd5 Nonlinear FEM display-fold solver: robustness, multi-core, hybrid viscoelastic
```

(본 세션은 커밋 전 — `.inp` 파이프라인 확장 작업 진행 중)

---

## 7. 남은 작업

| 항목 | 우선순위 | 설명 |
|------|----------|------|
| MPC→RBE2 빌더 키워드 인자 수정 | 높음 | `master_node=`/`slave_nodes=` → `master_id=`/`slave_ids=` + `mesh` 인자 추가. 버그이지만 현재 미사용 코드 |
| `*RIGID BODY` 복수 offset 검증 테스트 | 중간 | 강체 바디 2개 이상 + MPC 조합 시 `_next_extra_primal_offset()` 누적 정확성 검증 |
| `*NSET,GENERATE` 비연속 간격 | 낮음 | `GENERATE`에서 `step != 1`인 경우 검증 — 현재 단순 `range(start, end+1, step)` 사용 |
