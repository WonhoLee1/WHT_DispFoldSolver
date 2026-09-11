# Commercial CAE Model Architecture & Smart GeneralSet (2026-09-12)

## 1. 개요 (Overview)

상용 CAE 소프트웨어(Abaqus / ANSYS / OptiStruct)의 표준 객체 지향 모델 아키텍처를 도입하고, 노드·요소·페이스를 범용적으로 다루는 **Smart `GeneralSet` (범용 스마트 세트)** 기능을 설계 및 구현 완료하였습니다.

```
Model
├── Material (Elastic, Hyperelastic, Plasticity, DOD UMAT)
├── Section (SolidSection, ShellSection -> references Material)
├── Part (Local coordinates, elements, GeneralSet)
│   └── SectionAssignment (region: GeneralSet/str -> Section)
├── Assembly (rootAssembly)
│   ├── Instance (Part instantiation with 3D affine transform: translation + Rodrigues rotation)
│   └── FlattenedSolverSystem (Instant CSR/CSC topology, zero-overhead global DOF mapping)
├── Step (StaticStep, DynamicStep)
│   └── BoundaryCondition / Load (references INSTANCE.GENERAL_SET)
└── Solver (DynamicSolver, DynamicSolver3D)
```

---

## 2. Smart GeneralSet (범용 세트) 핵심 기능

### 2.1 연관 노드 자동 해석 (Associative Node Resolution)
- 요소(Element)만 지정된 세트라 하더라도, `get_nodes(include_elements=True)` 호출 시 해당 요소를 구성하는 고유 노드 ID들을 자동으로 수집·반환합니다.
- 사용자가 요소 영역에 대해 구속조건(BC)이나 하중을 인가할 때 별도의 NodeSet을 수작업으로 관리할 필요가 없습니다.

### 2.2 내부 공유면 자동 상쇄 및 외곽 경계 추출 (Boundary Face & Segment Cancellation)
- `get_faces(exterior_only=True)` / `get_segments()`:
  - 2D Quad 요소(4개 에지) 및 3D Hex 요소(6개 페이스)의 위상(Topology)을 자동 분석합니다.
  - 내부 요소 간에 공유되는 공통 인터페이스는 빈도수 분석($\ge 2$)을 통해 자동으로 상쇄(cancel)하고, 외부로 노출된 경계 에지/페이스(외곽 경계면)만을 완벽히 추출합니다.
  - 접촉(Contact) 및 Surface Tie 정의 시 유저가 수작업으로 면 번호를 일일이 지정할 필요가 없습니다.

### 2.3 기하학적 바운딩 박스 필터링 (Bounding Box Filtering)
- `part.create_set_from_box(name, x_range, y_range, z_range, entity_type="ALL")`:
  - 2D/3D 공간 좌표 범위를 지정하여 영역 내의 노드 및 요소(요소 중심 기준)를 자동으로 필터링하여 `GeneralSet`으로 생성합니다.

### 2.4 세트 대수 연산 (Set Algebra)
- Python 연산자 오버로딩을 통해 직관적인 집합 연산 지원:
  - 합집합: `set_a | set_b`
  - 교집합: `set_a & set_b`
  - 차집합: `set_a - set_b`

### 2.5 Section Assignment 및 Assembly 연동
- `part.assign_section(region, section_name)`: `region` 인자로 문자열 이름뿐만 아니라 `GeneralSet` 객체를 직접 전달 가능.
- `assembly.build_solver_system()`: 인스턴스화된 파트의 `GeneralSet`을 전역 스코프(`"INSTANCE.SET"`)로 승격하여 `global_nsets` 및 `global_elsets`에 자동 등록.

---

## 3. 검증 결과 (Verification Results)

### `pytest tests/test_cae_model_hierarchy.py tests/test_general_set.py -v`

```text
tests/test_cae_model_hierarchy.py::test_transform3d_affine_math PASSED   [ 10%]
tests/test_cae_model_hierarchy.py::test_part_and_section_assignment PASSED [ 20%]
tests/test_cae_model_hierarchy.py::test_assembly_multi_instance_flattening PASSED [ 30%]
tests/test_cae_model_hierarchy.py::test_model_to_solver3d_integration PASSED [ 40%]
tests/test_general_set.py::test_general_set_basic_and_associative_nodes PASSED [ 50%]
tests/test_general_set.py::test_general_set_2d_exterior_cancellation PASSED [ 60%]
tests/test_general_set.py::test_general_set_3d_hex_exterior_cancellation PASSED [ 70%]
tests/test_general_set.py::test_general_set_bounding_box PASSED          [ 80%]
tests/test_general_set.py::test_general_set_algebra PASSED               [ 90%]
tests/test_general_set.py::test_general_set_integration_with_model PASSED [100%]

======================= 10 passed, 1 warning in 31.17s ========================
```

- 10개 전 테스트 케이스 무결성 검증 통과.
- 3D 솔버 비선형 1스텝 평형 수렴 검증 완료.
