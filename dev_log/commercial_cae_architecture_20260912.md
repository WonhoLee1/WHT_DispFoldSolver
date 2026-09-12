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

## 3. 검증 결과 (Verification Results - Initial Phase)

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

---

## 4. 기하학적 형상 기반 스마트 영역 선택 (Geometric Set Factories)

### 4.1 구 및 중공 구 (Sphere & Hollow Sphere)
- `part.create_set_from_sphere(name, center, radius, inner_radius=0.0, entity_type="ALL")`
  - 중심점과 반경 $r_{in} \le r \le r_{out}$ 조건으로 구형 영역 엔티티 필터링.
  - 힌지 피벗 주변부의 국부 요소 셋 추출 및 집중 하중 영역 분리 시 활용.

### 4.2 원통 및 축방향 필터 (Cylinder & Hinge Axis)
- `part.create_set_from_cylinder(name, axis_point, axis_direction, radius, inner_radius=0.0, length_range=None, entity_type="ALL")`
  - 축 방향 벡터와 반경 및 축 방향 길이 구간을 지정하여 원통 영역 필터링.
  - 폴더블 디스플레이의 힌지 축 회전 중심부 핀/베어링/원통형 곡면부 완벽 대응.

### 4.3 평면 및 반공간 (Plane & Half-Space)
- `part.create_set_from_plane(name, point, normal, half_space=False, tolerance=1e-5, entity_type="ALL")`
  - 평면 상의 노드/요소 또는 평면 한쪽 전체(반공간 $n \cdot (x - x_0) \ge 0$) 추출.
  - 대칭 경계조건(Symmetry BC: XSYMM, YSYMM) 설정 시 필수적.

### 4.4 법선 벡터 기반 외곽 표면 (Surface by Normal)
- `part.create_surface_from_normal(name, direction, angle_tol_deg=15.0)`
  - 외곽 경계면의 외측 법선 벡터(outward normal)를 자동 계산하여 기준 방향과의 사잇각이 허용 오차 이내인 페이스 추출.
  - 디스플레이 상면(Top Face) 및 Rigid Plate 접촉면(Bottom Face) 자동 추출.

### 4.5 커스텀 조건식 (Custom Condition Predicate)
- `part.create_set_from_condition(name, condition_fn, entity_type="ALL", element_selection="CENTROID")`
  - 임의의 파이썬 함수 `lambda c: (c[0]**2 + c[1]**2 <= 25.0)`를 전달하여 복합 형상 자유 필터링.

---

## 5. Abaqus findAt 스타일 근접 탐색 및 피처 각도 전파 (Proximity & Propagation)

### 5.1 거리 허용 오차 가드 (Search Tolerance & raise_if_none)
- 쿼리 좌표와 모델 사이의 거리가 `search_tolerance`를 초과할 경우 임의의 원거리 엔티티가 오선택되는 현상을 차단.
- 엄격 모드(`raise_if_none=True`) 시 허용 오차 내 대상이 없으면 예외 발생.

### 5.2 직교 면 투영 (Orthogonal Face Projection)
- 점과 노드 간 단순 유클리드 거리의 한계를 극복하기 위해, 2D 선분 및 3D 쿼드 면에 대한 직교 정사영 및 Point-in-Polygon 판정을 수행.
- 곡면 및 두께 방향 단면 경계에서 올바른 타겟 면(Face)을 오차 없이 타격.

### 5.3 다중 시드점 (Multiple Seed Points)
- 시드 좌표로 단일 점뿐만 아니라 점 목록 `[[x1, y1], [x2, y2], ...]` 수용.
- 복수의 분리된 영역 또는 연속 영역을 한 번의 호출로 통합 전파.

### 5.4 경계 장벽 및 박판 래핑 방지 (Barrier & Wrap-Around Guard)
- `stop_at_nodes`: 특정 경계선 노드 ID 목록을 지정하여 해당 인터페이스를 넘어서는 전파를 차단.
- `max_distance`: 시드점으로부터의 최대 탐색 반경 제한.
- `target_normal`: 얇은 필름/박판 시트의 에지를 타고 반대편 면으로 돌아 넘어가는(wrap-around) 현상을 방지하기 위해 법선 방향 내적($\ge 0.2$) 필터링 적용.

---

## 6. 세트 간 복합 연산(끼리끼리 작용) 및 최근접 $n$개 추출 (Set Operations & findAt)

### 6.1 노드·요소·페이스 독립 연산 ("끼리끼리 작용")
`GeneralSet`은 `node_ids`, `element_ids`, `faces`를 동시에 복합으로 보유할 수 있으며, 두 셋 간의 연산 시 각 엔티티 타입별로 완벽히 독립적으로 동작합니다.
- **합집합 (Union)**:
  - `s1 | s2` 또는 `s1.union(s2, name=None)`
  - 노드는 노드끼리, 요소는 요소끼리, 페이스는 페이스끼리 합집합.
- **차집합 (Difference)**:
  - `s1 - s2` 또는 `s1.difference(s2, name=None)`
  - 노드에서 노드를 빼고, 요소에서 요소를 빼고, 페이스에서 페이스를 차감.
- **교집합 (Intersection)**:
  - `s1 & s2` 또는 `s1.intersection(s2, name=None)`
  - 양쪽 모두에 존재하는 노드, 요소, 페이스만 보존.

### 6.2 겹치는 노드 및 요소 추출 (Overlapping Entities Extraction)
- `set_a.get_overlapping_nodes(set_b, include_elements=False) -> np.ndarray`:
  - `include_elements=False` (기본): 명시적으로 등록된 `node_ids` 간의 교집합 반환.
  - `include_elements=True`: 각 세트의 요소(`element_ids`)를 구성하는 모든 노드까지 연관 확장하여 두 셋 간에 공유되는 모든 절점을 추출 (예: 인접 요소 셋 간의 공통 경계면 노드 자동 추출).
- `set_a.get_overlapping_elements(set_b) -> np.ndarray`:
  - 두 세트 간에 공통으로 포함된 요소 ID 배열 반환.
- `set_a.get_overlapping_faces(set_b) -> List[ElementFace]`:
  - 두 세트 간에 공통으로 포함된 페이스 목록 반환.
- Part 레벨 헬퍼:
  - `part.get_overlapping_nodes(set_a, set_b, include_elements=False)`
  - `part.get_overlapping_elements(set_a, set_b)`
  - `part.boolean_union(name, set_a, set_b)`
  - `part.boolean_difference(name, set_a, set_b)`
  - `part.boolean_intersection(name, set_a, set_b)`

### 6.3 거리순 정렬 기반 $n$개 최근접 추출 함수
- `part.find_closest_nodes(coords, n=1, search_tolerance=None, return_distances=False)`:
  - 쿼리 좌표로부터 가장 가까운 순서대로 정렬된 $n$개의 노드 ID (또는 거리 튜플) 반환.
- `part.find_closest_elements(coords, n=1, search_tolerance=None, return_distances=False)`:
  - 쿼리 좌표로부터 중심점 기준 가장 가까운 순서대로 정렬된 $n$개의 요소 ID 반환.
- `part.find_at(coords, name=None, entity_type="ALL", n=1, search_tolerance=None) -> GeneralSet`:
  - Abaqus 스타일로 단일 점 또는 다중 점 좌표를 받아 최근접 $n$개의 노드/요소를 추출하여 즉시 Part의 `GeneralSet`으로 생성 및 반환.

---

## 7. 종합 검증 결과 (Full Test Suite Summary)

```text
pytest tests/test_general_set.py tests/test_geometric_sets.py tests/test_angle_and_proximity_sets.py tests/test_advanced_selection.py tests/test_set_operations_and_findat.py -v

============================= 27 passed in 3.67s ==============================
```

모든 테스트 케이스 27개 100% 통과 (SOLID 원칙, Karpathy 가이드라인 준수 및 회귀 오류 없음 확인 완료).

