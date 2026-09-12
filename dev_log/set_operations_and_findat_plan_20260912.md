# Implementation Plan: Set Operations & findAt-Style Proximity Extraction

## 1. 개요 및 요구사항 분석
1. **셋(GeneralSet) 간 연산 및 복합 엔티티 독립 작용 (끼리끼리 연산)**
   - `GeneralSet`은 `node_ids`, `element_ids`, `faces`를 동시에 복합으로 가질 수 있음.
   - 연산 시 엔티티 타입별로 완벽히 독립적으로 동작:
     - `node_ids`: 노드는 노드끼리 합치기(`union`), 빼기(`difference`), 겹치기(`intersection`)
     - `element_ids`: 요소는 요소끼리 합치기, 빼기, 겹치기
     - `faces`: 페이스는 페이스끼리 합치기, 빼기, 겹치기
   - **겹치는 노드/요소 추출 기능**:
     - `set_a.get_overlapping_nodes(set_b, include_elements=False) -> np.ndarray`
       - `include_elements=False`: 명시적 `node_ids` 간 교집합
       - `include_elements=True`: 요소 구성 노드까지 연관 확장(`get_nodes(include_elements=True)`)한 뒤 교집합 노드 반환
     - `set_a.get_overlapping_elements(set_b) -> np.ndarray`
     - `set_a.get_overlapping_faces(set_b) -> List[ElementFace]`
   - 연산자 및 메서드 API:
     - `set_a | set_b` / `set_a.union(set_b, name=None)`
     - `set_a - set_b` / `set_a.difference(set_b, name=None)`
     - `set_a & set_b` / `set_a.intersection(set_b, name=None)`
   - Part 레벨 헬퍼:
     - `part.boolean_union(name, set_a, set_b)`
     - `part.boolean_difference(name, set_a, set_b)`
     - `part.boolean_intersection(name, set_a, set_b)`
     - `part.get_overlapping_nodes(set_a, set_b, include_elements=False)`

2. **findAt 스타일의 가까운 순서로 $n$개 노드/요소 추출 함수**
   - 좌표(Point)로부터 유클리드 거리 기준으로 가까운 순서대로 정렬된 $n$개의 엔티티를 반환:
     - `part.find_closest_nodes(coords, n=1, search_tolerance=None, return_distances=False) -> List[int] | List[Tuple[int, float]]`
     - `part.find_closest_elements(coords, n=1, search_tolerance=None, return_distances=False) -> List[int] | List[Tuple[int, float]]`
   - Abaqus 스타일 `find_at`:
     - `part.find_at(name, coords, entity_type="ALL", n=1, search_tolerance=None) -> GeneralSet`
       - 단일 점 또는 다중 점 좌표 지원
       - $n$개의 가까운 노드/요소를 찾아 즉시 새 `GeneralSet` 생성 및 Part에 등록

---

## 2. 세부 구현 계획

### Step 1: `dispsolver/model/set.py`
- `GeneralSet.union(other, name=None)`
- `GeneralSet.difference(other, name=None)`
- `GeneralSet.intersection(other, name=None)`
- `GeneralSet.get_overlapping_nodes(other, include_elements=False) -> np.ndarray`
- `GeneralSet.get_overlapping_elements(other) -> np.ndarray`
- `GeneralSet.get_overlapping_faces(other) -> List[ElementFace]`

### Step 2: `dispsolver/model/part.py`
- `Part.find_closest_nodes(coords, n=1, search_tolerance=None, return_distances=False)`
- `Part.find_closest_elements(coords, n=1, search_tolerance=None, return_distances=False)`
- `Part.find_at(name, coords, entity_type="ALL", n=1, search_tolerance=None)`
- `Part.boolean_union(name, set_a, set_b)`
- `Part.boolean_difference(name, set_a, set_b)`
- `Part.boolean_intersection(name, set_a, set_b)`
- `Part.get_overlapping_nodes(set_a, set_b, include_elements=False)`
- `Part.get_overlapping_elements(set_a, set_b)`

### Step 3: 단위 테스트 작성 (`tests/test_set_operations_and_findat.py`)
- 노드/요소/페이스 복합 세트 간의 합집합, 차집합, 교집합 끼리끼리 작용 검증
- `get_overlapping_nodes` (직접 노드 교집합 vs include_elements=True 연관 노드 교집합)
- `find_closest_nodes` 및 `find_closest_elements` 거리순 정렬 및 $n$개 추출, tolerance 필터링 검증
- `part.find_at`으로 생성된 GeneralSet의 유효성 검증

### Step 4: 개발 로그(`dev_log/commercial_cae_architecture_20260912.md`) 기록 및 최종 검증
- 모든 테스트(총 26개 이상) 일괄 통과 확인 후 커밋.
