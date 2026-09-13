# [Tech Note] Abaqus Scripting Interface 호환 2D/3D 단일 통합 SurfaceTie API 구축

**문서 일자**: 2026-09-13  
**대상 모듈**: `dispsolver/constraint/surface_tie_unified.py`, `dispsolver/constraint/__init__.py`  
**테스트 스위트**: `tests/test_surface_tie_unified.py` (100% PASS)

---

## 1. 배경 및 사용자 핵심 질문

> **사용자 질의**:  
> *"abaqus python api에서는 SurfaceTieConstraint, SurfaceTieConstraint3D 가 구분을 따로 하지 않은 호출 함수 아닌가?"*

Abaqus Python API (`abaqus.py`, `mdb.models[...].Tie()`)의 실제 설계 철학을 검토한 결과, **사용자의 지적이 100% 정확**했습니다.

---

## 2. Abaqus Python API의 공식 설계 철학

### 2.1 Abaqus의 단일 `model.Tie()` 문법
Abaqus/CAE Scripting Reference Guide에 따르면, 모델이 2D(평면응력, 평면변형률, 축대칭)이든 3D(솔리드, 쉘)이든 관계없이 오직 **단 하나의 일관된 메서드**로 제약조건을 정의합니다:

```python
mdb.models['Model-1'].Tie(
    name='Constraint-1',
    master=region_master,
    slave=region_slave,
    positionToleranceMethod=COMPUTED,  # 또는 SPECIFIED
    positionTolerance=1.0,
    adjust=ON,
    tieRotations=ON
)
```

Abaqus INP 키워드 파일에서도 2D/3D 구분 없이 완전히 동일합니다:
```abaqus
*TIE, NAME=Constraint-1, ADJUST=YES
slave_surface, master_surface
```

### 2.2 Abaqus 내부 차원 자동 판별 원리
1. **Region / Surface 객체의 차원 추상화**:
   - 사용자가 `Region` 또는 `Surface`를 넘길 때, Abaqus는 해당 서피스를 구성하는 기저 요소의 위상 차원(Topological Dimension)을 내부 메쉬 데이터베이스에서 즉각 조회합니다.
   - 2D 모델 서피스: 1차원 선분 요소면(Edge)들의 집합.
   - 3D 모델 서피스: 2차원 사각형/삼각형 요소면(Facet)들의 집합.
2. **Solver 커널 자동 디스패치**:
   - 솔버 런타임에서 위상이 1D Edge이면 **2D Node-to-Segment** 페널티/라그랑주 투영 커널로 전달.
   - 위상이 2D Facet이면 **3D Node-to-Face (Quad4/Tri3)** 페널티/라그랑주 투영 커널로 전달.
   - 사용자는 저수준 2D/3D 수학 커널의 차이를 전혀 신경 쓸 필요가 없습니다.

---

## 3. 우리 솔버의 기존 한계 vs 신규 통합 팩토리 아키텍처

### 3.1 기존 구조 (분리형 - 인지 부하 발생)
```
[ 기존 분리형 구조 ]
  2D: from dispsolver.constraint.surface_tie import SurfaceTieConstraint
      -> 인자: slave_node_ids, master_node_ids (선분 절점열), coords(2D), nid_to_idx
  3D: from dispsolver.constraint3d.surface_tie3d import SurfaceTieConstraint3D
      -> 인자: slave_node_ids, master_faces (4절점 튜플 리스트), coords(3D), nid_to_idx

  단점:
  1. 2D/3D마다 모듈 경로와 클래스명이 다름.
  2. 3D에서는 마스터 서피스의 4절점 Quad face 튜플 리스트를 사용자가 수동 조합해야 함.
```

### 3.2 신규 구조 (Abaqus 완전 패리티 통합 팩토리)
[`dispsolver/constraint/surface_tie_unified.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/constraint/surface_tie_unified.py):

```python
from dispsolver.constraint import SurfaceTie

# 사용법 1: Mesh 객체 기반 (2D / 3D 완전 자동 감지)
tie = SurfaceTie(
    slave=slave_nids,
    master=master_nodes_or_faces,
    mesh=mesh,                          # Mesh2D 또는 Mesh3D
    penalty_stiffness=1e5,
    name="TIE_1"
)

# 사용법 2: 순수 좌표 배열 기반
tie = SurfaceTie(
    slave=slave_nids,
    master=master_nodes_or_faces,
    coords=coords,                      # shape (N, 2) 또는 (N, 3)
    nid_to_idx=nid_to_idx,
    penalty_stiffness=1e5,
    name="TIE_1"
)
```

---

## 4. 통합 팩토리의 다형성 및 지능형 기능

1. **차원 자동 감지 (Dimension Auto-Detection)**:
   - `mesh` 객체의 타입(`Mesh2D` vs `Mesh3D`) 또는 `coords.shape[1]`(2 vs 3)을 검사하여 2D/3D를 100% 무결하게 판별.
2. **3D 마스터 절점열의 Quad Face 자동 추출 (Auto-Surface Reconstruction)**:
   - 3D에서 사용자가 4절점 튜플 대신 단순 **마스터 표면 절점 ID 리스트(`List[int]`)**만 넘기더라도, `mesh`의 요소 외곽 면들을 역추적하여 일치하는 Quad4 Face들을 자동으로 추출 및 페어링.
3. **100% 하위 호환성 (Backward Compatibility)**:
   - 기존의 `from dispsolver.constraint import SurfaceTieConstraint` 및 `from dispsolver.constraint3d import SurfaceTieConstraint3D` 코드도 아무런 수정 없이 그대로 작동.

---

## 5. 단위 테스트 검증 결과

`tests/test_surface_tie_unified.py`:
- `test_surface_tie_2d_coords`: 2D 좌표 배열 기반 생성 검증 $\to$ **PASS**
- `test_surface_tie_2d_mesh`: `Mesh2D` 객체 기반 생성 검증 $\to$ **PASS**
- `test_surface_tie_3d_faces`: 3D 좌표 + Quad Face 튜플 기반 생성 검증 $\to$ **PASS**
- `test_surface_tie_3d_mesh_auto_faces`: `Mesh3D` 객체 + 절점 ID 리스트 기반 외곽 Face 자동 복원 및 생성 검증 $\to$ **PASS**

**결과: 4 passed in 0.94s (100% 무결점 통과)**
