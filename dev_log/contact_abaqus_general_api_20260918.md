# Abaqus Python API Parity for Contact General & Part Exterior Self-Contact
**Date**: 2026-09-18  
**Author**: Antigravity Core FEA Architecture Team  
**Status**: Completed & Verified (31/31 Passed)

---

## 1. 개요 및 요구사항
사용자가 Abaqus Python 스크립팅(`mdb.models['Model-1']...`)에 익숙하므로, 기존 수치해석 솔버의 복잡한 접촉면 수동 정의(마스터/슬레이브 면 픽킹) 없이도 **Abaqus와 완전히 동일한 문법과 동작**을 제공하는 것을 목표로 구현되었습니다.

### 핵심 달성 사항
1. **Contact General (`ContactStd`, `*CONTACT, GENERAL`) 완벽 지원**:
   - 사용자가 별도의 마스터/슬레이브 면을 지정하지 않아도 `useAllExterior=True` 하나로 전체 모델의 모든 외곽 경계면(ALL EXTERIOR)을 자동 탐지하여 접촉 해석을 수행.
   - 접히는 디스플레이의 자가 접촉(Self-Contact) 및 여러 파트 간의 접근 접촉을 모두 단일 인터페이스로 자동 해석.
2. **파트 외곽면 자동 추출 (`part.exterior`, `part.get_exterior_faces()`)**:
   - `part.exterior` 속성을 통해 파트의 외곽면을 즉시 추출하여 `includedPairs=[(partA.exterior, partB.exterior)]` 형태로 직관적 지정 가능.
3. **Abaqus Python API 일치화**:
   - `model.ContactProperty('IntProp-1', normal_behavior='HARD', ...)`
   - `model.ContactStd('Int-1', createStepName='Step-1', useAllExterior=True, ...)`
   - `model.SurfaceToSurfaceContactStd('Pair-1', createStepName='Step-1', main=..., secondary=..., ...)`

---

## 2. 수치역학 및 아키텍처 상세

### 2.1 3D 외곽 경계면 자동 탐색 알고리즘 (`auto_detect_exterior_3d`)
- **파일**: [`dispsolver/contact/contact_surface.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/contact/contact_surface.py)
- **지원 요소**: `C3D8`, `C3D8R`, `C3D8I`, `C3D8H`, `C3D8_FBAR`, `C3D8_CR`, `SOLID_SHELL`, `C3D4`, `C3D4_ANP`, `C3D6`, `C3D10`
- **알고리즘**:
  1. 각 요소의 로컬 면(Face) 정의에 따른 노드 ID 튜플을 정렬하여 해시 카운팅.
  2. 요소 간 공유되는 내부 면은 발생 횟수 $\ge 2$로 상쇄 소거.
  3. 외곽에 노출된 면은 발생 횟수 $= 1$로 식별.
  4. 요소 중심 $\mathbf{C}_{\text{elem}}$과 면 중심 $\mathbf{C}_{\text{face}}$ 간의 방향 벡터를 기준으로 외곽 법선 벡터 $\mathbf{n}_{\text{out}}$을 엄밀히 계산하고, 필요 시 노드 와인딩(Winding) 순서를 반전하여 **완벽한 바깥쪽 외곽 법선 벡터**를 보장.

### 2.2 3D 동적 BVH 접촉 엔진 (`GeneralContact3D`)
- **파일**: [`dispsolver/contact/general_contact3d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/contact/general_contact3d.py)
- **Broad-Phase (광역 충돌 감지)**:
  - 30-bit Morton Code 공간 정렬 및 Dynamic BVH Octree 활용 ($O(N \log N)$).
  - 매 뉴턴 반복(Newton Iteration)마다 변형된 좌표 $\mathbf{x} = \mathbf{X} + \mathbf{u}$로 바텀업 AABB 리핏(`bvh.refit(u)`).
  - 요소 크기 연동 무차원 스킨(`auto_skin = 0.05 * L_char`)으로 메쉬 크기에 상관없이 스킨 관통 누락 원천 방지.
  - 노드를 공유하는 인접 면은 위상 필터(`face_nodes_set[a] & face_nodes_set[b]`)를 통해 자가 접촉 후보군에서 사전 배제.
- **Narrow-Phase (정밀 접촉 역학)**:
  - Quad4 (2×2 Gauss-Legendre) 및 Tri3 (면적 가중 무게중심) 정밀 적분.
  - 법선 대향 필터($\mathbf{n}_a \cdot \mathbf{n}_b < -0.05$)로 배면 허위 접촉 차단.
  - 대칭 랭크-1 외적 강성 합성:
    $$\mathbf{K}_c = \sum_{qp} \gamma \cdot w_{qp} (\mathbf{c}_{qp} \otimes \mathbf{c}_{qp}^T)$$
  - 라그랑주 승수 없이 완전 대칭(SPD) 유지 $\implies$ PARDISO Cholesky ($LL^T$) 고속 분해 완벽 호환.

### 2.3 Abaqus Python API 계층 연동
- [`dispsolver/model/model.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/model/model.py):
  - `model.ContactProperty()`, `model.ContactStd()`, `model.SurfaceToSurfaceContactStd()` 팩토리 등록.
  - `model.create_solver3d()`에서 `sys.constraints` 및 활성 단계(`Step`)의 상호작용(`step.interactions`)을 `solver.constraints`로 자동 바인딩.
- [`dispsolver/model/part.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/model/part.py) & [`dispsolver/model/instance.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/model/instance.py):
  - `part.get_exterior_faces()` 및 `part.exterior` 추가.
  - `instance.get_exterior_faces()` 및 `instance.exterior` 추가.
- [`dispsolver/solver3d/dynamic3d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/solver3d/dynamic3d.py):
  - `solve()` 진입 시 스텝 내 활성 `GeneralContact3D`를 자동 등록하여 해석 루프에서 실시간 조립.

---

## 3. 검증 결과

### 3.1 신규 Abaqus API 전수 검증 (`tests/test_abaqus_contact_api.py`)
| 테스트 케이스 | 검증 항목 | 결과 |
|:---|:---|:---:|
| `test_auto_detect_exterior_3d_c3d8` | 2×2×2 C3D8 격자에서 24개 외곽 Quad4 면 및 외향 법선 정밀도 | **PASS** |
| `test_auto_detect_exterior_3d_c3d4` | 공유면을 가진 2개 C3D4 사면체에서 6개 외곽 Tri3 면 추출 | **PASS** |
| `test_part_exterior_methods` | `part.get_exterior_faces()` 및 `part.exterior` 동작 검증 | **PASS** |
| `test_abaqus_contact_python_api_declarations` | `ContactProperty`, `ContactStd`, `SurfaceToSurfaceContactStd` 선언 | **PASS** |
| `test_general_contact_all_exterior_two_cubes_solve` | **면 지정 없는** 접근하는 2개 큐브 압축 접촉 비선형 해석 완주 | **PASS** (6 iters) |
| `test_general_contact_part_exterior_pairs` | `includedPairs=[(top.exterior, bot.exterior)]` 명시적 파트 접촉 | **PASS** |

### 3.2 전체 회귀 테스트 (31/31 통과)
```text
tests/test_abaqus_contact_api.py::test_auto_detect_exterior_3d_c3d8 PASSED
tests/test_abaqus_contact_api.py::test_auto_detect_exterior_3d_c3d4 PASSED
tests/test_abaqus_contact_api.py::test_part_exterior_methods PASSED
tests/test_abaqus_contact_api.py::test_abaqus_contact_python_api_declarations PASSED
tests/test_abaqus_contact_api.py::test_general_contact_all_exterior_two_cubes_solve PASSED
tests/test_abaqus_contact_api.py::test_general_contact_part_exterior_pairs PASSED
tests/test_bvh_tree3d.py (5 tests) PASSED
tests/test_nitsche_contact3d.py (3 tests) PASSED
tests/test_emcs_contact3d.py (5 tests) PASSED
tests/test_dual_mortar_2d.py (5 tests) PASSED
tests/test_cae_model_hierarchy.py (4 tests) PASSED
tests/test_multi_step_branching.py (3 tests) PASSED
======================= 31 passed in 37.58s ========================
```
기존의 Dual Mortar, Nitsche, EMCS, CAE 모델 계층 구조, 멀티스텝 분기 엔진 전반에 걸쳐 **회귀 오류 0건**을 달성했습니다.
