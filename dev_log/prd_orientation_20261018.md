# PRD: `*ORIENTATION` 연결 (3D 직교이방성 + 재료 방향)

**작성일**: 2026-10-18
**목록 번호**: `abaqus_feature_gap_dispfold_20261018.md` 12번
**브랜치**: `feat/p7-restart-orientation`
**상태**: 구현 완료 (2026-10-18). `tests/test_orientation.py` 11개 통과.

## 1. 배경

PET·PI 등 연신 필름은 방향성이 있다(MD/TD 탄성계수 차). 방향이 있는 재료를 Abaqus 방식(`*ELASTIC, TYPE=ENGINEERING CONSTANTS` + `*ORIENTATION` + `*SOLID SECTION, ORIENTATION=`)으로 쓸 수 있어야 한다. 지금 폴더블 프레임워크의 재료는 모두 등방성이라 당장의 모델에는 영향이 없고, 이방성 필름을 도입하는 시점의 선행 조건이다.

## 2. 현황 (코드 확인)

- 리더는 `*ORIENTATION`을 파싱해 `MaterialOrientation.frame`(3×3 방향 코사인, RECTANGULAR/COORDINATES만)을 저장하고, 섹션에 `orientation_name`을 연결한다.
- `finalize_materials`(`io/keywords/material.py`)는 `ENGINEERING CONSTANTS` 재료에 방향이 걸린 섹션을 `orientation-not-connected`로 **거부**한다(침묵 아님). 등방성 재료는 정보 로그만.
- 3D: `ENGINEERING CONSTANTS`는 C3D20 전용 사설 경로(재료 id 17)로만 허용되고, 다른 요소는 `NotImplementedError`. 카탈로그는 이 상수를 `MAT_CUSTOM_ELASTIC`의 6×6 강성 `C`로 만들 수 있다(`material/catalog.py`).
- 2D: `ENGINEERING CONSTANTS` 전체가 `NotImplementedError`("not implemented in 2D"). 이방성 자체가 미구현.
- `*ORIENTATION`의 둘째 줄(추가 회전각)은 0이 아니면 `Unsupported`.

## 3. 요구사항

R1. 3D 모든 솔리드 요소(C3D8 계열, C3D4, C3D10, C3D6 등)에서 `ENGINEERING CONSTANTS` 재료가 방향 없이도 동작한다 (전역 축 = 재료 축). C3D20은 기존 사설 경로를 유지.
R2. `*ORIENTATION`(RECTANGULAR, COORDINATES 정의)이 걸린 섹션의 요소는 회전된 강성 `C' = R C Rᵀ`(4차 텐서 회전)를 쓴다. 프레임의 행 = 국부 축 1, 2, 3.
R3. `*ORIENTATION`의 추가 회전(둘째 줄: 회전 기준 국부 방향, 각도)을 지원한다.
R4. 지원하지 않는 정의(CYLINDRICAL, SPHERICAL, NODES, USER)는 지금처럼 명시적으로 거부하되 이유를 정확히 말한다("요소마다 달라지는 프레임은 미구현").
R5. 2D의 이방성은 이번 범위 밖: 거부 유지, 문서화.
R6. 등방성 재료에 방향이 걸린 경우는 지금처럼 정보 로그(영향 없음).

## 4. 비범위

- 2D 이방성 평면변형률 강성, 원통·구 좌표 방향, 요소별 방향 분포(`*DISTRIBUTION`), 대변형에서 방향이 재료와 함께 회전하는 처리는 co-rotational 요소의 국부 프레임에 맡김(요소 내부에서 강성이 국부 프레임에 있다는 가정 그대로).

## 5. 시임과 경계 분류

| 시임 | 경계 | 기대값의 독립 출처 |
|---|---|---|
| S1. 방향 없는 직교이방성 C3D8 | 프로세스 내 (실제 리더+솔버) | 직교이방성 강성의 닫힌 형식 (공학 상수에서 S를 만들어 numpy로 역행렬) |
| S2. 평면 내 회전 θ에서 C′₁₁, C′₁₂, C′₁₆ | 프로세스 내 | 고전 적층 이론의 변환식(Jones): C′₁₁=m⁴C₁₁+2m²n²(C₁₂+2C₆₆)+n⁴C₂₂ 등 |
| S3. 추가 회전각 | 프로세스 내 | S2와 같은 식에서 θ = 프레임 각 + 추가각 |
| S4. 거부 경로 (원통, 2D, 비등방+NODES) | 프로세스 내 | 요구사항 R4, R5 |

균일 변형장 u = ε·x을 모든 절점에 처방하고 반력으로 응력을 읽는다: 정의상 σ = C′ε이므로 요소 안의 어떤 적분 방식과도 무관하다.

## 6. 완료 기준

- S1~S4 먼저 실패하고 최소 수정으로 통과.
- 전체 `pytest` 실패 목록을 시작 전(깨끗한 HEAD)과 비교.
- `python -m verification.run_all --changed`.
- hand-off (AGENTS.md §3.2).

## 7. 결과 (2026-10-18)

- 신규 `dispsolver/material/orientation.py`(`orthotropic_stiffness`, `rotate_stiffness`, `rotate_frame`, `oriented_material`). 카탈로그의 직교이방성 강성 계산을 이 한 곳으로 통합.
- 3D: `ENGINEERING CONSTANTS` 재료가 C3D8 계열·C3D6·C3D4·C3D10 등 모든 솔리드 요소에서 동작 (이전에는 C3D20 사설 경로만). 섹션에 `ORIENTATION=`이 있으면 어셈블리가 회전된 `C_mat`을 가진 파생 재료를 쓴다. C3D20은 기존 경로 유지(방향이 걸리면 파생 재료를 만들지 않고 기존 동작).
- 검증: 균일 변형장 u=ε·x를 처방하고 반력으로 응력을 읽어 C′ε와 비교 (닫힌 형식 대비: 컴플라이언스 역행렬 + 4차 텐서 회전, 평면 내 C′₁₁ 변환식과 교차 확인). θ=30°, 90°, −45°, 추가 회전(둘째 줄) 로컬 축 3과 1. 회전 방향을 일부러 뒤집은 변이가 4개 테스트를 깨뜨리는 것을 확인.
- 리더: 추가 회전각 지원(`rotation_axis`, `rotation_angle`, 우수 법칙). 각도가 있는데 방향이 없으면 오류. 원통·구·NODES 정의와 2D는 정확한 사유와 함께 거부.
- 한계: (a) 요소마다 달라지는 프레임(CYLINDRICAL 등)은 미구현. (b) 2D 이방성은 강성 자체가 없음. (c) 카탈로그 리팩터에서 특이 행렬일 때 `eye*E1`로 대체하던 조용한 폴백을 없앴음(이제 예외). (d) C3D20 + 방향은 미연결.
