# 각 벤치마크 케이스별 고해상도 시각화 이미지 캡처 계획서

## 1. 개요 및 목적 (Goal Description)
본 계획은 이전 단계에서 해석 완료된 **총 17개 벤치마크 케이스**(FrontISTR 2D 5단계, FrontISTR 3D 9개 케이스, WHT 2D 3개 케이스)에 대해, **PyVista 및 Matplotlib 기반의 표준화된 고해상도 시각화 이미지(ParaView Dark Style)**를 일괄 자동 캡처하고, 이를 `dev_log/figures/cases/`에 저장한 뒤 보고서(`dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md`)에 체계적으로 연동하는 것을 목적으로 합니다.

---

## 2. 시각화 표준 규칙 준수 (Visualization Standards)
사용자 정의 가이드라인 및 ParaView Dark Style 원칙을 100% 준수합니다:
1. **배경 및 글자색**: Black 배경 (`#000000`), White 폰트 (`#ffffff`), 기본 폰트 크기 **12pt**.
2. **메쉬 엣지 표시**: Dark Gray (`#444444`) 선으로 요소 경계를 명확히 렌더링하여 **메쉬 세분화(100% $\to$ 6.25%) 및 3D 깊이 분할($nz=1, 10, 20$) 차이가 한눈에 시각적으로 구별**되도록 구성.
3. **컬러바 규칙**: 뷰포트당 최대 **1개의 컬러바**만 배치 (좌측 재질 뷰는 컬러바 비활성화, 우측 von Mises 응력 뷰에만 1개 배치).
4. **좌표축 표시**: 좌/우 뷰포트 모두 직관적인 3D/2D 좌표축(Coordinate Axes) 활성화.
5. **듀얼 뷰포트 구성**:
   - **좌측 뷰포트**: 180° 폴딩 전체 형상 (재질 구분 렌더링: 회색=플레이트, 청록=커버글래스, 파랑=PET, 주황=PSA)
   - **우측 뷰포트**: 힌지 영역 Tear-Drop 오메가 루프 확대 뷰 (von Mises 응력 컨투어 + 터보 컬러맵)

---

## 3. 대상 캡처 케이스 목록 (17개 케이스)

### [Group A] FrontISTR 2D 메쉬 세분화 5단계 (Z=1층 압출 평면변형률)
1. `FISTR_2D_Refine_100.0pct` (360 elems, 962 nodes)
2. `FISTR_2D_Refine_50.0pct` (720 elems, 1,902 nodes)
3. `FISTR_2D_Refine_25.0pct` (1,440 elems, 3,782 nodes)
4. `FISTR_2D_Refine_12.5pct` (2,880 elems, 7,542 nodes)
5. `FISTR_2D_Refine_6.25pct` (5,760 elems, 15,062 nodes — 초고해상도 메쉬)

### [Group B] FrontISTR 3D 평면 내 세분화 (nz=2)
6. `FISTR_3D_InPlane_100.0pct` (720 elems)
7. `FISTR_3D_InPlane_50.0pct` (1,440 elems)
8. `FISTR_3D_InPlane_25.0pct` (2,880 elems)

### [Group C] FrontISTR 3D 폭 방향(Z-depth) 스케일링 (nz = 1, 10, 20)
9. `FISTR_3D_Depth100_nz1` (360 elems, 962 nodes)
10. `FISTR_3D_Depth100_nz10` (3,600 elems, 5,291 nodes)
11. `FISTR_3D_Depth100_nz20` (7,200 elems, 10,101 nodes — 고밀도 3D)
12. `FISTR_3D_Depth50_nz1` (720 elems, 1,902 nodes)
13. `FISTR_3D_Depth50_nz10` (7,200 elems, 10,461 nodes)
14. `FISTR_3D_Depth50_nz20` (14,400 elems, 19,971 nodes — 초고밀도 3D)

### [Group D] WHT Solver 2D 대조 케이스
15. `WHT_2D_Refine_100.0pct` (360 elems)
16. `WHT_2D_Refine_50.0pct` (720 elems)
17. `WHT_2D_Refine_25.0pct` (1,440 elems)

---

## 4. 구현 및 변경 계획 (Proposed Changes)

### [NEW] `example_disp_fold/capture_benchmark_cases_pyvista.py`
- 각 케이스의 결과 파일(`fistr_teardrop.res.0.30` 또는 WHT 상태 벡터)을 직접 읽어 변형 형상 그리드 생성.
- PyVista 오프스크린 렌더러(`window_size=(1920, 1000)`)로 각 케이스별 듀얼 뷰포트 고해상도 PNG 이미지 렌더링.
- 2D 비교 및 3D 깊이 비교를 직관적으로 비교할 수 있는 몽타주(Composite Grid) 이미지 추가 생성:
  - `montage_2d_refinement_5steps.png` (5단계 메쉬 밀도 나란히 비교)
  - `montage_3d_depth_scaling_nz1_10_20.png` (3D 등각 투영 뷰에서 폭 방향 분할 비교)

### [MODIFY] `dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md`
- 각 섹션(2D 세분화, 3D 깊이 스케일링, WHT 대조) 하단에 생성된 캡처 이미지를 캐러셀 또는 갤러리 형태로 배치하여 보고서의 시각적 완성도 극대화.

---

## 5. 검증 계획 (Verification Plan)
1. **스크립트 구동**: `python example_disp_fold/capture_benchmark_cases_pyvista.py` 실행
2. **산출물 확인**:
   - `dev_log/figures/cases/` 내 17개 개별 PNG 파일 정상 생성 확인
   - 몽타주 이미지 정상 생성 확인
3. **규칙 점검**:
   - 검은 배경, 흰색 텍스트(12pt), 짙은 회색 메쉬 엣지, 컬러바 단 1개 적용 여부 전수 검증
