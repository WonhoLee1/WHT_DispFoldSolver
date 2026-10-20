# 3D 깊이 방향 요소 수 증가 케이스 등각투영(Isometric) 뷰 캡처 계획서

## 1. 개요 및 목적 (Goal Description)
본 계획은 3D 폴딩 벤치마크 케이스 중 **폭 방향(Z-depth) 요소 수 스케일링 케이스($nz = 1, 10, 20$)**에 대해, 2D 정면도(XY 평면) 대신 **3D 등각투영 뷰(Isometric 3D View)**로 고해상도 시각화를 렌더링하는 것을 목적으로 합니다.
이를 통해 폭 방향(1.0 mm)을 따라 요소가 1개, 10개, 20개로 분할되는 메쉬 밀도 차이와 3D 가장자리 경계층 효과(Anticlastic Poisson Bulging)를 명확히 시각화합니다.

---

## 2. 시각화 및 카메라 파라미터 설계 (Camera & Style Standards)
1. **PyVista ParaView 다크 스타일 준수**:
   - Background: Black (`#000000`), Font: White (`#ffffff`, 12pt).
   - Mesh Edges: Dark Gray (`#444444`, line_width=1.0)를 렌더링하여 **$Z$축 방향 요소 분할 선(1개 $\to$ 10개 $\to$ 20개)이 선명하게 보이도록 구성**.
   - 단 1개의 컬러바 (우측 von Mises 응력 뷰에만 배치, 좌측 재질 뷰는 컬러바 비활성화).
   - 3D 좌표축(Coordinate Axes: $X, Y, Z$) 표시.
2. **카메라 구도 (3D Isometric Setup)**:
   - **좌측 뷰포트 (전체 3D Iso 뷰)**:
     - 구도: 피벗 중심 위쪽에서 비스듬히 내려다보는 3D 등각 뷰 (`azimuth = 30°`, `elevation = 20°`)
     - 표시 내용: 재질 구분 (Gray=Plate, Cyan=Glass, Blue=PET, Orange=PSA) + 3D 메쉬 엣지
   - **우측 뷰포트 (힌지 Tear-Drop 3D Iso 루프 확대 뷰)**:
     - 구도: 힌지 만곡부($X \in [-5, +5]\,\text{mm}, Z \in [-0.5, +0.5]\,\text{mm}$) 중심의 초점 확대
     - 표시 내용: von Mises 응력 [MPa] (터보 컬러맵, 0~2,800 MPa)

---

## 3. 대상 케이스 목록 (6개 케이스 + 3D 기본 케이스)

### [Group 1] 100% 평면 내 메쉬 기준 깊이 스케일링
1. `FISTR_3D_Depth100_nz1` (360 요소, $nz=1$)
2. `FISTR_3D_Depth100_nz10` (3,600 요소, $nz=10$)
3. `FISTR_3D_Depth100_nz20` (7,200 요소, $nz=20$)

### [Group 2] 50% 평면 내 메쉬 기준 깊이 스케일링
4. `FISTR_3D_Depth50_nz1` (720 요소, $nz=1$)
5. `FISTR_3D_Depth50_nz10` (7,200 요소, $nz=10$)
6. `FISTR_3D_Depth50_nz20` (14,400 요소, $nz=20$)

---

## 4. 변경 및 신규 파일 구조 (Proposed Changes)

### [NEW] `example_disp_fold/capture_3d_depth_iso_pyvista.py`
- PyVista 기반 3D Isometric 오프스크린 렌더러 함수 `render_pyvista_iso_capture()` 구현.
- 6개 깊이 방향 케이스에 대해 `_iso.png` 고해상도 이미지(1920×1000) 저장.
- $nz=1, 10, 20$의 깊이 방향 메쉬 밀도 변화를 한눈에 볼 수 있는 **3단 비교 몽타주(`montage_3d_depth_iso_nz1_10_20.png`)** 자동 합성.

### [MODIFY] `dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md`
- 제3절(3D 폴딩 깊이 방향 요소 수 스케일링) 하단에 3D Iso 몽타주 및 개별 케이스 캡처 이미지 링크 삽입.

### [MODIFY] `walkthrough.md`
- 3D 깊이 스케일링 캐러셀에 신규 생성된 3D Iso 뷰 캡처 추가.

---

## 5. 검증 계획 (Verification Plan)
1. **스크립트 실행**: `python example_disp_fold/capture_3d_depth_iso_pyvista.py` 실행.
2. **산출물 확인**:
   - `dev_log/figures/cases/FISTR_3D_Depth100_nz*.png` (Iso 뷰) 3종 생성.
   - `dev_log/figures/cases/FISTR_3D_Depth50_nz*.png` (Iso 뷰) 3종 생성.
   - `dev_log/figures/montage_3d_depth_iso_nz1_10_20.png` 몽타주 생성.
3. **시각적 품질 검증**:
   - 폭 방향(Z)을 따라 메쉬 선이 1개, 10개, 20개로 뚜렷하게 나뉘어 보이는지 육안 확인.
