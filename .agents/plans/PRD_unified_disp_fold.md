# 📄 PRD: Unified 2D/3D Display Folding Solver Framework (`example_disp_fold`)
**Document ID**: PRD-DISPFOLD-20261018-V1  
**Date**: 2026-10-18  
**Author**: Antigravity Multi-Agent Engineering Team  
**Status**: APPROVED & IN-PROGRESS  
**Target Module**: `example_disp_fold/`  

---

## 1. Executive Summary & Vision (제품 비전 및 개요)
본 제품은 플렉서블 디스플레이의 180° 티어드롭(Teardrop) 대변형 폴딩 해석을 위한 **차세대 2D/3D 유한요소 해석 통합 프레임워크**입니다.
기존에 `ex12`(2D)와 `ex13`(3D)로 분리되어 코드 중복과 유지보수 오버헤드를 유발하던 구조를 단일 진실 공급원(SSOT) 기반의 `example_disp_fold/` 패키지로 전면 통합하고, Abaqus Python API 수준의 모델링 추상화, Gmsh 기반 적응형 편향 격자 생성, Kitware VTKHDF v2.2 단일 파일 데이터베이스, PySide6 + PyVistaQT 통합 CAE 뷰어 및 1-Page 자동 엔지니어링 브리프 리포트를 일체형으로 제공합니다.

---

## 2. User Personas & Core Workflows (사용자 페르소나 및 워크플로우)
1. **플렉서블 디스플레이 구조/역학 연구원 (Display Mechanics Researcher)**
   - 다양한 적층 구조(PET, PSA, UTG/Glass 등 4~14층)의 굽힘 강성($D_{\text{slip}}$, $D_{\text{mono}}$), 곡률반경($R_{\text{apex}}$), 층간 슬립, 힌지 구동 반력 모멘트를 신속히 해석하고 설계 마진을 검증.
2. **CAE 해석 엔지니어 (Finite Element Analyst)**
   - 표준 상용 솔버(Abaqus)와의 1:1 대조를 위한 `.inp` 파일 출력, Abaqus Python API 스크립팅 연동, ParaView 직접 로딩을 위한 VTKHDF v2.2 포맷 활용.
3. **해석 프로젝트 매니저 / 리뷰어 (Project Manager / Reviewer)**
   - 해석 완료 직후 생성되는 1-Page HTML Engineering Brief, GIF 애니메이션, 통계 JSON을 통해 해석 수렴성, 물리적 수치 안정성, 에너지 보존을 직관적으로 확인.

---

## 3. Functional Requirements (기능 요구사항 - FR)

### FR-01: 단일 진실 공급원(SSOT) 모델 사양 (`model_spec.py`)
- **FR-01.1**: 레이어별 두께, 탄성계수($E$), 포아송비($\nu$), 항복강도($\sigma_y$), 경화계수($H$), 초탄성/점탄성 물성($G_0$, $K$), 요소 타입(2D/3D)을 객체지향형 `LayerSpec` 및 `DisplayStackConfig`로 관리.
- **FR-01.2**: 표준 14층 정밀 모델($t = 0.630\,\text{mm}$), 4층 콤팩트 모델(`compact_4`), U자 굽힘(`u_shape`), 수적형 갭(`compact_waterdrop`), 영간극(`crease_zero_gap`) 프리셋 내장.
- **FR-01.3**: 자유 슬립 굽힘 강성 $D_{\text{slip}} = \sum \frac{E'_i b t_i^3}{12}$ 및 완전 접착 굽힘 강성 $D_{\text{mono}} = \sum E'_i (I_{0,i} + A_i (y_i - y_{NA})^2)$의 해석적 닫힌해 자동 계산.
- **FR-01.4**: 힌지 회전 시 자가 접촉(Self-Contact) 발생 여부를 판정하는 지능형 접촉 가드(`requires_self_contact()`) 제공.

### FR-02: Gmsh 기반 2D/3D 고품질 편향 격자 생성 (`gmsh_mesh_generator.py`)
- **FR-02.1**: Gmsh Python API를 활용하여 힌지 구간(고밀도 곡률부)과 외곽 평탄부(점진적 완화 편향) 간의 부드러운 전이 격자 생성.
- **FR-02.2**: 2D 사각 요소(CPE4/CPE4I/CPE4H) 및 3D 육면체 요소(C3D8/C3D8I/C3D8H) 정밀 격자화.
- **FR-02.3**: 테이프 보이드(Tape Void, 힌지 구간 무요소 영역)의 정확한 위상 처리.
- **FR-02.4**: 서포트 플레이트(좌/우 강체 가이드) 생성 및 면 절점 위상 기반 단위 외향 법선(Outward Normal) 보장, 음체적(`det J <= 0`) 원천 방지.

### FR-03: Abaqus Python API 패리티 및 INP 키워드 생성기 (`abaqus_api_builder.py`)
- **FR-03.1**: Abaqus 파이썬 API (`Part`, `Assembly`, `Step`, `Interaction`, `BoundaryCondition`, `FieldOutputRequest`)와 1:1 매핑되는 빌더 클래스 구조.
- **FR-03.2**: Abaqus 표준 키워드(`.inp`) 생성 및 내장 파서(`read_abaqus_input`)를 통한 무손실 라운드트립(Roundtrip) 검증 통과.

### FR-04: Kitware VTKHDF v2.2 전처리/후처리 데이터베이스 (`vtkhdf_unified_io.py`)
- **FR-04.1**: 단일 `.vtkhdf` HDF5 파일 내에 전처리 메쉬, 절점/요소 세트(`PreProcessing/Sets`), 파트 정보(`Part_ID`, `Layer_ID`, `Material_Type`) 계층적 저장.
- **FR-04.2**: 비선형 과도 해석 진행에 따른 스텝별 변위($U$), 응력($S$), 변형률($LE$), 소성변형률($PEEQ$) 스트리밍 저장 (`Steps`, `Transient` 데이터 그룹).

### FR-05: 엔드투엔드 솔버 파이프라인 (`ex1_dispfold_teardrop.py`)
- **FR-05.1**: 2D 및 3D 해석을 CLI 플래그(`--dim 2D`, `--dim 3D`) 및 프로그래밍 API로 단일 구동.
- **FR-05.2**: 모든 터미널 출력을 콘솔과 로그 파일(`run.log`)에 실시간 동시 기록하는 `TeeLogger`.
- **FR-05.3**: 타임스탬프 결과 폴더(`results/DYYYYMMDD_HHMMSS/`) 자동 생성 및 다음 산출물 패키징:
  - `run.log`: 전체 수치 해석 터미널 로그
  - `model.inp`: Abaqus 호환 키워드 데크
  - `model_and_results.vtkhdf`: VTKHDF v2.2 전후처리 데이터베이스
  - `summary_plots.png`: 4분할 엔지니어링 성능 대시보드
  - `folding_animation.gif`: 폴딩 대변형 애니메이션
  - `result_summary.json`: 정량적 KPI 및 수렴성 지표 요약
  - `report.html`: 1-Page 인터랙티브 HTML 엔지니어링 브리프
- **FR-05.4**: 기구학적 회전 예측자(Kinematic Rotation Predictor) On/Off 스위치(`--enable_predictor` / `--no_predictor`).

### FR-06: PySide6 + PyVistaQT 통합 뷰어 (`viewer_qt_app.py`)
- **FR-06.1**: 좌측 3D/2D 인터랙티브 뷰어(`QtInteractor`) + 우측 Matplotlib 엔지니어링 패널(`FigureCanvasQTAgg`).
- **FR-06.2**: Matplotlib 패널은 Noto Sans KR 9pt 폰트, 한국어 지원, 상단/하단 반응 곡선 및 Elastica 이론 한계선($D_{\text{slip}}$ vs $D_{\text{mono}}$) 동시 플롯.
- **FR-06.3**: 시간 슬라이더 연동 3D 변형 메쉬-2D 그래프 마커 실시간 동기화.
- **FR-06.4**: 레이어별 가시성(Visibility) 토글 체크리스트 및 투명도 조절 슬라이더.

---

## 4. Non-Functional Requirements (비기능 요구사항 - NFR)
- **NFR-01: 수치 무결성 및 안티 모킹 (Anti-Mocking Protocol)**:
  - 어떠한 수치해석, 강성 행렬, 잔차, 응력 필드에도 임의의 난수(`np.random`)나 하드코딩된 더미 모킹 데이터를 일절 배제.
  - 모든 수식은 Bathe, Belytschko, Timoshenko 등 공인 원전에 입각한 엄밀 전개.
- **NFR-02: Karpathy AI 엔지니어링 4대 원칙 준수**:
  - Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution.
- **NFR-03: 플랫폼 및 환경 호환성**:
  - Windows PowerShell 7 환경 지원, 명시적 UTF-8 인코딩, BOM 없는 UTF-8 파일 저장.
- **NFR-04: 테스트 주도 개발 (TDD Red-Green Loop)**:
  - 모든 기능은 공용 경계 인터페이스(Seam)에 대한 단위/통합 테스트를 우선 작성하고 통과 여부를 검증.

---

## 5. Architectural Seams & TDD Test Matrix (경계 인터페이스 및 테스트 계획)

| Seam ID | 모듈 / 인터페이스 | 검증 내용 | 검증 테스트 파일 |
|:---:|:---|:---|:---|
| **Seam-1** | `model_spec.py` | 14층/4층 적층 유효성, 물성 일관성, $D_{\text{slip}}$, $D_{\text{mono}}$ 해석해, 접촉 가드 | `tests/test_dispfold_model_spec.py` |
| **Seam-2** | `gmsh_mesh_generator.py` | 2D(CPE4/CPE4I/CPE4H), 3D(C3D8/C3D8H) 격자 연결성, 외향 법선, 양체적 | `tests/test_dispfold_gmsh.py` |
| **Seam-3** | `abaqus_api_builder.py` | 키워드 생성 및 `read_abaqus_input` 라운드트립 무손실 파싱 | `tests/test_dispfold_abaqus_inp.py` |
| **Seam-4** | `vtkhdf_unified_io.py` | VTKHDF v2.2 HDF5 스키마, 파트/세트 및 과도 필드 데이터 I/O | `tests/test_dispfold_vtkhdf.py` |
| **Seam-5** | `ex1_dispfold_teardrop.py` | 2D/3D 솔버 루프 실행, 뉴턴 수렴성, 6대 산출물(log, inp, vtkhdf, png, gif, html) 완결성 | `tests/test_dispfold_solver_e2e.py` |
| **Seam-6** | `viewer_qt_app.py` | PySide6 + PyVistaQt 뷰어 인스턴스화, 데이터 연동, 헤드리스 렌더링 무오류 | `tests/test_dispfold_gui.py` |
