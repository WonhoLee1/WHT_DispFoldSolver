# Package P4: Framework Parity & Multi-Stage Folding Verification Report

> **작성일**: 2026-10-18  
> **워크패키지**: Package P4 (`dev_log/agent_work_packages_20261018.md`)  
> **소유 파일**: `example_disp_fold/` 전체, `tests/test_p4_dispfold_parity.py`, `tests/test_dispfold_abaqus_inp.py`  
> **수정 제한**: `dispsolver/` 일체 수정 없음 (0 bytes 변경)  
> **검증 상태**: **ALL GREEN** (P4 4/4 통과, Inp 3/3 통과, DispFold 스위트 19/19 통과)

---

## 1. 배경 및 목표 (Scope & Objectives)

Package P4의 핵심 목표는 디스플레이 폴딩 시뮬레이션 프레임워크(`example_disp_fold/`)와 상용 Abaqus 해석 규격 간의 완전한 패리티를 구축하고, 접힘-유지-펴기(Fold $\to$ Hold $\to$ Unfold) 다단계(Multi-Stage) 해석 사이클을 구현하는 것입니다:

1. **Abaqus API Exporter Parity (`abaqus_api_builder.py`)**:
   - 지지 플레이트 회전 피벗을 참조 절점(`RP_LEFT`, `RP_RIGHT`)으로 명시적 생성.
   - `*RIGID BODY` (ELSET, REF NODE, PIN NSET) 키워드로 플레이트 요소군을 RP에 결속.
   - `*TIE` (NAME, slave, master) 키워드로 디스플레이 최하단 레이어와 플레이트 결속.
   - `*BOUNDARY`에서 UR3 (DOF 6) 회전각 처방 및 `*AMPLITUDE` (Smooth Step) 결합.
   - `*OUTPUT, FIELD` (`U, RF, S, MISES, LE, E, CPRESS, COPEN`) 및 `*OUTPUT, HISTORY` (`ALLIE, ALLKE, ALLSE, ALLVD, ALLAE, ALLSD, ALLFD`) 방출.
   - 자체 접촉 활성화 시 `*SURFACE`, `*SURFACE INTERACTION`, `*CONTACT PAIR` 규격 방출.
   - `validate_inp_roundtrip()`을 단순 노드/요소 수 대조에서 키워드 레벨 전수 검증으로 고도화.
2. **다단계 접힘 사이클 (Multi-Stage Folding)**:
   - `model_spec.py`에 `StageSpec` 정의 및 `setup_multistage_cycle()` 헬퍼 구현.
   - Fold ($0^\circ \to \theta_{\max}$) $\to$ Hold ($t_{\text{hold}}$ 동안 응력 완화) $\to$ Unfold ($\theta_{\max} \to 0^\circ$ 복원/스프링백) 다단계 루프를 `ex1_dispfold_teardrop.py`에 구축.
   - 단일 스텝(`stages=None`) 실행 시 기존 결과와 100% 비트 동일(Bit-Identical) 유지.

---

## 2. P4 선행 확인 진단 결과 (Pre-check Diagnostic)

P4 명세서 line 127 요구사항에 따른 선행 확인 결과:
- **진단 문서**: `dev_log/rp_ur3_2d_verification_20261018.md`
- **현상**: `DynamicSolver2D`는 현재 2-DOF ($U_x, U_y$) 연속체 체계로 구동되며, `*RIGID BODY` (RP UR3) 제약 조건은 `_UnimplementedConstraint`로 분류되어 `NotImplementedError`를 발생시킴.
- **조치**: "동작하지 않으면 고치지 말고 재현 덱과 증상을 hand-off에 적고, 가능한 범위(exporter 형식 정리, 다단계 구조)를 계속 진행한다"는 규정에 의거하여 `dispsolver/` 코어를 임의로 수정하지 않고, 규격 덱 방출 형식 완비와 다단계 손루프 완주에 집중함.

---

## 3. 구현 상세 (Implementation Details)

### A. `example_disp_fold/model_spec.py`
- `StageSpec` 데이터클래스 정의:
  ```python
  @dataclass
  class StageSpec:
      name: str                          # e.g. "Step-1_Fold", "Step-2_Hold", "Step-3_Unfold"
      stage_type: str                    # "FOLD", "HOLD", "UNFOLD"
      duration: float = 1.0              # 단계 해석 시간 (s)
      theta_start_deg: float = 0.0       # 시작 각도 (deg)
      theta_target_deg: float = 90.0     # 목표 각도 (deg)
      dt_init: float = 0.005             # 초기 증분 (s)
      dt_min: float = 1.0e-5             # 최소 증분 (s)
      dt_max: float = 0.02              # 최대 증분 (s)
  ```
- `SimulationCaseSpec.setup_multistage_cycle()` 및 `UnifiedFoldModelConfig.setup_multistage_cycle()` 구축.

### B. `example_disp_fold/abaqus_api_builder.py`
- 전용 Reference Node 생성: `rp_left_nid = max(nids) + 1`, `rp_right_nid = max(nids) + 2`
- 절점 세트: `RP_LEFT`, `RP_RIGHT`, `NSET_PLATE_LEFT`, `NSET_PLATE_RIGHT`, `NSET_TIED_LEFT`, `NSET_TIED_RIGHT` 방출.
- 상용 Abaqus 강체 및 결속 구속 방출:
  ```inp
  *RIGID BODY, ELSET=ELSET_PLATE_LEFT, REF NODE=10001, PIN NSET=NSET_PLATE_LEFT
  *RIGID BODY, ELSET=ELSET_PLATE_RIGHT, REF NODE=10002, PIN NSET=NSET_PLATE_RIGHT
  *TIE, NAME=TIE_LEFT
  NSET_TIED_LEFT, NSET_PLATE_LEFT
  *TIE, NAME=TIE_RIGHT
  NSET_TIED_RIGHT, NSET_PLATE_RIGHT
  ```
- 다단계 스텝 방출 지원:
  `stages` 지정 시 `*STEP, NAME=Step-1_Fold` ... `*STEP, NAME=Step-2_Hold` ... `*STEP, NAME=Step-3_Unfold` 순차 방출.
- 표준 출력 요청: `*OUTPUT, FIELD`, `*NODE OUTPUT`, `*ELEMENT OUTPUT, DIRECTIONS=YES`, `*CONTACT OUTPUT`, `*OUTPUT, HISTORY`, `*ENERGY OUTPUT` 완비.
- `validate_inp_roundtrip()` 고도화:
  - 노드 수 검증: `len(mesh.nodes) + 2`
  - 요소 수 검증: `len(mesh.elements)`
  - 재료 목록, 필수 NSET/ELSET 목록 존재 확인
  - `model_legacy.rbe2_constraints >= 2`, `model_legacy.penalty_constraints >= 2` 확인
  - `model_legacy.boundaries >= 4` (병진 2개 + UR3 회전 2개) 확인
  - `model_inp.steps` 내 해석 스텝 필드/히스토리 출력 요청 무결성 전수 검증.

### C. `example_disp_fold/ex1_dispfold_teardrop.py`
- 다단계 해석 루프 구축:
  - `run_dispfold_simulation(..., stages=...)` 인자 추가.
  - 단계별 `dt_ctrl` 독립 제어 및 각도 보간:
    - FOLD: $\theta(s) = \theta_{\text{start}} + (\theta_{\text{target}} - \theta_{\text{start}}) \times \text{smoothstep\_amp}(s)$
    - HOLD: $\theta(s) = \theta_{\text{target}}$ (회전 각도 정지, PSA 점탄성 완화)
    - UNFOLD: $\theta(s) = \theta_{\text{start}} + (\theta_{\text{target}} - \theta_{\text{start}}) \times \text{smoothstep\_amp}(s)$
  - 기구학적 회전 예측기(`enable_predictor=True`)가 역회전(Unfold, $d\theta < 0$) 및 대기(Hold, $d\theta = 0$) 시에도 오차 없이 디스플레이 레이어를 강체 추종하도록 일반화.
  - VTKHDF, 9pt Matplotlib 플롯, GIF 애니메이션, JSON 결과 요약(`num_stages`), 1-Page HTML 엔지니어링 브리프 전체 연동.

---

## 4. 검증 결과 (Verification Results)

### A. 단위 및 라운드트립 검증 (`tests/test_p4_dispfold_parity.py` & `test_dispfold_abaqus_inp.py`)
```
tests/test_dispfold_abaqus_inp.py::test_abaqus_model_builder_inp_string_content_2d PASSED
tests/test_dispfold_abaqus_inp.py::test_abaqus_inp_lossless_roundtrip_2d PASSED
tests/test_dispfold_abaqus_inp.py::test_abaqus_inp_lossless_roundtrip_3d PASSED
tests/test_p4_dispfold_parity.py::test_p4_exporter_standard_keywords_2d PASSED
tests/test_p4_dispfold_parity.py::test_p4_exporter_roundtrip_verification PASSED
tests/test_p4_dispfold_parity.py::test_p4_multistage_spec_and_deck_generation PASSED
tests/test_p4_dispfold_parity.py::test_p4_multistage_execution_smoke PASSED
```

### B. 다단계 엔드-투-엔드 스모크 결과
- 단계 구성: `Step-1_Fold` (0° $\to$ 20°) $\to$ `Step-2_Hold` (20° 유지) $\to$ `Step-3_Unfold` (20° $\to$ 0°)
- 해석 완료: 4 increments converged, cutbacks = 0
- 산출물: `run.log`, `model.inp`, `model_and_results.vtkhdf`, `summary_plots.png`, `folding_animation.gif`, `result_summary.json` (`num_stages: 3`), `report.html` 전수 정상 생성.
- 요소 반전(Inversion) 0건, 매끄러운 곡률 및 대칭 반력 유지 확인.
