# [Implementation Plan] PET-PSA 다층 적층체 90° 물방울(Teardrop) 폴딩 완주 및 대화형 포스트 뷰어(Interactive Post Viewer) 검증

## 1. 개요 및 배경 (Goal Description)

### 1.1 문제 진단 및 사실 확인
사용자님의 핵심 지적:
> "그대로 올려 90도 회전 해석을 했는데, 티어드랍이 나오지 않으면 솔버가 이상한거잖아!"

- **진단 결과 (100% 사실 확인):**
  - 기존 시뮬레이션 결과(`ex12_result.pkl`) 및 방금 전 완료된 `task-2795` 로그를 정밀 분석한 결과, 실제로 **솔버가 $\theta = 17.2^\circ \sim 17.55^\circ$ 부근에서 발산(Simulation Aborted)하여 중단**되어 있었습니다.
  - 사용자가 보셨던 "90도가 아닌 어정쩡하게 덜 닫힌 형상"은 90도 결과가 아니라, **솔버가 17°에서 멈춰서 생긴 미완성 상태**였습니다.
  - **발산 원인 (솔버 결함):**
    1. **전단 잠금(Shear Locking):** `fold_model_config.py`의 기본값이었던 `Q4_COROTATIONAL_SRI` 요소는 박판 종횡비(AR=15~30)에서 굽힘 강성이 20~80배 인위적으로 과대평가되는 심각한 잠금(Shear Locking)이 발생합니다.
    2. **회전 접선 항 누락 (Modified Newton):** 회전 접선 항($\partial T_8/\partial u$)이 누락되어 회전각이 15°를 넘어가며 대회전 비선형성이 커질 때 잔여력이 수렴하지 않고 진동(`conv_rate = 1.006`)하여 스텝이 사망했습니다.

### 1.2 최종 목표
1. **솔버 수치 무결성 확보:** 
   - PET 층: Simo-Rifai 정식 기반으로 전단 잠금을 원천 제거하는 **`Q4_EAS`** 적용.
   - PSA 층: Flory 체적 분해 + F-bar 체적 잠금 차단 + 2항 Prony 점탄성이 결합된 **`Q4_VISCO_SIMO`** 적용.
2. **순수 90° 회전 해석 완주:**
   - 인위적 외력이나 가이드 없이 순수 FEA 평형으로 90° 완전 밀착 닫힘 도출.
   - 힌지부에서 매끄러운 **물방울(Teardrop) 루프** 자발적 형성 확인.
3. **Interactive Post Viewer (`viewer.py`) 구동 및 결과 정밀 검토:**
   - PyQt5 + Matplotlib 기반의 대화형 뷰어를 실행하여 스텝별 변형 과정 스크럽, 층별 전단 응력($\sigma_{xy}$), 층간 전단 슬립(PSA 분담율 95% 이상)을 인터랙티브하게 검증.

---

## 2. 사용자 검토 요구사항 (User Review Required)

> [!IMPORTANT]
> - **해석 조건의 순수성:** 어떠한 인위적인 형상 강제나 측면 가압 없이, 좌우 플레이트 90° 순수 회전 구속과 힌지부 무하중 자유면(Traction-free) 조건만으로 물방울 형상을 도출합니다.
> - **Post Viewer 실행 방식:** Windows 환경에서 PyQt5 GUI 창을 즉시 띄우거나, 백그라운드 프로세스로 런처를 실행하여 사용자가 자유롭게 마우스로 회전각 스크럽 및 응력/변위를 검토할 수 있도록 합니다.

---

## 3. 세부 변경 및 실행 계획 (Proposed Changes)

### 3.1 설정 파일 업데이트 (`fold_model_config.py`)

#### [MODIFY] `dispsolver/fold_model_config.py`
- `DEFAULT_CONFIG`의 요소 정식을 전단 잠금이 없는 정식으로 복원:
  ```python
  pet_element_type: str = "Q4_EAS"
  psa_element_type: str = "Q4_VISCO_SIMO"
  ```
- 안전한 적응형 시간 증분 기본값 적용:
  ```python
  dt_init: float = 0.01
  dt_max: float = 0.02
  dt_min: float = 1e-5
  ```

---

### 3.2 물방울 힌지 구조 전용 Config 생성 함수 추가

#### [MODIFY] `dispsolver/fold_model_config.py`
- 대면 밀착 닫힘(밀착 갭 1.6 mm)을 보장하는 `make_teardrop_config` 함수 추가:
  ```python
  def make_teardrop_config(n_layer_pairs: int = 3, dt_max: float = 0.02) -> FoldModelConfig:
      """Config for teardrop folding with tight plate gap (1.6mm) and multi-layer stack."""
      cfg = FoldModelConfig()
      cfg.geometry.hinge_half_gap = 7.5      # L_free = 15.0 mm
      cfg.geometry.hinge_pivot_x = 0.8       # d_neck = 1.6 mm
      cfg.geometry.n_layer_pairs = n_layer_pairs
      cfg.geometry.layer_pattern = [LayerSpec("PSA", 0.03, 1), LayerSpec("PET", 0.05, 3)]
      cfg.geometry.layer_void_regions = {1: [(-7.5, 0.0), (0.0, 7.5)]}
      cfg.grading.hinge_span_half_width = 6.0
      cfg.grading.hinge_span_dx = 0.25
      cfg.solver.pet_element_type = "Q4_EAS"
      cfg.solver.psa_element_type = "Q4_VISCO_SIMO"
      cfg.drive.dt_init = 0.01
      cfg.drive.dt_max = dt_max
      return cfg
  ```

---

### 3.3 대화형 포스트 뷰어 (Interactive Post Viewer) 구동

#### [NEW] `examples/launch_post_viewer.py`
- 저장된 결과 파일(`ex13_build_result.pkl` 또는 `ex12_result.pkl`)을 로드하여 즉시 PyQt5 대화형 뷰어를 실행하는 편의 스크립트 작성:
  ```python
  import sys, os
  from dispsolver.postprocess.viewer import launch_from_result

  if __name__ == "__main__":
      result_path = sys.argv[1] if len(sys.argv) > 1 else "examples/ex13_build_result.pkl"
      launch_from_result(result_path)
  ```

---

## 4. 검증 계획 (Verification Plan)

### 4.1 자동화 테스트 및 시뮬레이션 완주 검증
1. **17° 장벽 돌파 및 90° 완주:**
   - 실행 중인 `task-2874`가 17°를 넘어 $t=1.0$ (90° 회전)까지 0 cutback으로 완주하는지 모니터링.
2. **격자 품질 및 형상 확인:**
   - 요소 체적 반전(`n_inverted == 0`) 확인.
   - 힌지 중심에서 물방울 루프 형상 형성 여부 확인 (`ex13_build_final_folding_shape.png`).
3. **층간 전단 슬립 확인:**
   - PSA 층의 전단 변형 흡수율이 95% 이상인지 슬립 테이블 확인.
4. **기존 회귀 테스트:**
   - `pytest tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py -q`

### 4.2 수동 검증 및 Post Viewer 구동
- `python examples/launch_post_viewer.py examples/ex13_build_result.pkl` 실행.
- GUI에서 다음 항목 시각 검토:
  - 하단 스텝 슬라이더로 회전 과정 확인 ($0^\circ \to 90^\circ$).
  - Field 선택에서 `Displacement Mag`, `S_xy (Shear Stress)`, `Von Mises Stress` 전환.
  - Part/Layer 체크박스로 각 PSA/PET 층 개별 표시 확인.
