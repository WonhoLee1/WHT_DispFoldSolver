# Implementation Plan: 수렴성 저하 원인 규명 및 뉴턴-랩슨 고속 수렴성(2 Iterations/step) 복원 계획

사용자님의 날카로운 지적(**"수렴성이 되게 좋지 않아졌는데?"**)에 따라, 직전 솔버 거동 및 로그를 정밀 분석한 결과 **수렴성을 심각하게 악화시킨 2가지 근본 원인**을 명확히 규명하였습니다.

---

## 1. Problem Analysis (수렴성 저하의 근본 원인)

### 원인 1: 인위적 최대 변위 보정량 클램핑(`max_displacement_corr = 0.05 mm`)의 부작용 (주원인)
- **현상**: 매 반복마다 `Max Disp.Corr`가 정확히 `5.0000e-02` ($0.05\text{ mm}$)에 하드 락(Hard Lock)되어 풀리지 않음.
- **메커니즘**:
  - 직전 세션에서 극박 PSA 요소의 뒤집힘을 막겠다는 의도로 추가된 `dynamic.py`의 `max_displacement_corr = 0.05` 로직이 불평형 상태($R_{\text{norm}} > 1.0\text{ N}$)에서 **모든 자유 절점의 변위 수정 벡터 $\Delta u$를 강제로 0.05 mm로 축소**시켰습니다.
  - 비선형 대변형 유한요소 해석에서 굽힘 모멘트가 가해질 때 절점들은 한 스텝당 자연스럽게 0.1 ~ 0.5 mm 정도 이동해야 평형에 도달할 수 있습니다.
  - 그러나 매 반복마다 0.05 mm의 아기 걸음(baby steps)만 걷도록 강제되면서, **Newton-Raphson 고유의 2차 수렴성(Quadratic Convergence)이 완전히 파괴**되었습니다.
  - 그 결과, 스텝당 1~3회에 끝나야 할 수렴이 **15~25회까지 반복**을 소모하게 되었고, 반복수 초과로 인한 Cutback이 연쇄 발생하여 $dt$가 $0.00002\text{ s}$까지 추락하는 극심한 정체 현상이 빚어졌습니다.

### 원인 2: 시간 증분 상한(`dt_max`)의 불일치 (`0.1s` vs `0.01s`)
- **현상**: 기존 `ex12`가 과거 $17.5^\circ$에서 멈췄던 진짜 원인은 변위 수정량이 커서가 아니라, `DEFAULT_CONFIG.drive.dt_max`가 `0.1`로 너무 크게 풀려 있었기 때문입니다.
- **메커니즘**:
  - `dt_max = 0.1` 상태에서는 초기에 수렴이 원활하면 $dt$가 $0.04\text{ s}$까지 급증하여 **스텝당 4도에 달하는 거대한 회전각**이 한 번에 가해집니다. 이로 인해 30 µm 극박 PSA 요소에 과도한 순간 변형이 걸려 발산했던 것입니다.
  - 원래 AGENTS.md 및 `ex11`/`ex12` 기준에서 권장하는 시간 증분 상한은 **`dt_max = 0.01` (스텝당 최대 ~0.9° 회전)**입니다. 이 값을 지켜주면 인위적인 변위 캡 없이도 요소 반전 없이 평형에 도달할 수 있습니다.

### 실증 A/B 테스트 결과 (방금 수행)
| 조건 | Newton 반복수 (스텝 1~3) | 수렴 거동 | 상태 |
| :--- | :---: | :---: | :---: |
| **변위 클램핑 적용 (`max_disp_corr = 0.05`)** | **25 회 (상한 도달)** | 잔여력 저하 지체 ($8.3 \to 5.9 \to 5.4 \to 3.8 \dots$), Cutback 연쇄, $dt \to 0.00002$ 추락 | **극심한 수렴성 악화 (실패)** |
| **변위 클램핑 해제 (`max_disp_corr = None`)** | **단 2 회 (Quadratic)** | 스텝 1: 2 iters, 스텝 2: 2 iters, 스텝 3: 2 iters, 부드러운 평형 도달 | **초고속 정상 수렴 (성공)** |

---

## 2. Proposed Changes (개선 및 정상화 계획)

### [Component 1] `dispsolver/solver/dynamic.py`
- `max_displacement_corr` 강제 클램핑 로직을 제거 또는 기본 `None`으로 완전히 무력화.
- Newton의 순수 탐색 방향 $\Delta u = -K^{-1} R$이 어떠한 인위적 축소 없이 온전한 2차 수렴성을 발휘하도록 복원.
- 안정성이 입증된 `PardisoNonlinearSolver` 연동 및 PARDISO Phase 11 캐싱은 그대로 유지하여 고속 연산 보장.

### [Component 2] `dispsolver/fold_model_config.py`
- `DEFAULT_CONFIG.drive.dt_max` 및 `make_teardrop_config`의 `dt_max`를 **`0.01` ~ `0.02`**로 명확히 설정. (스텝당 0.9° 이하 안정적 회전 보장)
- `SolverTuningConfig.max_displacement_corr = None` 설정.
- 안정된 요소 정식: PET `Q4_COROTATIONAL_SRI` + PSA `Q4_UP` 유지.

### [Component 3] `examples/ex13_unified_model_io.py`
- `run_build`에 `max_steps` 매개변수 옵션을 추가하여, 사용자가 원하는 스텝 수만큼 프로파일링 및 테스트가 가능하도록 편의 기능 보완.

---

## 3. Verification Plan (검증 계획)

1. **소규모 스텝 수렴성 검증 (Smoke Test)**:
   - 3스텝 적분을 실행하여 스텝당 반복수가 **1~3회**로 즉각 수렴하는지 확인.
2. **다층 Teardrop 90° 폴딩 해석 완주**:
   - `dt_max = 0.01` 조건으로 $t=1.0$ (90° 폴딩) 완주 실행.
   - 요소 반전 없음(`n_inverted == 0`), PSA 층간 전단 분담율 >95% 확인.
   - `examples/ex13_build_final_folding_shape.png`로 완벽한 물방울 형상 확인.
3. **대화형 포스트 뷰어 연동**:
   - 완성된 `ex13_build_result.pkl` 파일을 화면의 Qt 뷰어로 로드하여 대화형 시각화 확인.
4. **기존 검증 스위트 회귀 테스트**:
   - `pytest tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py -q`
   - `python -m verification.run_all --quiet` (9/9 PASS 확인)

---

## 4. User Review Required

> [!IMPORTANT]
> 인위적 변위 클램핑(`max_displacement_corr = 0.05`)을 해제하고, 물리적으로 타당한 시간 증분(`dt_max = 0.01`)을 적용하여 스텝당 1~3회의 초고속 Newton-Raphson 수렴성을 복원하고자 합니다.
> 위 계획에 동의하시면 승인해 주시면 즉시 반영하여 완주 해석을 진행하겠습니다.
