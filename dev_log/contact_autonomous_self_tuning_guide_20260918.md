# Autonomous In-Situ Contact Auto-Tuner & Parameter Recommendation Architecture
Date: 2026-09-18
Author: Antigravity Core FEA Architect (lucy-orcha pipeline)

## 1. Motivation & Practical Background
유한요소 비선형 접촉 해석에서 실무 엔지니어가 가장 많은 시간을 낭비하는 원인은 다음과 같다:
1. 해석이 진행되다가 특정 스텝에서 접촉 비선형성(채터링, 접촉면 충돌 충격, 조건수 악화)으로 수렴하지 못하고 컷백이 나거나 해석이 중단됨.
2. 엔지니어는 해석을 처음부터 다시 돌리며 수동으로 접촉 강성을 줄이거나, 안정화 계수를 높이거나, 침투 허용량을 늘리는 시행착오를 수십 번 반복함.

**우리의 혁신적 해결책: Autonomous In-Situ Contact Auto-Tuner**:
- 해석이 멈추거나 사용자가 처음부터 다시 정의할 필요 없이,
- **솔버가 런타임(In-situ)에 능동적으로 불안정성을 감지**하여 접촉 완충 거리(`lead`), 강성 완화율(`softnessScale`), 국소 안정화(`damping`)를 적응적으로 완화·조정.
- 스텝을 즉시 복구(Self-healing)하여 성공적으로 수렴시키고, 그 안정된 파라미터를 다음 스텝들에도 확정 유지.
- **해석 로그 및 반환 객체에 결정된 최적 파라미터 가이드를 출력**하여, 다음 번 해석 스크립트 작성 시 그대로 복사해서 처음부터 최적 설정으로 구동할 수 있도록 안내.

---

## 2. Dynamic Self-Tuning Protocol

```
[Newton-Raphson Iteration in solve_step]
                │
         (Non-convergence or Chattering > max_iters/2)
                │
                ▼
      [ContactAutoTuner Triggered]
                │
  1. 진단 (Diagnostic Analysis):
     - Severe Discontinuity Iterations (SDI) 빈발? -> 접촉 개폐 진동 (Chattering)
     - Residual Divergence / Line search halving? -> 접촉 초기 충격 (Impulse Shock)
                │
  2. 능동 파라미터 완화 (Autonomous Relaxation):
     - lead (c0) *= 1.5 ~ 2.0 (접촉 전 부드러운 전이 구간 확장)
     - softnessScale *= 0.5 ~ 0.6 (초기 접선 기울기 완화)
     - stabilization_coeff 추가/보강 (점성 댐핑 완충)
                │
  3. 재시도 및 확정 (In-Situ Recovery & Lock-in):
     - 현재 스텝 재계산 -> 수렴 성공!
     - 갱신된 파라미터를 GeneralContact3D.law 에 즉시 확정 적용
                │
  4. 가이드 리포트 출력 (Parameter Recommendation):
     - 콘솔/로그에 결정된 파라미터 및 스크립트 복사용 코드 제공
```

---

## 3. Python API & Architecture

```python
# 1. 자동 튜너가 활성화된 상태로 solve_step 구동 (기본 auto_tune_contact=True)
converged, iters = solver.solve_step(dt=0.01, auto_tune_contact=True)

# 2. 해석 후 결정된 접촉 파라미터 가이드 조회
guide = solver.get_contact_tuning_guide()
print(guide["code_snippet"])
# 출력 예시:
# prop.AdaptiveSoftBehavior(lead=0.0035, softnessScale=0.036, power=2.0)
```
