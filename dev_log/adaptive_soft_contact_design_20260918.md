# Adaptive Soft Contact & Power-Law Stiffness Auto-Tuning System Design
Date: 2026-09-18
Author: Antigravity Core FEA Architect (lucy-orcha pipeline)

## 1. Problem Statement & User Need
Abaqus 및 기존 접촉 해석에서 `pressureOverclosure=TABULAR` 또는 `EXPONENTIAL`을 사용할 때 가장 큰 실무적 고충은 **"적절한 접촉 강성과 압력 수치를 사용자가 직접 계산해서 하드코딩해야 한다"**는 점이다:
- 재료 물성치(스틸 200 GPa vs 고무 10 MPa vs 박막 PSA 0.1 MPa)와 메쉬 밀도에 따라 적정 접촉 강성이 수만 배 이상 차이남.
- 너무 강한 수치를 넣으면 뉴턴 반복이 튕겨나가거나 컷백(Cutback)이 폭증하고, 너무 약하면 과도한 침투가 발생함.
- 사용자는 복잡한 수치 대신 다음과 같은 직관적인 물리적 제어를 원함:
  1. **접촉 전으로 조금 당기는 수준 (Pre-contact Lead / Clearance)**: 접촉 발생 직전 미세 갭에서 미리 완충 강성을 시작.
  2. **강성 기울기를 낮춰주는 수준 (Softness Scale)**: 접촉 초기 단계에서 기본 강성을 일정 비율(예: 1%~10%)로 완화하여 충격 완화.
  3. **지수/차수 수준 (Power / Exponent)**: 1.5차, 2차(Quadratic), 3차(Cubic) 등 부드러운 전이를 결정하는 곡률 차수.
  4. **적응형 자동 스케일링 (Adaptive Auto-Tuning)**: 솔버가 메쉬 특성 길이 $L_{\text{char}}$와 재료 강성 $k_{\text{ref}}$를 바탕으로 안정된 최적의 강성 곡선을 자동 유도.

---

## 2. Mathematical Formulation of Adaptive Power-Law Soft Contact

### 2.1 Variables & Parameters
- $h$: Overclosure (침투량, $h = -g_n = -\text{gap}$)
- $c_0$: Pre-contact Lead (접촉 전 당김 거리, $c_0 > 0 \implies h = -c_0$에서 완충 시작)
- $\delta$: Transition Depth (정상 페널티 강성에 도달하는 목표 침투 깊이)
- $K_{\text{target}}$: 정상 상태 목표 접촉 강성 ($= \text{softness\_scale} \cdot K_{\text{default}}$)
- $m$: Power-law Exponent (기본값 $m = 2.0$, $C^1$ 연속)

### 2.2 Normalized Coordinate & Governing Curves
정규화 침투 좌표 $\xi$:
$$
\xi(h) = \frac{h + c_0}{\delta + c_0}
$$

1. **비접촉 구간 ($\xi \le 0 \iff h \le -c_0$)**:
   $$p(h) = 0, \quad \frac{dp}{dh} = 0$$

2. **적응형 완충 전이 구간 ($0 < \xi \le 1 \iff -c_0 < h \le \delta$)**:
   거듭제곱 완충 압력:
   $$p(h) = p_{\text{ref}} \cdot \xi^m$$
   접선 강성:
   $$K(h) = \frac{dp}{dh} = \frac{m \cdot p_{\text{ref}}}{\delta + c_0} \cdot \xi^{m-1}$$
   경계 조건: $\xi = 1$ ($h = \delta$)에서 접선 강성이 목표 강성 $K_{\text{target}}$과 완벽히 일치해야 하므로:
   $$K(\delta) = \frac{m \cdot p_{\text{ref}}}{\delta + c_0} = K_{\text{target}} \implies p_{\text{ref}} = \frac{K_{\text{target}} \cdot (\delta + c_0)}{m}$$
   따라서 최종 압력 공식:
   $$p(h) = \frac{K_{\text{target}} \cdot (\delta + c_0)}{m} \left( \frac{h + c_0}{\delta + c_0} \right)^m$$
   접선 강성 공식:
   $$\frac{dp}{dh} = K_{\text{target}} \left( \frac{h + c_0}{\delta + c_0} \right)^{m-1}$$

3. **정상 접촉 저항 구간 ($\xi > 1 \iff h > \delta$)**:
   $\xi = 1$ 지점에서 값과 기울기가 $C^1$ 연속인 선형 확장:
   $$p(h) = p_{\text{ref}} + K_{\text{target}} \cdot (h - \delta)$$
   $$\frac{dp}{dh} = K_{\text{target}}$$

### 2.3 Mathematical Advantages
- **$C^1$ 매끄러운 연속성 (Smooth Transitions)**:
  $m \ge 2.0$일 때 접촉 개시점 $h = -c_0$에서 $p = 0$ 및 $K = 0$이므로, 접촉 활성화 시 강성 행렬의 불연속 충격이 0이며 뉴턴 반복 수렴성이 비약적으로 향상됨.
- **수치적 무결성 (Anti-Mocking)**:
  임의의 난수나 가짜 스케일링이 아닌, 변분 에너지의 2계 도함수(Tangent Consistency)를 기계 정밀도로 만족.

---

## 3. Python API Architecture

```python
# 1. Abaqus-parity NormalBehavior 내 키워드 확장
prop = model.ContactProperty('AdaptiveProp')
prop.NormalBehavior(
    pressureOverclosure=ADAPTIVE_SOFT, # or ADAPTIVE, POWER_LAW
    lead=0.001,                        # 접촉 전 1 um부터 완충 시작
    softnessScale=0.2,                 # 초기 강성 기울기 배율 (기본 페널티의 20%)
    power=2.0,                         # 지수/차수 (2.0: 2차 포물선 완충)
    allowablePenetration=0.02,         # 20 um 침투 시 정상 강성 도달
)

# 2. 직관적인 전용 편의 메서드
prop.AdaptiveSoftBehavior(
    lead=0.001,          # 접촉 전 당김 거리
    leadRatio=0.01,      # 메쉬 특성 길이 비율 기준 자동 당김
    softnessScale=0.1,   # 강성 기울기 완화 배율
    power=2.0,           # 차수
)
```
