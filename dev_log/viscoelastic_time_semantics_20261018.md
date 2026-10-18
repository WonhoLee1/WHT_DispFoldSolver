# 점탄성 시간 의미 조사 및 *VISCO 해석 절차 설계 보고서
(Viscoelastic Time Semantics & *VISCO Procedure Design Report)
**문서 작성일**: 2026-10-18  
**작업 패키지**: P3 (해석 절차: STABILIZE 포팅 + 점탄성 시간 의미 조사)  
**브랜치**: `feat/p3-procedures`  
**책임자**: Antigravity (P3 Agent)  

---

## 1. 개요 및 배경 (Executive Summary)

Abaqus/Standard와 상용 FEA 솔버에서는 준정적(Quasi-static) 해석 절차 간에 **물리적 시간($t$)과 증분 시간($\Delta t$)의 역학적 의미**가 엄격하게 구분됩니다:
1. **`*STATIC` (기본값, rate-independent)**:
   - 의사 시간(pseudo-time) $t \in [0, 1]$을 하중 증분 제어 파라미터로만 사용합니다.
   - 점탄성(Viscoelastic) 재료가 정의된 경우, 기본적으로 점성 완화가 완료된 **장기 평형 탄성 계수($g_\infty = 1 - \sum g_i$)**를 적용하여 정적 평형을 해석합니다.
2. **`*STATIC, TIME DEPENDENT` 또는 `*VISCO` (transient viscoelastic/creep)**:
   - $\Delta t$를 실제 물리적 시간 증분으로 취급하여 Prony 급수 컨볼루션 적분 및 오버스트레스 이완($\beta_i = e^{-\Delta t / \tau_i}$)을 계산합니다.
   - 크리프/점탄성 변형률 증분 오차 허용치인 `CETOL` 파라미터에 기반한 자동 시간 증분 조절이 활성화됩니다.
3. **`*DYNAMIC` (implicit dynamics)**:
   - 질량 관성력($M \mathbf{a}$)과 감쇠력, 점탄성 이완이 실제 물리 시간 $t$에 따라 동시에 적분됩니다.

본 연구에서는 `WHT_DispFoldSolver`의 단일 PSA(점착제) 4절점 2D 요소(`QUAD4` / `CPE4H`)를 대상으로 일정 변형률(10% 인장) 하에서 시간 증분 $\Delta t$ 변화에 따른 응력 이완 거동을 수치 계측하고, 연속시간 Prony 해석해와 비교·분석하여 향후 `*VISCO` 절차의 아키텍처 설계를 도출하였습니다.

---

## 2. 점탄성 정식화 및 오차 수식 유도 (Theoretical Formulation)

### 2.1 Simo (1987) 점탄성 재귀 점화식
`dispsolver/material/viscoelastic.py` 및 `q4_visco_*_numba.py`에 구현된 대변형 점탄성은 체적/편차 분리(Flory split) 및 Simo (1987)의 선형 중점 적분법(linearized midpoint rule)을 따릅니다:

$$S_{\text{eff}} = S_{\text{vol}} + S_{\text{eff,dev}}$$
$$S_{\text{eff,dev}}(t + \Delta t) = g_\infty S_{\text{dev,el}}(t + \Delta t) + \sum_{i=1}^M \mathbf{h}_i(t + \Delta t)$$

여기서 $M$개의 Prony 감쇠 성분 내부 변수 $\mathbf{h}_i$는 다음과 같이 전진합니다:
$$\mathbf{h}_i(t + \Delta t) = \beta_i \mathbf{h}_i(t) + g_i \gamma_i \Delta S_{\text{dev,el}}$$
$$\beta_i = \exp\left(-\frac{\Delta t}{\tau_i a_T}\right), \quad \gamma_i = \frac{1 - \beta_i}{\Delta t / (\tau_i a_T)}$$
$$g_\infty + \sum_{i=1}^M g_i = 1$$

### 2.2 일정 변형률(Stress Relaxation) 하에서의 해석해
시간 $t=0$에서 급격히 변형률 $\varepsilon_0$이 부하된 후 유지될 때($\Delta S_{\text{dev,el}} = 0$ for $t > \Delta t$):
- **연속시간 정해(Continuous Analytical Solution)**:
  $$S_{\text{dev}}^{\text{exact}}(t) = S_{\text{dev,0}} \left[ g_\infty + \sum_{i=1}^M g_i \exp\left(-\frac{t}{\tau_i}\right) \right]$$
- **이산 점화식 수치해(Discrete Numerical Solution)**:
  1단계($t \in [0, \Delta t]$)에서 $S_{\text{dev,el}}$이 $0$에서 $S_{\text{dev,0}}$으로 도달하므로:
  $$\mathbf{h}_i(\Delta t) = g_i \gamma_i S_{\text{dev,0}} = g_i \frac{1 - e^{-\Delta t / \tau_i}}{\Delta t / \tau_i} S_{\text{dev,0}}$$
  이후 $k$번째 증분($t = k \Delta t$)에서는:
  $$\mathbf{h}_i(t) = \mathbf{h}_i(\Delta t) \cdot e^{-(t - \Delta t)/\tau_i} = g_i S_{\text{dev,0}} \left( \frac{e^{\Delta t / \tau_i} - 1}{\Delta t / \tau_i} \right) e^{-t / \tau_i}$$

테일러 전개 시:
$$\frac{e^{\Delta t / \tau_i} - 1}{\Delta t / \tau_i} = 1 + \frac{1}{2} \frac{\Delta t}{\tau_i} + O(\Delta t^2)$$
따라서 이산 수치해는 연속시간 정해 대비 $O(\Delta t)$의 1차 수렴 오차를 가집니다.

---

## 3. 수치 실험 결과 (Empirical Measurement Results)

### 3.1 실험 조건
- **메쉬**: $1.0\,\text{mm} \times 1.0\,\text{mm}$ 단일 4절점 평면변형률 요소 (`QUAD4` / `CPE4H`)
- **기저 재료**: NeoHookean ($E = 1000.0\,\text{MPa}$, $\nu = 0.30$)
- **Prony 급수 물성치 (PSA 전형값)**:
  - 항 1 (고속 이완): $g_1 = 0.40$, $\tau_1 = 0.05\,\text{s}$
  - 항 2 (중속 이완): $g_2 = 0.30$, $\tau_2 = 0.50\,\text{s}$
  - 장기 평형: $g_\infty = 0.30$ ($g_1 + g_2 + g_\infty = 1.0$)
- **하중 조건**: 상단 변위 $u_y = 0.10\,\text{mm}$ (공칭 변형률 10%) 부하 후 $t = 1.0\,\text{s}$까지 유지
- **초기 순간 탄성 편차 응력**: $S_{\text{dev,0}} = 384.615\,\text{MPa}$
- **장기 평형 편차 응력 ($t \to \infty$)**: $S_{\infty} = g_\infty S_{\text{dev,0}} = 115.385\,\text{MPa}$

---

### 3.2 시간 증분 $\Delta t$ 미세화에 따른 수렴도 분석 (STATIC 모드)

해석해:
- $S_{\text{dev}}^{\text{exact}}(t = 0.1\,\text{s}) = 230.674\,\text{MPa}$
- $S_{\text{dev}}^{\text{exact}}(t = 1.0\,\text{s}) = 131.000\,\text{MPa}$

| $\Delta t$ (s) | 증분 수 (Steps) | $S_{\text{dev}}(t=0.1\,\text{s})$ (MPa) | $S_{\text{dev}}(t=1.0\,\text{s})$ (MPa) | 최대 상대오차 (%) | RMS 상대오차 (%) | 최종 상대오차 (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.001** | 1000 | 230.978 | 131.016 | 0.428% | 0.093% | **0.012%** |
| **0.002** | 500 | 231.286 | 131.031 | 0.853% | 0.186% | **0.024%** |
| **0.005** | 200 | 232.225 | 131.079 | 2.107% | 0.464% | **0.060%** |
| **0.010** | 100 | 233.853 | 131.157 | 4.124% | 0.927% | **0.120%** |
| **0.020** | 50 | 237.369 | 131.317 | 7.870% | 1.851% | **0.242%** |
| **0.050** | 20 | 250.514 | 131.808 | 16.662% | 4.540% | **0.616%** |
| **0.100** | 10 | 286.476 | 132.671 | 24.191% | 8.340% | **1.276%** |

> [!NOTE]
> **1차 수렴성($O(\Delta t)$) 완벽 입증**:
> $\Delta t = 0.01\,\text{s}$에서 최종 오차 0.120%이던 것이 $\Delta t = 0.001\,\text{s}$에서 0.012%로 정확히 10배 정밀해짐을 확인하였습니다.
> 또한 $t = 1.0\,\text{s}$에서는 고속 감쇠 모드($\tau_1 = 0.05\,\text{s}$, $e^{-20} \approx 2 \times 10^{-9}$)가 완전 소멸되어 오차가 $0.01 \sim 1.2\%$ 미만으로 극도로 안정화됩니다.

---

### 3.3 해석 절차 모드별 비교: STATIC vs DYNAMIC ($\Delta t = 0.05\,\text{s}$)

| 절차 모드 | 시간 $t$ (s) | 수치 응력 $S_{\text{eff,dev}}$ (MPa) | 연속 해석 정해 (MPa) | 오차 vs 정해 (%) | 뉴턴 반복수 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **STATIC** | 0.1 | 250.514484 | 230.674361 | 8.601% | 1.0 |
| **STATIC** | 0.5 | 160.039177 | 157.839228 | 1.394% | 1.0 |
| **STATIC** | 1.0 | 131.807696 | 131.000225 | 0.616% | 1.0 |
| **DYNAMIC** | 0.1 | 250.514484 | 230.674361 | 8.601% | 1.0 |
| **DYNAMIC** | 0.5 | 160.039177 | 157.839228 | 1.394% | 1.0 |
| **DYNAMIC** | 1.0 | 131.807696 | 131.000225 | 0.616% | 1.0 |

> [!IMPORTANT]
> **핵심 발견 (Core Empirical Finding)**:
> 1. `DynamicSolver2D` / `DynamicSolver3D` 및 `DynamicSolver`는 **`STATIC` 모드에서도 $\Delta t$를 요소 조립 루프(`assemble_system(u, dt=dt)`)에 그대로 전달**합니다.
> 2. 그 결과, 현재 코드베이스의 `*STATIC` 절차는 준정적 탄성 평형이 아니라 **물리 시간 $t$를 소비하며 점탄성 응력 이완을 동반하는 `*VISCO` 모드로 동작**하고 있습니다.
> 3. STATIC과 DYNAMIC 간의 유효 응력은 소수점 12자리까지 비트 단위로 완벽히 일치합니다 (`250.51448434... MPa`).

---

## 4. Abaqus 표준 규격 대조 및 의미론적 격차 (Semantic Gap Analysis)

| 기능 / 동작 | Abaqus/Standard 표준 규격 | 현재 WHT_DispFoldSolver 현황 | 격차 및 영향 |
|:---|:---|:---|:---|
| **`*STATIC` (기본값)** | 물리 시간 무관 (rate-independent). 점탄성 재료는 장기 평형 탄성률 $g_\infty$ 적용 | $\Delta t$를 물리 시간으로 취급하여 Prony 응력 이완 진행 | 정적 탄성 해석을 기대하는 범용 Abaqus .inp 덱에서 과도적 이완 발생 |
| **`*STATIC, TIME DEPENDENT`** | $\Delta t$를 실제 물리 시간으로 취급하여 점탄성/크리프 과도 응답 적분 | 지원되지 않는 파라미터로 무시되거나 기본 STATIC으로 동작 | 명시적 옵션 플래그 누락 |
| **`*VISCO` 절차** | 준정적 점탄성/크리프 전용 해석 절차. `CETOL` 기반 자동 시간 증분 제어 | `*VISCO` 키워드 미구현 (`STATIC`으로 대체 구동) | Abaqus .inp 리더에서 `*VISCO` 파싱 불가 또는 GAP 발생 |
| **시간 증분 제어 (`CETOL`)** | 점탄성 변형률 변화율에 따라 $\Delta t$ 자동 컷백/증가 | 고정 $\Delta t$ 또는 잔차 수렴성 기반 적응 증분만 작동 | 시간 적분 오차를 직접 제어하지 못함 |

---

## 5. `*VISCO` 절차 도입을 위한 아키텍처 설계안 (Proposed Design for *VISCO)

Abaqus Feature Parity를 완성하고 기존 디스플레이 폴딩 해석과의 하위 호환성을 보장하기 위해 다음과 같은 단계적 설계를 제안합니다:

### 5.1 Step 및 Reader 확장
1. **Model API (`dispsolver/model/step.py`)**:
   ```python
   class ViscoStep(Step):
       """Abaqus-compatible *VISCO step for transient viscoelasticity & creep."""
       def __init__(self, name: str, cetol: float = 1e-3, **kwargs):
           super().__init__(name=name, procedure="VISCO", **kwargs)
           self.cetol = float(cetol)
   ```
2. **Keyword Reader (`dispsolver/io/keywords/steps.py`)**:
   - `*VISCO` 키워드 핸들러 추가:
     ```python
     @register_keyword("VISCO", contexts=[KeywordContext.STEP])
     def _visco(rows, ctx, p):
         step = _current_step(ctx)
         step.procedure = "VISCO"
         step.cetol = p.cetol
     ```
   - `*STATIC`에 `TIME DEPENDENT` 파라미터 인식 추가.

### 5.2 솔버 점탄성 시간 의미 플래그 (`visco_time_active`)
`DynamicSolver2D` 및 `DynamicSolver3D`에 스텝별 시간 전진 정책을 명시화:
```python
if step.procedure == "VISCO" or (step.procedure == "STATIC" and getattr(step, "time_dependent", False)):
    dt_elem = dt
elif step.procedure == "STATIC":
    # Abaqus standard: rate-independent static solve
    # Option A (Strict Parity): dt_elem = None (forces material to long-term g_inf)
    # Option B (DispFold Legacy Compatibility): env flag or config-controlled fallback
    dt_elem = dt if self.allow_static_visco_relaxation else None
```

### 5.3 `CETOL` 기반 자동 시간 증분 어댑터 (Adaptive Time Stepper)
Simo 점탄성 정식화에서 한 스텝 동안 발생한 최대 점탄성 과도 변형률 증분 $\Delta \varepsilon_{\text{visco}}$를 추정하여 다음 스텝 $\Delta t_{\text{next}}$를 자동 조절:
$$\Delta t_{\text{next}} = \Delta t_{\text{curr}} \cdot \left( \frac{\text{CETOL}}{\max_e \|\Delta \varepsilon_{\text{visco}}\|} \right)^{1/2}$$
허용치 초과 시 즉시 증분을 컷백(cutback)하여 시간 적분 정밀도를 보장.

---

## 6. 결론 (Conclusion)

1. **검증 완료**: Simo (1987) 점탄성 재귀 점화식은 $\Delta t \to 0$에 따라 연속시간 정해로 엄밀한 $O(\Delta t)$ 1차 수렴함을 7개 시간 증분 구간에서 정량적으로 반증 완료하였습니다.
2. **현황 진단**: 현재 솔버의 `STATIC` 모드는 내부적으로 이미 `*VISCO`와 동일한 물리적 시간 적분을 수행하고 있으므로, `*VISCO` 키워드 및 `ViscoStep` API를 추가하는 작업은 물리 커널 수정 없이 **인터페이스 바인딩과 CETOL 적응 증분기 추가만으로 즉시 완성 가능**합니다.
3. 본 설계안은 향후 P3 이후의 절차 고도화 패키지에서 우선순위에 따라 즉시 구현될 수 있습니다.
