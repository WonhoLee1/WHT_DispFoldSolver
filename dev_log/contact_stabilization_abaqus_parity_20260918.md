# Abaqus-Parity Contact Stabilization & Property Sub-Methods System Design
Date: 2026-09-18
Author: Antigravity Core FEA Architect (lucy-orcha pipeline)

## 1. Executive Summary & Problem Context
사용자는 Abaqus Python Scripting Interface를 오랜 기간 실무에서 사용해 온 전문가이며, WHT_DispFoldSolver에서도 Abaqus Python API의 문법과 옵션을 100% 동일하게(Abaqus Parity) 사용하기를 요구하고 있다.

구체적으로:
1. **NormalBehavior & TangentialBehavior 서브 메서드 체이닝**:
   - 기존의 평탄한 생성자 인수 전달 방식에서 벗어나, `ContactProperty` 객체의 서브메서드인 `prop.NormalBehavior(...)` 및 `prop.TangentialBehavior(...)`를 통해 법선/접선 접촉 거동을 정의.
   - `pressureOverclosure=TABULAR, table=((0.0, -0.001), (1.0, 0.0), (1000.0, 0.1))`와 같이, **(Pressure, Overclosure)** 순서의 Abaqus 튜플 및 접촉 전 음수 오버클로저(Soft Tangent Pre-contact) 완전 지원.
2. **Contact Stabilization & Damping 완전 패리티**:
   - 접촉 전 물체 간의 미소 갭에서 발생하는 수치적 특이성(강체 운동 모드)을 억제하기 위해 Abaqus가 제공하는 **Contact Damping** 및 **Contact Stabilization** 옵션을 동일한 수준으로 제공.
   - `prop.Damping(definition=DAMPING_COEFFICIENT, dampingCoefficient=..., clearance=..., tangentFraction=...)`
   - `model.ContactControl(name=..., stabilize=AUTOMATIC, factor=..., rampDownFactor=...)`
   - `model.ContactStabilization(name=..., definition=AUTOMATIC, dampingFactor=..., tangentFraction=...)`
   - General Contact 및 Contact Pair에서 `contactControls` / `contactStabilization` 인자를 통한 자동 연동.

---

## 2. Abaqus Theoretical Basis & Governing Equations

### 2.1 Viscous Contact Damping (*CONTACT DAMPING)
Abaqus/Standard에서 접촉 감쇠력은 접촉면 간의 상대 속도(Relative Velocity)에 비례하여 작용합니다:
$$
\mathbf{f}_{vd} = \mu_{0} \cdot A_{\text{node}} \cdot \mathbf{v}_{\text{rel}}
$$
여기서:
- $\mu_0$ [$\text{N}\cdot\text{s}/\text{mm}^3$]: 단위 면적당 감쇠 계수 (`DAMPING COEFFICIENT`).
- $A_{\text{node}}$ [$\text{mm}^2$]: 절점 유효 접촉 면적.
- $\mathbf{v}_{\text{rel}} = (\mathbf{u}_{n+1} - \mathbf{u}_n) / \Delta t$: 상대 변위 속도.

법선 방향 및 접선 방향으로 분해:
$$
\mathbf{f}_{vd} = \mu_{0} A \left[ (\mathbf{v}_{\text{rel}} \cdot \mathbf{n}) \mathbf{n} + \eta_{\text{tangent}} (\mathbf{v}_{\text{rel}} - (\mathbf{v}_{\text{rel}} \cdot \mathbf{n}) \mathbf{n}) \right]
$$
- $\eta_{\text{tangent}}$: `tangentFraction` (Abaqus 기본값 = 0.0 for Standard, 즉 순수 법선 감쇠).

Clearance $c$에 따른 감쇠 계수 램핑:
$$
\mu(c) = \begin{cases}
\mu_0, & c \le 0 \quad (\text{접촉 중}) \\
\mu_0 \left(1 - \frac{c}{c_0}\right), & 0 < c < c_0 \quad (\text{접촉 전 완충 구간}) \\
0, & c \ge c_0 \quad (\text{비접촉})
\end{cases}
$$
이 수식은 접촉 직전 $c_0$ 거리 이내로 접근할 때 점진적으로 감쇠력을 발생시켜, 뉴턴 반복 중 발생하는 채터링(Chattering)과 충돌 발산을 완벽히 방지합니다.

### 2.2 Automatic Contact Stabilization (*CONTACT CONTROLS, STABILIZE)
강체 모드(Rigid Body Motion) 방지를 위한 자동 점성 안정화:
- 요소 대표 강성 $K_{\text{elem}}$과 스텝 시간 $\Delta t$를 기반으로 기본 감쇠 계수 자동 계산:
  $$
  c_{\text{auto}} = \alpha_{\text{factor}} \cdot \frac{K_{\text{elem}} \cdot \Delta t}{A_{\text{contact}}} \times 10^{-3}
  $$
- 스텝 진행에 따른 램프다운:
  $$
  S(t) = 1.0 - (1.0 - f_{\text{ramp}}) \cdot \frac{t - t_0}{\Delta t_{\text{step}}}
  $$
  기본적으로 스텝 종료 시 $f_{\text{ramp}} = 0.0$으로 완전히 소거되어 최종 수렴 상태의 물리적 정밀도를 전혀 왜곡하지 않음.

### 2.3 Consistent Tangent Matrix Contribution
준정적(Quasistatic) 해석에서 $\mathbf{v}_{\text{rel}} \approx \Delta \mathbf{u} / \Delta t$ 이므로, 뉴턴-랩슨의 2차 수렴성을 보장하기 위한 일관 접선 강성 행렬 기여는 다음과 같습니다:
$$
\mathbf{K}_{\text{stab}} = \frac{\partial \mathbf{f}_{vd}}{\partial \mathbf{u}} = \frac{\mu(c) A}{\Delta t} \left[ \mathbf{n} \otimes \mathbf{n}^T + \eta_{\text{tangent}} (\mathbf{I} - \mathbf{n} \otimes \mathbf{n}^T) \right]
$$
이 강성 행렬은 완전 대칭(SPD)이며, 접촉 개시 전의 시스템 행렬 특이성(Singular Matrix, $K \mathbf{u} = \mathbf{0}$)을 엄밀하게 제거합니다.

---

## 3. Abaqus Python API Architecture & Signatures

```python
# 1. Contact Property with Sub-methods
prop = model.ContactProperty('IntProp-1')
prop.NormalBehavior(
    pressureOverclosure=TABULAR,     # HARD | TABULAR | EXPONENTIAL | LINEAR
    table=((0.0, -0.001), (1.0, 0.0), (1000.0, 0.1)),
    constraintEnforcementMethod=DEFAULT,
    stiffnessScaleFactor=1.0
)
prop.TangentialBehavior(
    formulation=PENALTY,             # FRICTIONLESS | PENALTY
    table=((0.2, ), )
)
prop.Damping(
    definition=DAMPING_COEFFICIENT,  # DAMPING_COEFFICIENT | CRITICAL_DAMPING_FRACTION
    dampingCoefficient=1.0e-3,
    clearance=0.1,
    tangentFraction=0.0
)

# 2. Contact Controls & Stabilization Objects
ctrl = model.ContactControl(
    name='ContCtrl-1',
    stabilize=AUTOMATIC,             # AUTOMATIC | None
    factor=1.0,
    tangentFraction=0.0,
    rampDownFactor=0.0
)
# Alias for Abaqus ContactStabilization
stab = model.ContactStabilization(
    name='Stab-1',
    definition=AUTOMATIC,
    dampingFactor=1.0,
    tangentFraction=0.0,
    clearance=0.1
)

# 3. Interactions with Stabilization
model.ContactStd(
    name='GeneralContact-1',
    createStepName='Step-1',
    useAllExterior=True,
    interactionProperty='IntProp-1',
    contactControls='ContCtrl-1',     # or contactStabilization='Stab-1'
)
```

---

## 4. Implementation Steps & Verification Plan

1. **`dispsolver/model/interaction.py`**:
   - `ContactProperty`에 `NormalBehavior`, `TangentialBehavior`, `Damping` 메서드 추가
   - `ContactControl`, `ContactStabilization` 데이터클래스 정의
   - Abaqus 심볼릭 상수 export (`HARD`, `TABULAR`, `EXPONENTIAL`, `LINEAR`, `PENALTY`, `FRICTIONLESS`, `AUTOMATIC`, `DAMPING_COEFFICIENT`, `CRITICAL_DAMPING_FRACTION`, `DEFAULT`)
   - `ContactPair` 및 `GeneralContact`의 `build_runtime_constraint`에서 law 및 stabilization 계수 전달 로직 고도화.
2. **`dispsolver/model/model.py`**:
   - `model.ContactControl(...)`, `model.ContactStabilization(...)` 생성 메서드 및 컨테이너 관리 (`model.contact_controls`, `model.contact_stabilizations`).
   - `model.interactionProperties` = `model.contact_properties` alias 제공.
3. **`dispsolver/contact/general_contact3d.py`**:
   - `law` 및 `damping_coeff`, `damping_clearance`, `tangent_fraction` 파라미터 수용.
   - `assemble()` 루프에서 $0 < \text{gap} < c_0$ 구간의 점성 감쇠 및 접선 행렬 가산.
4. **독립 검증 (`tests/test_abaqus_contact_api.py`)**:
   - 전 문법 검증 및 수치적 감쇠력/강성 일관성 테스트.
