# 📋 Abaqus Verification Reference 26개 문서 실시간 분석 및 상태 매트릭스

**작성 일시**: 2026-09-19 19:55  
**오케스트레이션 상태**: lucy-orcha Non-Gemini 4인 패널 (DeepSeek-V4-Pro, Qwen3.8-Max, GLM-5.3, Claude-Opus) 진행 중  

---

## 1. 26개 Abaqus Verification Reference 문서 전수 분류 매트릭스

사용자 지시 조건:
- **열전달(Heat Transfer) 제외**: 열전도도, 비열, 접촉 열전도 등은 1차 구현 및 벤치마크 대상에서 제외.
- **온도 의존 필드 및 열팽창 필수 포함**: 온도에 따른 물성 변화 $E(T), \sigma_y(T)$ 및 열변형률 $\boldsymbol{\varepsilon}^{th} = \alpha(T - T_0)\mathbf{I}$ 연계 반영.
- **연계 .inp 벤치마크 준비**: 각 문서에 명시된 Abaqus 표준 입력 파일명 매핑.

| No | 문서명 (HTML) | 물리 현상 및 핵심 이론 | 사용자 조건 | 우리 솔버(`WHT_DispFoldSolver`) 반영 사양 | 연계 Abaqus 검증 .inp 파일 |
|:---:|:---|:---|:---:|:---|:---|
| **01** | `simaver-c-elastic.htm` | 선형 등방/직교이방성/완전이방성 탄성 | **포함** | Engineering Constants ($E_1, E_2, E_3, \nu_{12}, G_{12}$), 국소 좌표계 | `elastic_ortho.inp`, `elastic_aniso.inp` |
| **02** | `simaver-c-viscoelastic.htm` | 시간 영역 Prony 점탄성 (완화/크립) | **완료됨** | 6-Prony 항 ($M=6$, 42-Float SDV) 이미 완비 | `visco_relax_prony.inp`, `visco_creep.inp` |
| **03** | `simaver-c-mullinseffect.htm` | Mullins 효과 (충전 고무 손상 연화) | **신규 도입** | Ogden-Roxburgh 에너지 소산 손상 함수 $1 - \frac{1}{r}\text{erf}\left(\frac{U_m - U}{m+\beta U_m}\right)$ | `mullins_damage_cyclic.inp` |
| **04** | `simaver-c-hysteretic.htm` | 초탄성 비탄성 이력 현상 | **신규 도입** | Bergstrom-Boyce 네트워크 이완 이력 모델 | `hyper_hysteresis.inp` |
| **05** | `simaver-c-tempelastic.htm` | **온도 의존 탄성 거동** | **★ 필수 포함** | 테이블형 $E(T), \nu(T)$ 선형 보간 및 일관 접선 행렬 계산 | `temp_dep_elastic_3d.inp` |
| **06** | `simaver-c-fieldelastic.htm` | 사용자 필드 변수 의존 탄성 | **포함** | Field Variables $f_1, f_2$ 의존 탄성 텐서 | `field_dep_elastic.inp` |
| **07** | `simaver-c-viscohyper.htm` | 초탄성 기반 대변형 점탄성 | **완료됨** | 초탄성 전제품군(Mooney, Ogden 등)과 6-Prony 결합 완료 | `visco_hyper_large_def.inp` |
| **08** | `simaver-c-nonlinviscohyper.htm`| 비선형 점탄성 (변형률 의존 완화) | **후순위 검토**| K-BKZ 또는 비선형 시간 스케일 완화 | `nonlin_viscohyper.inp` |
| **09** | `simaver-c-rateindepplast.htm` | 속도 무관 J2 Mises 소성 및 이동경화 | **완료됨** | J2 Plasticity 및 Radioss LAW36 / 테이블 소성 완비 | `mises_plastic_kinematic.inp` |
| **10** | `simaver-c-viscocylinder.htm` | 내압 받는 두꺼운 점탄성 실린더 | **검증용** | 원통 좌표계 해석해 대조 벤치마크 | `visco_thick_cylinder.inp` |
| **11** | `simaver-c-ratedepplastexp.htm`| 동적 속도 의존 소성 (Cowper-Symonds) | **신규 도입** | $\sigma_y = \sigma_0 \left[1 + \left(\frac{\dot{\bar{\varepsilon}}^{pl}}{D}\right)^{1/p}\right]$ | `rate_dep_cowper_symonds.inp` |
| **12** | `simaver-c-annealtemp.htm` | 열처리 어닐링 온도 제어 | **포함** | $T \ge T_{\text{anneal}}$ 시 누적 소성 변형률 $\bar{\varepsilon}^{pl} \to 0$ 리셋 | `anneal_temperature_reset.inp` |
| **13** | `simaver-c-tempplastic.htm` | **온도 의존 소성 거동** | **★ 필수 포함** | $\sigma_y(\bar{\varepsilon}^{pl}, T)$ 2차원 그리드 보간 및 열-소성 연계 | `temp_dep_plastic_mises.inp` |
| **14** | `simaver-c-fieldplastic.htm` | 필드 변수 의존 소성 거동 | **포함** | $\sigma_y(\bar{\varepsilon}^{pl}, f_1)$ 보간 | `field_dep_plastic.inp` |
| **15** | `simaver-c-johnsoncook.htm` | Johnson-Cook 소성 및 열-기계 파손 | **신규 도입** | $(A + B\varepsilon^n)(1 + C\ln\dot{\varepsilon}^*)(1 - T^{*m})$ | `johnson_cook_dynamic_damage.inp` |
| **16** | `simaver-c-damage.htm` | 연성/전단 손상 개시 및 진전 | **신규 도입** | Ductile & Shear Damage initiation, Hillerborg 파괴에너지 진전 | `ductile_damage_evolution.inp` |
| **17** | `simaver-c-creep.htm` | 고온 크립 변형 (Power-law/Norton) | **신규 도입** | $\dot{\bar{\varepsilon}}^{cr} = A \tilde{q}^n t^m$ 또는 Norton 법칙 | `creep_power_law.inp` |
| **18** | `simaver-c-viscous.htm` | 재료 점성 감쇠 및 안정화 점성 | **포함** | Viscous damping for Newton stabilization | `material_viscous_damping.inp` |
| **19** | `simaver-c-concrete.htm` | 콘크리트 소성 모델 | **제외/후순위** | Flexible Display 해석과 무관 | `concrete_plasticity.inp` |
| **20** | `simaver-c-concretedamaged.htm`| 콘크리트 손상 소성 | **제외/후순위** | Flexible Display 해석과 무관 | `concrete_damaged_plasticity.inp` |
| **21** | `simaver-c-damping.htm` | Rayleigh 감쇠 및 복소 구조 감쇠 | **완료됨** | $\mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K}$ 완비 | `rayleigh_damping_dynamic.inp` |
| **22** | `simaver-c-simpleexpansion.htm`| **열팽창 계수 $\alpha \Delta T$ 거동** | **★ 필수 포함** | $\boldsymbol{\varepsilon}^{th} = \alpha(T - T_0)\mathbf{I}$, 열변형률에 의한 잔차 하중 $\mathbf{f}^{th}$ | `thermal_expansion_simple.inp` |
| **23** | `simaver-c-plasticity-corrections.htm`| 소성 수치 보정 및 Return Mapping | **완료됨** | Radioss LAW36 $C^1$ 스무딩 및 $H_{\text{tail}}$ 완비 | `plasticity_algo_correction.inp` |
| **24** | `simaver-c-fieldvardependentconductivity.htm`| 필드 변수 의존 열전도도 | **🚫 제외** | 열전달 해석 제외 조건 | `field_conductivity.inp` |
| **25** | `simaver-c-conductivityspecificheat.htm`| 비열 및 열전도도 | **🚫 제외** | 열전달 해석 제외 조건 | `thermal_conductivity_heat.inp` |
| **26** | `simaver-c-gapconductance.htm` | 접촉면 열전도 (Gap Conductance) | **🚫 제외** | 접촉 열전달 제외 조건 | `gap_thermal_conductance.inp` |

---

## 2. 우리 솔버(`WHT_DispFoldSolver`)에 신규 개발할 핵심 기능 3대 패키지

### [패키지 1: 열-역학 연계 (Thermal-Mechanical Coupling)] ★ 최우선
- **대상 문서**: `simaver-c-tempelastic.htm`, `simaver-c-tempplastic.htm`, `simaver-c-simpleexpansion.htm`, `simaver-c-annealtemp.htm`
- **구현 내용**:
  1. **열변형률(Thermal Strain) 분할**:
     $$\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^{me} + \boldsymbol{\varepsilon}^{th}, \quad \boldsymbol{\varepsilon}^{th} = \alpha (T - T_0) \mathbf{I}$$
  2. **온도 의존 탄-소성 보간**:
     $$E = E(T), \quad \nu = \nu(T), \quad \sigma_y = \sigma_y(\bar{\varepsilon}^{pl}, T)$$
  3. **내력 벡터 및 강성 행렬 확장**:
     $$\mathbf{f}_{\text{int}} = \int \mathbf{B}^T \boldsymbol{\sigma}(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^{th})\, dV$$
  4. **Abaqus API 연동**:
     ```python
     material.Elastic(table=((E_val, nu_val, temp_val), ...))
     material.Plastic(table=((yield_val, strain_val, temp_val), ...))
     material.Expansion(type="ISO", table=((alpha_val, temp_val), ...))
     ```

### [패키지 2: 속도 의존 및 고온 소성/크립 (Rate-dependent Plasticity & Creep)]
- **대상 문서**: `simaver-c-ratedepplastexp.htm`, `simaver-c-johnsoncook.htm`, `simaver-c-creep.htm`
- **구현 내용**:
  1. **Cowper-Symonds / Johnson-Cook 동적 항복비율**:
     $$R = \frac{\sigma_y}{\sigma_0} = 1 + \left(\frac{\dot{\bar{\varepsilon}}^{pl}}{D}\right)^{1/p}$$
  2. **Norton / Power-law 크립 적분기**:
     $$\Delta \bar{\varepsilon}^{cr} = A q^n \Delta t$$

### [패키지 3: 고무 손상 및 재료 연화 (Mullins Effect & Ductile Damage)]
- **대상 문서**: `simaver-c-mullinseffect.htm`, `simaver-c-hysteretic.htm`, `simaver-c-damage.htm`
- **구현 내용**:
  1. **Mullins 효과 고무 연화 함수**:
     $$\tilde{\boldsymbol{\sigma}} = \eta \boldsymbol{\sigma}_{\text{hyper}}, \quad \eta = 1 - \frac{1}{r} \text{erf}\left(\frac{U_m - U}{m + \beta U_m}\right)$$
  2. **Ductile / Shear Damage 소성 파손 연동**:
     $$\boldsymbol{\sigma} = (1 - D) \tilde{\boldsymbol{\sigma}}$$

---

## 3. 벤치마크 수행 로드맵
- **원칙**: 사용자 지시대로 **기능 구현과 점검이 완료된 이후**에 Abaqus Verification Reference Manual에 기재된 표준 `.inp` 벤치마크 파일을 생성/실행하여 Abaqus 정해와 대조 검증 수행.
