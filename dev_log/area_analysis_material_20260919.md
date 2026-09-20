# 🔬 WHT_DispFoldSolver: Material Domain Deep Analysis (Living Document)

> **문서 버전**: 2026-09-19  
> **기준 마스터 플랜**: `.agents/plans/material_master_plan_20260919_team_ABC.md`  
> **상태**: S1 (최소 P0) 및 S2 (P1) 완료 / 70개 회귀 테스트 100% 통과  

---

## 1. 열-역학 연계 (Thermal-Mechanical Coupling) 수치해석적 분석

### 1.1 소변형 및 대변형 분할 기구학
Abaqus Theory Manual §4.4.1 표준에 따라 열변형률은 부피 팽창에만 기여하는 구형(Spherical) 텐서로 정의된다:
$$\boldsymbol{\varepsilon}^{th} = \alpha(T)(T - T_0)\mathbf{I}$$

대변형(Large Deformation, Finite Strain) 체계에서는 변형 구배 $\mathbf{F}$를 순수 기계적 변형 $\mathbf{F}_{me}$와 등방 열팽창 $\mathbf{F}_{th}$로 승산 분할(Multiplicative Split)한다:
$$\mathbf{F} = \mathbf{F}_{me} \mathbf{F}_{th}$$
여기서 등방 열팽창의 경우 $\mathbf{F}_{th} = \vartheta \mathbf{I}$, $\vartheta = 1 + \alpha(T)(T - T_0)$ 이다. 따라서:
$$\mathbf{F}_{me} = \frac{1}{\vartheta} \mathbf{F}$$
이 공식은 기존의 고성능 초탄성/탄소성 PK2 응력 및 탄젠트 커널에 변형 구배 $\mathbf{F}$를 입력하기 전에 스칼라 축척 $\vartheta^{-1}$만 곱해주는 어댑터 래퍼(`dispsolver/material/thermal_expansion.py`)로 완벽히 캡슐화된다. 기존 검증된 수만 라인의 요소 커널을 단 한 줄도 손대지 않고 재사용할 수 있는 핵심 이점을 제공한다.

### 1.2 Frozen-Field 원칙과 접선 행렬 보존
사전 정의된 온도 필드($T$)는 기계적 변위($\mathbf{u}$)에 영향을 받지 않는 순방향 결합(One-way Coupling)이다.
따라서 가상일의 원리에 따른 등가 열하중 벡터는:
$$\mathbf{f}^{th} = \int_{V_0} \mathbf{B}^T \mathbf{C} \boldsymbol{\varepsilon}^{th} dV$$
이며, Newton-Raphson 평형 방정식의 잔차 벡터는 다음과 같이 구성된다:
$$\mathbf{r}(\mathbf{u}) = \mathbf{f}_{int}(\mathbf{u}) - \mathbf{f}^{th} - \mathbf{f}_{ext} = \mathbf{0}$$
온도 $T$가 고정(Frozen)되어 있으므로:
$$\frac{\partial \mathbf{f}^{th}}{\partial \mathbf{u}} = \mathbf{0}$$
따라서 구조 접선 강성 행렬 $\mathbf{K} = \frac{\partial \mathbf{f}_{int}}{\partial \mathbf{u}}$의 완전한 대칭성($\mathbf{K} = \mathbf{K}^T$)이 엄밀하게 보존된다. 이는 Intel oneMKL PARDISO의 대칭 Cholesky($LL^T$) 솔버를 100% 안정적으로 구동할 수 있게 한다.

---

## 2. 고급 소성 및 속도/온도 의존 구성방정식

### 2.1 Prager-Ziegler 선형 이동 경화 (Linear Kinematic Hardening)
- **항복 함수**:
  $$f(\boldsymbol{\sigma}, \boldsymbol{\alpha}) = \sqrt{\frac{3}{2}} \|\mathbf{s} - \boldsymbol{\alpha}\| - \sigma_{y0} = 0$$
- **후방 응력(Backstress) 진화식**:
  $$\dot{\boldsymbol{\alpha}} = \frac{2}{3} C_{kin} \dot{\boldsymbol{\varepsilon}}^p$$
- **Bauschinger 역항복 이론치**:
  전방 인장 시 소성 변형에 의해 후방 응력 $\alpha_{11} > 0$이 축적된다. 하중을 반전하여 압축으로 진입할 때:
  $$\sigma_{rev} = \sigma_{fwd} - 2\sigma_{y0}$$
  `tests/test_plastic_hardening.py`에서 $\Delta \sigma = 1.99\sigma_{y0}$까지는 탄성 유지, $\Delta \sigma > 2.0\sigma_{y0}$에서 즉시 압축 항복이 발생하는 거동을 기계 정밀도로 입증하였다.

### 2.2 Johnson-Cook 점소성 모델
고속 충격 및 열연화를 동시에 고려하는 Johnson-Cook 모델:
$$\sigma_y = \left[A + B (\bar{\varepsilon}^p)^n\right] \left[1 + C \ln \left(\max\left(1.0, \frac{\dot{\varepsilon}^p}{\dot{\varepsilon}_0}\right)\right)\right] \left[1 - (T^*)^m\right]$$
- **상동온도(Homologous Temperature)**:
  $$T^* = \text{clamp}\left(\frac{T - T_{room}}{T_{melt} - T_{room}}, 0.0, 1.0\right)$$
- **축퇴 무결성 (Strict Degeneracy)**:
  $T \le T_{room}$ 및 $\dot{\varepsilon}^p \le \dot{\varepsilon}_0$인 기준 조건에서 속도항과 온도항이 정확히 $1.0$이 되어 정적 등온 경화식 $A + B(\bar{\varepsilon}^p)^n$과 오차 $< 10^{-12}$ 수준으로 일치함을 보장한다.
- **완전 용융**: $T \ge T_{melt} \implies \sigma_y = 0.0$.

### 2.3 Cowper-Symonds 동적 속도 의존 모델
금속 판재 및 충격 흡수재의 전형적인 변형률 속도 의존 항복:
$$\sigma_y = \sigma_{y,static} \left[1 + \left(\frac{\dot{\varepsilon}^p}{D}\right)^{1/p}\right]$$
- 정적 극한($\dot{\varepsilon}^p = 0 \implies \sigma_y = \sigma_{static}$)
- 특성 속도($\dot{\varepsilon}^p = D \implies \sigma_y = 2\sigma_{static}$)

---

## 3. 검증 실측치 데이터 매트릭스 (70/70 All Green)

2026-09-19 독립 재측정 결과:
### 2.4 Ogden-Roxburgh 초탄성 Mullins 손상 모델
- **변형률 에너지 밀도**:
  $$W(\mathbf{F}, \eta) = \eta W_{dev}(\mathbf{F}) + W_{vol}(J) + \phi(\eta)$$
- **손상 변수**:
  $$\eta = 1 - \frac{1}{r} \text{erf}\left(\frac{U_{max} - U_{dev}}{m + \beta U_{max}}\right)$$
  무손상 상태($U_{dev} = U_{max}$)에서는 $\eta = 1.0$을 엄밀히 유지하며, 제하 시 $\eta < 1.0$로 연화되고, 재재하하여 $U_{max}$를 돌파하면 무손상 포락선 곡선으로 정확히 복귀함을 검증 완료(`tests/test_mullins_and_bergstrom_boyce.py`).

### 2.5 Bergström-Boyce 2-네트워크 점탄소성 모델
- **네트워크 병렬 분할**: $\mathbf{S} = \mathbf{S}_A + \mathbf{S}_B$
  - Network A: 평형 Arruda-Boyce 8-chain 초탄성 응력
  - Network B: 비선형 점소성 흐름에 의한 완화 응력
- **비선형 유량식**:
  $$\dot{\gamma}_B = \dot{\gamma}_0 \left(\frac{\tau_B}{\hat{\tau}}\right)^m$$
  장기 정적 유지 시 Network B의 응력이 $0$으로 점근 완화되어 평형 Network A 응력으로 수렴함을 검증 완료.

---

## 3. 검증 실측치 데이터 매트릭스 (81/81 All Green)

2026-09-19 독립 재측정 결과:
```
tests\test_dependence_table.py ....                                      [  4%]
tests\test_thermal_mechanical_coupling.py .....                          [ 11%]
tests\test_solver_thermal_expansion_2d.py ..                             [ 13%]
tests\test_solver_thermal_expansion_3d.py ..                             [ 16%]
tests\test_plastic_hardening.py .....                                    [ 22%]
tests\test_mullins_and_bergstrom_boyce.py .....                          [ 28%]
tests\test_radioss_law36_smoothing_and_tail.py ....                      [ 33%]
tests\test_tabular_plasticity_radioss.py ....                            [ 38%]
tests\test_hyperelastic.py ................                              [ 58%]
tests\test_viscoelastic.py ............................                  [ 92%]
tests\test_creep_power_law.py ....                                       [ 97%]
tests\test_bimetallic_strip.py ..                                        [100%]
======================= 81 passed, 1 warning in 84.53s ========================
```

- **3D 자유 팽창 변위**: $\Delta L = \alpha \Delta T L = 0.003000\,\text{mm}$ (오차 $< 10^{-6}\,\text{mm}$)
- **3D 1축 구속 반력**: $R = 210.000\,\text{N}$ (정밀 일치)
- **2D 평면 변형률 팽창 변위**: $\varepsilon_{xx} = (1+\nu)\alpha\Delta T = 1.95\times 10^{-3}$ (정밀 일치)
- **2D 1축 구속 반력**: $R = 210.000\,\text{N}$ (정밀 일치)
- **Abaqus 공인 4대 벤치마크 (`benchmark_abaqus_material_inp.py`)**:
  - `temp_elastic_ef1`: 오차 $0.010\%$ (200.0208 N / 160.0166 N) **[PASS]**
  - `temp_plastic_mises`: 소성 가공 연화 정밀 일치 **[PASS]**
  - `thermal_expansion_simple`: 구속 반력 $210.0000\,\text{N}$ (오차 $0.0000\%$) **[PASS]**
  - `johnson_cook_dynamic`: 고속 충격 속도 경화 및 열연화 일치 **[PASS]**
- **다요소 구조 열-역학 연계 벤치마크 (`benchmark_bimetallic_strip.py`)**:
  - **2D Plane Strain (CPE4I 9-모드 비적합 요소)**:
    - Timoshenko 1925 이론해: $4.49140\,\text{mm}$
    - FEM 실측 처짐: $4.48648\,\text{mm}$ (상대 오차 **$0.110\%$**, PASS)
  - **3D Solid Beam (SOLID_SHELL 8절점 소변형 솔리드 셸)**:
    - Timoshenko 1925 이론해: $3.31383\,\text{mm}$
    - FEM 실측 처짐: $3.30695\,\text{mm}$ (상대 오차 **$0.208\%$**, PASS)
- **14층 플렉서블 디스플레이 열 팽창 & 층간 전단 벤치마크 (`benchmark_display_thermal_rollup.py`)**:
  - $\Delta T = 40\,\text{K}$ 구동 시 PET 팁 자유 신장량 이론치 $31.200\,\mu\text{m}$ 대비 FEM 실측 $31.234\,\mu\text{m}$ (상대 오차 **$0.11\%$**, PASS)
  - 연질 PSA 층간 전단 흡수 차이 $0.127\,\mu\text{m}$ 발생, 2회 뉴턴 반복만으로 무결점 수렴.

---

## 4. 📌 Open 항목 및 향후 로드맵 (Living Task List)

- **[x] P2: Mullins 효과 및 진보형 탄성 손상 모델**:
  - 충전 고무 및 디스플레이 PSA의 반복 하중 연화(Mullins damage) 완료 (`mullins_damage.py`).
- **[x] P2: Bergström-Boyce 2-네트워크 점탄성/소성 하이브리드 모델**:
  - PSA 대변형 크리프 및 회복 거동 해석 완료 (`bergstrom_boyce.py`).
- **[x] P3: 점소성 크리프(Creep) 및 2층(Two-layer) 점탄성-점소성**:
  - Abaqus `*CREEP` 거듭제곱 법칙($\dot{\varepsilon}^{cr} = A q^n t^m$) 음함수 방사형 리턴 매핑 및 일관 탄젠트 완료 (`creep_numba.py`, `tests/test_creep_power_law.py`).
- **[x] P3: 다요소 구조 열-역학 연계 벤치마크 검증**:
  - Timoshenko 바이메탈 스트립 2D/3D (오차 0.11%, 0.21%) 및 14층 디스플레이 열-역학 롤업 검증 완료.

