# FrontISTR 이론 매뉴얼 (지배방정식, 이산화, 구성칙, 접촉역학, 비선형 해법, 선형 솔버, 병렬화)

> 크롤링 출처: `https://docs.frontistr.com/ko/` (수록 페이지 수: 33개)

## 수록 문서 목차

- `theory/99_reference.md` — 참고문헌
- `theory/appendix_static_linear.md` — 선형 탄성 정적 해석 (입문)
- `theory/constitutive/01_linear_elastic.md` — 선형 탄성
- `theory/constitutive/02_hyperelastic.md` — 초탄성
- `theory/constitutive/03_elastoplastic.md` — 탄소성
- `theory/constitutive/04_viscoelastic.md` — 점탄성
- `theory/constitutive/05_creep.md` — 크리프
- `theory/constitutive/06_thermal_properties.md` — 열물성
- `theory/contact/01_contact.md` — 접촉 해석
- `theory/discretization/01_shape_functions.md` — 형상함수와 유한요소 근사
- `theory/discretization/02_global_derivative.md` — 형상함수의 공간 미분
- `theory/discretization/03_internal_force.md` — 내력 가상일의 이산화
- `theory/discretization/04_external_force_and_assembly.md` — 외력의 가상일과 전체 방정식의 조립
- `theory/discretization/05_numerical_integration.md` — 수치 적분
- `theory/discretization/08_element_solid.md` — 3차원 솔리드 요소의 형상 함수
- `theory/discretization/09_element_advanced.md` — 고성능 요소 정식화
- `theory/governing/01a_kinematics.md` — 운동·변형·변형률
- `theory/governing/01b_stress_balance.md` — 응력과 보존 법칙
- `theory/governing/01c_virtual_work.md` — 가상일의 원리
- `theory/governing/01d_incremental_analysis.md` — 증분해석의 틀
- `theory/governing/02_heat_equation.md` — 열전도 방정식
- `theory/index.md` — 이론 매뉴얼
- `theory/linear_solver/01_solver.md` — 선형 솔버
- `theory/notation/01_tensor_notation.md` — 텐서 표기법과 수학적 기초
- `theory/notation/02_symbol_list.md` — 물리량 기호 목록
- `theory/parallel/01_parallel.md` — 병렬 유한요소법
- `theory/solution_methods/01_newton_raphson.md` — Newton-Raphson 법
- `theory/solution_methods/01b_convergence.md` — 수렴 판정
- `theory/solution_methods/02_dynamic.md` — 동적 해석 기법
- `theory/solution_methods/03_eigen.md` — 모드 해석
- `theory/solution_methods/04_frequency_response.md` — 주파수 응답 해석
- `theory/solution_methods/05_transient_heat.md` — 비정상 열전도 해석
- `theory/solution_methods/06_geometric_bc.md` — 기하학적 경계 조건의 처리

---


<!-- ========================================== -->
<!-- FILE: theory/99_reference.md | TITLE: 참고문헌 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/99_reference.html -->

# 참고문헌

 
## 연속체역학·유한요소법

 
- Hisada·Noguchi, 비선형 유한요소법의 기초와 응용, Maruzen(1995).
- O.C.Zienkiewicz, R.L.Taylor: The Finite Element Method, 6th Ed., Vol.2: McGraw-Hill, 2005
- 계산역학 핸드북 제I권 유한요소법(구조편), 일본기계학회(1998).
- Washizu Kyuichiro·Miyamoto Hiroshi·Yamada Yoshiaki·Yamamoto Yoshiyuki·Kawai Tadahiko, 유한요소법 핸드북, (I 기초편), Baifukan(1982).

 
## 모드 해석

 
- Mori Masatake·Sugihara Masaaki·Murota Kazuo, 선형 계산, Iwanami Shoten(1994).
- Lois Komzsik: The Lanczos Method Evolution and Application: Siam, 2003.

 
## 동적 해석

 
- Togawa Hayato, 유한요소법에 의한 진동 해석, Science-sha(1997)

 
## 열전도 해석

 
- Yagawa Genki·Miyazaki Noriyuki, 유한요소법에 의한 열응력·크리프. 열전도 해석, Science-sha(1985)


---


<!-- ========================================== -->
<!-- FILE: theory/appendix_static_linear.md | TITLE: 선형 탄성 정적 해석 (입문) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/appendix_static_linear.html -->

# 선형 탄성 정적 해석 (입문)

 이 절에서는 미소변형 이론에 기반한 탄성 정적 해석의 정식화를 설명합니다. 응력-변형률 관계에는 선형 탄성을 가정합니다. 이 장은 유한요소 구조 해석의 전체 구성을 이해하기 위한 입문으로, 하나의 장에서 완결되도록 구성되어 있습니다.

 가상일의 원리의 일반론(현재 배치 표현, 초기 배치 표현 및 미소변형으로의 귀착)은 [가상일의 원리](governing/01c_virtual_work.html)를 참조하십시오. 선형 탄성 구성 법칙의 자세한 내용은 [선형 탄성](constitutive/01_linear_elastic.html), 텐서 및 Voigt 표기 규약은 [텐서 표기와 수학적 기초](notation/01_tensor_notation.html), 유한변형의 일반 정식화는 [운동·변형·변형률](governing/01a_kinematics.html), 비선형 문제의 해법은 [접선 강성 행렬](solution_methods/01_newton_raphson.html)을 참조하십시오.

 
## 기본 방정식

 미소변형 및 선형 탄성의 가정하에서 고체역학의 경계값 문제는 평형 방정식, 역학적 경계 조건 및 기하학적 경계 조건(본질적 경계 조건)으로 구성됩니다(그림 2.1.1 참조).

 \[\begin{equation} \nabla \cdot \boldsymbol{\sigma} + \overline{\boldsymbol{b}} = \boldsymbol{0} \quad \text{에서} \ V \label{eq:2.1.1} \end{equation}\] \[\begin{equation} \boldsymbol{\sigma} \cdot \boldsymbol{n} = \overline{\boldsymbol{t}} \quad \text{위에서} \ S_t \label{eq:2.1.2} \end{equation}\] \[\begin{equation} \boldsymbol{u} = \overline{\boldsymbol{u}} \quad \text{위에서} \ S_u \label{eq:2.1.3} \end{equation}\] 여기서 \(\boldsymbol{\sigma}\)는 Cauchy 응력, \(\overline{\boldsymbol{b}}\)는 단위 체적당 체적력, \(\overline{\boldsymbol{t}}\)는 지정 표면력, \(\overline{\boldsymbol{u}}\)는 지정 변위이며, \(S_t, S_u\)는 각각 역학적 경계와 기하학적 경계입니다.

 ![고체역학의 경계값 문제(미소변형 문제)](/theory/image/theory01_01.png)

 그림 2.1.1 고체역학의 경계값 문제(미소변형 문제)

 [대칭 기울기 연산자](notation/01_tensor_notation.html)를 사용하면 변형률-변위 관계는 다음과 같습니다.

 \[\begin{equation} \boldsymbol{\varepsilon} = \nabla_S \boldsymbol{u} \label{eq:2.1.4} \end{equation}\] 선형 탄성 구성 방정식은 다음과 같습니다.

 \[\begin{equation} \boldsymbol{\sigma} = \boldsymbol{C} : \boldsymbol{\varepsilon} \label{eq:2.1.5} \end{equation}\] 여기서 \(\boldsymbol{C}\)는 4차 탄성 텐서입니다.

 
## 가상일의 원리

 가상일의 원리의 일반 형식(현재 배치 표현, 초기 배치 표현 및 미소변형으로의 귀착)은 [가상일의 원리](governing/01c_virtual_work.html)에 정리되어 있습니다. 미소변형 및 선형 탄성의 가정하에서 약형은 다음과 같습니다.

 \[\begin{equation} \int_V \boldsymbol{\sigma} : \delta \boldsymbol{\varepsilon}\, dV = \int_{S_t} \overline{\boldsymbol{t}} \cdot \delta \boldsymbol{u}\, dS + \int_V \overline{\boldsymbol{b}} \cdot \delta \boldsymbol{u}\, dV \label{eq:2.1.6} \end{equation}\] \[\begin{equation} \delta \boldsymbol{u} = \boldsymbol{0} \quad \text{위에서} \ S_u \label{eq:2.1.7} \end{equation}\] 구성 방정식 \eqref{eq:2.1.5}를 대입하고 [Voigt 표기](notation/01_tensor_notation.html)로 \(\hat{\sigma} = D\, \hat{\varepsilon}\)라고 쓰면, 이산화에 직접 사용하는 다음 형식을 얻습니다.

 \[\begin{equation} \int_V \delta \hat{\varepsilon}^T\, D\, \hat{\varepsilon}\, dV = \int_{S_t} \delta \boldsymbol{u}^T\, \overline{\boldsymbol{t}}\, dS + \int_V \delta \boldsymbol{u}^T\, \overline{\boldsymbol{b}}\, dV \label{eq:2.1.10} \end{equation}\] 여기서 \(D\)는 [선형 탄성](constitutive/01_linear_elastic.html)에서 정의한 탄성 행렬입니다. 식 \eqref{eq:2.1.10}과 \eqref{eq:2.1.7}이 아래에서 이산화하는 가상일의 원리를 구성합니다.

 
## 이산화와 전체 방정식의 조립

 식 \( \eqref{eq:2.1.10} \)의 가상일의 원리를 유한요소별로 이산화하면 다음과 같습니다.

 \[\begin{equation} \sum_e \int_{V^e} \delta \hat{\varepsilon}^T\, D\, \hat{\varepsilon}\, dV = \sum_e \int_{S^e_t} \delta \boldsymbol{u}^T\, \overline{\boldsymbol{t}}\, dS + \sum_e \int_{V^e} \delta \boldsymbol{u}^T\, \overline{\boldsymbol{b}}\, dV \label{eq:2.1.11} \end{equation}\] 각 요소에 대해 요소를 구성하는 절점의 변위를 사용하여 변위장을 다음과 같이 보간합니다.

 \[\begin{equation} \boldsymbol{u} = \sum^m_{i=1} N_i\, \boldsymbol{u}_i = \boldsymbol{N}\, \boldsymbol{U} \label{eq:2.1.12} \end{equation}\] 그러면 식 \(\eqref{eq:2.1.4}\)를 사용하여 변형률은 다음과 같이 주어집니다.

 \[\begin{equation} \hat{\varepsilon} = \boldsymbol{B}\, \boldsymbol{U} \label{eq:2.1.13} \end{equation}\] 식 \(\eqref{eq:2.1.12}\)와 \(\eqref{eq:2.1.13}\)을 식 \(\eqref{eq:2.1.11}\)에 대입하면 다음을 얻습니다.

 \[\begin{equation} \sum_e \delta \boldsymbol{U}^T \left( \int_{V^e} \boldsymbol{B}^T D\, \boldsymbol{B}\, dV \right) \boldsymbol{U} = \sum_e \delta \boldsymbol{U}^T \int_{S_t^e} \boldsymbol{N}^T\, \overline{\boldsymbol{t}}\, dS + \sum_e \delta \boldsymbol{U}^T \int_{V^e} \boldsymbol{N}^T\, \overline{\boldsymbol{b}}\, dV \label{eq:2.1.14} \end{equation}\] 식 \(\eqref{eq:2.1.14}\)는 다음과 같이 쓸 수 있습니다.

 \[\begin{equation} \delta \boldsymbol{U}^T\, \boldsymbol{K}\, \boldsymbol{U} = \delta \boldsymbol{U}^T\, \boldsymbol{F} \label{eq:2.1.15} \end{equation}\] 여기서 식 \(\eqref{eq:2.1.16}\) 및 \(\eqref{eq:2.1.17}\)로 정의되는 행렬과 벡터의 성분은 각 유한요소에 대해 계산하고 중첩하여 조립할 수 있습니다.

 \[\begin{equation} \boldsymbol{K} = \sum_e \int_{V^e} \boldsymbol{B}^T\, D\, \boldsymbol{B}\, dV \label{eq:2.1.16} \end{equation}\] \[\begin{equation} \boldsymbol{F} = \sum_e \left( \int_{S_t^e} \boldsymbol{N}^T\, \overline{\boldsymbol{t}}\, dS + \int_{V^e} \boldsymbol{N}^T\, \overline{\boldsymbol{b}}\, dV \right) \label{eq:2.1.17} \end{equation}\] 식 \(\eqref{eq:2.1.15}\)는 임의의 가상 변위 \(\delta \boldsymbol{U}\)에 대해 성립하므로 다음 방정식을 얻습니다.

 \[\begin{equation} \boldsymbol{K}\, \boldsymbol{U} = \boldsymbol{F} \label{eq:2.1.18} \end{equation}\] 한편 식 \(\eqref{eq:2.1.3}\)의 변위 경계 조건은 다음과 같이 표현됩니다.

 \[\begin{equation} \boldsymbol{U} = \overline{\boldsymbol{U}} \label{eq:2.1.19} \end{equation}\] 식 \(\eqref{eq:2.1.19}\)의 구속 조건을 만족하도록 식 \(\eqref{eq:2.1.18}\)을 풀면 절점 변위 \(\boldsymbol{U}\)를 구할 수 있습니다.

 
## 관련 항목


---


<!-- ========================================== -->
<!-- FILE: theory/constitutive/01_linear_elastic.md | TITLE: 선형 탄성 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/constitutive/01_linear_elastic.html -->

# 선형 탄성

 이 장에서는 선형 탄성 재료의 구성 법칙(Hooke 법칙)을 다룬다. 모델 선택과 입력 지정에 대한 자세한 내용은 기능 편 `03_material`을 참조한다.

 
## 현상(선형 탄성 응답)

 선형 탄성체란 하중에 대한 변형률이 충분히 작은 영역에서 응력과 변형률이 선형 관계를 가지며, 제하 시 모든 변형률이 원래 상태로 되돌아가는 가역적이고 이력 비의존적인 응답을 보이는 재료를 말한다. 소성이나 크리프와 달리 상태를 나타내는 내부 변수(소성 변형률이나 점성 변형률 등)를 가지지 않으며, 응력은 현재의 변형률만으로 유일하게 정해진다.

 미소 변형 영역에서는 Cauchy 응력 \(\boldsymbol{\sigma}\)와 미소 변형률 \(\boldsymbol{\varepsilon}\)을 사용하고, 유한 변형 영역(큰 회전을 포함하지만 변형률 자체는 작은 문제)에서는 제2 Piola-Kirchhoff 응력 \(\boldsymbol{S}\)와 Green-Lagrange 변형률 \(\boldsymbol{E}\)를 사용한다(St.Venant-Kirchhoff 재료).

 
## 구성 법칙

 
### 등방성 선형 탄성체

 미소 변형 영역에서의 등방성 Hooke 법칙은 Lamé 상수 \(\lambda, \mu\)를 사용하여

 \[ \boldsymbol{\sigma} = \lambda\, \mathrm{tr}(\boldsymbol{\varepsilon})\, \boldsymbol{I} + 2\mu\, \boldsymbol{\varepsilon} \] 으로 표현된다. 성분 표기에서는

 \[ C_{ijkl} = \lambda\, \delta_{ij}\delta_{kl} + \mu\, \bigl( \delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk} \bigr) \] 가 되며, 변형률과의 이중 축약에 의해 \(\sigma_{ij} = C_{ijkl}\,\varepsilon_{kl}\)을 얻는다. Lamé 상수는 영률 \(E\)와 포아송비 \(\nu\)와 다음 관계에 있다:

 \[ \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}, \qquad \mu = \frac{E}{2(1+\nu)}. \] Voigt 표기에서 변형률 벡터 \(\hat{\varepsilon}\)과 응력 벡터 \(\hat{\sigma}\)의 관계는 \(\hat{\sigma} = D\, \hat{\varepsilon}\)로 쓰며, 3차원 재료 행렬 \(D\)는

 \[ D = \begin{bmatrix} \lambda + 2\mu & \lambda & \lambda & 0 & 0 & 0 \\ \lambda & \lambda + 2\mu & \lambda & 0 & 0 & 0 \\ \lambda & \lambda & \lambda + 2\mu & 0 & 0 & 0 \\ 0 & 0 & 0 & \mu & 0 & 0 \\ 0 & 0 & 0 & 0 & \mu & 0 \\ 0 & 0 & 0 & 0 & 0 & \mu \end{bmatrix} \] 으로 주어진다(구현에서는 이 행렬 표현을 사용한다).

 
### 직교이방성 선형 탄성체

 주축 방향마다 탄성 상수가 다른 재료에서는 9개의 독립 상수를 갖는 직교이방성 모델을 사용한다. 즉, 주축 방향 영률 \(E_1, E_2, E_3\), 주축 간 포아송비 \(\nu_{12}, \nu_{23}, \nu_{31}\), 전단 탄성계수 \(G_{12}, G_{23}, G_{31}\)이다. 구체적인 \(D\) 행렬의 형식은 일반적인 참고문헌을 참조한다.

 
### 유한 변형 영역: St.Venant-Kirchhoff 재료

 큰 회전을 포함하지만 변형률 자체는 작은 문제에서는 제2 Piola-Kirchhoff 응력 \(\boldsymbol{S}\)와 Green-Lagrange 변형률 \(\boldsymbol{E}\)에 대한 선형 탄성 법칙으로 St.Venant-Kirchhoff 재료를 사용한다:

 \[ \boldsymbol{S} = \boldsymbol{C} : \boldsymbol{E}, \qquad C_{ijkl} = \lambda\, \delta_{ij}\delta_{kl} + \mu\, \bigl( \delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk} \bigr). \] Lamé 상수 \(\lambda, \mu\)의 정의는 미소 변형의 경우와 같다. 다만 응력과 변형률 정의의 조합이 다르므로, 미소 변형의 Hooke 법칙과는 별도의 구성 법칙임에 유의한다.

 
### 아탄성 재료

 Updated Lagrange법에서는 상대 Kirchhoff 응력 텐서의 Jaumann 속도 \(\hat{\boldsymbol{\sigma}}^{\nabla J}\)와 변형 속도 텐서 \(\boldsymbol{D}\) 사이의 선형 탄성 법칙

 \[ \hat{\boldsymbol{\sigma}}^{\nabla J} = \boldsymbol{C} : \boldsymbol{D} \] 을 사용하는 아탄성 재료를 이용한다. 이 경우도 Lamé 상수에 의한 \(\boldsymbol{C}\)의 정의는 위와 동일하지만, 응력과 변형률의 조합이 다르므로 별도의 구성 법칙이다. 시간 적분 절차(전진 Euler 적분과 중앙차분 근사)에 의해 \(\boldsymbol{\sigma}_{t_{n+1}}\)가 응력 갱신되는 자세한 내용은 tmptexdocs/11b_continuum_mechanics.tex을 참조한다.

 
## 2차원 해석 및 셸 요소로의 축약

 
### 평면 응력(\(\sigma_{33} = 0\))

 얇은 판이나 셸의 하중을 나타내는 평면 응력에서는 \(\sigma_{33} = \sigma_{13} = \sigma_{23} = 0\)을 가정하고, 면외 방향 변형률 성분 \(\varepsilon_{33}\)을 대수적으로 제거한 \(D\)를 사용한다:

 \[ D = \frac{E}{1 - \nu^2} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \dfrac{1-\nu}{2} \end{bmatrix}. \] 
### 평면 변형률(\(\varepsilon_{33} = 0\))

 두께 방향이 구속된 긴 구조물의 단면을 나타내는 평면 변형률에서는 \(\varepsilon_{33} = \varepsilon_{13} = \varepsilon_{23} = 0\)을 가정한다:

 \[ D = \frac{E}{(1+\nu)(1-2\nu)} \begin{bmatrix} 1-\nu & \nu & 0 \\ \nu & 1-\nu & 0 \\ 0 & 0 & \dfrac{1-2\nu}{2} \end{bmatrix}. \] 
### 축대칭

 원통 좌표 \((r, \theta, z)\)에서 \(\theta\) 방향으로 균일한 문제를 다루는 축대칭 해석에서는 \(\boldsymbol{\sigma} = (\sigma_{rr}, \sigma_{\theta\theta}, \sigma_{zz}, \sigma_{rz})^T\), \(\boldsymbol{\varepsilon} = (\varepsilon_{rr}, \varepsilon_{\theta\theta}, \varepsilon_{zz}, 2\varepsilon_{rz})^T\)의 4개 성분을 유지하고,

 \[ D = \frac{E}{(1+\nu)(1-2\nu)} \begin{bmatrix} 1-\nu & \nu & \nu & 0 \\ \nu & 1-\nu & \nu & 0 \\ \nu & \nu & 1-\nu & 0 \\ 0 & 0 & 0 & \dfrac{1-2\nu}{2} \end{bmatrix} \] 이 된다.

 
### 셸 요소(평면 응력 + 횡전단)

 셸 요소에서는 면내 응력을 평면 응력으로 취급하고, 추가로 횡전단 변형률 성분 \((2\varepsilon_{13}, 2\varepsilon_{23})\)에 대해 전단 보정계수 \(\kappa\)(일반적으로 \(\kappa = 5/6\))를 적용한 \(\kappa\, G\)를 사용한다. 요소별 구체적인 정식화는 [고성능 요소 정식화](../discretization/09_element_advanced.html)를 참조한다. 기능 편 `03_material`에 있는 "셸 요소는 선형 탄성만 지원"이라는 사양은 면내 상태를 평면 응력으로 취급하는 이 단순화에 기반한다.

 
## 온도 의존성

 영률 \(E\), 포아송비 \(\nu\), 선팽창계수 \(\alpha\)는 일반적으로 온도 \(T\)의 함수로 지정된다. FrontISTR에서는 유한한 온도점 열 \(T_1 < T_2 < \cdots < T_n\)에 대응하는 물성값을 입력하며, 계산 온도 \(T\)에서의 해당 물성값은 인접한 온도점 사이의 선형 보간으로 평가한다. 범위 밖의 온도(\(T < T_1\) 또는 \(T > T_n\))에 대해서는 가장 가까운 끝점 \(T_1\) 또는 \(T_n\)의 값으로 외삽한다(상수 외삽).

 열 물성(밀도, 비열, 열전도율)의 온도 의존성은 [열 물성](06_thermal_properties.html)을 참조한다. 보간 규칙은 이들 물성과 공통이다.

 
## 관련 항목

 
- [텐서 표기와 수학적 기초](../notation/01_tensor_notation.html) — Voigt 표기의 규약과 재료 행렬 \(D\)의 위치
- [운동·변형·변형률](../governing/01a_kinematics.html) — 변형 구배와 변형률 텐서에 대한 일반론
- [초탄성](02_hyperelastic.html) — 큰 변형률을 포함하는 가역 응답으로의 확장
- [탄소성](03_elastoplastic.html) — 소성 변형률을 포함하는 아탄성 및 소성 모델
- [열 물성](06_thermal_properties.html) — 열전도 및 열응력 연계에 사용하는 물성
- [재료 데이터(기능)](../../functions_reference/03_material.html) — 모델 선택지와 선택 방법


---


<!-- ========================================== -->
<!-- FILE: theory/constitutive/02_hyperelastic.md | TITLE: 초탄성 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/constitutive/02_hyperelastic.html -->

# 초탄성

 본 장에서는 FrontISTR가 지원하는 초탄성 구성 법칙을 다룬다. 모델 선택과 입력 지정의 자세한 내용은 기능편 `03_material`을 참조한다.

 
## 현상(대변형 가역 응답)

 TBD(다음 단계에서 본문화).

 
## 구성 법칙의 틀

 등방성 초탄성 재료의 탄성 포텐셜 에너지는 응력이 작용하지 않는 초기 상태로부터의 등방성 응답에서 얻어지며, 우 Cauchy-Green 변형 텐서 \(C\)의 주불변량 \((I_1, I_2, I_3)\) 또는 체적 변화를 제거한 변형 텐서의 주불변량 \((\overline{I}_1, \overline{I}_2, \overline{I}_3)\)의 함수, 즉 \(W = W(I_1, I_2, I_3)\) 또는 \(W = W(\overline{I}_1, \overline{I}_2, \overline{I}_3)\)로 나타낼 수 있다.

 초탄성 재료의 구성식은 제2 Piola-Kirchhoff 응력과 Green-Lagrange 변형률의 관계로 정의되며, 그 변형 해석에는 Total Lagrange법을 적용한다.

 탄성 포텐셜 에너지 \(W\)를 알면 다음과 같이 제2 Piola-Kirchhoff 응력 및 응력-변형률 관계를 계산할 수 있다.

 \[ S = 2\frac{\partial W}{\partial C} \] \[ \mathbb{C} = 4 \frac{\partial^2 W}{\partial C \partial C} \] 이하에서는 FrontISTR가 채택하는 초탄성 모델의 탄성 포텐셜 에너지 \(W\)를 열거한다.

 
## Neo-Hookean 초탄성 모델

 Neo-Hookean 초탄성 모델은 등방성을 갖는 선형법칙(Hooke 법칙)을 대변형 문제에 적용할 수 있도록 확장한 것이다. 그 탄성 포텐셜은 다음과 같다.

 \[ W = C_{10} (\overline{I}_1 - 3) + \frac{1}{D} (J - 1)^2 \] 여기서 \(C_{10}\)과 \(D\)는 재료 상수이다.

 
## Mooney-Rivlin 초탄성 모델

 \[ W = C_{10} (\overline{I}_1 - 3) + C_{01} (\overline{I}_2 - 3) + \frac{1}{D} (J - 1)^2 \] 여기서 \(C_{10}\), \(C_{01}\)과 \(D\)는 재료 상수이다.

 
## Mooney-Rivlin 이방성 초탄성 모델

 \[ W = C_{10} (\overline{I}_1 - 3) + C_{01} (\overline{I}_2 - 3) + \frac{1}{D} (J - 1)^2 + C_{42} (\overline{I}_4 - 1)^2 + C_{43} (\overline{I}_4 - 1)^3 \] 여기서 \(C_{10}\), \(C_{01}\)과 \(D\)는 등방성 모델과 공통인 재료 상수이며, \(C_{42}\)와 \(C_{43}\)은 이방성 전용 재료 상수이다.

 
## Arruda-Boyce 초탄성 모델

 \[ \begin{aligned} W &= \mu \left[ \frac{1}{2}(\overline{I}_1 - 3) + \frac{1}{20\lambda_m^2}(\overline{I}_1^2 - 9) + \frac{11}{1050\lambda_m^4}(\overline{I}_1^3 - 27) \right. \\ &\quad \left. + \frac{19}{7000\lambda_m^6}(\overline{I}_1^4 - 81) + \frac{519}{673750\lambda_m^8}(\overline{I}_1^5 - 243) \right] + \frac{1}{D} \left( \frac{J^2 - 1}{2} - \ln J \right) \end{aligned} \] \[ \mu = \frac{\mu_0}{1 + \dfrac{3}{5\lambda_m^2} + \dfrac{99}{175\lambda_m^4} + \dfrac{513}{875\lambda_m^6} + \dfrac{42039}{67375\lambda_m^8}} \] 여기서 \(\mu_0\), \(\lambda_m\)과 \(D\)는 재료 상수이다.

 
## 관련 항목

 
- [선형 탄성](01_linear_elastic.html) — 미소변형의 탄성 응답
- [탄소성](03_elastoplastic.html) — 비가역 변형을 포함하는 구성 법칙
- [재료 데이터(기능)](../../functions_reference/03_material.html) — 모델 선택지와 사용 구분


---


<!-- ========================================== -->
<!-- FILE: theory/constitutive/03_elastoplastic.md | TITLE: 탄소성 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/constitutive/03_elastoplastic.html -->

# 탄소성

 이 장에서는 FrontISTR가 채택하는 탄소성 구성 법칙의 틀을 다룬다. 선택과 입력 지정에 대한 자세한 내용은 기능 편 `03_material`을 참조한다.

 
## 현상(소성 변형)

 TBD(다음 단계에서 본문화).

 
## 구성 법칙의 틀

 FrontISTR에서는 연관 유동 법칙에 따르는 탄소성 구성식을 채택하고 있다. 또한 이 구성식은 Kirchhoff 응력의 Jaumann 속도와 변형 속도 텐서의 관계를 나타내며, 변형 해석에는 갱신 라그랑주법을 적용한다.

 탄소성체의 항복 조건이 다음과 같이 주어진다고 한다.

 초기 항복 조건:

 \[ F(\sigma, \sigma_{y_0}) \] 후속 항복 조건:

 \[ F(\sigma, \sigma_y(\overline{e}^p)) \] 여기서

 
- \(F\): 항복 함수
- \(\sigma_{y_0}\): 초기 항복 응력
- \(\sigma_y\): 후속 항복 응력
- \(\sigma\): 응력 텐서
- \(e\): 미소 변형률 텐서
- \(e^p\): 소성 변형률 텐서
- \(\overline{e}^p\): 등가 소성 변형률

 항복 응력-등가 소성 변형률 관계는 단축 상태의 응력-소성 변형률 관계와 일치한다고 한다.

 **단축 상태의 응력-소성 변형률 관계**

 \[ \sigma = H(e^p) \] \[ \frac{d\sigma}{de^p} = H' \] 여기서 \(H'\)는 변형 경화 계수이다.

 **등가 응력-등가 소성 변형률 관계**

 \[ \overline{\sigma} = H(\overline{e}^p) \] \[ \dot{\overline{\sigma}} = H' \dot{\overline{e}^p} \] 후속 항복 함수는 일반적으로 온도와 소성 변형률 일의 함수이지만, 여기서는 단순화를 위해 등가 소성 변형률 \(\overline{e}^p\)만의 함수라고 한다. 소성 변형이 진행되는 동안 \(F = 0\)이 계속 만족되어야 하므로 다음 식이 성립해야 한다.

 \[ \dot{F} = \frac{\partial F}{\partial \sigma} : \dot{\sigma} + \frac{\partial F}{\partial e^p} : \dot{e}^p = 0 \] 여기서 \(\dot{F}\)는 \(F\)의 시간 미분을 나타낸다. 이후 어떤 양 \(A\)의 시간 미분을 \(\dot{A}\)로 나타낸다.

 여기서 소성 포텐셜 \(\Theta\)의 존재를 가정하고 소성 변형률 속도를 다음 식으로 표현한다.

 \[ \dot{e}^p = \dot{\lambda} \frac{\partial \Theta}{\partial \sigma} \] 여기서 \(\dot{\lambda}\)는 계수이다.

 또한 소성 포텐셜 \(\Theta\)가 항복 함수 \(F\)와 같다고 하여 다음의 연관 유동 법칙을 가정한다.

 \[ \dot{e}^p = \dot{\lambda} \frac{\partial F}{\partial \sigma} \] 이 식을 정합 조건식에 대입하면 다음 식을 얻는다.

 \[ \dot{\lambda} = \frac{a^T : d_D}{A + a^T : D : a} \dot{e} \] 여기서 \(D\)는 탄성 행렬이며,

 \[ a^T = \frac{\partial F}{\partial \sigma}, \qquad d_D = D a^T, \qquad A = -\frac{a}{\dot{\lambda}} \frac{\partial F}{\partial e^p} : \dot{e}^p \] 탄소성의 응력-변형률 관계식은 다음과 같이 쓸 수 있다.

 \[ \dot{\sigma} = \left\{ D - \frac{d_D \otimes d_D^T}{A + d_D^T a} \right\} : \dot{e} \] 탄소성 재료의 항복 함수 \(F\)를 알면 이 식으로부터 그 구성식을 얻을 수 있다.

 
## 항복 함수

 아래에 FrontISTR가 채택하는 탄소성 항복 함수를 열거한다.

 
### Von Mises 항복 함수

 \[ F = \sqrt{3 J_2} - \sigma_y = 0 \] 여기서 \(J_2\)는 편차 응력 텐서의 제2 불변량이다.

 
### Mohr-Coulomb 항복 함수

 \[ F = \sigma_1 - \sigma_3 + (\sigma_1 + \sigma_3) \sin\phi - 2 c \cos\phi = 0 \] 여기서 \(\sigma_1, \sigma_3\)는 최대·최소 주응력, \(c\)는 점착력, \(\phi\)는 내부 마찰각이다.

 
### Drucker-Prager 항복 함수

 \[ F = \sqrt{J_2} - \alpha \sigma : I - \sigma_y = 0 \] 여기서는 재료 상수 \(\alpha\)와 \(\sigma_y\)를 재료의 점착력과 마찰각으로부터 다음과 같이 계산한다.

 \[ \alpha = \frac{2 \sin\phi}{3 + \sin\phi}, \qquad \sigma_y = \frac{6 c \cos\phi}{3 + \sin\phi} \] 
## 관련 항목

 
- [선형 탄성](01_linear_elastic.html) — 탄성 응답(탄소성의 탄성 부분)
- [접선 강성 행렬](../solution_methods/01_newton_raphson.html) — 재료 비선형을 포함하는 증분 해석의 접선 강성
- [재료 데이터(기능)](../../functions_reference/03_material.html) — 항복 함수·경화 법칙의 선택지와 선택 방법


---


<!-- ========================================== -->
<!-- FILE: theory/constitutive/04_viscoelastic.md | TITLE: 점탄성 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/constitutive/04_viscoelastic.html -->

# 점탄성

 본 장에서는 FrontISTR가 채택하는 점탄성 구성 법칙을 다룬다. 사용 구분·입력 지정의 상세는 기능편 [재료 데이터](../../functions_reference/03_material.html)를 참조한다.

 
## 현상(점탄성 응답)

 TBD(다음 단계에서 본문 작성).

 
## 일반화 Maxwell 모델

 FrontISTR에서는 일반화 Maxwell 모델을 채택하고 있다. 편차 변형률을 \(e\), 가지 \(m\)에서의 편차 점성 변형률을 \(q^{(m)}\)로 하여 편차 탄성 변형률 \(h^{(m)}\)를

 \[ h^{(m)} \equiv e - q^{(m)} \] 로 정의한다. 그러면 구성식은 다음과 같이 \(e\)와 \(h^{(m)}\)의 함수가 된다.

 \[ \sigma(t) = K \, \operatorname{tr}\varepsilon \, I + 2 G_0 (\mu_\infty e(t) + \mu h) \] 여기서

 \[ \mu h = \sum_{m=1}^{M} \mu_m h^{(m)}(t), \qquad \mu_\infty + \sum_{m=1}^{M} \mu_m = 1 \] 이다. 또한 \(h^{(m)}\)는

 \[ \dot{h}^{(m)}(t) + \frac{1}{\lambda_m} h^{(m)}(t) = \dot{e}(t) \] 로부터 구한다. 여기서 \(\lambda_m\)는 완화(relaxation) 시간이다.

 
## Prony 급수

 완화계수 \(G\)는 다음 Prony 급수로 나타낸다.

 \[ G(t) = G_0 \left[ \mu_\infty + \sum_{m=1}^{M} \mu_m \exp\!\left( \frac{-t}{\lambda_m} \right) \right] \] 
## 관련 항목

 
- [선형 탄성](01_linear_elastic.html) — 순간 탄성 응답
- [크리프](05_creep.html) — 응력이 일정한 상태에서의 시간 의존 변형
- [재료 데이터(기능)](../../functions_reference/03_material.html) — 점탄성 매개변수 지정 방법


---


<!-- ========================================== -->
<!-- FILE: theory/constitutive/05_creep.md | TITLE: 크리프 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/constitutive/05_creep.html -->

# 크리프

 이 장에서는 FrontISTR가 채택하는 크리프 구성 법칙을 다룬다. 사용 구분과 입력 지정에 대한 자세한 내용은 기능편 `03_material`을 참조한다.

 
## 크리프 현상과 변형률의 가산 분해

 응력이 일정한 상황에서 나타나는 시간 의존적 변위는 「크리프」라고 하는 현상이다.

 앞에서 설명한 점탄성 거동도 일종의 선형 크리프 현상으로 생각할 수 있다. 여기서는 몇 가지 비선형 크리프에 대해 설명한다. 이 현상은 순간적으로 발생하는 변형률에 추가하는 방식으로 구성식을 만드는 방법이 일반적으로 사용되며, 일정한 하중이 지속되는 동안의 변형률을 크리프 변형률 \(\varepsilon^c\)로 한다. 크리프를 고려한 구성식에서는 일반적으로 응력과 전체 크리프 변형률의 함수로 정의되는 크리프 변형률 속도 \(\dot{\varepsilon}^c\)를 사용한다.

 \[ \dot{\varepsilon}^c \equiv \frac{\partial \varepsilon^c}{\partial t} = \beta(\sigma, \varepsilon^c) \] 여기서 순간적으로 발생하는 변형률을 탄성 변형률 \(\varepsilon^e\)라고 하면 전체 변형률은 크리프 변형률을 더한 다음 식으로 나타낸다.

 \[ \varepsilon = \varepsilon^e + \varepsilon^c \] 여기서,

 \[ \varepsilon^e = c^{-1} : \sigma \] 이다(\(c\)는 탄성계수 텐서).

 
## Norton 법칙

 크리프 구성식의 구체적인 형태로 FrontISTR에서는 다음 Norton 모델을 채택한다. 그 구성식에서는 다음과 같이 등가 크리프 변형률 속도 \(\dot{\varepsilon}^{cr}\)가 Mises 응력 \(q\)와 시간 \(t\)의 함수로 표현된다.

 \[ \dot{\varepsilon}^{cr} = A q^n t^m \] 여기서 \(A\), \(m\), \(n\)은 재료 상수이다.

 
## 시간 적분과 응력 갱신

 소성 재료와 마찬가지로 크리프를 나타내는 구성식에 대해서도 수치해석상의 시간 적분 방법을 제시해야 한다. 크리프를 고려한 구성식은,

 \[ \sigma_{n+1} = c : (\varepsilon_{n+1} - \varepsilon_{n+1}^c) \] \[ \varepsilon_{n+1}^c = \varepsilon_n^c + \Delta t \, \beta_{n+\theta} \] 여기서 \(\beta_{n+\theta}\)는,

 \[ \beta_{n+\theta} = (1 - \theta) \beta_n + \theta \beta_{n+1} \] 로 한다. 또한 크리프 변형률 증분 \(\Delta \varepsilon^c\)는 비선형 방정식을 단순화한

 \[ R_{n+1} = \varepsilon_{n+1} - c^{-1} : \sigma_{n+1} - \varepsilon_n^c - \Delta t \, \beta_{n+\theta} = \mathbf{0} \] 으로 한다.

 Newton-Raphson 법의 반복 계산에서는 초기값을 \(\sigma_{n+1} = \sigma_n\) 및 유한요소법에서 구한 변형률 증분으로 두며, 반복해와 증분해는 다음 식으로 한다.

 \[ R_{n+1}^{(k+1)} = \mathbf{0} = R_{n+1}^{(k)} - (c^{-1} + \Delta t \, c_{n+1}^c) \, d\sigma_{n+1}^{(k)} \] 여기서,

 \[ c_{n+1}^c = \left.\frac{\partial \beta}{\partial \sigma}\right|_{n+\theta} = \theta \left.\frac{\partial \beta}{\partial \sigma}\right|_{n+1} \] 으로 한다. 잔차 \(R\)이 \(\mathbf{0}\)이 될 때까지 반복해법을 수행할 때 응력 \(\sigma_{n+1}\)과 접선계수

 \[ c_{n+1}^* = (c^{-1} + \Delta t \, c_{n+1}^c)^{-1} \] 를 사용한다.

 
## 관련 항목

 
- [점탄성](04_viscoelastic.html) — 선형 크리프에 해당하는 점탄성 거동
- [접선 강성 행렬](../solution_methods/01_newton_raphson.html) — 크리프를 포함하는 증분 해석의 접선 강성
- [재료 데이터(기능)](../../functions_reference/03_material.html) — Norton 모델의 매개변수 지정


---


<!-- ========================================== -->
<!-- FILE: theory/constitutive/06_thermal_properties.md | TITLE: 열물성 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/constitutive/06_thermal_properties.html -->

# 열물성

 이 장에서는 FrontISTR의 열전도 해석과 열응력 연성에서 사용하는 열물성 데이터의 정식화를 다룹니다. 선택 방법과 입력 사양에 대한 자세한 내용은 기능편 `03_material`을 참조하십시오.

 
## 열물성 데이터의 종류

 
## 온도 의존 테이블과 보간 규칙

 
## 열전도율의 이방성

 
## 관련 항목

 
- [열전도 방정식](../governing/02_heat_equation.html) — 열물성이 나타나는 지배 방정식
- [비정상 열전도](../solution_methods/05_transient_heat.html) — 시간 이산화와 비선형 반복
- [선형 탄성](01_linear_elastic.html) — 탄성 상수의 온도 의존성
- 기능 참조: [재료 데이터](../../functions_reference/03_material.html) — 열물성 입력 사양


---


<!-- ========================================== -->
<!-- FILE: theory/contact/01_contact.md | TITLE: 접촉 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/contact/01_contact.html -->

# 접촉 해석

 두 물체가 접촉하면 접촉면을 통해 접촉력\(t_c\)이 전달된다. 가상일의 원리 식을 다음과 같이 다시 쓴다.

 \[\begin{equation} \int^{t'}_{^{t'}v}\,^{t'} \sigma : \delta^{t'} A_{(L)} d^{t'}v = \int^{t'}_{^{t'}S_{t}}\,^{t'} \overline{t} \cdot \delta u d^{t'}s+ \int^{t'}_{V} \overline{b} \cdot \delta u d^{t'}v + \int^{t'}_{^{t'} S _{c}}t_{c}[\delta u^{(1)} - u^{(2)}] \label{eq:2.2.70} \end{equation}\] 여기서 \(S_c\)는 접촉 면적을, \(u^{(1)}\)과 \(u^{(2)}\)는 각각 접촉 물체 1과 접촉 물체 2의 변위를 나타낸다.

 접촉 해석에서는 접촉할 가능성이 있는 면을 한 쌍으로 지정한다. 이 면 쌍의 한쪽을 마스터 면, 다른 쪽을 슬레이브 면으로 한다. 이 마스터-슬레이브 해석 방법에서는 접촉 구속 조건을 다음과 같이 가정한다.

 
1. 슬레이브 절점은 마스터 면을 관통하지 않는다.
2. 접촉이 발생하면 슬레이브 절점은 접촉 위치에 놓이며, 이 접촉점을 통해 마스터 면과 슬레이브 면이 서로 접촉력과 마찰력을 전달한다.

 식\(\eqref{eq:2.2.70}\)의 마지막 항을 유한요소법으로 이산화하면 다음 식을 얻는다.

 \[\begin{equation} \int^{t'}_{^{t'}S_{c}} t_c [\delta u^{(1)} - \delta u^{(2)}] \approx \delta UK_c \Delta U + \delta UF_c \label{eq:2.2.71} \end{equation}\] 여기서 \(K_c\)와 \(F_c\)는 각각 접촉 강성 행렬과 접촉력을 나타낸다. 접촉 구속을 고려한 total Lagrange법 및 updated Lagrange법의 유한요소 정식화는 다음과 같다.

 \[\begin{equation} \delta U^T ( ^t_0 K_L + ^t_0 K_{NL} + K_c ) \Delta U = \delta U^T {_0^{t'} F} - \partial U^T {^t_0 Q} + \delta UF_c \label{eq:2.2.72} \end{equation}\] \[\begin{equation} \delta U^T (^t_t K_L + ^t_t K_{NL} + K_c ) \Delta U = \delta U^T {_{t} ^{t'} F} - \partial U^T {^t_t Q}+ \delta UF_c \label{eq:2.2.73} \end{equation}\] 
## 관련 항목

 
- [접선 강성 행렬](../solution_methods/01_newton_raphson.html) — 접촉을 포함하는 비선형 해석의 접선 강성
- [Newton-Raphson법](../solution_methods/01_newton_raphson.html) — 반복 해법
- [접촉 및 임베딩](../../functions_reference/05_contact.html) — 접촉 유형 및 알고리즘 선택지


---


<!-- ========================================== -->
<!-- FILE: theory/discretization/01_shape_functions.md | TITLE: 형상함수와 유한요소 근사 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/discretization/01_shape_functions.html -->

# 형상함수와 유한요소 근사

 [가상일의 원리](../governing/01c_virtual_work.html)의 약형을 계산기로 다루기 위해 물체 영역을 유한개의 요소로 분할하고, 요소 내부의 물질점 좌표·변위·시험함수를 절점값과 형상함수로 보간한다. 형상함수의 공간 미분은 [형상함수의 공간 미분](02_global_derivative.html), 약형의 이산화는 [내력 가상일의 이산화](03_internal_force.html), 요소 종류별 형상함수의 구체적인 형태는 [요소 번호 체계와 형상함수 라이브러리](../../coding_guide/01_element_taxonomy.html) 이후에서 다룬다.

 
## 영역 분할과 적분의 요소별 합

 기준 배치의 영역 \(\Omega_0\) 및 현재 배치의 영역 \(\Omega\)를 각각 요소 \(\Omega^e_0\), \(\Omega^e\)의 합집합으로 근사한다:

 \[ \Omega_0 \approx \Omega_0^h = \bigcup_e \Omega^e_0, \qquad \Omega \approx \Omega^h = \bigcup_e \Omega^e \] (\(e\)는 요소 번호이며, 요소 경계는 인접 요소 사이에서 공유된다). 이에 따라 [가상일의 원리](../governing/01c_virtual_work.html)의 체적·면적 적분은 요소별 적분의 합으로 분해된다:

 \[ \int_{\Omega_0} (\cdot)\, dV \approx \sum_e \int_{\Omega^e_0} (\cdot)\, dV, \qquad \int_{\Gamma_{0t}} (\cdot)\, d\Gamma \approx \sum_e \int_{\Gamma^e_{0t}} (\cdot)\, d\Gamma \] (현재 배치에 대해서도 \(dV \to dv\), \(\Omega^e_0 \to \Omega^e\), \(\Gamma^e_{0t} \to \Gamma^e_t\)로 치환하면 동일하다). 이후 약형의 평가는 요소 단위 적분의 구성으로 귀착된다.

 
## 절점값과 형상함수에 의한 보간(아이소파라메트릭 요소)

 각 요소 \(\Omega^e_0\)에는 \(n_e\)개의 절점이 할당된다. 요소 절점 \(\alpha = 1, \ldots, n_e\)의 기준 배치 좌표와 절점 변위를 \(\boldsymbol{X}^e_\alpha, \boldsymbol{u}^e_\alpha\)라 하고, 이들을 배열한 **요소 절점 벡터** \(\boldsymbol{X}^e = (\boldsymbol{X}^{eT}_1, \ldots, \boldsymbol{X}^{eT}_{n_e})^T\), \(\boldsymbol{u}^e = (\boldsymbol{u}^{eT}_1, \ldots, \boldsymbol{u}^{eT}_{n_e})^T\)는 전체 절점 벡터 \(\boldsymbol{X}^n, \boldsymbol{u}^n\)(\(n_g\)는 전체 절점 수)에서 요소 \(e\)를 구성하는 절점의 성분만 추출한 것이다.

 요소 내부의 로컬 좌표인 **자연좌표** \(\boldsymbol{r}\)를 매개변수로 하는 **형상함수** \(N_\alpha^e(\boldsymbol{r})\)에 의해 요소 내부의 물질좌표·변위·시험함수를 공통 형상함수로 보간한다(아이소파라메트릭 요소·Galerkin 법):

 \[ \boldsymbol{X} = \sum_{\alpha=1}^{n_e} N_\alpha^e(\boldsymbol{r})\, \boldsymbol{X}^e_\alpha, \qquad \boldsymbol{u} = \sum_{\alpha=1}^{n_e} N_\alpha^e(\boldsymbol{r})\, \boldsymbol{u}^e_\alpha, \qquad \delta\boldsymbol{u} = \sum_{\alpha=1}^{n_e} N_\alpha^e(\boldsymbol{r})\, \delta\boldsymbol{u}^e_\alpha. \] 형상함수는 다음 두 성질을 만족하도록 구성하고, 자연좌표에서 물질좌표로의 사상 \(\boldsymbol{r}\mapsto\boldsymbol{X}\)가 요소 내부에서 일대일이 되도록 요소 형상을 선택한다:

 \[ \sum_{\alpha=1}^{n_e} N_\alpha^e(\boldsymbol{r}) = 1, \qquad N_\beta^e(\boldsymbol{r}_\alpha) = \delta_{\alpha\beta} \] (\(\boldsymbol{r}_\alpha\)는 절점 \(\alpha\)에 대응하는 자연좌표의 점이고, \(\delta_{\alpha\beta}\)는 Kronecker 델타이다). 첫 번째 식은 강체 병진의 재현성을, 두 번째 식은 보간값이 절점에서 절점값과 일치함을 보장한다. 요소 종류별 \(n_e\), \(N_\alpha^e\)의 구체적인 형태는 [요소 번호 체계와 형상함수 라이브러리](../../coding_guide/01_element_taxonomy.html) 이후에서 제시한다. 표기의 번잡함을 피하기 위해 요소 종류에 대한 의존성은 요소별 위첨자 \(e\)로 대표한다.

 위의 보간 규칙에 따라 약형의 피적분함수는 요소 절점값 \(\boldsymbol{u}^e, \delta\boldsymbol{u}^e\)와 \(N_\alpha^e\)만으로 표현할 수 있다. 한편 변형률은 보간된 변위와 [변형률-변위 관계](../governing/01a_kinematics.html)로부터, 응력은 그 변형률과 [재료 구성 법칙](../constitutive/01_linear_elastic.html)으로부터 유도되는 양이며 절점값에서 직접 보간되는 것이 아니다. 이들은 요소 내부의 적분점에서 평가된다([수치 적분](05_numerical_integration.html)).

 
## 전체 절점 벡터의 배열 규칙

 절점에 할당되는 물리량은 전체 절점 벡터에서 **절점 번호 → 자유도** 순으로 오름차순 배열한다. 절점 \(\alpha\)에서 자유도 \(i\) 성분을 \(u_{i\alpha}\)라 하면, 3차원(\(i=1,2,3\))과 2차원(\(i=1,2\))에서 각각

 \[ \boldsymbol{u}^n = (u_{11}, u_{21}, u_{31},\ u_{12}, u_{22}, u_{32},\ \ldots,\ u_{1 n_g}, u_{2 n_g}, u_{3 n_g})^T, \] \[ \boldsymbol{u}^n = (u_{11}, u_{21},\ u_{12}, u_{22},\ \ldots,\ u_{1 n_g}, u_{2 n_g})^T \] 가 된다. 좌표 \(\boldsymbol{X}^n\) 및 시험함수 \(\delta\boldsymbol{u}^n\)도 같은 배열을 따른다. 이후 행렬·벡터 형식의 식 전개는 3차원을 대표로 기술한다.

 
## 관련 항목

 
- [가상일의 원리](../governing/01c_virtual_work.html) — 이산화 대상이 되는 약형
- [형상함수의 공간 미분](02_global_derivative.html) — Jacobian·B 행렬의 준비
- [요소 번호 체계와 형상함수 라이브러리](../../coding_guide/01_element_taxonomy.html) — 요소 유형별 형상함수


---


<!-- ========================================== -->
<!-- FILE: theory/discretization/02_global_derivative.md | TITLE: 형상함수의 공간 미분 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/discretization/02_global_derivative.html -->

# 형상함수의 공간 미분

 [형상함수와 유한요소 근사](01_shape_functions.html)에서 도입한 형상함수 \(N_\alpha^e(\boldsymbol{r})\)는 요소의 로컬 좌표인 자연좌표 \(\boldsymbol{r}\)의 함수로 주어진다. 한편 약형의 피적분 함수에는 변형률-변위 관계를 통해 물리 좌표(기준 배치 \(\boldsymbol{X}\) 또는 현재 배치 \(\boldsymbol{x}\))에 대한 편미분 \(\partial N_\alpha^e/\partial \boldsymbol{X}\) 또는 \(\partial N_\alpha^e/\partial \boldsymbol{x}\)가 나타난다. 이 장에서는 이러한 **형상함수의 공간 미분**을 요소 절점 좌표와 자연좌표 미분으로부터 계산하는 절차를 정리한다.

 
## 연쇄법칙에 의한 공간 미분 표현

 보간식 \(\boldsymbol{X} = \sum_\alpha N_\alpha^e(\boldsymbol{r})\,\boldsymbol{X}^e_\alpha\)([이전 장](01_shape_functions.html))에 연쇄법칙을 적용하면, 자연좌표에서 물리 좌표로의 사상의 **Jacobian 행렬** \(J_{ij} = \partial X_i/\partial r_j\)를 사용하여

 \[ \frac{\partial \boldsymbol{N}^e}{\partial \boldsymbol{X}} = \frac{\partial \boldsymbol{N}^e}{\partial \boldsymbol{r}}\,\left(\frac{\partial \boldsymbol{X}}{\partial \boldsymbol{r}}\right)^{-1} \] 을 얻는다. 여기서 \(\partial \boldsymbol{N}^e/\partial \boldsymbol{X}\) 및 \(\partial \boldsymbol{N}^e/\partial \boldsymbol{r}\)은 행 \(\alpha\)가 절점 \(\alpha\)의 편미분 벡터를 나타내는 \(n_e \times 3\) 행렬이다.

 
## Jacobian 행렬의 구성

 보간식을 \(\boldsymbol{r}\)로 편미분하면

 \[ \frac{\partial \boldsymbol{X}}{\partial \boldsymbol{r}} = \left[\boldsymbol{X}^e_1, \ldots, \boldsymbol{X}^e_{n_e}\right]\frac{\partial \boldsymbol{N}^e}{\partial \boldsymbol{r}} \] 이 되며, 요소 절점 좌표 \(\boldsymbol{X}^e_\alpha\)와 자연좌표 미분 \(\partial \boldsymbol{N}^e/\partial \boldsymbol{r}\)로부터 직접 구성할 수 있다. 자연좌표 미분은 형상함수의 함수 형태로 결정되며 요소 종류별로 미리 구현된다.

 Jacobian 행렬의 행렬식 \(\det(\partial \boldsymbol{X}/\partial \boldsymbol{r})\)은 [수치 적분](05_numerical_integration.html)에서 체적 요소의 변환 \(dV = \det(\partial \boldsymbol{X}/\partial \boldsymbol{r})\,d\boldsymbol{r}\)에 사용한다. 행렬식이 \(0\)이 되는 경우 요소가 퇴화되어 계산이 실패한다.

 
## 계산 흐름

 이상의 유도에서 공간 미분 \(\partial \boldsymbol{N}^e/\partial \boldsymbol{X}\)의 계산은 다음 4단계로 나눌 수 있다.

 
1. **자연좌표 미분 계산**: 요소 종류와 평가점의 자연좌표 \(\boldsymbol{r}\)를 입력으로 하여 자연좌표 미분 행렬 \(\partial \boldsymbol{N}^e/\partial \boldsymbol{r}\)을 계산한다. 요소 종류마다 형상함수의 함수 형태가 다르므로 요소 종류에 따라 분기한다.
2. **Jacobian 행렬 계산**: 요소 절점 좌표 \(\boldsymbol{X}^e\)와 \(\partial \boldsymbol{N}^e/\partial \boldsymbol{r}\)로부터 앞 절의 식에 따라 Jacobian 행렬 \(\partial \boldsymbol{X}/\partial \boldsymbol{r}\)을 조립한다.
3. **역행렬 및 행렬식 계산**: Jacobian 행렬의 역행렬 \((\partial \boldsymbol{X}/\partial \boldsymbol{r})^{-1}\)과 행렬식을 계산한다. 행렬식은 수치 적분의 가중치에 사용한다.
4. **공간 미분 계산**: \(\partial \boldsymbol{N}^e/\partial \boldsymbol{r}\)과 \((\partial \boldsymbol{X}/\partial \boldsymbol{r})^{-1}\)의 곱으로 공간 미분 \(\partial \boldsymbol{N}^e/\partial \boldsymbol{X}\)를 얻는다.

 이 4단계는 요소 종류와 공간 차원(2차원 또는 3차원)에 따라 분기하지만 전체 흐름은 공통이다.

 
## 기준 배치와 현재 배치에서의 공통화

 위 절차는 \(\boldsymbol{X}^e\)를 현재 배치의 절점 좌표 \(\boldsymbol{x}^e_\alpha = \boldsymbol{X}^e_\alpha + \boldsymbol{u}^e_\alpha\)로 바꾸기만 하면 현재 배치의 공간 미분 \(\partial \boldsymbol{N}^e/\partial \boldsymbol{x}\)에도 적용할 수 있다. 따라서 Total Lagrange법과 Updated Lagrange법에서 입력 절점 좌표만 전환하여 동일한 절차를 공용할 수 있다.

 
## 관련 항목

 
- [형상함수와 유한요소 근사](01_shape_functions.html) — 형상함수와 요소 절점 좌표의 정의
- [내력 가상일의 이산화](03_internal_force.html) — 공간 미분으로부터 B 행렬 구성
- [수치 적분](05_numerical_integration.html) — Gauss 적분에서 Jacobian 행렬식 사용


---


<!-- ========================================== -->
<!-- FILE: theory/discretization/03_internal_force.md | TITLE: 내력 가상일의 이산화 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/discretization/03_internal_force.html -->

# 내력 가상일의 이산화

 [증분 해석의 틀](../governing/01d_incremental_analysis.html)에서 제시한 시각 \(t + \Delta t\)의 가상일 방정식은 기준 배치의 선택에 따라 Updated Lagrange 형식과 Total Lagrange 형식의 두 형태를 취합니다. 이 장에서는 [형상 함수와 유한요소 근사](01_shape_functions.html) 및 [형상 함수의 공간 미분](02_global_derivative.html)에서 도입한 유한요소 근사를 사용하여 두 형식의 내력의 가상일을 공간적으로 이산화하고, 요소 내력 벡터 \(\boldsymbol{q}^e\) (UL)와 \(\boldsymbol{Q}^e\) (TL)를 구합니다.

 요소 \(e\)에 대해 구성 절점을 \(\alpha = 1, \ldots, n_e\), 변위를 \(\boldsymbol{u}^e_\alpha\)라 하고, 요소 절점 변위 벡터를 \(\boldsymbol{u}^e = (\boldsymbol{u}^{eT}_1, \ldots, \boldsymbol{u}^{eT}_{n_e})^T\)의 순서로 배열합니다. 가상 변위 \(\delta \boldsymbol{u}^e\)도 같은 순서로 정의합니다. 요소 내부의 변위는 형상 함수에 의해 \(\boldsymbol{u} = \sum_\alpha N_\alpha^e \boldsymbol{u}^e_\alpha\)로 보간됩니다.

 
## Updated Lagrange 형식의 내력 가상일

 Updated Lagrange 형식에서는 시각 \(t\)의 현재 배치 \({}^{t}\Omega\)를 기준 배치로 사용하고, 내력의 가상일을 Cauchy 응력 \(\boldsymbol{\sigma}\)와 Almansi 변형률의 선형부 \(\boldsymbol{A}_{(L)}\)를 사용하여 다음과 같이 나타냅니다.

 \[ \delta W^{\mathrm{int}} = \sum_e \int_{\Omega^e} \boldsymbol{\sigma} : \delta \boldsymbol{A}_{(L)}\, dv \] \(\delta \boldsymbol{A}_{(L)}\)의 각 성분은 현재 배치 좌표 \(\boldsymbol{x}\)를 사용하여 형상 함수 미분 \(\partial N_\alpha^e/\partial x_i\)와 절점 가상 변위 \(\delta u^e_{i\alpha}\)의 선형 결합으로 나타낼 수 있습니다. 이를 Voigt 표기로 정리하면 다음과 같습니다.

 \[ \delta \boldsymbol{A}_{(L)} = \boldsymbol{B}_L\, \delta \boldsymbol{u}^e, \qquad \boldsymbol{B}_L = [\boldsymbol{B}_{L1}, \ldots, \boldsymbol{B}_{Ln_e}] \] 절점 블록 \(\boldsymbol{B}_{L\alpha}\)는 \(\partial N_\alpha^e/\partial x_i\)를 Voigt 규약에 따라 \(6 \times 3\) 행렬로 배열하여 구성하며, \(\boldsymbol{B}_L\)은 UL 형식의 변형률-변위 행렬입니다. 이를 내력의 가상일에 대입하고 \(\delta \boldsymbol{u}^e\)를 묶어내면 다음과 같습니다.

 \[ \delta W^{\mathrm{int}} = \sum_e \delta \boldsymbol{u}^{eT}\, \boldsymbol{q}^e, \qquad \boldsymbol{q}^e = \int_{\Omega^e} \boldsymbol{B}_L^T \boldsymbol{\sigma}\, dv \] \(\boldsymbol{q}^e\)에서 절점 블록 \(\boldsymbol{q}^e_\alpha = \int_{\Omega^e} \boldsymbol{B}_{L\alpha}^T \boldsymbol{\sigma}\, dv\)는 요소 \(\Omega^e\)가 구성 절점 \(\alpha\)에 작용시키는 내력입니다.

 
## Total Lagrange 형식의 내력 가상일

 Total Lagrange 형식에서는 초기 배치 \(\Omega_0\)를 기준 배치로 사용하고, 내력의 가상일을 제2 Piola-Kirchhoff 응력 \(\boldsymbol{S}\)와 Green-Lagrange 변형률 \(\boldsymbol{E}\)를 사용하여 다음과 같이 나타냅니다.

 \[ \delta W^{\mathrm{int}} = \sum_e \int_{\Omega^e_0} \boldsymbol{S} : \delta \boldsymbol{E}\, dV \] 변분 \(\delta \boldsymbol{E}\)는 가상 변위에 선형인 항과 현재 변위 구배 \(\partial u_k/\partial X_j\)와의 곱을 포함하는 비선형 항으로 분리됩니다.

 \[ \delta E_{(L)ij} = \frac{1}{2}\left( \frac{\partial \delta u_i}{\partial X_j} + \frac{\partial \delta u_j}{\partial X_i} \right), \quad \delta E_{(NL)ij} = \frac{1}{2}\left( \frac{\partial \delta u_k}{\partial X_i}\, \frac{\partial u_k}{\partial X_j} + \frac{\partial u_k}{\partial X_i}\, \frac{\partial \delta u_k}{\partial X_j} \right). \] 선형항은 UL 형식과 동일한 배열 규칙을 \(\partial N_\alpha^e/\partial X_i\)에 적용하여 절점 블록 \(\boldsymbol{B}_{L\alpha}\)로 나타낼 수 있습니다.

 \[ \delta \boldsymbol{E}_{(L)} = \boldsymbol{B}_L\, \delta \boldsymbol{u}^e \] 기준 배치가 다르기 때문에 구성 미분만 \(\partial N_\alpha^e/\partial x_i\)에서 \(\partial N_\alpha^e/\partial X_i\)로 바뀌며, 기호는 UL 형식과 동일하게 사용합니다. 비선형항은 현재 변위 구배 \(\partial u_k/\partial X_j\)와 \(\partial N_\alpha^e/\partial X_i\)의 곱을 사용하고, 이를 Voigt 규약에 따라 배열하여 절점 블록 \(\boldsymbol{B}_{NL\alpha}\)를 구성합니다.

 \[ \delta \boldsymbol{E}_{(NL)} = \boldsymbol{B}_{NL}\, \delta \boldsymbol{u}^e, \qquad \boldsymbol{B}_{NL} = [\boldsymbol{B}_{NL1}, \ldots, \boldsymbol{B}_{NLn_e}] \] 따라서 \(\delta \boldsymbol{E} = (\boldsymbol{B}_L + \boldsymbol{B}_{NL})\, \delta \boldsymbol{u}^e\)이며, \(\boldsymbol{B}_L + \boldsymbol{B}_{NL}\)은 TL 형식의 변형률-변위 행렬입니다. 이를 내력의 가상일에 대입하면 다음과 같습니다.

 \[ \delta W^{\mathrm{int}} = \sum_e \delta \boldsymbol{u}^{eT}\, \boldsymbol{Q}^e, \qquad \boldsymbol{Q}^e = \int_{\Omega^e_0} (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^T \boldsymbol{S}\, dV \] 절점 블록 \(\boldsymbol{Q}^e_\alpha\)는 요소 \(\Omega^e_0\)가 구성 절점 \(\alpha\)에 작용시키는 내력입니다.

 
## UL/TL의 대응 관계와 계산 흐름

 Updated Lagrange 형식과 Total Lagrange 형식의 요소 내력 벡터는 다음과 같이 대응합니다.

 
| 항목 | Updated Lagrange 형식 | Total Lagrange 형식 |
| --- | --- | --- |
| 기준 배치 | 현재 배치 \({}^{t}\Omega^e\) | 초기 배치 \(\Omega^e_0\) |
| 응력 텐서 | Cauchy 응력 \(\boldsymbol{\sigma}\) | 제2 PK 응력 \(\boldsymbol{S}\) |
| 변형률 변분 | \(\delta \boldsymbol{A}_{(L)}\) | \(\delta \boldsymbol{E} = \delta \boldsymbol{E}_{(L)} + \delta \boldsymbol{E}_{(NL)}\) |
| B 행렬 | \(\boldsymbol{B}_L\) (\(\partial N/\partial x\)) | \(\boldsymbol{B}_L + \boldsymbol{B}_{NL}\) (\(\partial N/\partial X\), \(\partial u/\partial X\)) |
| 요소 내력 | \(\boldsymbol{q}^e = \int_{\Omega^e} \boldsymbol{B}_L^T \boldsymbol{\sigma}\, dv\) | \(\boldsymbol{Q}^e = \int_{\Omega^e_0} (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^T \boldsymbol{S}\, dV\) |

 두 형식 모두 형상 함수의 공간 미분으로부터 \(\boldsymbol{B}_L\)을 구성하고, TL 형식에서는 현재 변위 구배로부터 \(\boldsymbol{B}_{NL}\)을 구성하여 더한 뒤, 구성 법칙에 따라 응력(\(\boldsymbol{\sigma}\) 또는 \(\boldsymbol{S}\))을 갱신하고, 적분점에서 요소 영역에 걸쳐 \(\boldsymbol{B}^T \boldsymbol{\sigma}\) 또는 \((\boldsymbol{B}_L+\boldsymbol{B}_{NL})^T \boldsymbol{S}\)를 수치 적분([수치 적분](05_numerical_integration.html))하는 동일한 절차로 처리됩니다. 기준 배치(절점 좌표와 \(\boldsymbol{B}\) 행렬의 구성)의 전환과 응력 텐서의 교체를 제외하면 처리가 공통이므로, FrontISTR에서는 두 형식의 내력 계산을 공통 서브루틴으로 구현합니다. 요소 내력 벡터 \(\boldsymbol{q}^e\)와 \(\boldsymbol{Q}^e\)를 전역 내력 벡터로 조립하는 과정은 [외력의 가상일과 전역 방정식의 조립](04_external_force_and_assembly.html)에서 다룹니다.

 
## 관련 항목

 
- [가상일의 원리](../governing/01c_virtual_work.html) — 출발 방정식
- [증분 해석의 틀](../governing/01d_incremental_analysis.html) — TL/UL 분기의 정의
- [형상 함수의 공간 미분](02_global_derivative.html) — B 행렬의 구성 요소
- [외력의 가상일과 전역 방정식의 조립](04_external_force_and_assembly.html) — 전역 내력 벡터로의 조립
- [접선 강성 행렬](../solution_methods/01_newton_raphson.html) — 내력의 시간 미분에 의한 선형화
- [물리량 기호 목록](../notation/02_symbol_list.html)


---


<!-- ========================================== -->
<!-- FILE: theory/discretization/04_external_force_and_assembly.md | TITLE: 외력의 가상일과 전체 방정식의 조립 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/discretization/04_external_force_and_assembly.html -->

# 외력의 가상일과 전체 방정식의 조립

 [내력 가상일의 이산화](03_internal_force.html)에서는 약형의 좌변을 요소 내력 벡터 \(\boldsymbol{q}^e\)(UL 법) 또는 \(\boldsymbol{Q}^e\)(TL 법)로 정리했다. 이 장에서는 외력의 가상일로부터 요소 절점 외력 벡터 \(\boldsymbol{F}^e\)를 도입하고, 요소 절점 물리량을 전체 절점 번호로 재배열하여 집계하는 **조립** 연산을 거쳐 FrontISTR의 비선형 구조 해석이 풀어야 하는 절점 변위에 관한 비선형 방정식계에 도달한다.

 
## 외력 가상일의 요소 분해

 [가상일의 원리](../governing/01c_virtual_work.html)의 우변은 체적력(물체력)과 역학적 경계에서의 규정 표면력으로 이루어진 외력의 가상일로서 요소별로 분해할 수 있다. [형상 함수와 유한요소 근사](01_shape_functions.html)에서 도입한 변위 보간을 행렬 형식으로 쓰기 위해, 절점 \(\alpha\)의 형상 함수 \(N_\alpha^e\)를 대각에 배치한 \(d \times d\) 블록 \(\boldsymbol{N}_\alpha\)와 이를 가로로 나열한 \(\boldsymbol{N} = [\boldsymbol{N}_1, \ldots, \boldsymbol{N}_{n_e}]\)를 사용하여 \(\delta\boldsymbol{u} = \boldsymbol{N}\, \delta\boldsymbol{u}^e\)로 둔다. 이를 기준 배치로 쓴 외력의 가상일에 대입하면

 \[ \delta W^{\mathrm{ext}} = \sum_e \delta\boldsymbol{u}^{eT} \boldsymbol{F}^e, \qquad \boldsymbol{F}^e_\alpha = \int_{\Omega^e_0} \boldsymbol{N}_\alpha^T \rho_0 \boldsymbol{g}\, dV + \int_{\Gamma^e_{0t}} \boldsymbol{N}_\alpha^T \bar{\boldsymbol{t}}_0\, d\Gamma_0 \] 를 얻는다. 여기서 **요소 절점 외력 벡터**를 \(\boldsymbol{F}^e = (\boldsymbol{F}^{eT}_1, \ldots, \boldsymbol{F}^{eT}_{n_e})^T\)로 배열했다. 이에 따라 외력의 가상일은 내력 측과 동일한 「요소 절점 벡터 × 시험 함수」의 형태로 정리된다(현재 배치로 써도 \(dV \to dv\), \(\rho_0 \to \rho\), \(\bar{\boldsymbol{t}}_0 \to \bar{\boldsymbol{t}}\)로 치환하면 같은 형태가 된다).

 
## 요소 절점 물리량의 조립

 요소별로 얻은 절점 물리량 \(\boldsymbol{Q}^e_\alpha, \boldsymbol{F}^e_\alpha\)를 전체 절점 번호 순으로 배열한 전체 벡터에 집계한다. 요소 \(\Omega^e\)의 요소 내 절점 번호 \(\alpha\)에 대응하는 전체 절점 번호를

 \[ \mathrm{gdx}(e, \alpha) = i_g \] 로 나타내면 요소 절점 물리량은 전체 절점 물리량의 대응 성분과 일치한다(\(\boldsymbol{u}^e_\alpha = \boldsymbol{u}_{i_g}\) 등). 일반적으로 절점 \(i_g\)는 여러 요소가 공유하므로, 전체 절점 번호가 \(i_g\)가 되는 \((e, \alpha)\)의 조합을 모은 집합

 \[ \mathcal{E}(i_g) = \{ (e, \alpha) \mid \mathrm{gdx}(e, \alpha) = i_g \} \] 을 사용하여 \(\sum_e \sum_\alpha = \sum_{i_g} \sum_{(e,\alpha) \in \mathcal{E}(i_g)}\)로 합을 다시 쓰면, 전체 절점 \(n_g\)에 걸친 **절점 내력** 및 **전체 내력 벡터**

 \[ \boldsymbol{Q}_{i_g} = \sum_{(e,\alpha) \in \mathcal{E}(i_g)} \boldsymbol{Q}^e_\alpha, \qquad \boldsymbol{Q} = (\boldsymbol{Q}^T_1, \ldots, \boldsymbol{Q}^T_{n_g})^T \] 를 얻는다. \(\boldsymbol{Q}_{i_g}\)는 절점 \(i_g\)에 작용하는 요소 절점 내력의 합력에 해당하며, 외력이 작용하지 않고 평형을 이루고 있다면 \(\boldsymbol{0}\)이 되는 양이다. UL 법에서도 같은 절차로 \(\boldsymbol{q}_{i_g}, \boldsymbol{q}\)를 얻으며 값으로는 \(\boldsymbol{q} = \boldsymbol{Q}\)이므로, 이하에서는 구분이 필요한 곳을 제외하고 \(\boldsymbol{Q}\)로 통일한다. **전체 외력 벡터** \(\boldsymbol{F}\)도 같은 집계로 얻어진다.

 구현에서는 집합 \(\mathcal{E}(i_g)\)를 명시적으로 만들지 않고 요소 루프 안에서 대응 성분에 더한다.

 
```text
전체 내력 벡터 Q를 0으로 초기화: Q_{i_g} = 0  (i_g = 1, ..., n_g)
for e = 1 to (요소 수)
    for α = 1 to n_e
        i_g = gdx(e, α)
        Q_{i_g} += Q^e_α
    end for
end for
```

 전체 외력 벡터 \(\boldsymbol{F}\)도 같은 절차로 구성된다. 전체 절점 번호로 번호가 매겨진 벡터·행렬에 요소 절점 물리량을 더하여 저장하는 이 연산을 **조립**(assemble)이라고 한다. 두 절점 번호에 대한 2계 텐서(강성 행렬 등)에 대해서는 집합 \(\mathcal{E}^2(i_g, i_h) = \{ (e, \alpha, \beta) \mid \mathrm{gdx}(e, \alpha) = i_g\ \mathrm{and}\ \mathrm{gdx}(e, \beta) = i_h \}\)를 사용하여 같은 종류의 조립 연산을 얻을 수 있다(구체적인 구성은 [접선 강성 행렬](../solution_methods/01_newton_raphson.html) 참조).

 
## 풀어야 할 비선형 방정식

 내력·외력의 조립 결과를 [가상일의 원리](../governing/01c_virtual_work.html)에 대입하고, 기하학적 경계 조건을 만족하는 임의의 시험 함수 \(\delta\boldsymbol{u}^n\)에 대해 성립한다는 점에서

 \[ \boldsymbol{Q}(\boldsymbol{u}^n) - \boldsymbol{F}(\boldsymbol{u}^n) = \boldsymbol{0} \] 를 얻는다. 증분 해석([증분 해석의 틀](../governing/01d_incremental_analysis.html))의 문맥에서 시간 첨자 \(_{n+1}\)를 다시 사용하고 전체 절점 벡터를 나타내는 위첨자 \(^n\)을 생략하면, 풀어야 할 방정식은

 \[ \boldsymbol{Q}(\boldsymbol{u}_{n+1}) - \boldsymbol{F}(\boldsymbol{u}_{n+1}) = \boldsymbol{0} \] 가 된다. 시각 \(t_{n+1}\)에서의 절점 변위 \(\boldsymbol{u}_{n+1}\)를 구하는 이산화된 경계값 문제는 이 변위에 관한 비선형 방정식을 기하학적 경계 조건과 함께 푸는 문제로 귀착된다. 방정식의 선형화와 접선 강성 행렬의 구성은 [접선 강성 행렬](../solution_methods/01_newton_raphson.html), 반복 해법은 [Newton-Raphson 법](../solution_methods/01_newton_raphson.html)에 맡긴다.

 
## 관련 항목

 
- [가상일의 원리](../governing/01c_virtual_work.html) — 약형의 출발점
- [증분 해석의 틀](../governing/01d_incremental_analysis.html) — 시간 첨자와 기준 배치의 선택
- [형상 함수와 유한요소 근사](01_shape_functions.html) — 변위·시험 함수의 보간과 요소 절점 벡터
- [내력 가상일의 이산화](03_internal_force.html) — 요소 내력 벡터 \(\boldsymbol{q}^e, \boldsymbol{Q}^e\)의 유도
- [접선 강성 행렬](../solution_methods/01_newton_raphson.html) — 강성 행렬의 조립(동일 절차)
- [Newton-Raphson 법](../solution_methods/01_newton_raphson.html) — 비선형 방정식의 반복 해법
- [물리량 기호 목록](../notation/02_symbol_list.html)


---


<!-- ========================================== -->
<!-- FILE: theory/discretization/05_numerical_integration.md | TITLE: 수치 적분 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/discretization/05_numerical_integration.html -->

# 수치 적분

 [내력 가상일의 이산화](03_internal_force.html) 및 [외력 가상일과 전체 방정식의 조립](04_external_force_and_assembly.html)에서 얻은 요소 내력 벡터 \(\boldsymbol{q}^e, \boldsymbol{Q}^e\)와 요소 강성 행렬 \(\boldsymbol{K}^e\)는 요소 영역 \(\Omega^e\) 또는 \(\Omega^e_0\) 위의 적분 형태를 취한다. FrontISTR에서는 이를 Gauss 적분 공식으로 수치적으로 평가한다.

 
## Gauss 적분 공식과 변수 변환

 Gauss 적분 공식은 표준 영역 \(\Xi\) 위의 적분을 적분점 \(\boldsymbol{\xi}_i \in \Xi\)에서의 피적분함수값과 가중치 \(w_i\)의 선형합으로 근사한다. 요소 영역 \(\Omega^e\)에 적용할 때에는 사상 \(\boldsymbol{x}: \Xi \to \Omega^e\)에 의한 변수 변환이 수반되며,

 \[ \int_{\Omega^e} f(\boldsymbol{x})\, dv \approx \sum_{i=1}^{n_q} w_i\, f(\boldsymbol{x}(\boldsymbol{\xi}_i))\, J_{\xi_i}, \qquad J_{\xi_i} = \left.\det\!\left(\frac{\partial \boldsymbol{x}}{\partial \boldsymbol{\xi}}\right)\right|_{\boldsymbol{\xi}_i} \] 로 쓸 수 있다. \(n_q\)는 적분점 수, \(J_{\xi_i}\)는 변환 Jacobian 행렬식이다. 표준 영역 \(\Xi\)는 요소 종류마다 정해지며(육면체는 \([-1,1]^3\), 삼각형·사면체·쐐기형은 각각의 기준 형상), 적분점 \(\boldsymbol{\xi}_i\)와 가중치 \(w_i\)는 수치표로 주어진다. 면적분도 요소 면을 2차원 표준 영역에서 사상함으로써 같은 형식으로 취급한다.

 FrontISTR가 채택하는 대표적인 적분점 수를 아래에 나타낸다(요소형과의 대응은 [요소 번호 체계와 형상함수 라이브러리](../../coding_guide/01_element_taxonomy.html)를 참조).

 
| 요소 유형 | 적분 공식 | 적분점 수 |
| --- | --- | --- |
| 4절점 사면체(`tet4n`) | 1점 공식 | 1 |
| 10절점 사면체(`tet10n`) | 4점 공식 | 4 |
| 6절점 삼각기둥(`prism6n`) | 2점 공식 | 2 |
| 15절점 삼각기둥(`prism15n`) | 9점 공식 | 9 |
| 8절점 육면체(`hex8n`) | 2×2×2 Gauss-Legendre | 8 |
| 20절점 육면체(`hex20n`) | 3×3×3 Gauss-Legendre | 27 |
| 4절점 사각형(`quad4n`) | 2×2 Gauss-Legendre | 4 |
| 8절점 사각형(`quad8n`) | 3×3 Gauss-Legendre | 9 |
| 3절점 삼각형(`tri3n`) | 1점 공식 | 1 |
| 6절점 삼각형(`tri6n`) | 3점 공식 | 3 |

 육면체·사각형·선분 요소에서는 각 축 방향 Gauss-Legendre 공식의 텐서곱을 취한다. 삼각형·사면체·삼각기둥은 단체 형상에 적합한 전용 공식(삼각형 자체 위에서 다항식을 정확히 적분하는 점 배치)을 사용한다.

 
## 요소 적분에의 적용

 요소 적분에서는 [자연좌표](01_shape_functions.html) \(\boldsymbol{r}\)을 표준 영역의 좌표로 하고(\(\boldsymbol{r} = \boldsymbol{\xi}\)), 물리좌표로의 사상은 형상함수에 의한 절점 좌표 보간으로 주어진다. 참조 배치의 선택([증분 해석의 틀](../governing/01d_incremental_analysis.html))에 따라 다음과 같이 구분하여 사용한다.

 **Total Lagrange 법**(기준 배치 \(\Omega^e_0\)에서 적분): 사상과 Jacobian을

 \[ \boldsymbol{X}^e(\boldsymbol{r}) = \sum_{\alpha} N_\alpha^e(\boldsymbol{r})\, \boldsymbol{X}^e_\alpha, \qquad J_{r_i} = \left.\det\!\left(\frac{\partial \boldsymbol{X}}{\partial \boldsymbol{r}}\right)\right|_{\boldsymbol{r}_i} \] 로 하고 요소 내력·강성 행렬을

 \[ \boldsymbol{Q}^e \approx \sum_{i=1}^{n_q} w_i\, (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^T\, \boldsymbol{S}\, J_{r_i}, \qquad \boldsymbol{K}^e \approx \sum_{i=1}^{n_q} w_i\, \boldsymbol{K}^e_{x}\, J_{r_i} \] 로 근사한다. \(\boldsymbol{B}_L, \boldsymbol{B}_{NL}, \boldsymbol{S}, \boldsymbol{K}^e_{x}\)는 모두 적분점 \(\boldsymbol{r}_i\)에서 평가한다.

 **Updated Lagrange 법**(현재 배치 \(\Omega^e\)에서 적분): 사상과 Jacobian을

 \[ \boldsymbol{x}^e(\boldsymbol{r}) = \sum_{\alpha} N_\alpha^e(\boldsymbol{r})\, \boldsymbol{x}^e_\alpha = \sum_{\alpha} N_\alpha^e(\boldsymbol{r})\, (\boldsymbol{X}^e_\alpha + \boldsymbol{u}^e_\alpha), \qquad J_{r_i} = \left.\det\!\left(\frac{\partial \boldsymbol{x}}{\partial \boldsymbol{r}}\right)\right|_{\boldsymbol{r}_i} \] 로 하고,

 \[ \boldsymbol{q}^e \approx \sum_{i=1}^{n_q} w_i\, \boldsymbol{B}_L^T\, \boldsymbol{\sigma}\, J_{r_i}, \qquad \boldsymbol{K}^e \approx \sum_{i=1}^{n_q} w_i\, \boldsymbol{K}^e_{x}\, J_{r_i} \] 로 근사한다.

 두 정식화의 차이는 사상 입력의 절점 좌표가 \(\boldsymbol{X}^e_\alpha\)인지 \(\boldsymbol{x}^e_\alpha\)인지라는 한 가지 점뿐이며, 적분점·가중치·적분점 루프의 구조는 공통이다.

 
## 완전 적분과 차수 저감 적분

 피적분함수의 다항식 차수를 정확히 적분할 수 있는 적분점 수를 채택하는 적분을 **완전 적분**, 그보다 한 단계 적은 적분점 수를 채택하는 적분을 **차수 저감 적분**이라고 한다. 차수 저감 적분은 전단·체적 locking 완화에 사용되지만 hourglass 모드와 같은 거짓 변형 모드에 대한 대처가 수반된다. 요소 종류별 적분점 수와 완전/차수 저감의 선택은 [요소 번호 체계와 형상함수 라이브러리](../../coding_guide/01_element_taxonomy.html) 이후 및 [고성능 요소 정식화](09_element_advanced.html)에서 다룬다.

 
## 관련 항목

 
- [형상함수의 공간 미분](02_global_derivative.html) — Jacobian 계산
- [내력 가상일의 이산화](03_internal_force.html) — 수치 적분 대상
- [고성능 요소 정식화](09_element_advanced.html) — 차수 저감 적분의 이용


---


<!-- ========================================== -->
<!-- FILE: theory/discretization/08_element_solid.md | TITLE: 3차원 솔리드 요소의 형상 함수 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/discretization/08_element_solid.html -->

# 3차원 솔리드 요소의 형상 함수

 이 장에서는 FrontISTR가 연속체 해석에 사용하는 3차원 솔리드 요소에 대해 [형상 함수와 유한요소 근사](01_shape_functions.html)에서 도입한 형상 함수 \(N_\alpha^e(\boldsymbol{r})\)의 자연 좌표계와 구체적인 형식을 정리합니다. 대상인 여섯 요소 유형은 1차/2차 사면체(341/342), 1차/2차 삼각기둥(351/352), 1차/2차 육면체(361/362)입니다. HEC-MW 메시 형식의 절점 순서와 면 번호는 기능 레퍼런스의 [요소 라이브러리](../../functions_reference/02_element.html)를 참조하십시오.

 이러한 형상 함수와 Jacobian 행렬식을 사용하여 요소 적분을 구성하는 절차는 [수치 적분](05_numerical_integration.html)을 따릅니다. 361의 B-bar, F-bar 및 비적합 정식화와 보 요소 및 셸 요소의 형상 함수는 [고성능 요소 정식화](09_element_advanced.html)에서 다룹니다.

 
## 자연 좌표계와 내부 절점 순서

 형상 함수는 각 요소의 자연 좌표 \(\boldsymbol{r}\) 위에서 정의됩니다. 3차원 솔리드 요소는 요소 형상에 따라 다음 세 종류의 자연 좌표계를 사용합니다.

 
- **육면체(361/362)**: \(\boldsymbol{r} = (\xi, \eta, \zeta)\), \(-1 \le \xi, \eta, \zeta \le 1\).
- **사면체(341/342)**: 체적 좌표 \((L_1, L_2, L_3, L_4)\)를 사용하며 \(L_1 + L_2 + L_3 + L_4 = 1\), \(L_i \ge 0\)입니다. \(\boldsymbol{r} = (L_2, L_3, L_4)\)를 독립 변수로 취하면 \(L_1 = 1 - L_2 - L_3 - L_4\)입니다.
- **삼각기둥(351/352)**: 밑면 삼각형의 체적 좌표 \((\xi, \eta)\)(\(0 \le \xi, \eta\), \(\xi + \eta \le 1\))와 축 방향 좌표 \(\zeta \in [-1, 1]\)를 결합하여 \(\boldsymbol{r} = (\xi, \eta, \zeta)\)를 구성합니다. 보조량 \(a = 1 - \xi - \eta\)를 밑면의 세 번째 체적 좌표로 사용합니다.

 1차 요소에서는 HEC-MW 입력 형식과 내부 형식의 요소 절점 번호 순서가 동일합니다. 2차 요소에서는 중간절점의 배열 규약이 다릅니다. FrontISTR는 HEC-MW 형식을 입력으로 받아들이며(모서리 절점을 요소 둘레 순서로 나열한 뒤 중간절점을 모서리 1-2, 2-3, … 순으로 나열), 요소 라이브러리의 형상 함수는 다른 절점 순서를 전제로 구현되어 있습니다.

 두 형식의 차이는 메시 입력 직후 호출되는 `hecmw2fstr_mesh_conv`가 각 요소의 절점을 재정렬하여 해소합니다. 3차원 솔리드 요소 중 다음 두 유형이 변환됩니다.

 
| 요소 유형 | 변환 테이블 | 설명 |
| --- | --- | --- |
| 342 (2차 사면체) | `Table342` | 10개 절점을 \(\{1,2,3,4,7,5,6,8,9,10\}\) 순서로 재정렬 |
| 352 (2차 삼각기둥) | `Table352` | 15개 절점을 \(\{1,2,3,4,5,6,9,7,8,12,10,11,13,14,15\}\) 순서로 재정렬 |

 재정렬은 요소 연결 정보를 저장하는 배열에 직접 수행되므로 이후 요소 라이브러리 호출은 모두 FrontISTR 내부 절점 순서를 사용합니다. 2차 육면체 요소 362는 HEC-MW 입력과 FrontISTR 내부 형식의 중간절점 순서가 동일하므로 변환 테이블이 없습니다. 결과 출력 시에는 역변환 `fstr2hecmw_mesh_conv`가 쓰기 전에 HEC-MW 절점 순서를 복원하므로 사용자는 입출력 모두 HEC-MW 형식만 인식하면 됩니다. 아래 형상 함수의 절점 번호는 변환 후의 FrontISTR 내부 형식을 사용합니다.

 
## 1차 요소의 형상 함수

 1차 사면체 요소(341)의 형상 함수는 체적 좌표 자체입니다.

 \[ N_1 = L_1, \qquad N_2 = L_2, \qquad N_3 = L_3, \qquad N_4 = L_4. \] 1차 삼각기둥 요소(351)의 형상 함수는 밑면 삼각형의 선형 형상 함수(\(a, \xi, \eta\))와 축 방향의 선형 형상 함수(\((1-\zeta)/2, (1+\zeta)/2\))의 곱입니다.

 \[ \begin{aligned} N_1 &= \tfrac{1}{2}\, a\,(1-\zeta), & N_2 &= \tfrac{1}{2}\, \xi\,(1-\zeta), & N_3 &= \tfrac{1}{2}\, \eta\,(1-\zeta), \\ N_4 &= \tfrac{1}{2}\, a\,(1+\zeta), & N_5 &= \tfrac{1}{2}\, \xi\,(1+\zeta), & N_6 &= \tfrac{1}{2}\, \eta\,(1+\zeta). \end{aligned} \] 1차 육면체 요소(361)의 형상 함수는 세 좌표 방향의 선형 형상 함수의 텐서곱입니다.

 \[ N_\alpha = \tfrac{1}{8}\,(1+\xi_\alpha \xi)(1+\eta_\alpha \eta)(1+\zeta_\alpha \zeta), \qquad \alpha = 1, \ldots, 8. \] 여기서 \((\xi_\alpha, \eta_\alpha, \zeta_\alpha)\)는 절점 \(\alpha\)의 자연 좌표 값입니다(각 성분은 \(\pm 1\)).

 
## 2차 요소의 형상 함수

 2차 사면체 요소(342)의 형상 함수는 네 개의 모서리 절점과 여섯 개의 중간절점으로 나뉩니다.

 \[ \begin{aligned} N_1 &= L_1(2L_1 - 1), & N_2 &= L_2(2L_2 - 1), \\ N_3 &= L_3(2L_3 - 1), & N_4 &= L_4(2L_4 - 1), \\ N_5 &= 4\,L_1 L_2, & N_6 &= 4\,L_2 L_3, \\ N_7 &= 4\,L_1 L_3, & N_8 &= 4\,L_1 L_4, \\ N_9 &= 4\,L_2 L_4, & N_{10} &= 4\,L_3 L_4. \end{aligned} \] 중간절점 5–10은 각각 모서리 1-2, 2-3, 1-3, 1-4, 2-4, 3-4의 중점에 대응합니다.

 2차 삼각기둥 요소(352)는 여섯 개의 모서리 절점, 삼각형 면 위의 여섯 개의 중간절점, 축 방향 모서리 위의 세 개의 중간절점으로 구성됩니다.

 \[ \begin{aligned} N_1 &= \tfrac{1}{2}\, a\,(1-\zeta)(2a - 2 - \zeta), & N_2 &= \tfrac{1}{2}\, \xi\,(1-\zeta)(2\xi - 2 - \zeta), \\ N_3 &= \tfrac{1}{2}\, \eta\,(1-\zeta)(2\eta - 2 - \zeta), & N_4 &= \tfrac{1}{2}\, a\,(1+\zeta)(2a - 2 + \zeta), \\ N_5 &= \tfrac{1}{2}\, \xi\,(1+\zeta)(2\xi - 2 + \zeta), & N_6 &= \tfrac{1}{2}\, \eta\,(1+\zeta)(2\eta - 2 + \zeta), \\ N_7 &= 2\,\xi\, a\,(1-\zeta), & N_8 &= 2\,\xi\, \eta\,(1-\zeta), \\ N_9 &= 2\,\eta\, a\,(1-\zeta), & N_{10} &= 2\,\xi\, a\,(1+\zeta), \\ N_{11} &= 2\,\xi\, \eta\,(1+\zeta), & N_{12} &= 2\,\eta\, a\,(1+\zeta), \\ N_{13} &= a\,(1 - \zeta^2), & N_{14} &= \xi\,(1 - \zeta^2), \\ N_{15} &= \eta\,(1 - \zeta^2). \end{aligned} \] 중간절점 7–9는 아래쪽 삼각형(\(\zeta = -1\))의 모서리 중점, 절점 10–12는 위쪽 삼각형(\(\zeta = +1\))의 모서리 중점, 절점 13–15는 축 방향 모서리의 중점에 대응합니다.

 2차 육면체 요소(362)는 Serendipity 계열의 20절점 요소입니다. 절점 \(\alpha\)의 자연 좌표 값을 \((\xi_\alpha, \eta_\alpha, \zeta_\alpha)\)라고 합니다. 모서리 절점(\(\alpha = 1, \ldots, 8\), 각 성분은 \(\pm 1\))에 대해서는 다음과 같습니다.

 \[ N_\alpha = -\tfrac{1}{8}\,(1+\xi_\alpha \xi)(1+\eta_\alpha \eta)(1+\zeta_\alpha \zeta)\bigl(2 - \xi_\alpha \xi - \eta_\alpha \eta - \zeta_\alpha \zeta\bigr). \] 중간절점(\(\alpha = 9, \ldots, 20\), 세 자연 좌표 중 하나가 \(0\)이고 나머지 두 개가 \(\pm 1\)) 중 \(\xi_\alpha = 0\)인 절점(\(\xi\) 방향의 모서리)에 대해서는 다음과 같습니다.

 \[ N_\alpha = \tfrac{1}{4}\,(1 - \xi^2)(1 + \eta_\alpha \eta)(1 + \zeta_\alpha \zeta). \] 모서리 방향이 \(\eta\) 또는 \(\zeta\) 축인 중간절점도 마찬가지로 해당 자연 좌표를 제곱항으로 제거한 형식으로 주어집니다.

 
## 관련 항목

 
- [형상 함수와 유한요소 근사](01_shape_functions.html) — 형상 함수 \(N_\alpha^e\)의 수학적 정의
- [형상 함수의 공간 미분](02_global_derivative.html) — 형상 함수의 전체 좌표 미분과 Jacobian
- [수치 적분](05_numerical_integration.html) — 적분점, 가중치 및 요소 적분의 근사
- [고성능 요소 정식화](09_element_advanced.html) — B-bar, F-bar, 비적합 요소, 쉘 및 빔
- [요소 라이브러리 (기능)](../../functions_reference/02_element.html) — HEC-MW 형식의 절점 순서와 면 번호
- [물리량 기호 목록](../notation/02_symbol_list.html)


---


<!-- ========================================== -->
<!-- FILE: theory/discretization/09_element_advanced.md | TITLE: 고성능 요소 정식화 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/discretization/09_element_advanced.html -->

# 고성능 요소 정식화

 [3차원 솔리드 요소의 정식화](08_element_solid.html)에서 제시한 표준 변위 기반 정식화는 거의 비압축성인 재료나 굽힘 지배 박벽 구조에 적용하면 로킹(체적 로킹 또는 전단 로킹)이라 불리는 과도한 인공 강성을 나타냅니다. 이를 피하기 위해 FrontISTR는 B 행렬 또는 변형 구배의 체적 성분만 치환하는 B-bar법과 F-bar법, 내부 자유도를 도입하는 비적합 요소, 압력을 변위와 독립된 미지장으로 취급하는 u-p 혼합 요소, 그리고 판 및 보 구조에 특화된 MITC 셸 요소와 보 요소를 제공합니다.

 이 장에서는 이러한 고성능 요소 및 구조 요소의 정식화를 요소별로 정리합니다.

 
## B-bar법

 8절점 1차 육면체 요소를 준비압축성 재료에 사용하면 하나의 요소 내부 변형률장이 일정 체적 제약과 충돌하여 체적 로킹이라 불리는 과도한 강성이 발생합니다. B-bar법은 B 행렬에서 체적 팽창에 기여하는 성분을 요소 중심에서 평가한 값으로 치환하여 이러한 과구속을 완화합니다 [Hughes1980].

 요소 중심 \(\boldsymbol{r} = \boldsymbol{0}\)에서 형상 함수의 공간 미분으로 계산한 B 행렬을 \(\bar{\boldsymbol{B}}\), 적분점 \(\boldsymbol{r}\)에서 계산한 일반 B 행렬을 \(\boldsymbol{B}(\boldsymbol{r})\)라 합니다. 절점 \(\alpha\)의 자유도 \(i\)에 대한 변위-변형률 관계에서 체적 변형률 성분 \((\varepsilon_{11}, \varepsilon_{22}, \varepsilon_{33})\)에는

 \[ \Delta B_{i\alpha} = \tfrac{1}{3}\bigl(\bar{B}_{i\alpha}(\boldsymbol{0}) - B_{i\alpha}(\boldsymbol{r})\bigr) \] 를 더하고, 전단 성분 \((\varepsilon_{12}, \varepsilon_{23}, \varepsilon_{31})\)에는 일반 \(\boldsymbol{B}\)를 사용합니다. 이렇게 얻은 B-bar 행렬을 사용해 요소 강성과 내력 벡터를 조립합니다.

 FrontISTR는 이 정식화를 8절점 1차 육면체 요소(요소 ID 361, [요소 번호 체계](../../coding_guide/01_element_taxonomy.html)) 전용으로 제공하며, 미소변형, Total Lagrange, Updated Lagrange 정식화 모두에 적용할 수 있습니다.

 
## F-bar법

 유한변형에서는 체적 변화가 변형 구배 \(\boldsymbol{F}\)를 통해 비선형적으로 작용하므로, B-bar법과 유사한 체적 로킹 대책을 변형 구배 수준에서 적용하는 것이 F-bar법 [deSouzaNeto1996]입니다.

 요소 중심 \(\boldsymbol{r} = \boldsymbol{0}\)에서 평가한 변형 구배의 체적비를 \(J_0 = \det \boldsymbol{F}(\boldsymbol{0})\), 적분점에서의 체적비를 \(J = \det \boldsymbol{F}(\boldsymbol{r})\)라 하고, 적분점의 변형 구배를

 \[ \bar{\boldsymbol{F}} = \left(\frac{J_0}{J}\right)^{1/3} \boldsymbol{F} \] 로 치환합니다. 이에 따라 \(\det \bar{\boldsymbol{F}} = J_0\)가 되어 요소 전체의 체적비가 요소 중심값과 같아집니다. 응력 평가와 변형률-변위 행렬 구성에는 치환된 \(\bar{\boldsymbol{F}}\)를 사용하고, 요소 접선 강성에는 이 치환에 따른 추가 항을 포함합니다.

 FrontISTR는 F-bar법을 8절점 1차 육면체 요소 전용으로 구현하고 있으며, 미소변형뿐 아니라 Total Lagrange / Updated Lagrange 비선형 기하에도 적용할 수 있습니다.

 
## 비적합 요소

 8절점 1차 육면체 요소는 굽힘에 필요한 변형률 모드가 부족하여 굽힘 지배 문제에서 굽힘 로킹을 나타냅니다. 비적합 요소 [Taylor1976]는 이 부족함을 보완하기 위해 요소 내부에 추가 변위 모드를 도입합니다.

 요소 절점 변위 \(\boldsymbol{u}^e\)에 더해 요소 내부에만 존재하는 비적합 모드 자유도 \(\boldsymbol{\alpha} \in \mathbb{R}^{9}\)를 요소당 3방향 × 3모드로 도입하고, 변위장을

 \[ \boldsymbol{u}(\boldsymbol{r}) = \sum_{\alpha=1}^{8} N_\alpha^e(\boldsymbol{r})\, \boldsymbol{u}^e_\alpha + \sum_{k=1}^{3} M_k(\boldsymbol{r})\, \boldsymbol{\alpha}_k \] 로 근사합니다. 자연좌표 \(\boldsymbol{r} = (\xi, \eta, \zeta)\)에 대해 비적합 형상 함수는 \(M_1 = 1 - \xi^2\), \(M_2 = 1 - \eta^2\), \(M_3 = 1 - \zeta^2\)로 둡니다. 이들은 요소 경계에서 연속성을 보장하지 않지만 굽힘 모드와 관련된 변형률을 재현할 수 있는 내부 공간을 추가합니다.

 요소 강성은 먼저 외부 자유도/내부 자유도의 블록 형태로 조립합니다.

 \[ \begin{bmatrix} \boldsymbol{K}_{dd} & \boldsymbol{K}_{d\alpha} \\ \boldsymbol{K}_{\alpha d} & \boldsymbol{K}_{\alpha\alpha} \end{bmatrix} \begin{bmatrix} d\boldsymbol{u}^e \\ d\boldsymbol{\alpha} \end{bmatrix} = \begin{bmatrix} \boldsymbol{F}^e_{\text{ext}} \\ \boldsymbol{0} \end{bmatrix} \] 그 후 \(d\boldsymbol{\alpha} = -\boldsymbol{K}_{\alpha\alpha}^{-1}\boldsymbol{K}_{\alpha d}\,d\boldsymbol{u}^e\)를 이용해 내부 자유도를 제거하는 정적 축약을 수행하여 외부 자유도만 포함하는 요소 강성

 \[ \boldsymbol{K}^e = \boldsymbol{K}_{dd} - \boldsymbol{K}_{d\alpha}\,\boldsymbol{K}_{\alpha\alpha}^{-1}\,\boldsymbol{K}_{\alpha d} \] 을 전체 조립에 전달합니다.

 FrontISTR는 비적합 요소를 8절점 1차 육면체 요소(C3D8IC) 전용으로 구현하고 있으며 미소변형, Total Lagrange, Updated Lagrange 정식화 모두에 적용할 수 있습니다.

 
## U-P 혼합 요소

 B-bar법과 F-bar법이 변위 기반 틀 안에서 체적 성분을 보정하는 데 비해, u-p 혼합(U-P) 요소는 압력 \(\lambda\)를 변위와 독립된 미지장으로 도입하는 혼합 정식화 [Bathe1996]를 사용합니다. 포아송비가 0.5에 매우 가까운 고무계 재료나 소성 변형 이후의 금속처럼 거의 비압축성인 재료에서 변위장만으로 일정 체적 제약을 강제하면 체적 로킹이 발생하므로, 압력을 독립 변수로 취급하여 이 제약을 완화합니다.

 응력을 편차 성분과 압력 성분으로 분리하면

 \[ \boldsymbol{\sigma} = \boldsymbol{\sigma}_{\mathrm{dev}} + \lambda\,\boldsymbol{I}, \qquad \boldsymbol{\sigma}_{\mathrm{dev}} = \mathbf{D}_{\mathrm{dev}}\,\boldsymbol{\varepsilon} \] 입니다. 여기서 \(\mathbf{D}_{\mathrm{dev}}\)는 탄성 행렬에서 체적 탄성률 \(K\)에 비례하는 체적 성분을 제거하여 얻은 편차 탄성 행렬입니다. 압력 \(\lambda\)와 체적 변형률 \(g = \mathrm{tr}\,\boldsymbol{\varepsilon}\)는 압축성 \(\alpha^{-1} = 1/K\)를 통해 다음 제약 관계를 갖습니다.

 \[ g - \alpha^{-1}\lambda = 0 \] 변위 \(\boldsymbol{u}\)와 압력 \(\lambda\)를 미지수로 이산화하면 요소 연성 시스템은

 \[ \begin{bmatrix} \mathbf{K}_{uu} & \mathbf{K}_{up} \\ \mathbf{K}_{up}^{T} & \mathbf{K}_{pp} \end{bmatrix} \begin{bmatrix} d\boldsymbol{u} \\ d\lambda \end{bmatrix} = \begin{bmatrix} \boldsymbol{f}_{u} \\ \boldsymbol{f}_{p} \end{bmatrix} \] 이 됩니다. 여기서 \(\mathbf{K}_{uu}\)는 편차 탄성 기여와 유한변형에서는 기하 강성을 포함하고, \(\mathbf{K}_{up}\)는 체적 변형률과 압력을 연성하며, \(\mathbf{K}_{pp} = -\int \alpha^{-1}\,\boldsymbol{N}_p \boldsymbol{N}_p^{T}\,dV\)는 압력 안정화 항입니다(\(\boldsymbol{N}_p\)는 압력 형상 함수). 압력 자유도는 요소 내부 자유도이므로

 \[ \mathbf{K}_{\mathrm{eff}} = \mathbf{K}_{uu} - \mathbf{K}_{up}\,\mathbf{K}_{pp}^{-1}\,\mathbf{K}_{up}^{T} \] 형태로 정적 축약을 수행하고 외부(변위) 자유도만 포함하는 유효 강성을 전체 조립에 전달합니다.

 FrontISTR는 U-P 요소를 8절점 1차 육면체 요소 전용으로 구현하며 요소당 하나의 압력 자유도(요소 내 일정)를 사용합니다. 미소변형, Total Lagrange, Updated Lagrange 정식화에 적용할 수 있습니다. Updated Lagrange에서는 객관 응력률(Jaumann/Hughes-Winget형)을 사용해 편차 응력을 갱신한 뒤, 정적 축약으로 얻은 값을 사용하여 압력 \(\lambda\,\boldsymbol{I}\)를 부여합니다.

 
## 셸 요소

 얇은 판 및 셸 구조에는 Reissner-Mindlin 판/셸 이론에 기반한 셸 요소를 사용합니다. 두께가 얇아지면 저차 변위 기반 판/셸 요소는 체적 로킹과 유사한 메커니즘으로 횡전단 변형률을 과대평가하는 전단 로킹을 일으켜 굽힘 모드의 강성이 지나치게 커집니다. MITC(Mixed Interpolation of Tensorial Components)법 [Dvorkin1984] [Bathe1986]은 전단 변형률 성분만 요소 내 미리 정한 타잉점(tying point)에서 다시 샘플링하고 그 값을 요소 전체에 보간하여 이 문제를 회피합니다.

 MITC 셸 요소의 절점은 중립면에 놓이며, 각 절점은 3개의 병진 성분과 중립면 법선 좌표계에 대한 3개의 회전 성분으로 구성된 6자유도를 갖습니다. 요소 강성은 중립면 자연좌표와 두께 방향에 대한 3차원 가우스 적분으로 평가하며, 두께 \(h\)는 구성 관계 평가 시 요소 특성으로 주어집니다.

 FrontISTR는 하나의 중립면 층으로 표현되는 MITC3(요소 ID 731), MITC4(741), MITC9(743)와, 두께 방향으로 두 층에 절점을 배치한 적층 셸 요소 MITC3-shell361(761, 3\(\times\)2 절점, 절점당 3자유도) 및 MITC4-shell361(781, 4\(\times\)2 절점, 절점당 3자유도)을 제공합니다. 적층 셸 요소의 절점 자유도는 3개의 병진 성분만으로 구성되며, 회전 자유도에 해당하는 굽힘 모드는 두 층 배치로 표현됩니다.

 
## 보 요소

 거더와 프레임 구조 같은 선형 부재는 보 요소로 이산화합니다. 보 요소는 단면 내부의 변형을 단면 상수로 대표하고, 보 축 위 절점의 병진과 회전만으로 신장, 굽힘, 비틀림을 표현합니다.

 보 요소의 각 절점은 병진 3성분과 회전 3성분의 6자유도를 갖습니다. 요소 강성은 요소 축과 단면 데이터에서 주어지는 기준축으로 정한 국부 좌표계에서 신장, 두 방향 굽힘, 비틀림에 대한 폐형식 강성 행렬을 구성한 뒤 전역 좌표계로 좌표 변환하여 얻습니다. 단면적 \(A\), 국부 \(y\)축과 \(z\)축에 대한 단면 2차 모멘트 \(I_y\), \(I_z\), 비틀림 상수는 단면 데이터로 주어지며, 재료의 영률 \(E\)와 영률 및 포아송비로부터 구한 전단 탄성계수 \(G\)와 함께 각 강성 계수를 구성합니다.

 굽힘의 취급에는 변형 후에도 단면이 보 축에 수직인 채로 유지된다고 가정하여 전단 변형을 무시하는 Bernoulli-Euler 보와 전단 변형을 고려하는 Timoshenko 보가 있습니다. 2절점 보 요소 611에서는 둘 다 선택할 수 있으며 기본값은 Bernoulli-Euler 보입니다. Timoshenko 보에서는 요소 길이를 \(L\)로 할 때 전단 변형의 기여를 나타내는 계수

 \[ \Phi_y = \frac{12 E I_z}{G A_s L^2}, \qquad \Phi_z = \frac{12 E I_y}{G A_s L^2} \] 를 각 굽힘 면의 강성 계수에 포함합니다. 예를 들어 국부 \(y\) 방향 처짐에 대한 강성 계수는 \(12 E I_z / \{ L^3 (1 + \Phi_y) \}\)입니다. \(A_s\)는 전단 단면적이며, FrontISTR의 단면 데이터에는 \(A_s\)가 별도로 없으므로 \(A_s = A\)로 둡니다. 이는 전단 보정 계수를 1로 두는 것과 같습니다. \(\Phi_y = \Phi_z = 0\)으로 두면 Bernoulli-Euler 보의 강성과 일치합니다.

 641은 병진 자유도를 갖는 2개 절점과 회전 자유도를 갖는 2개 절점으로 보를 표현하는 혼합 자유도용 요소이며, Bernoulli-Euler 보로 정식화되어 있습니다.

 
## 관련 항목

 
- [3차원 솔리드 요소의 정식화](08_element_solid.html) — 표준 정식화와의 비교
- [요소 번호 체계와 형상 함수 라이브러리](../../coding_guide/01_element_taxonomy.html) — 요소 유형 ID
- [요소 라이브러리(기능)](../../functions_reference/02_element.html) — 각 요소 선택 지침


---


<!-- ========================================== -->
<!-- FILE: theory/governing/01a_kinematics.md | TITLE: 운동·변형·변형률 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/governing/01a_kinematics.html -->

# 운동·변형·변형률

 
## 기준 배치와 현재 배치

 
### 배치의 정의

 물리량을 "어느 시점의 형상"을 기준으로 기술하는지 구분하기 위해 이 매뉴얼에서는 두 종류의 배치를 사용한다.

 
- **기준 배치(reference configuration)**: 해석 시작 시점(시각 \(0\))의 변형 전 형상(**초기 배치**라고도 함). 기준 배치 내의 점을 **물질좌표** \(\boldsymbol{X}\)로 나타내고, 체적을 \(V\), 표면을 \(S\)로 한다.
- **현재 배치(current configuration)**: 시각 \(t\)에서 변형 후의 형상. 현재 배치 내의 점을 **공간좌표** \(\boldsymbol{x}\)로 나타내고, 체적을 \(v\), 표면을 \(s\)로 한다.

 운동은 이 두 배치 사이의 사상 \(\boldsymbol{x} = \boldsymbol{x}(\boldsymbol{X}, t)\)으로 나타내며, 변위는 \(\boldsymbol{u} = \boldsymbol{x} - \boldsymbol{X}\)로 정의한다.

 
### 배치에 의존하는 기호 규약(대문자/소문자)

 물리량이 어느 배치에 속하는지를 기호의 대소문자로 구분한다.

 
- 기준 배치에 속하는 양은 **대문자**로 표기한다(예: \(\boldsymbol{X}, \boldsymbol{F}, \boldsymbol{S}, \boldsymbol{E}, V, S\)).
- 현재 배치에 속하는 양은 **소문자**로 표기한다(예: \(\boldsymbol{x}, \boldsymbol{\sigma}, \boldsymbol{e}, v, s\)).
- 제1 Piola-Kirchhoff 응력 \(\boldsymbol{P}\)와 같이 두 배치에 걸친 양은 관례적인 기호를 사용하고 본문에서 필요할 때마다 명시한다.
- 미소변형 이론에서는 두 배치의 구별이 사라지므로 소문자(\(\boldsymbol{\sigma}, \boldsymbol{\varepsilon}\))만 사용한다.

 
### 기준 배치 기울기와 현재 배치 기울기

 공간 미분을 어느 배치에 대해 취하는지에 따라 기호와 의미가 달라진다.

 
- **기준 배치 기울기** \(\nabla_X = \partial / \partial \boldsymbol{X}\): 물질좌표에 관한 기울기. 예: \(\boldsymbol{F} = \partial \boldsymbol{x} / \partial \boldsymbol{X}\).
- **현재 배치 기울기** \(\nabla_x = \partial / \partial \boldsymbol{x}\): 공간좌표에 관한 기울기. 예: \(\boldsymbol{L} = \partial \boldsymbol{v} / \partial \boldsymbol{x}\).

 두 기울기는 \(\nabla_X = \boldsymbol{F}^T \nabla_x\)로 관계된다. 시간 미분 규약(물질 시간 미분 \(\dot{(\cdot)}\))은 [텐서 표기법과 수학적 기초](../notation/01_tensor_notation.html)를 참조한다.

 
## 운동의 기술

 
### 운동 사상과 물질 기술·공간 기술

 연속체를 시각 \(0\)에서의 위치 \(\boldsymbol{X}\)를 라벨로 하는 물질점의 집합으로 보면, 운동은 물질점 \(\boldsymbol{X}\)에 시각 \(t\)에서의 위치 \(\boldsymbol{x}\)를 대응시키는 사상 \(\phi\)에 의해

 \[ \boldsymbol{x} = \phi(\boldsymbol{X}, t) \] 로 나타낸다. 기준 배치를 초기 배치로 취하는 규약에서는 \(\boldsymbol{X} = \phi(\boldsymbol{X}, 0)\)이다.

 독립변수의 선택에 따라 두 종류의 기술을 구분한다.

 
- **물질 기술(Lagrange 기술)**: 물리량을 \((\boldsymbol{X}, t)\)의 함수로 나타낸다. 고체역학에서는 기준 배치가 고정 형상으로 주어지므로 자연스러운 선택이다.
- **공간 기술(Euler 기술)**: 물리량을 \((\boldsymbol{x}, t)\)의 함수로 나타낸다. 유체역학에서 표준적으로 사용된다.

 FrontISTR의 구조 해석은 기본적으로 물질 기술에 기반하지만, Updated Lagrange 법에서는 현재 배치를 새로운 기준 배치로 사용하므로 공간 기술의 양도 함께 사용한다.

 
### 변위·속도·가속도

 변위 \(\boldsymbol{u}\)는 앞 절과 같이 \(\boldsymbol{u} = \boldsymbol{x} - \boldsymbol{X}\)로 정의한다. 크기에 관한 가정은 두지 않고 유한변형까지 포함하여 다룬다. 속도 벡터와 가속도 벡터는 물질점을 고정한 \(\boldsymbol{u}\)의 물질 시간 미분으로서

 \[ \boldsymbol{v} = \dot{\boldsymbol{u}}, \qquad \boldsymbol{a} = \dot{\boldsymbol{v}} = \ddot{\boldsymbol{u}} \] 로 정의된다.

 
## 변형 구배와 변형 텐서

 
### 변형 구배 텐서

 유한변형 연속체역학의 기본량은 **변형 구배 텐서** \(\boldsymbol{F}\)이다. 이는 기준 배치의 미소 선분 \(d\boldsymbol{X}\)를 현재 배치의 미소 선분 \(d\boldsymbol{x}\)로 사상하는 선형 사상이며, 운동 사상의 물질좌표에 관한 기울기로

 \[ \boldsymbol{F} = \frac{\partial \boldsymbol{x}}{\partial \boldsymbol{X}} = \boldsymbol{I} + \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}}, \qquad F_{ij} = \frac{\partial x_i}{\partial X_j} \] 정의된다(\(d\boldsymbol{x} = \boldsymbol{F}\, d\boldsymbol{X}\)). 무변형 상태에서는 \(\boldsymbol{F} = \boldsymbol{I}\)가 된다. \(\boldsymbol{F}\)는 변형률과 응력의 정의, 배치 사이의 물리량 변환에 널리 사용된다.

 
### 체적 변화비

 기준 배치의 미소 체적 \(dV\)와 현재 배치의 미소 체적 \(dv\)의 비를 **체적 변화비** \(J\)라고 하며

 \[ J = \frac{dv}{dV} = \det \boldsymbol{F} \] 로 주어진다. 연속체로서의 가역성으로부터 \(J > 0\)을 가정한다. 질량 보존법칙은 \(\rho_0 = J\rho\)로 나타낸다.

 
### 오른쪽·왼쪽 Cauchy-Green 변형 텐서

 변형 구배 \(\boldsymbol{F}\)는 강체 회전을 포함하므로 변형 자체를 나타내는 척도에는 강체 회전을 제외한 양을 사용한다. **오른쪽 Cauchy-Green 변형 텐서** \(\boldsymbol{C}\)(기준 배치)와 **왼쪽 Cauchy-Green 변형 텐서** \(\boldsymbol{b}\)(현재 배치)를

 \[ \boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}, \qquad \boldsymbol{b} = \boldsymbol{F} \boldsymbol{F}^T \] 로 정의한다. 둘 다 대칭 텐서이며 동일한 고유값(주신장비의 제곱)을 갖는다. 초탄성 변형 에너지 함수의 주불변량은 이들을 사용하여 주어진다.

 
## 변형률 텐서

 
### Green-Lagrange 변형률 텐서

 기준 배치를 기준으로 하는 변형률 척도로 **Green-Lagrange 변형률 텐서** \(\boldsymbol{E}\)를

 \[ \boldsymbol{E} = \frac{1}{2}(\boldsymbol{C} - \boldsymbol{I}) = \frac{1}{2}(\boldsymbol{F}^T \boldsymbol{F} - \boldsymbol{I}) \] 로 정의한다. 기준 배치에서 미소 벡터 내적의 변화를 나타내는 대칭 텐서이며, 강체 회전에 대해 불변이고 무변형 상태에서 \(\boldsymbol{E} = \boldsymbol{0}\)이 된다. \(\boldsymbol{F} = \boldsymbol{I} + \partial \boldsymbol{u} / \partial \boldsymbol{X}\)를 사용하면

 \[ \boldsymbol{E} = \frac{1}{2}\left\{ \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} + \left(\frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}}\right)^T + \left(\frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}}\right)^T \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} \right\} \] 와 같이 변위 기울기의 1차항과 2차항의 합으로 나타내며, 2차항이 기하학적 비선형성을 나타낸다. Total Lagrange 법에서는 일 공액쌍 \((\boldsymbol{S}, \boldsymbol{E})\)으로 사용한다.

 
### Almansi 변형률 텐서

 현재 배치를 기준으로 하는 변형률 척도로 **Almansi 변형률 텐서** \(\boldsymbol{e}\)를

 \[ \boldsymbol{e} = \frac{1}{2}(\boldsymbol{I} - \boldsymbol{b}^{-1}) \] 로 정의한다. Green-Lagrange 변형률과는 \(\boldsymbol{E} = \boldsymbol{F}^T \boldsymbol{e}\, \boldsymbol{F}\)(pull-back)으로 연결된다.

 
### 미소 변형률 텐서

 변위 기울기가 작은 경우(\(|\partial u_i / \partial X_j| \ll 1\)), Green-Lagrange 변형률의 2차항을 무시할 수 있고 두 배치의 구별도 사라지므로 기울기를 공간좌표로 나타낸 **미소 변형률 텐서** \(\boldsymbol{\varepsilon}\)로 귀착된다:

 \[ \boldsymbol{\varepsilon} = \frac{1}{2}\left\{ \nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T \right\}, \qquad \varepsilon_{ij} = \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) \] FrontISTR의 선형 정적 해석·모드 해석·주파수 응답 해석·선형 동적 해석에서는 이 변형률 척도를 사용한다. 유한변형 해석에서는 Total Lagrange 법에서 \(\boldsymbol{E}\)를, Updated Lagrange 법에서는 현재 배치를 기준으로 하는 변형 속도 텐서 \(\boldsymbol{D}\)를 사용한다.

 
## 관련 항목

 
- [응력과 보존법칙](01b_stress_balance.html)
- [응력과 보존법칙](01b_stress_balance.html)
- [가상일의 원리](01c_virtual_work.html)
- [텐서 표기법과 수학적 기초](../notation/01_tensor_notation.html)
- [물리량 기호 목록](../notation/02_symbol_list.html)


---


<!-- ========================================== -->
<!-- FILE: theory/governing/01b_stress_balance.md | TITLE: 응력과 보존 법칙 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/governing/01b_stress_balance.html -->

# 응력과 보존 법칙

 이 장에서는 응력 텐서의 정의와 상호 관계, 그리고 질량·운동량·각운동량의 각 보존 법칙에서 유도되는 평형 방정식과 응력의 대칭성을 정리한다. 세 종류의 응력 텐서(Cauchy 응력 \(\boldsymbol{\sigma}\), 제1 Piola-Kirchhoff 응력 \(\boldsymbol{P}\), 제2 Piola-Kirchhoff 응력 \(\boldsymbol{S}\))는 기준면과 힘 벡터에 어느 배치를 사용하는지의 차이에 따라 도입된다. 배치·운동·변형 구배의 정의는 [운동·변형·변형률](01a_kinematics.html)을, 기호 목록은 [물리량 기호 목록](../notation/02_symbol_list.html)을 참조한다.

 
## Cauchy 응력 텐서

 현재 배치의 점 \(\boldsymbol{x}\)에서 미소면(면적 \(d\Gamma\), 외향 단위 법선 \(\boldsymbol{n}\))에 작용하는 힘을 \(d\boldsymbol{f}\)라 하고, **응력 벡터** \(\boldsymbol{t}\)를 단위 면적당 힘 \(d\boldsymbol{f} = \boldsymbol{t}\, d\Gamma\)로 정의한다. \(\boldsymbol{t}\)는 위치 \(\boldsymbol{x}\)와 법선 \(\boldsymbol{n}\) 모두에 의존한다. 미소 사면체에 운동량 보존 법칙을 적용하면 \(\boldsymbol{t}\)는 \(\boldsymbol{n}\)에 대해 선형이 되며, 2계 텐서 \(\boldsymbol{\sigma}(\boldsymbol{x}, t)\)가 존재하여

 \[ \boldsymbol{t} = \boldsymbol{\sigma} \boldsymbol{n}, \qquad t_i = \sigma_{ij} n_j \] 가 임의의 \(\boldsymbol{n}\)에 대해 성립한다(**Cauchy 응력 정리**). 이 텐서 \(\boldsymbol{\sigma}\)를 **Cauchy 응력 텐서**라고 하며, 현재 배치의 기하 형상을 기준으로 한 단위 면적당 힘이라는 의미에서 **진응력**이라고도 한다. 성분 \(\sigma_{ij}\)는 현재 배치에서 좌표축 \(x_j\)에 직교하는 미소면의 단위 면적에 작용하는 힘의 \(x_i\) 성분을 나타낸다.

 뒤에서 설명하는 각운동량 보존 법칙에 따라 Cauchy 응력은 대칭 텐서

 \[ \boldsymbol{\sigma} = \boldsymbol{\sigma}^T, \qquad \sigma_{ij} = \sigma_{ji} \] 이며, 3차원에서는 독립적인 6개 성분(\(\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{23}, \sigma_{31}\))을 갖는다. Voigt 표기에 의한 응력 벡터화는 이 대칭성을 전제로 한다([텐서 표기와 수학적 기초](../notation/01_tensor_notation.html) 참조). Cauchy 응력은 FrontISTR의 Updated Lagrange 법에 의한 기하학적 비선형 해석 및 미소 변형 해석의 응력 출력에 사용된다.

 
## 제1·제2 Piola-Kirchhoff 응력

 유한 변형에서는 기준 배치를 기준면·기준 벡터로 하여 응력을 표현하는 편이 편리한 경우가 많으므로, 두 종류의 **Piola-Kirchhoff 응력 텐서**를 도입한다. 기준 배치의 미소면 면적을 \(d\Gamma_0\), 외향 단위 법선을 \(\boldsymbol{N}\)이라 한다.

 **제1 Piola-Kirchhoff 응력**(**제1 PK 응력**, **공칭 응력**) \(\boldsymbol{P}\)는 현재 배치의 힘 \(d\boldsymbol{f}\)를 기준 배치의 미소면에 작용시켰을 때의 응력 텐서로서

 \[ d\boldsymbol{f} = \boldsymbol{P} \boldsymbol{N}\, d\Gamma_0 \] 로 정의된다. Nanson 공식 \(\boldsymbol{n}\, d\Gamma = J \boldsymbol{F}^{-T} \boldsymbol{N}\, d\Gamma_0\)와 Cauchy 응력 정리로부터 Cauchy 응력과의 관계

 \[ \boldsymbol{P} = J\, \boldsymbol{\sigma} \boldsymbol{F}^{-T}, \qquad \boldsymbol{\sigma} = \frac{1}{J} \boldsymbol{P} \boldsymbol{F}^T \] 를 얻는다. 제1 PK 응력은 일반적으로 비대칭 텐서이다.

 **제2 Piola-Kirchhoff 응력**(**제2 PK 응력**) \(\boldsymbol{S}\)는 현재 배치의 힘 \(d\boldsymbol{f}\)를 \(\boldsymbol{F}^{-1}\)로 기준 배치에 끌어온 뒤 기준 배치의 미소면에 작용시켰을 때의 응력 텐서로서

 \[ \boldsymbol{F}^{-1}\, d\boldsymbol{f} = \boldsymbol{S} \boldsymbol{N}\, d\Gamma_0 \] 로 정의된다. 힘 벡터와 작용면 모두 기준 배치의 양으로 표현되며, 강체 회전에 대해 불변이고 대칭 텐서이다. 제1 PK 응력과의 관계 및 Cauchy 응력과의 변환식은

 \[ \boldsymbol{P} = \boldsymbol{F} \boldsymbol{S}, \qquad \boldsymbol{\sigma} = \frac{1}{J} \boldsymbol{F} \boldsymbol{S} \boldsymbol{F}^T, \qquad \boldsymbol{S} = J\, \boldsymbol{F}^{-1} \boldsymbol{\sigma} \boldsymbol{F}^{-T} \] 이다. 제2 PK 응력은 초탄성 변형률 에너지 함수 \(W(\boldsymbol{C})\)로부터 \(\boldsymbol{S} = 2\,\partial W / \partial \boldsymbol{C}\)로 유도되며, Total Lagrange 법에서는 Green-Lagrange 변형률 \(\boldsymbol{E}\)와 짝을 이루는 응력 공액쌍 \((\boldsymbol{S}, \boldsymbol{E})\)으로 사용된다.

 기준 배치와 대칭성은 다음 표와 같다. 미소 변형의 극한(\(\boldsymbol{F} \to \boldsymbol{I}\), \(J \to 1\))에서는 세 종류의 응력이 일치한다.

 
| 응력 텐서 | 기준면 | 힘 벡터 | 대칭성 | 용도 |
| --- | --- | --- | --- | --- |
| Cauchy 응력 \(\boldsymbol{\sigma}\) | 현재 배치 \(d\Gamma, \boldsymbol{n}\) | 현재 배치 \(d\boldsymbol{f}\) | 대칭 | Updated Lagrange 법·미소 변형 해석 |
| 제1 PK 응력 \(\boldsymbol{P}\) | 기준 배치 \(d\Gamma_0, \boldsymbol{N}\) | 현재 배치 \(d\boldsymbol{f}\) | 일반적으로 비대칭 | 기준 배치의 평형 방정식 |
| 제2 PK 응력 \(\boldsymbol{S}\) | 기준 배치 \(d\Gamma_0, \boldsymbol{N}\) | 기준 배치 \(\boldsymbol{F}^{-1} d\boldsymbol{f}\) | 대칭 | Total Lagrange 법·초탄성 |

 
## 질량·운동량 보존 법칙과 평형 방정식

 현재 배치의 질량 밀도를 \(\rho\), 기준 배치의 질량 밀도를 \(\rho_0\)라 하면, 질량 보존 법칙 \(\int_{\Omega} \rho\, dv = \int_{\Omega_0} \rho_0\, dV\)는 체적 요소의 변환 \(dv = J\, dV\)에 의해 국소형

 \[ \rho_0 = J \rho \] 으로 귀착된다.

 물체력(단위 질량당)을 \(\boldsymbol{g}\), 가속도를 \(\boldsymbol{a}\)라 하고, 운동량 보존 법칙(Euler의 제1 운동 법칙)에 Cauchy 응력 정리와 Gauss 발산 정리를 적용하면 현재 배치에서의 국소형 **평형 방정식(운동 방정식)**

 \[ \nabla_x \cdot \boldsymbol{\sigma} + \rho \boldsymbol{g} = \rho \boldsymbol{a}, \qquad \frac{\partial \sigma_{ij}}{\partial x_j} + \rho g_i = \rho a_i \] 을 얻는다. Nanson 공식과 \(\boldsymbol{P} = J \boldsymbol{\sigma} \boldsymbol{F}^{-T}\)에 의해 기준 배치의 적분으로 다시 쓰면 기준 배치에서의 국소형

 \[ \nabla_X \cdot \boldsymbol{P} + \rho_0 \boldsymbol{g} = \rho_0 \boldsymbol{a}, \qquad \frac{\partial P_{ij}}{\partial X_j} + \rho_0 g_i = \rho_0 a_i \] 을 얻는다. 두 형식은 응력 변환 법칙과 \(\rho_0 = J\rho\)에 의해 동등하다. 관성항을 무시한

 \[ \nabla_x \cdot \boldsymbol{\sigma} + \rho \boldsymbol{g} = \boldsymbol{0}, \qquad \nabla_X \cdot \boldsymbol{P} + \rho_0 \boldsymbol{g} = \boldsymbol{0} \] 은 정적 평형 방정식이며 FrontISTR 정적 해석(선형·비선형)의 출발식이 된다.

 
## 각운동량 보존 법칙과 응력 대칭성

 각운동량 보존 법칙(Euler의 제2 운동 법칙)에 운동량 보존 법칙에서 얻은 평형 방정식을 결합하면 **Cauchy 응력 텐서의 대칭성**

 \[ \boldsymbol{\sigma} = \boldsymbol{\sigma}^T, \qquad \sigma_{ij} = \sigma_{ji} \] 이 유도되며, 3차원에서는 독립적인 6개 성분으로 이루어진다. 변환식 \(\boldsymbol{\sigma} = J^{-1} \boldsymbol{F} \boldsymbol{S} \boldsymbol{F}^T\)의 양변을 전치하고 \(\boldsymbol{F}\)의 정칙성을 이용하면 제2 PK 응력도 대칭

 \[ \boldsymbol{S} = \boldsymbol{S}^T \] 이며 독립적인 6개 성분으로 이루어진다. 한편 제1 PK 응력 \(\boldsymbol{P} = \boldsymbol{F} \boldsymbol{S}\)는 일반적으로 대칭이 아니며, 관계식 \(\boldsymbol{P} \boldsymbol{F}^T = \boldsymbol{F} \boldsymbol{P}^T\)(즉 \(\boldsymbol{P} \boldsymbol{F}^T\)가 대칭)만 성립하고 독립 성분 수는 9개이다.

 이러한 대칭성은 Voigt 표기에 의한 응력의 6성분 벡터화의 전제가 된다. Voigt 표기의 규약과 재료 행렬의 구성은 [텐서 표기와 수학적 기초](../notation/01_tensor_notation.html) 및 [선형 탄성](../constitutive/01_linear_elastic.html)을 참조한다.

 
## 관련 항목

 
- [운동·변형·변형률](01a_kinematics.html)
- [운동·변형·변형률](01a_kinematics.html)
- [가상일의 원리](01c_virtual_work.html)
- [텐서 표기와 수학적 기초](../notation/01_tensor_notation.html)
- [물리량 기호 목록](../notation/02_symbol_list.html)


---


<!-- ========================================== -->
<!-- FILE: theory/governing/01c_virtual_work.md | TITLE: 가상일의 원리 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/governing/01c_virtual_work.html -->

# 가상일의 원리

 [응력과 보존 법칙](01b_stress_balance.html)에서 유도한 평형방정식과 경계조건을 바탕으로, 연속체역학 경계값 문제의 약형인 **가상일의 원리**를 유도한다. 유한요소법에 의한 이산화는 이 약형에서 출발한다. 본 장에서는 현재 배치(Cauchy 응력과 Almansi 변형률의 선형 부분으로 표현)와 기준 배치(제2 Piola-Kirchhoff 응력과 Green-Lagrange 변형률로 표현)의 두 형식을 제시하고, 양자가 등가임을 보인 뒤 미소변형으로의 귀착을 확인한다.

 
## 평형방정식과 경계조건

 연속체에 작용하는 단위 질량당 물체력을 \(\boldsymbol{g}\)라 하고, 현재 배치에서 영역 \(\Omega\)를 차지하는 물체를 생각한다. 경계 \(\Gamma\)는 변위가 \(\bar{\boldsymbol{u}}\)로 규정되는 기하학적 경계 \(\Gamma_B\)와 표면력이 \(\bar{\boldsymbol{t}}\)로 규정되는 역학적 경계 \(\Gamma_t\)로 분할되며, \(\Gamma = \Gamma_B \cup \Gamma_t\), \(\Gamma_B \cap \Gamma_t = \emptyset\)로 한다. 정적 문제에서는 [응력과 보존 법칙](01b_stress_balance.html)에 제시된 운동량 보존법칙에서 관성항을 제외한

 \[ \nabla_x \cdot \boldsymbol{\sigma} + \rho \boldsymbol{g} = \boldsymbol{0} \quad \text{영역} \ \Omega \] 을 평형방정식으로 한다. 경계조건은

 \[ \boldsymbol{\sigma} \boldsymbol{n} = \bar{\boldsymbol{t}} \quad \text{경계} \ \Gamma_t \] \[ \boldsymbol{u} = \bar{\boldsymbol{u}} \quad \text{경계} \ \Gamma_B \] 으로 주어진다. 이하에서는 가상일의 원리를 평형방정식과 역학적 경계조건 \(\boldsymbol{\sigma} \boldsymbol{n} = \bar{\boldsymbol{t}}\)의 약형으로 유도한다. 기하학적 경계조건 \(\boldsymbol{u} = \bar{\boldsymbol{u}}\)는 시험함수의 선택을 통해 반영된다.

 
## 현재 배치 표현의 약형

 약형에서는 미지 변위의 허용공간과 시험함수 공간을 각각

 \[ \mathcal{U} = \{ \boldsymbol{u} \in [H^1(\Omega)]^d \mid \boldsymbol{u} = \bar{\boldsymbol{u}} \ \text{경계} \ \Gamma_B \} \] \[ \mathcal{V} = \{ \delta \boldsymbol{u} \in [H^1(\Omega)]^d \mid \delta \boldsymbol{u} = \boldsymbol{0} \ \text{경계} \ \Gamma_B \} \] 로 둔다. 여기서 \(d\)는 공간 차원, \(H^1(\Omega)\)는 1차 약미분까지 제곱적분 가능한 Sobolev 공간이며, \(\delta\)는 변분 기호이다. 현재 배치 표현에서 \(\Omega\)는 변형 후의 배치이며, 실제 수치해법에서는 기준 배치 또는 알려진 중간 배치로 되돌려 취급한다.

 평형방정식에 가중함수 \(\delta \boldsymbol{u} \in \mathcal{V}\)를 곱하고 Gauss의 발산정리와 역학적 경계조건을 적용하면, 현재 배치에서의 가상일의 원리는 다음 식이 된다.

 \[ \int_{\Omega} \boldsymbol{\sigma} : \delta \boldsymbol{A}_{(L)}\, dv = \int_{\Gamma_t} \delta \boldsymbol{u}^T \bar{\boldsymbol{t}}\, d\Gamma + \int_{\Omega} \delta \boldsymbol{u}^T \rho \boldsymbol{g}\, dv \] 여기서 \(\boldsymbol{A}_{(L)}\)는 **Almansi 변형률 텐서의 선형 부분**이며,

 \[ \boldsymbol{A}_{(L)} = \frac{1}{2}\left( \nabla_x \boldsymbol{u} + (\nabla_x \boldsymbol{u})^T \right), \qquad A_{(L)ij} = \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) \] 로 정의된다. 그 변분은 \(\delta \boldsymbol{A}_{(L)} = \tfrac{1}{2}(\nabla_x \delta \boldsymbol{u} + (\nabla_x \delta \boldsymbol{u})^T)\)이다. 즉, \(\boldsymbol{u} \in \mathcal{U}\)를 구하고 임의의 \(\delta \boldsymbol{u} \in \mathcal{V}\)에 대해 가상일 식을 만족하도록 요구한다. 좌변은 내력의 가상일, 우변은 규정 표면력과 물체력에 의한 외력의 가상일이다.

 이 식은 변형 후(현재 배치)의 영역에서 기술되어 있으므로, 실제 해법에서는 초기 배치 \(\Omega_0\)(기준 배치) 또는 알려진 중간 배치를 참조 배치로 다시 선택하고 증분형으로 고쳐 쓴 뒤 해를 구한다. 참조 배치의 구체적인 선택(Total Lagrange / Updated Lagrange)과 증분 분해는 [증분해석의 틀](01d_incremental_analysis.html)을 참조한다.

 
## 초기 배치 표현의 약형

 기준 배치에서 영역 \(\Omega_0\)를 차지하는 물체를 생각하고, 그 경계 \(\Gamma_0\)를 \(\Gamma_{0B} \cup \Gamma_{0t}\)로 분할한다. 현재 배치 표현을 기준 배치로 되돌리면 응력-변형률의 공액쌍은 제2 Piola-Kirchhoff 응력 \(\boldsymbol{S}\)와 Green-Lagrange 변형률 \(\boldsymbol{E}\)가 된다. 이때 초기 배치에서의 가상일의 원리는

 \[ \int_{\Omega_0} \boldsymbol{S} : \delta \boldsymbol{E}\, dV = \int_{\Gamma_{0t}} \delta \boldsymbol{u}^T \bar{\boldsymbol{t}}\, d\Gamma_0 + \int_{\Omega_0} \delta \boldsymbol{u}^T \rho_0 \boldsymbol{g}\, dV \] 로 쓸 수 있다. 여기서 \(\rho_0\)는 기준 배치의 질량 밀도이며, 질량보존법칙 \(\rho_0 = J\rho\)에 의해 현재 배치의 물체력 표현과 등가이다.

 
### 현재 배치 표현과 초기 배치 표현의 등가성

 두 표현의 내력 가상일은 변형구배 \(\boldsymbol{F}\)와 체적비 \(J = \det \boldsymbol{F}\)에 의한 변환을 통해 일치한다. 즉,

 \[ \int_{\Omega_0} \boldsymbol{S} : \delta \boldsymbol{E}\, dV = \int_{\Omega} \boldsymbol{\sigma} : \delta \boldsymbol{A}_{(L)}\, dv \] 이다. 외력항도 질량보존법칙과 표면력의 변환에 의해 등가이다. 따라서 현재 배치의 가상일 식과 초기 배치의 가상일 식은 동일한 원리를 서로 다른 배치에서 표현한 것이다. 기준 배치를 참조하는 해법은 Total Lagrange법에, 현재 배치(직전의 수렴 배치)를 참조하는 해법은 Updated Lagrange법에 대응한다.

 
## 미소변형으로의 귀착

 미소변형 가정 \(\boldsymbol{F} \approx \boldsymbol{I}\), \(J \approx 1\) 아래에서는 현재 배치와 기준 배치의 구별이 사라지고, 제2 PK 응력은 Cauchy 응력과 일치하며(\(\boldsymbol{S} \to \boldsymbol{\sigma}\)), Green-Lagrange 변형률과 Almansi 변형률의 선형 부분은 모두 미소변형률 \(\boldsymbol{\varepsilon}\)로 귀착된다.

 \[ \boldsymbol{\varepsilon} = \nabla_S \boldsymbol{u} = \frac{1}{2}\left( \nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T \right), \qquad \varepsilon_{ij} = \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) \] 이때 가상일의 원리는 Cauchy 응력 \(\boldsymbol{\sigma}\)와 미소변형률 \(\boldsymbol{\varepsilon}\)에 의한 약형

 \[ \int_{\Omega} \boldsymbol{\sigma} : \delta \boldsymbol{\varepsilon}\, dV = \int_{\Gamma_t} \delta \boldsymbol{u}^T \bar{\boldsymbol{t}}\, d\Gamma + \int_{\Omega} \delta \boldsymbol{u}^T \rho \boldsymbol{g}\, dV \] \[ \delta \boldsymbol{u} = \boldsymbol{0} \quad \text{경계} \ \Gamma_B \] 으로 귀착된다. 이는 미소변형·선형탄성 정적해석에서 이산화에 직접 사용하는 약형이다([선형탄성 정적해석(입문·부록)](../appendix_static_linear.html)에서는 이 형식을 출발점으로 하여 요소 강성 \(\boldsymbol{K}^e\)의 구성부터 전체 방정식 \(\boldsymbol{K}\boldsymbol{U} = \boldsymbol{F}\)의 조립까지 제시한다).

 선형탄성 구성 법칙 \(\boldsymbol{\sigma} = \boldsymbol{\mathsf{C}} : \boldsymbol{\varepsilon}\)를 대입하고 [Voigt 표기](../notation/01_tensor_notation.html)에서 \(\hat{\sigma} = D\, \hat{\varepsilon}\)로 쓰면, 약형은

 \[ \int_{\Omega} \delta \hat{\varepsilon}^T D\, \hat{\varepsilon}\, dV = \int_{\Gamma_t} \delta \boldsymbol{u}^T \bar{\boldsymbol{t}}\, d\Gamma + \int_{\Omega} \delta \boldsymbol{u}^T \rho \boldsymbol{g}\, dV \] 의 형태가 된다.

 
## 관련 항목

 
- [운동·변형·변형률](01a_kinematics.html)
- [운동·변형·변형률](01a_kinematics.html)
- [응력과 보존 법칙](01b_stress_balance.html)
- [선형탄성 정적해석(입문·부록)](../appendix_static_linear.html) — 미소변형에서의 귀결
- [증분해석의 틀](01d_incremental_analysis.html) — 시간 방향 이산화와 TL/UL 분기의 선언
- [내력 가상일의 이산화](../discretization/03_internal_force.html) — 약형의 공간 이산화
- [접선강성행렬](../solution_methods/01_newton_raphson.html) — 선형화와 Newton-Raphson 해법
- [물리량 기호 목록](../notation/02_symbol_list.html)


---


<!-- ========================================== -->
<!-- FILE: theory/governing/01d_incremental_analysis.md | TITLE: 증분해석의 틀 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/governing/01d_incremental_analysis.html -->

# 증분해석의 틀

 
## 증분해석의 필요성

 앞에서 설명한 바와 같이 미소변형 문제의 해석에서는 평형방정식 등의 기초방정식과 등가인 가상일의 원리를 이용하고, 이 식을 유한요소로 이산화함으로써 유한요소해석을 수행할 수 있다. 구조물의 대변형을 다루는 유한변형 문제의 해석에서도 기본적으로 가상일의 원리를 사용한다는 점은 동일하다.

 그러나 유한변형 문제에서는 재료의 선형성을 가정하더라도 가상일의 원리 식은 변위에 대해 비선형 방정식이 된다. 비선형식을 풀기 위해서는 일반적으로 반복법에 의한 반복 계산이 사용된다.

 그 반복 계산에서는 작은 하중 증분마다 구간별로 계산하고 이를 누적하여 최종 변형 상태에 도달하는 증분해석 기법이 사용된다. 미소변형 문제를 가정한 경우 변형률과 응력을 정의하기 위한 배치는 변형 전과 변형 후를 특별히 구별하지 않았다. 즉, 미소변형을 가정한 경우에는 기초방정식을 기술하는 배치가 변형 전이든 변형 후이든 문제가 되지 않았다.

 유한변형 문제에서 증분해석을 수행할 경우 참조 배치로 초기 배치를 취하는지, 증분 시작 시점의 배치를 취하는지에 따라 정식화가 달라진다. 전자를 total Lagrange법, 후자를 updated Lagrange법이라 한다. 자세한 내용은 장 말의 참고문헌 등을 참조하기 바란다.

 
## 출발식: 시각 \(t'\)에서의 가상일의 원리

 시각 \(t\)까지의 상태가 알려져 있고 시각 \(t' = t + \Delta t\)의 상태를 미지로 하는 증분해석을 가정한다(그림 2.2.1 참조). 시각 \(t'\)에서의 평형방정식·경계조건·가상일의 원리(현재 배치 표현, Cauchy 응력 \(^{t'}\sigma\)와 Almansi 변형률의 선형 부분 \(^{t'} A_{(L)}\)에 의한 표현)는 [가상일의 원리](01c_virtual_work.html)에 제시된 바와 같다.

 ![증분해석의 개념](/theory/image/theory02_01.png)

 그림 2.2.1 증분해석의 개념

 다만 이 출발식은 시각 \(t'\)의 (미지의) 배치에서 기술되어 있으므로, 실제 해법에서는 시각 \(0\)의 초기 배치 \(V\) 또는 시각 \(t\)의 현재 배치 \(^{t'} v\)를 참조 배치로 다시 선택하고 증분형으로 고쳐 쓴 뒤 해를 구한다:

 
- **Total Lagrange법** — 초기 배치 \(V\)를 참조한다. 제2 PK 응력 \(\boldsymbol{S}\)와 Green-Lagrange 변형률 \(\boldsymbol{E}\)를 사용한다.
- **Updated Lagrange법** — 증분 시작 시점의 현재 배치 \(^{t}v\)를 참조한다. Cauchy 응력 \(\boldsymbol{\sigma}\)와 Almansi 변형률의 선형 부분 \(\boldsymbol{A}_{(L)}\)을 사용한다.

 두 정식화의 공간 이산화(B 행렬·내력 벡터)는 [내력 가상일의 이산화](../discretization/03_internal_force.html)를, 선형화와 접선강성의 구성은 [접선강성행렬](../solution_methods/01_newton_raphson.html)을 참조한다.

 
## 관련 항목

 
- [가상일의 원리](01c_virtual_work.html) — 출발식의 본체(현재 배치·초기 배치의 약형)
- [내력 가상일의 이산화](../discretization/03_internal_force.html) — TL/UL의 B 행렬과 내력 벡터
- [접선강성행렬](../solution_methods/01_newton_raphson.html) — 선형화에 의한 \(\boldsymbol{K}^{tl}, \boldsymbol{K}^{up}\)
- [Newton-Raphson법](../solution_methods/01_newton_raphson.html) — 반복 알고리즘


---


<!-- ========================================== -->
<!-- FILE: theory/governing/02_heat_equation.md | TITLE: 열전도 방정식 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/governing/02_heat_equation.html -->

# 열전도 방정식

 이 장에서는 FrontISTR가 대상으로 하는 열전도의 지배방정식(연속체 수준)과 경계 조건을 정리한다. 시간 이산화 및 FEM 이산화는 이산화(L3d)와 해법 알고리즘(L3e: `solution_methods/05_transient_heat`)에서 다룬다.

 
## 지배방정식

 연속체에서의 열전도 방정식은 다음과 같다.

 \[\begin{equation} \rho c \frac{\partial T}{\partial t} = \frac{\partial}{\partial x}\left(k_x \frac{\partial T}{\partial x}\right) + \frac{\partial}{\partial y}\left(k_y \frac{\partial T}{\partial y}\right) + \frac{\partial}{\partial z}\left(k_z \frac{\partial T}{\partial z}\right) + Q \label{eq:gov_he_main} \end{equation}\] 여기서 \(\rho=\rho(x)\)는 질량 밀도, \(c=c(x,T)\)는 비열, \(T=T(x,t)\)는 온도, \(k=k(x,T)\)는 열전도율, \(Q=Q(x,T,t)\)는 발열량, \(x\)는 위치, \(t\)는 시간을 나타낸다. \(k_x, k_y, k_z\)는 이방성 열전도에서 각 방향의 열전도율이다.

 구체적인 열물성(온도 의존성·이방성)은 [열물성](../constitutive/06_thermal_properties.html)을 참조한다.

 
## 경계 조건

 고려하는 영역을 \(S\), 그 경계를 \(\Gamma\)라 한다. \(\Gamma\) 위의 모든 점에서 Dirichlet형 또는 Neumann형 경계 조건 중 하나가 주어진다고 가정하면 경계 조건은 다음과 같다.

 \[\begin{equation} T = T_1(x,t), \; x \in \Gamma_1 \label{eq:gov_he_dirichlet} \end{equation}\] \[\begin{equation} k \frac{\partial T}{\partial n} = q(x,T,t), \; x \in \Gamma_2 \label{eq:gov_he_neumann} \end{equation}\] 단, \(T_1\)과 \(q\)의 함수형은 알려져 있다고 한다. \(q\)는 경계로부터 유출되는 열유속이다. 경계 열유속 \(q\)는 다음 세 종류의 성분으로 구성된다.

 \[\begin{equation} q=-q_s+q_c+q_r \label{eq:gov_he_flux} \end{equation}\] \[\begin{equation} q_s=q_s(x,t) \label{eq:gov_he_qs} \end{equation}\] \[\begin{equation} q_c=hc(T-Tc) \label{eq:gov_he_qc} \end{equation}\] \[\begin{equation} q_r=hr(T^4-Tr^4) \label{eq:gov_he_qr} \end{equation}\] 여기서 \(q_s\)는 분포 열유속, \(q_c\)는 대류 열전달에 의한 열유속, \(q_r\)는 복사 열전달에 의한 열유속이다.

 단, \(Tc=Tc(x,t)\)는 대류 열전달의 주위 온도, \(hc=hc(x,t)\)는 대류 열전달계수, \(Tr=Tr(x,t)\)는 복사 열전달의 주위 온도, \(hr=\varepsilon \sigma F = hr(x,t)\)는 복사 열전달계수, \(\varepsilon\)는 방사율, \(\sigma\)는 Stefan-Boltzmann 상수, \(F\)는 형상계수이다.

 입력 파일에서 지정하는 방법은 기능편 [경계 조건·하중](../../functions_reference/04_boundary_load.html)을 참조한다.

 
## 약형

 
## 구조 해석과의 연성 경계

 
## 관련 항목

 
- [비정상 열전도](../solution_methods/05_transient_heat.html) — 시간 이산화와 비선형 반복
- [열물성](../constitutive/06_thermal_properties.html) — 물성 데이터의 정식화와 온도 의존성
- [형상 함수와 유한요소 근사](../discretization/01_shape_functions.html) — FEM 이산화의 틀
- 기능 레퍼런스 [경계 조건·하중](../../functions_reference/04_boundary_load.html) — 열전도 해석의 경계 조건 입력


---


<!-- ========================================== -->
<!-- FILE: theory/index.md | TITLE: 이론 매뉴얼 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/index.html -->

# 이론 매뉴얼

 본 매뉴얼에서는 FrontISTR 내부에서 채택하고 있는 정식화를 기술한다. 일반적인 유한요소법 교과서식 입문서가 아니라, FrontISTR의 해석 기능을 뒷받침하는 구체적인 정식화 및 알고리즘의 설명으로 한정한다.

 각 기능의 개요와 사용 구분은 [기능 레퍼런스](../functions_reference/01_analysis_type.html), 입력 파일 형식은 [키워드 레퍼런스](../keyword_reference/cnt/solution.html)를 참조하기 바란다.

 
## 장 구성

 
- [선형 탄성 정적 해석(입문·부록)](appendix_static_linear.html) — 미소 변형·선형 탄성의 가정하에서 약형식으로부터 전체 방정식 \(KU=F\)의 유도까지를 한 장에서 완결하여 제시하는 입문 총정리
- **표기** — 텐서 표기법·Voigt 표기의 규약
- **장의 지배방정식** — 연속체역학의 기초(운동·변형·변형률 / 응력과 보존법칙 / 가상일의 원리), 열전도 방정식
- **구성 법칙** — 선형 탄성, 초탄성, 탄소성, 점탄성, 크리프, 열물성
- **이산화** — 약형식과 유한요소법, 요소 정식화
- **해법 알고리즘** — 비선형 정적 해석(증분 해석의 틀 / Total Lagrange / Updated Lagrange / Newton-Raphson 법과 수렴 판정), 동적 해석, 모드 해석, 주파수 응답 해석, 비정상 열전도 해석
- **접촉** — 접촉 문제의 정식화
- **선형 솔버** — 반복법·직접법의 수학적 원리
- **병렬 계산** — 영역 분할법에 기반한 병렬 유한요소법
- [참고문헌](99_reference.html)


---


<!-- ========================================== -->
<!-- FILE: theory/linear_solver/01_solver.md | TITLE: 선형 솔버 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/linear_solver/01_solver.html -->

# 선형 솔버

 TBD

 
## 관련 항목

 
- [솔버·전처리](../../functions_reference/06_solver.html) — 솔버 선택지와 사용 구분


---


<!-- ========================================== -->
<!-- FILE: theory/notation/01_tensor_notation.md | TITLE: 텐서 표기법과 수학적 기초 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/notation/01_tensor_notation.html -->

# 텐서 표기법과 수학적 기초

 이 장에서는 FrontISTR 이론 매뉴얼의 이후 장에서 사용하는 텐서 표기법·지표 표기법·Voigt 표기법의 규약을 정리한다. 여기서는 물리량의 정의에 의존하지 않는 **순수한 표기 규칙**만 설명한다. 연속체 운동에 수반되는 배치(기준 배치/현재 배치)에 의존하는 기호 규약(대문자·소문자의 구분, 물질 미분과 공간 미분의 구분)은 배치 개념을 도입하는 [운동·변형·변형률](../governing/01a_kinematics.html)에 둔다.

 
## 텐서 표기법

 
### 굵은 글꼴에 의한 벡터·텐서 표기

 서체에 관한 원칙으로 스칼라량 및 벡터·텐서의 **성분**은 보통 글꼴(\(\rho\), \(x_i\) 등)로 표기하고, 벡터·텐서 **양 자체**는 굵은 글꼴(\(\boldsymbol{a}\), \(\boldsymbol{E}\), \(\boldsymbol{\sigma}\) 등)로 표기한다.

 
### Einstein 합산 규약

 특별한 언급이 없는 한 동일한 지표가 같은 항에 두 번 나타나는 경우 그 지표에 대해 합을 취하는 Einstein 합산 규약을 적용한다. 예를 들어

 \[ a_i b_i = \sum_i a_i b_i, \quad A_{ij} B_{ij} = \sum_{i,j} A_{ij} B_{ij}, \quad \frac{\partial a_i}{\partial x_i} = \sum_i \frac{\partial a_i}{\partial x_i} \] 이고, 텐서 \(\boldsymbol{A}\), \(\boldsymbol{B}\)의 곱 \(\boldsymbol{C} = \boldsymbol{A}\boldsymbol{B}\)의 \(i,j\) 성분은

 \[ C_{ij} = A_{ik} B_{kj} \] 가 된다. 지표 규약으로서 특별한 언급 없이 \(i, j, k, l, \ldots\)가 지표에 사용된 경우 자유도에 관한 지표(3차원이라면 \(1, 2, 3\))를 의미하며, 그리스 소문자 \(\alpha, \beta, \gamma, \ldots\)가 지표에 사용된 경우 요소를 구성하는 절점에 관한 지표를 의미한다.

 
### 내적·전치·텐서곱

 텐서 \(\boldsymbol{A}\)의 전치를 \(\boldsymbol{A}^T\)로 나타낸다. 2계 텐서 \(\boldsymbol{A}\), \(\boldsymbol{B}\)의 내적(이중 축약)을

 \[ \boldsymbol{A} : \boldsymbol{B} \equiv A_{ij} B_{ij} \] 로 표기한다. 벡터 \(\boldsymbol{a}\), \(\boldsymbol{b}\)의 내적은 \(\boldsymbol{a} \cdot \boldsymbol{b} = a_i b_i\), 텐서곱은 \((\boldsymbol{a} \otimes \boldsymbol{b})_{ij} = a_i b_j\)이다.

 
## Voigt 표기

 응력과 변형률은 자유도에 관한 2계 대칭 텐서이며, 그 사이의 선형 관계를 나타내는 계수는 4계 텐서가 된다. 이를 그대로 프로그램에서 다루는 것은 계산량과 코딩(다차원 배열·깊은 다중 루프) 측면에서 불리하므로, 일반적인 유한요소법 프로그램에서는 대칭성을 고려하여 **응력·변형률은 열벡터**로, **4계 탄성 텐서는 2차원 행렬**로 압축하여 다룬다. 이를 Voigt 표기라고 한다.

 이후 텐서량 \(\boldsymbol{A}\)의 행렬·벡터 표기를 \(\hat{A}\)로 나타내어 원래 텐서와 구별한다.

 
### 응력·변형률의 벡터 표기

 대칭 응력 텐서 \(\boldsymbol{\sigma}\) 및 변형률 텐서 \(\boldsymbol{\varepsilon}\)에 대해 2차원인 경우

 \[ \hat{\sigma} = \begin{bmatrix} \sigma_{11} \\ \sigma_{22} \\ \sigma_{12} \end{bmatrix}, \qquad \hat{\varepsilon} = \begin{bmatrix} \varepsilon_{11} \\ \varepsilon_{22} \\ 2 \varepsilon_{12} \end{bmatrix} \] 3차원인 경우

 \[ \hat{\sigma} = \begin{bmatrix} \sigma_{11} \\ \sigma_{22} \\ \sigma_{33} \\ \sigma_{12} \\ \sigma_{23} \\ \sigma_{31} \end{bmatrix}, \qquad \hat{\varepsilon} = \begin{bmatrix} \varepsilon_{11} \\ \varepsilon_{22} \\ \varepsilon_{33} \\ 2 \varepsilon_{12} \\ 2 \varepsilon_{23} \\ 2 \varepsilon_{31} \end{bmatrix} \] 와 같이 열벡터로 다룬다. 변분이나 미분 등의 파생량에도 같은 규칙을 적용한다. **변형률 쪽의 전단 성분에는 계수 2가 붙는다**는 점에 주의한다(응력 쪽에는 붙지 않는다). 이 비대칭성에 의해 텐서 내적은 벡터 내적으로

 \[ \boldsymbol{\sigma} : \delta\boldsymbol{\varepsilon} = \sigma_{ij}\,\delta \varepsilon_{ij} = \sum_{i=j} \sigma_{ij}\,\delta \varepsilon_{ij} + 2\sum_{i<j} \sigma_{ij}\,\delta \varepsilon_{ij} = \hat{\sigma}^T\,\delta\hat{\varepsilon} \] 의 형태로 간결하게 쓸 수 있다. 구현에서는 전단 변형률 성분의 2배 계수를 잘못 처리하기 쉬우므로 항상 "응력에는 없음/변형률에는 있음" 규약을 의식해야 한다. 기준 배치의 양(제2 Piola-Kirchhoff 응력 \(\boldsymbol{S}\)와 Green-Lagrange 변형률 \(\boldsymbol{E}\))에도 완전히 동일한 Voigt 규약을 적용한다.

 
### 4계 텐서의 행렬 표기

 응력과 변형률의 선형 관계 \(\boldsymbol{\sigma} = \boldsymbol{C} : \boldsymbol{\varepsilon}\)(성분 표기로 \(\sigma_{ij} = C_{ijkl} \varepsilon_{kl}\))에 대해 \(\boldsymbol{\varepsilon}\)의 대칭성을 이용하면

 \[ \sigma_{ij} = \tilde{C}_{ijkl} \varepsilon_{kl}, \qquad \tilde{C}_{ijkl} = \frac{1}{2}\bigl( C_{ijkl} + C_{ijlk} \bigr) \] 로 다시 쓸 수 있다. 3차원인 경우 \(\tilde{C}\)를 6×6 재료 행렬 \(\hat{\tilde{C}}\)로 배열하면 원래 텐서 관계는

 \[ \hat{\sigma} = \hat{\tilde{C}}\, \hat{\varepsilon} \] 로 나타낼 수 있다. 이후 이 매뉴얼에서는 특별한 언급이 없는 한 재료 행렬을 \(D\)(또는 \(\hat{C}\))로 표기한다. 구체적인 등방 선형 탄성의 성분은 [선형 탄성](../constitutive/01_linear_elastic.html)을 참조한다.

 
## 미분 연산자 규약

 
### 대칭 기울기 연산자

 벡터장 \(\boldsymbol{u}\)에 대한 **대칭 기울기 연산자** \(\nabla_S\)를

 \[ \nabla_S \boldsymbol{u} = \frac{1}{2}\bigl( \nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T \bigr) \] 로 정의한다. 성분 표기로는 \((\nabla_S \boldsymbol{u})_{ij} = \tfrac{1}{2}(\partial u_i / \partial x_j + \partial u_j / \partial x_i)\)이다. 미소변형의 변형률 \(\boldsymbol{\varepsilon} = \nabla_S \boldsymbol{u}\)는 이 연산자로 간결하게 나타낼 수 있다. 유한변형에서 기울기를 취하는 배치(기준/현재)의 구분은 [운동·변형·변형률](../governing/01a_kinematics.html)에서 다룬다.

 
### 물질 시간 미분

 어떤 양 \(A\)의 물질 시간 미분(동일한 물질점을 따라간 시간 미분)을 위첨자 점으로 \(\dot{A}\)라 나타낸다:

 \[ \dot{A} \equiv \frac{D A}{D t}. \] 속도와 가속도는 이 규약을 따른다.

 
## 관련 항목

 
- [운동·변형·변형률](../governing/01a_kinematics.html) — 기준 배치/현재 배치의 개념과 배치에 의존하는 기호 규약(대문자/소문자, 물질 미분/공간 미분)
- [응력과 보존 법칙](../governing/01b_stress_balance.html) — 이 표기법을 사용한 지배방정식의 유도
- [가상일의 원리](../governing/01c_virtual_work.html) — 약형과 Voigt 표기의 구분
- [선형 탄성](../constitutive/01_linear_elastic.html) — 재료 행렬 \(D\)의 구체적인 형태
- [형상함수와 유한요소 근사](../discretization/01_shape_functions.html) — Voigt 표기에 의한 요소 강성 계산


---


<!-- ========================================== -->
<!-- FILE: theory/notation/02_symbol_list.md | TITLE: 물리량 기호 목록 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/notation/02_symbol_list.html -->

# 물리량 기호 목록

 이 페이지는 FrontISTR 이론 매뉴얼에서 사용하는 물리량의 기호를 일람한다. 표기 규칙(벡터·텐서의 굵은 글씨 표기, Einstein 축약 규약, Voigt 표기)에 대해서는 [텐서 표기법과 수학적 기초](01_tensor_notation.html)를 참조하기 바란다.

 
---

 
## 배치·좌표계

 
| 기호 | 설명 |
| --- | --- |
| \(\boldsymbol{X}\) | 물질점의 기준 배치(초기 배치)에서의 위치 벡터(물질좌표) |
| \(\boldsymbol{x}\) | 물질점의 현재 배치에서의 위치 벡터(공간좌표) |
| \(\phi(\boldsymbol{X}, t)\) | 운동 사상: \(\boldsymbol{x} = \phi(\boldsymbol{X}, t)\) |
| \(\Omega_0\) | 기준 배치에서 물체가 차지하는 영역 |
| \(\Omega\) | 현재 배치에서 물체가 차지하는 영역 |
| \(\Gamma_0\) | \(\Omega_0\)의 경계 |
| \(\Gamma\) | \(\Omega\)의 경계 |
| \(\Gamma_B, \Gamma_{0B}\) | 기하학적(변위) 경계: \(\boldsymbol{u} = \bar{\boldsymbol{u}}\)가 규정되는 현재 배치·기준 배치의 경계 |
| \(\Gamma_t, \Gamma_{0t}\) | 역학적 경계: \(\boldsymbol{\sigma}\boldsymbol{n} = \bar{\boldsymbol{t}}\) (또는 \(\boldsymbol{P}\boldsymbol{N} = \bar{\boldsymbol{t}}\))가 규정되는 현재 배치·기준 배치의 경계 |
| \(\boldsymbol{n}\) | 현재 배치에서 면의 바깥쪽 단위 법선 벡터 |
| \(\boldsymbol{N}\) | 기준 배치에서 면의 바깥쪽 단위 법선 벡터 |
| \(t\) | 시각 |

 **규약**: 기준 배치의 물리량에는 대문자 또는 아래첨자 \(0\)을, 현재 배치의 물리량에는 소문자를 할당한다.

 
---

 
## 시각과 증분(증분 해석)

 
| 기호 | 설명 |
| --- | --- |
| \(\Delta t\) | 시각 증분: \(\Delta t = t_{n+1} - t_n\) |
| \(^{t}(\cdot)\) | 시각 \(t\)에서의 물리량(앞쪽 위첨자). 예: \(^{t}\boldsymbol{\sigma}\)는 시각 \(t\)에서의 Cauchy 응력 |
| \(^{t}\Omega\) | 시각 \(t\)에서의 현재 배치 영역 |
| \(^{t}\Gamma\) | 시각 \(t\)에서의 현재 배치 경계 |

 **규약**: 증분 해석에서는 시각 \(t\)까지의 상태를 기지값, 시각 \(t + \Delta t\)의 상태를 미지값으로 하여 구한다. 시각을 명시할 필요가 없는 경우에는 앞쪽 위첨자를 생략한다. 참조 배치로 기준 배치 \(\Omega_0\)을 선택하는 정식화를 Total Lagrange 법, 증분 시작 시의 현재 배치 \(^{t}\Omega\)를 선택하는 정식화를 Updated Lagrange 법이라고 한다.

 
---

 
## 변위·속도·가속도·체적력

 
| 기호 | 설명 |
| --- | --- |
| \(\boldsymbol{u}\) | 변위 벡터: \(\boldsymbol{u} = \boldsymbol{x} - \boldsymbol{X}\) |
| \(\bar{\boldsymbol{u}}\) | 기하학적 경계 \(\Gamma_B\)에서 규정되는 변위 |
| \(\delta \boldsymbol{u}\) | 가상 변위(시험함수): \(\Gamma_B\)에서 \(\delta \boldsymbol{u} = \boldsymbol{0}\) |
| \(\boldsymbol{v}\) | 속도 벡터: \(\boldsymbol{v} = \dot{\boldsymbol{u}}\) |
| \(\boldsymbol{a}\) | 가속도 벡터: \(\boldsymbol{a} = \dot{\boldsymbol{v}}\) |
| \(\boldsymbol{g}\) | 체적력(단위 질량당) |
| \(\boldsymbol{t}\) | 표면력 벡터(단위 면적당): \(\boldsymbol{t} = \boldsymbol{\sigma}\boldsymbol{n}\) |
| \(\bar{\boldsymbol{t}}\) | 역학적 경계 \(\Gamma_t\)에서 규정되는 표면력(Neumann 데이터) |

 
---

 
## 미분 연산자

 
| 기호 | 설명 |
| --- | --- |
| \(\dot{(\cdot)}\) | 물질 시간 미분(동일한 물질점을 추적한 시간 미분): \(\dot{A} \equiv DA/Dt\) |
| \(\nabla_X\) | 기준 배치 구배(물질좌표 \(\boldsymbol{X}\)에 관한 구배): \(\nabla_X = \partial / \partial \boldsymbol{X}\) |
| \(\nabla_x\) | 현재 배치 구배(공간좌표 \(\boldsymbol{x}\)에 관한 구배): \(\nabla_x = \partial / \partial \boldsymbol{x}\) |
| \(\nabla_S\) | 대칭 구배 연산자: \(\nabla_S \boldsymbol{u} = \tfrac{1}{2}(\nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T)\) |

 **비고**: \(\nabla_X\)와 \(\nabla_x\)는 \(\nabla_X = \boldsymbol{F}^T \nabla_x\)로 관계지어진다. 아래첨자가 없는 \(\nabla\)는 문맥으로 배치를 판별하거나, 미소 변형 시와 같이 구별이 사라지는 경우에 사용한다.

 
---

 
## 변형 텐서

 
| 기호 | 설명 |
| --- | --- |
| \(\boldsymbol{F}\) | 변형 구배 텐서: \(F_{ij} = \partial x_i / \partial X_j\) |
| \(J\) | 체적 변화비: \(J = dv/dV = \det \boldsymbol{F}\) |
| \(\boldsymbol{C}\) | 오른쪽 Cauchy-Green 변형 텐서: \(\boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}\) |
| \(\boldsymbol{b}\) | 왼쪽 Cauchy-Green 변형 텐서: \(\boldsymbol{b} = \boldsymbol{F}\boldsymbol{F}^T\) |
| \(\boldsymbol{L}\) | 속도 구배 텐서: \(L_{ij} = \partial v_i / \partial x_j = \dot{F}_{ik}F^{-1}_{kj}\) |
| \(\boldsymbol{D}\) | 변형 속도 텐서(\(\boldsymbol{L}\)의 대칭 부분): \(\boldsymbol{D} = \tfrac{1}{2}(\boldsymbol{L}+\boldsymbol{L}^T)\) |
| \(\boldsymbol{W}\) | 스핀 텐서(\(\boldsymbol{L}\)의 반대칭 부분): \(\boldsymbol{W} = \tfrac{1}{2}(\boldsymbol{L}-\boldsymbol{L}^T)\) |

 
---

 
## 변형률 텐서

 
| 기호 | 설명 |
| --- | --- |
| \(\boldsymbol{E}\) | Green-Lagrange 변형률 텐서(기준 배치): \(\boldsymbol{E} = \tfrac{1}{2}(\boldsymbol{C}-\boldsymbol{I})\) |
| \(\boldsymbol{e}\) | Almansi 변형률 텐서(현재 배치): \(\boldsymbol{e} = \tfrac{1}{2}(\boldsymbol{I}-\boldsymbol{b}^{-1})\) |
| \(\boldsymbol{A}_{(L)}\) | Almansi 변형률 텐서의 선형 부분(현재 배치의 대칭 변위 구배): \(\boldsymbol{A}_{(L)} = \tfrac{1}{2}(\nabla_x \boldsymbol{u} + (\nabla_x \boldsymbol{u})^T)\) |
| \(\boldsymbol{\varepsilon}\) | 미소 변형률 텐서(선형 근사): \(\varepsilon_{ij} = \tfrac{1}{2}(\partial u_i/\partial x_j + \partial u_j/\partial x_i)\) |

 **비고**: \(\boldsymbol{E}\)는 Total Lagrange 법, \(\boldsymbol{D}\)는 Updated Lagrange 법에서 각각 주로 사용된다.

 
---

 
## 응력 텐서

 
| 기호 | 설명 |
| --- | --- |
| \(\boldsymbol{\sigma}\) | Cauchy 응력 텐서(진응력, 현재 배치): \(d\boldsymbol{f} = \boldsymbol{\sigma}\boldsymbol{n}\,d\Gamma\) |
| \(\boldsymbol{P}\) | 제1 Piola-Kirchhoff 응력 텐서(공칭 응력): \(d\boldsymbol{f} = \boldsymbol{P}\boldsymbol{N}\,d\Gamma_0\) |
| \(\boldsymbol{S}\) | 제2 Piola-Kirchhoff 응력 텐서(기준 배치, 대칭): \(\boldsymbol{F}^{-1}d\boldsymbol{f} = \boldsymbol{S}\boldsymbol{N}\,d\Gamma_0\) |

 각 응력 텐서 사이의 변환 관계:

 \[ \boldsymbol{P} = \boldsymbol{F}\boldsymbol{S}, \qquad \boldsymbol{\sigma} = \frac{1}{J}\boldsymbol{F}\boldsymbol{S}\boldsymbol{F}^T = \frac{1}{J}\boldsymbol{P}\boldsymbol{F}^T \] **규약**: Total Lagrange 법에서는 \((\boldsymbol{S}, \boldsymbol{E})\), Updated Lagrange 법에서는 \((\boldsymbol{\sigma}, \boldsymbol{D})\)의 쌍을 사용한다.

 
---

 
## 밀도·질량

 
| 기호 | 설명 |
| --- | --- |
| \(\rho\) | 현재 배치에서의 질량 밀도 |
| \(\rho_0\) | 기준 배치에서의 질량 밀도 |

 질량 보존법칙: \(\rho_0 = J\rho\).

 
---

 
## 재료 상수(선형 탄성체)

 
| 기호 | 설명 |
| --- | --- |
| \(E\) | 영률 |
| \(\nu\) | 포아송비 |
| \(\lambda\) | Lamé 제1상수: \(\lambda = E\nu / [(1+\nu)(1-2\nu)]\) |
| \(\mu\) | Lamé 제2상수(전단탄성계수, 횡탄성계수): \(\mu = E / [2(1+\nu)]\) |
| \(\boldsymbol{\mathsf{C}}\) | 탄성 텐서(4계): \(\boldsymbol{S} = \boldsymbol{\mathsf{C}}:\boldsymbol{E}\), 성분 \(C_{ijkl}\) |
| \(D\)(또는 \(\hat{\tilde{C}}\)) | 재료 행렬(Voigt 표기, 3차원에서는 \(6\times6\)): \(\hat{\sigma} = D\,\hat{\varepsilon}\) |

 등방성 선형 탄성체의 탄성 텐서 성분:

 \[ C_{ijkl} = \lambda\,\delta_{ij}\delta_{kl} + \mu\,(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}) \] 
---

 
## 초탄성 재료

 
| 기호 | 설명 |
| --- | --- |
| \(W(\boldsymbol{C})\) | 탄성 포텐셜 함수(변형률 에너지 밀도 함수) |
| \(I_1, I_2, I_3\) | 오른쪽 Cauchy-Green 텐서 \(\boldsymbol{C}\)의 주불변량 |
| \(\tilde{I}_1, \tilde{I}_2, \tilde{I}_3\) | \(\boldsymbol{C}\)의 축소 불변량(체적 변화분을 분리) |
| \(C_1, C_2\) | Mooney-Rivlin 모델의 재료 상수 |
| \(D\) | 체적 탄성에 관한 재료 상수 |

 
---

 
## 탄소성 재료

 
| 기호 | 설명 |
| --- | --- |
| \(\boldsymbol{D}^e\) | 변형 속도 텐서의 탄성 성분 |
| \(\boldsymbol{D}^p\) | 변형 속도 텐서의 소성 성분 |
| \(F(\boldsymbol{\sigma}, \kappa)\) | 항복 함수 |
| \(\kappa\) | 소성 상태를 나타내는 내부 변수(등방 경화 변수) |
| \(\lambda^p\) | 소성 승수(소성 변형률 속도 승수): \(\lambda^p \geq 0\) |
| \(\Theta\) | 소성 포텐셜(연관 유동법칙에서는 \(\Theta = F\)) |
| \(\bar{\sigma}\) | 등가응력(von Mises 응력 등) |
| \(\bar{\varepsilon}^p\) | 등가소성변형률 |
| \(H(\bar{\varepsilon}^p)\) | 등방 경화 함수 |

 상보성 조건: \(\lambda^p F(\boldsymbol{\sigma}, \kappa) = 0\), \(\lambda^p \geq 0\), \(F \leq 0\).

 
---

 
## 유한요소법

 
| 기호 | 설명 |
| --- | --- |
| \(\Omega^h, \Omega_0^h\) | 유한요소 분할로 근사된 현재 배치·기준 배치의 영역: \(\Omega^h = \bigcup_e \Omega^e\) |
| \(\Omega^e, \Omega^e_0\) | 요소의 현재 배치·기준 배치에서의 영역 |
| \(\Gamma^e_t, \Gamma^e_{0t}\) | 요소 경계 중 역학적 경계에 속하는 부분 |
| \(\boldsymbol{r}\) | 자연좌표(요소의 로컬 좌표) |
| \(\boldsymbol{r}_\alpha\) | 절점 \(\alpha\)에 대응하는 자연좌표의 점 |
| \(N_\alpha^e(\boldsymbol{r})\) | 요소 \(e\)의 절점 \(\alpha\)에 대응하는 형상함수 |
| \(n_e\) | 요소의 구성 절점 수 |
| \(n_g\) | 전체 절점 수 |
| \(\boldsymbol{X}^e_\alpha, \boldsymbol{u}^e_\alpha\) | 요소 절점 \(\alpha\)의 좌표·변위 |
| \(\boldsymbol{X}^e, \boldsymbol{u}^e\) | 요소 절점 좌표·변위 벡터: \(\boldsymbol{X}^e = (\boldsymbol{X}^{eT}_1, \ldots, \boldsymbol{X}^{eT}_{n_e})^T\) 등 |
| \(\boldsymbol{X}^n, \boldsymbol{u}^n\) | 전체 절점 좌표·변위 벡터(절점 번호 → 자유도 순으로 배열) |
| \(\boldsymbol{B}\) | 변형률-변위 관계 행렬(B 행렬) |
| \(\boldsymbol{N}_\alpha, \boldsymbol{N}\) | 형상함수 행렬: \(\boldsymbol{N}_\alpha\)는 절점 \(\alpha\)의 형상함수 \(N_\alpha^e\)를 대각에 배열한 \(d \times d\) 블록, \(\boldsymbol{N} = [\boldsymbol{N}_1, \ldots, \boldsymbol{N}_{n_e}]\). \(\delta\boldsymbol{u} = \boldsymbol{N}\, \delta\boldsymbol{u}^e\) |
| \(\boldsymbol{K}^e\) | 요소 강성 행렬 |
| \(\boldsymbol{K}^e_X, \boldsymbol{K}^e_x\) | 요소 강성 행렬의 피적분항(기준 배치·현재 배치로 표기한 것): \(\boldsymbol{K}^e = \int_{\Omega^e_0} \boldsymbol{K}^e_X\, dV = \int_{\Omega^e} \boldsymbol{K}^e_x\, dv\) |
| \(\boldsymbol{Q}^e\)(TL법), \(\boldsymbol{q}^e\)(UL법) | 요소 내력 벡터 |
| \(\boldsymbol{F}^e_\alpha, \boldsymbol{F}^e\) | 요소 절점 \(\alpha\)에 작용하는 절점 외력 및 요소 절점 외력 벡터: \(\boldsymbol{F}^e = (\boldsymbol{F}^{eT}_1, \ldots, \boldsymbol{F}^{eT}_{n_e})^T\) |
| \(\boldsymbol{Q}, \boldsymbol{F}\) | 전체 내력 벡터·전체 외력 벡터: \(\boldsymbol{Q} = (\boldsymbol{Q}^T_1, \ldots, \boldsymbol{Q}^T_{n_g})^T\), \(\boldsymbol{F}\)도 동일 |
| \(\boldsymbol{Q}_{i_g}, \boldsymbol{F}_{i_g}\) | 전체 절점 \(i_g\)에서의 절점 내력·절점 외력(\(\boldsymbol{Q}, \boldsymbol{F}\)의 절점 블록) |
| \(\boldsymbol{K}\) | 전체 접선 강성 행렬: \(\boldsymbol{K} = \partial \boldsymbol{Q}/\partial \boldsymbol{u}\). \(3\times 3\) 블록 \(\boldsymbol{K}_{i_gi_h}\)를 \(i_g\)행 \(i_h\)열에 배열한 행렬 |
| \(\boldsymbol{R}_i\) | Newton-Raphson 반복에서의 잔차 벡터: \(\boldsymbol{R}_i = \boldsymbol{F} - \boldsymbol{Q}(\boldsymbol{u}_n + \Delta\boldsymbol{u})\) |
| \(d\boldsymbol{u}_i\) | Newton-Raphson 반복의 수정량(제 \(i\) 반복에서 선형방정식 \(\boldsymbol{K}_i\, d\boldsymbol{u}_i = \boldsymbol{R}_{i-1}\)을 풀어 얻음) |
| \(i_g\) | 전체 절점 번호(\(1 \leq i_g \leq n_g\)) |
| \(\mathrm{gdx}(e, \alpha)\) | 요소 \(e\)의 요소 내 절점 번호 \(\alpha\)에 대응하는 전체 절점 번호: \(\mathrm{gdx}(e, \alpha) = i_g\) |
| \(\mathcal{E}(i_g)\) | 전체 절점 번호 \(i_g\)에 대응하는 \((e, \alpha)\) 쌍의 집합: \(\mathcal{E}(i_g) = \{ (e, \alpha) \mid \mathrm{gdx}(e, \alpha) = i_g \}\). 조립 연산에 사용 |
| \(\mathcal{E}^2(i_g, i_h)\) | 전체 절점 번호 쌍 \((i_g, i_h)\)에 대응하는 \((e, \alpha, \beta)\) 조합의 집합: 강성 행렬 조립에 사용 |
| \(\alpha, \beta, \gamma, \ldots\) | 요소 구성 절점에 관한 첨자 |
| \(i, j, k, l, \ldots\) | 자유도에 관한 첨자(3차원에서는 \(1, 2, 3\)) |

 
---

 
## 고성능 요소·구조 요소

 
| 기호 | 설명 |
| --- | --- |
| \(\bar{\boldsymbol{B}}\) | B-bar 법에서 체적부가 수정된 변형률-변위 관계 행렬(요소 중심에서 평가한 B 행렬로 체적 성분을 치환) |
| \(\bar{\boldsymbol{F}}\) | F-bar 법에서 체적부가 수정된 변형 구배: \(\bar{\boldsymbol{F}} = (J_0/J)^{1/3} \boldsymbol{F}\) |
| \(J_0\) | 요소 중심 \(\boldsymbol{r}=\boldsymbol{0}\)에서 평가한 변형 구배의 체적비: \(J_0 = \det \boldsymbol{F}(\boldsymbol{0})\) |
| \(\boldsymbol{\alpha}\) | 비적합 모드 요소에서의 내부 자유도 벡터(요소 경계에서 연속성을 보장하지 않는 추가 변위 모드의 계수) |
| \(M_k(\boldsymbol{r})\) | 비적합 모드의 형상함수: \(M_1=1-\xi^2\), \(M_2=1-\eta^2\), \(M_3=1-\zeta^2\) |
| \(h\) | 셸 요소의 판 두께 |
| \(A\) | 보 요소 단면의 단면적 |
| \(I\) | 보 요소 단면의 단면 2차 모멘트(로컬 \(y\), \(z\)축에 대해서는 \(I_y\), \(I_z\)로 표기) |
| \(G\) | 전단탄성계수: \(G = E/[2(1+\nu)]\) |
| \(L\) | 보 요소의 요소 길이 |
| \(A_s\) | 보 요소 단면의 전단 면적(FrontISTR에서는 \(A_s = A\)) |
| \(\Phi_y, \Phi_z\) | 티모셴코 보의 전단 변형 기여를 나타내는 계수: \(\Phi_y = 12EI_z/(GA_sL^2)\), \(\Phi_z = 12EI_y/(GA_sL^2)\) |

 
---

 
## 관련 항목

 
- [텐서 표기법과 수학적 기초](01_tensor_notation.html) — 굵은 글씨 표기·Einstein 축약 규약·Voigt 표기의 규칙
- [운동·변형·변형률](../governing/01a_kinematics.html) — 배치·변형 구배·변형률의 정의
- [운동·변형·변형률](../governing/01a_kinematics.html) — 지배방정식의 유도
- [선형 탄성](../constitutive/01_linear_elastic.html) — 재료 행렬 \(D\)의 구체적인 성분


---


<!-- ========================================== -->
<!-- FILE: theory/parallel/01_parallel.md | TITLE: 병렬 유한요소법 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/parallel/01_parallel.html -->

# 병렬 유한요소법

 TBD

 
## 관련 항목

 
- [분할](../../functions_reference/10_partition.html) — 병렬화 방법의 선택지와 사용 구분
- [솔버·전처리](../../functions_reference/06_solver.html) — 병렬 솔버 선택


---


<!-- ========================================== -->
<!-- FILE: theory/solution_methods/01_newton_raphson.md | TITLE: Newton-Raphson 법 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/solution_methods/01_newton_raphson.html -->

# Newton-Raphson 법

 
## 선형화와 반복 점화식

 [외력의 가상일과 전체 방정식의 조립](../discretization/04_external_force_and_assembly.html)에서 얻은, 시각 \(t_{n+1}\)에서의 절점 변위 \(\boldsymbol{u}_{n+1}\)에 관한 비선형 방정식을 Newton-Raphson 법으로 푼다. 시각 \(t_n\)까지의 절점 변위 \(\boldsymbol{u}_n\)은 알려져 있다고 하고, 변위 증분 \(\Delta\boldsymbol{u}\)를 미지수로 하여

 \[ \boldsymbol{u}_{n+1} = \boldsymbol{u}_n + \Delta\boldsymbol{u} \] 를 구한다. 이후에는 외력 벡터의 절점 변위 의존성을 무시하고, \(\boldsymbol{F}(\boldsymbol{u}_{n+1}) = \boldsymbol{F}_{n+1}\)로 두어

 \[ \boldsymbol{Q}(\boldsymbol{u}_n + \Delta\boldsymbol{u}) - \boldsymbol{F}_{n+1} = \boldsymbol{0} \] 를 푼다.

 현재 해 \(\Delta\boldsymbol{u}\)에서의 **접선 강성**

 \[ \boldsymbol{K} = \left. \frac{\partial \boldsymbol{Q}}{\partial \boldsymbol{u}} \right|_{\boldsymbol{u}_n + \Delta\boldsymbol{u}} \] 을 사용하여 비선형 방정식을 선형화하면

 \[ \boldsymbol{K}\, d\boldsymbol{u} + \boldsymbol{Q}(\boldsymbol{u}_n + \Delta\boldsymbol{u}) - \boldsymbol{F}_{n+1} = \boldsymbol{0} \] 이 된다. 제 \(i\) 반복의 수정량을 \(d\boldsymbol{u}_i\), 반복 시작 시의 잔차 벡터를

 \[ \boldsymbol{R}_{i-1} = \boldsymbol{F}_{n+1} - \boldsymbol{Q}(\boldsymbol{u}_n + \Delta\boldsymbol{u}) \] 라고 쓰면, 반복 점화식은

 \[ \boldsymbol{K}_i\, d\boldsymbol{u}_i = \boldsymbol{R}_{i-1}, \qquad \Delta\boldsymbol{u} \leftarrow \Delta\boldsymbol{u} + d\boldsymbol{u}_i \] 의 형태가 된다. 잔차 \(\boldsymbol{R}_i\)는 평형 상태에서 벗어난 힘의 불균형에 해당하는 양이다.

 
## 접선 강성 행렬의 구성

 접선 강성 \(\boldsymbol{K} = \partial\boldsymbol{Q}/\partial\boldsymbol{u}\)는 [내력의 가상일의 이산화](../discretization/03_internal_force.html)에서 얻은 요소 내력 벡터를 절점 변위로 편미분하고, 요소별 피적분항을 요소 영역에서 적분한 것을 모아 구성한다. 요소 수준의 피적분항을 \(\boldsymbol{K}^e_X\)(기준 배치 표기, TL 법) 또는 \(\boldsymbol{K}^e_x\)(현재 배치 표기, UL 법)라고 쓰면, 요소 접선 강성은

 \[ \boldsymbol{K}^e = \int_{\Omega^e_0} \boldsymbol{K}^e_X\, dV \quad (\text{TL}), \qquad \boldsymbol{K}^e = \int_{\Omega^e} \boldsymbol{K}^e_x\, dv \quad (\text{UL}) \] 로 주어진다. 다음에 TL/UL 피적분항의 최종 형태를 나타낸다. 어느 경우나 **재료 강성 항**(초기 변위 항)과 **기하 강성 항**(초기 응력 항)의 합으로 분해된다.

 
### Total Lagrange 법

 Total Lagrange 법에서는 제2 Piola-Kirchhoff 응력률 \(\dot{\boldsymbol{S}}\)와 Green-Lagrange 변형률률 \(\dot{\boldsymbol{E}}\)의 선형 관계 \(\dot{\boldsymbol{S}} = \boldsymbol{\mathsf{C}}:\dot{\boldsymbol{E}}\)를 가정한다. 이는 선형 탄성체(St.Venant-Kirchhoff 체)와 초탄성체의 구성 법칙에 대응하며, FrontISTR에서는 Total Lagrange 법을 이러한 재료에 사용한다. 이때 요소 접선 강성의 피적분항은 텐서 형식으로

 \[ \delta\boldsymbol{u}^{eT}\, \boldsymbol{K}^e_X\, \dot{\boldsymbol{u}}^e = \dot{\boldsymbol{S}}:\delta\boldsymbol{E} + \boldsymbol{S}:(\delta\boldsymbol{F}^T \dot{\boldsymbol{F}}) \] 라고 쓸 수 있다. 우변의 제1항이 **재료 강성 항**(초기 변위 항), 제2항이 **기하 강성 항**(초기 응력 항)이다.

 FrontISTR 구현에서는 이 피적분항을 Voigt 표기를 사용한 행렬 형식

 \[ \boldsymbol{K}^e_X = (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^T\, \tilde{\boldsymbol{C}}\, (\boldsymbol{B}_L + \boldsymbol{B}_{NL}) + \boldsymbol{F}_9^T\, \boldsymbol{S}_9\, \boldsymbol{F}_9 \] 으로 계산한다. 각 행렬은 다음과 같다. \(\boldsymbol{B}_L, \boldsymbol{B}_{NL}\)은 [내력의 가상일의 이산화](../discretization/03_internal_force.html)에서 도입한 B 행렬이며, \(\tilde{\boldsymbol{C}}\)는 구성 텐서 \(\boldsymbol{\mathsf{C}}\)를 Voigt 표기로 나타낸 \(6\times 6\) 재료 강성 행렬([텐서 표기와 수학적 기초](../notation/01_tensor_notation.html))이다. \(\boldsymbol{S}_9, \boldsymbol{F}_9\)는 기하 강성 항을 행렬곱으로 나타내기 위해 사용하는 다음의 재배열 행렬이다. 먼저 \(3\times 3\)의 2계 텐서 \(\boldsymbol{A}\)를 9성분 벡터로 재배열하는 기호 \([\,\cdot\,]\)를

 \[ [\boldsymbol{A}] = (A_{11}, A_{21}, A_{31}, A_{12}, A_{22}, A_{32}, A_{13}, A_{23}, A_{33})^T \] 로 정의하면, \(\boldsymbol{F}_9\)는 변형 구배의 변분을 \([\delta\boldsymbol{F}] = \boldsymbol{F}_9\, \delta\boldsymbol{u}^e\)의 형태로 나타내는 \(9\times d n_e\) 행렬이며, 요소 절점 \(\alpha = 1, \ldots, n_e\)에 대응하는 \(9\times d\) 블록

 \[ [\boldsymbol{F}_9]_\alpha = \begin{bmatrix} (\partial N_\alpha^e/\partial X_1)\, \boldsymbol{I} \\ (\partial N_\alpha^e/\partial X_2)\, \boldsymbol{I} \\ (\partial N_\alpha^e/\partial X_3)\, \boldsymbol{I} \end{bmatrix} \qquad (\boldsymbol{I} \text{는 } 3\times 3 \text{ 단위행렬}) \] 을 요소 절점 순서대로 가로로 배열한 \(\boldsymbol{F}_9 = [[\boldsymbol{F}_9]_1, \ldots, [\boldsymbol{F}_9]_{n_e}]\)로 주어진다. \(\boldsymbol{S}_9\)는 이것과 조합하여 기하 강성 항이 \(\delta\boldsymbol{u}^{eT}\, \boldsymbol{F}_9^T \boldsymbol{S}_9 \boldsymbol{F}_9\, \dot{\boldsymbol{u}}^e\)로 표현되도록 선택된 \(9\times 9\) 행렬

 \[ \boldsymbol{S}_9 = \begin{bmatrix} S_{11} \boldsymbol{I} & S_{12} \boldsymbol{I} & S_{13} \boldsymbol{I} \\ S_{21} \boldsymbol{I} & S_{22} \boldsymbol{I} & S_{23} \boldsymbol{I} \\ S_{31} \boldsymbol{I} & S_{32} \boldsymbol{I} & S_{33} \boldsymbol{I} \end{bmatrix} \] 이다.

 
### Updated Lagrange 법

 Updated Lagrange 법에서는 상대 Kirchhoff 응력 텐서의 Jaumann 응력률 \(\hat{\boldsymbol{\sigma}}^{\nabla J}\)와 변형률 속도 텐서 \(\boldsymbol{D}\)의 선형 관계 \(\hat{\boldsymbol{\sigma}}^{\nabla J} = \boldsymbol{\mathsf{C}}:\boldsymbol{D}\)를 가정한다. 이는 선형 탄성체·탄소성체·크리프 재료에 공통되는 hypoelastic 구성 법칙의 형식이며, FrontISTR에서는 Updated Lagrange 법을 이러한 재료에 사용한다. 이때 현재 배치로 나타낸 요소 접선 강성의 피적분항은 텐서 형식으로

 \[ \delta\boldsymbol{u}^{eT}\, \boldsymbol{K}^e_x\, \dot{\boldsymbol{u}}^e = \boldsymbol{\sigma}^{\nabla T}:\delta\boldsymbol{A}_{(L)} + \boldsymbol{\sigma}:(\delta\boldsymbol{F}_t^T\, \boldsymbol{L}) \] 라고 쓸 수 있다(\(\boldsymbol{\sigma}^{\nabla T}\)는 Truesdell 응력률, \(\boldsymbol{A}_{(L)}\)은 Almansi 변형률의 선형부, \(\boldsymbol{F}_t = \partial\boldsymbol{u}/\partial\boldsymbol{x}\)는 현재 배치를 기준으로 한 변위 구배, \(\boldsymbol{L}\)은 속도 구배 텐서). 우변의 제1항이 **재료 강성 항**, 제2항이 **기하 강성 항**이다.

 FrontISTR 구현에서는 이 피적분항을 Voigt 표기를 사용한 행렬 형식

 \[ \boldsymbol{K}^e_x = \boldsymbol{b}^T\, (\tilde{\boldsymbol{C}} - \boldsymbol{G})\, \boldsymbol{b} + \boldsymbol{f}_9^T\, \boldsymbol{\sigma}_9\, \boldsymbol{f}_9 \] 으로 계산한다. \(\boldsymbol{b}\)는 현재 배치에서 구성한 B 행렬([내력의 가상일의 이산화](../discretization/03_internal_force.html))이다. \(\boldsymbol{\sigma}_9, \boldsymbol{f}_9\)는 TL 법에서 정의한 \(\boldsymbol{S}_9, \boldsymbol{F}_9\)에서 제2 PK 응력 \(\boldsymbol{S}\)를 Cauchy 응력 \(\boldsymbol{\sigma}\)로, 기준 배치 구배 \(\partial N_\alpha^e/\partial X_i\)를 현재 배치 구배 \(\partial N_\alpha^e/\partial x_i\)로 치환한 것이다.

 \(\boldsymbol{G}\)는 hypoelastic 구성 법칙 \(\hat{\boldsymbol{\sigma}}^{\nabla J} = \boldsymbol{\mathsf{C}}:\boldsymbol{D}\)를 Truesdell 응력률 기반의 구성 법칙으로서 접선 강성의 틀과 일치시키는 데 필요한 Cauchy 응력 의존 보정 행렬이며, 4계 텐서 성분 \(G_{ijkl} = \delta_{il}\sigma_{kj} + \delta_{jl}\sigma_{ik}\)를 \(6\times 6\) Voigt 표기로 배열한

 \[ \boldsymbol{G} = \begin{bmatrix} 2\sigma_{11} & 0 & 0 & \sigma_{12} & 0 & \sigma_{31} \\ 0 & 2\sigma_{22} & 0 & \sigma_{12} & \sigma_{23} & 0 \\ 0 & 0 & 2\sigma_{33} & 0 & \sigma_{23} & \sigma_{31} \\ \sigma_{12} & \sigma_{12} & 0 & \tfrac{\sigma_{11}+\sigma_{22}}{2} & \tfrac{\sigma_{12}}{2} & \tfrac{\sigma_{23}}{2} \\ 0 & \sigma_{23} & \sigma_{23} & \tfrac{\sigma_{12}}{2} & \tfrac{\sigma_{22}+\sigma_{33}}{2} & \tfrac{\sigma_{12}}{2} \\ \sigma_{31} & 0 & \sigma_{31} & \tfrac{\sigma_{23}}{2} & \tfrac{\sigma_{12}}{2} & \tfrac{\sigma_{33}+\sigma_{11}}{2} \end{bmatrix} \] 이다.

 
### 전체 강성 행렬의 조립

 전체 접선 강성 \(\boldsymbol{K}\)는 요소 강성 \(\boldsymbol{K}^e\)를 절점별 \(d\times d\) 블록 \(\boldsymbol{K}^e_{\alpha\beta}\)로 나누고, [요소 절점 물리량의 조립](../discretization/04_external_force_and_assembly.html#요소-절점-물리량의-조립)에서 도입한 2계 텐서용 조립 집합 \(\mathcal{E}^2(i_g, i_h)\)를 사용하여

 \[ \boldsymbol{K}_{i_gi_h} = \sum_{(e,\alpha,\beta) \in \mathcal{E}^2(i_g, i_h)} \boldsymbol{K}^e_{\alpha\beta} \] 를 \(i_g\)행 \(i_h\)열에 배치한 행렬로 얻어진다. 구현에서는 집합 \(\mathcal{E}^2\)를 명시적으로 만들지 않고, 요소 루프 안에서 대응하는 블록에 직접 더한다. 절점당 자유도 \(\times\) 전체 절점 수 \(n_g\)차의 정방행렬이 되지만, 요소를 통해 연결된 절점 사이 이외의 성분은 \(0\)이므로 희소 행렬 형식으로 저장된다.

 TL 법과 UL 법의 요소 강성은 기준 배치의 전환(절점 좌표와 B 행렬의 구성 원천)과 \(\boldsymbol{G}\) 행렬의 유무를 제외하면 같은 형태이므로, FrontISTR에서는 양자를 공통 서브루틴으로 구현한다.

 
## 반복 알고리즘

 이상을 정리하면, 반복 초기에는 \(\Delta\boldsymbol{u} = \boldsymbol{0}\)으로 두고 초기 잔차 \(\boldsymbol{R}_0 = \boldsymbol{F}_{n+1} - \boldsymbol{Q}(\boldsymbol{u}_n)\)를 구한 다음, 제 \(i\) 반복에서 다음 절차를 수행한다.

 
1. 현재 변위 \(\boldsymbol{u}_n + \Delta\boldsymbol{u}\)에서의 접선 강성 \(\boldsymbol{K}_i = \partial\boldsymbol{Q}/\partial\boldsymbol{u}|_{\boldsymbol{u}_n + \Delta\boldsymbol{u}}\)를 [접선 강성 행렬의 구성](#접선-강성-행렬의-구성)의 절차로 계산한다.
2. 기하학적 경계 조건을 반영하기 위해 변위 구속이 부과된 자유도에 대해 접선 강성 행렬과 잔차 벡터를 변형하여 \(\tilde{\boldsymbol{K}}_i, \tilde{\boldsymbol{R}}_{i-1}\)를 얻는다([기하학적 경계 조건의 처리](06_geometric_bc.html)).
3. 선형 방정식 \(\tilde{\boldsymbol{K}}_i\, d\boldsymbol{u}_i = \tilde{\boldsymbol{R}}_{i-1}\)를 풀어 수정량 \(d\boldsymbol{u}_i\)를 구한다. 이 절차는 반복 계산에서 계산 부하의 대부분을 차지하는 경우가 많다.
4. 변위 증분을 \(\Delta\boldsymbol{u} \leftarrow \Delta\boldsymbol{u} + d\boldsymbol{u}_i\)로 갱신하고, 이에 따라 내력 벡터 \(\boldsymbol{Q}(\boldsymbol{u}_n + \Delta\boldsymbol{u})\) 및 잔차 \(\boldsymbol{R}_i = \boldsymbol{F}_{n+1} - \boldsymbol{Q}(\boldsymbol{u}_n + \Delta\boldsymbol{u})\)를 계산한다.
5. 수렴 판정을 수행하고, 수렴하면 반복을 종료한다. 잔차 \(\boldsymbol{R}_i\) 중 기하학적 경계 조건이 부과된 자유도에는 구속 반력에 해당하는 성분이 나타나므로, 이들을 제외한 성분 \(\tilde{\boldsymbol{R}}_i\)로 판정 지표를 구성한다. 구체적인 판정 지표와 임계값은 [수렴 판정](01b_convergence.html)에서 다룬다. 수렴하지 않고 반복 상한에 도달한 경우에는 반복 실패로 한다.

 반복이 수렴한 시점의 \(\Delta\boldsymbol{u}\)를 \(\boldsymbol{u}_n\)에 더하여 시각 \(t_{n+1}\)의 누적 변위 \(\boldsymbol{u}_{n+1} = \boldsymbol{u}_n + \Delta\boldsymbol{u}\)로 하고, 다음 시간 단계로 진행한다.

 
## 관련 항목

 
- [외력의 가상일과 전체 방정식의 조립](../discretization/04_external_force_and_assembly.html) — 풀어야 할 비선형 방정식의 출발점
- [내력의 가상일의 이산화](../discretization/03_internal_force.html) — \(\boldsymbol{B}_L, \boldsymbol{B}_{NL}, \boldsymbol{b}\)의 구성
- [기하학적 경계 조건의 처리](06_geometric_bc.html) — 변위 구속을 반영한 접선 강성 행렬·잔차 벡터의 변형
- [수렴 판정](01b_convergence.html) — 잔차 노름에 의한 정지 조건
- [텐서 표기와 수학적 기초](../notation/01_tensor_notation.html) — 재료 행렬 \(\tilde{\boldsymbol{C}}\)의 Voigt 표기
- [비선형 반복·시간 적분(기능)](../../functions_reference/06b_solution_algorithm.html) — 기능 레퍼런스에서의 용도별 선택


---


<!-- ========================================== -->
<!-- FILE: theory/solution_methods/01b_convergence.md | TITLE: 수렴 판정 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/solution_methods/01b_convergence.html -->

# 수렴 판정

 TBD

 
## 관련 항목

 
- [Newton-Raphson 법](01_newton_raphson.html) — 반복 본체
- [비선형 반복·시간 적분(기능)](../../functions_reference/06b_solution_algorithm.html) — 지표의 구분 사용과 기본값


---


<!-- ========================================== -->
<!-- FILE: theory/solution_methods/02_dynamic.md | TITLE: 동적 해석 기법 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/solution_methods/02_dynamic.html -->

# 동적 해석 기법

 본 절에서는 직접 시간 적분법을 적용한 동적 문제 해석 기법을 설명한다. 아래에 암시적 방법과 명시적 방법의 정식화를 나타낸다.

 
## 운동방정식의 이산화(공통 틀)

 TBD(다음 단계에서 본문 작성).

 
## 암시적 방법(Newmark-β 법)

 동적 문제를 대상으로, 아래 식에 나타낸 운동방정식의 해법에 직접 시간 적분법을 적용하였다.

 \[\begin{equation} M( t + \Delta t ) \ddot{U} (t + \Delta t) + C( t + \Delta t ) \dot{U}(t + \Delta t) + Q( t + \Delta t ) = F( t + \Delta t ) \label{eq:2.5.1} \end{equation}\] 여기서 \(M\)은 질량 행렬, \(C\)는 감쇠 행렬, \(Q\)는 내력 벡터, \(F\)는 외력 벡터이다. 또한 질량 행렬은 비선형 해석에서도 변형과 무관하게 일정하다고 한다.

 시간 증분 \(\Delta t\) 내에서의 변위, 속도 및 가속도의 변화는 Newmark-\(\beta\) 법을 사용하여 식 \(\eqref{eq:2.5.2}\) 및 식 \(\eqref{eq:2.5.3}\)과 같이 근사한다.

 \[\begin{equation} \dot{U}(t + \Delta t) = \frac{\gamma}{\beta \Delta t} \Delta U( t + \Delta t ) - \frac{\gamma - \beta}{\beta} \dot{U}( t ) - \Delta t \frac{\gamma - 2\beta}{2\beta} \ddot{U}(t) \label{eq:2.5.2} \end{equation}\] \[\begin{equation} \ddot{U}(t + \Delta t) = \frac{1}{\beta \Delta t^2}\Delta U(t + \Delta t) - \frac{1}{\beta \Delta t} \dot{U}(t) - \frac{1 - 2\beta}{2\beta} \ddot {U}(t) \label{eq:2.5.3} \end{equation}\] 여기서 \(\gamma\), \(\beta\)는 Newmark-\(\beta\) 법의 매개변수이다.

 잘 알려진 바와 같이 \(\gamma\) 및 \(\beta\)를 다음 값으로 설정하면 선형 가속도법 또는 사다리꼴 법칙과 일치한다.

 \(\gamma = \displaystyle \frac{1}{2}\), \(\beta = \displaystyle \frac{1}{6}\)(선형 가속도법)

 \(\gamma = \displaystyle \frac{1}{2}\), \(\beta = \displaystyle \frac{1}{4}\)(사다리꼴 법칙)

 식 \(\eqref{eq:2.5.2}\), 식 \(\eqref{eq:2.5.3}\)을 식 \(\eqref{eq:2.5.1}\)에 대입하면 다음 식을 얻는다.

 \[\begin{align} \nonumber \left( \frac{1}{\beta \Delta t^2} \mathbf{M} + \frac{\gamma}{\beta \Delta t} C + K \right) \Delta U ( t + \Delta t ) &= F ( t + \Delta t ) - Q ( t + \Delta t ) \\\ \nonumber &+ \frac{1}{\beta \Delta t} M \dot{U} ( t ) + \frac{1 - 2\beta}{2\beta} M \ddot{U} ( t ) \\\ &+ \frac{\gamma - \beta}{\beta} C \dot{U} (t) + \Delta t \frac{\gamma - 2\beta}{2 \beta} C \ddot{U}(t) \label{eq:2.5.4} \end{align}\] 특히 선형 문제에서는 \(K_L\)을 선형 강성 행렬로 하고, \(Q ( t + \Delta t ) = K_L U (t + \Delta t)\)가 되며, 이 식을 위 식에 대입하면 다음 식을 얻는다.

 \[\begin{align} \nonumber M \left\lbrace -\frac{1}{\beta \Delta t^2} U(t) -\frac{1}{\beta \Delta t}\dot U(t) - \frac{2\beta}{1-2\beta} \ddot U(t) \right\rbrace &+ C\left\lbrace - \frac{\gamma}{\beta \Delta t} U(t) + \left(1 - \frac{\gamma}{\beta}\right) \dot U(t) + \Delta{t}\frac{ 2\beta-\gamma}{2\beta}\ddot U(t)\right\rbrace \\\ & + \frac{1}{\beta \Delta{t}^2} M + \frac{\gamma}{\beta \Delta{t}} C + K_L U(t+\Delta{t}) = F(t+\Delta{t}) \label{eq:2.5.5} \end{align}\] 또한 기하학적 경계조건으로 가속도가 지정된 위치에서는 식 \(\eqref{eq:2.5.2}\)로부터 다음 식의 변위를 얻는다.

 \[\begin{equation} u_{is} (t+\Delta{t}) = u_{is} (t) + \Delta t \dot{u}(t) + \Delta t^2 \left(\frac{1}{2} -\beta \right) {\ddot{u}}_{is} (t + \Delta t) \label{eq:2.5.6} \end{equation}\] 마찬가지로 속도가 지정된 위치에서는 식 \(\eqref{eq:2.5.6}\)으로부터 다음 식의 변위를 얻는다.

 \[\begin{equation} u_{is}(t+\Delta{t})= u_{is}(t)+\Delta t \frac{ \gamma - \beta}{ \gamma}\dot{u_{is}}(t) +(\Delta{t})^2 \frac{ \gamma - 2\beta}{ 2\gamma} \ddot{u_{is}}(t) +\Delta t \frac{\beta}{ \gamma}\dot{u_{is}}(t+\Delta{t}) \label{eq:2.5.7} \end{equation}\] 여기서 \(u_{is}(t+\Delta{t})\)는 시각 \(t+\Delta{t}\)에서의 절점 변위, \(\dot{u_{is}}(t+\Delta{t})\)는 시각 \(t+\Delta{t}\)에서의 절점 속도, \(\ddot{u_{is}}(t+\Delta{t})\)는 시각 \(t+\Delta{t}\)에서의 절점 가속도, \(i\)는 절점 자유도 번호, \(s\)는 절점 번호이다. 또한 질량 항과 감쇠 항의 취급은 다음과 같다.

 
### 질량 항의 취급

 질량 행렬은 원칙적으로 집중 질량 행렬로 취급한다.

 
### 감쇠 항의 취급

 감쇠 항은 식 \(\eqref{eq:2.5.8}\)로 표현되는 Rayleigh 감쇠로 취급한다.

 \[\begin{equation} C = R_m M + R_k K_L \label{eq:2.5.8} \end{equation}\] 여기서 \(R_m\), \(R_k\)는 Rayleigh 감쇠의 매개변수이다.

 `!DYNAMIC` 카드에서 지정하는 \(R_m\), \(R_k\)는 모델 전체에 일률적으로 적용된다. 재료별로 서로 다른 \(R_m\), \(R_k\)를 지정하려면 해당 재료의 `!MATERIAL` 블록 안에 [`!DAMPING`](../../keyword_reference/cnt/damping.ja.md) 카드를 지정한다. `!DAMPING`이 지정된 재료에 속하는 요소는 해당 요소의 질량 행렬 \(M_i\)와 접선 강성 행렬 \(K_i\)로부터 \(C_i = R_m M_i + R_k K_i\)로 요소 감쇠 행렬을 계산하여 전역 감쇠 행렬에 조립한다. 이 기능은 암시적 방법에서만 유효하다.

 
## 명시적 방법(중앙 차분법)

 명시적 방법에서는 아래 식에 나타낸 시각 t에서의 운동방정식을 바탕으로 한다.

 \[\begin{equation} M \ddot{U}(t) + C (t) \dot{U}(t) + Q(t) = F(t) \label{eq:2.5.9} \end{equation}\] 여기서는 시각 \(t + \Delta t\) 및 시각 \(t - \Delta t\)에서의 변위를 시각 \(t\)에서의 Taylor 전개로 나타내고, \(\Delta t\)에 관한 2차 항까지 취하면 다음과 같다.

 \[\begin{equation} U(t+\Delta{t}) = U(t)+\dot{U}(t)(\Delta{t}) +\frac{1}{2!}\ddot{U}(\Delta{t})^2 \label{eq:2.5.10} \end{equation}\] \[\begin{equation} U(t-\Delta{t})=U(t)-\dot{U}(t)(\Delta{t}) +\frac{1}{2!}\ddot{U}(\Delta{t})^2 \label{eq:2.5.11} \end{equation}\] 식 \(\eqref{eq:2.5.3}\), 식 \(\eqref{eq:2.5.4}\)의 차와 합으로부터 다음 식을 얻는다.

 \[\begin{equation} \dot{U}(t)=\frac{1}{2\Delta{t}} (U(t+\Delta{t})-U(t-\Delta{t})) \label{eq:2.5.12} \end{equation}\] \[\begin{equation} \ddot{U}= \frac{1}{(2\Delta{t})^2} (U(t+\Delta{t})-2U(t)+U(t-\Delta{t})) \label{eq:2.5.13} \end{equation}\] 식 \(\eqref{eq:2.5.12}\), 식 \(\eqref{eq:2.5.13}\)을 식 \(\eqref{eq:2.5.9}\)에 대입하면 다음 식을 얻는다.

 \[\begin{equation} \left( \frac{1}{\Delta t^{2}} M + \frac{1}{2\Delta t} C \right) U ( t + \Delta t ) \\\ = F(t) - Q(t) - \frac{1}{\Delta t^{2}} 2 U(t) - U( t - \Delta t) - \frac{1}{2\Delta t} C U(t - \Delta t) \label{eq:2.5.14} \end{equation}\] 특히 선형 문제에서는 \(Q(t) = K_L U(t)\)가 되며, 위 식은 다음과 같다

 \[\begin{equation} \left( \frac{1}{\Delta t^{2}} M + \frac{1}{2\Delta t} C \right) U( t + \Delta t ) \\\ = F(t) - K_L U(t) - \frac{1}{\Delta t^{2}} M U(t) - U(t - \Delta t) - \frac{1}{2\Delta t} C U (t - \Delta t) \label{eq:2.5.15} \end{equation}\] 여기서 질량 행렬 \(M\)을 집중 질량 행렬로, 감쇠 행렬을 비례 감쇠 행렬 \(C = R_m M\)과 같이 두면 식 \(\eqref{eq:2.5.15}\)은 연립방정식을 푸는 조작이 필요하지 않다.

 따라서 식 \(\eqref{eq:2.5.15}\)로부터 \(U(t+\Delta t)\)는 다음 식으로 구할 수 있다.

 \[\begin{equation} U( t + \Delta t ) \\\ = \frac{1}{( \frac{1}{\Delta t^{2}} M + \frac{1}{2\Delta t} C )} \{ F(t) - Q(t) - \frac{1}{\Delta t^{2}} M U(t) - U(t - \Delta t) - \frac{1}{2\Delta t} C U(t - \Delta t) \} \label{eq:2.5.17} \end{equation}\] 
## 관련 항목

 
- [해석 종류](../../functions_reference/01_analysis_type.html) — 동적 해석의 기능 개요와 사용 구분
- [참고문헌](../99_reference.html)


---


<!-- ========================================== -->
<!-- FILE: theory/solution_methods/03_eigen.md | TITLE: 모드 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/solution_methods/03_eigen.html -->

# 모드 해석

 
## 일반화 고유값 문제

 연속체의 자유 진동을 해석할 때는 공간 이산화를 수행하고, 그림 2.3.1과 같이 집중 질량을 갖는 다자유도계로 모델링한다. 감쇠가 없는 자유 진동 문제의 지배 방정식(운동 방정식)은 다음과 같다.

 \[\begin{equation} M \ddot{u} + K u = 0 \label{eq:2.3.1} \end{equation}\] 여기서 \(u\)는 일반화 변위 벡터, \(M\)은 질량 행렬, \(K\)는 강성 행렬이다. 고유 각진동수를 \(\omega\), 둘 다 0은 아닌 임의의 상수를 \(a\)와 \(b\), 벡터를 \(x\)라 하고 다음 함수를 정의한다.

 \[\begin{equation} u(t) = (a \sin \omega t + b \cos \omega t ) x \label{eq:2.3.2} \end{equation}\] 이 함수의 2계 미분은 다음과 같다.

 \[\begin{equation} \ddot{u}(t) = -\omega^2 (a \sin \omega t + b \cos \omega t) x \label{eq:2.3.3} \end{equation}\] 식 \(\eqref{eq:2.3.2}\)와 식 \(\eqref{eq:2.3.3}\)을 식 \(\eqref{eq:2.3.1}\)에 대입하면 다음을 얻는다.

 \[\begin{equation} M \ddot{u} + K u = (a \sin \omega t + b \cos \omega t) (K-\omega^2 M) x = 0 \label{eq:2.3.4} \end{equation}\] 따라서 위 식이 성립한다.

 비자명한 진동에서는 \(a \sin \omega t + b \cos \omega t\)가 항등적으로 0이 아니므로

 \[ (K-\omega^2M)x=0 \] 위 관계를 얻는다. 따라서 \(\lambda=\omega^2\)로 두면

 \[\begin{equation} K x = \lambda M x \label{eq:2.3.5} \end{equation}\] 위 식을 얻는다.

 스칼라 \(\lambda\)를 고유값, 벡터 \(x\)를 고유벡터라 하며, 식 \(\eqref{eq:2.3.5}\)로 표현되는 문제를 일반화 고유값 문제라 한다.

 고유값 \(\lambda=\omega^2\)에서 고유 각진동수 \(\omega\)를 구할 수 있으며, 대응하는 고유벡터 \(x\)는 진동 모드를 나타낸다.

 ![감쇠가 없는 자유 진동 다자유도계의 예](/theory/image/theory05_01.png)

 그림 2.3.1 감쇠가 없는 자유 진동 다자유도계의 예

 
## 행렬의 성질과 가정

 앞 절에서 얻은 일반화 고유값 문제 \(Kx=\lambda Mx\)에 대해 이 매뉴얼에서는 대상 행렬이 대칭이라고 가정한다. 복소 행렬에서는 에르미트 행렬에, 실수 행렬에서는 대칭 행렬에 해당한다.

 행렬 \(K\)의 \(ij\) 성분을 \(k_{ij}\)라 하면 에르미트 성질은 다음과 같이 표현된다.

 \[\begin{equation} k_{ij} = \bar{k}_{ji} \label{eq:2.3.6} \end{equation}\] 여기서 \(\bar{k}_{ji}\)는 \(k_{ji}\)의 복소켤레이다. 실수 행렬에서는 이 관계가 \(k_{ij}=k_{ji}\)가 된다.

 실대칭 행렬 \(H\)가 양의 정부호라는 것은 임의의 0이 아닌 벡터 \(x\)에 대해

 \[\begin{equation} x^{t} H x > 0 \label{eq:2.3.7} \end{equation}\] 위 부등식이 성립한다는 뜻이다. 이 경우 \(H\)의 모든 고유값은 양수이다.

 구조 고유값 문제에서 질량 행렬 \(M\)은 일반적으로 양의 정부호 행렬로 취급한다. 반면 강성 행렬 \(K\)는 구속조건에 따라 양의 준정부호 행렬이 될 수 있으며, 강체 모드에 대응하는 영 고유값을 가질 수 있다.

 
## 시프트 역반복법

 유한요소법을 이용한 실제 구조 해석에서는 일반적으로 모든 고유값이 필요하지 않으며, 몇 개의 저차 고유값만으로 충분한 경우가 많다. HEC-MW는 대규모 문제를 대상으로 하므로 행렬의 크기가 크고 영 성분이 많은 매우 희소한 행렬을 다룬다. 따라서 이러한 특성을 고려하여 저차 모드의 고유값을 효율적으로 구하는 것이 중요하다.

 시프트량을 \(\sigma\)라 할 때, \(-\sigma\)가 어떤 고유값과도 일치하지 않고 \(K+\sigma M\)이 비특이이면 식 \(\eqref{eq:2.3.5}\)를 다음과 같이 변환할 수 있다.

 \[\begin{equation} (K + \sigma M)^{-1} M x = \frac{1}{\lambda+\sigma} x \label{eq:2.3.8} \end{equation}\] 이 변환에서는 고유벡터 \(x\)가 변하지 않으며, 고유값 \(\lambda\)는 \(1/(\lambda+\sigma)\)로 사상된다.

 따라서 \(\lambda\)가 \(-\sigma\)에 가까울수록 변환된 고유값의 절댓값이 커진다. 구조 고유값 문제에서는 \(\lambda \geq 0\), \(\sigma \geq 0\)이므로 가장 낮은 고유값이 절댓값이 가장 큰 고유값으로 사상된다. 이 성질을 이용하여 절댓값이 큰 고유값부터 수렴하기 쉬운 반복법을 식 \(\eqref{eq:2.3.8}\)에 적용하면 저차 고유값부터 오름차순으로 효율적으로 구할 수 있다.

 이 방법을 시프트 역반복법이라 한다.

 FrontISTR는 구속조건이 있는 해석에서 \(\sigma = 0\)으로 설정하므로 식 \(\eqref{eq:2.3.8}\)은 \(K^{-1} M x = \frac{1}{\lambda} x\), 즉 시프트가 없는 역반복이 된다. 구속조건이 없는 자유-자유 해석에서는 \(K\)가 강체 모드에 대응하는 영 고유값을 가져 특이 행렬이 되므로, \(\sigma\)에 양의 값을 지정하여 \(K+\sigma M\)을 정칙화한다. 이때 \(\sigma\)는 `!EIGEN`의 `SIGMA`로 지정한다.

 
## Lanczos 법

 
### 채택 이유(Jacobi 법과의 비교)

 고전적인 방법으로는 Jacobi 법이 잘 알려져 있다.

 이 방법은 행렬의 크기가 작고 밀행렬일 때 효과적이다. 그러나 HEC-MW에서 다루는 행렬은 대규모 희소 행렬이므로 Jacobi 법을 사용하지 않고 Lanczos 반복법을 사용한다.

 
### 알고리즘과 특징

 1950년대에 C. Lanczos가 제안한 이 방법은 행렬을 삼중대각화하는 알고리즘이며 다음과 같은 특징이 있다.

 
- 반복법이므로 행렬의 희소성을 유지한 채 계산할 수 있다.
- 연산의 대부분이 행렬-벡터 곱으로 구성되어 병렬화에 적합하다.
- 유한요소 메시와 연계된 기하학적 영역 분할법에 적합하다.
- 구할 고유값의 개수나 모드 범위를 제한하여 효율적으로 계산할 수 있다.

 Lanczos 법은 초기 벡터에서 시작하여 직교 벡터를 차례로 구성하고 Krylov 부분공간의 기저를 구한다.

 유한 정밀도 연산에서는 반올림 오차로 인해 벡터의 직교성이 손실될 수 있다. 이 영향을 억제하기 위해 FrontISTR 구현에서는 이미 구한 Lanczos 기저 벡터에 대해 재직교화를 수행한다.

 
### 기하학적 의미(Krylov 부분공간)

 식 \(\eqref{eq:2.3.8}\)에 다음 변수 변환을 적용하면

 \[ A = (K + \sigma M)^{-1} M \] \[\begin{equation} \frac{1}{\lambda+\sigma}= \zeta \label{eq:2.3.9} \end{equation}\] 문제를 다음과 같이 다시 쓸 수 있다.

 \[\begin{equation} A x = \zeta x \label{eq:2.3.10} \end{equation}\] 따라서 위 식을 얻는다.

 임의의 0이 아닌 벡터 \(q_0\)에 대해

 \[ q_0,\ Aq_0,\ A^2q_0,\ldots,A^{m-1}q_0 \] 이 벡터들이 생성하는 공간

 \[ \mathcal{K}_m(A,q_0) = \operatorname{span} \{q_0,Aq_0,A^2q_0,\ldots,A^{m-1}q_0\} \] 을 Krylov 부분공간이라 한다.

 Lanczos 법은 이 Krylov 부분공간의 기저를 차례로 구성한다.

 FrontISTR에서는 질량 행렬 \(M\)에 관한 내적

 \[ \langle x,y\rangle_M = x^T M y \] 을 사용하여 기저를 정규직교화한다. 다음 그림에 나타낸 내적 \(\langle x,y\rangle\)과 노름 \(\|x\|\)은 FrontISTR 계산에서 각각 이 \(M\)-내적과 이에 대응하는 \(M\)-노름

 \[ \|x\|_M=\sqrt{x^T M x} \] 으로 해석한다.

 임의의 벡터 \(q_0\)에 행렬 \(A\)로 정의되는 선형 변환을 적용한다(그림 2.3.2 참조).

 ![행렬 \(A\)에 의한 \(q_0\)의 선형 변환](/theory/image/theory05_02.png)

 그림 2.3.2 행렬 \(A\)에 의한 \(q_0\)의 선형 변환

 변환된 벡터를 원래 벡터와 함께 생성하는 공간 안에서 직교화한다. 즉, 그림 2.3.3과 같이 Gram-Schmidt 직교화를 수행한다. 그 결과 얻은 벡터를 \(r_1\)이라 하고, 이를 정규화하여 \(q_1\)을 얻는다.

 ![\(q_0\)에 직교하는 벡터 \(q_1\)](/theory/image/theory05_03.png)

 그림 2.3.3 \(q_0\)에 직교하는 벡터 \(q_1\)

 같은 알고리즘으로 \(q_1\)에서 \(q_2\)를 구한다. 이때 \(q_2\)는 \(q_1\)과 \(q_0\) 모두에 직교한다(그림 2.3.4).

 ![\(q_1\)과 \(q_0\)에 직교하는 벡터 \(q_2\)](/theory/image/theory05_04.png)

 그림 2.3.4 \(q_1\)과 \(q_0\)에 직교하는 벡터 \(q_2\)

 이와 같이 Lanczos 법은 Krylov 부분공간의 정규직교 기저를 차례로 구성한다. 이론적으로는 대상 고유값 문제의 대칭성을 이용하여 이 반복을 최근의 기저 벡터를 사용하는 삼항 점화식으로 나타낼 수 있다.

 한편 FrontISTR 구현에서는 유한 정밀도 연산으로 인한 직교성 손실을 억제하기 위해 이미 구한 Lanczos 기저 벡터에 대해 \(M\)-내적을 이용한 재직교화를 수행한다.

 
### 삼중대각화

 FrontISTR의 Lanczos 반복에서는 앞 절에서 설명한 \(M\)-내적에 대해 기저 벡터를 정규직교화하므로

 \[ q_i^T M q_j = \delta_{ij} \] 위 관계가 성립한다.

 식 \(\eqref{eq:2.3.10}\)의 행렬 \(A\)를 사용하면 이론적인 Lanczos 반복은

 \[\begin{equation} A q_i = \beta_i q_{i-1} + \alpha_i q_i + \beta_{i+1} q_{i+1} \label{eq:2.3.11} \end{equation}\] 위의 삼항 점화식으로 나타낼 수 있다.

 먼저 \(\alpha_i\)를 다음과 같이 정의한다.

 \[ \alpha_i = q_i^T M A q_i \] 다음으로 임시 잔차를

 \[ \tilde{r}_{i+1} = Aq_i - \beta_i q_{i-1} - \alpha_i q_i \] 위와 같이 둔다.

 FrontISTR 구현에서는 유한 정밀도 연산으로 인한 직교성 손실을 억제하기 위해 \(\tilde{r}_{i+1}\)을 이미 구한 Lanczos 기저 벡터에 대해 \(M\)-내적으로 재직교화한다. 재직교화 후의 잔차를 \(r_{i+1}\)이라 하면

 \[\begin{equation} \beta_{i+1} = \sqrt{r_{i+1}^T M r_{i+1}}, \qquad q_{i+1} = \frac{r_{i+1}}{\beta_{i+1}} \label{eq:2.3.12} \end{equation}\] 위 관계가 성립한다.

 Lanczos 반복으로 얻은 \(m\)개의 기저 벡터를 다음과 같이 모으면

 \[ Q_m=[q_0,q_1,\ldots,q_{m-1}] \] 유한 횟수의 Lanczos 반복 후에는

 \[\begin{equation} A Q_m = Q_m T_m + \beta_m q_m e_m^T \label{eq:2.3.13} \end{equation}\] 위 관계가 성립한다.

 여기서 \(e_m\)은 제\(m\)성분만 1인 \(m\)차원 단위 벡터이며

 \[\begin{equation} T_m= \begin{pmatrix} \alpha_0 & \beta_1 & & &\\ \beta_1 & \alpha_1 & \beta_2 & &\\ & \ddots & \ddots & \ddots &\\ & & \beta_{m-2} & \alpha_{m-2}& \beta_{m-1}\\ & & & \beta_{m-1} & \alpha_{m-1} \end{pmatrix} \label{eq:2.3.14} \end{equation}\] 는 대칭 삼중대각 행렬이다.

 따라서 삼중대각 행렬 \(T_m\)의 고유값을 계산하면 원래의 대규모 고유값 문제의 고유값을 근사할 수 있다.

 
## 관련 항목

 
- [주파수 응답 해석](04_frequency_response.html) — 모드 해석 결과를 이용한 주파수 응답
- [해석 종류](../../functions_reference/01_analysis_type.html) — 모드 해석의 기능 개요


---


<!-- ========================================== -->
<!-- FILE: theory/solution_methods/04_frequency_response.md | TITLE: 주파수 응답 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/solution_methods/04_frequency_response.html -->

# 주파수 응답 해석

 
## 문제 정식화

 주파수 응답 해석은 외력이 시간에 따라 조화적으로 변할 때 시스템의 정상상태 응답 진폭과 위상을 주파수 영역에서 평가합니다. 감쇠 유무에 따라 지배 운동방정식의 취급이 달라지므로, 이 장에서는 먼저 무감쇠 자유진동으로부터 고유모드를 유도하고 이를 모드 기저로 사용하여 감쇠가 있는 조화 응답을 전개합니다.

 
### 무감쇠 자유진동과 고유모드

 감쇠를 무시하면 운동방정식은 다음과 같습니다.

 \[\begin{equation} M \ddot{U} + K U = 0 \label{eq:2.6.1} \end{equation}\] 변위를 각 고유모드로 전개하면

 \[\begin{equation} U = U_j e^{i \omega_j t} \label{eq:2.6.2} \end{equation}\] 이를 식 \(\eqref{eq:2.6.1}\)에 대입하면

 \[\begin{equation} K U_j = \omega_j^2 M U_j \label{eq:2.6.3} \end{equation}\] 을 얻습니다. 이는 일반화 고유값 문제입니다. 시프트 역반복법과 Lanczos법을 포함한 수치 해법은 [모드 해석](03_eigen.html)에서 다룹니다. 여기서는 이후 조화 응답 전개에 사용하는 고유진동수와 고유모드의 성질을 정리합니다.

 
### 고유진동수의 실수성

 고유진동수가 실수임은 다음과 같이 보일 수 있습니다. \(\omega_j^2 = \lambda_j\)라 둡니다. 식 \(\eqref{eq:2.6.3}\)의 복소켤레를 취하면 식 \(\eqref{eq:2.6.4}\)를 얻습니다.

 \[\begin{equation} K U_j = \lambda_j M U_j K \overline{UJ} = \overline{\lambda_J} M \overline{U_J} \label{eq:2.6.4} \end{equation}\] \(\overline{U}_J^T\)를 곱하면

 \[\begin{equation} U_j^T K \overline{U}_J = \overline{\lambda}_J U_j^T M \overline{U}_J \overline{U}_J^T K U_j = \lambda_j \overline{U}_J^T M U_j \label{eq:2.6.5} \end{equation}\] 식 \(\eqref{eq:2.6.5}\)로부터

 \[\begin{equation} 0 = ( \lambda_j - \overline{\lambda_J} ) \overline{U_J}^T M U_j \label{eq:2.6.6} \end{equation}\] 입니다. 질량 행렬은 대칭 양의 정부호이므로 0이 아닌 고유벡터에 대해

 \[\begin{equation} \overline{U_J} M U_j > 0 \label{eq:2.6.7} \end{equation}\] 입니다. 따라서

 \[\begin{equation} \lambda_j =\overline{\lambda_J} \label{eq:2.6.8} \end{equation}\] 이고, 그러므로 \(\omega_j^2 = \lambda_j\)는 실수입니다.

 
### 고유모드의 직교성과 정규화

 서로 다른 두 모드를 생각합니다.

 \[\begin{equation} K U_i = \lambda_i M U_i K U_j = \lambda_j M U_j \label{eq:2.6.9} \end{equation}\] 그러면

 \[\begin{equation} ( \lambda_i - \lambda_j ) U_j^T M U_i = 0 \label{eq:2.6.10} \end{equation}\] 이고, 고유값이 서로 다르면

 \[\begin{equation} U_j^T M U_i = 0 \label{eq:2.6.11} \end{equation}\] 입니다. 따라서 서로 다른 고유모드는 질량 행렬에 대해 직교합니다. 같은 모드에 대해서는 식 \(\eqref{eq:2.6.12}\)와 같이 질량 행렬에 대해 정규화하면 이후의 취급이 간단해집니다.

 \[\begin{equation} U_i^T M U_i = 1 \label{eq:2.6.12} \end{equation}\] 
## 감쇠가 있는 조화 응답

 다음으로 감쇠를 고려한 주파수 응답 해석의 정식화를 제시합니다. 지배 운동방정식은 식 \(\eqref{eq:2.6.13}\)과 같습니다.

 \[\begin{equation} M \ddot{U} + C \dot{U} + K U = F \label{eq:2.6.13} \end{equation}\] 감쇠항은 Rayleigh형이라고 가정하며 식 \(\eqref{eq:2.6.14}\)로 나타냅니다.

 \[\begin{equation} C = \alpha M + \beta K \label{eq:2.6.14} \end{equation}\] 모드 해석에서 얻은 고유벡터를 사용하면 시각 *t*에서의 변위 벡터는 식 \(\eqref{eq:2.6.15}\)와 같이 전개할 수 있습니다.

 \[\begin{equation} U(t) = \sum_i b_i(t) U_i \label{eq:2.6.15} \end{equation}\] 외력항이 다음의 조화형이라고 가정합니다.

 \[\begin{equation} F(t) = ( F_R + i F_I )e^{i \Omega t} \label{eq:2.6.16} \end{equation}\] 그러면 \(b_{j}(t)\)를 구합니다. 식 \(\eqref{eq:2.6.13}\)의 운동방정식은 강제진동 형태이므로

 \[\begin{equation} b_j (t) = (b_{jR} + b_{jI}) e^{i \Omega t} \label{eq:2.6.17} \end{equation}\] 가 성립합니다. 전개 계수 \(b_{j}(t)\)의 실수부와 허수부는 각각 식 \(\eqref{eq:2.6.18}i\) 및 \(\eqref{eq:2.6.19}\)로 주어집니다.

 \[\begin{equation} b_{jR} = \frac{ U^T_j F_R (\omega^2_j - \Omega^2) + U^T_j F_I (\alpha + \beta \omega_j^2) \Omega}{ (\omega^2_j - \Omega^2)^2 + (\alpha + \beta \omega_j^2)^2 \Omega^2} \label{eq:2.6.18} \end{equation}\] \[\begin{equation} b_{jI} = \frac{ U^T_j F_I(\omega^2_j - \Omega^2) - U^T_j F_R(\alpha + \beta \omega_j^2) \Omega}{ (\omega^2_j - \Omega^2)^2 + (\alpha + \beta \omega_j^2)^2 \Omega^2} \label{eq:2.6.19} \end{equation}\] 이로써 원하는 조화 응답 계수를 얻습니다.

 
## 관련 항목

 
- [모드 해석](03_eigen.html) — 기초 고유값 문제의 정식화
- [동적 해석 방법](02_dynamic.html) — 시간 영역의 동적 해석
- [해석 종류](../../functions_reference/01_analysis_type.html) — 주파수 응답 해석의 기능 개요


---


<!-- ========================================== -->
<!-- FILE: theory/solution_methods/05_transient_heat.md | TITLE: 비정상 열전도 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/solution_methods/05_transient_heat.html -->

# 비정상 열전도 해석

 유한요소법(Finite Element Method)을 이용한 고체 열전도 해석의 시간 이산화와 반복해법을 설명한다. 연속체의 지배방정식과 경계조건은 [열전도 방정식](../governing/02_heat_equation.html)을 참조한다.

 
## 이산화 방정식(출발점)

 열전도 방정식([열전도 방정식의 식 (gov_he_main)](../governing/02_heat_equation.html))을 Galerkin법으로 이산화하면,

 \[\begin{equation} K T + M \frac{\partial T}{\partial t} = F \label{eq:2.4.8} \end{equation}\] 단,

 \[\begin{equation} K = \int\left( k_x \frac{\partial N^T}{\partial x}\frac{\partial N}{\partial x} + k_y \frac{\partial N^T}{\partial y}\frac{\partial N}{\partial y} + k_z \frac{\partial N^T}{\partial z}\frac{\partial N}{\partial z} \right) dV + \int hc N^T N ds + \int hr N^T N ds \label{eq:2.4.9} \end{equation}\] \[\begin{equation} M = \int \rho c N^T N dV \label{eq:2.4.10} \end{equation}\] \[\begin{equation} F = \int Q N^T dV - \int q_s N^T dS + \int{hc} T c N^T dS + \int{hcTr} ({T+Tr}) ({T^2 + T r^2}) N^T dS \label{eq:2.4.11} \end{equation}\] \[\begin{equation} N = (N^1, N^2, \ldots, Ni) \label{eq:2.4.12} \end{equation}\] 여기서 \(K\), \(M\), \(F\), \(N\)은 각각 열전도 행렬(경계 기여의 대류·복사 항 포함), 질량 행렬, 열하중 벡터, 형상함수 행렬이다. 물성 기호(\(\rho\), \(c\), \(k_x, k_y, k_z\), \(Q\), \(hc\), \(hr\) 등)의 정의는 [열전도 방정식](../governing/02_heat_equation.html)을 따른다.

 
## 시간 이산화와 반복해법

 방정식 \(\eqref{eq:2.4.8}\)은 비선형 비정상 방정식이다. 이제 시간에 대해 후진 Euler법으로 이산화하고, 시각 \(t=t_0\)에서의 온도가 알려져 있을 때 시각 \(t=t_0+\Delta t\)에서의 온도를 다음 식으로 계산한다.

 \[\begin{equation} K_{t=t_0+\Delta t} T_{t=t_0+\Delta t} + M_{t=t_0+\Delta t} \frac{T_{t=t_0+\Delta t} - T_{t=t_0}}{\Delta t} = F_{t=t_0+\Delta t} \label{eq:2.4.13} \end{equation}\] 여기서 식 \(\eqref{eq:2.4.13}\)을 근사적으로 만족하는 온도 벡터 \(T_{t=t_0+\Delta t}^{(i)}\) 를 개선하여 더 정확한 해 \(T_{t=t_0+\Delta t}^{(i)+1}\)을 구하는 것을 생각한다.

 이를 위해 먼저 온도 벡터를 다음과 같이 나타낸다.

 \[\begin{equation} T_{t=t_0+\Delta t}= T_{t=t_0+\Delta t}^{(i)} + \Delta T_{t=t_0+\Delta t}^{(i)} \label{eq:2.4.14} \end{equation}\] 열전도 행렬과 온도 벡터의 곱, 질량 행렬 등을 다음 식과 같이 근사적으로 나타낸다.

 \[\begin{equation} K_{t=t_0+\Delta t} T_{t=t_0+\Delta t} = K_{t=t_0+\Delta t}^{(i)} T_{t=t_0+\Delta t}^{(i)} + \frac{\partial \big(K_{t=t_0+\Delta t}^{(i)} T_{t=t_0+\Delta t}^{(i)}\big) } {\partial T_{t=t_0+\Delta t}^{(i)} } \Delta T_{t=t_0+\Delta t}^{(i)} \label{eq:2.4.15} \end{equation}\] \[\begin{equation} M_{t=t_0+\Delta t} = M_{t=t_0+\Delta t}^{(i)} + \frac{\partial M_{t=t_0+\Delta t}^{(i)}}{\partial T_{t=t_0+\Delta t}^{(i)}} \Delta T_{t=t_0+\Delta t}^{(i)} \label{eq:2.4.16} \end{equation}\] 식 \(\eqref{eq:2.4.14}\), 식 \(\eqref{eq:2.4.15}\), 식 \(\eqref{eq:2.4.16}\)을 식 \(\eqref{eq:2.4.13}\)에 대입하고 2차 이상의 항을 생략하면 다음 식을 얻는다.

 \[\begin{equation} \bigg(\frac{M_{t=t_0+\Delta t}^{(i)}}{\Delta t} + \frac {\partial M_{t=t_0+\Delta t}^{(i)} } { \partial T_{t=t_0+\Delta t}^{(i)} } \frac{T_{t=t_0+\Delta t}^{(i)} - T_{t=t_0}}{\Delta t} + \frac{\partial \big(K_{t=t_0+\Delta t}^{(i)} T_{t=t_0+\Delta t}^{(i)}\big)} {\partial T_{t=t_0+\Delta t}^{(i)}} \bigg) \Delta T_{t=t_0+\Delta t}^{(i)} \\\ = F_{t=t_0+\Delta t} - M_{t=t_0+\Delta t}^{(i)} \frac{T_{t=t_0+\Delta t}^{(i)} - T_{t=t_0}}{\Delta t} - K_{t=t_0+\Delta t}^{(i)} T_{t=t_0+\Delta t}^{(i)} \label{eq:2.4.17} \end{equation}\] 또한 좌변의 계수 행렬을 다음 식으로 근사 평가한다.

 \[\begin{equation} K^{(i)} = \frac{M_{t=t_0+\Delta t}^{(i)}}{\Delta t} + \frac{\partial \big( K_{t=t_0+\Delta t}^{(i)} T_{t=t_0+\Delta t}^{(i)} \big)}{\partial T^{(i)}_{t=t_0+\Delta t}} = \frac{M_{t=t_0+\Delta t}^{(i)}}{\Delta t} + K_{T_{t=t_0+\Delta t}}^{(i)} \label{eq:2.4.18} \end{equation}\] 여기서 \(K_{T_{t=t_0+\Delta t}}^{(i)}\)는 접선강성행렬이다.

 결국 다음 식을 이용해 반복 계산을 수행함으로써 시각 \(t=t_0+\Delta t\)에서의 온도를 계산할 수 있다.

 \[\begin{equation} K^{(i)} \Delta T_{t=t_0+\Delta t}^{(i)} = F_{t=t_0+\Delta t} - M_{t=t_0+\Delta t}^{(i)} \frac{T_{t=t_0+\Delta t}^{(i)} - T_{t=t_0}}{\Delta t} - K^{(i)} T_{t=t_0+\Delta t}^{(i)} \label{eq:2.4.19} \end{equation}\] 특히 정상 해석에서는 다음 식을 이용해 반복 계산을 수행한다.

 \[ K_T^{(i)} \Delta T_{t=\infty}^{(i)} = F_{t=\infty} - K_T^{(i)} \Delta T_{t=\infty}^{(i)} \] \[\begin{equation} T_{t=\infty}^{(i+1)} = T_{t=\infty}^{(i)} + \Delta{T}_{t=\infty}^{(i)} \label{eq:2.4.20} \end{equation}\] 비정상 해석에서 시간 증분 \(\Delta t\)의 선택은 시간 이산화에 암시적 해법을 사용하므로 일반적으로 그 크기에 대한 제약을 받지 않는다. 다만 시간 증분 \(\Delta t\)가 지나치게 크면 반복 계산의 수렴 횟수가 증가한다. 일반적으로 시간 증분 \(\Delta t\)가 지나치게 크면 반복 횟수가 증가한다. 구현에서는 잔차 벡터의 크기를 감시하여 수렴이 느리면 \(\Delta t\)를 줄이고, 반복 횟수가 적으면 \(\Delta t\)를 늘리는 자동 증분 제어를 사용한다(→ 자세한 내용은 [스텝 제어](../../functions_reference/07_step.html) 참조).

 
## 관련 항목

 
- [해석 종류](../../functions_reference/01_analysis_type.html) — 열전도 해석의 기능 개요
- [스텝 제어](../../functions_reference/07_step.html) — 자동 증분 제어의 상세
- [참고문헌](../99_reference.html)


---


<!-- ========================================== -->
<!-- FILE: theory/solution_methods/06_geometric_bc.md | TITLE: 기하학적 경계 조건의 처리 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/theory/solution_methods/06_geometric_bc.html -->

# 기하학적 경계 조건의 처리

 [Newton-Raphson법](01_newton_raphson.html)에서 각 반복마다 푸는 선형 방정식 \(\boldsymbol{K} d\boldsymbol{u} = \boldsymbol{R}\)에는 아직 기하학적 경계 조건(변위 구속)이 포함되어 있지 않다. 이 장에서는 변위 구속 \(\boldsymbol{u} = \bar{\boldsymbol{u}}\)가 주어진 자유도에 대해 강성 행렬 \(\boldsymbol{K}\)와 잔차 벡터 \(\boldsymbol{R}\)을 어떻게 수정하면 반복 후 변위가 구속값에 도달하는지를 보인다. 다점 구속(MPC)이나 접촉에 따른 비선형 구속은 별도 계통으로 처리하며, [다점 구속](../../functions_reference/04b_mpc.html)과 [접촉](../../functions_reference/05_contact.html)을 참조한다.

 
## 변위 구속을 미지 변수로 남기는 방침

 기하학적 경계 조건이 주어진 절점의 변위는 더 이상 미지 변수가 아니라 지정값 \(\bar{u}_i\)로 고정된다. 이를 상수로 간주하여 선형 방정식 \(\boldsymbol{K} d\boldsymbol{u} = \boldsymbol{R}\)에서 제거할 수도 있지만, FrontISTR에서는 구속 자유도도 다른 자유도와 동일한 수정량 벡터 \(d\boldsymbol{u}\)에 포함한 채 \(\boldsymbol{K}\)와 \(\boldsymbol{R}\) 쪽을 다시 써서 구속을 구현한다.

 이 방식에서는 구속 유무에 따라 계수 행렬의 차원과 자유도 순서가 바뀌지 않으므로 변수를 제거하고 비어 있는 행과 열을 다시 채우는 처리가 필요 없다. 희소행렬 형식으로 계수 행렬을 유지하는 구현에서는 이러한 재배치가 복잡해지므로, FrontISTR는 계수 행렬과 우변의 갱신만으로 구속을 달성하는 후자의 방식을 채택한다.

 
## Newton-Raphson 수정량의 치환 목표

 변위가 구속된 자유도 \(i\)에 대해 Newton-Raphson 반복에서 구하는 수정량 \(du_i\)가

 \[ du_i = d\bar{u}_i \equiv \begin{cases} \bar{u}_i & (\text{1차 반복}) \\ 0 & (\text{2차 반복 이후}) \end{cases} \] 이 되도록 \(\boldsymbol{K}\)와 \(\boldsymbol{R}\)을 수정한다. 이에 따라 첫 번째 반복에서 변위 증분은 구속값 \(\bar{u}_i\)에 도달하고, 두 번째 반복 이후에는 수정량이 \(0\)이 되어 구속값이 유지된다.

 
## 행렬·잔차 벡터의 변형 절차

 앞 절의 치환 목표는 선형 방정식 \(\boldsymbol{K} d\boldsymbol{u} = \boldsymbol{R}\)의 형태를 유지한 채 다음 두 가지 변형으로 달성된다.

 
1. **\(i\)행의 대각화**: \(du_i\)의 계수인 대각항 \(K_{ii}\)를 \(1\)로, 그 밖의 계수 \(K_{ij}\ (j \not= i)\)를 모두 \(0\)으로 치환하고 우변을 \(R_i \to d\bar{u}_i\)로 한다. 그러면 \(i\)행의 방정식은 \(du_i = d\bar{u}_i\)가 된다.
2. **\(j(\not= i)\)행에서의 이항**: \(j(\not= i)\)행에서 \(K_{ji} \not= 0\)이면 \(K_{ji} du_i = K_{ji} d\bar{u}_i\) 항을 우변으로 이항한다. 즉, 우변에서 \(K_{ji} d\bar{u}_i\)를 빼고 \(du_i\)의 계수 \(K_{ji}\)를 \(0\)으로 한다.

 성분 표기로 쓰면 변형 전후의 방정식은 다음과 같이 대응한다.

 \[ \begin{bmatrix} K_{11} & K_{12} & \cdots & K_{1i} & \cdots & K_{1N} \\ K_{21} & K_{22} & \cdots & K_{2i} & \cdots & K_{2N} \\ \vdots & \vdots & \ddots & \vdots & & \vdots \\ K_{i1} & K_{i2} & \cdots & K_{ii} & \cdots & K_{iN} \\ \vdots & \vdots & & \vdots & \ddots & \vdots \\ K_{N1} & K_{N2} & \cdots & K_{Ni} & \cdots & K_{NN} \\ \end{bmatrix} \begin{bmatrix} du_1 \\ du_2 \\ \vdots \\ du_i \\ \vdots \\ du_N \end{bmatrix} = \begin{bmatrix} R_1 \\ R_2 \\ \vdots \\ R_i \\ \vdots \\ R_N \end{bmatrix} \] \[ \Longrightarrow \quad \begin{bmatrix} K_{11} & K_{12} & \cdots & 0 & \cdots & K_{1N} \\ K_{21} & K_{22} & \cdots & 0 & \cdots & K_{2N} \\ \vdots & \vdots & \ddots & \vdots & & \vdots \\ 0 & 0 & \cdots & 1 & \cdots & 0 \\ \vdots & \vdots & & \vdots & \ddots & \vdots \\ K_{N1} & K_{N2} & \cdots & 0 & \cdots & K_{NN} \\ \end{bmatrix} \begin{bmatrix} du_1 \\ du_2 \\ \vdots \\ du_i \\ \vdots \\ du_N \end{bmatrix} = \begin{bmatrix} R_1 - K_{1i} d\bar{u}_i \\ R_2 - K_{2i} d\bar{u}_i \\ \vdots \\ d\bar{u}_i \\ \vdots \\ R_N - K_{Ni} d\bar{u}_i \\ \end{bmatrix} \] 변형 후의 계수 행렬과 우변을 각각 \(\tilde{\boldsymbol{K}}, \tilde{\boldsymbol{R}}\)로 쓰면, 최종적으로 풀어야 할 선형 방정식은

 \[ \tilde{\boldsymbol{K}}\, d\boldsymbol{u} = \tilde{\boldsymbol{R}} \] 이 된다. 구속 자유도가 여러 개 존재하는 경우 각 구속 자유도 \(i\)에 대해 위의 1, 2 변형을 차례로 적용한다.

 
## 희소행렬 구현에서의 처리 범위

 구현상 계수 행렬 \(\boldsymbol{K}\)는 희소행렬 형식으로 유지되며, 앞 절의 절차 2에서 \(j(\not= i)\)행의 처리는 \(i\)열에 0이 아닌 성분을 갖는 행, 즉 구속 자유도 \(i\)와 요소를 통해 연결된 절점의 자유도에 대해서만 수행된다. 연결되지 않은 자유도에서는 \(K_{ji} = 0\)이므로 이항에 의해 우변이 변하지 않는다. 이러한 제한으로 전체 절점 수에 대해 국소적인 조작만으로 구속 처리가 완료된다.

 
## 관련 항목

 
- [외력의 가상일과 전체 방정식 조립](../discretization/04_external_force_and_assembly.html) — 수정 전 \(\boldsymbol{K}, \boldsymbol{R}\)의 구성
- [Newton-Raphson법](01_newton_raphson.html) — 수정량 \(d\boldsymbol{u}\)를 구하는 반복과의 관계
- [수렴 판정](01b_convergence.html) — 구속 처리 후 잔차 \(\tilde{\boldsymbol{R}}\)에 대한 노름 평가


---

