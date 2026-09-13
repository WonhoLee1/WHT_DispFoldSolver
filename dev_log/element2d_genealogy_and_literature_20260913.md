# 2D 솔리드 유한요소 학술 계보, 이론 정식화 및 발전사 총람

## 1. 개요 (Overview)

본 문서는 `WHT_DispFoldSolver`의 2D 평면 변형률(Plane Strain, `CPE`) 및 평면 응력(Plane Stress, `CPS`) 유한요소 체계를 상용 CAE(Abaqus)와 1:1 완전 호환 구조로 전면 재설계하기 위한 학술 문헌 조사, 최초 개발자 및 계보도, 수치적 정식화, 그리고 요소별 장단점 총람입니다.

기존 2D 구현의 기술적·이론적 결함(좌표 회전 시 비가역적 전단 잠김 부활, 메쉬의 요소 타입 문자열 무시, 비적합 모드 정적 응축 누락 등)을 근본적으로 극복하고, 3D 요소 라이브러리(`dispsolver/element3d/`, `dispsolver/solver3d/`)와 완벽히 대칭되는 데이터 지향 설계(DOD) 기반 Numba 고속 어셈블리 아키텍처를 도입합니다.

---

## 2. Abaqus 2D 솔리드 요소 계보 및 개발자 총람 (Academic Genealogy)

| 요소 명칭 | 차수 및 형상 | 최초 개발자 및 소속 연구기관 | 핵심 원저 문헌 (Foundational Paper) | 최신 발전자 및 개량 문헌 (Recent Advances) | 상용 CAE 대응 |
|:---|:---:|:---|:---|:---|:---|
| **`CPE4` / `CPS4`** | 1차 4절점 사각형 (2×2 완전적분) | Bruce M. Irons, Olgierd C. Zienkiewicz, J. G. Ergatoudis (Swansea Univ., UK, 1968) | Ergatoudis, Irons & Zienkiewicz (1968), *IJSS* | Belytschko et al. (2000), *Nonlinear FE for Continua & Structures* | Abaqus CPE4/CPS4 / Ansys SOLID182 (Full) |
| **`CPE4I` / `CPS4I`** | 1차 4절점 비적합/EAS 사각형 (4-mode) | Edward L. Wilson, Robert L. Taylor (UC Berkeley, 1973); Juan C. Simo, M. S. Rifai (Stanford, 1990) | Wilson et al. (1973) / Simo & Rifai (1990), *IJNME* | Simo & Armero (1992), *IJNME*; Korelc & Wriggers (1996), *CMAME* | Abaqus CPE4I/CPS4I / Ansys SOLID182 (Enhanced Strain) |
| **`CPE4R` / `CPS4R`** | 1차 4절점 1점 감차적분 사각형 + FB Hourglass | Dennis P. Flanagan (Sandia Labs), Ted Belytschko (Northwestern Univ., 1981) | Flanagan & Belytschko (1981), *IJNME* | Puso (2000), *IJNME*; Felippa (2005), *CMAME*; Abaqus Theory Guide §3.2.4 | Abaqus CPE4R/CPS4R / LS-DYNA ELFORM 1 |
| **`CPE4H` / `CPS4H`** | 1차 4절점 혼합 하이브리드 사각형 (u-P) | Leonard R. Herrmann (UC Davis, 1965); Juan C. Simo, Robert L. Taylor, Karl S. Pister (UC Berkeley, 1985) | Herrmann (1965), *AIAA J.*; Simo, Taylor & Pister (1985), *CMAME* | Brink & Stein (1996), *IJNME*; Wriggers (2008), *Nonlinear FEM* | Abaqus CPE4H / Ansys Mixed u-P SOLID182 |
| **`CPE4_FBAR`** | 1차 4절점 체적 F-bar 투영 사각형 | Thomas J. R. Hughes (Stanford, 1980); Eduardo A. de Souza Neto, Djordje Peric, D. R. J. Owen (Swansea Univ., 1996) | Hughes (1980), *IJNME*; de Souza Neto et al. (1996), *IJSS* | de Souza Neto et al. (2008), *Computational Plasticity* | Abaqus CPE4 with F-bar / Large-Strain Plasticity |
| **`CPE4_CR`** | 1차 4절점 공회전(Co-rotational) 사각형 + SectionControls | Ted Belytschko, B. J. Hsieh (Northwestern Univ., 1973); Carlos A. Felippa, Bjorn Haugen (CU Boulder / NTNU, 2005) | Belytschko & Hsieh (1973), *IJNME*; Felippa & Haugen (2005), *CMAME* | Battini & Pacoste (2002), *CMAME*; Abaqus *SECTION CONTROLS | Abaqus CPE4 with Distortion Control |
| **`CPE3` / `CPS3`** | 1차 3절점 일정변형률 삼각형 (CST) | M. Jon Turner, Ray W. Clough, Harold C. Martin, LeRoy J. Topp (Boeing / UC Berkeley, 1956) | Turner, Clough, Martin & Topp (1956), *J. Aero. Sci.* | Felippa (2004), *CMAME* (ANDES Triangle) | Abaqus CPE3/CPS3 / Ansys PLANE182 (Tri) |
| **`CPE6` / `CPS6`** | 2차 6절점 등매개 삼각형 | John H. Argyris (Imperial College / Univ. of Stuttgart, 1965); Olgierd C. Zienkiewicz (Swansea Univ., 1971) | Argyris (1965), *J. Royal Aero. Soc.*; Zienkiewicz (1971), *The Finite Element Method* | Dunavant (1985), *IJNME* (Symmetric Gauss Rules) | Abaqus CPE6/CPS6 / Ansys PLANE183 (Tri) |
| **`CPE6M`** | 2차 6절점 개량 삼각형 (체적 B-bar + 접촉 양수 반력) | Hibbitt, Karlsson & Sorensen (HKS / Abaqus Team, 1999); A. Czekanski, S. A. Meguid (Univ. of Toronto, 2001) | Abaqus Theory Guide §3.2.6; Czekanski & Meguid (2001), *FEAD* | Gee, Dohrmann, Key & Wall (2009), *CMAME* | Abaqus CPE6M (Modified 6-node Tri) |
| **`CPE8` / `CPE8R`** | 2차 8절점 세렌디피티 사각형 (Full 3×3 / Reduced 2×2) | Bruce M. Irons, Olgierd C. Zienkiewicz (Swansea Univ., 1968) | Zienkiewicz, Irons et al. (1969), *Isoparametric and Associated Element Families* | Bathe (1996), *Finite Element Procedures* | Abaqus CPE8 / CPE8R / Ansys PLANE183 |

---

## 3. 요소별 정식화 메커니즘, 장단점 및 적용 분야

### 3.1 CPE4 / CPS4 (Standard Bilinear Quad)
- **정식화**: $u(x, y) = \sum_{i=1}^4 N_i(\xi, \eta) u_i$, 완전 2×2 가우스 수치적분(4 GPs).
- **장점**: 기하학적 대칭성, 완전한 수치 랭크 (8 DOF - 3 강체모드 = 5 양의 고유치).
- **단점**: 
  - **전단 잠김 (Shear Locking)**: 순수 굽힘 하중 하에서 인공적인 기생 전단 변형률($\gamma_{xy} \ne 0$)이 발생하여 요소 형상비(Aspect Ratio)가 커질수록 굽힘 강성이 비정상적으로 급증함 ($1 + 0.35 \text{AR}^2$).
  - **체적 잠김 (Volumetric Locking)**: 비압축성 극한($\nu \to 0.5$)에서 체적 팽창 구속으로 인해 요소가 굳어버림.
- **용도**: 인장/압축 우세 영역의 기본 해석.

### 3.2 CPE4I / CPS4I (Incompatible / Enhanced Assumed Strain Quad)
- **정식화**: Simo & Rifai (1990) 4-모드 Enhanced Assumed Strain.
  $$\tilde{\boldsymbol{E}} = \boldsymbol{E}^{compat} + \boldsymbol{M}(\xi, \eta) \boldsymbol{\alpha}$$
  여기서 $\boldsymbol{\alpha} = [\alpha_1, \alpha_2, \alpha_3, \alpha_4]^T$는 요소 내부 비적합 변형률 자유도이며, 요소 단위에서 정적 응축(Static Condensation)되어 전역 강성 행렬의 크기는 일반 4절점 요소와 동일함.
- **장점**: 
  - **전단 잠김 완전 제거**: 1개 요소만으로 캔틸레버 보의 순수 굽힘 이론해를 100.0% 오차 없이 정확히 재현.
  - **프레임 객관성 (Frame Invariance)**: 요소가 임의의 각도(0°~90°)로 회전해도 전단 잠김 방지 성능이 1.000으로 완벽 유지됨.
- **단점**: 요소 형상이 평행사변형을 벗어나 심하게 사다리꼴(Trapezoidal)로 왜곡될 경우 Wilson-Taylor 형상 함수 특성상 감쇠가 필요함 (Simo의 $J_0/J$ 자코비안 스케일링으로 완화).
- **용도**: 얇은 디스플레이 박판 폴딩, 굽힘 우세 판재 성형 해석의 **최우선 권장 요소**.

### 3.3 CPE4R / CPS4R (Reduced Integration Quad + Orthogonal Hourglass Control)
- **정식화**: 중심점 1점 가우스 적분($\xi=\eta=0$) + Flanagan & Belytschko (1981) 직교 아워글래스 안정화 벡터 $\boldsymbol{\gamma}$.
  $$\boldsymbol{\gamma} = \frac{1}{A} \left[ \boldsymbol{h} - (\boldsymbol{h}^T \boldsymbol{x}) \boldsymbol{b}_x - (\boldsymbol{h}^T \boldsymbol{y}) \boldsymbol{b}_y \right]$$
  여기서 $\boldsymbol{h} = [1, -1, 1, -1]^T$는 아워글래스 모드 형상.
- **장점**:
  - **초고속 연산**: 완전적분 대비 3~4배 빠른 어셈블리 속도.
  - **전단 잠김 및 체적 잠김 원천 차단**: 1점 적분이므로 기생 전단 변형률 및 과구속 체적 변형률이 0이 됨.
- **단점**: 점하중(Point Load) 작용 시 아워글래스 격자 뒤틀림 모드가 발생할 수 있으므로 SectionControls의 Hourglass Stiffness 계수($Q_{HG} \approx 0.005 \sim 0.05$) 관리 필수.
- **용도**: 초탄성 고무(PSA) 대변형, 고속 접촉/충돌 해석.

### 3.4 CPE4H (Mixed u-P Hybrid Quad)
- **정식화**: Herrmann (1965) 변분 원리. 변위 $\boldsymbol{u}$와 함께 정수압(Hydrostatic Pressure) $p$를 독립 변수로 취급.
  $$\Pi(\boldsymbol{u}, p) = \int_{\Omega} \left( W_{dev}(\boldsymbol{F}_{dev}) + \frac{1}{2} p (J - 1) - \frac{1}{2K} p^2 \right) d\Omega$$
- **장점**: $\nu = 0.49999$의 완전 비압축성 조건에서도 체적 잠김(Volumetric Locking)이 완벽히 방지됨.
- **용도**: PSA, 실리콘, 고무 등 초탄성 비압축성 층 해석.

### 3.5 CPE4_FBAR (Multiplicative Centroid F-bar Quad)
- **정식화**: Hughes (1980) B-bar의 유한변형률 승법 분해 확장.
  $$\bar{\boldsymbol{F}} = \left( \frac{J_0}{J} \right)^{1/2} \boldsymbol{F}, \quad J_0 = \det \boldsymbol{F}(\mathbf{0})$$
- **장점**: 추가적인 내부 압력 자유도 없이도 중심점 체적비 $J_0$ 투영을 통해 체적 비압축성 대변형 소성을 완벽히 해석.
- **용도**: 대변형 탄소성 굽힘.

### 3.6 CPE6 / CPE6M (Quadratic Triangles)
- **CPE6**: 6절점 2차 삼각형. 곡면 경계 적합성이 뛰어나며 1차 삼각형(CST)의 전단 잠김을 극복.
- **CPE6M**: Abaqus가 자체 개발한 수정 2차 삼각형. 일반 2차 삼각형은 접촉면 모서리 절점과 중간 절점의 등가 하중비가 -1:2가 되어 접촉 해석 시 인공적인 인장력(Chattering)이 발생하지만, CPE6M은 수정된 체적 B-bar 및 양수 접촉 가중치 정식화를 통해 모든 절점에서 양의 접촉 압력을 보장함.

---

## 4. 기존 2D 개발 실패 사례 분석 및 극복 대책

### [실패 1] 회전 시 전단 잠김 부활 현상 (AGENTS.md §4.1, §4.14)
* **원인**: 기존의 단순 감차적분 SRI 및 공회전(Corotational) 정식화는 데카르트 축 정렬 상태에서는 전단 잠김이 없었으나, 45° 회전 시 상대 전단 샘플링 좌표축이 틀어지며 **전단 잠김이 20배~250배 수준으로 폭발**함.
* **극복 대책**: Simo & Rifai (1990) 정식 EAS 정식화(`CPE4I`)를 통해, 국소 자연좌표계 $(\xi, \eta)$에서 정의된 공변 기저 변환을 사용하여 임의의 회전각에서도 일정한 1.000의 굽힘 강성을 유지하도록 구현.

### [실패 2] 메쉬의 요소 타입 문자열 무시 버그 (AGENTS.md §4.2)
* **원인**: 레거시 2D 솔버는 메쉬에 지정된 `elem_type` 문자열을 읽지 않고 생성자 인자로만 전달받아, 복합 요소 메쉬에서 단일 요소로 강제 실행되던 결함.
* **극복 대책**: 3D와 동일하게 `_setup_numba_topology()`에서 메쉬 요소별 타입을 스캔하여 커널 그룹(`elem_kernel_groups`)으로 자동 분기하는 아키텍처 도입.

### [실패 3] 다종 요소 어셈블리 시 토폴로지 뒤섞임 버그 (2026-09-13 금일 해결)
* **원인**: `create_solver`에서 모델의 `sys.rows_topo`가 솔버의 커널 그룹 정렬 토폴로지를 무단 덮어써 대각 강성이 1/1200로 붕괴.
* **극복 대책**: `create_solver2d` 설계 시 `DynamicSolver2D` 자체의 Numba 커널 그룹 정렬 토폴로지만을 일관되게 사용하도록 보장.
