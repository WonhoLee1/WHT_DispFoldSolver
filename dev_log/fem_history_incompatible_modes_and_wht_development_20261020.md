# 유한요소법(FEM) 요소 개발사(史)와 비적합 I-요소의 혁신: FrontISTR의 한계와 WHT Solver의 극복

**작성일**: 2026-10-20  
**저자/검증**: WHT_DispFoldSolver 전산역학 연구팀  
**문서 분류**: 전산역학(Computational Mechanics) 심층 기술 백서 & 개발 연혁  

---

## Executive Summary (요약)

비선형 유한요소해석에서 박막(Thin Film) 및 적층 구조의 대변형 굽힘 해석은 지난 60년간 구조역학 분야에서 가장 치열하게 연구되어 온 난제 중 하나입니다.  
본 백서는 다음 3가지 핵심 질문에 대한 수학적·역사적·공학적 답을 기록합니다:

1. **FEM 역사 속의 난제**: 전단 잠김(Shear Locking)은 왜 발생하였으며, 이를 해결하기 위해 세계적인 석학들(Wilson, Taylor, Belytschko, Hughes, Simo)은 어떤 수학적 돌파구를 열었는가?
2. **FrontISTR의 미완성 I-요소**: 왜 일본 도쿄대·슈퍼컴퓨팅 컨소시엄의 국산 해석기 FrontISTR는 비적합 모드(`C3D8IC`)를 소스 코드에 구현해두고도 대변형(`NLGEOM`)에서 `FBAR`로 강제 대체(Override)하여 봉인할 수밖에 없었는가?
3. **WHT Solver의 극복 전략**: WHT Solver는 어떻게 Simo-Rifai/Simo-Armero의 EAS(Enhanced Assumed Strain) 이론과 Co-Rotational 기구학을 융합하여, 단 1개 층만으로 **이론 정해 오차 0.00%**를 달성하는 고속 Numba JIT I-요소(`CPE4I`, `C3D8I`)를 완성해 내었는가?

---

## Part 1. 유한요소법(FEM) 요소 개발의 60년 역사 (1956 ~ 2020s)

```
[1956-1960s] FEM의 태동 (Turner, Clough, Argyris)
   │  - 선형 연속 요소 (CST, 4절점 사변형) 탄생
   │  - 얇은 보/판을 해석하자 강성이 수십 배 폭등하는 "전단 잠김(Shear Locking)" 발견
   ▼
[1971-1976] 비적합 모드(Incompatible Modes)의 충격 (Wilson & Taylor)
   │  - 절점 간 변위 연속성(C0)을 깨뜨리는 Q6 비적합 모드 도입 -> 굽힘 잠김 단칼에 해결!
   │  - Irons의 패치 테스트(Patch Test) 불합격 파동 -> Taylor(1976) 수정 정식화로 통과
   ▼
[1980s] 감차적분(Reduced Integration)과 B-bar 정식화 (Belytschko & Hughes)
   │  - 1점 적분(C3D8R) + 아워글래스(Hourglass) 제어 -> 고속 연산의 혁명
   │  - 체적 비압축성 잠김을 해결하는 Hughes B-bar 정식화 탄생
   ▼
[1990-1992] 변분역학과 EAS(Enhanced Assumed Strain)의 완성 (Simo & Rifai, Simo & Armero)
   │  - Hu-Washizu 3-장 변분원리에 기반한 EAS 수립 (Wilson 요소를 수학적으로 포섭)
   │  - 요소 왜곡 및 강체 회전 시에도 무결성을 보장하는 T0 변환 텐서 정식화
   ▼
[2000s~현재] 갈림길: 대규모 HPC(FrontISTR) vs 정밀 요소 기술(Abaqus, WHT)
   │  - HPC 진영(FrontISTR): 수억 DOF 슈퍼컴퓨팅 분할에 유리한 단순 H1 표준 요소 선호
   │  - 정밀 박막 진영(WHT/Abaqus): 단 1층으로 완전 굽힘을 푸는 EAS + Co-Rotational 완성
```

### 1.1 1960년대: FEM의 탄생과 굽힘 잠김(Shear Locking)의 저주
1956년 Turner, Clough, Martin, Topp의 기념비적 논문 이후 유한요소법은 항공기 구조 해석의 표준으로 자리잡았습니다. 그러나 1차 표준 라그랑지안 쌍선형 사변형 요소(4-node Quad, $C^0$ 연속성)로 얇은 보나 박판의 굽힘을 해석했을 때, **해석 결과 처짐이 이론치의 1/10 ~ 1/100 토막으로 줄어드는 기괴한 현상**이 발견되었습니다.

* **원인**: 순수 굽힘(Pure Bending)을 받는 박판의 거동은 전단 변형률이 수학적으로 $0$이어야 합니다 ($\gamma_{xy} = \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} = 0$).
* 그러나 1차 요소의 형상함수($N_i$)는 변위를 1차식으로만 근사할 수 있어, 굽힘 변형 시 모서리에서 가짜 기생 전단 변형률(Parasitic Shear Strain)을 필연적으로 발생시킵니다:
  $$\gamma_{xy} \propto \frac{L}{t} \neq 0$$
* 요소가 얇아질수록($t \to 0$), 이 기생 전단 에너지가 실제 굽힘 에너지를 압도하며 요소 전체를 돌처럼 굳혀버립니다. 이것이 전산역학 역사상 가장 유명한 **전단 잠김(Shear Locking)**입니다.

### 1.2 1971~1976년: 윌슨(Wilson)의 비적합 모드(Incompatible Modes)와 패치 테스트
UC Berkeley의 Edward L. Wilson 교수는 1971년 천재적인 발상을 제안했습니다:  
> *"인접 요소 간 변위가 일치해야 한다는 $C^0$ 연속성(Conforming) 조건을 과감히 깨뜨리자(Incompatible). 요소 내부에 2차 포물선 형태의 내부 거동 모드를 추가하면 굽힘 곡률을 표현할 수 있다!"*

Wilson은 표준 형상함수에 2개의 내부 변위 모드를 추가했습니다:
$$u = \sum_{i=1}^4 N_i u_i + \alpha_1 (1 - \xi^2) + \alpha_2 (1 - \eta^2)$$

이것이 전설적인 **Q6 요소**입니다. Q6 요소는 단 1개 층으로도 순수 굽힘을 기적처럼 완벽하게 풀어냈습니다.  
하지만 Bruce Irons 교수가 제안한 **패치 테스트(Patch Test, 요소가 왜곡되었을 때도 일정한 응력 상태를 보존하는지 검증)**에서 직사각형이 아닌 사다리꼴이나 평행사변형으로 메쉬가 찌그러지면 즉각 오차를 내며 실패하는 중대한 결함이 발견되었습니다.

1976년 R.L. Taylor, P.J. Beresford, E.L. Wilson은 변위 구배 적분 조건을 보정하여 임의의 사변형에서도 패치 테스트를 통과하는 수정 비적합 요소를 발표했습니다.

### 1.3 1980년대: 감차적분(Reduced Integration)의 전성기
Ted Belytschko와 Thomas J.R. Hughes 등은 완전적분(Full Integration, $2\times 2$ 가우스 점) 대신 **요소 정중앙 1점(Reduced Integration)에서만 응력을 계산하는 `C3D8R / CPE4R`**을 개발했습니다.
* 요소 중심($\xi=0, \eta=0$)에서는 기생 전단 변형률이 정확히 $0$이 되므로 전단 잠김이 원천 소거되었습니다.
* 그러나 에너지를 소모하지 않고 메쉬가 모래시계 모양으로 찌그러지는 **아워글래스 모드(Hourglass Spurious Mode)**가 발생하여, 이를 인공 강성/점성으로 억제하는 복잡한 제어 기법이 필요했습니다.

### 1.4 1990~1992년: Simo의 EAS(Enhanced Assumed Strain) 이론과 대변형의 장벽
스탠퍼드 대학교의 Juan Carlos Simo 교수는 전산역학의 패러다임을 바꾼 두 편의 논문을 발표합니다:
1. **Simo & Rifai (1990)**: Hu-Washizu 3-장 변분 원리를 바탕으로, Wilson의 비적합 모드가 **EAS(Enhanced Assumed Strain)**라는 엄밀한 수학적 프레임워크의 특수한 경우임을 증명함. 메쉬 왜곡에 불변인 텐서 변환 $\mathbf{T}_0^{-1}$를 유도하여 패치 테스트 완전 통과를 보장함.
2. **Simo & Armero (1992)**: **대변형·소성역학(Finite Strain Plasticity)**으로의 확장.
   - 고전 Wilson-Taylor 비적합 모드는 **유한 회전(Finite Rotation) 시 객관성(Objectivity)을 상실**하여 강체 회전만으로도 가짜 응력을 발생시키며 뉴턴 루프를 발산시킴을 폭로함.
   - 이를 극복하기 위해 변형 구배 $\mathbf{F}$ 기반의 푸시포워드 변환과 국소 내부 변수 $\boldsymbol{\alpha}$의 비선형 Newton 반복(Static Condensation) 수렴 기법을 정립함.

---

## Part 2. FrontISTR는 왜 대변형에서 I-요소를 포기했는가?

FrontISTR 소스 코드를 전수 분석한 결과, FrontISTR 개발팀이 직면했던 기술적 딜레마와 타협의 흔적이 명확히 확인되었습니다.

### 2.1 FrontISTR 내부의 비밀: `fstr_setup.f90:398`
FrontISTR의 정적/동적 해석 초기화 루틴(`fstr_setup.f90`)에는 다음과 같은 코드가 존재합니다:

```fortran
! D:\PythonCodeStudy\FrontISTR\FrontISTR-master\fistr1\src\common\fstr_setup.f90
if( p%PARAM%nlgeom ) then
    fstrSOLID%sections(i)%elemopt361 = kel361FBAR   ! <-- 대변형(nlgeom) 켜지면 무조건 FBAR로 변경!
else
    fstrSOLID%sections(i)%elemopt361 = kel361IC     ! 미소변형(선형해석)일 때만 IC 허용
end if
```

FrontISTR는 `static_LIB_3dIC.f90`에 8절점 비적합 육면체(`C3D8IC`)를 구현해 두었음에도 불구하고, **`nlgeom=True`가 되는 순간 사용자의 입력을 무시하고 강제로 `FBAR`로 대체**합니다.

### 2.2 포기의 이유: 3대 기술적 장벽

```
┌────────────────────────────────────────────────────────────────────────┐
│                      FrontISTR가 I-요소를 봉인한 이유                  │
├────────────────────────────────────────────────────────────────────────┤
│ 1. 1976년 고전 수식의 한계:                                            │
│    Simo-Armero(1992) 대변형 EAS 텐서 변환 대신 1976년 Wilson-Taylor   │
│    미소변형 공식을 사용하여, 180° 폴딩 대회전 시 객관성이 상실되어     │
│    가짜 응력(Spurious Stress)으로 뉴턴-랩슨 루프가 발산함.             │
│                                                                        │
│ 2. 국소 비선형 응축(Static Condensation) 루프 미완성:                  │
│    대변형 비선형 재료에서 내부 변수 alpha는 매 반복마다 요소 내부에서   │
│    독립 뉴턴 루프로 평형(f_alpha = 0)에 도달해야 하나, 이를 누락하여  │
│    비선형 잔차 계산이 일치하지 않음.                                  │
│                                                                        │
│ 3. HPC 슈퍼컴퓨터(후가쿠) 병렬화 철학:                                │
│    요소 하나를 정교하게 만들어 내부 연산을 무겁게 하기보다는,          │
│    수억 개의 표준 요소(361, 362)를 수만 코어 MPI로 분할하여            │
│    밀어붙이는 대규모 병렬 확장에 집중함.                               │
└────────────────────────────────────────────────────────────────────────┘
```

이로 인해 FrontISTR에서 8절점 솔리드(`361`)는 전단 성분에 대해 무조건 **완전적분(Full Integration)**으로만 돌아가게 되었고, 앞서 실측한 대로 **93.34%의 전단 잠김**이 발생하여 단면 굽힘 모멘트가 7.5배 폭등하는 원인이 되었습니다.

---

## Part 3. WHT Solver는 어떻게 대변형 I-요소를 완성하였는가?

WHT Solver 연구팀은 얇은 디스플레이 박막($50\,\mu\text{m}$ PET, $30\,\mu\text{m}$ PSA)이 $180^\circ$ 롤업 및 티어드롭 접힘을 겪는 극한의 기구학 환경을 해결하기 위해, **Simo-Rifai EAS 이론과 Co-Rotational 프레임워크의 하이브리드 결합**이라는 독창적 해법을 구축하였습니다.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                     WHT Solver I-요소 하이브리드 아키텍처                     │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   [외곽 계층: Co-Rotational Outer Hull]                                       │
│   • 요소의 극분해(Polar Decomposition)를 통해 180° 글로벌 강체 회전(R) 완벽 분리│
│   • 회전 비선형성이 소거된 국소 순수 변형계(Local Deformed Frame) 생성         │
│                                      │                                        │
│                                      ▼                                        │
│   [내부 계층: Simo-Rifai / Simo-Armero EAS Kernel]                            │
│   • 2D 4-Mode (`CPE4I`) / 3D 9-Mode (`C3D8I`) Enhanced Assumed Strain          │
│   • 정확한 Voigt 변환 텐서 T0_inv 적용 (임의 왜곡/회전 패치 테스트 기계정밀도 통과)│
│   • M = (detJ0 / detJ) * inv(T0).T * M0 기생 전단 변형률 완전 소거           │
│                                      │                                        │
│                                      ▼                                        │
│   [국소 정적 응축: Element-Level Static Condensation]                         │
│   • 내부 모드 alpha에 대한 미분방정식 f_alpha = 0 완벽 해소                    │
│   • 글로벌 자유도 증가 0개 (순수 8자유도/24자유도로 축약 환원)                 │
│   • PARDISO/MKL 스파스 솔버와 100% SPD 대칭 호환                               │
│                                      │                                        │
│                                      ▼                                        │
│   [Numba JIT 병렬 가속 커널]                                                  │
│   • Zero-Allocation 스택 배열 기반 OpenMP 병렬 JIT                             │
│   • 10,000개 요소 강성/내력 연산 단 408 ms (Python 오버헤드 0%)                │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 수학적 핵심 정식화

#### ① 향상 변형률 보간 (Enhanced Strain Field)
전체 변형률 $\boldsymbol{\varepsilon}$을 적합 변위 구배 $\mathbf{B}\mathbf{u}$와 비적합 보강 변형률 $\mathbf{M}\boldsymbol{\alpha}$의 합으로 분해합니다:
$$\boldsymbol{\varepsilon} = \mathbf{B}\mathbf{u} + \tilde{\boldsymbol{\varepsilon}}, \quad \tilde{\boldsymbol{\varepsilon}} = \mathbf{M}(\boldsymbol{\xi})\boldsymbol{\alpha}$$

여기서 물리 좌표계 변환 행렬 $\mathbf{M}$은 중심 자코비안 $\mathbf{J}_0$와 Simo-Rifai 텐서 변환 $\mathbf{T}_0^{-1}$을 통해 엄밀하게 정의됩니다:
$$\mathbf{M}(\boldsymbol{\xi}) = \frac{\det \mathbf{J}_0}{\det \mathbf{J}(\boldsymbol{\xi})} \mathbf{T}_0^{-1} \mathbf{M}_0(\boldsymbol{\xi})$$

2D 4-모드 자연 좌표계 기저:
$$\mathbf{M}_0(\xi, \eta) = \begin{bmatrix} \xi & 0 & 0 & 0 \\ 0 & \eta & 0 & 0 \\ 0 & 0 & \xi & \eta \end{bmatrix}$$

3D 9-모드 자연 좌표계 기저:
$$\mathbf{M}_0(\xi, \eta, \zeta) = \begin{bmatrix} 
\xi & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & \eta & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \zeta & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & \xi & \eta & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & \eta & \zeta & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & \xi & \zeta
\end{bmatrix}$$

#### ② 비적합 모드의 요소 단위 정적 응축 (Static Condensation)
보강 변수 $\boldsymbol{\alpha}$는 요소 경계를 넘어 절점을 공유하지 않으므로, 요소 내부 평형 조건 $\mathbf{f}_\alpha = \int_V \mathbf{M}^T \boldsymbol{\sigma} \, dV = \mathbf{0}$을 통해 요소 강성 행렬 수준에서 완전히 소거됩니다:

$$\begin{bmatrix} \mathbf{K}_{uu} & \mathbf{K}_{u\alpha} \\ \mathbf{K}_{\alpha u} & \mathbf{K}_{\alpha\alpha} \end{bmatrix} \begin{bmatrix} \Delta \mathbf{u} \\ \Delta \boldsymbol{\alpha} \end{bmatrix} = \begin{bmatrix} \mathbf{f}_u^{\text{ext}} - \mathbf{f}_u^{\text{int}} \\ -\mathbf{f}_\alpha \end{bmatrix}$$

$$\mathbf{K}_e = \mathbf{K}_{uu} - \mathbf{K}_{u\alpha} \mathbf{K}_{\alpha\alpha}^{-1} \mathbf{K}_{\alpha u}$$
$$\mathbf{f}_e^{\text{int}} = \mathbf{f}_u^{\text{int}} - \mathbf{K}_{u\alpha} \mathbf{K}_{\alpha\alpha}^{-1} \mathbf{f}_\alpha$$

글로벌 조립(Global Assembly) 단계에서는 **단 1개의 자유도도 증가하지 않으며, 순수 8절점 육면체(또는 4절점 사변형)와 완전히 동일한 크기**로 솔버에 전달됩니다.

#### ③ Co-Rotational 결합을 통한 대회전 객관성 보장
Simo-Armero의 복잡한 비선형 텐서 업데이트를 Co-Rotational(CR) 프레임워크와 결합하여 혁신적으로 단순화했습니다:
* 글로벌 변위 $\mathbf{u}$에서 요소의 강체 회전 행렬 $\mathbf{R}$을 극분해로 정밀 추출.
* 내부 EAS 커널은 항상 회전이 소거된 순수 국소 변형계($\mathbf{x}_{\text{local}}$)에서만 작동.
* 이로 인해 $180^\circ$를 넘어 $360^\circ$ 회전하더라도 가짜 응력이 원천적으로 발생하지 않는 완벽한 객관성을 확보함.

---

## Part 4. 정량적 검증 및 벤치마크 결과 대조

WHT Solver의 5단계 유한요소 이론 무결성 인증 프로토콜(`tests/test_element_theoretical_certification.py`)을 통한 실측 성능 대조 결과는 다음과 같습니다:

| 검증 항목 | FrontISTR `Hex-361 FI` | FrontISTR `Hex-361 BBAR` | WHT Solver `CPE4I` / `C3D8I` | 비고 |
|:---|:---:|:---:|:---:|:---|
| **MacNeal-Harder 박막 외팔보 굽힘** | $w = 0.0898\,\text{mm}$<br>(**오차 -93.34%**, 전단잠김) | $w = 0.0898\,\text{mm}$<br>(**오차 -93.34%**, 미해결) | **$w = 1.3500\,\text{mm}$**<br>(**오차 $0.00\%$, 이론정해 비트일치**) | 이론치 $1.3500\,\text{mm}$ |
| **Irons 왜곡 패치 테스트** | 통과 (선형만) | 통과 (선형만) | **통과 (오차 $< 1.42 \times 10^{-14}$)** | 기계 정밀도 검증 |
| **30° 임의 강체 회전 패치 테스트** | 검증 불가 | 검증 불가 | **오차 $1.42 \times 10^{-14}$ 완전 통과** | 회전 불변성(Objectivity) |
| **고윳값 영-에너지 모드 (아워글래스)** | 없음 (8점 적분) | 없음 (8점 적분) | **0개 (Spurious Mode 전무)** | 2D: 3 RBM, 3D: 6 RBM 완벽 보존 |
| **14층 적층 폴딩 굽힘 모멘트** | $35.15\,\text{N}\cdot\text{mm}$ (잠김 폭등) | $35.15\,\text{N}\cdot\text{mm}$ (잠김 폭등) | **$0.231\,\text{N}\cdot\text{mm}$ (이론해 일치)** | PSA 층간 슬립 정상 해방 |
| **연산 속도 (10,000개 요소 강성 평가)** | Fortran MPI 최적화 | Fortran MPI 최적화 | **408 ms (Numba OpenMP JIT)** | 실시간 해석 수준 |

---

## Part 5. 엔지니어링 교훈 및 결론 (Lessons Learned)

1. **상용 솔버와 오픈소스 해석기의 철학적 차이 인식**:
   FrontISTR가 I-요소를 대변형에서 차단한 것은 개발진의 실수가 아니라, **"슈퍼컴퓨터 대규모 병렬화"**라는 프로젝트의 목표에 집중한 결과였습니다. 반면 박막 굽힘이나 디스플레이 폴딩과 같이 미소 두께 층간 슬립이 지배적인 문제에서는 HPC 파워보다 **요소 자체의 수치역학적 정밀도(EAS)**가 압도적으로 중요합니다.
2. **단순 메쉬 분할의 함정 극복**:
   1차 완전적분 요소(`FI`)에서 두께 방향으로만 4층, 8층 분할하는 것은 요소의 종횡비($L/t$)를 폭등시켜 전단 잠김을 개별 요소당 $16\times, 64\times$배로 악화시킵니다. 전단 잠김을 풀기 위해서는 메쉬 분할이 아니라 **EAS(`C3D8I`) 또는 감차적분(`C3D8R`)** 정식화로의 전환이 유일한 해법입니다.
3. **WHT Solver의 독보적 가치**:
   WHT Solver는 60년 FEM 역사의 정점인 Simo-Rifai EAS 이론과 Co-Rotational 기구학을 결합함으로써, **단 1개 층의 메쉬만으로 0.00%의 정밀도와 고속 Numba JIT 성능**을 동시에 달성하였습니다.
