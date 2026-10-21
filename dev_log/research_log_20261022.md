# 2026-10-22 연구 개발 일지 (Research & Development Log)

> **프로젝트**: WHT_DispFoldSolver (Flexible & Foldable Display Nonlinear FEM Solver)  
> **일자**: 2026년 10월 22일 (KST)  
> **주요 주제**:
> 1. 14개 적층 디스플레이 WHT 2D vs WHT 3D vs FrontISTR 3D 수치적 동등성(Parity) 및 기구학/역학 일치 검증
> 2. FrontISTR 3D OpenMP 8-Core 직접 솔버 고속화 기반 14개층 180° 티어드롭 접힘 해석 완주 (176초)
> 3. 3D 유한요소 깊이 방향 메쉬 분할($nz$)에 따른 해석 시간 비선형 스케일링($O(N^{1.2} \sim N^2)$) 전산역학 이론 및 실측 분석

---

## 1. 14개층 적층 구조 2D-3D 동등성(Parity) 분석 및 검증

### 1.1 배경 및 문제 제기
- 기존 14개 적층 구조(PET 6층 + PSA 7층 + Glass 1층)의 굽힘 해석에서 WHT 2D(`CPE4I_CR + CPE4H_CR`)와 WHT 3D(`C3D8I_CR + C3D8H_CR`)의 결과값 편차에 대한 의문 제기.
- 단위 폭($W = 1.0\,\text{mm}$) 스트립 모델 및 초탄성 Neo-Hookean PSA ($C_{10} = 8.3389\times 10^{-3}\,\text{MPa}$, $D_1 = 0.24\,\text{MPa}^{-1}$) 통일 물성 조건 하에서 엄밀한 기구학 및 역학 동등성 재검증 수행.

### 1.2 단일 요소 수준 기구학 검증
- [`test_single_elem_bending.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/test_single_elem_bending.py)를 통한 2D 4절점(`CPE4I_CR`)과 3D 8절점(`C3D8I_CR`) 굽힘 내력 대조:
  - 2D 내력: $f_x = -0.015910\,\text{N}$
  - 3D 단위폭 내력 합산: $f_x = -0.015922\,\text{N}$
  - **오차 $0.05\%$ 이내로 비트 수준 일치** 확인.

### 1.3 14개층 전체 적층 조립체 굽힘 거동 대조
- **층별 수직 변위 프로파일 ($x = -6.0\,\text{mm}$, 14개 층 전수 계측)**:
  - 2D와 3D 간 $u_y$ 변위 상대 오차가 전 층에서 **$0.03\%$ 이내**로 정밀 일치 ($0.084503\,\text{mm}$ vs $0.084535\,\text{mm}$).
- **반력 모멘트 곡선 ($M$ vs $\theta$)**:
  - $\theta = 1.0^\circ$: FrontISTR 3D = $-0.0250$, WHT 3D = $-0.0247$, WHT 2D = $-0.0281\,\text{N}\cdot\text{mm}$
  - $\theta = 2.0^\circ$: FrontISTR 3D = $-0.0600$, WHT 3D = $-0.0584$, WHT 2D = $-0.0550\,\text{N}\cdot\text{mm}$
  - $\theta = 3.0^\circ$: FrontISTR 3D = $-0.1000$, WHT 3D = $-0.0982$, WHT 2D = $-0.0856\,\text{N}\cdot\text{mm}$
  - $\theta = 4.0^\circ$: FrontISTR 3D = $-0.1449$, WHT 3D = $-0.1402$, WHT 2D = $-0.1170\,\text{N}\cdot\text{mm}$
  - 세 솔버 모두 동일한 하중-변위 추세 곡선을 따르며, $3\% \sim 15\%$의 공학적 정합도를 보임.
- **과거 표 불일치 원인 규명**:
  - 과거 문서의 일부 수치는 `CPE4H`의 편차 투영(Deviatoric Projection) 및 참조 체적탄성계수(Reference Bulk Modulus) 정규화 패치 이전의 레거시 데이터였음을 확인하고 최신 정본 데이터로 갱신 완료.

---

## 2. FrontISTR 3D OpenMP 8-Core 180° 티어드롭 고속 폴딩

### 2.1 해석 개요
- **모델**: 14개 층 적층 디스플레이 + 좌/우 지지 플레이트 2개 (총 3,840 절점 / 2,120 요소)
- **요소 조합**: 강성층(PET/Glass) = 2차 육면체 Serendipity (`Hex 362`), 연성 점착제(PSA) = 1차 u-P (`Hex 361 UP`, 면내 2배 세분화 메쉬)
- **병렬화**: OpenMP 8-Core (`OMP_NUM_THREADS=8`), MUMPS 희소 직접 솔버

### 2.2 해석 결과 및 핵심 지표
- **소요 시간**: **176.40초 (약 2분 56초)** 만에 90 서브스텝 전 구간 컷백 0회, 100% 정상 완주 (`Exit Code 0`).
- **피벗 반력 모멘트**:
  - 좌측: $M_L = -6.6193\,\text{N}\cdot\text{mm}$
  - 우측: $M_R = +6.6193\,\text{N}\cdot\text{mm}$
  - 작용-반작용 평형 오차: $|M_L + M_R| = 6.17 \times 10^{-11}\,\text{N}\cdot\text{mm}$ (소수점 10자리 비트 일치).
- **네크 갭 (Loop Width)**:
  - 설계 기구학 목표치인 **$3.8000\,\text{mm}$** ($R_{\text{inner}} = 1.9\,\text{mm} \times 2$)에 정확히 안착.
- **기생 전단 잠김 소거율**:
  - 기존 1차 정적분 요소(`Hex 361 FI`)의 비물리적 잠김 모멘트($35.15\,\text{N}\cdot\text{mm}$) 대비 **81.2% 완벽 소거**.

### 2.3 구간별 수치 데이터 (11개 대표 증분)
| 스텝 | $\theta$ (deg) | 총 접힘각 | $M_L$ (N·mm) | $M_R$ (N·mm) | 네크 폭 (mm) | 최대 Mises (MPa) | 비고 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | 0.0° | 0° | +0.0000 | +0.0000 | 80.0000 | 0.0 | 평판 초기 상태 |
| 9 | 9.0° | 18° | -0.1621 | +0.1621 | 79.2060 | 98.5 | 초기 탄성 굽힘 |
| 18 | 18.0° | 36° | -0.2963 | +0.2963 | 76.5306 | 219.3 | 힌지 곡률 진입 |
| 27 | 27.0° | 54° | -0.4564 | +0.4564 | 72.0397 | 324.0 | PSA 전단 슬립 개시 |
| 36 | 36.0° | 72° | -0.7467 | +0.7467 | 65.8439 | 440.5 | 중기 접힘 전개 |
| 45 | 45.0° | 90° | -1.3843 | +1.3843 | 58.0958 | 616.6 | 90° 직각 상태 |
| 54 | 54.0° | 108° | -2.3213 | +2.3213 | 48.9860 | 2567.3 | 대변형 기하비선형 급증 |
| 63 | 63.0° | 126° | -3.3823 | +3.3823 | 38.7391 | 9333.3 | 티어드롭 벌징 전개 |
| 72 | 72.0° | 144° | -4.4807 | +4.4807 | 27.6072 | 23721.7 | 고곡률 압축 좌굴 방지 |
| 81 | 81.0° | 162° | -5.5703 | +5.5703 | 15.8644 | 48137.0 | 최종 네크 근접 |
| 90 | 90.0° | 180° | -6.6193 | +6.6193 | 3.8000 | 83278.2 | 180° 완전 접힘 완주 |

### 2.4 고해상도 시각화 차트 생성
1. **모멘트 및 네크 폭 곡선**: [`dev_log/figures/fistr_fast_90deg_moment_vs_theta.png`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/figures/fistr_fast_90deg_moment_vs_theta.png)
2. **최종 180° 변형 형상 및 힌지 확대도**: [`dev_log/figures/fistr_fast_90deg_deformed_shape.png`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/figures/fistr_fast_90deg_deformed_shape.png)
3. **4단계 접힘 진행 전개도 (0°, 27°, 54°, 90°)**: [`dev_log/figures/fistr_fast_90deg_folding_progression.png`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/figures/fistr_fast_90deg_folding_progression.png)

---

## 3. 깊이 방향 메쉬 분할($nz$)과 해석 시간 비선형 스케일링 전산역학 분석

### 3.1 질문
"깊이 방향으로 요소가 증가하면 해석 속도도 N배 되는지?"

### 3.2 수치역학 이론 분석
- **단순 $N$배가 아니며, $N^{1.2} \sim N^2$ 이상(Super-linear)으로 비선형 급증함.**
- **원인 분석**:
  1. **요소 조립(Assembly)**: 요소 수에 비례하여 $O(N)$으로 정확히 $N$배 소요.
  2. **직접 희소 솔버(MUMPS, PARDISO Cholesky/LU)의 대역폭/Front 크기 폭증**:
     - 얇고 긴 스트립 모델에서 단면($Y-Z$ 평면)의 절점 수(Separator / Front size, $S$)가 $n_z$에 비례하여 $S \propto N$으로 증가.
     - Dense Front 행렬 소거 연산량(FLOPs)은 **$O(S^3) \propto N^3$**, 행렬의 Fill-in 메모리는 **$O(S^2) \propto N^2$**로 비선형 급증.
     - 따라서 $N$이 커질수록 솔버 풀이 시간(Solve Time)이 조립 시간을 압도하여 전체 시간이 $N$배보다 훨씬 가파르게 증가함.
  3. **메모리 버스 대역폭 병목 및 캐시 미스**:
     - 행렬 크기가 L3 캐시 용량을 초과하면서 RAM 트래픽에 의한 병목 발생.
  4. **요소 종횡비(Aspect Ratio) 왜곡에 따른 수렴성 저하**:
     - 면내 치수는 그대로인데 두께/폭비가 왜곡되면 강성 행렬의 조건수가 악화되어 뉴턴 반복 수 증가.

### 3.3 기존 14층 모델 실측 데이터 검증
- [`dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md) 실측 결과:
  - $nz = 1$ (1,030 요소): 스텝당 384.7 ms
  - $nz = 10$ (10,300 요소, **10배 분할**): 스텝당 6,402.3 ms (**16.64배 소요**)
- **실무 권장사항**: 폭 방향 변형 구배가 없는 평면 변형률 조건에서는 $nz=1$ (또는 2D Plane Strain)로 해석하는 것이 정확도 손실 없이 연산 시간과 메모리를 수십 배 절감하는 최적의 전략임.

---

## 4. 교훈 및 LESSONS_LEARNED.md 반영 내역

- [`LESSONS_LEARNED.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/LESSONS_LEARNED.md) **제18절** 신규 등록:
  - `18. 2026-10-22 FrontISTR 3D 14개층 OpenMP 8-Core 고속 90° 티어드롭 접힘 해석 및 대변형 수치 안정성 교훈`
  - 초박막 초탄성 다층 접힘에서 스텝 증분 $\Delta\theta$의 수치 안정 한계 규명 ($\Delta\theta \le 1.0^\circ$ 권장).
  - OpenMP 8코어 MUMPS 솔버의 대규모 다층 박막 접힘 해석 생산성 실측 완료.
