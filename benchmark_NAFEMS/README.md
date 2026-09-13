# 🏛️ NAFEMS Standard & Nonlinear Benchmarks Suite (`benchmark_NAFEMS`)

본 디렉토리는 유한요소해석(FEA)의 세계적 공인 표준 기관인 **NAFEMS(National Agency for Finite Element Methods and Standards)**의 공인 벤치마크 테스트 스위트를 현재 솔버(`DynamicSolver2D`, `DynamicSolver3D`) 및 보유 중인 21개 유한요소(2D 10개 + 3D 11개)로 전수 검증하기 위한 전용 패키지입니다.

---

## 1. 벤치마크 구성 및 공식 레퍼런스

### 📘 Standard Benchmarks: Linear Elastic Tests (LE1 ~ LE11)
- **공식 출처**: NAFEMS Target Publication *The Standard NAFEMS Benchmarks (TNSB Rev. 3)*, Abaqus Benchmarks Guide §4.2

| 번호 | 벤치마크 명칭 | 해석 차원 / 범주 | 주요 역학 현상 및 평가 척도 | NAFEMS 공인 정해 | 허용 오차 |
|:---:|:---|:---:|:---|:---:|:---:|
| **`LE1`** | **Elliptic Membrane** | 2D / 3D Solid | 타원 멤브레인 인장집중, 점 D 수직응력 $\sigma_y$ | $92.7\,\text{MPa}$ | $\pm 1.5\%$ |
| **`LE2`** | **Scordelis-Lo Roof** | 3D Solid Shell | 원통형 쉘 자중 처짐, 자유단 중앙 $w$ | $-0.09217\,\text{m}$ | $\pm 2.0\%$ |
| **`LE3`** | **Hemispherical Shell** | 3D Solid Shell | 개구 반구형 쉘 점하중, 반경방향 변위 $\delta$ | $0.185\,\text{m}$ | $\pm 2.0\%$ |
| **`LE4`** | **Thick Cylinder under Pressure** | **2D & 3D 전수** | 후육 원통 Lame 후프응력 $\sigma_\theta(r_i)$ | $166.67\,\text{MPa}$ | $\pm 1.0\%$ |
| **`LE5`** | **Z-Section Cantilever** | 3D Solid Beam | Z-단면 비틀림-전단, 접합부 축응력 $\sigma_x$ | $-108.0\,\text{MPa}$ | $\pm 2.0\%$ |
| **`LE6`** | **Morley 30° Skew Plate** | 3D Solid Plate | 30도 사교판 특이 굽힘, 중앙부 처짐 $w$ | $-0.641\,\text{mm}$ | $\pm 2.5\%$ |
| **`LE7`** | **Axisymmetric Cylinder Thermal**| 3D Solid Thermal | 반경방향 포물선 온도경사, 외경 축응력 $\sigma_z$ | $145.83\,\text{MPa}$ | $\pm 2.0\%$ |
| **`LE8`** | **Axi-symmetric Hyperboloidal** | 3D Solid Shell | 쌍곡면 쉘 링 인장응력 | $1.58\,\text{MPa}$ | $\pm 2.0\%$ |
| **`LE9`** | **Thick Solid Sphere** | 3D Solid Spherical | 후육 구체 3차원 구면 내압, 내경 후프응력 $\sigma_\theta$ | $133.33\,\text{MPa}$ | $\pm 1.5\%$ |
| **`LE10`** | **Thick Plate under Pressure** | **3D & 2D 전수** | 마인들린(Mindlin) 후판 굽힘 전단잠김, 하면 중앙 $w$ | $-0.1106\,\text{mm}$ | $\pm 1.5\%$ |
| **`LE11`** | **Solid Cylinder Parabolic Temp**| 3D Solid Thermal | 원통 중심축 열응력 $\sigma_z(r=0)$ | $-150.0\,\text{MPa}$ | $\pm 2.0\%$ |

---

### 📕 Proposed Nonlinear Benchmarks (NL1 ~ NL7)
- **공식 출처**: NAFEMS Report R0024 *"Selected Benchmarks for Material and Geometric Nonlinearity"*, Abaqus Benchmarks Guide §1.2

| 번호 | 벤치마크 명칭 | 비선형 범주 | 주요 물리 현상 및 평가 지표 | 타겟 정해 및 레퍼런스 | 허용 오차 |
|:---:|:---|:---:|:---|:---:|:---:|
| **`NL1`** | **Large Deflection Cantilever** | 기하 비선형 | 대회전 탄성곡선(Elastica), 끝단 수직 처짐 $w_{\text{tip}}$ | $8.11\,\text{m}$ ($w/L=0.811$, Bisshopp & Drucker) | $\pm 1.0\%$ |
| **`NL2`** | **Circular Plate Large Deflection**| 기하 비선형 | 멤브레인 신장 강성 증가(Membrane Stiffening), 중심 $w$ | $18.42\,\text{mm}$ (Way 1934 / Timoshenko) | $\pm 4.0\%$ |
| **`NL3`** | **Large Deflection Diamond Frame** | 기하 비선형 | 다이아몬드 프레임 / 원환 대회전 압축 변위 $\delta_y$ | $0.450\,\text{m}$ (NAFEMS R0024) | $\pm 5.0\%$ |
| **`NL4`** | **Snap-Through of Shallow Arch** | 기하/안정성 | 한계점 좌굴(Limit-point Instability) 스냅스루 하중 $P_{\text{lim}}$ | $1240.0\,\text{N}$ (Roorda 1965) | $\pm 5.0\%$ |
| **`NL5`** | **Elastic-Plastic Thick Cylinder** | 재료 비선형 | Multiplicative $J_2$ 소성 완전 붕괴 한계 압력 $p_{\text{coll}}$ | $160.08\,\text{MPa}$ (Hill 1950 정해) | $\pm 1.5\%$ |
| **`NL6`** | **3D Cantilever Bending-Torsion** | 기하 비선형 | 3D 공간 대회전-비틀림 연성 변위 놈 $\|u_{\text{tip}}\|$ | $4.65\,\text{m}$ (Argyris & Symeonidis) | $\pm 3.0\%$ |
| **`NL7`** | **Tensile Bar Localized Necking** | 기하+재료 | 유한 변형 소성 국부 네킹, 등가 소성 변형률 $\bar{\varepsilon}^p$ | $1.15$ (Needleman 1972) | $\pm 5.0\%$ |

---

## 2. 평가 대상 유한요소 라이브러리 (총 21개)

### 3D Solid Elements (11개)
- `C3D8` : 완전적분 8절점 1차 6면체
- `C3D8I` : 9-모드 비적합 모드 / EAS 확장 변형률 6면체 (Wilson-Taylor-Simo)
- `C3D8_FBAR` : F-bar 체적 투영 대변형 6면체 (de Souza Neto-Hughes)
- `C3D8_CR` : Polar 분해 Co-Rotational + Hughes B-bar 6면체
- `C3D8H` : Herrmann u-P 혼합 정식화 완전 비압축성 6면체
- `C3D8R` : 1점 감차적분 + Flanagan-Belytschko 아워글래스 제어 6면체
- `C3D4` : 4절점 선형 사면체 (CST)
- `C3D4_ANP` : Bonet-Burton 체적 절점 압력 평균화 (ANP) 사면체
- `C3D10` : 10절점 2차 사면체 (Full 4-point Gauss)
- `C3D10M` : 체적 B-bar + 접촉 아워글래스 안정화 사면체
- `C3D6` : 6절점 선형 쐐기(Wedge/Prism) 요소

### 2D Plane-Strain Elements (10개)
- `CPE4`, `CPE4I`, `CPE4R`, `CPE4H`, `CPE4_FBAR`, `CPE4_CR`, `CPE3`, `CPE6`, `CPE6M`, `CPE8`

---

## 3. 실행 방법 및 CLI 인터페이스

### 전체 통합 벤치마크 및 도면 생성 일괄 실행
```bash
python benchmark_NAFEMS/run_all_nafems.py --mode all
```

### 선형 탄성 LE1 ~ LE11 전용 실행
```bash
python benchmark_NAFEMS/run_nafems_le.py
```

### 비선형 NL1 ~ NL7 전용 실행
```bash
python benchmark_NAFEMS/run_nafems_nl.py
```

### 출판 품질 비교 그래프 재생성
```bash
python benchmark_NAFEMS/figures_nafems.py
```

---

## 4. 결과 산출물 구조

```
benchmark_NAFEMS/
├── results/
│   ├── results_le.json                 # LE1~LE11 전수 수치 결과 및 오차율
│   └── results_nl.json                 # NL1~NL7 비선형 수치 결과 및 하중 이력
├── figures/
│   ├── nafems_le_elements_comparison.png       # LE 종합 4패널 비교 차트
│   └── nafems_nl_benchmarks_comparison.png     # NL 종합 4패널 비교 차트
└── README.md
```
종합 학술 리포트: [`dev_log/benchmark_nafems_20260913.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/benchmark_nafems_20260913.md)
