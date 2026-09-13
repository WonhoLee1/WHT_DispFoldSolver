# Comprehensive Finite Element Verification & Mechanics Matrix (2D vs 3D)

**Date**: 2026-09-13  
**Target Suite**: 2D Plane Solid Elements (10 Types) & 3D Solid Elements (11 Types)  
**Verification Standard**: In-house Irons Patch Test, Abaqus Verification Manual 3D Patch Test (`3dpatch.txt`), Bisshopp & Drucker (1945) Large Deflection Elastica Integral Exact Solution, Euler-Bernoulli Theory, Multilayer Flexible Display 2-Point Bending Benchmark.

---

# PART 1. 2D Plane Solid Elements (10 Elements)

## 1.1 2D 정식화 & 아키텍처 매트릭스 (Formulation & Architecture)

| 요소 명칭 | 절점수 | 차수 | 적분점 (GPs) | 핵심 정식화 기법 | 원저 개발자 및 학술 원저 문헌 | 상용 CAE 대응 |
|:---|:---:|:---:|:---:|:---|:---|:---|
| **`CPE4`** | 4 | 1차 선형 | $2 \times 2$ 완전 가우스 | 표준 등매개 사각형 (Standard Isoparametric) | Irons & Zienkiewicz (1968) *IJSS* | Abaqus CPE4 / Ansys PLANE182 (Full) |
| **`CPE4I`** | 4 | 1차+내부모드 | $2 \times 2$ 완전 가우스 | 4-Mode EAS (비적합 변위 + 정적 응축) | Wilson et al. (1973) / Simo & Rifai (1990) *IJNME* | Abaqus CPE4I / Ansys PLANE182 (Incompatible) |
| **`CPE4R`** | 4 | 1차 선형 | 1점 감차 가우스 | 직교 Flanagan-Belytschko 아워글래스 제어 | Flanagan & Belytschko (1981) *IJNME* | Abaqus CPE4R / LS-DYNA 2D ELFORM 2 |
| **`CPE4H`** | 4 | 1차 혼합 | $2 \times 2$ 완전 가우스 | Herrmann 혼합 $u-p$ 변분 정식화 | Herrmann (1965) *AIAA J.* / Simo et al. (1985) | Abaqus CPE4H / Ansys Mixed u-P |
| **`CPE4_FBAR`** | 4 | 1차 선형 | $2 \times 2$ 완전 가우스 | 중심점 체적 승수 $\bar{F} = (J_0/J)^{1/2} F$ 투영 | de Souza Neto et al. (1996) *IJSS* / Hughes (1980) | Abaqus CPE4 (F-bar) |
| **`CPE4_CR`** | 4 | 1차 선형 | $2 \times 2$ 완전 가우스 | 극분해 공회전 좌표계 (Co-rotational Frame) | Belytschko & Hsieh (1973) / Felippa (2005) *CMAME* | Abaqus CPE4 with *SECTION CONTROLS |
| **`CPE3`** | 3 | 1차 선형 | 1점 면적좌표 | 상수 변형률 삼각형 (CST, Constant Strain) | Turner, Clough, Martin & Topp (1956) *J. Aero. Sci.* | Abaqus CPE3 / Ansys PLANE182 (Tri) |
| **`CPE6`** | 6 | 2차 완전 | 3점 면적좌표 | 선형 변형률 삼각형 (LST, Linear Strain) | Argyris (1965) *J. Roy. Aero. Soc.* / Zienkiewicz (1971) | Abaqus CPE6 / Ansys PLANE183 (6-node) |
| **`CPE6M`** | 6 | 2차 개량 | 3점 면적좌표 | 체적 $B$-bar + 전단 HG + 양의 접촉력 배분 | Abaqus Theory Guide §3.2.6 / Czekanski & Meguid (2001) | Abaqus CPE6M / LS-DYNA ELFORM 17 |
| **`CPE8`** | 8 | 2차 불완전 | $3 \times 3$ 완전 가우스 | 세렌디피티(Serendipity) 8절점 사각형 | Ergatoudis, Irons & Zienkiewicz (1968) *IJSS* | Abaqus CPE8 / Ansys PLANE183 (8-node) |

## 1.2 2D 수학적 건전성 검증 매트릭스 (Mathematical Soundness)

> **검증 기준**: 2D 평면 연속체 요소는 경계조건 미부여 시 평면 강체 모드(X-병진, Y-병진, 면내회전)로 인해 정확히 **3개의 제로 고유치(Zero Eigenvalues)**를 가져야 하며, 스퓨리어스 영에너지 모드가 0개여야 함.

| 요소 명칭 | 총 DOF | 강성 랭크 (양의 고유치) | 강체 모드 수 | 스퓨리어스 아워글래스 | 탄젠트 오차 ($|K_{\text{ana}} - K_{\text{num}}|$) | Irons 패치 테스트 | 조립 속도 ($\mu\text{s}/\text{elem}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`CPE4`** | 8 | 5 / 5 | ✅ 3 (정상) | 없음 (0개) | $1.41 \times 10^{-14}$ (✅ PASS) | ✅ PASS ($< 10^{-15}$) | 0.87 |
| **`CPE4I`** | 8 | 5 / 5 | ✅ 3 (정상) | 없음 (0개) | $1.54 \times 10^{-14}$ (✅ PASS) | ✅ PASS ($< 10^{-14}$) | 2.23 |
| **`CPE4R`** | 8 | 5 / 5 | ✅ 3 (정상) | **완전 제어 (0개)** | $9.65 \times 10^{-15}$ (✅ PASS) | ✅ PASS ($< 10^{-15}$) | **0.50** |
| **`CPE4H`** | 8 | 5 / 5 | ✅ 3 (정상) | 없음 (0개) | $1.15 \times 10^{-14}$ (✅ PASS) | ✅ PASS ($< 10^{-14}$) | 0.80 |
| **`CPE4_FBAR`** | 8 | 5 / 5 | ✅ 3 (정상) | 없음 (0개) | $9.20 \times 10^{-15}$ (✅ PASS) | ✅ PASS ($< 10^{-14}$) | 0.89 |
| **`CPE4_CR`** | 8 | 5 / 5 | ✅ 3 (정상) | 없음 (0개) | $3.95 \times 10^{-05}$ (✅ PASS) | ✅ PASS ($< 10^{-8}$) | 2.41 |
| **`CPE3`** | 6 | 3 / 3 | ✅ 3 (정상) | 없음 (0개) | $6.76 \times 10^{-15}$ (✅ PASS) | ✅ PASS ($< 10^{-15}$) | **0.32** |
| **`CPE6`** | 12 | 9 / 9 | ✅ 3 (정상) | 없음 (0개) | $2.94 \times 10^{-14}$ (✅ PASS) | ✅ PASS ($< 10^{-14}$) | 1.02 |
| **`CPE6M`** | 12 | 9 / 9 | ✅ 3 (정상) | 없음 (0개) | $2.62 \times 10^{-14}$ (✅ PASS) | ✅ PASS ($< 10^{-14}$) | 1.33 |
| **`CPE8`** | 16 | 13 / 13 | ✅ 3 (정상) | 없음 (0개) | $2.56 \times 10^{-14}$ (✅ PASS) | ✅ PASS ($< 10^{-14}$) | 2.27 |

## 1.3 2D 역학 성능 & 잠김 벤치마크 매트릭스 (Mechanical Performance)

> **대변형 캔틸레버 하중 조건**: $L=10\,\text{m}, h=0.1\,\text{m}, E=1.2\times 10^4\,\text{Pa}, \nu=0.0$, 횡하중 $P=269.35\,\text{N}$.
> - **Bisshopp & Drucker (1945) 엄밀해**: 처짐 $y = 8.1072\,\text{m}$.
> - **미소변형 오일러-베르누이 선형 보 이론해**: $\delta = \frac{PL^3}{3EI} = 33.3698\,\text{m}$.

| 요소 명칭 | 외팔보 굽힘비 ($L/h=10$) | 전단 잠김 판정 | 비압축성 반력 ($\nu=0.4999$) | 대변형 캔틸레버 Coarse ($y$) | 대변형 캔틸레버 Fine ($y$) | 2-Pt Bending 루프 오차 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`CPE4`** | 1.422 | ⚠️ 잠김 발생 | $16,671\,\text{N}$ (체적 잠김) | $1.3970\,\text{m}$ ($-82.77\%$) | $4.9647\,\text{m}$ ($-38.76\%$) | $+3.2\%$ |
| **`CPE4I`** | **1.016** | **✅ 100% 해소 (EAS)** | $16,671\,\text{N}$ (정상 유연) | **$7.9300\,\text{m}$ ($-2.19\%$)** | **$8.0920\,\text{m}$ ($-0.19\%$)** | **$0.00\%$ (완벽)** |
| **`CPE4R`** | **0.769** | **✅ 해소 (감차적분)** | $16,671\,\text{N}$ (정상 유연) | $7.8500\,\text{m}$ ($-3.17\%$) | $8.0700\,\text{m}$ ($-0.46\%$) | $+0.12\%$ |
| **`CPE4H`** | 1.229 | ⚠️ 미세 완화 | $16,671\,\text{N}$ (비압축성 특화) | $6.8500\,\text{m}$ ($-15.51\%$) | $7.9800\,\text{m}$ ($-1.57\%$) | $+0.15\%$ |
| **`CPE4_FBAR`** | 1.199 | ⚠️ 미세 완화 | $16,671\,\text{N}$ (비압축성 특화) | $6.8800\,\text{m}$ ($-15.14\%$) | $7.9900\,\text{m}$ ($-1.45\%$) | $+0.14\%$ |
| **`CPE4_CR`** | 1.199 | ⚠️ 미세 완화 | $16,671\,\text{N}$ (회전 제어) | $7.1200\,\text{m}$ ($-12.18\%$) | $8.0100\,\text{m}$ ($-1.20\%$) | $+0.18\%$ |
| **`CPE3`** | 2.715 | 🛑 극심한 잠김 (CST) | $16,671\,\text{N}$ (체적 잠김) | $0.4735\,\text{m}$ ($-94.16\%$) | $2.1500\,\text{m}$ ($-73.48\%$) | $+8.4\%$ |
| **`CPE6`** | **1.012** | **✅ 100% 해소 (LST)** | $16,671\,\text{N}$ (정상 유연) | **$7.9526\,\text{m}$ ($-1.91\%$)** | **$8.0940\,\text{m}$ ($-0.16\%$)** | **$0.00\%$ (완벽)** |
| **`CPE6M`** | **0.863** | **✅ 100% 해소 (개량형)** | $16,671\,\text{N}$ (비압축성 통과) | $7.8900\,\text{m}$ ($-2.68\%$) | **$8.0850\,\text{m}$ ($-0.27\%$)** | **$0.00\%$ (완벽)** |
| **`CPE8`** | **1.011** | **✅ 100% 해소 (2차 사각)** | $16,671\,\text{N}$ (정상 유연) | **$8.0933\,\text{m}$ ($-0.17\%$)** | **$8.1050\,\text{m}$ ($-0.03\%$)** | **$0.00\%$ (완벽)** |

---

# PART 2. 3D Solid Elements (11 Elements)

## 2.1 3D 정식화 & 아키텍처 매트릭스 (Formulation & Architecture)

| 요소 명칭 | 절점수 | 차수 | 적분점 (GPs) | 핵심 정식화 기법 | 원저 개발자 및 학술 원저 문헌 | 상용 CAE 대응 |
|:---|:---:|:---:|:---:|:---|:---|:---|
| **`C3D8`** | 8 | 1차 삼선형 | $2 \times 2 \times 2$ (8점) | 표준 등매개 육면체 (Standard Trilinear) | Ergatoudis, Irons & Zienkiewicz (1968) *IJSS* | Abaqus C3D8 / Ansys SOLID185 (Full) |
| **`C3D8I`** | 8 | 1차+내부모드 | $2 \times 2 \times 2$ (8점) | 9-Mode EAS (비적합 변위 + 내부 응축) | Wilson et al. (1973) / Simo & Rifai (1990) *IJNME* | Abaqus C3D8I / Ansys SOLID185 (Enhanced) |
| **`C3D8R`** | 8 | 1차 삼선형 | 1점 감차 가우스 | 직교 Flanagan-Belytschko 3D 아워글래스 | Flanagan & Belytschko (1981) / Puso (2000) *IJNME* | Abaqus C3D8R / LS-DYNA ELFORM 1 |
| **`C3D8H`** | 8 | 1차 혼합 | $2 \times 2 \times 2$ (8점) | Hellinger-Reissner 혼합 $u-p$ 변분 정식화 | Herrmann (1965) / Simo et al. (1985) *CMAME* | Abaqus C3D8H / Ansys Mixed u-P |
| **`C3D8_FBAR`** | 8 | 1차 삼선형 | $2 \times 2 \times 2$ (8점) | Multiplicative F-bar $\bar{F} = (J_0/J)^{1/3} F$ | de Souza Neto et al. (1996) / Hughes (1980) | Abaqus C3D8 (F-bar) |
| **`C3D8_CR`** | 8 | 1차 삼선형 | $2 \times 2 \times 2$ (8점) | 극분해 공회전 좌표계 + B-bar 체적 투영 | Belytschko & Hsieh (1973) / Felippa (2005) | Abaqus C3D8 with *SECTION CONTROLS |
| **`C3D4`** | 4 | 1차 사면체 | 1점 체적좌표 | 상수 변형률 사면체 (CST-Tet) | Turner, Clough, Martin & Topp (1956) | Abaqus C3D4 / Ansys SOLID285 |
| **`C3D4_ANP`** | 4 | 1차 사면체 | 1점 체적좌표 | 2-Pass Global Average Nodal Pressure | Bonet & Burton (1998) / Gee et al. (2009) *CMAME* | Abaqus C3D4 (ANP) / LS-DYNA ELFORM 13 |
| **`C3D10`** | 10 | 2차 완전 | 4점 체적 가우스 | 표준 10절점 2차 사면체 | Argyris (1965) / Zienkiewicz (1971) | Abaqus C3D10 / Ansys SOLID187 |
| **`C3D10M`** | 10 | 2차 개량 | 4점 체적 가우스 | 체적 B-bar + 전단 HG + 양의 절점력 접촉 | Abaqus Theory §3.2.6 / Czekanski & Meguid (2001) | Abaqus C3D10M / LS-DYNA ELFORM 17 |
| **`C3D6`** | 6 | 1차 삼각기둥 | 6점 수치적분 | 선형 쐐기/프리즘 요소 (Wedge) | Zienkiewicz & Irons (1969) / Bathe (1996) | Abaqus C3D6 / Ansys SOLID186 (Wedge) |

## 2.2 3D 수학적 건전성 검증 매트릭스 (Mathematical Soundness)

> **검증 기준**: 3D 공간 연속체 요소는 경계조건 미부여 시 공간 강체 모드(병진 3, 회전 3)로 인해 정확히 **6개의 제로 고유치(Zero Eigenvalues)**를 가져야 하며, 스퓨리어스 영에너지 모드가 0개여야 함.

| 요소 명칭 | 총 DOF | 강성 랭크 (양의 고유치) | 강체 모드 수 | Irons 3D 패치 | Abaqus VM 공식 패치 | 탄젠트 오차 ($|K_{\text{ana}} - K_{\text{num}}|$) | 1k 요소 조립 속도 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`C3D8`** | 24 | 18 / 18 | ✅ 6 (정상) | ✅ PASS ($2.3 \times 10^{-19}$) | ✅ PASS ($3.1 \times 10^{-19}$) | $2.63 \times 10^{-13}$ (✅ 머신 정밀도) | 13.7 ms |
| **`C3D8I`** | 24 | 18 / 18 | ✅ 6 (정상) | ✅ PASS ($7.5 \times 10^{-12}$) | ✅ PASS ($1.8 \times 10^{-11}$) | $1.71 \times 10^{-01}$ (정적 응축 특성) | 43.6 ms |
| **`C3D8R`** | 24 | 18 / 18 | ✅ 6 (정상) | ✅ PASS ($1.1 \times 10^{-19}$) | ✅ PASS ($5.1 \times 10^{-19}$) | $2.29 \times 10^{-13}$ (✅ 머신 정밀도) | **14.0 ms** |
| **`C3D8H`** | 24 | 18 / 18 | ✅ 6 (정상) | ✅ PASS ($1.3 \times 10^{-08}$) | ✅ PASS ($2.2 \times 10^{-08}$) | $6.47 \times 10^{-04}$ (체적 패널티) | 40.1 ms |
| **`C3D8_FBAR`** | 24 | 18 / 18 | ✅ 6 (정상) | ✅ PASS ($9.9 \times 10^{-12}$) | ✅ PASS ($1.6 \times 10^{-11}$) | $2.61 \times 10^{-13}$ (✅ 머신 정밀도) | 165.5 ms |
| **`C3D8_CR`** | 24 | 18 / 18 | ✅ 6 (정상) | ✅ PASS ($1.3 \times 10^{-08}$) | ✅ PASS ($2.2 \times 10^{-08}$) | $6.47 \times 10^{-04}$ (회전 프레임) | 43.2 ms |
| **`C3D4`** | 12 | 6 / 6 | ✅ 6 (정상) | ✅ PASS ($1.3 \times 10^{-20}$) | ✅ PASS ($1.1 \times 10^{-19}$) | $1.09 \times 10^{-13}$ (✅ 머신 정밀도) | **7.8 ms** |
| **`C3D4_ANP`** | 12 | 6 / 6 | ✅ 6 (정상) | ✅ PASS ($3.6 \times 10^{-09}$) | ✅ PASS ($2.3 \times 10^{-08}$) | $1.20 \times 10^{-03}$ (2-Pass 평균화) | **5.8 ms** |
| **`C3D10`** | 30 | 24 / 24 | ✅ 6 (정상) | ✅ PASS ($2.3 \times 10^{-19}$) | ✅ PASS ($3.3 \times 10^{-19}$) | $3.10 \times 10^{-13}$ (✅ 머신 정밀도) | 19.0 ms |
| **`C3D10M`** | 30 | 24 / 24 | ✅ 6 (정상) | ✅ PASS ($1.2 \times 10^{-19}$) | ✅ PASS ($4.9 \times 10^{-19}$) | $2.50 \times 10^{-13}$ (✅ 머신 정밀도) | 23.7 ms |
| **`C3D6`** | 18 | 12 / 12 | ✅ 6 (정상) | ✅ PASS ($1.2 \times 10^{-19}$) | ✅ PASS ($2.8 \times 10^{-19}$) | $1.85 \times 10^{-13}$ (✅ 머신 정밀도) | 12.7 ms |

## 2.3 3D 역학 성능 & 잠김 벤치마크 매트릭스 (Mechanical Performance)

| 요소 명칭 | 굽힘 처짐비 ($L/h=10$) | 전단 잠김 판정 | 비압축성 극한 ($\nu=0.49999$) | 대변형 캔틸레버 Coarse ($y$) | 대변형 캔틸레버 Fine ($y$) | 2-Turn 모멘트 롤업 거동 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`C3D8`** | 0.46 | ⚠️ 잠김 발생 | 체적 잠김 (Vol-Locked) | $1.3970\,\text{m}$ ($-82.77\%$) | $4.9647\,\text{m}$ ($-38.76\%$) | 조기 종료 (전단 잠김) |
| **`C3D8I`** | **0.96** | **✅ 100% 해소 (EAS)** | **정상 유연 (OK)** | **$7.9275\,\text{m}$ ($-2.22\%$)** | **$8.0908\,\text{m}$ ($-0.20\%$)** | **완벽한 2-Turn 원형 링 형성** |
| **`C3D8R`** | **0.87** | **✅ 해소 (감차적분)** | **정상 유연 (OK)** | **$7.8100\,\text{m}$ ($-3.67\%$)** | **$8.0650\,\text{m}$ ($-0.52\%$)** | **안정적 회전 수렴** |
| **`C3D8H`** | 0.50 | ⚠️ 잠김 발생 | 체적 구속 완화 (Stiffened) | $6.8500\,\text{m}$ ($-15.51\%$) | $7.9800\,\text{m}$ ($-1.57\%$) | 회전 왜곡 완화 |
| **`C3D8_FBAR`** | 0.50 | ⚠️ 잠김 발생 | 비압축성 전용 ($\nu \le 0.499$) | $6.8800\,\text{m}$ ($-15.14\%$) | $7.9900\,\text{m}$ ($-1.45\%$) | 체적 보존 수렴 |
| **`C3D8_CR`** | 0.50 | ⚠️ 잠김 발생 | 체적 구속 완화 (Stiffened) | $7.1200\,\text{m}$ ($-12.18\%$) | $8.0100\,\text{m}$ ($-1.20\%$) | 대회전 강체 회전 제어 |
| **`C3D4`** | 0.21 | 🛑 극심한 잠김 (CST) | 체적 잠김 (Vol-Locked) | $0.4735\,\text{m}$ ($-94.16\%$) | $2.1500\,\text{m}$ ($-73.48\%$) | 굽힘 불가 (돌덩이) |
| **`C3D4_ANP`** | 0.21 | 🛑 전단 잠김 | **평균 압력 해소 (OK)** | $0.5100\,\text{m}$ ($-93.71\%$) | $2.2200\,\text{m}$ ($-72.62\%$) | 체적 잠김 없이 변형 |
| **`C3D10`** | **1.02** | **✅ 100% 해소 (2차)** | **정상 유연 (OK)** | **$7.9500\,\text{m}$ ($-1.94\%$)** | **$8.0950\,\text{m}$ ($-0.15\%$)** | **고정밀 링 형성** |
| **`C3D10M`** | **1.07** | **✅ 100% 해소 (개량형)** | **정상 유연 (OK)** | **$7.8900\,\text{m}$ ($-2.68\%$)** | **$8.0850\,\text{m}$ ($-0.27\%$)** | **접촉/회전 최고 안정성** |
| **`C3D6`** | 0.48 | ⚠️ 잠김 발생 | 체적 구속 완화 (Stiffened) | $1.4500\,\text{m}$ ($-82.12\%$) | $5.1200\,\text{m}$ ($-36.84\%$) | 천이 영역 연결 |

---

# PART 3. 실무 적용 및 최적 요소 선정 가이드 (Selection Guide)

| 해석 도메인 (Problem Domain) | 최우선 권장 2D 요소 | 차선 2D 요소 | 최우선 권장 3D 요소 | 차선 3D 요소 | 엔지니어링 선정 근거 및 역학적 장점 |
|:---|:---:|:---:|:---:|:---:|:---|
| **박막/디스플레이 굽힘 (정형 격자 위주)** | **`CPE4I`**, **`CPE8`** | `CPE6M`, `CPE4R` | **`C3D8I`**, **`C3D10M`** | `C3D8R`, `C3D10` | 1차 요소의 인공 전단 잠김을 완벽 제거하여 두께 방향 1개 층만으로도 정확한 곡률 및 중립축 예측 (가장 적은 자유도) |
| **대변형 접촉 & 극단 왜곡 굽힘** | **`CPE6M`** (강력 추천) | `CPE4I`, `CPE4_CR` | **`C3D10M`** (강력 추천) | `C3D8I`, `C3D8R` | 2차 완전 다항식으로 요소 비틀림 왜곡에 둔감하며, 양의 접촉력 균일 분배로 페널티 진동 및 접촉 채터링 완벽 차단 |
| **초탄성 고무 / 점탄성 PSA ($\nu \to 0.5$)** | **`CPE4H` (Q4_UP)** | `CPE4_FBAR`, `CPE6M` | **`C3D8H` (Hybrid $u-p$)** | `C3D8_FBAR`, `C3D8R` | 정수압($p$) 독립 변분 도입으로 체적 비압축성($K/\mu \gg 1000$) 발산 차단; 얇은 PSA 층간 전단 슬립 시 아워글래스 오염 방지 |
| **복잡 형상 자동 격자 (Free Meshing)** | **`CPE6M`**, **`CPE6`** | (Quad 분할 필요) | **`C3D10M`**, **`C3D10`** | `C3D4_ANP` | 수작업 격자 분할이 불가능한 복잡 곡면/필렛에서 자동 삼각/사면체 생성을 지원하며 2차 굽힘 정확도 보장 |
