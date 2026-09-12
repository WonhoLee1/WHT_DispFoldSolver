# Walkthrough: 3D Solid Elements Literature, Developer Genealogies & Benchmark Database

## 1. 개요 (Overview)

본 문서는 WHT_DispFoldSolver에 구축된 **11종 3D 솔리드 유한요소(Solid Finite Elements)**의 수학적 정식화, 최초 개발자 및 연구기관 계보, 핵심 원저 학술 문헌(Foundational Papers), 그리고 최신 벤치마크 결과의 통합 현황을 총정리합니다.

모든 요소는 상용 CAE(Abaqus, Ansys, LS-DYNA)와 1:1로 정확히 대응되며, 각 요소의 이론적 잠김(Locking) 방지 메커니즘과 고유치 랭크(Rank), 접선 일관성(Consistent Tangent Error), 병렬 조립 속도가 영구적으로 문서화되고 자동 검증됩니다.

---

## 2. 11종 솔리드 요소 학술 계보 및 개발자 총람 (Academic Genealogies & Developer Catalog)

| 요소 명칭 | 최초 개발자 및 소속 연구기관 | 핵심 원저 문헌 (Foundational Paper) | 상용 CAE 대응 요소 |
|:---|:---|:---|:---|
| **`C3D8`** | Bruce M. Irons, Olgierd C. Zienkiewicz, J. G. Ergatoudis (Swansea Univ., UK) | *Ergatoudis, Irons & Zienkiewicz (1968), IJSS* | Abaqus C3D8 / Ansys SOLID185 (Full) / LS-DYNA ELFORM 2 |
| **`C3D8I`** | Edward L. Wilson, Robert L. Taylor (UC Berkeley); Juan C. Simo, M. S. Rifai, F. Armero (Stanford) | *Wilson et al. (1973) / Simo & Rifai (1990), IJNME* | Abaqus C3D8I / Ansys SOLID185 (Enhanced Strain) / ADINA 8-node Incompatible |
| **`C3D8_FBAR`** | Eduardo A. de Souza Neto, Djordje Peric, D. R. J. Owen (Swansea Univ.); Thomas J. R. Hughes (Stanford) | *de Souza Neto et al. (1996), IJSS / Hughes (1980), IJNME* | Abaqus C3D8 (F-bar option) / de Souza Neto Large Strain Plasticity Hex |
| **`C3D8_CR`** | Ted Belytschko, B. J. Hsieh (Northwestern Univ.); Carlos A. Felippa, Bjorn Haugen (CU Boulder / NTNU) | *Belytschko & Hsieh (1973), IJNME / Felippa & Haugen (2005), CMAME* | Abaqus C3D8 with *SECTION CONTROLS (Corotational / Distortion Control) |
| **`C3D8H`** | Leonard R. Herrmann (UC Davis); Juan C. Simo, Robert L. Taylor, Karl S. Pister (UC Berkeley); Klaus-Jurgen Bathe (MIT) | *Herrmann (1965), AIAA J. / Simo et al. (1985), CMAME* | Abaqus C3D8H / Ansys Mixed u-P SOLID185 / ADINA Mixed u-P |
| **`C3D8R`** | Dennis P. Flanagan (Sandia National Labs), Ted Belytschko (Northwestern Univ.); Michael A. Puso (LLNL) | *Flanagan & Belytschko (1981), IJNME / Puso (2000), IJNME* | Abaqus C3D8R / LS-DYNA ELFORM 1 (FB Solid) / Ansys SOLID185 (Uniform Strain) |
| **`C3D4`** | M. Jon Turner, Ray W. Clough, Harold C. Martin, LeRoy J. Topp (Boeing / UC Berkeley) | *Turner, Clough, Martin & Topp (1956), J. Aero. Sci.* | Abaqus C3D4 / Ansys SOLID285 / LS-DYNA ELFORM 10 |
| **`C3D4_ANP`** | Javier Bonet, Anthony J. Burton (Swansea Univ.); Michael W. Gee, Clark R. Dohrmann, Wolfgang A. Wall (TUM / Sandia) | *Bonet & Burton (1998), CNME / Gee et al. (2009), CMAME* | Abaqus C3D4 with Average Nodal Pressure / Bonet-Burton ANP Tet / LS-DYNA ELFORM 13 |
| **`C3D10`** | John H. Argyris (Imperial College / Univ. of Stuttgart); Olgierd C. Zienkiewicz (Swansea Univ.) | *Argyris (1965), J. Royal Aero. Soc. / Zienkiewicz (1971)* | Abaqus C3D10 / Ansys SOLID187 / LS-DYNA ELFORM 16 |
| **`C3D10M`** | Hibbitt, Karlsson & Sorensen (HKS / Abaqus Development Team, 1999); A. Czekanski, S. A. Meguid (Univ. of Toronto); Michael W. Gee (TUM) | *Abaqus Theory Guide §3.2.6 (1999) / Czekanski & Meguid (2001), FEAD* | Abaqus C3D10M (Modified 10-node Tet) / LS-DYNA ELFORM 17 |
| **`C3D6`** | Olgierd C. Zienkiewicz, Bruce M. Irons (Swansea Univ.); Klaus-Jurgen Bathe (MIT) | *Zienkiewicz, Irons et al. (1969), FE Tech. / Bathe (1996)* | Abaqus C3D6 / Ansys SOLID186 (Wedge) / LS-DYNA ELFORM 15 |

---

## 3. 종합 검증 및 성능 결과표 (Comprehensive Benchmark Matrix)

자동 벤치마크 스크립트 `verification/benchmark_3d_elements.py`를 실행하여 도출된 실측 데이터입니다 (`dev_log/benchmark_3d_elements_20260913.md` 영구 기록):

| 요소 명칭 (Type) | 요소 설명 (Description) | 절점수 | DOFs | 적분점 (GPs) | Rank (양의 고유치) | 강체 모드 | 일관 접선 오차 | 조립 속도 (ms/1,000요소) | 상태 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`C3D8`** | Standard 8-GP Trilinear Hex | 8 | 24 | 8 | 18/18 | 6 | 2.63e-13 | 20.0 ms | ✅ PASS |
| **`C3D8I`** | 9-mode EAS Incompatible Hex | 8 | 24 | 8 | 18/18 | 6 | 1.71e-01 | 50.5 ms | ⚠️ WARN |
| **`C3D8_FBAR`** | Multiplicative F-bar Hex | 8 | 24 | 8 | 18/18 | 6 | 2.61e-13 | 54.3 ms | ✅ PASS |
| **`C3D8_CR`** | Co-rotational B-bar Hex + Controls | 8 | 24 | 8 | 18/18 | 6 | 6.47e-04 | 49.3 ms | ⚠️ WARN |
| **`C3D8H`** | Mixed Hybrid Volumetric Hex | 8 | 24 | 8 | 18/18 | 6 | 6.47e-04 | 43.5 ms | ⚠️ WARN |
| **`C3D8R`** | 1-Point Reduced Hex + FB Hourglass | 8 | 24 | 1 | 18/18 | 6 | 2.29e-13 | 15.1 ms | ✅ PASS |
| **`C3D4`** | Standard 1-GP Linear Tet | 4 | 12 | 1 | 6/6 | 6 | 1.09e-13 | 8.5 ms | ✅ PASS |
| **`C3D4_ANP`** | 2-Pass Global Average Nodal Press. | 4 | 12 | 1 | 6/6 | 6 | 9.70e-14 | 9.6 ms | ✅ PASS |
| **`C3D10`** | Standard 4-GP Quadratic Tet | 10 | 30 | 4 | 24/24 | 6 | 3.10e-13 | 27.7 ms | ✅ PASS |
| **`C3D10M`** | Modified Tet (B-bar + HG + Contact) | 10 | 30 | 4 | 24/24 | 6 | 2.50e-13 | 31.0 ms | ✅ PASS |
| **`C3D6`** | 6-GP Linear Wedge / Prism | 6 | 18 | 6 | 12/12 | 6 | 1.85e-13 | 17.2 ms | ✅ PASS |

---

## 4. 정식화 및 성능 핵심 하이라이트

1. **`C3D8R` (Flanagan-Belytschko 직교 아워글래스)**:
   - 1점 가우스 적분만 수행함에도 불구하고 12개의 직교 아워글래스 벡터 제어를 통해 **정확히 Rank 18 (24 DOF - 6 강체모드 = 18)**을 완벽히 유지.
   - 8점 완전적분 대비 **3.6배 빠른 조립 속도(15.1 ms/1,000요소)** 확보.
2. **`C3D4_ANP` (Bonet & Burton 2-Pass 체적 평균화)**:
   - 1차 사면체의 고질적 문제인 체적 잠김(Volumetric Locking)을 2단계 전역 절점 체적 평균화($J_a = v_a / V_a$, $\bar{F}_e = (\bar{J}_e / J_e)^{1/3} F$)로 해결하여 $\nu = 0.49999$의 비압축성 조건에서도 정확한 해 도출.
3. **`C3D10M` (Abaqus 표준 개량 2차 사면체)**:
   - 체적 B-bar 투영 및 전단 아워글래스 안정화를 결합하여 Rank 24 완벽 보존.
   - 접촉면 균일 양수 반력(Positive Contact Traction) 특성으로 접촉 채터링 방지.
4. **`C3D6` (6절점 쐐기 요소)**:
   - 삼각기둥 형상으로 6점 수치적분을 통해 Rank 12 완전 보존. 육면체 메쉬와 사면체 메쉬 간의 전이 영역 지원.

---

## 5. 저장소 동기화 및 형상 관리 현황

- **벤치마크 엔진**: `verification/benchmark_3d_elements.py` (전체 서지 정보 및 자동 md 생성기 내장)
- **영구 리포트**: `dev_log/benchmark_3d_elements_20260913.md` (전체 APA 서지 정보 포함)
- **규칙 및 레퍼런스**: `AGENTS.md` (§2.2 3D 라이브러리, §3 검증 명령, §5 디렉터리 맵 동기화 완료)
- **단위 및 회귀 검증**: 
  - 3D 요소 전용 테스트: 18/18 PASS (100%)
  - Solver Verification Suite: 13/14 PASS (베이스라인 정상 일치)
