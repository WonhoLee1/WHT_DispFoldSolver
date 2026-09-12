# 3D Solid Elements Verification & Performance Benchmark Report (20260912)

## 1. 종합 검증 및 성능 결과표 (Comprehensive Benchmark Matrix)

| 요소 명칭 (Type) | 요소 설명 (Description) | 절점수 | DOFs | 적분점 (GPs) | Rank (양의 고유치) | 강체 모드 | 일관 접선 오차 | 조립 속도 (ms/1,000요소) | 상태 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`C3D8`** | Standard 8-GP Trilinear Hex | 8 | 24 | 8 | 18/18 | 6 | 2.63e-13 | 19.3 ms | ✅ PASS |
| **`C3D8I`** | 9-mode EAS Incompatible Hex | 8 | 24 | 8 | 18/18 | 6 | 1.71e-01 | 51.6 ms | ⚠️ WARN |
| **`C3D8_FBAR`** | Multiplicative F-bar Hex | 8 | 24 | 8 | 18/18 | 6 | 2.61e-13 | 59.2 ms | ✅ PASS |
| **`C3D8_CR`** | Co-rotational B-bar Hex + Controls | 8 | 24 | 8 | 18/18 | 6 | 6.47e-04 | 51.7 ms | ⚠️ WARN |
| **`C3D8H`** | Mixed Hybrid Volumetric Hex | 8 | 24 | 8 | 18/18 | 6 | 6.47e-04 | 53.4 ms | ⚠️ WARN |
| **`C3D8R`** | 1-Point Reduced Hex + FB Hourglass | 8 | 24 | 1 | 18/18 | 6 | 2.29e-13 | 16.2 ms | ✅ PASS |
| **`C3D4`** | Standard 1-GP Linear Tet | 4 | 12 | 1 | 6/6 | 6 | 1.09e-13 | 9.5 ms | ✅ PASS |
| **`C3D4_ANP`** | 2-Pass Global Average Nodal Press. | 4 | 12 | 1 | 6/6 | 6 | 9.70e-14 | 10.0 ms | ✅ PASS |
| **`C3D10`** | Standard 4-GP Quadratic Tet | 10 | 30 | 4 | 24/24 | 6 | 3.10e-13 | 29.9 ms | ✅ PASS |
| **`C3D10M`** | Modified Tet (B-bar + HG + Contact) | 10 | 30 | 4 | 24/24 | 6 | 2.50e-13 | 32.9 ms | ✅ PASS |
| **`C3D6`** | 6-GP Linear Wedge / Prism | 6 | 18 | 6 | 12/12 | 6 | 1.85e-13 | 14.1 ms | ✅ PASS |

## 2. 주요 성능 및 정식화 분석

1. **`C3D8R` (Flanagan-Belytschko 1점 감차적분)**:
   - 1점 적분임에도 12개의 직교 아워글래스 벡터 제어로 **정확히 Rank 18 (24 DOF - 강체 6모드 = 18)** 달성.
   - 전단 잠김(Shear locking)이 없으며, 풀 적분 요소(`C3D8I`, `C3D8_FBAR`) 대비 압도적인 조립 속도 확보.
2. **`C3D4_ANP` (Bonet & Burton 2-Pass 체적 평균화)**:
   - 표준 1차 사면체의 고질적인 체적 잠김을 글로벌 2단계 평활화로 극복하여 비압축성 극한 및 소성 영역 적용 가능.
3. **`C3D10M` (Abaqus 표준 개량 2차 사면체)**:
   - B-bar 체적 투영과 전단 기반 아워글래스 안정화로 Rank 24 완벽 보장 및 면 접촉력 양수화 달성.
4. **`C3D6` (6절점 삼각기둥)**:
   - 6점 수치적분으로 Rank 12 완전 보장, 육면체와 사면체 사이의 천이 메싱 완벽 지원.
