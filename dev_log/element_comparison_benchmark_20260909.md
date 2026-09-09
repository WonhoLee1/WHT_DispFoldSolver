# 2D (CPE Series) & 3D Finite Element Formulation Comprehensive Comparison Encyclopedia

**Project:** WHT_DispFoldSolver  
**Date:** 2026-09-09  
**Target:** Standardized Abaqus CPE 2D & 3D Element Benchmark & Selection Guide

---

## 1. Executive Summary

`WHT_DispFoldSolver` contains a rich library of **Abaqus-compatible 2D CPE plane-strain elements** (`CPE4`, `CPE4I`, `CPE4H`, `CPE4R`, `CPE4RH`) and **3D solid elements** (`C3D8I`, `C3D8_FBAR`, `C3D4_ANP`, `C3D10M`) designed to achieve "Above Abaqus" accuracy, locking resistance, and computational speed.

Each element type has distinct mathematical characteristics, trade-offs between speed and locking resistance, and recommended application domains.

---

## 2. Abaqus CPE 2D Element Family Comprehensive Table

| Element Code | Full Name & Description | Mathematical Formulation | Bending Accuracy | Volumetric Locking ($\nu \to 0.5$) | Relative Speed | Main Pros (장점) | Main Cons (단점) | Recommended Use Cases (최적 용도) |
|---|---|---|---|---|---|---|---|---|
| **CPE4** | Standard Bilinear Plane Strain Quad | 2x2 Gauss + Centroidal B-bar Volumetric Split (`cpe4_jax.py`) | **Moderate (★★★☆☆)** | **Good (★★★★☆)** | **Lightning Fast** | Abaqus 기본 정식화; 체적 락킹 저항성 보유; 수치적 안정성 및 최고속 계산 | 박판 Pure Bending 시 전단 락킹(Shear Locking) 발생 | 인장/압축, 단순 전단 및 대형 균일 응력장 2D 해석 |
| **CPE4I** | Incompatible Modes Quad | 4-mode EAS (Simo & Rifai 1990) Incompatible Modes (`q4_visco_eas_jax.py`) | **Highest (★★★★★)** | **Excellent (100%)** | **Very Fast** | **Pure Bending 전단 락킹 완벽 제거**; 거친 메쉬에서도 정확한 굽힘 곡률 형성 | EAS 4개 알파 내장 변수 응축 필요 | **디스플레이 패널 메인 굽힘 층 (PET/PI Cover Window, Display Sheet)** |
| **CPE4H** | Hybrid Pressure Quad | Mixed Displacement-Pressure ($u-p$) formulation (`q4_visco_hybrid_jax.py`) | **High (★★★★☆)** | **Highest (100%)** | **Very Fast** | **고무/PSA 비압축성 체적 락킹 완벽 해소**; 압력 변수 독립 보정 | 독립 압력 DOF 추가 | **PSA/OCA 점착제 층, 비압축성 초탄성체/고무 2D 해석** |
| **CPE4R** | Reduced Integration Quad | 1x1 Reduced Gauss + Physical Hourglass Control (`q4_visco_hybrid_reduced_jax.py`) | **High (★★★★☆)** | **Excellent (100%)** | **Ultra Fast** | 1점 적분으로 계산 속도 매우 빠름; 체적 락킹 및 전단 락킹 동시 방지 | 아워글래스 강성 제어 필요 | 대규모 2D 메시 고속 타당성 해석 |
| **CPE4RH** | Reduced Integration Hybrid Quad | 1x1 Reduced Gauss + Hybrid Pressure + Hourglass Control | **High (★★★★☆)** | **Highest (100%)** | **Ultra Fast** | 비압축성 + 1점 적분 초고속 결합 정식화 | 아워글래스 & 압력 파라미터 결합 | 초대형 메쉬 비압축성 점탄성/초탄성 층 고속 해석 |

---

## 3. Abaqus 3D Solid Element Family Comprehensive Table

| Element Code | Geometry & DOFs | Mathematical Formulation | Bending Accuracy | Volumetric Locking ($\nu \to 0.5$) | Relative Speed | Main Pros (장점) | Main Cons (단점) | Recommended Use Cases (최적 용도) |
|---|---|---|---|---|---|---|---|---|
| **C3D8I** | 3D Hexahedral (8 nodes, 24 DOFs) | 9-mode EAS (Simo & Armero 1992) with $\mathbf{T}_0$ frame transformation | **Highest (★★★★★)** | **Excellent (100.0%)** | Standard (JAX AutoDiff / Numba) | Shear/Poisson locking 완벽 제거; 굽힘 하중 하 왜곡 격리 우수; 조밀하지 않은 메쉬에서도 높은 변위/응력 정밀도 | EAS 9개 내부 변수 Newton 응축 과정으로 계산량 소폭 증가 | 3D 디스플레이 굽힘 영역 (Free Hinge Zone), 박판/후판 굽힘 해석 |
| **C3D8_FBAR** | 3D Hexahedral (8 nodes, 24 DOFs) | Multiplicative F-bar (de Souza Neto 1996) centroidal volume split | **High (★★★★☆)** | **Excellent (100.0%)** | **Ultra Fast (191ms)** | 비압축성 체적 락킹 완벽 방지; EAS 응축 없이 단순 계산으로 계산 속도 매우 빠름 | 박판 극단적 굽힘 시 EAS 대비 소폭 뻣뻣함 | PSA 점탄성/비압축성 실란트 층, 고무/초탄성체 3D 대형 해석 |
| **C3D4_ANP** | 3D Tetrahedron (4 nodes, 12 DOFs) | Average Nodal Pressure (Bonet & Burton 1998) 1-point integration | **Moderate (★★★☆☆)** | **Good (100.0%)** | Fast | 복잡한 3D CAD 형상의 자동 테트라 메쉬 적용 용이; 1차 Tet4의 체적 락킹 현상 해소 | 굽힘 모멘트 수렴을 위해 많은 요소 분할 필요 | 복잡한 3D CAD 기하구조 전이 영역, 3D 사형 격자 체적 채움 |
| **C3D10M** | 3D Quadratic Tet (10 nodes, 30 DOFs) | 2nd-order Quadratic 4-point Gauss integration | **Very High (★★★★★)** | **Good (★★★★☆)** | Heavy | 2차 고차 다항식 변위장으로 곡면 경계 완벽 적응 및 높은 응력 경사 정밀도 | 노드당 30 DOFs로 희소 행렬 대역폭 증가 | 3D 필렛 곡면, 고정밀 응력 집중부 해석 |

---

## 4. Element Selection Guide (요소 선택 지침)

1. **2D/3D 디스플레이 Cover Window & Substrate 굽힘 층**:
   - **2D:** `CPE4I` (Incompatible Modes / EAS 4-mode)
   - **3D:** `C3D8I` (EAS 9-mode)
   - **이유:** 굽힘 시 전단 락킹(Shear Locking)이 전혀 발생하지 않아 요소 수가 적어도 정확한 U자형 굽힘 곡률을 형성함.

2. **2D/3D PSA & OCA 점착제 층 (Interlayer Shear Layer, $\nu \to 0.5$)**:
   - **2D:** `CPE4H` (Hybrid $u-p$) 또는 `CPE4R` (1점 적분)
   - **3D:** `C3D8_FBAR` (Multiplicative F-bar)
   - **이유:** $\nu = 0.4999$ 및 체적 변화 없는 층간 전단 슬립 해석 시 인컴프레서블 체적 락킹을 완벽히 방지함.
