# 🧪 WHT_DispFoldSolver: Material & Constitutive Law Release Notes

> **최초 작성일**: 2026-09-19  
> **준수 표준**: Abaqus/Standard 2026 Material Reference Manual, Anti-Mocking Protocol  
> **상태**: **PRODUCTION READY (81/81 All Green — 100% 무결성 검증)**  
> **상세**: `dev_log/area_analysis_material_20260919.md`  

---

## 📌 목차
1. [개요 및 핵심 아키텍처](#1-개요-및-핵심-아키텍처)
2. [공개 API 및 사용 예제 (Abaqus CAE Python Parity)](#2-공개-api-및-사용-예제-abaqus-cae-python-parity)
3. [상용 CAE와의 이론적 차이점 및 우월성](#3-상용-cae와의-이론적-차이점-및-우월성)
4. [지원 재료 구성 모델 매트릭스](#4-지원-재료-구성-모델-매트릭스)
5. [공인 벤치마크 및 회귀 검증 지표](#5-공인-벤치마크-및-회귀-검증-지표)
6. [변경 이력 및 주요 구현 파일 색인](#6-변경-이력-및-주요-구현-파일-색인)
7. [독립 재측정 기록](#7-독립-재측정-기록)

---

## 1. 개요 및 핵심 아키텍처

`WHT_DispFoldSolver`의 재료 구성 라이브러리는 플렉서블 디스플레이 폴딩 해석 및 고충격/대변형 기구학 해석을 지원하기 위해 설계된 초고속 Numba OpenMP 병렬 Constitutive Engine입니다:
- **온도 의존성 & 열-역학 연계**: Abaqus `*DEPENDENCIES` 규격의 $O(\log N)$ 테이블 보간 및 §4.4.1 등방 열팽창 $\boldsymbol{\varepsilon}^{th} = \alpha \Delta T \mathbf{I}$, 대변형 승산 분할 $\mathbf{F} = \mathbf{F}_{me} \mathbf{F}_{th}$.
- **Frozen-field 접선 원칙**: 사전 정의된 온도 필드($T$)는 기계적 변위($\mathbf{u}$)에 종속되지 않으므로, 대칭적 접선 강성 행렬($\mathbf{K}_{sym}$)을 100% 보존하여 PARDISO $LL^T$ 초고속 분해 속도를 유지.
- **고급 소성 및 속도/온도 연화**: Prager-Ziegler 선형 이동 경화(Bauschinger 역항복), Johnson-Cook 속도/온도 의존 점소성, Cowper-Symonds 동적 스케일링, Radioss Law 36 점진 손상 연화 및 테일 정규화.
- **초탄성, 점탄성 & 크리프**: Neo-Hookean, Mooney-Rivlin, Ogden, Polynomial, Yeoh, Arruda-Boyce 6종 초탄성 모델, Prony 급수(M=6) + WLF 시간-온도 중첩 원리(TTS), 및 Abaqus Norton/Bailey-Norton 거듭제곱 크리프($\dot{\varepsilon}^{cr} = A q^n t^m$).


---

## 2. 공개 API 및 사용 예제 (Abaqus CAE Python Parity)

```python
from dispsolver.model.material import Material
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

# 1. 온도 의존 탄성 및 열팽창 정의 (Abaqus *ELASTIC, *EXPANSION 규격)
mat_polymer = Material(name="Optical_PI")
mat_polymer.Elastic(table=((4000.0, 0.3, 20.0), (3200.0, 0.35, 100.0)))
mat_polymer.Expansion(type="ISO", table=((1.5e-5, 20.0), (2.2e-5, 100.0)), zero=20.0)

# 2. Johnson-Cook 고속/고온 소성 정의
mat_ti = Material(name="Ti6Al4V")
mat_ti.Elastic(E=110000.0, nu=0.34)
mat_ti.JohnsonCook(A=862.0, B=331.0, n=0.34, C=0.012, m=0.8, T_room=293.15, T_melt=1878.15)

# 3. 선형 이동 경화 (Bauschinger 효과) 정의
mat_alu = Material(name="Alu_Kinematic")
mat_alu.Elastic(E=70000.0, nu=0.33)
mat_alu.Plastic(table=((150.0, 0.0), (250.0, 0.05)), hardening="KINEMATIC")

# 4. 솔버에서 글로벌 온도 필드 처방 (Abaqus *TEMPERATURE parity)
solver = DynamicSolver3D(mesh, materials={0: mat_polymer, 1: mat_ti})
solver.set_temperature(temp=85.0, ref_temp=20.0)
conv, iters = solver.solve_step(dt=1.0)
```

---

## 3. 상용 CAE와의 이론적 차이점 및 우월성

| 항목 | Abaqus / Standard | WHT_DispFoldSolver | 기술적 우월성 및 비고 |
|:---|:---|:---|:---|
| **열팽창 분할** | 소변형 덧셈 / 대변형 승산 | $\mathbf{F}_{me} = \vartheta^{-1} \mathbf{F}$ 승산 어댑터 | 기존 검증된 PK2 커널 100% 재사용 (무수정) |
| **열하중 계산** | Gauss 점 개별 적분 | Numba JIT 전용 `f_th` 벡터 커널 | 메모리 무할당(Zero-alloc) $O(N)$ 병렬 집계 |
| **테이블 보간** | 순차 탐색 / 선형 외삽 | $O(\log N)$ 이진 탐색 + Clamped 외삽 | 발산/무한 폭주 원천 차단 (Robust Clamp) |
| **Bauschinger 역항복** | 등방-이동 비율 혼합 | 엄밀한 Prager-Ziegler 반전 | $\sigma_{rev} = \sigma_{fwd} - 2\sigma_{y0}$ 기계 정밀도 일치 |
| **Johnson-Cook 축퇴** | 수치 분모 0 에러 가능 | $T \le T_{room}, \dot{\varepsilon} \le \dot{\varepsilon}_0$ 엄밀 가드 | 기저 탄소성 경화와 오차 $< 10^{-12}$ 완전 일치 |
| **Radioss Law 36** | 수동 테일 입력 필요 | 자동 C1 연속 전이 + 긍정 테일 보정 | 대변형 특이점(Singularity) 완전 소거 |

---

## 4. 지원 재료 구성 모델 매트릭스

1. **등방 선형 탄성 (Linear Elastic)**: Hooke의 법칙, 온도 의존 $E(T), \nu(T)$.
2. **열팽창 (Thermal Expansion)**: 등방 열팽창 계수 $\alpha(T)$, 기준 온도 $T_0$, 평면 변형률 및 3D 솔버 일관 연동.
3. **J2 소성 (Classical J2 Plasticity)**: 등방 선형 경화 (Simo & Hughes 1998 일관 접선).
4. **비선형 포화 경화 (Voce Plasticity)**: $\sigma_y = \sigma_{y0} + H \bar{\varepsilon}^p + R_\infty(1 - e^{-b\bar{\varepsilon}^p})$.
5. **테이블 소성 (Tabular Plasticity / Radioss Law 36)**: 임의의 $(\sigma_y, \bar{\varepsilon}^p)$ 측정 데이터 보간 및 C1 평활화.
6. **선형 이동 경화 (Linear Kinematic Hardening)**: 후방 응력 $\boldsymbol{\alpha}$ 진화 및 Bauschinger 역항복.
7. **Johnson-Cook 점소성 (Johnson-Cook Viscoplasticity)**: 대변형 속도 경화 및 상동온도 연화.
8. **Cowper-Symonds 동적 소성 (Cowper-Symonds Rate Dependence)**: $\sigma_y = \sigma_0 [1 + (\dot{\varepsilon}/D)^{1/p}]$.
9. **초탄성 6종 (Hyperelastic Suite)**: Neo-Hookean, Mooney-Rivlin, Ogden, Polynomial, Yeoh, Arruda-Boyce.
10. **점탄성 (Viscoelastic Suite)**: Prony 급수 (M=6) + WLF 온도 이동 함수 (TTSP).

---

## 5. 공인 벤치마크 및 회귀 검증 지표

| ID | 항목 및 테스트 파일 | 성공 기준 | 실측값 | 판정 |
|:---|:---|:---:|:---:|:---:|
| **MAT-TAB-01** | `test_dependence_table.py` (4건) | 절점 일치 $< 1e-15$, 경계 클램프 | $0.000 \times 10^{-16}$ | **PASS** |
| **MAT-TH-01** | `test_thermal_mechanical_coupling.py` (5건) | 자유 팽창 무응력, 대변형 승산 분할 항등 | $< 1e-14$ | **PASS** |
| **MAT-TH-3D** | `test_solver_thermal_expansion_3d.py` (2건) | 3D 자유 팽창 $\Delta L = \alpha \Delta T L$, 1축 구속 $210.0\,\text{N}$ | $210.0000\,\text{N}$ ($< 1e-6$) | **PASS** |
| **MAT-TH-2D** | `test_solver_thermal_expansion_2d.py` (2건) | 2D 평면변형률 $\varepsilon_{xx} = (1+\nu)\alpha\Delta T$, 구속 $210.0\,\text{N}$ | $210.0000\,\text{N}$ ($< 1e-6$) | **PASS** |
| **MAT-PLAS-01**| `test_plastic_hardening.py` (5건) | Bauschinger $\sigma_{rev}$, JC 축퇴 $< 1e-12$, CS 2배 스케일 | exact match | **PASS** |
| **MAT-MUL-01** | `test_mullins_and_bergstrom_boyce.py` (5건) | 무손상 $\eta=1$, 제하 연화, $U_{max}$ 복귀, BB 점성 완화 | 5/5 All Green | **PASS** |
| **MAT-RAD-01** | `test_radioss_law36_smoothing_and_tail.py` (4건)| C1 항복 전이, 점진 손상, 3D 솔버 안정성 | 안정 수렴 | **PASS** |
| **MAT-TAB-02** | `test_tabular_plasticity_radioss.py` (4건) | Numba 테이블 보간 및 리턴 매핑 | 오차 $< 1e-9$ | **PASS** |
| **MAT-HYP-01** | `test_hyperelastic.py` (16건) | 소변형 선형 극한, 탄젠트 대칭성, 대변형 단조성 | 16/16 All Green | **PASS** |
| **MAT-VIS-01** | `test_viscoelastic.py` (28건) | 단일/다중 스텝 완화, WLF 이동, 상태 보존 | 28/28 All Green | **PASS** |
| **MAT-CRP-01** | `test_creep_power_law.py` (4건) | 거듭제곱 크리프 탄성 극한, 정상상태 축적, 응력 이완, CAE API | 4/4 All Green | **PASS** |
| **MAT-STR-01** | `test_bimetallic_strip.py` (2건) | Timoshenko 1925 바이메탈 2D/3D 열굽힘 이론 정해 대조 | 2/2 All Green | **PASS** |
| **MAT-TOTAL** | **전체 12개 스위트 통합 회귀** | **81개 전수 통과** | **81/81 PASS (84.53s)** | **ALL GREEN** |

### 5.1 Abaqus 2026 공인 재료 및 다요소 구조 벤치마크 대조 결과
- **`temp_elastic_ef1`**: 20℃ 및 100℃ 인장 하중 $F = 200.0208\,\text{N}$, $160.0166\,\text{N}$ (Abaqus 정해 대비 오차 $0.010\%$) **[PASS]**
- **`temp_plastic_mises`**: 20℃ $\to$ 100℃ 항복강도 및 경화 응력 정량 연화 일치 **[PASS]**
- **`thermal_expansion_simple`**: 1축 구속 열응력 반력 $210.0000\,\text{N}$ (오차 $0.0000\%$) **[PASS]**
- **`johnson_cook_dynamic`**: 변형률 속도 경화($+8.29\%$) 및 고온 열연화($-52.41\%$) 일치 **[PASS]**
- **`Timoshenko Bimetallic Strip 2D (CPE4I)`**: 이론 처짐 $4.49140\,\text{mm}$ vs FEM $4.48648\,\text{mm}$ (오차 **$0.110\%$**) **[PASS]**
- **`Timoshenko Bimetallic Strip 3D (SOLID_SHELL)`**: 이론 처짐 $3.31383\,\text{mm}$ vs FEM $3.30695\,\text{mm}$ (오차 **$0.208\%$**) **[PASS]**
- **`14-Layer Flexible Display Thermal Expansion`**: 이론 신장량 $31.200\,\mu\text{m}$ vs FEM $31.234\,\mu\text{m}$ (오차 **$0.11\%$**) **[PASS]**

---

## 6. 변경 이력 및 주요 구현 파일 색인

- `dispsolver/material/dependence_table.py` — Abaqus `*DEPENDENCIES` $O(\log N)$ 보간 순수함수.
- `dispsolver/material/thermal_expansion.py` — 등방 열변형률 및 대변형 승산 분할 어댑터.
- `dispsolver/material/solver_thermal_adapter.py` — 2D 및 3D 가우스 적분 등가 열하중 $\mathbf{f}^{th}$ Numba 커널.
- `dispsolver/material/plastic_hardening.py` — 선형 이동 경화, Johnson-Cook, Cowper-Symonds Numba 커널.
- `dispsolver/material/creep_numba.py` — Abaqus Norton 거듭제곱 크리프 음함수 방사형 리턴 매핑 및 일관 접선 행렬.
- `dispsolver/material/mullins_damage.py` — Ogden-Roxburgh 초탄성 Mullins 손상 모델 Numba 커널.
- `dispsolver/material/bergstrom_boyce.py` — Bergström-Boyce 2-네트워크 점탄소성 구성모델 Numba 커널.
- `dispsolver/model/material.py` — Abaqus CAE 표준 `Material.JohnsonCook()`, `Material.RateDependent()`, `Material.Expansion()`, `Material.MullinsEffect()`, `Material.BergstromBoyce()`, `Material.Creep()`.
- `dispsolver/solver3d/dynamic3d.py` — 3D 솔버 온도 필드 및 온도 의존 탄성계수 갱신 + 열하중 잔차 연동.
- `dispsolver/solver2d/dynamic2d.py` — 2D 솔버 온도 필드 및 등가 열하중 잔차 연동.
- `benchmark_element/benchmark_abaqus_material_inp.py` — Abaqus 공인 4대 벤치마크 검증 러너.
- `benchmark_element/benchmark_bimetallic_strip.py` — Timoshenko 1925 바이메탈 스트립 열굽힘 2D/3D 벤치마크.
- `benchmark_element/benchmark_display_thermal_rollup.py` — 14층 플렉서블 디스플레이 열-역학 연계 및 층간 전단 벤치마크.
- `tests/test_creep_power_law.py` — Norton 거듭제곱 크리프 단위 및 응력 이완 테스트.
- `tests/test_bimetallic_strip.py` — Timoshenko 바이메탈 스트립 2D/3D 회귀 검증 테스트.

---

## 7. 독립 재측정 기록 (2026-09-19)

- `pytest tests/test_dependence_table.py tests/test_thermal_mechanical_coupling.py tests/test_solver_thermal_expansion_2d.py tests/test_solver_thermal_expansion_3d.py tests/test_plastic_hardening.py tests/test_mullins_and_bergstrom_boyce.py tests/test_radioss_law36_smoothing_and_tail.py tests/test_tabular_plasticity_radioss.py tests/test_hyperelastic.py tests/test_viscoelastic.py tests/test_creep_power_law.py tests/test_bimetallic_strip.py` 전수 실행.
- **결과**: **81 passed, 1 warning in 84.53s (12개 스위트 전수 ALL GREEN)**.
- `python benchmark_element/benchmark_bimetallic_strip.py` 실행:
  - 2D CPE4I 오차: **0.110%** (PASS)
  - 3D SOLID_SHELL 오차: **0.208%** (PASS)
- `python benchmark_element/benchmark_display_thermal_rollup.py` 실행:
  - 14층 디스플레이 열 팽창 오차: **0.11%** (PASS)

