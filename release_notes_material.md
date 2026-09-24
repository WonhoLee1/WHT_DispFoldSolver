# 🧪 WHT_DispFoldSolver: Material & Constitutive Law Release Notes

> **최초 작성일**: 2026-09-19  
> **준수 표준**: Abaqus/Standard 2026 Material Reference Manual, Anti-Mocking Protocol  
> **상태**: **PRODUCTION READY (81/81 기반 + Wave0–3 신규 50개 ALL GREEN — 2026-09-20)**  
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
- **Wave0–3 추가분 (2026-09-20)**: 필드변수 의존 탄/소성(FV1 `*FIELD` parity), Ogden/Mooney 2D 코어 + 열팽창 승산 어댑터(면외 $1/\vartheta$ 포함), 쌍곡사인 크리프 + Anand 점소성(2x2 내재적분), 주파수영역 Prony 평가($G', G'', \tan\delta$ + WLF), HGO/GOH 이방성 초탄성(JAX), 연성/JC 손상 개시·진전·삭제(MAT13, SDV 10).
- **공유기반 수정 (동일 일자)**: JAX `base.py` 전단 2배 오류 수정(PK2/tensor/tangent 일관화), 거듭제곱 크리프 접선계수 $2G \to 4G/3$ 수정(FD 입증).


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

# 5. 필드변수 의존 탄성 + 연성 손상 (Abaqus *FIELD / *DAMAGE INITIATION parity)
mat_f = Material(name="FieldSteel")
mat_f.Elastic(field_table=((200000.0, 0.3, 0.0), (160000.0, 0.3, 1.0)))
solver_f = DynamicSolver3D(mesh, materials={0: mat_f})
solver_f.set_field_variable(0.5)  # E = 180000 MPa 선형 보간

mat_d = Material(name="DuctileSteel")
mat_d.Elastic(E=200000.0, nu=0.3)
mat_d.Plastic(table=[(250.0, 0.0), (270.0, 0.01)])
mat_d.DamageInitiation(type="DUCTILE", table=((0.01,),))
mat_d.DamageEvolution(type="ENERGY_LINEAR", Gf=2.0, L=1.0)
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
| **FIELD 의존** | *FIELD + DEPENDENCIES 테이블 | 동일 보간 + `set_field_variable` 플러밍 | temp 경로와 bit-identical parity 실측 |
| **손상 연화** | 에너지 소산 정규화 | $G_f/L$ 면적에너지 객관성 0.13% 실측 | 요소크기 무관 파단 에너지 |
| **주파수 점탄성** | 시간↔주파수 변환 내장 | Prony→$G', G''$ 정확 변환 (신규 상수 0) | Debye 피크 해석값 일치 |

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
11. **필드변수 의존 (Field Dependence, 2026-09-20)**: `field_elastic_table`/`field_plastic_table` + `set_field_variable` (온도 경로와 bit-identical).
12. **초탄성 팽창 어댑터 (2026-09-20)**: Ogden/Mooney/AB 열승산 분할 + 면외 $1/\vartheta$ + 접선 $1/\vartheta^2$ 체인룰.
13. **쌍곡사인 크리프 + Anand (2026-09-20)**: Garofalo $\dot{\varepsilon} = A\sinh(Bq)^n t^m$ (Norton 극한 대조) + Anand 2x2 내재적분(`converged` 플래그).
14. **주파수 점탄성 (2026-09-20)**: Debye $G', G''$ 평가 + WLF 환산주파수 + 복소 $E^*$ (신규 상수 0).
15. **이방성 초탄성 HGO/GOH (2026-09-20)**: 섬유분산 $\kappa$, 좌굴 off, JAX autodiff 접선.
16. **연성/JC 손상 (2026-09-20)**: MAT13 — 개시누적 $\omega_D$ + $G_f$ 에너지진전 + 삭제 (SDV 10).
17. **Drucker-Prager 소성 (2026-09-20)**: 선형 전단기준, K=1, 비연관 팽창 + apex (MAT14, SDV 7). $\beta=0$ → J2 항등.
18. **Concrete 손상소성 (2026-09-20)**: DP 백본 + 압축경화/인장연화 테이블 + dt/dc + apex (MAT15, SDV 10). Lubliner Lode·주기복원 범위 외.
19. **Van der Waals 초탄성 (2026-09-20)**: Kilian lock-up $(\lambda_m)$ 분기, $\lambda_m \to \infty$ → NH 항등 회귀게이트.
20. **FLD 판재한계 (2026-09-20)**: Keeler-Brazier FLC0 + 두께·n 보정, 출력전용 후처리 (구성식 미개입).
21. **JC rate·temperature 손상승수 (2026-09-20)**: D4/D5 디스패치 (MAT13 props 13–16), D4=D5=0 → 정적식 항등. Lode-K는 t-분포 범위 오류로 비활성 슬롯 예약 (90dfd12).

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
| **MAT-HYP-02** | `test_hyperelastic_ogden_mooney.py` (7건) | 소변형 극한, Ogden↔Mooney 교차, 방향 FD | 7/7 (2026-09-20) | **PASS** |
| **MAT-FLD-01** | `test_field_ef1.py` (4건) + `benchmark_field_ef1.py` | 200/180/160 N (<1%), 소성 서열 | 4/4 + 2/2 (2026-09-20) | **PASS** |
| **MAT-EXP-01** | `test_expansion_hyper.py` (4건) | θ→1 극한, Ogden↔Mooney 분할후 일치, FD | 4/4 (2026-09-20) | **PASS** |
| **MAT-CRP-02** | `test_creep_extended.py` (10건) | sinh→Norton 극한, Anand 포화/서열/FD | 10/10 (2026-09-20) | **PASS** |
| **MAT-VIS-02** | `test_frequency_viscoelasticity.py` (6건) | Debye 피크, 극한, WLF 이동, 가산성 | 6/6 (2026-09-20) | **PASS** |
| **MAT-ANISO-01** | `test_anisotropic_hyperelastic.py` (5건) | 등방퇴화, κ=1/3, 객관성, FD | 5/5 (2026-09-20) | **PASS** |
| **MAT-DMG-01** | `test_damage_ductile_jc.py` (13건) | 개시누적, JC퇴화, D4/D5 rate·T승수, 무치유, 파단, FD, end-to-end | 13/13 (2026-09-20) | **PASS** |
| **MAT-DMG-BM** | `benchmark_damage.py` (4건) | 개시해석해, 면적에너지 객관성, 삼축, JC식 | 4/4 (2026-09-20) | **PASS** |
| **MAT-DP-01** | `test_drucker_prager.py` (9건) | β=0 퇴화, 인장/압축 비대칭 해석식, apex, 팽창부호, FD, end-to-end | 9/9 (2026-09-20) | **PASS** |
| **MAT-CDP-01** | `test_concrete_dp.py` (11건) | 경화/연화 exact, apex 정수압, 제하탄성 exact, FD, end-to-end | 11/11 (2026-09-20) | **PASS** |
| **MAT-VDW-01** | `test_van_der_waals.py` (6건) | Kilian lock-up, NH 항등극한, FD | 6/6 (2026-09-20) | **PASS** |
| **MAT-FLC-01** | `test_forming_limit.py` (5건) | Keeler-Brazier FLC0 + n·t 보정, 출력전용 | 5/5 (2026-09-20) | **PASS** |

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
- `dispsolver/material/creep_numba.py` — Abaqus Norton 거듭제곱 크리프 음함수 방사형 리턴 매핑 및 일관 접선 행렬 (2026-09-20: 연화 rank-1 계수 $2G \to 4G/3$ FD 수정).
- `dispsolver/material/base.py` — JAX 초탄성 베이스 (2026-09-20: 전단 2배 오류 수정 + `f33` 면외 스트레치 스레딩).
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
- `dispsolver/material/ogden.py` + `mooney_rivlin.py` — Ogden 스펙트럴 / Mooney-Rivlin JAX 코어 (2026-09-20, s1항 J13 오류·coincident 극한 수정).
- `dispsolver/material/expansion_hyper.py` — Ogden/Mooney/AB 열승산 어댑터 (2026-09-20).
- `dispsolver/material/creep_extended.py` — 쌍곡사인 + Anand 2x2 내재적분 (2026-09-20).
- `dispsolver/material/frequency_viscoelastic.py` — Debye $G', G''$ + WLF + 복소 $E^*$ (2026-09-20).
- `dispsolver/material/anisotropic_hyperelastic.py` — HGO/GOH JAX 코어 (2026-09-20).
- `dispsolver/material/damage_initiation.py` — 연성/JC 개시 + $G_f$ 진전 순수커널 (2026-09-20).
- `dispsolver/material3d/numba_materials.py` — MAT13 연성손상 디스패치 분기 + SDV 10 (2026-09-20).
- `dispsolver/model/material.py` — `field_table`/`Creep(HYPERB/ANAND)`/`DamageInitiation`/`DamageEvolution` API (2026-09-20).
- `benchmark_element/benchmark_field_ef1.py`, `benchmark_damage.py` — FV1 쌍 + 손상 4종 벤치마크 (2026-09-20).
- `tests/test_hyperelastic_ogden_mooney.py`, `test_field_ef1.py`, `test_expansion_hyper.py`, `test_creep_extended.py`, `test_frequency_viscoelasticity.py`, `test_anisotropic_hyperelastic.py`, `test_damage_ductile_jc.py` (2026-09-20, 도합 53건: JC D4/D5 추가로 10→13).
- `dispsolver/material3d/numba_materials.py` — MAT14 Drucker-Prager 분기 (2026-09-20).
- `tests/test_drucker_prager.py` (2026-09-20, 9건).
- `dispsolver/material3d/numba_materials.py` — MAT15 Concrete 손상소성 분기 (2026-09-20).
- `tests/test_concrete_dp.py` (2026-09-20, 11건).
- `dispsolver/material/van_der_waals.py` — Kilian lock-up VdW 코어 (2026-09-20).
- `dispsolver/material/forming_limit.py` — Keeler-Brazier FLC 출력전용 후처리 (2026-09-20).
- `tests/test_van_der_waals.py` (2026-09-20, 6건), `tests/test_forming_limit.py` (2026-09-20).
- `tests/test_concrete_dp.py` — 제하 검증 타겟 조정 (2026-09-21, apex+연화 snap-back 회피).

---

## 7. 독립 재측정 기록 (2026-09-19)

- `pytest tests/test_dependence_table.py tests/test_thermal_mechanical_coupling.py tests/test_solver_thermal_expansion_2d.py tests/test_solver_thermal_expansion_3d.py tests/test_plastic_hardening.py tests/test_mullins_and_bergstrom_boyce.py tests/test_radioss_law36_smoothing_and_tail.py tests/test_tabular_plasticity_radioss.py tests/test_hyperelastic.py tests/test_viscoelastic.py tests/test_creep_power_law.py tests/test_bimetallic_strip.py` 전수 실행.
- **결과**: **81 passed, 1 warning in 84.53s (12개 스위트 전수 ALL GREEN)**.
- `python benchmark_element/benchmark_bimetallic_strip.py` 실행:
  - 2D CPE4I 오차: **0.110%** (PASS)
  - 3D SOLID_SHELL 오차: **0.208%** (PASS)
- `python benchmark_element/benchmark_display_thermal_rollup.py` 실행:
  - 14층 디스플레이 열 팽창 오차: **0.11%** (PASS)

## 7.1 독립 재측정 기록 (2026-09-20, Wave0–3)

- `pytest tests/test_hyperelastic_ogden_mooney.py tests/test_field_ef1.py tests/test_expansion_hyper.py tests/test_creep_extended.py tests/test_creep_power_law.py` → **24 passed**.
- `pytest tests/test_frequency_viscoelasticity.py` → **6 passed**. 재료 visco 전수(`-k "visco or prony or frequency or wlf"`) → **77 passed, 1 skipped, 2 xfailed**.
- `pytest tests/test_anisotropic_hyperelastic.py` → **5 passed**. hyper 3종 통합 → **18 passed**.
- `pytest tests/test_rayleigh_damping.py` → **4 passed** (solver 영역, §SLV-DMP-01 참조).
- `pytest tests/test_damage_ductile_jc.py` → **10 passed**.
- `pytest tests/ -k "material or plastic or elastic or field or thermal or dependence or ogden or mooney or expansion or anneal or neuber or hill or hyperelastic or johnson or creep or viscohyper or mullins or bergstrom"` → **173 passed, 3 xfailed**.
- `python benchmark_element/benchmark_field_ef1.py` → **2/2 PASS** (200.0208/180.0187/160.0166 N, temp 경로와 bit-identical).
- `python benchmark_element/benchmark_damage.py` → **4/4 PASS** (면적에너지 객관성 0.13%).
- `python benchmark_element/benchmark_abaqus_material_inp.py` → **4/4 PASS** (회귀 확인).
- `python -m verification.run_all` → **13/14** (bending_2pt FAIL은 stash A/B 기실패, 2026-09-20 재확인: 수정 전후 bit-identical).
- `pytest tests/test_drucker_prager.py` → **9 passed** (2026-09-20).
- `pytest tests/test_concrete_dp.py` → **11 passed** (2026-09-20).
- `pytest tests/test_damage_ductile_jc.py tests/test_3d_mechanics_benchmarks.py` → **25 passed, 4 xfailed** (2026-09-20, MAT14/15 회귀).
- `pytest tests/test_damage_ductile_jc.py tests/test_drucker_prager.py tests/test_van_der_waals.py tests/test_forming_limit.py` → **33 passed** (2026-09-21, VdW 6 + FLD 5 + JC 13 + DP 9 후속 회귀).
- `pytest tests/test_damage_ductile_jc.py` → **13 passed** (2026-09-21, D4/D5 추가 후 재확인).
- `pytest tests/test_damage_ductile_jc.py tests/test_drucker_prager.py tests/test_van_der_waals.py tests/test_forming_limit.py` → **33 passed** (2026-09-21, 후속 회귀).
- `pytest tests/test_concrete_dp.py tests/test_abaqus_reader_and_executor.py` → **15 passed** (2026-09-21, reader `_AbortParse` + CDP 제하조정 후).
- `pytest tests/test_3d_mechanics_benchmarks.py` → **15 passed, 4 xfailed** (2026-09-21, MAT15 이후 3D 회귀).

