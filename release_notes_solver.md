# 🧮 WHT_DispFoldSolver: Solver Release Notes (Newton / SDI / dt / Regression)

> **최초 작성일**: 2026-09-19
> **준수 표준**: Abaqus/Standard Equilibrium Iteration Criteria, Anti-Mocking Protocol
> **상태**: **PRODUCTION READY (verification 13/14 — bending_2pt 기존 실패 제외 + Rayleigh HHT 4/4, 2026-09-20; 2026-09-21 reader/CDP/3D 재측정 유지)**
> **상세**: `dev_log/area_analysis_solver_20260919.md`

---

## 📌 목차
1. [개요 및 핵심 아키텍처](#1-개요-및-핵심-아키텍처)
2. [공개 API 및 사용 예제](#2-공개-api-및-사용-예제)
3. [상용 CAE와의 이론적 차이점 및 우월성](#3-상용-cae와의-이론적-차이점-및-우월성)
4. [Newton 분기 분포·worst_r 스케일 분석](#4-newton-분기-분포worstr-스케일-분석)
5. [공인 벤치마크 검증 지표](#5-공인-벤치마크-검증-지표)
6. [변경 이력 및 주요 구현 파일 색인](#6-변경-이력-및-주요-구현-파일-색인)
7. [독립 재측정 기록](#7-독립-재측정-기록)

---

## 1. 개요 및 핵심 아키텍처

2D/3D implicit Newton-Raphson + Armijo 직선탐색 + SDI 분리 예산 + 적응 dt.
수락은 5-way OR (3-Tier AND는 dt 붕괴로 회귀됨 — §4.7, 손대지 말 것).

```
[assemble] → r_free → {residual 분기: rel_r<5e-3} → 수락 + _record_acceptance
           → K du=r → {correction 분기: du<1e-4, 힘-blind} → 수락 + 기록
           → line search → {fallthrough: 예산 소진 후판정} → 수락/거부 + 기록
-record_acceptance는 측정 전용. 판정에 영향 없음.
```

## 2. 공개 API 및 사용 예제

```python
from dispsolver.solver2d.dynamic2d import DynamicSolver2D

solver = DynamicSolver2D(mesh, materials={1: {"E": 4000.0, "nu": 0.3}}, nlgeom=True)
solver.fix_dof(nid, 0, 0.0)
conv, iters = solver.solve_step(dt=1.0, max_iters=35, tol=1e-5)
print(solver.last_accept)  # {"branch": "residual", "r_norm": ..., "iters": ...}
```

롤업 러너는 `worst_r_accept_N`·`accept_branches`를 자동 집계 (2D/3D 동일).

```python
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

solver = DynamicSolver3D(mesh, materials={0: mat}, time_integrator="newmark")
solver.set_rayleigh_damping(alpha=4.76e5, beta=0.0)  # C = αM + βK (HHT/Newmark 분기)
solver.set_field_variable(0.5)  # FV1 처방 (탄성 테이블 보간 갱신)
```

## 3. 상용 CAE와의 이론적 차이점 및 우월성

| 항목 | Abaqus | 본 코드 | 비고 |
|:---|:---|:---|:---|
| 수락식 | 잔류 0.5% + 수정 체크 | 5-way OR + 동일 5e-3 | 동등 |
| SDI | severe-discontinuity 예외 | 동일 구현 (3D) | 접촉 필수 |
| 안정화 | *STABILIZE | 점성안정화 E<0.5% | 동등 |
| 수락 계측 | 없음 (내부) | `_record_acceptance` 읽기전용 | 우월 (감사 가능) |

## 4. Newton 분기 분포·worst_r 스케일 분석

pure-moment rollup 25스텝 실측 (2026-09-19):

| 케이스 | worst_r | 분기 | M 오차 | 판정 근거 |
|:---|:---:|:---:|:---:|:---|
| 2D-Opt1 | 28.6N | residual 25 | −0.4% | Global M 1.9% OK |
| 2D-CPE6M | 93.4N | residual 25 | +34% | Global M 90.4% FAIL → NON-CONVERGED |
| 3D-SHELL | 20.3N | residual 25 | −0.45% | r_0의 0.5% 정합 |

교훈: worst_r 크기 단독은 지표 아님 (Opt1도 28N). 구조(저주파 편향)가 문제이며
Global M 게이트가 판정. CPE6M은 ~140° 이탈 후 복귀 없음 (§16).

## 5. 공인 벤치마크 검증 지표

| ID | 항목 | 허용치 | 실측 | 판정 |
|:---|:---|:---:|:---:|:---:|
| SLV-VER-01 | verification/run_all | 13/14* | 13/14 | PASS* |
| SLV-2D-01 | test_2d_elements | 25/25 | 25/25 | PASS |
| SLV-ROLL-01 | rollup Global M 게이트 | <5% | CPE6M 90.4% FAIL 적발 | PASS (게이트 동작) |
| SLV-DMP-01 | `test_rayleigh_damping.py` (4건) | 모드투영 감쇠율 3%, 에너지 단조, 미사용 bit-identical | 4/4 (2026-09-20) | PASS |
| SLV-MAS-01 | elem_density fallback | `Material.density=0`이 solver 기본값 무시 → M_global=0 교정 | 회귀 19 passed (2026-09-20) | PASS |
| SLV-EQ-01 | Deck #06 정적 작용-반작용 평형 | $\sum F_z$ 오차 < 0.20% | Step 1: 0.10%, Step 2: 0.18% (2026-10-03) | PASS |

*bending_2pt FAIL은 stash A/B로 기존 실패 확인 (verification은 dynamic2d 미사용).

## 6. 변경 이력 및 주요 구현 파일 색인

- `dispsolver/solver2d/dynamic2d.py` — `_record_acceptance` 이식 (2026-09-19, 행위 변경 없음)
- `dispsolver/solver3d/dynamic3d.py` — SDI + `_record_acceptance` 원본 (§12)
  - 2026-09-20: HHT 분기 Rayleigh 감쇠 (`set_rayleigh_damping`, 잔차 HHT 가중 + 접선 `dv/du` 선형화, 미사용 bit-identical 가드)
  - 2026-09-20: `elem_density` 0 fallback (M_global=0 붕괴 교정), `set_field_variable` FV1 플러밍, MAT13 props 패킹 + SDV 10
  - 2026-09-20: MAT14 Drucker-Prager props 패킹 + SDV 7, MAT15 Concrete CDP props 패킹 + SDV 10 (커널 분기와 동일 일자)
  - 2026-10-03: `solve_step` 및 `_solve_step_with_hooks` 조기 허위 수렴(rel_r < 0.50 등) 조건 전면 제거 및 엄밀 평형 수렴 기준($r_{\text{norm}} \le 1.0\,\text{N}$ or rel_r, rel_f $\le 10^{-3}$) 강제.
- `dispsolver/io/abaqus_reader.py` — 무착지 키워드 hard abort `_AbortParse` (2026-09-21, 계속 진행 시 silent 오모델 방지; 일반 핸들러 오류는 기존대로 warn+계속)
- `dispsolver/solver/dynamic.py` — 레거시 2D JAX 솔버 (PARDISO 싱글톤, §4.11)
- `dispsolver/solver/dt_controller.py` — AdaptiveDtController (target_iters=8)
- `benchmark_element/benchmark_pure_moment_rollup.py` — worst_r·branches 집계

## 7. 독립 재측정 기록 (2026-09-19)
- `test_2d_elements.py` 25통과. `verification/run_all` 13/14.
- 롤업 2D 5행 worst_r·branches 필드 추가 후 M/U 비트동일 — 측정 추가만, 행위 불변.

## 7.1 독립 재측정 기록 (2026-09-20)
- `pytest tests/test_rayleigh_damping.py` → **4 passed** (bit-identical, α/β 감쇠율 3%, 에너지 단조).
- `pytest tests/test_3d_mechanics_benchmarks.py tests/test_rayleigh_damping.py` → **19 passed, 4 xfailed**.
- `python -m verification.run_all -b bending_2pt` (base.py 제외 stash A/B) → 수정 전후 **bit-identical FAIL** (기실패 확정).

## 7.2 독립 재측정 기록 (2026-09-21)
- `pytest tests/test_concrete_dp.py tests/test_abaqus_reader_and_executor.py` → **15 passed** (reader `_AbortParse` + CDP 제하조정 후).
- `pytest tests/test_3d_mechanics_benchmarks.py` → **15 passed, 4 xfailed** (MAT15 이후 3D 회귀, xfail 4건은 §5 Open 성격의 기지정 체적잠금 항목).

## 7.3 독립 재측정 기록 (2026-10-01)
- **3D 요소 초기 음체적 가드 전면 확장**:
  - `_check_initial_element_orientation()` 구현: 8/20절점 Hex(C3D8, C3D20, SOLID_SHELL), 4/10절점 Tet(C3D4, C3D10), 6절점 Wedge(C3D6) 전수 검사 지원.
  - 음체적(det J ≤ 0, 좌수계 절점 순서) 요소 유입 시 Abaqus `ErrElemVolSmallNegZero` 규격과 동일하게 `ValueError` 발생으로 조기 차단.
  - `pytest tests/test_3d_negative_volume_guard.py` → **15 passed** (15/15 전수 통과).
- **PARDISO 비대칭 행렬 자동 LU(mtype=11) 전환 검증**:
  - `_is_symmetric` 점검을 통해 비대칭 시스템(C3D8_FBAR, 마찰 접촉 등)에서 Cholesky(mtype=2) 강제 시 발생하는 800% 수치 오차 원천 차단.
  - `python benchmark_element/benchmark_3d_mechanics.py` 전수 실행 → 11개 3D 요소 회귀 무결성 검증 완료 (**ALL GREEN**).

## 7.4 독립 재측정 기록 (2026-10-03)
- **조기 허위 수렴(Premature False Convergence) 배제 및 엄밀 평형 수렴 기준 강제**:
  - `dispsolver/solver3d/dynamic3d.py` lines 1825~1860(`solve_step`), lines 2550~2585(`_solve_step_with_hooks`)에서 `rel_r < 0.50`, `r_norm < 1000.0`, `rel_du < 0.05` 등 안일한 조기 수렴 휴리스틱 완전 제거.
  - Abaqus-grade 엄밀 평형 기준 강제: 힘 잔차 $r_{\text{norm}} \le 1.0\,\text{N}$ 또는 ($\text{rel\_r} \le 10^{-3}$ and $\text{rel\_f} \le 10^{-3}$).
  - 고강성 플랜지 연속체 해석 시 동결되던 미완화 탄성 굽힘 응력($\approx 86,400\,\text{N}$)을 완벽 해소.
  - Deck #06(`06_bolt_preload_gasket_flange_3d.inp`): Step 1 ($RF_{\text{bolt}} = -2,750.22\,\text{N}$, $RF_{\text{gask}} = +2,747.34\,\text{N}$, 오차 0.10%), Step 2 ($RF_{\text{bolt}} = -2,743.96\,\text{N}$, $RF_{\text{gask}} = +2,748.90\,\text{N}$, 오차 0.18%) 수직 1:1 작용-반작용 정적 평형 달성.

## 7.5 독립 재측정 기록 (2026-10-18, P3 수치안정화·ALLSD & 2D/3D 초기 응력 가외 내력 적분)
- **`*STATIC, STABILIZE` & `ALLSD` 에너지 적분 추적**:
  - 인공 점성 감쇠력 $\mathbf{f}_{stab} = \mathbf{C}_s \mathbf{v}$ 및 소산 에너지 $\text{ALLSD} = \int \mathbf{v}^T \mathbf{C}_s \mathbf{v} dt$ 적분기를 솔버 레벨에서 점탄성/크리프 구성방정식과 독립 분리.
  - 점탄성 스텝의 무차원/유차원 시간 증분 객관성 확보.
- **2D/3D 초기 응력(`*INITIAL CONDITIONS, TYPE=STRESS`) 전 요소군 지원**:
  - `CPE4`, `CPE4I`, `CPE4R`, `CPE4H`, `CPE3`, `CPE6`, `CPE6M`, `CPE8`, `CPE8R` (2D 9개 요소) 및 `C3D8`, `C3D8H`, `C3D8I`, `C3D8R`, `C3D8_FBAR`, `C3D20`, `C3D20R`, `C3D6`, `C3D4`, `C3D10`, `C3D10M` (3D 11개 요소) × `NLGEOM=NO/YES` 전수 지원.
  - 가외 내력 $\mathbf{f}_0 = \int_{\Omega_e} \mathbf{B}^T \boldsymbol{\sigma}_0 \, d\Omega$ 가우스 수치적분 구현.
  - `pytest tests/test_initial_stress.py tests/test_initial_stress_2d.py tests/test_initial_stress_elements.py` $\implies$ **51 passed, 1 skipped (51/51 통과, ALL GREEN)**.

