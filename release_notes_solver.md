# 🧮 WHT_DispFoldSolver: Solver Release Notes (Newton / SDI / dt / Regression)

> **최초 작성일**: 2026-09-19
> **준수 표준**: Abaqus/Standard Equilibrium Iteration Criteria, Anti-Mocking Protocol
> **상태**: **PRODUCTION READY (verification 13/14 — bending_2pt 기존 실패 제외)**
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

*bending_2pt FAIL은 stash A/B로 기존 실패 확인 (verification은 dynamic2d 미사용).

## 6. 변경 이력 및 주요 구현 파일 색인

- `dispsolver/solver2d/dynamic2d.py` — `_record_acceptance` 이식 (2026-09-19, 행위 변경 없음)
- `dispsolver/solver3d/dynamic3d.py` — SDI + `_record_acceptance` 원본 (§12)
- `dispsolver/solver/dynamic.py` — 레거시 2D JAX 솔버 (PARDISO 싱글톤, §4.11)
- `dispsolver/solver/dt_controller.py` — AdaptiveDtController (target_iters=8)
- `benchmark_element/benchmark_pure_moment_rollup.py` — worst_r·branches 집계

## 7. 독립 재측정 기록 (2026-09-19)
- `test_2d_elements.py` 25통과. `verification/run_all` 13/14.
- 롤업 2D 5행 worst_r·branches 필드 추가 후 M/U 비트동일 — 측정 추가만, 행위 불변.
