# 🔗 WHT_DispFoldSolver: Constraint Release Notes (Tie / RBE2 / Contact Constraints)

> **최초 작성일**: 2026-09-19
> **준수 표준**: Abaqus *TIE / *COUPLING / *CONTACT PAIR Semantics, Anti-Mocking Protocol
> **상태**: **PRODUCTION READY (릴리즈 지정 9파일 58/58, 2026-09-19 재측정)**
> **상세**: `dev_log/area_analysis_constraint_20260919.md`

---

## 📌 목차
1. [개요 및 핵심 아키텍처](#1-개요-및-핵심-아키텍처)
2. [공개 API 및 사용 예제](#2-공개-api-및-사용-예제)
3. [상용 CAE와의 이론적 차이점 및 우월성](#3-상용-cae와의-이론적-차이점-및-우월성)
4. [강성·간극 수치 및 병목 분석](#4-강성간극-수치-및-병목-분석)
5. [공인 벤치마크 검증 지표](#5-공인-벤치마크-검증-지표)
6. [변경 이력 및 주요 구현 파일 색인](#6-변경-이력-및-주요-구현-파일-색인)
7. [독립 재측정 기록](#7-독립-재측정-기록)

---

## 1. 개요 및 핵심 아키텍처

디스플레이-플레이트 결합(tie), 강체 구동(RBE2), 3D 접촉구속을 세 갈래로 분리.
구동은 exact 응축, 결합은 penalty tie, 접촉은 dual mortar — 각각 SPD 유지가 원칙.

```
        [구동: RigidBodyPart + RBE2 응축]      [결합: SurfaceTie]      [접촉: Dual Mortar]
        u_s = u_m + (R-I)(X_s-X_m)            penalty node-to-seg     u_s = D^-1 M u_m
                   │                                     │                         │
                   └──────────────────┬──────────────────┘                         │
                                      ▼                                            ▼
                         [공유 조립 꼬리 (tie/RBE2 penalty)]              [SPD Cholesky]
                         신규 조립 분기는 return 금지·fallthrough (§4.8)
```

## 2. 공개 API 및 사용 예제

```python
from dispsolver.part.rigid_body import RigidBodyPart
from dispsolver.constraint.surface_tie import SurfaceTieConstraint

# 강체 플레이트 구동: 매 스텝 slave 변위를 직접 Dirichlet BC로
u_slave = plate.get_slave_displacements(u_master, theta)
solver.set_prescribed_dofs(plate_dofs, u_slave)

# display-plate 결합
tie = SurfaceTieConstraint(slave_nodes, master_segments, penalty_stiffness=1e5)
```

`RBE2HingeElement`로 플레이트를 구동하지 말 것 — penalty+AL 방식은 이미 실패 확정(§4.10).
`ContactPair`/`ContactStd` 예제는 `release_notes_contact.md` §2 참조.

## 3. 상용 CAE와의 이론적 차이점 및 우월성

| 항목 | Abaqus | 본 코드 | 비고 |
|:---|:---|:---|:---|
| Tie | *TIE, position tolerance | penalty node-to-seg + 투영 | ex12 간극 1e-5 mm |
| Rigid drive | *COUPLING kinematic | RBE2 응축 = Dirichlet (exact) | penalty 회귀 금지 |
| Mortar | S2S + AL | Dual mortar + Wohlmuth 응축, SPD | saddle-point 없음 |

## 4. 강성·간극 수치 및 병목 분석

- Tie 간극: `max_gap ≈ 1e-5 mm` (`max|u| ≈ 1 mm`, k=1e5) — k를 1e2→1e10으로 바꿔도
  궤적 동일 (플레이트가 exact rigid라 tie가 병목이 아님, §4.10).
- 팁 역전: tie 강성이 아니라 **메시 grading**으로 해결 (0.25 mm 클러스터링, §4.12).
  tie를 stiffen하면 오히려 악화 — 반대 방향.
- `skip_fully_prescribed` (구현済·기본 OFF): 완전구속 요소 조립 스킵. 켤 때 sanity_report 동반.

## 5. 공인 벤치마크 검증 지표

| ID | 항목 | 허용치 | 실측 | 판정 |
|:---|:---|:---:|:---:|:---:|
| CTC-TIE-01 | ex12 tie 간극비 | < 1e-3 | ~1e-5 | PASS |
| CTC-API-01 | 9파일 접촉 스위트 | 58/58 | **58/58** (48.6s) | PASS |
| CTC-ORD-01 | 테스트 순서 독립성 | 순서 무관 | friction→api 2실패 | **FAIL (Open)** |

## 6. 변경 이력 및 주요 구현 파일 색인

- `dispsolver/constraint/surface_tie.py` — penalty tie (§4.8 fallthrough 수정済)
- `dispsolver/constraint/rbe2_condensed.py` — exact 응축 매니저
- `dispsolver/part/rigid_body.py` — RigidBodyPart (Dirichlet 구동)
- `dispsolver/constraint3d/dual_mortar_clipping3d.py` — Stage 5 dual mortar (untracked)
- `dispsolver/contact/general_contact3d.py` — mu/k_t/앵커/반환사상 (untracked)

## 7. 독립 재측정 기록 (2026-09-19)
- §6.2 9파일 명령 재실행: **58 passed**. 58/58 주장 확인.
- 순서 의존성 신규 발견: friction→abaqus_api 2파일 실행 시 후자 2건 실패.
  역순·단독·9파일은 통과 → 테스트 격리 결함 판정, 수정 대기.
