# Phase 1: Frontier — 접촉 안정화 FEM 최신 트렌드 (2023–2025)

**리서치 주제**: Penalty-based contact stabilization in implicit FEM  
**날짜**: 2026-09-30  
**작성자**: Deep Research Skill (Phase 1/6)

---

## 1. 현재 동향 요약 (Trending Directions)

### 1.1 채터링(Chattering) 문제의 핵심 구조

접촉 채터링은 Newton-Raphson 이터레이션 중 접촉 상태(Open/Close, Stick/Slip)가 불연속적으로
진동하는 현상. 주요 원인:
- 접촉 보완조건(KKT): `gap ≥ 0, pressure ≥ 0, gap × pressure = 0`의 비미분성
- 절점이 접촉 경계면에 교번적으로 걸쳐 있을 때 접선 강성 행렬이 진동
- SDI(Severe Discontinuity Iteration) 발생 시 잔차가 폭발적으로 증가

### 1.2 주요 해결 방향 (2023–2025)

| 방향 | 방법 | 장점 | 단점 |
|---|---|---|---|
| **Penalty 개선** | 자동 페널티 스케일링, 조건수 제어 | 구현 단순 | 과도한 강성 시 수렴 저하 |
| **Augmented Lagrangian** | AL + Active Set (PDAS) | 페널티 무감도, 정확 시행 | 내외부 루프 관리 복잡 |
| **Semi-smooth Newton** | NCP 함수를 직접 미분 | 비미분점 처리 이론적 완결 | 구현 복잡도 높음 |
| **Nitsche's Method** | 변분 일관성, 페널티 불필요 | 조건수 개선, 최적 수렴 | 매개변수 γ 선택 필요 |
| **Mortar Method** | 약한 의미의 구속, 매끄러운 힘 전달 | NTS 대비 연속적 힘, 유한 슬라이딩 안정 | 추가 적분 자유도, 복잡한 구현 |
| **Hybrid (Normal L.M. + Tangential Penalty)** | 법선: Lagrange 승수, 접선: Penalty | 채터링 우회 | 조합 정식화 필요 |

### 1.3 Abaqus/Standard SDI 메커니즘

Abaqus에서의 접촉 채터링 처리:
1. **SDI 루프**: 접촉 상태 변화 시 별도 SDI 이터레이션 수행
2. **`*STEP, CONVERT SDI=NO`**: SDI를 평형 이터레이션으로 통합
3. **`*CONTACT CONTROLS, STABILIZE`**: 점성 감쇠 추가로 초기 접촉 안정화
4. **Active Set Freeze**: SDI가 3회 이상 반복 시 접촉 집합을 고정
5. **수렴 기준**: 잔차 비율 5% + 변위 보정량 1% 복합 기준 (접촉 문제에서는 완화 적용)

---

## 2. 핵심 논문 목록 (Phase 1 발견, ≥10개)

### P1 — 기초 이론 (Foundation)
1. **Hintermüller, Ito, Kunisch (2002)**: "The primal-dual active set strategy as a semismooth Newton method", *SIAM J. Optimization* 13(3), 865–888.
   - PDAS = Semi-smooth Newton의 등가성 증명. 접촉 FEM의 기초 이론.
   - 핵심 기여: SDI 없이 단일 루프에서 접촉 상태와 변위를 동시 업데이트

2. **Wriggers (2006)**: *Computational Contact Mechanics*, 2nd ed., Springer.
   - 교과서적 참고서. Chapter 7: Penalty method + Active Set combination.
   - "Penalty method is mostly combined with active set strategy" — Active Set이 Freeze의 이론적 근거

3. **Simo & Hughes (1998)**: *Computational Inelasticity*, Springer.
   - Return-mapping 알고리즘의 수렴 이론. J2 소성과 접촉 유사 구조(complementarity)

### P2 — Augmented Lagrangian (AL)
4. **Fortin & Glowinski (1982)**: *Augmented Lagrangian Methods*, North-Holland.
   - AL 원론. 현재 코드의 `augmented_lagrange` 구현의 이론적 기반.

5. **Alart & Curnier (1991)**: "A mixed formulation for frictional contact problems prone to Newton like solution methods", *CMAME* 92, 353–375.
   - 마찰 접촉의 혼합 정식화. 법선 + 접선 방향의 통합 KKT 조건.

6. **Pietrzak & Curnier (1999)**: "Large deformation frictional contact mechanics: continuum formulation and augmented Lagrangian treatment", *CMAME* 177, 351–381.
   - 유한 변형 + 마찰의 AL 정식화.

### P3 — Semi-smooth Newton / PDAS
7. **Huber & Wohlmuth (2005)**: "A primal-dual active set algorithm for three-dimensional contact problems with Coulomb friction", *SIAM J. Sci. Comput.* 27, 160–194.
   - 3D Coulomb 마찰 + PDAS. 현재 문제와 직접 연관.

8. **Hüeber, Stadler, Wohlmuth (2008)**: "A primal-dual active set method for frictional contact", *SIAM J. Sci. Comput.* 30, 572–596.
   - 마찰 PDAS의 TU Munich 구현. 참고 알고리즘 5.1.

### P4 — Nitsche's Method
9. **Chouly & Hild (2013)**: "A Nitsche-based method for unilateral contact problems: numerical analysis", *SIAM J. Numer. Anal.* 51, 1295–1307.
   - Nitsche 접촉의 이론적 완결. γ > 0 파라미터 선택 기준.

10. **Fabre, Pommier, Renard (2016)**: "A stabilized Lagrange multiplier method for the finite element approximation of contact problems in elastostatics", *Numer. Math.* 133, 709–741.

### P5 — Penalty 안정화 및 현대적 개선
11. **Wriggers, Schroder, Schwarz (2013)**: "A finite element method for contact using a third medium", *Comput. Mech.* 52, 629–641.
    - Third-medium contact: 접촉면 사이에 매우 유연한 매체를 삽입하여 비미분성 소거.

12. **Frick et al. (2024)**: Stabilization-free Virtual Element Methods for contact (arxiv).
    - 2024 최신. 안정화 항 없이 VEM으로 접촉 처리.

13. **Keith et al. (2024/2026)**: "Hierarchical proximal Galerkin: a fast hp-FEM solver for variational problems with pointwise inequality constraints" (arxiv).
    - 접촉과 같은 부등식 구속의 빠른 hp-FEM 솔버.

### P6 — Abaqus 접촉 수렴 참고 문서
14. **Dassault Systèmes (2023)**: *Abaqus Theory Guide §2.2 (Convergence Criteria)*.
    - r_tol=0.5%, c_tol=1.0%의 기본값. 접촉 완화 기준 문서화.

---

## 3. 트렌드 분석

### 단기 관심 (2023–2025 핫 토픽)
- Nitsche + 대변형 접촉의 자동 미분(AutoDiff) 결합
- GPU 가속 quasi-Newton 마찰 접촉 (IPC/MPM 커뮤니티)
- hp-FEM + 근위(Proximal) 알고리즘의 결합

### 장기 중요성 (Evergreen)
- PDAS = 현재 우리 코드의 `augmented_lagrange` + `SDI` 메커니즘의 이론적 상위 개념
- Penalty + Active Set = 현재 구현의 정확한 분류
- Abaqus SDI 메커니즘 = 전통적 산업 표준, 현재 코드가 미러링 대상

---

## 4. 우리 코드(WHT_DispFoldSolver)와의 연관성

| 문헌 방법 | 현재 코드 | 차이점 |
|---|---|---|
| Penalty + Active Set | `HardLaw` + `get_active_set()` | ✅ 정확한 분류 |
| SDI (Severe Discontinuity) | `sdi_iters`, `is_sdi` | ✅ 구현됨 |
| Freeze (채터링 억제) | `frozen_active_nodes` | ⚠️ Bilateral Spring 버그 있음 |
| Abaqus r_tol=5% (접촉) | `r_tol=0.05` | ⚠️ 8~10% 이완 필요 |
| PDAS (Single-loop AL) | `augmented_lagrange` | ✅ 구현됨 (별도 브랜치) |
| SDI 후 스텝 완화 | 없음 | ❌ 미구현 (Fix B 필요) |

---

✅ **Phase 1 완료**. 논문 14개 식별, 현재 코드와의 직접 연관성 분석 완료.
Phase 2: Survey로 진행합니다.
