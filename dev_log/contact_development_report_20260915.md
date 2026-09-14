# 3D Contact 개발 보고서 (2026-09-13 ~ 2026-09-15)

## 요약

Phase 1(강체 평면 접촉)부터 Phase 2(변형체-변형체 접촉, augmented Lagrangian,
Abaqus급 penalty/soft contact, PDASS 단일 루프 업그레이드)까지 구현 완료.
모든 단계가 실측 검증됨(단순 "수렴했다"가 아니라 물리적으로 올바른지까지
확인 — AGENTS.md §4.9의 프로젝트 원칙에 따름).

---

## 1. Phase 1 — 강체 평면 접촉 (기반)

- `dispsolver/model/interaction.py`: `ContactProperty`/`AnalyticalRigidSurface`/`ContactPair`
  — Abaqus CAE 스타일 API, `Tie`와 동일한 master/slave 패턴.
- `dispsolver/constraint3d/surface_contact3d.py`: `SurfaceContactConstraint3D`
  — frictionless, hard, small-sliding, node-to-rigid-plane penalty contact.
- 검증: `tests/test_contact_phase1.py` 통과.

## 2. Newton solver 자체의 실제 버그 2건 발견 및 수정

Phase 1 검증 중 Hertz 접촉 벤치마크(`benchmark_element/benchmark_3d_contact.py`)가
특정 다중-노드 동시 접촉 지점에서 발산하는 걸 발견 — 근본 원인 추적 결과
**접촉과 무관한, solver 전체에 영향 있는 진짜 버그 2개**를 찾음:

1. **Severe Discontinuity Iteration (SDI) 미처리**: 접촉 노드가 새로 활성화되는
   순간 잔차가 불연속적으로 튀는데, 기존 Newton loop는 이걸 그냥 "수렴 안 됨"으로만
   처리 — Abaqus/Standard의 실제 SDI 처리 방식(활성 집합이 바뀐 iteration은
   일반 수렴 체크에서 예외 처리, 별도 예산으로 처리)을 `solve_step()`에 구현.
2. **Armijo line search 버그**: 6단계 backtracking 중 첫 시도가 무조건 "유효"로
   기록되던 로직 — 잔차가 증가하는 스텝도 그냥 받아들이고 있었음. 수정 후
   진짜 Armijo 조건(개선 없으면 reject)으로 동작.

두 수정 다 전체 3D 회귀 테스트(46/47, 무관한 기존 실패 1건 제외) 영향 없음 확인.

## 3. Augmented Lagrangian — 구현 + 심각한 부호 버그 수정

- `SurfaceContactConstraint3D.augmented_lagrange=True`, `update_augmented_multipliers()`,
  `DynamicSolver3D.solve_step_augmented()` — Uzawa 방식 outer loop.
- **첫 구현에서 기하급수적으로 발산**(cycle마다 ~1.42배씩 나빠짐) — Fable 에이전트가
  근본 원인 발견: contact force를 전역 `f_int_global`에 누적할 때 부호가 반대였음
  (contact은 tie와 달리 물리적 힘이 자기 gap 벡터와 같은 방향이라, `f_int = -F_physical`
  관례에 맞추려면 명시적 부호 반전이 필요했는데 안 하고 있었음).
- 수정 후: 같은 케이스에서 penetration이 cycle마다 ~0.77배씩 **수축**(설계상 예상과
  정확히 일치), 노드별 힘 균형 residual ~1e-10 (진짜 평형, 느슨한 수렴 기준 통과가
  아니라).

## 4. Abaqus급 Penalty/Soft Contact 하드닝

전문 설계 에이전트(Fable) 파견 → `dev_log/contact_abaqus_grade_design_20260915.md`.

- **Nonlinear (4구간) penalty** — 접촉 활성화 순간 강성을 1배(K_i)로 낮추고 실제
  침투가 쌓인 후에만 100배(K_f)로 올리는 Abaqus 방식. **Hertz 벤치마크의 실제
  발산 문제를 해결한 핵심 수정** — 이전엔 45개 이상 노드 동시 접촉에서 막히던 게
  전체 목표까지 cutback 없이 완전 수렴, Hertz 이론값과의 일치도도 개선됨
  (비율 1.40~1.61 → 0.78).
- **Soft contact 3종** (LINEAR/EXPONENTIAL/TABULAR) — `pressure_overclosure.py`의
  공유 `PressureOverclosureLaw` 인터페이스로 hard/nonlinear/soft 전부 통합.
- **자동 강성 계산** (`k_ref`) — 재료(E)와 요소 크기로 자동 유도, 사용자가 절대값
  입력 안 해도 됨. 사용자 요청에 따라 soft contact도 이 기준 강성에서 스케일
  파라미터(`softness_scale`, `allowable_penetration`)로만 조절 가능하게 설계 —
  exponential 곡선 파라미터(c0, p0)까지 닫힌 형식(closed-form)으로 유도.
- Abaqus 자체 문서 확인: augmented Lagrangian은 HARD contact에만 적용됨(soft는
  direct method만) — 그대로 코드에 반영, 위반 시 명시적 에러.
- AL을 Hertz curved-block 케이스에서도 재검증 — 완전 수렴, penetration이
  nonlinear penalty보다 한 자릿수 더 작음(8.7e-5 vs 2.3e-4).

## 5. 변형체-변형체 접촉 (Phase 2 핵심)

`dispsolver/constraint3d/surface_contact3d_deformable.py`:
`DeformableSurfaceContactConstraint3D`.

- 기존 `surface_tie3d.py`의 실전 검증된 node-to-Quad4-face projection을 재사용,
  one-sided(접촉) 버전으로 일반화.
- Master 표면 자체가 진짜로 변형(움직이는 평면이 아님) — 두 큐브를 접촉으로만
  연결한 테스트에서 master 쪽 접촉면도 실제로 압축되는 것 확인(강체였다면
  절대 안 나오는 결과).
- Modified Newton 근사(법선 벡터는 매 호출마다 재계산하지만 tangent에는 미분 안 함)
  — 이 프로젝트가 이미 코로테이셔널 요소에 쓰던 방식과 동일한 정당한 근사.
- `ContactPair.master`가 `AnalyticalRigidSurface` 대신 `Surface`/face 리스트도
  받을 수 있게 CAE API 확장.
- 현재 스코프: HARD+LINEAR penalty만 (soft laws/nonlinear penalty는 아직 미연결,
  다음 확장 대상).

## 6. "Abaqus를 넘어서는" 알고리즘 조사 + PDASS 업그레이드

전문 연구 에이전트(Fable) 2차 파견 →
`dev_log/contact_beyond_abaqus_research_20260915.md`,
`dev_log/contact_pdass_precise_design_20260915.md`.

**핵심 발견**: 우리가 이미 만든 augmented Lagrangian의 수식(`p = max(0, λ+k(-gap))`)이
정확히 **Alart-Curnier NCP projection**과 같음. Hüeber & Wohlmuth (2005)가
증명한 PDASS(Primal-Dual Active Set Strategy) ≡ semi-smooth Newton이 바로
이 수식 위에서 성립 — 차이는 λ를 outer loop에서 고정하느냐(기존 방식), 아니면
Newton iteration 안에서 매번 업데이트하느냐(PDASS/Simo & Laursen 1992 single-loop
구성) 뿐.

- Nitsche's method, Dual Mortar도 조사 — **어떤 방법도 접촉 활성화 시점의
  불연속 자체를 없애지는 못함**(complementarity 조건의 구조적 특성, 이미
  soft contact에서 찾은 결론과 동일하게 재확인됨).
- **구현**: `update_augmented_multipliers()` 호출을 outer loop에서 빼서
  `solve_step()`의 Newton loop 안(accepted iterate가 확정되는 두 지점)으로 이동.
  `assemble()`의 수식은 전혀 안 바뀜 — 오직 λ가 "언제" 갱신되느냐만 바뀜.
- 측정 결과: 기존 10 outer cycle → **4 cycle**로 감소, cycle당 inner iteration도
  5~9개 → **2개**로 감소. 설계가 예측한 "정확히 1 cycle"까지는 아니지만(Newton
  자체 수렴 기준이 λ의 완전 수렴보다 먼저 만족되는 경우가 있어서), 총 작업량은
  확실히 줄었고 정확도(penetration)는 그대로 유지됨.
- 변형체-변형체 접촉에도 동일하게 포팅 — penetration 400배 감소(0.0314→0.0000752)
  확인, 물리적으로 힘이 진짜 평형값에 더 가까워짐.

---

## 검증 상태 (전부 실측, 추측 없음)

| 항목 | 상태 |
|---|---|
| Phase 1 rigid-plane 접촉 | ✅ 검증됨 |
| SDI + line search 수정 | ✅ 검증됨, 회귀 없음 |
| Augmented Lagrangian (부호 수정 후) | ✅ 검증됨 |
| Nonlinear penalty | ✅ 검증됨 (Hertz 발산 해결) |
| Soft contact 3종 + 자동 스케일 | ✅ 검증됨 |
| AL on Hertz 재검증 | ✅ 검증됨 |
| 변형체-변형체 접촉 | ✅ 검증됨 (실제 두 물체 상호작용 확인) |
| PDASS single-loop | ✅ 검증됨 (부분 개선, 정확한 이상치는 아님) |

---

## 다음 단계 (우선순위 순, 사용자와 논의된 내용 반영)

1. **Dual Mortar discretization** — 비정합 메쉬 접촉면에서 결합행렬을 대각행렬로
   만들어 요소 수준 정적 응축 가능하게 하는 업그레이드 (Wohlmuth 2000,
   Popp/Gee/Wall 2010). PDASS와 결합하면 접촉 있는 해석이 접촉 없는 일반
   탄성해석과 같은 속도로 풀림 — 다음 실제 구현 후보.
2. **변형체-변형체에 nonlinear penalty/soft contact 연결** — 지금은 HARD+LINEAR만.
3. **Phase 3: 마찰(Coulomb friction)** — 아직 시작 안 함.
4. **§A.3 real contact stabilization** — 낮은 우선순위, 필요한 예제 생길 때.
5. **§A.4 interference-fit 밀어내기** — 사용자 지시로 뒤로 미룸.
6. **Nitsche's method SPD 주장 검증** — 사용자가 제시한 외부 자료의 강한 주장
   (페널티 민감도 극히 낮음 + SPD 유지)이 우리 연구와 결이 달라 원문 검증 필요.

## 관련 문서 전체 목록

- `dev_log/3d_contact_implementation_design_20260913.md` — 최초 Phase 1/2 설계
- `dev_log/hertz_contact_benchmark_20260913.md` — SDI/line search 수정 + 발견 기록
- `dev_log/contact_abaqus_grade_design_20260915.md` — penalty 하드닝 + soft contact 설계
- `dev_log/contact_beyond_abaqus_research_20260915.md` — Abaqus 이후 알고리즘 조사
- `dev_log/contact_pdass_precise_design_20260915.md` — PDASS 정밀 설계
- `dev_log/contact_development_report_20260915.md` — 본 보고서
