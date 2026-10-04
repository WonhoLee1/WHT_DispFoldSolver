# Deck #06 정정 요약 (2026-10-04)

Gemini 세션(`dev_log/session_20261004_handoff.md`)과 그 결과를 옮긴 문서들의 Deck #06 관련 주장 중 **틀린 것**과 **올바른 값**을 한곳에 모은다. 근거는 `dev_log/plan_deck06_contact_fix_20261004.md`(§1b, §9)와 커밋 `21d62fb`..`1ef2024`. 수치는 모두 freeze를 해제하고 실제 단방향 접촉 법칙으로 다시 조립한 수렴 상태에서 잰 값이다.

## 1. 틀린 주장 → 올바른 내용

| # | 주장 (출처) | 상태 | 올바른 내용 |
|---|---|---|---|
| C1 | "PENALTY(10x)와 NITSCHE(1x) 반력이 2.4% 이내로 일치 → 유일 물리 정해(Unique Ground Truth) 수렴 입증" (GEMINI.md L207-211, release_notes_contact.md §14 L357, lessons_learned_contact_harmonization L87, handoff §2.4) | **틀림** | 메시 오차만 약 5%(m=1→2 세분화). 2.4%는 그보다 작은 차이이므로 아무것도 입증하지 않는다. 당시 "NITSCHE"는 penalty였다(C3). |
| C2 | NITSCHE 반력 −6,412.48 N, 평형오차 0.0017% (같은 출처), −6,346.3 N (release_notes Case 3/4) | **대체됨** | 진짜 Nitsche(`3686fd0`): x1/x10/x100 = −6,584.6 / −6,569.0 / −6,567.8 N. PENALTY x10..x300 = −6,543 .. −6,664, 극한 ≈ −6,668. 두 방법 간 1.5% 차이(NTS vs STS 이산화)는 남음. |
| C3 | "대칭 Nitsche (Chouly et al. 2017), 극단적 penalty 없이 정확한 비침투 수렴" (nitsche_contact3d.py docstring, GEMINI.md L222, release_notes L31/L304) | **틀림** | 해당 클래스는 `p = γ·g`만 있는 면적분 penalty였다(patch test 오차가 1/γ에 비례: −16.6%/−2.0%/−0.045%). 이제 θ=0 Nitsche로 교체됨. 진짜 Nitsche 접선은 **비대칭**이다. 옛 penalty는 `consistency=False`로만 사용 가능. |
| C4 | SDI freeze 범위를 `recent_active_sets[-2:]`로 수정 완료 (handoff §2.2, lessons L55-61) | **불완전했음** | `.inp` 경로(`_solve_step_with_hooks`)에는 적용되지 않았다. P1(`8f38bca`)에서 두 루프 판정 로직 단일화, P2(`40d09f6`)에서 범위 통일. |
| C5 | "엄격한 Abaqus급 수렴 기준" (handoff §2.3, dynamic3d 주석) | **틀림** | 절대 공차(`r<1.0 N`, `r<10 N`)와 부풀려진 `r_0`, 소진 분기의 `rel_r<5e-3`, line search `min_r<1e5` 우회. PENALTY x50이 r=90 N으로 수렴 판정됨. P3(`4bc3eb9`)에서 상대 기준으로 교체. |
| C6 | "Abaqus Chattering Freeze standard: bilateral spring" (surface_contact3d_deformable.py 주석) | **틀림** | 정상 상태를 진동으로 오판해 3번째 iteration마다 freeze, freeze 노드는 압력 법칙을 선형 스프링으로 대체(soft exp 미수렴 원인). P3/P2에서 수정. |
| C7 | Step 2 내압 해석 수렴 (handoff, summary.json) | **무의미했음** | 5 MPa 내압이 −z 합력 5 N으로 들어갔다. P4(`9064a43`) 후 합력 (487.5, 487.5, 0) N = p·R·H, 정상 수렴. |
| C8 | `.sta`: CONVERGED, cutback 0 | **가짜였음** | step당 고정 1행. P5(`6db3500`) 후 실제 증분 기록(Step 1 = 4 증분). |
| C9 | 0-gap 부호 가드(`abs(gap0) < 1e-5`)로 부호 반전 해결 | **부분 해결이었음** | 억지끼움(초기 겹침 > 10 nm)에서는 여전히 법선 반전. P6(`911d5e0`)에서 master 요소 바깥 방향 기준으로 교체. Mortar도 같은 계열 결함(`1ef2024`). |
| C10 | deck 경로 `benchmarks/abaqus_standard_inp/...` (handoff, lessons dev_log) | **틀림** | `benchmark_inp/phase2_methodology/06_bolt_preload_gasket_flange_3d.inp` |
| C11 | 가스켓 "초탄성 EPDM (Mooney-Rivlin)" | **틀림** | `*ELASTIC 1000, 0.38` 선형 탄성 (GEMINI.md L261은 이미 이를 지적함) |
| C12 | Mortar: "positive-preserving dual basis" 주석 | **미구현** | 주석만 있고 기존 clamp가 삭제됨. 별도 조사 대상. |
| C13 | DUAL_MORTAR는 미소 슬라이딩 볼트 체결에서 클리핑 꼭짓점 진동(4↔5) 때문에 수렴하지 못한다 (lessons_learned dev_log §3.3, GEMINI.md L219-221) | **틀린 진단** | 실제 원인은 무작위 법선, λ_h<0 적분점 누락, 과거 기하 활성 판정의 세 결함(LESSONS_LEARNED 6.11). 수정 후 x1 −6,570.8 N, x10 −6,698.5 N 수렴. |

## 2. 이번 작업에서 새로 발견된 결함 (문서에 없던 것)

- **Pardiso 대칭 판정**: `_is_symmetric`이 `numpy.bool_`을 반환하고 resolver가 `is False`로 검사 → 모든 비대칭 접선이 상삼각 대칭화로 풀림(`3686fd0`). 2026-10-01 이후 deformable contact consistent tangent, 마찰 consistent tangent 결과는 재검토 대상.
- `.gitignore`의 `output/`이 `dispsolver/output/` 패키지를 숨김(`fdda733`).
- reader: section ELSET을 못 찾으면 기본 재료 E=4000으로 진행, part 간 같은 ELSET 이름 충돌, 3D `*TIE` 미지원(원시 객체 → AttributeError). 미수정.

## 3. GEMINI.md 반영 제안 문안 (사용자 관리 문서이므로 직접 수정하지 않음)

L207-213("컨택 정식화 간 유일 물리 정해 일치성") 대체안:

> - **컨택 정식화 간 극한 비교 (2026-10-04 재측정, `dev_log/deck06_corrections_20261004.md`)**:
>   - 이전 "2.4% 일치 = 유일 물리 정해" 결론은 철회. 메시 오차(약 5%)보다 작은 차이였고 당시 NITSCHE는 penalty였다.
>   - 수정 후 Step 1 반력: NITSCHE(진짜, θ=0) x1/x10/x100 = −6,584.6 / −6,569.0 / −6,567.8 N(γ 무관에 가까움), PENALTY x10..x300 → 극한 ≈ −6,668 N. NTS/STS 이산화 차이 1.5%.
>   - 접촉 정식화의 정확성은 정해가 있는 접촉 patch test로 판정한다(`tests/test_deck06_contact.py`).

L222("대칭 Nitsche … 가장 강건한 표준 해법") 대체안:

> - 미소 변형/정적 볼트 체결: Nitsche(θ=0, 비대칭 접선, γ₀=50) 또는 deformable PENALTY(강성 x10 이상). DUAL_MORTAR는 deck06에서 미수렴(조사 중).
