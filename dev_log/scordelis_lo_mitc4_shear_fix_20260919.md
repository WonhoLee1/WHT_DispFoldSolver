# Scordelis-Lo MITC4+ 전단 스케일 수정 완료 보고 (2026-09-19)

## 1. 증상
`benchmark_element/benchmark_shell_mitc.py` 실행 시 MITC4+ 오차 63% (8x8 −0.111, 16x16 −0.113),
정제해도 flat. 이론해 w_ref = −0.3024 대비 2.7배 뻣뻣.

## 2. 진단 과정 (제외된 용의자)
- 하중·BC·형상·단위: 정석대로 (원통 R=25, L=50, t=0.25, q=90, 1/4 대칭, 다이어프램). 총하중 3.93e4 일치.
- 솔버: PARDISO vs SciPy 대조, 자유잔류 1e-8 — 평형 성립. 풀이 자체는 정확.
- 드릴링 패널티: 0으로 꺼도 −0.1118 (0.6% 변화) — 무죄.
- 막·굽힘 요소 에너지: 해석해 대비 1.0000 — 무죄.

## 3. 참원인: 전단 B 행렬 관례 혼합 (dispsolver/element_shell/mitc4_plus_numba.py)
`_mitc4_shear_B_at_tying_point`에서 w 항은 공학전단(계수 1),
회전 항은 텐서 관례(계수 0.5·J·N)로 혼합. D_s = kappa·G·t는 공학전단 기준이라
회전 채널 전단 에너지가 정확히 1/4로 축소 → MITC 구속이 어긋나 조립 굽힘 응답이
메쉬 무관하게 정확히 4배 뻣뻣해짐 (평판 단순지지 수렴값이 이론값의 0.2504로 flat).

결정적 증거:
- 단일요소 회전-전단 에너지비 0.2500 → 수정 후 1.0000
- 평판 수렴비 0.2436/0.2490/0.2504 → 수정 후 0.9738/0.9948/0.9997

## 4. 수정 (4줄 + 독스트링 정정)
회전 항 `0.5*J*_t*N_t[i]` → `J*_t*N_t[i]` (행 0/1, θx/θy).
재발 방지 주석 포함: D_s가 공학전단 기준임을 명시 (Bathe 표기 1/2을 그대로 옮기면 재발).

## 5. 검증 결과
| 케이스 | 수정 전 | 수정 후 |
|---|---|---|
| Scordelis-Lo MITC4+ 8x8 | 63.27% FAIL | 2.51% PASS |
| Scordelis-Lo MITC4+ 16x16 | 62.72% FAIL | **0.28% PASS** (목표 <2%) |
| 고윳값 6RBM | PASS | PASS 유지 |

## 6. MITC3+ 잔류 (84~91% FAIL) — 수정 보류 권장
`mitc3_plus_numba.py`는 진짜 MITC3(Lee-Bathe edge tying)이 아니라
raw Mindlin Tri3 전부적분 (전단 B 회전항 = 무차원 L[i], tying 없음) — 전단 잠김 체질.
8x8 −0.026 → 16x16 −0.048로 수렴 추세는 있으나 정식화 교체급 작업 필요.
디스플레이 해석은 SOLID_SHELL_CR로 커버되므로 진짜 쉘이 필요할 때까지 보류 권장.

## 7. 산출물 상태
- 수정 파일: `dispsolver/element_shell/mitc4_plus_numba.py` (untracked — 커밋 대기)
- 관련 untracked: `dispsolver/element_shell/` 전체, `dispsolver/solver_shell/`,
  `benchmark_element/benchmark_shell_mitc.py`, `benchmark_element/eigenvalue_check.py`
- 진단 스크래치: `scratch/probe_shell_{shear,triage,sing,solver,conv}.py`
- 계획서 원본: `~/.gemini/antigravity-cli/brain/d68310e6-.../shell_elements_completion_plan.md`
  (Component 1·2는 계획서 작성 후 이미 실행된 상태였음 — 본 보고가 실제 검증 완료 기록)
