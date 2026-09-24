# 영역별 분석: Solver (living doc, 초판 2026-09-19)

본 문서는 단일 소스 `GEMINI.md`의 영역 인덱스에서 참조된다. 수치 주장은 실측 기반만 기재.

## 1. Newton 수락 구조 (2D/3D 공통)
- 5-way OR (AGENTS.md §2): 변위비·에너지·KKT·Abaqus식·절대수정 중 하나면 수락.
  3-Tier AND 시도는 dt 붕괴로 회귀됨 (§4.7) — 손대지 말 것.
- 수락 분기 3종: `residual` (rel_r<5e-3), `correction` (du<1e-4, 힘-blind),
  `fallthrough` (예산 소진 후판정). 개수 제한이 아니라 종류가 핵심.
- `_record_acceptance` (판정 불변, 측정 전용): 3D (§12) + 2D (2026-09-19 이식).
  `last_accept={branch, r_norm, iters}`. 롤업 러너에 `worst_r_accept_N`·`accept_branches` 배선済 (2D/3D).

## 2. 측정된 수락 잔류 스케일 (pure-moment rollup, 25스텝)
- 2D-Opt1: worst 28.6N, 전원 residual, M −0.4% → 잔류 크기는 단독 지표 아님.
- 2D-CPE6M: worst 93.4N, ~140° 이탈 후 M +34% → 구조(저주파 편향)가 문제.
  판정은 Global M 게이트(90.4% FAIL)가 담당. CPE6M행은 정확도 NON-CONVERGED 표기.
- 3D-SHELL: worst 20.3N = r_0의 0.5% (Abaqus식 스케일 정합).

## 3. SDI·직선탐색·dt
- SDI 활성집합 분리 예산 (3D): 접촉 해석 필수. 2D 롤업은 SDI 없음.
- Armijo 직선탐색: contact 개발 과정에서 첫시도 무조건 유효 버그 수정済.
- `AdaptiveDtController`: target_iters=8 기준 증감. ex12는 dt_max 고정으로 완주.

## 4. 회귀 상태
- `verification/run_all`: 13/14. bending_2pt FAIL은 stash A/B로 기존 실패 확인
  (verification은 dynamic2d 미사용 — 코드경로 무관).
- `tests/test_2d_elements.py`: 25통과.

## 5. Open (미해결)
- [ ] bending_2pt 손대지 않기로 합의済 (레거시 스택). 단 jax≡numba 6자리 동일은
  디스패치 지문 — 10분 타임박스 판별 제안 상태.
  - 2026-09-21 판별 완료: jax≡numba 7자리 완전일치(3.194262e+02, err 30.1716%
    쌍방동일). 디스패치 분기 아님 → formulation-level 실패 확정, 합의 유지.
- [ ] CPE6M 140° 방아쇠 (B-bar checkerboard/HG α 의심, 미입증).
  - 2026-09-21 입증 완료: 요소 무죄, 수렴판정 유죄. 근거 3점 —
    (1) CR 프레임 추출식은 강체회전에 정확, (2) B-bar 구적오차는 이산평형을
    깰 수 없으나 실측 모멘트 불균형 87%, (3) 허용오차 1000배 강화에도
    비트동일(25스텝 전원 residual 분기 수렴, worst_r 93.4N).
    즉 뉴턴 정체를 수렴으로 오인한 false-convergence acceptance.
    NON-CONVERGED 표기 유지 정당. 수정은 전모델 영향이라 미실시 —
    수렴판정 강화는 별도 배치로.
