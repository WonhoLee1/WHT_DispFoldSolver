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
- 3D 솔리드 음체적 가드 및 PARDISO 비대칭 LU 검증 (2026-10-01):
  - `tests/test_3d_negative_volume_guard.py` 15/15 통과.
  - `benchmark_element/benchmark_3d_mechanics.py`: 11개 3D 요소 회귀 무결성 검증 완료.

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
- [!] 3D solver 수락 기준 엄밀화 재설계 (2026-10-04 감사 지적):
  - 조치 이력: `rel_r < 0.50`, `r_norm < 1000.0` 조기 탈출 제거.
  - 잔존 결함 (감사 적발):
    1) correction 분기가 여전히 10 N 또는 1% 잔차를 허용함.
    2) fallthrough 분기는 2번째 반복의 스파이크값($r_0 \approx 2 \times 10^5\,\text{N}$) 기준으로 수백 N의 잔차를 통과시킴.
    3) $r_{\text{norm}} \le 1.0\,\text{N}$, $10^{-4}\,\text{mm}$ 등 절대값 공차가 특정 단위계에 종속됨.
    4) Line Search에서 `min_r_norm < 1e5`로 인해 잔차 증가 스텝을 무조건 수락하는 바이패스 존재.
  - 향후 계획: 특성 하중($q_{\text{avg}}$) 기반 무차원화 수렴 기준 및 엄격한 Armijo 강제.
- [!] 접촉면 초기 법선 위상학적 외향 법선(Outward Normal) 재설계 (2026-10-04 감사 지적):
  - 조치 이력: 동일 평면($z=1.5\,\text{mm}$)에서 `if abs(gap0) < 1e-5: sign = 1.0`로 35노드 정상화.
  - 잔존 결함 (감사 적발):
    - 초기 겹침이 10 nm를 초과하는 억지끼움(Interference fit) 모델에서는 여전히 법선이 내부로 뒤집혀 분리 상태로 오판되는 땜질식 처방임.
  - 향후 계획: 간극 부호가 아닌 C3D8 요소 면(Face)의 절점 위상(CCW) 기준 외향 법선으로 근본 전환.
- [!] SDI 활성 집합 동결 및 실행 경로 일원화 재설계 (2026-10-04 감사 지적):
  - 조치 이력: `dynamic3d.py:1804` `solve_step`에 `recent_active_sets[-2:]` 적용.
  - 잔존 결함 (감사 적발):
    1) 실제 `.inp` 실행 경로인 `_solve_step_with_hooks`(L2546)에는 수정이 누락되어 여전히 전체 과거 이력을 합집합함 (실행 경로 이중화 결함).
    2) 동결된 노드가 양방향 스프링이 되어 비물리적 인장력(접착)을 허용하고, 미동결 침투 노드를 묵인하며, 수렴 선언 전 접촉 상태를 재검증하지 않음.
  - 향후 계획: `_solve_step_with_hooks` 단일화, 인장 발생 즉시 방출 및 수렴 전 접촉 상태 재평가 루프 확립.



