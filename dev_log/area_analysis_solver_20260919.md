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
- [x] 3D solver 수락 기준 엄밀화 검토 (2026-10-03 해결 완료):
  - 현상: `rel_r < 0.50`, `r_norm < 1000.0` 등의 조기 탈출 휴리스틱으로 인해 뉴턴-라프슨이 초기 잔차 대비 절반 수준에서 조기 종료되어, 고강성 플랜지 내 미완화 탄성 굽힘 응력($\approx 86,400\,\text{N}$)이 동결되어 볼트 반력이 $-89,013\,\text{N}$으로 왜곡됨.
  - 조치: `dynamic3d.py`의 임의 이완 조건을 전면 제거하고 $r_{\text{norm}} \le 1.0\,\text{N}$ 또는 ($\text{rel\_r} \le 10^{-3}$ and $\text{rel\_f} \le 10^{-3}$) 기준을 강제함.
  - 결과: Deck #06 Step 1에서 $RF_{\text{bolt}} = -2,750.22\,\text{N}$ vs $RF_{\text{gask}} = +2,747.34\,\text{N}$ (오차 < 0.10%, 99.89% 정적 평형) 완벽 달성.
- [x] 접촉면 0-간극 머신 엡실론 부호 반전으로 인한 반력 반토막 왜곡 해결 (2026-10-04 해결 완료):
  - 현상: 초기 형상에서 마스터/슬레이브 접촉면이 일치($z = 1.500000\,\text{mm}$)할 때, 부동소수점 오차($-1.38 \times 10^{-17}$)로 인해 삼항 연산자 `gap0 >= 0`가 False가 되어 접촉 법선 벡터 부호가 요소 내부로 역전. 35개 접촉 절점 중 10개가 분리(Separated)로 오인되어 반력이 $-2,750\,\text{N}$으로 왜곡됨.
  - 조치: `surface_contact3d_deformable.py` 및 `nitsche_contact3d.py`에 `if abs(gap0) < 1e-5: sign = 1.0` 공차 가드 배치.
  - 결과: 35개 접촉 절점 100% 정상 활성화, PENALTY (10x) $-6,569.86\,\text{N}$ vs NITSCHE (1x) $-6,412.48\,\text{N}$ (오차 2.4%)로 물리적 유일 정해로 상호 일치 달성.
- [x] SDI 활성 집합 동결 시 과거 이력 무차별 합집합 버그 해결 (2026-10-04 해결 완료):
  - 현상: Severe Discontinuity Iteration (SDI) 발생 시 `dynamic3d.py`가 과거 전체 이터레이션의 활성 절점을 무차별 합집합(`set().union(*recent_active_sets)`)하여 이미 분리된 비접촉 노드 20개까지 총 29개가 동결되어 잔차가 $1,360\,\text{N}$에서 정체(Stall).
  - 조치: 동결 범위를 최근 2회 진동 노드(`recent_active_sets[-2:]`) 및 현재 침투 노드로 제한.


