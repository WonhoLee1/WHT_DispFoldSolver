# Deck #06 접촉/솔버 수정 계획 (2026-10-04)

대상: `benchmark_inp/phase2_methodology/06_bolt_preload_gasket_flange_3d.inp`
배경: Gemini 세션(`dev_log/session_20261004_handoff.md`)이 "버그 박멸"을 선언했으나
독립 재검증에서 반박됨. 측정 근거는 본 문서 §1, 스크립트는 §8.

---

## 0. 원칙

1. **정해 기준값을 먼저 고정하고, 모든 수정은 그 기준값 대비로만 판정한다.**
   수렴 여부(`ok=True`)·`.sta`·"평형 오차"는 증거가 아니다 (AGENTS.md §4.9).
2. **수렴 판정은 "freeze 해제 + 실제 단방향 접촉 법칙"으로 재조립한 상태에서 한다.**
   솔버 내부 상태(freeze, bilateral spring)로 계산한 residual은 판정 근거가 아니다.
3. **silent fallback 금지.** 지원 안 되는 조합은 경고가 아니라 명시적 오류.
4. **Newton 루프는 하나의 판정 로직만 갖는다.** `solve_step` / `_solve_step_with_hooks` /
   `solve_step_augmented`의 중복이 이번 "수정이 실행 경로에 안 들어간" 사고의 원인.
5. AGENTS.md §4.7 경고 준수: 수렴 기준 강화는 dt 붕괴를 부를 수 있음 →
   기준 변경마다 Hertz roller 등 기존 접촉 deck 회귀를 함께 돌린다.

---

## 1. 현재 측정값 (수정 전 baseline)

Step 1 (`U_z=-0.060`) `RF_bolt` [N], freeze 해제 재조립 기준:

| 방법 | 1x | 10x | 100x | 1000x |
|---|---|---|---|---|
| NITSCHE | -6,412.48 | -6,548.32 | -6,565.53 | -6,567.44 |
| PENALTY | -5,623.95 | -6,562.18 | 실패 | 실패 |
| DUAL_MORTAR | 실패 (6회 increment 재시도, r≈1e3 N 정체) | | | |

- **기준값 R\* ≈ -6,567 N** (NITSCHE γ→∞ 극한). 독립 확인 필요(§P0-3).
- Step 2 `f_ext`: 합력 `(0, 0, -5.00) N`, t=0.5/1.0 동일 → 압력 미적용·ramp 없음.
  기대값: 1/4 모델 `Σf_x = Σf_y = p·R·H = 5·15·6.5 = 487.5 N`, `Σf_z = 0`.

---

## 1b. P0 결과 (2026-10-04) — §1의 "R\* ≈ -6,567"은 철회

**단일 정해 R\*는 이 deck에서 아직 정할 수 없다.** 측정:

| 메시 | NITSCHE 극한 (x1000) | PENALTY x10 / x20 / x30 | PENALTY 외삽 극한 |
|---|---|---|---|
| m=1 (원본, 비정합) | -6,567.44 | -6,562 / -6,622 / -6,636 | ≈ -6,670 |
| m=2 (세분화) | -6,239.03 | -6,270 / -6,305 / 실패 | ≈ -6,339 |
| 정합 메시 (r 절점 일치) | -6,565.38 | -6,557 / -6,620 / 실패 | ≈ -6,682 |
| TIE 접합 (상한) | | | -8,695 |

- 메시 오차 ≈ 5% (m=1→2). Gemini의 "2.4% 일치 = 물리 정해"는 메시 오차보다 작은 차이를 논한 것.
- 방법 간 극한 차이 1.6%가 세분화·정합 메시에서도 줄지 않음 → 이산화 차이로 설명 안 됨.
  그러나 PENALTY x20 이상은 수용 판정 결함(아래)의 영향권이라 어느 쪽이 맞는지 **P3 이후 재측정**.

**접촉 patch test (정해 존재)** — 적층 2블록, 무마찰, F = |δ|/(h/E₁+h/E₂):

| | x1 | x10 | x100 | x1000 |
|---|---|---|---|---|
| PENALTY | -2.7% | -0.28% | -0.025% | +0.28% (r=13 N, 거짓 수렴) |
| NITSCHE | **-16.6%** | -2.0% | -0.22% | -0.045% |

- 두 방법 모두 극한에서 정해로 수렴 → 정식화의 극한은 일관됨.
- **NITSCHE 오차 ∝ 1/γ = penalty처럼 동작.** consistent Nitsche는 γ와 무관하게 patch test 통과해야 함 →
  consistency 항 비활성 확증 수준 (P7 우선순위 상향).

**거짓 수렴 실증**:
- deck06 PENALTY x50: `ok=True`, **fallthrough 분기**에서 80 iteration 후 r=90 N 수용, RF -6,957 (극한 초과).
  freeze 해제 residual도 동일 90 N → bilateral 인장이 아니라 fallthrough의 부풀려진 `r_0` 기준이 원인.
- patch test를 E≈1e3으로 돌리면 하중 0.2 N 대비 residual 0.02 N(10%)이 `r_norm < 1.0` 절대 공차로 수용됨.

**P0에서 새로 발견한 결함 (deck06 밖, 같은 silent-fallback 계열)**:
- R1: `*SOLID SECTION`의 ELSET을 못 찾으면 경고만 내고 **기본 재료 E=4000, ν=0.3**로 해석 진행 (`strict=False`).
- R2: 서로 다른 part가 같은 part-local ELSET 이름을 쓰면 section 재료가 덮어써짐 (instance 범위 분리 안 됨).
- R3: 3D `.inp` 경로의 `*TIE`가 원시 `Tie` 객체로 constraint 목록에 들어가 `AttributeError: 'Tie' object has no attribute 'assemble'`.

**P0 산출물**:
- `tests/test_deck06_contact.py`: deck06 방법별 극한 게이트, freeze 해제 평형 게이트,
  **접촉 patch test 정해 게이트**, Step 2 하중 벡터 게이트. xfail은 측정값 주석 포함.
- `DynamicSolver3D._record_acceptance`: `r_norm_unfrozen`, `frozen` 기록 (측정 전용, 판정 불변).

**A1 수정**: 단일 R\* 대신 (a) patch test 정해 0.5% 이내, (b) deck06은 방법별 자체 극한 대비
(PENALTY 2%, NITSCHE 1%), (c) 방법 간 극한 일치는 P3 후 재측정해서 기준 확정.

---

## 2. 수용 기준 (Acceptance)

| ID | 기준 |
|---|---|
| A1 | PENALTY(기본 설정), NITSCHE(기본 γ), DUAL_MORTAR 모두 `|RF - R*|/|R*| ≤ 1%` |
| A2 | 수용 시점 상태를 freeze 해제·단방향 법칙으로 재조립: `‖r_free‖ ≤ 1e-3·q_ref` |
| A3 | 수용 상태에 인장 접촉력 0, 비활성인데 침투한 노드 0 (공차 `1e-3·L_elem`) |
| A4 | PENALTY 10x / 100x / 1000x 모두 수렴, A1 만족 |
| A5 | Step 2 `f_ext` 합력이 해석값과 1e-6 상대오차 이내, 시간에 비례해 ramp |
| A6 | `.sta`가 실제 increment / attempt / SDI / iteration 수를 기록 |
| A7 | 회귀: `python -m verification.run_all` 9/9, `pytest tests/` 실패 집합이 baseline과 동일 (A/B), Hertz roller deck 완주 |

---

## 3. 작업 단계

### P0. 안전망 먼저 (코드 수정 전)

1. **Baseline 스냅샷.** 현재 미커밋 변경(Gemini 작업)을 별도 커밋으로 보존 →
   이후 수정이 diff로 분리되게 함. (결정 D4)
2. **회귀 테스트 추가** `tests/test_deck06_contact.py` (`@pytest.mark.slow`):
   - A1/A2/A3 측정 함수 = `scratchpad/verify06.py` 로직 이식.
   - 현재는 xfail로 시작 → 수정되면 xpass 감지 (`tests/element_contract`의 `_XFAIL` 관례).
3. **R\* 독립 확인**: (a) 접촉 대신 `*TIE`로 묶은 모델(상한, 인장 허용),
   (b) 메시 1단계 세분화 NITSCHE 1000x. 둘 다 R\*과 대조. Abaqus 결과가 있으면 그것이 우선.
4. `_record_acceptance`에 **freeze 해제 재조립 residual**과 접촉 일관성(A3) 측정 추가.
   측정만, 판정에는 아직 미사용.

### P1. Newton 루프 판정 로직 단일화 — `dispsolver/solver3d/dynamic3d.py`

- SDI/oscillation 감지, freeze 갱신, 수렴 판정, line search 수용 규칙을
  private helper로 추출해 세 루프가 공유:
  `_sdi_update(...)`, `_freeze_update(...)`, `_check_converged(...)`, `_line_search(...)`.
- 이 단계는 **동작 불변 리팩터**. 검증: deck06 / Hertz 결과가 리팩터 전과 bit 단위로 동일.
- 리팩터 직후 현재 `.inp` 경로(`_solve_step_with_hooks`, L2546)의 전체 union 동결이
  `solve_step`(L1804)과 같은 규칙을 쓰게 됨 → 핸드오프 2.2 주장이 실제 경로에 처음 적용됨.

### P2. Freeze(chatter 억제) 의미 수정

현재 결함:
- freeze된 노드 = bilateral spring → 인장(접착) 허용.
- freeze에 없는 노드 = 침투해도 힘 0 (`surface_contact3d_deformable.py`의 `continue`).
- freeze 상태로 수렴하면 그대로 수용, 해제 후 재확인 없음.

수정:
1. freeze 비대상 노드는 **정상 단방향 법칙 유지** (`continue` 제거).
2. freeze는 "상태 라벨 고정"으로만 사용. 수렴 직전 **freeze 해제 → 재조립 → A2/A3 확인**.
   위반 시: freeze 해제하고 iteration 계속, 예산 소진 시 cutback.
3. freeze 대상 집합 = 최근 2개 상태의 **교집합 + 현재 침투 노드** (합집합 대신 검토;
   Hertz deck으로 비교 측정 후 결정).
4. 적용 대상: deformable penalty, mortar 둘 다. Nitsche는 `frozen_active_nodes`를
   무시하므로 solver가 속성을 쓰지 않도록 `get_active_set`/freeze 지원 여부를 명시 플래그로.

### P3. 수렴·line search 규칙 정상화

1. `min_r_norm < 1e5`이면 아무 step이나 수용 → **삭제**. 비하강 step 수용은 SDI iteration에 한해,
   최대 증가율 제한(예: 2.5x) 하에서만.
2. `r_0_norm` = iteration 중 최대 residual(2번째 iteration의 Dirichlet 튐 2e5 N) →
   **Abaqus식 평균 force flux `q̃`** (활성 DOF의 |f_int| 평균) 기반 상대 기준으로 교체.
3. 절대 공차(`r<1.0 N`, `r<10 N`, `du<1e-4 mm`, `du_limit 0.05/0.10 mm`) →
   `q̃`, 대표 요소 길이로 무차원화.
4. fallthrough(예산 소진) 분기도 동일 `_check_converged` 사용 (현재 옛 `rel_r<5e-3`).
5. 매 변경마다 A7 회귀 + AGENTS §4.7 dt 붕괴 여부 확인.

### P4. Step 2 DSLOAD — `_compute_step_external_loads`

현재: 압력 면적분은 20-node hex(`_HEX20_FACES`)만. C3D8은
"크기 ÷ 노드 수, −z 방향" silent fallback. `*SURFACE` 이름 region 미해석. STATIC ramp 없음.

수정:
1. C3D8 Quad4 면 압력 적분 (2×2 Gauss, `p·n·dA`), 면 정의는
   `interaction._HEX_FACE_LOCAL_NODES`(바깥 법선 순서) 재사용.
2. region이 element-based `*SURFACE`이면 (요소, 면) 목록으로 해석.
3. 압력에도 STATIC 기본 ramp 적용 (`get_force_vector` 분기와 동일 규칙).
4. 해석 불가 region/요소형 → **명시적 오류**. −z 분배 fallback 삭제.
5. NLGEOM 하 follower(현재 형상 기준) 압력: 1차는 기준 형상, follower + 하중 강성은 후속 항목으로 명기.
6. 단위 테스트: 1/4 원통 내압 합력 = `p·R·H` (x, y), 0 (z).

### P5. `.sta` 실기록 — `dispsolver/cli/run_inp.py:217`, `dynamic3d.solve()`

- `solve()` 증분 루프가 attempt마다 `(inc, att, n_sdi, n_iter, t_total, t_step, dt, status)`를
  logger 콜백으로 보고. CLI의 step당 고정 1행 기록 삭제.
- 테스트: deck06 Step 1이 `*STATIC 0.25`에서 최소 4 increment로 기록되는지.

### P6. 초기 간극 부호 — `surface_contact3d_deformable.py`, `nitsche_contact3d.py`

현재: `|gap0|<1e-5`면 sign=+1, 아니면 gap 부호로 법선 뒤집기.
→ 초기 겹침 > 10 nm(억지끼움)이면 법선이 뒤집혀 분리로 처리되는 원래 결함 유지.

수정: 법선 방향을 **gap 부호가 아니라 면의 위상**으로 결정.
면 법선이 자기 요소 중심 쪽을 향하면 반전 (면 순서가 바깥 법선이면 항상 +1).
테스트: 0.01 mm 초기 overclosure 모델에서 접촉력이 압축으로 발생.

### P7. 기본 침투 강성·Nitsche γ

- PENALTY 기본(1x)은 14% 과소. Abaqus 기본은 대표 요소 강성의 10배. (결정 D1)
- NITSCHE: 침투가 γ에 반비례(1.57→0.003 µm), RF가 γ에 2.4% 의존 →
  consistent Nitsche라면 γ 둔감해야 함. **consistency 항 누락/비활성 가설** 조사.
  `nitsche_contact3d.py`의 consistency term을 단순 2-block patch(해석해 있음)로 FD 검증.

### P8. DUAL_MORTAR 비수렴

- 증상: SDI 후 `rel_f≈0.99` 유지 = 접촉력이 내부력을 거의 상쇄 못 함.
- 조사 후보: (a) fast topology update(`topology_update_tol=1e-3`)가 투영을 고정해 접촉력 오류,
  (b) `D_f` positivity clamp 삭제(주석의 "positive-preserving dual basis"는 미구현),
  (c) freeze와 signed pressure 상호작용.
- 시간 제한 조사. 제한 내 해결 안 되면 small-sliding 문제에서 DUAL_MORTAR를
  명시적 미지원(오류)으로 표기. (결정 D3)

### P9. 문서 정정

`GEMINI.md` Lesson 12, `release_notes_contact.md` §14,
`dev_log/lessons_learned_contact_harmonization_20261004.md`, `session_20261004_handoff.md`:
- "2.4% 차이 = 동일 정해" 논리 오류 정정, R\*와 sweep 표로 대체.
- deck 경로(`benchmarks/abaqus_standard_inp/` → `benchmark_inp/phase2_methodology/`), 가스켓 재료
  (초탄성 EPDM 아님, 선형 `*ELASTIC E=1000, ν=0.38`) 정정.
- "엄격한 수렴 기준", "Abaqus Chattering Freeze standard" 표현 정정.

### 부수 항목

- `tangential_stiffness_ratio` 기본값 1.0 → 0.1이 섞여 들어옴 (마찰 deck 영향). (결정 D2)
- `finite_sliding_mortar3d.py.bak_20261004` 삭제.
- `contact_stiffness.py` 초탄성 branch: `d1` 기본값 0.01 등 임의값 → 근거 확인 또는 오류.

---

## 4. 순서와 검증 게이트

```
P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9
     (동작불변)  (핵심 정확성)    (Step2)  (.sta)
```

각 단계 종료 게이트:
1. deck06 Step 1 sweep (`sweep06.py`) + A2/A3 측정
2. Hertz roller deck 완주 여부
3. `python -m verification.run_all`
4. `pytest tests/` 실패 집합 A/B (`git stash`)

---

## 5. 위험

| 위험 | 대응 |
|---|---|
| 수렴 기준 강화 → dt 붕괴 (§4.7 재현) | P3 변경을 항목별로 분리, 각각 Hertz/deck06 측정 |
| freeze 수정 → Hertz roller chatter 재발 | P2-3 교집합/합집합 비교 측정 후 결정 |
| R\* 자체 오류 | P0-3 TIE 상한·메시 세분화·(가능 시) Abaqus 대조 |
| 리팩터(P1) 중 동작 변경 | bit 단위 동일성 확인 전 다음 단계 진입 금지 |

---

## 6. 결정 (사용자, 2026-10-04 확정)

- **D1** PENALTY 기본 강성 → **Abaqus식 10x**. P7에서 적용, 기존 접촉 deck 회귀 확인.
- **D2** `tangential_stiffness_ratio` 기본값 → **0.1 유지**. 단 Hertz roller 등 마찰 deck에서
  1.0 대비 영향 측정해 release note에 기록.
- **D3** DUAL_MORTAR → **시간 제한 조사 후 판단**. 미해결 시 small-sliding에서 명시적 미지원(오류).
- **D4** Gemini 미커밋 변경 → **스냅샷 커밋 후 진행** (P0-1).

---

## 8. 측정 스크립트 (현재 scratchpad, P0에서 tests/로 이식)

- `verify06.py` — freeze 해제 재조립 residual, RF, 침투 노드
- `sweep06.py` — 방법 × 강성 배율 sweep
- `step2_06.py`, `fext06.py` — Step 2 하중 벡터 검사

---

## 9. 진행 상태 (2026-10-04 갱신)

| 단계 | 상태 | 커밋 | 비고 |
|---|---|---|---|
| P0 | 완료 | `ba139af` | 회귀 게이트, 접촉 patch test, freeze 해제 수용 probe |
| (저장소) | 완료 | `fdda733` | `.gitignore`가 숨긴 `dispsolver/output/` 커밋 |
| P1 | 완료 | `8f38bca` | Newton 루프 판정 단일화, 404회 호출 비트 단위 동일 |
| P3 | 완료 | `4bc3eb9` | 상대 수렴 기준, `<1e5` 우회 삭제, 진동 오판·freeze 법칙 교체 수정(soft exp) |
| P2 | 완료 | `40d09f6` | freeze 상태 수용 금지 게이트, freeze 범위 통일 |
| P4 | 완료 | `9064a43` | C3D8 면 압력 적분, ramp, −z fallback 제거 |
| P5 | 완료 | `6db3500` | `.sta` 실제 증분·시도·SDI 기록 |
| P6 | 완료 | `911d5e0` | 법선을 master 요소 바깥 방향으로(억지끼움), roundoff floor `|X+u|` |
| P7 | 완료 | `3686fd0` | **Pardiso 대칭 판정 버그(np.bool_)**: 모든 비대칭 접선이 대칭화되어 풀렸음. NITSCHE를 진짜 Nitsche(θ=0)로 |
| P8 | 부분 | (이번 커밋) | Mortar 법선 방향 결함 수정(평평한 접촉면에서 반올림 잡음으로 방향 결정 → 침투 0으로 인식). **deck06 미수렴은 미해결** |
| P9 | 대기 | | 문서 정정 |

### P8 미해결 내용 (시간 제한 도달)
- 법선 수정 후 Mortar는 penalty 해 상태에서 35노드 활성, 접촉력 5,634 N을 정상 감지.
- 그러나 Newton이 r ≈ 2e4 (r/q ≈ 10~17), 활성 30노드에서 정체.
- 접선 대 유한차분 상대오차 6e-4 (중앙값 1.5e-4) — 부정확하나 이것만으로 정체를 설명하기 어려움.
- 조립 1회 3.3 s (Python 루프, triplet 30만) — deck06 Step 1에 수십 분.
- 결정 D3 원안은 "small-sliding에서 명시적 미지원(오류)"이었으나, 이 deck은 sliding 형식을 지정하지 않았고 gear tooth(#05) 등 다른 deck이 같은 클래스를 쓰므로 오류 처리는 보류. 다음 조사 후보: (a) 접선 오차 원인(클리핑 다각형 변화의 미분 누락 추정), (b) 적분점 단위 활성 경계, (c) 조립 Numba화.

### P7 결과 요약
- deck06 Step 1: NITSCHE x1/x10/x100 = −6,584.6 / −6,569.0 / −6,567.8 N (γ 무관에 가까움, 일관 정식화).
  PENALTY x10..x300 = −6,543 .. −6,664, 극한 ≈ −6,668. 두 방법 간 1.5% 차이(NTS vs STS)는 남음.
- 접촉 patch test: Nitsche 기본값 오차 1.3e-4 (이전 −16.6%).
- PENALTY x100 정체(40분+)는 대칭 판정 버그가 원인이었고 해결(7초).
