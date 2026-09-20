# Abaqus 26 Verification 문서 분석 + WHT_DispFoldSolver 재료 물성 고도화 마스터 플랜

- 일자: 2026-09-19
- 성격: 패널 심층 분석 보고서 (6단계: 설계/조사/이론평가/계획/문제분석/해결방안)
- 근거: 코드베이스 직접 확인 — `material_dispatch_3d` 시그니처(numba_materials.py:543), `response_interface.py` 정준 계약, `viscoelastic.py` WLF, SDV 레이아웃(7/8/12/42)

---

## 0. 문서 트리아지 (26 → 17 in-scope)

| 그룹 | 문서 | 내용 | 우선순위 |
|---|---|---|---|
| **P0 인프라** | 5, 6, 22 | 온도 의존 탄성, 필드변수 탄성, 열팽창 — 필드 파이프라인 자체가 전제조건 | ★★★ (사용자 필수 지정) |
| **P0 인프라** | 13, 14 | 온도/필드 의존 소성 | ★★★ (사용자 필수 지정) |
| **P1 소성 확장** | 9, 11, 15, 23 | Mises 등방/이동 경화 완성, Cowper-Symonds/Johnson-Cook 속도 의존, JC 파손, return-map 수치보정 | ★★★ |
| **P2 고무 비탄성** | 3, 4, 8 | Mullins, Hysteretic(Bergström-Boyce 계열), 비선형 점탄성(변형률 의존 완화) | ★★ (PSA 사이클링 수명에 필수) |
| **P3 파손/기타** | 16, 12, 17, 1 | Ductile/Shear damage, annealing, creep, orthotropic 탄성 | ★★ |
| **벤치마크 전용** | 2, 7, 10 | 기존 구현(Prony M=6 + 초탄성 전제품군) 검증 시나리오 — 신규 물성 아님 | 검증 |
| **솔버측/기존** | 18, 21 | viscous stabilization(기존 `stabilization.py`), Rayleigh 감쇠(동적 모드 도입 시) | 후순위 |
| **제외** | 19, 20, 24, 25, 26 | 콘크리트 2종, 열전달 3종 | 사용자 지침대로 제외 |

**핵심 통찰**: 문서 5/6/13/14/22는 "물성 모델"이 아니라 **"물성 데이터가 온도·필드의 함수가 되는 인프라"**다. 개별 모델 10개 추가보다 필드 파이프라인 1개가 먼저다. 이 없으면 온도 의존 모델 전부 접속 불가.

---

## 1. System Design (설계)

### 1.1 현황 시그니처 (코드 확인됨)

```python
# 3D Numba 경로 (dispsolver/material3d/numba_materials.py:543)
material_dispatch_3d(mat_type, props, sdv_prev, E_voigt, F, detF, dt)
    -> (S_voigt, C_mat, sdv_new, error_flag)

# 2D JAX 경로 (dispsolver/material/response_interface.py, 정준 계약)
response_jax(F_2d, state, dt, params) -> (S_voigt, state_new)
tangent_response_jax(...)             -> (S_voigt, C_voigt, state_new)  # optional
```

설계가 고쳐야 할 결함 3가지:
1. **온도/필드 인자 없음.** WLF는 `viscoelastic.py`에 존재하나 `temperature=20.0` 고정 스칼라 — 어떤 필드에도 미접속.
2. **SDV 레이아웃 매직넘버.** `sdv_prev[6]`, `sdv[36+c]` 하드코딩 산재 — 신규 모델 추가 시 충돌 위험.
3. **props 위치기반 + 휴리스틱 디코딩.** NeoHookean `if p1 <= 0.05` 분기 — 테이블 데이터(온도 의존 물성곡선)를 담을 구조 자체가 없음.

### 1.2 확장 시그니처 (v2 계약)

```python
# 3D
material_dispatch_3d_v2(
    mat_type, props, sdv_prev,
    E_voigt, F, detF,
    dt, ttime,               # ttime: 총 누적 시간 (creep/annealing용)
    temp, dtemp,             # GP 온도, 온도 증분 (스칼라)
    fields,                  # (NFV,) float64 — NFV=4 고정 (Abaqus 관례)
) -> (S_voigt, C_mat, sdv_new, error_flag)

# 2D — 시그니처 불변, params NamedTuple 확장
# J2Params(...) -> temp: float, fields: (4,) 필드 추가
```

원칙: **온도·필드는 미지수가 아닌 처방 데이터**. Level-1(비연성 열-기계)에서는 T가 DOF가
아니므로 접선 강성에 새 블록 없음 — 기존 K_uu 그대로 일관접선 유지. 완전 연성(Level-2,
K_Tu 커플링)은 열전달 문서(24–26)와 함께 후순위로 설계 보류.

### 1.3 온도 필드 공급 아키텍처

```
TemperatureField (신규: dispsolver/load/temperature_field.py)
├─ 모드 A: 균일 T(t)            — 스칼라 램프/스텝 (대부분의 벤치마크)
├─ 모드 B: 해석 함수 T(X, t)    — 두께 방향 구배 등
├─ 모드 C: 절점 배열 T[n_nodes] — 향후 연성 해석/실험 데이터 맵핑
└─ evaluate_gp(coords_gp, t) -> (temp, dtemp, fields)
```

솔버 접속: `DynamicSolver.set_temperature_field(tf)` → `_assemble_*` 계열이 GP 좌표로
평가된 `temp_gp[n_gp]`, `fields_gp[n_gp,4]` 배열을 배치 커널에 전달.
`_bc_base_vals` 처방 패턴과 동일 계열 — 데이터이지 미지수 아님.

### 1.4 SDV 레지스트리 (매직넘버 제거)

```python
# dispsolver/material3d/sdv_layout.py (신규) — single source of truth
SDV_LAYOUTS = {
    MAT_J2_PLASTICITY:         {"ep": (0,6), "eqps": (6,1)},                     # =7
    MAT_J2_PLASTICITY_TABULAR: {"ep": (0,6), "eqps": (6,1), "failed": (7,1)},    # =8
    MAT_HYPER_VISCOELASTIC:    {"h": (0,36), "S_dev_el": (36,6)},                # =42
    # ---- 신규 ----
    MAT_MULLINS:   {"wm_max": (0,1), "eta_prev": (1,1), "branch": (2,1)},        # =3
    MAT_HYSTERETIC:{"F_in": (0,9), "s_ev": (9,1)},                               # =10
    MAT_JC_DAMAGE: {"ep": (0,6), "eqps": (6,1), "D": (7,1),
                    "w_duc": (8,1), "w_shr": (9,1)},                             # =10
    MAT_CREEP:     {"ep_cr": (0,6), "eq_cr": (6,1)},                             # =7
}
```

Numba에는 오프셋 상수를 모듈 레벨 상수로 생성(코드젠 또는 수동 미러+일치 테스트).
`get_default_sdv_count`는 이 레지스트리에서 파생. 복합 모델(점탄+소성+손상)은
**합성 레이아웃**: `[base_block | plastic_block | damage_block]`, 각 블록 오프셋은
구성 시 Python에서 계산해 props 헤더로 전달.

### 1.5 props v2 — 자기기술(self-describing) 블록 레이아웃

```
props = [ n_tables(i4) |
          table_0: id(i4), n_axes(i4), n_pts_per_axis(4×i4),
                   axis_breakpoints(concat f8), values(f8, 행=조합) |
          ... |
          scalar_block: model별 스칼라 (f8 × k) ]
```

- 온도 의존 물성 = 테이블 id로 참조 (예: `TAB_E`, `TAB_SIGY`, `TAB_ALPHA`).
- **휴리스틱 디코딩 금지** — v2 모델은 오프셋을 Python `PropsLayout`이 생성, 커널은 상수 오프셋으로만 읽음. 기존 v1 모델은 전환 기간 동안 현행 유지(외과적 변경 원칙).
- 보간 커널: `interp_table_1d/2d` njit — 축 정렬 브레이크포인트 + 선형 보간 (Abaqus *DEPVAR 계열과 동일 규칙: 범위 밖은 끝값 고정(clamp), 외삽 금지).

### 1.6 열팽창 삽입 지점 — 다중적(multiplicative) 열 분해

```
F = F_mech · F_th,   F_th = γ(T) · R_orth (등방: γ=exp(∫α dT)의 대각)
```

- **단일 삽입 지점**: 커널 진입부에서 `F_mech = F · F_th⁻¹` 계산 후 기존 구성법칙에 전달. 14개 초탄성/점탄/소성 모델 **수식 무수정 재사용**.
- 등방: `F_th = γI`, `γ = exp(ᾱ(T−T_ref))`, ᾱ는 ∫α(T)dT/ΔT (α가 T 의존이면 테이블 적분).
- Orthotropic(문서 1/22 결합): 재료축 대각 `diag(γ1,γ2,γ3)` → 요소 배향 회전 `R_orth`.
- 평면변형률 임베딩(F33=1 고정)에서 `F_mech,33 = 1/γ ≠ 1` → S33 반력 발생 = 두께 방향 구속 열응력. **물리적으로 정확** (평면변형률이 면외 열팽창을 막는 것이 실제 거동).
- J 분할: `J_mech = J / J_th` — 비압축성 모델(PSA ν=0.499)의 체적 분리는 J_mech 기준으로 수행. 열팽창이 체적 변화를 전량 흡수하므로 기계적 응답은 거의 등체적 유지 → **하이브리드 u-p 요소(CPE4H 계열)와 자연 정합**.

---

## 2. Technical Research (조사) — 문서별 정식화 요체

### A. 열/필드 의존 그룹 (5, 6, 13, 14, 22)

**문서 22 (simpleexpansion)** — `*EXPANSION`: ε_th = α(θ)(θ−θ_ref), 등방/직교이방/전 이방.
검증 시나리오: 단순 인봉 봉, ΔT 처방, ε = αΔT 정확 재현 + 구속 봉 열응력 σ = −EαΔT.
`.inp` 사양: `*EXPANSION, TYPE=ISOTROPIC` / α 테이블 `α, θ` 쌍.

**문서 5 (tempelastic)** — E(θ), ν(θ) 선형 보간 테이블. 검증: 여러 온도에서 인장 시험,
기울기 = E(θ_i) 대조. Abaqus 규칙: **보간은 테이블 축에서 선형, 온도는 GP에서 절점 평균**.

**문서 6 (fieldelastic)** — 동일 구조, 독립변수만 필드변수 f1..f4. Abaqus는 물성당 최대
4개 필드 + 온도 동시 의존 허용(`DEPENDENCIES=n`). 우리 설계도 동일: `(T, f1..f4)` 다축 테이블.

**문서 13 (tempplastic)** — σ_y(θ), 경화곡선 가족 σ_y(ε̄p, θ). 검증: 온도별 인장 → 항복점 대조.
핵심 수치 이슈: **return mapping 중 σ_y 평가 온도를 언제 froze하는가** — Abaqus는 증분
시작 온도(θ_n)로 평가(비연성 가정). 우리도 동일: 증분 내 T 상수.

**문서 14 (fieldplastic)** — 13과 동일 + 필드변수 축.

### B. 고무 비탄성 그룹 (3, 4, 8)

**문서 3 (Mullins)** — Abaqus *MULLINS EFFECT: 두 네트워크(탄성 + 손상).
에너지 W = η·W̄_dev(λ̄) + W_vol(J) + 손상항, η = 1 − (1/r)·erf⁻¹((W̄_max−W̄)/m) 형태의
연화 함수(Abaqus 공식: `η = 1 − (1/r) erf((W_max − W)/(m + β W_max))` 변형, r,m,β 3개 파라미터).
상태: `W̄_max` 이력 + 현재 분기(재하/하중제거) — **卸载 시 η=1 복귀(탄성)**, 재하 시에만 연화.
검증 시나리오: λ=1→λ_max→1→λ_max 2사이클, 2차 재하 곡선 = 1차(η 이력) + 영구연화 대조.

**문서 4 (hysteretic)** — Abaqus *HYSTERETIC (2018+, Bergström-Boyce 계열):
평행 네트워크 N개, 각 네트워크 = 초탄성 스케일 α_i·W̄(λ̄_ev) + 비선형 점성 진화.
진화: `γ̇_ev = τ·(λ̄−1)^p·|τ_ev|^q·sign(τ:deviatoric)` — 내부 변수는 비탄성 변형 F_in(또는 등가 진화 스칼라).
에너지 소산(히스테리시스 루프 면적)이 검증 대상: 사이클링 시 W_diss = ∮σ dε를 Abaqus 결과와 대조.
기존 Prony 점탄성과의 차이: **완화시간이 변형률 크기에 의존**(비선형) — 문서 8과 연결.

**문서 8 (nonlinviscohyper)** — 변형률 의존 완화: Prony g_i(ε̄) 또는 완화시간 τ_i(ε̄) 스케일링.
Abaqus 구현: `*VISCOELASTIC, TIME=PRONY` + 변형률 의존 승수 테이블, 또는 NONLINEAR option.
기존 M=6 Prony 커널에 **ε̄ 승수 테이블 1개** 추가가 최소 변경 — h_recurrence 구조 불변.

### C. 소성 확장 그룹 (9, 11, 15, 23, 12, 17)

**문서 9 (rateindepplast)** — 이미 대부분 존재(J2 등방/비선형/테이블). 갭: **이동 경화
(kinematic hardening)의 Numba 배치 경로 완성도** — 현재 SDV 12(백응력6+편응력6) 레이아웃은
있으나 계약 스위트 C10 통과 여부 미확인. Abaqus 이동경화 = 선형 Ziegler + 비선형(Chaboche):
`α̇ = C·(2/3)·ε̇p − γ·α·ε̇p` (Abaqus 비선형 형식: `α̇ = (C/σ⁰)·(σ−α)·ε̇p − γ·α·ε̇p`).

**문서 11 (ratedepplastexp)** — 두 오버스트레스 모델:
- Cowper-Symonds: `σ_y = σ_ref·(1 + (ε̇p/D)^(1/p))` — D,p 재료상수.
- Johnson-Cook: `σ_y = (A + B·ε̄p^n)·(1 + C·ln(ε̇p/ε̇0))·(1 − T*^m)`, `T* = (T−T_room)/(T_melt−T_room)`.
수치 이슈: 항복조건이 ε̇p의 비선형 함수 → **return mapping이 국소 비선형 방정식**화.
해법: ε̄p에 대한 1D Newton(또는 radial-return의 폐형이 존재하는 CS는 `Δp` 스칼라 Newton 3–5회).
검증: 홉킨슨 바 계열 시나리오 대신, 단일 GP에서 ε̇p 스윕 → σ_y(ε̇p) 곡선 대조.

**문서 15 (johnsoncook)** — JC 소성 + JC 손상: `ε̄_D = (D1 + D2·exp(D3·η))·(1 + D4·ln ε̇*)·(1 + D5·T*)`,
η = 삼축응력도(p/q). 손상 진화: 에너지 기반 또는 변위 기반, 손상 후 `σ → (1−D)·σ` + 접선 축소.
열연성 파손(adiabatic heating) 옵션은 제외 가능(준정적 폴더 시나리오와 무관).

**문서 23 (plasticity-corrections)** — 알고리즘 검증 문서. 우리 대응물: 계약 C8(FD Jacobian
대조)이 이미 이 역할. 추가할 것: **cutback 후 상태 롤백 정확성** — Abaqus는 증분 재시도 시
STATEV를 증분 시작으로 복원. 우리 `state_new` 승격 경로(_ul_F_n 승격 규칙, F5)와 동일 원칙을
새 SDV 전체로 확장: **수렴 시에만 sdv_new 승격, cutback 시 sdv_prev 유지**.

**문서 12 (annealtemp)** — `*ANNEAL TEMPERATURE`: T ≥ T_anneal 이면 등가소성변형/경화 이력
부분·전부 소거(테이블로 소거율). 구현: return-map 직후 `if temp >= T_ann(softening_frac)`:
`eqps *= (1−frac); ep *= (1−frac)`. SDV 신규 불필요 — 기존 ep/eqps에 연산만 추가. **저비용 고효과**.

**문서 17 (creep)** — `*CREEP`: ε̇_cr = A·σ^n·t^m (time-hardening) 또는 strain-hardening.
준정적 폴더에서는 PSA의 장기 처짐(sag)만 관련 — 후순위. 구현: 점탄성 h_recurrence와 유사한
**연산자 분리(operator split)**: 탄성 예측 → 크립 보정(ε_cr 명시적 적분, 작으면 안정).

### D. 파손 그룹 (16)

**ductile**: `ω_d = ∫ dε̄p/ε̄_D(η, ε̇p)` — 누적 ≥1 시 개시. **shear**: `ω_s = ∫ dε̄p/ε̄_D(k_s)`,
k_s = 편각(θ_s, 정규화 3응력 편차 각). 개시 후 진화: 변위/에너지 기반 선형 연화 `D: 0→1`,
`σ_eff = (1−D)σ`. 요소 삭제는 아님(2D 폴더에서는 D=1 접선 축소까지만 — mesh deletion은 후순위).

### E. 벤치마크 전용 (2, 7, 10)

**문서 10 (viscocylinder)** — 두꺼운 점탄성 실린더 내압: **해석해 존재**(radial/hoop 응력
이완 곡선). 우리 평면변형률 2D 솔버에 정확 대응: 두꺼운 원통(내경 a, 외경 b), 내압 p₀
step → σ_θθ(t), u_r(t)를 Prony M항 해석해와 대조. **기존 M=6 구현의 가장 강한 검증** —
형상이 축대칭이라 Q4 평면변형 요소로 b/a 스윕 3점이면 충분. 최우선 벤치마크로 채택 권고.

---

## 3. Theoretical Evaluation (이론평가)

### 3.1 일관 접선(consistent tangent) 평가 — 모델별

| 모델 | 접선 난이도 | 판정 |
|---|---|---|
| 온도/필드 의존 탄성·소성 | **쉬움** — T는 데이터, ∂C/∂u = 0. 보간 테이블 미분 불필요 | 기존 구조 그대로 |
| 열팽창 (multiplicative) | **쉬움** — F_th 상수, ∂S/∂F_mech · ∂F_mech/∂F = γ⁻¹ 스케일 | 체인룰 1줄 |
| Mullins | 중간 — 재하 분기에서 ∂η/∂W̄ 항 추가. 분기 전환(η=1 ↔ η<1)은 **C⁰ 불연속** | 분기 감지 후 국소 Newton; 전환점 근처 cutback 허용 |
| Hysteretic (BB) | **어려움** — F_in 진화식 비선형, 접선 = 진화식의 선형화(국소 Newton + FD 또는 해석) | 국소 Newton 2–4회 + 수정 Newton 전역 (AGENTS §4.4 선례 준수: 전역 자동미분 금지) |
| Cowper-Symonds/JC 소성 | 중간 — Δp 스칼라 Newton, 접선 = `d(σ)/d(Δp)` 폐형 유도 가능 | 해석 접선 필수(수정 Newton 시 38° 벽 재발 위험) |
| Ductile/Shear damage | 중간 — 손상 접선 `∂((1−D)σ)/∂ε`에 ∂D/∂ε 항 | D 연화 구간에서 접선 비대칭 가능 → 대칭화 또는 line-search 허용 |
| Creep | 쉬움(명시적 적분 시) — 탄성 접선 재사용 | operator split, dt 제한만 관리 |

### 3.2 수치 안정성

- **J2 eigh NaN 문제(AGENTS §4.3)**: 신규 return-map(JC, 이동경화)도 spectral 분해 경유 시
  동일 위험. 대칙: 신규 커널은 **deviatoric radial-return 폐형**(eigh 불필요, 2D/평면변형에서는
  deviator가 3성분)으로 작성 — `plastic_jax.py`의 tanh-smoothed yield 선례 유지.
- **cutback 시 SDV 롤백(문서 23 대응)**: 모든 신규 상태변수는 `sdv_prev`에서 읽고 수렴
  승격 시에만 기록. F5 교훈(_ul_F_n 중복 합성)과 동일 패턴 — **승격 지점 단일화**.
- **테이블 보간 clamp**: 범위 밖 외삽은 물성 음수화 위험(E(T) < 0). 보간 후 `E>0, σ_y>0`
  가드 + error_flag 승격(요소 0강성화 대신 cutback 유발).
- **Mullins 분기 불연속**: W̄ = W̄_max 경계에서 η 미분 불연속. Abaqus도 동일 — Newton
  1–2회 추가 소요는 정상, 수렴 판정 기준(§2의 5-way OR) 변경 불요.

### 3.3 체적 비압축성과의 결합 (PSA ν=0.499, 폴딩 디스플레이 핵심)

1. **열팽창 × 비압축성**: F_th = γI는 순수 체적 변형. J_mech = J/γ³ 분리 시 deviatoric
   응답 불변 → 하이브리드 u-p(CPE4H/CPE4IH)와 **충돌 없음**. 단, 평면변형률에서 면외
   열응력 S33 = −λ_th 항이 압력 p와 커플링 — u-p 혼합식에서 p의 물리적 의미가 "기계적 압력"
   으로 유지되므로 오히려 더 깨끗함.
2. **Mullins × 비압축성**: Abaqus 형식 자체가 dev/vol 분리(η는 deviatoric 에만 적용,
   체적부는 손상 없음) → 기존 F̄ 기반 커널 구조와 정합.
3. **점탄성 × 온도**: WLF aT(T)가 dt_eff = dt/aT로 들어감. T가 공간적으로 변하면 GP별
   aT 다름 — 이미 배치 커널은 GP 루프이므로 배열 전달만으로 해결. 안정성: aT가 증분 내에서
   동결(T_n 기준)이면 h_recurrence는 기존 지수 적분 그대로 — **무조건부 안정 유지**.
4. **JC 소성 × 온도 연화**: T* → 1(용융 근접)에서 σ_y → 0, 접선 특이. 폴딩 디스플레이
   작동 온도(−20~60°C)에서는 T* ≪ 1 — 실질 위험 없음. 가드만.

### 3.4 프레임 공변성 (AGENTS §4.14 교훈)

신규 커널 전부 **구성 레저(`kinematics/frame.py`의 `gp_internal_force`) 경유 의무화** —
F4/B1 결함(워크 켤레 아닌 stress-B 수축) 재발 방지. 특히:
- F_mech = F·F_th⁻¹ 분해는 **REF0 기준**으로 수행(F_th는 초기构形 온도에 정의) — UL 경로에서
  F_n과 혼동 금지. UL에서는 F_mech,n = F_n·F_th,n⁻¹, F_inc는 기계적 부분만.
- 계약 C6/C7/C8(회전 기준 스윕, 등방성, FD Jacobian)을 **신규 mat_type마다 통과 의무화**.
  C8은 회전된 기준에서 — §4.15의 "부분 push-forward가 축퇴를 무효화" 교훈.

---

## 4. Strategic Planning (계획) — 로드맵

벤치마크 수행은 각 단계 구현+단위점검 완료 후(사용자 지침). 단계마다 `verification.run_all` +
계약 스위트(C1–C11) 게이트 통과 의무.

### Phase 0 — 필드 인프라 (문서 5, 6, 22, 13, 14 동시 해결)
1. `sdv_layout.py` + `PropsLayout`(props v2, 테이블 블록) — 기존 모델 무수정, 신규 전용.
2. `TemperatureField` + 솔버 접속(`set_temperature_field`, GP 배열 전달).
3. `material_dispatch_3d_v2`(temp/dtemp/fields/ttime 인자) + 2D params 확장.
4. 열팽창: multiplicative F_th 분해 커널 진입부 삽입 + 평면변형 S33 처리.
5. 온도 의존 E(T)/ν(T)/σ_y(T) 테이블 보간 njit 커널.
6. 검증: 문서 22/5/13 대응 `.inp` 벤치마크 4종(§6.3).

### Phase 1 — 속도 의존 소성 + JC (문서 9, 11, 15, 23)
1. 이동경화(Chaboche) 배치 경로 계약 C10 확인/완성.
2. Cowper-Symonds 오버스트레스: Δp 스칼라 Newton return-map + 해석 접선.
3. Johnson-Cook 소성(T*, ln ε̇*, ε̄p^n 3항 곱) — CS 커널의 일반화로 구현(공유 서브루틴).
4. JC 손상(ε̄_D(η,ε̇,T)) + 손상 진화 (1−D)σ.
5. 검증: GP 수준 σ_y(ε̇p) 스윕 + 문서 11/15 `.inp` 벤치마크.

### Phase 2 — 고무 비탄성 (문서 3, 4, 8)
1. Mullins(η 연화, W̄_max 이력) — 기존 초탄성 dev 부분 래퍼로 구현.
2. 비선형 점탄성: g_i 또는 τ_i에 ε̄ 승수 테이블(최소 변경).
3. Hysteretic(BB): 평행 네트워크 + F_in 진화 — 최고 난이도, 마지막.
4. 검증: 2사이클 재하(Mullins), 히스테리시스 루프 면적(Hysteretic), 문서 3/4/8 `.inp`.

### Phase 3 — 파손/크립/어닐링/이방성 (문서 16, 12, 17, 1)
1. Ductile/Shear damage (ω 누적 + 선형 연화).
2. Annealing (SDV 소거, 저비용).
3. Creep (time-hardening, operator split).
4. Orthotropic 탄성 (요소 배향 배열 `elem_orient` 도입).
5. 검증: 문서 16/12/17/1 `.inp` 벤치마크.

### Phase V — 기존 기능 검증 (문서 2, 7, 10, 병행 가능)
- **문서 10 viscous cylinder 해석해 대조**를 최우선 실행 — 기존 Prony M=6의 가장 강한 검증.
- 문서 2/7: 소변형 완화·크립 + 대변형 점탄已有 구현 회귀 확인.

---

## 5. Core Problem Analysis (문제분석) — 폴딩 디스플레이 관점

### 5.1 열팽창 미스매치 (문서 22가 필수인 이유)

14층 PET-PSA 스택: α_PET ≈ 15–20 ppm/K, α_PSA ≈ 100–300 ppm/K(고무계). ΔT = 40 K에서:
- 자유 수축差: Δε = (α_PSA − α_PET)·ΔT ≈ 0.3–1.1% 면내 변형률 불일치.
- 층간 구속 → **PSA에 전단 선행 변형 + PET에 축응력**. 폴더 온도 사이클(−20↔60°C)에서
  매회 누적 → 층간 박리(delamination) 개시 조건과 직결.
- 현재 솔버: α 개념 자체가 없음 → 저온 폴더 시나리오의 초기 응력 상태(라미네이션 잔류응력)
  를 표현 불가. **열팽창은 "추가 기능"이 아니라 다층 모델의 물리적 완전성 요건.**
- 절점 수 관점: 두께 0.5mm에 14층, Δα·ΔT 변형률 구배는 층 두께(30µm) 스케일 —
  기존 graded mesh(§4.12)로 해상 충분, 새 메징 불요.

### 5.2 온도 의존 재료 (문서 5/13이 필수인 이유)

- PSA: T_g 부근에서 E가 2–3자리수 변화, WLF aT로 점성도 연동. 60°C에서는 거의 액상 점성,
  −20°C에서는 유리상 — **폴딩 각도별 반력 모멘트가 온도별로 완전히 다른 곡선**이 됨.
- PET: σ_y(T) 하강(상온 80 MPa → 60°C에서 약 60–65 MPa급). 저온에서는 취성화.
  폴더 최저 사용온도(−20°C)에서 항복 마진이 줄어 **크리칭(영구 주름) 위험 증가** —
  온도 의존 소성 없이 이를 예측 불가.

### 5.3 속도 의존 소성 (문서 11 relevance)

- 폴더 개폐는 준정적(수 초)이지만 **낙하/급속 폴딩(0.1 s)** 시나리오에서는 ε̇ ≈ 1–10 s⁻¹.
  PET의 항복응력 속도 강화(CS D,p)를 무시하면 급속 폴딩 시 응력을 과소평가.
- PSA의 경우 속도 의존은 이미 점탄성(WLF+Prony)이 담당 — JC 소성은 PET/Glass 층 전용.

### 5.4 Mullins/Hysteretic (문서 3/4 relevance)

- PSA는 폴더 전 수명(20만 회 폴딩) 동안 **매 사이클마다 Mullins 연화 + 히스테리시스 소산**.
  1사이클 모델(현행)은 1000사이클 후 반력 모멘트 저하를 예측 못 함.
- 실무 중요도: 반력 모멘트(힌지 토크 설계값)의 **사이클 열화 곡선** = 폴더 힌지 스프링
  설계 입력. 현재 모델로는 1회차 값만 나옴.

### 5.5 파손 (문서 16 relevance)

- 최대 위험 위치: §4.12의 팁 전단 집중 영역 + 힌지 내면 압축/외면 인장 극점.
- PET 외면 인장 ε̄ > ε_fail(≈2–5%) 시 크랙 개시 — ductile damage ω 누적기가 없으면
  "수렴했으나 이미 찢어진" 상태를 놓침(§4.9 교훈의 재료 버전).

---

## 6. Concrete Resolution Plan (해결방안)

### 6.1 Numba 커널 시그니처 (Phase 0/1 핵심)

```python
# --- 온도 필드/테이블 인프라 ---
@njit(fastmath=True)
def interp_table_1d(x: float, n: int, xs: np.ndarray, ys: np.ndarray) -> float:
    """선형 보간 + 양끝 clamp. xs 오름차순 보장."""

@njit(fastmath=True)
def thermal_stretch(temp: float, t_ref: float,
                    n_aT: int, tab_T: np.ndarray, tab_alpha: np.ndarray) -> float:
    """gamma = exp( integral_{t_ref}^{temp} alpha(T) dT )  -- 사다리꼴 적분."""

@njit(fastmath=True)
def apply_thermal_split(F: np.ndarray, gamma: float) -> np.ndarray:
    """F_mech = F / gamma  (등방). orthotropic이면 diag(g1,g2,g3) 역행렬 곱."""

# --- dispatch v2 ---
@njit(fastmath=True)
def material_dispatch_3d_v2(
    mat_type, props, sdv_prev,
    E_voigt, F, detF,
    dt, ttime, temp, dtemp, fields,
) -> (S_voigt, C_mat, sdv_new, error_flag):
    # 1) 테이블에서 E(T,f), nu(T,f), sigma_y(T,f), alpha(T) 보간
    # 2) gamma = thermal_stretch(...); F_mech = F/gamma; detF_mech = detF/gamma**3
    # 3) 기존 구성법칙 코어 호출 (무수정 재사용)
    # 4) 신규: rate-dependent / damage / mullins 분기

# --- Cowper-Symonds return map (평면 deviatoric, eigh 불필요) ---
@njit(fastmath=True)
def cs_return_map(sig_trial_dev: np.ndarray, J2_trial: float,
                  eqps_old: float, ep_old: np.ndarray,
                  sigy0: float, H: float, D_cs: float, p_cs: float,
                  dt: float) -> (sig_dev_new, ep_new, eqps_new, dsig_dep):
    """
    항복조건: f(dp) = sqrt(3*J2(dp)) - [sigy0 + H*(eqps_old+dp)] * (1+(dp/(D*dt))**(1/p))
    dp에 대해 1D Newton (초기값: rate-independent 폐형해). 3-5회 수렴.
    접선 dsig_dep = d(sigma)/d(eps) 폐형 (Consistent tangent, 알고리즘 변별항 포함).
    """

# --- Johnson-Cook (CS의 일반화) ---
@njit(fastmath=True)
def jc_yield_stress(eqps: float, eqps_rate: float, temp: float,
                    A,B,n_jc,C_jc,m_jc, t_room, t_melt, eps_dot0) -> float:
    """sigma_y = (A + B*eqps^n) * (1 + C*ln(max(eqps_rate,eps_dot0)/eps_dot0))
                 * (1 - Tstar^m),  Tstar = (temp-t_room)/(t_melt-t_room)"""

# --- Mullins 래퍼 (초탄성 코어 재사용) ---
@njit(fastmath=True)
def mullins_eta(w_bar_dev: float, wm_max_old: float,
                r: float, m: float, beta: float) -> (eta, wm_max_new, on_reload):
    """
    Abaqus 형식: on_reload = (w_bar_dev >= wm_max_old)
    eta = 1 - (1/r)*erf_arg_form(...)   # Abaqus 공칭 손상함수
    하중제거 분기: eta = 1 (완전 탄성 복귀), wm_max 동결.
    S_dev = eta * S_dev_hyper(w_bar),  S_vol 무변경 (비압축 분리 유지)
    """

# --- Ductile damage 누적 ---
@njit(fastmath=True)
def ductile_damage_update(deqps: float, triax: float, w_duc_old: float,
                          tab_eta: np.ndarray, tab_epsD: np.ndarray) -> (w_new, D_new):
    """w_new = w_old + deqps / epsD_interp(triax);  D = clip(w,0,1) (선형 진화);
       sigma_eff = (1-D)*sigma,  C_eff = (1-D)*C - 손상 연화항"""
```

### 6.2 솔버 연동 수식 (열팽창, 평면변형 2D)

```
F_th = gamma*I,  gamma = exp(alpha_bar*(T - T_ref))
F_mech = F * F_th^-1 = F/gamma,   J_mech = J/gamma^2   (2D 평면변형 임베딩: J_th = gamma^2로 취급 주의
                                                        -- 면외는 F33=1 고정이므로 F_mech,33 = 1/gamma)
S = S_hyper(F_mech)               # 기존 구성법칙 무수정
C_tangent: dS/dF = (dS/dF_mech) * (1/gamma)   # 상수 스케일, 체인룰 1줄
반력(라미네이션 잔류): 자유 층은 S=0, 구속 층은 eps_th 불일치만큼 S 축적
```

WLF 연동(문서 6/5 결합): `dt_eff = dt / aT(T_gp)`, aT는 기존 `wlf_shift()` 재사용 —
GP별 온도 배열만 공급하면 됨(커널 구조 불변).

### 6.3 벤치마크 `.inp` 사양 (구현 완료 후 실행)

| # | 문서 | 시나리오 | 검증량 | 합격 기준 |
|---|---|---|---|---|
| B1 | 22 | 봉 인장, ΔT=100K, α=1e-5 | ε_total = αΔT + σ/E | 해석해 대비 < 0.1% |
| B2 | 22 | 완전 구속 봉 ΔT | σ = −EαΔT | < 0.1% |
| B3 | 5 | E(T) 3점 테이블, T별 인장 | 기울기 = E(T_i) | 테이블 정확 |
| B4 | 13 | σ_y(T) 테이블, T별 항복 | 항복점 | < 0.5% |
| B5 | 11 | 단일 GP ε̇p 스윕 (0.001–100/s) | σ_y(ε̇p) = CS/JC 공식 | < 0.1% |
| B6 | 15 | JC 손상: 노치 봉 삼축도 스윕 | ε̄_D(η) 곡선 | 공식 대조 |
| B7 | 3 | Mullins 2사이클 λ=1→3→1→3 | 2차 재하 곡선 연화율 | Abaqus r,m,β 공식 대조 |
| B8 | 4 | Hysteretic 3사이클 | ∮σdε 루프 면적 | 에너지 균형 < 2% |
| B9 | 8 | g(ε̄) 승수 완화시험 | 완화 곡선 family | < 1% |
| B10 | 16 | Ductile ω=1 개시 시점 | ε̄p at D>0 | 테이블 epsD 대조 |
| B11 | 12 | T > T_anneal 후 재하 | eqps 소거율 | 테이블 frac 대조 |
| B12 | 10 | **두꺼운 점탄 실린더 내압** (기존기능 검증) | σ_θθ(t), u_r(t) vs 해석해 | < 1%, 3개 b/a |
| B13 | 1 | Orthotropic 판 인장 (θ=0/30/45/90°) | E_eff(θ) = CLT 공식 | < 0.1% |

각 B#에 대해 `verification/abaqus_benchmarks/` 신규 스크립트 + 계약 스위트 등록.
B12는 Phase 0 전에라도 즉시 실행 가능(기존 기능) — **Phase V 첫 항목으로 선행 권고**.

### 6.4 리스크 레지스터

| 리스크 | 완화책 |
|---|---|
| dispatch v2 전환 중 기존 14개 모델 회귀 | v1 유지, v2 신규 전용; 계약 C10 게이트 |
| BB Hysteretic 국소 Newton 발산 | 수정 Newton + line search (AGENTS §4.4 선례) |
| SDV 블록 합성 오프셋 오류 | sdv_layout 레지스트리 + 라운드트립 단위테스트 |
| Mullins 분기 전환 수렴 저하 | 전환점 감지 시 dt 자동 축소 (dt_controller 연동) |
| props v1/v2 혼용 휴리스틱 충돌 | mat_type별로 버전 태깅, v2는 휴리스틱 디코딩 완전 배제 |

### 6.5 구현 순서 요약 (의존성 그래프)

```
Phase 0: sdv_layout ─┬─> props_v2 ─> dispatch_v2 ─┬─> thermal_split(22)
                     │                            ├─> temp_tables(5,13)
                     └─> TemperatureField ────────┴─> field_tables(6,14)
Phase 1: CS/JC return map(11) ─> JC damage(15) ─> ductile/shear(16, Phase3)
Phase 2: Mullins(3) ─> nonlinear visco(8) ─> Hysteretic(4)
Phase V: viscous cylinder(10) — 즉시 선행 가능
```

---

## 부록: 기존 자산 재확인 (중복 구현 방지)

- 초탄성 8종(NeoHookean, Arruda-Boyce, Yeoh, Mooney-Rivlin, Polynomial, Ogden, Marlow, Linear) — `numba_materials.py` 완료.
- Prony M=6 대변형 점탄성 + WLF — 완료(2D `viscoelastic.py`, 3D `MAT_HYPER_VISCOELASTIC` SDV 42).
- 테이블 탄-소성(Radioss LAW36) + H_tail=1e-3·E 안정화 테일 — 완료(SDV 8).
- J2 등방/비선형(이동경화 SDV 12) — 존재, 계약 C10 확인 필요.
- DuctileDamage 모델층 stub(`model/material.py`) — 존재, 커널 미구현.
- **부재**: 온도 필드, 필드변수, 열팽창, 속도 의존, Mullins, Hysteretic, creep, annealing, orthotropic, damage 커널.

