# ex13 기반 2D FEM Transient Implicit Solver & 요소 모델 심층 리뷰 — Abaqus 초월 설계안

> **대상:** `examples/ex13_unified_model_io.py` + `ex12_abaqus_inp_plate_fold.py`/`gen_ex12_inp.py` + `dispsolver/solver/dynamic.py` + `dispsolver/element/*` + `dispsolver/material/*` + `dispsolver/constraint/*`
> **목표:** 최신 수학/공학 이론 기준으로 현 solver·요소 모델을 리뷰하고, Abaqus 대비 속도·정확도 모두에서 우위를 점하기 위한 설계안 도출. 코드 수정 없음 — 리뷰·설계만.
> **작성:** 2026-08-30 / Sisyphus
> **관련 문서:** `AGENTS.md` §1–5, `dev_log/ex12_tip_inversion_fix_and_full_closure_20260725.md`, `dev_log/rbe2_condensation_review.md`, `dev_log/perf_assembly_batch_neohookean_20260726.md`, `verification/RULES.md`

---

## 0. 요약 (TL;DR)

| 영역 | 현 상태 한줄 진단 | Abaqus 대비 핵심 격차 | 최우선 개선안 |
|---|---|---|---|
| **시간적분** | `mode="quasistatic"` 고정 — 관성 제거, HHT-α 코드는 존재하나 미사용. 진짜 transient 아님 | Abaqus는 `*DYNAMIC, ALPHA=-0.05~-0.33` 로 HF ringing 제어 + `*VISCO` 크리프 동시 처리. 현 코드는 rate 의존성만 dt로 전달 | HHT-α/Generalized-α를 quasistatic 루프에도 활성화(관성 0이어도 수치감쇠 유효) + `dt`-제약 없는 WLF/프루니 일관성 보정 |
| **Newton** | 단일 변위비 수렴 `tol=1e-3` + 5-way OR — Abaqus 3단계보다 느슨. line search는 NaN 가드만 | Abaqus는 `R_max ≤0.5%·q_avg AND Δu_max≤1%·du_max` 이중 + 에너지. 현 OR는 false-positive 쉬움 | OC-O3 등급: `||R_u||/||q_avg||` + `||Δu||/||u||` + `ΔE` 의 AND-OR 하이브리드 + KKT 전용 merit 복구 |
| **선형해석** | SPD condensation + equilibration + PARDISO singleton은 우수. 그러나 매 Newton factorize, dense 환원, slave 미제거 assembly | Abaqus는 symbolic 재사용·MPC 제거·iterative AMG fallback. 현 코드는 58% assembly 낭비 잔존 | Prescribed-DOF 요소 스킵 + symbolic 재사용 + equilibration 캐시 + iterative fallback |
| **Constraint** | RBE2 condensation으로 10¹⁶→10³ 개선 — 탁월. Tie는 `k=1e4` 단순 penalty, AL 없음 | Abaqus `*TIE, POSITION TOLERANCE + ADJUST + MPC` 는 정확·무조건 안정. penalty는 k 의존·침투 | Mortar/segment-to-segment + Augmented Lagrangian + `k` 자동 스케일링 |
| **요소** | `Q4_COROTATIONAL_SRI` + `Q4_UP(=F-bar)` + `Q4_VISCO_SIMO` 조합은 modern. EAS-4만, dT/du 누락, SRI/F-bar 혼선 | Abaqus CPE4I(비호환모드)+ CPE4H(hybrid u-p) + CPE4R(시간감쇠 hourglass). SRI만으론 bending+volumetric 동시 해결 불완전 | EAS-7/ANS+Q1P0 진짜 hybrid + corot rotational tangent 보완 + F-bar/EAS 통합 |
| **재료** | J2 multiplicative + Arruda-Boyce+Prony/WLF Flory split은 이론적으로 최첨단 | Abaqus는 Bergstrom-Boyce, Parallel Rheological Framework(PRF) 다중네트워크. 현재 1-term Prony는 빈약 | 2~3-term PRF + Mullins + 측정 기반 `lambda_m` 캘리브레이션 |
| **속도** | JAX vmap per-pid 캐시로 20분→85초 달성. 여전히 assembly 58%, PARDISO 85% 과거 병목 유사 패턴 잔존 | Abaqus는 domain decomposition + multi-front + GPU. Numba 경로는 미완 | Numba F-bar/EAS 전체 커버 + COO→CSR 일괄 + element 스킵 |
| **검증** | 12-벤치 verification(패치·굽힘·수렴차수) 견고. 그러나 접촉·대변형 벤치 없음 | Abaqus는 NAFEMS 벤치 200+ | 대변형 elastica·crease·interlayer shear 벤치 추가 |

**결론:** 이론적 토대(Flory split, Simo 1987 overstress, EAS, corotational, condensation)는 2024년 기준 Abaqus와 동급 이상이다. 격차는 **알고리즘 완성도(수렴판정·line search·tie)와 성능 엔지니어링(assembly 스킵·factor 재사용)** 에 집중돼 있다. 이 두 축만 메워도 동일 mesh에서 Abaqus 대비 1.5–2× 속도 + 동등 이상 정확도 달성 가능.

---

## 1. ex13 아키텍처 리뷰

### 1.1 잘한 점 — Single Source of Truth

`dispsolver/fold_model_config.py` + `dispsolver/mesh/display_builder.py` 로 `.inp` 생성과 Python 직접 빌드(`run_build`)가 동일 topology를 공유하는 구조는 **탁월**하다. 과거 drift(`element_type` dict desync로 첫 Newton 20분) 교훈을 정확히 반영. `DisplayGrid`가 void edge를 `xs` partition에 강제 포함해 sliver(`~1e-15`)를 원천 차단한 것도 이론적으로 올바른 COMSOL/Abaqus CAE 방식.

### 1.2 남은 리스크

- **`FoldModelConfig` 기본값이 `uniform=True`, `plate_body_dx=0.5` 이나 `hinge_span_dx=0.25` 와 `hinge_half_gap=1.0`, `hinge_span_half_width=0.25` 조합:** `AGENTS.md §4.12`의 graded 핵심(팁 0.25 / 힌지엣지 0.25 / hinge span 0.25 / plate body 1.0)이 `uniform=True` 일 때 모두 `0.5`로 무력화된다. `DEFAULT_CONFIG` 가 현재 "graded 끈" 상태 — ex12 성공 조건과 불일치. Abaqus `*ELGEN` 처럼 `grading.uniform` 기본을 `False` 로 바꿔야 한다.
- **`run_folding_from_result` 의 `params["E"]>10000 → 1e-6` 하드코딩:** STEEL plate를 사실상 rigid로 만드는 트릭이지만 `E` 를 1e-6 으로 두면 plate 요소의 `K_e≈0` 에 `eps_reg=1e-8·min_diag` 가 상대적으로 커져 plate DOF가 regularization에 지배된다. Condensation이 plate를 이미 제거하므로 실해는 없으나, 향후 condensation 미사용 경로에선 위험. `ex13` build 경로처럼 plate를 condensation으로만 처리하고 `K_plate` assembly 자체를 스킵하는 게 정석.
- **`display_half_length=40, hinge_half_gap=1.0, hinge_pivot_x=2.5` — AGENTS.md §1.0 허용 조건 `hinge_pivot_x < hinge_half_gap` 위반:** 문서상 피벗은 hinge gap *안*(`|x|<1`)에 있어야 plate가 display를 crush하지 않는다. 현재 `2.5>1.0` 이면 피벗이 plate 내부(`x∈[1,40]`)에 들어가긴 하나 gap 바깥. `gen_ex12_inp.py` 레거시와 충돌. 일관성 점검 필요.
- **Void가 `layer_void_regions={1:[(-10,0),(0,10)]}` 로 bottom PSA에만 적용:** 의도는 tape cutout 모델이나, 현재 6개 PSA layer 중 최하단 1개만 void. 실제 display는 OCA 패턴이 층마다 다르다. `build_display_grid` 는 이를 지원하나 config 문서화가 부족.

---

## 2. Transient Implicit Solver 리뷰

### 2.1 Time Integration — Quasistatic 고정과 HHT-α 사장

**현 코드 (`dynamic.py` L476-486, `ex12...py` L175-183):**

```python
INTEGRATION_MODES["quasistatic"] = dict(static_mode=True, alpha=0.0, beta=0.25, gamma=0.5)
# ex13:
solver = DynamicSolver(..., mode="quasistatic", ...)  # static_mode=True, M·a 항 제거
```

- 문서상 HHT-α (`α∈[-0.3,0], β=(1-α)²/4, γ=½-α`) 일반식은 완벽히 구현(Hilber-Hughes-Taylor 1977, Hughes 1987 Ch.9). 그러나 ex13/ex12 실행 경로는 **항상 quasistatic** — `K_eff = K_T` , `R_u = f_ext - f_int` 로 관성 완전 제거. `dt` 는 Newton 증분 크기 + 점탄성 `dt` 전달에만 쓰인다.
- 즉, 소위 "transient implicit" 라는 파일 헤더와 달리 **실행은 static arc-length 없는 `*STATIC`** 이다. Abaqus `*DYNAMIC, APPLICATION=QUASI-STATIC` 과 동일한 한계.

**최신 이론 대비 평가 (2020–2025):**

- **Generalized-α (Chung & Hulbert 1993, Jansen et al. 2000):** HHT-α를 `α_m, α_f` 2-파라미터로 확장해 HF 감쇠를 유지하면서 저주파 정확도를 더 높인다. Abaqus 2022+ `*DYNAMIC, ALPHA` 는 내부적으로 Generalized-α. 현 코드가 `α_m=α, α_f=0` 특수화로 머물러 있는 것은 1990년대 수준.
- **Quasistatic에서도 수치감쇠가 필요한 이유:** 힌지 bifurcation 직후 `K_T` 가 near-singular → Newton 수정 `Δu` 에 HF 진동이 섞인다. Abaqus `*STATIC, STABILIZE` 는 점성력을 더해 이를 잡는다. 현 코드는 `stabilization.py`(`F_stab=c·M·v`) 를 정의만 해두고 **실제 solve 루프에서 호출하지 않는다.** `c=2e-4`, `E_stab/E<0.5%` 자동 스케일은 Abaqus와 동일 아이디어이나 배선 끊김.
- **점탄성의 dt 일관성:** `ViscoelasticMaterial.pk2_voigt(F, h_prev, dt)` 에 `dt` 가 들어가므로 quasistatic이라도 `dt` 가 물리적 의미를 가진다. 그러나 `AdaptiveDtController` 가 `n_iter` 만 보고 `dt` 를 키웠다 줄였다 하므로, 점탄성 완화시간 `τ=3.33s` 대비 물리적 `dt` 가 들쭉날쭉 — WLF/프루니 정확도가 dt 변동에 민감.

**설계안 (Abaqus 초월):**

1. **모드 분리 명확화:** `mode="quasistatic"` 를 `mode="static"` 로 개명하고, 점탄성용 `mode="visco"`(여전히 `M=0` 이나 `dt` 물리시간 고정 + Generalized-α 감쇠만 활성화) 를 신설. `dt_controller` 와 물리 `dt` 를 분리: 점탄성에선 `dt` 를 `τ/10` 정도로 고정하고 Newton sub-increment만 적응.
2. **Generalized-α 승격:** `INTEGRATION_MODES` 에 `alpha_m, alpha_f` 도입, `ρ_∞` 로 파라미터화(`α_m=(2ρ∞-1)/(ρ∞+1), α_f=ρ∞/(ρ∞+1)`). Abaqus 기본 `ρ∞=0.8` (`α=-0.05`) 과 `ρ∞=0.5` 두 프리셋 제공.
3. **Stabilization 배선:** `DynamicSolver` 에 `stabilization: Optional[AbaqusViscousStabilization]` 인자 추가, `_assemble` 직후 `F_stab, K_stab` 를 `R/K` 에 더한다. 단, `E_stab/E_strain` 를 매 step 누적해 0.5% 초과 시 `c` 자동 감소(현재 로직) 대신 **Abaqus처럼 `*ENERGY OUTPUT` 로 노출**.
4. **HHT-α 관성 경로 부활 옵션:** `density=1e-9` 는 현재 사실상 0. 실제 inertia가 필요한 drop/impact 시나리오를 위해 `rho` 를 실제 값으로 두고 `mode="transient"` 를 쓸 수 있게 회귀 테스트 추가.

### 2.2 Newton-Raphson & Convergence — 5-way OR의 함정

**현 코드 (`dynamic.py` L1503-1567, `_solve_step_impl` 수렴 판정 L2068 이후 — 실제 판정 로직은 `dynamic.py` 2180줄 근처, `tol/rtol/atol` + energy + KKT residual):**

`AGENTS.md §2` 에 따르면 5-way OR: `||Δu||/||u||<tol` OR `|R·Δu|<1e-15` OR `||R||/||q_avg||<rtol` OR Abaqus식 `R_max≤0.005·q_avg AND c_max≤0.01·du_max` OR `||Δu||<atol`.

- 2026-07-25에 3-Tier AND 재작성이 시도됐으나 `t≈0` 에서 cutback 캐스케이드로 revert — 현재 OR 유지.
- `tol=1e-3` (`SolverTuningConfig`) 은 Abaqus 기본(`0.5% residual + 1% correction`) 보다 느슨. `max_iter=25` 는 Abaqus 기본 16보다 관대.

**이론적 문제점:**

- **OR는 false-positive:** 변위비가 작아도 잔류력이 크면 비평형 상태서 수렴 선언. 얇은 PET/PSA 접합부처럼 `K_e` 스케일이 층마다 10⁵ 배 차이나는 문제에서 치명. Abaqus가 AND를 고집하는 이유.
- **Abaqus식 `R_max/q_avg` 분모 `q_avg` 의 정의 불명:** 현 코드는 `q_avg` 를 단순 평균 외력으로 두나, Abaqus는 시간평균 수렴력 `q_avg` 를 별도 정의(증분력·반력 포함). 이 차이로 초기 `t≈0` 에서 분모≈0 → 수렴 판정이 불안정해 3-Tier가 실패한 것으로 보임.
- **KKT saddle-point에서 `||R||` 은 merit가 아님:** 코드 주석(L1536-1545)이 정확히 지적 — Greenstadt 1967, Dennis&Schnabel §6.4. `||R||` 감소가 평형 개선을 보장하지 않는다. 그런데 line search가 `||R_active||` 로 Armijo를 시도 — 모순.
- **Line search는 NaN 가드 + 완화된 Armijo(`c=1e-4`) 만:** `alpha_max=1.0` 은 올바르나(과거 0.7 cap 제거), saddle-point에선 Armijo 자체가 무의미. Abaqus는 `*CONTROLS, LINE SEARCH` 로 별도 merit(에너지)를 쓴다.

**설계안:**

1. **수렴판정 3-Tier 부활 — 단, 분모 안정화:** `q_avg = max(||f_ext||, ||f_int||, ||R_prev||, ε)` 로 재정의해 `t≈0` 에서 0 분모 방지. `tol=5e-3`, `rtol=5e-3`, `atol=1e-8` (변위 mm 스케일) 로 Abaqus 기본과 정렬. OR 대신 `(force_converged AND disp_converged) OR energy_converged` 로 변경하되, ex03 단위테스트가 아닌 **ex12 90° 풀런** 으로 검증(AGENTS.md §4.7 교훈).
2. **Merit 함수 교체:** KKT 잔류 대신 **증분 에너지 `Δu·R_u` (mechanical block만)** 를 Armijo 대상으로. `R_lam` (제약 잔류)는 별도 `||g||<tol` 로 체크. 이는 Abaqus `*CONTROLS, PARAMETERS=FIELD` 의 두 블록 분리 전략과 동일.
3. **Arc-length (Riks) 옵션:** 90° 이후 post-buckling snap-through를 위해 Crisfield arc-length를 선택적 활성화. 현재 `theta_targets` 가 변위 제어라 snap을 못 잡는다.
4. **Jacobian 재사용 (Modified Newton + Broyden):** 연속 step에서 `K_eff` 변화가 작으면 factorize 재사용 + Broyden rank-1 업데이트. PARDISO factorize가 Newton당 1회인데 이를 2–3회당 1회로 절감.

### 2.3 Linear Solver — PARDISO와 KKT→SPD 변환

**현 코드:**

- `_equilibrate` (대각 equilibration, `scale=1/√|diag|`) + `_solve_linear_system` (PARDISO `mtype=11` unsym + iterative refinement 4회) — **매우 우수**. KKT 스케일 10⁵ 차이를 `<10²` 로 축소.
- `RBE2CondensationManager` 가 `TᵀKT` 로 KKT를 SPD( `n_independent ≈ n_dofs - 2·n_slave + n_extra` ) 로 축소, `κ 10¹⁶→10³` — 이론적 최적해(Belytschko Ch.6, AGENTS.md §4.10).
- `_get_pardiso_solver` singleton으로 `__init` glob 비용 제거(410s→152s) — 탁월.

**남은 병목 (AGENTS.md §4.11, §4.13):**

- **매 Newton factorize:** `p_solver.factorize(Js)` 가 iteration마다 호출. PARDISO는 symbolic+numeric을 분리할 수 있는데 매번 numeric만이라도 비용. Abaqus는 `*STEP, INC` 에서 front 재사용.
- **Condensation 후 dense 변환:** `K_red.toarray()` (L1855) — `n_independent≈~3000` 에서 dense `3000²·8B≈72MB` 로 아직 괜찮으나, mesh 2× 시 4× 메모리. Abaqus는 sparse 유지.
- **Equilibration 매번 재계산:** `scale` 이 `K_red` 대각에만 의존하므로 Newton 초기 1회만 계산해도 됨.
- **Prescribed-DOF 요소 assembly 낭비:** plate 요소 480개가 매 Newton `lil→coo` 로 조립되나 자유 DOF에 기여 0. AGENTS.md §4.13 TODO가 정확.
- **CG/AMG fallback 없음:** PARDISO 실패 시 `spsolve` 하나뿐. Abaqus는 iterative + direct 하이브리드.

**설계안:**

1. **Prescribed 요소 스킵 (최우선, 58% 절감):** `DynamicSolver._assemble_multi_material_batch` 진입 전 `fully_prescribed = np.all(np.isin(dof_indices[e], bc_dofs))` 마스크로 `f_int/K_T` 기여 스킵. 단, tie/RBE2 coupling이 있는 요소는 제외 조건에 포함(AGENTS.md §4.8 교훈).
2. **Symbolic 재사용:** `p_solver.factorize` 대신 `p_solver._mkl_pardiso.pardiso(..., phase=12)` symbolic을 step 첫 Newton에서만, 이후 `phase=22` numeric만. `scipy` fallback에선 `splu` 재사용.
3. **Sparse 유지 + equilibration 캐시:** `K_red` 를 CSR 유지, `scale` 은 첫 Newton에서 1회 계산해 재사용. `K_eq = D·K·D` 는 CSR 곱으로 O(nnz).
4. **Iterative fallback:** 큰 모델(`n>20000`)에선 `pyamg`/`scipy.sparse.linalg.cg` + equilibration을 PARDISO 대체로 제공. 수렴 실패 시에만 direct로 폴백.

### 2.4 Constraints — RBE2 vs Tie

**RBE2 (우수):** `u_s = u_m + (R(θ)-I)d0` 정확 운동학, `T` 야코비안(`∂u_s/∂θ = dR/dθ·d`) 매 Newton 재구축, SPD 변환 — Abaqus `*RIGID BODY` 와 동등. `theta_penalty_k=1e8` 는 `K_extra` 대각에만 더해져 θ 자유도 안정화 — 깔끔.

**Tie (취약):**

- `SurfaceTieConstraint` 는 3-node penalty(`slave + 2 master`, `N1=(1-ξ)/2, N2=(1+ξ)/2`, `f=k·gap, K=k·[I, -N1 I, -N2 I; ...]`) — 단순하고 빠르나 **Abaqus `*TIE` 와 본질적 차이**:
  - Abaqus tie는 MPC 제거(제약행렬) 또는 mortar 적분으로 **k 무관 정확** 구속. penalty는 `k` 선택 딜레마: `k` 작으면 gap(침투), `k` 크면 `κ(K)` 악화.
  - 현재 `k=1e4` 고정. 과거 1e2~1e10 스윕에서 gap 40% 포화(AGENTS.md §4.10) — RBE2 penalty 시절 이야기나, 현재 condensation 후에도 `k` 민감도는 남음.
  - Position tolerance 0.5mm, 최근접 세그먼트 1개만 — 곡면(폴딩 후 U자)에서 master 세그먼트가 회전하면 초기 투영 `ξ` 가 틀어진다. Abaqus는 매 increment 재투영 + `ADJUST=YES`.

**설계안:**

1. **Augmented Lagrangian (AL) 래퍼:** penalty loop 바깥에 `λ_{k+1}=λ_k + k·gap` Uzawa 업데이트 2–3회. Abaqus `*CONTACT, PENALTY` 와 동일. `k` 는 그대로 두고 gap을 1e-5→1e-8로 감소.
2. **Mortar segment-to-segment 통합:** 현재 node-to-segment 1점 적분을 segment 중점 2점 Gauss로 확장. `N1,N2` 가 매 Newton 현재 위치서 재계산되므로 곡률 추종 개선.
3. **k 자동 스케일링:** `k = β·E·h / A` (β=10~100, E=PET 4000MPa, h=요소크기) 로 mesh/objective 스케일 독립화. 현재 고정 1e4는 두께 0.5/0.017 모두에 동일 — 불균형.
4. **재투영 옵션:** 큰 회전(>30°)에선 `xi` 를 매 Newton 현재 투영으로 갱신. 비용은 O(n_slave·log n_master) 로 무시.

---

## 3. 요소 모델 리뷰

### 3.1 Q4 B-bar (SRI) — `dispsolver/element/q4.py`

**이론:** Hughes (1980, 1987 Ch.4) B-bar `B̄ = B_dev(ξ)+B_vol(ξ=0)`. 체적 잠금(ν→0.5) 제거, 패치테스트 보존(`∫B̄ dV = ∫B dV` 는 아니나 평균 dilatation으로 상수 변형률 재현). 2×2 deviatoric + 1점 volumetric — Abaqus CPE4와 동일 계열.

**평가:** 체적 잠금엔 효과적이나 **굽힘 잠금(shear locking) 미해결**. 0.017mm PET 단층을 B-bar만으로 굽히면 `K_bending` 이 `AR²` 에 비례해 10–300× 과대(AGENTS.md §4.1 실측). 그래서 PET에 EAS/corot이 필수 — B-bar 단독은 display에 부적합, plate(Steel)처럼 변형 없는 곳에만 적합. 현 코드가 `element_type` 미지정 시 B-bar로 fallback하는 것은 안전하나, PET/PSA에 B-bar가 실수로 배정되면 20분 stall 재발.

### 3.2 Q4_EAS — `q4_eas.py` / `q4_eas_jax.py`

**이론:** Simo & Rifai (1990) EAS-4, `ε = B·u + M·α`, `M(ξ)=[[ξ,0,0,0],[0,η,0,0],[0,0,ξ,η]]` (자연좌표). 정적 응축 `K_eas = K_uu - K_uα K_αα⁻¹ K_αu`, 4개 내부파라미터 `α` 는 요소별 Newton(12회)으로 응축. 패치테스트는 `∫M dV=0` + `(detJ0/detJ)T0⁻¹` 스케일로 보장.

**현 구현 강점:**

- 유한변형 TL: `F_enh = (detJ0/detJ) D_k J0⁻¹`, `F_t = F_c + Σα·F_enh`, `B_L`+기하강성, `C` 는 J2 return map과 일관 — **교과서적 정확**.
- JAX 버전은 `tangent_voigt_jax` autodiff로 `C` 일관성 확보 + `while_loop` 고정 20회로 vmap 안전, `tanh` 포화 + NaN freeze로 강건.

**약점 vs 최신 (2020–2025):**

- **EAS-4 vs EAS-7:** EAS-4는 면내 굽힘만 커버. 두께 방향 Poisson 잠금·횡전단 잠금은 EAS-7(Andelfinger & Ramm 1993)이 필요. PET 0.017mm 박판에선 EAS-4로 충분하나, 향후 3D shell 확장 시 EAS-7이 요구.
- **ANS (Assumed Natural Strain, Dvorkin & Bathe 1984) 미결합:** 횡전단 `γ_xy` 에 EAS만 쓰고 ANS를 안 쓰면 aspect ratio 30에서 잔류 전단 잠금. Abaqus CPE4I는 EAS+ANS 하이브리드. 현 `M` 은 전단 모드 2개뿐.
- **Q4_EAS의 FD 접선 (numpy 경로):** `q4_eas.py:_stress_and_tangent` 가 `h=1e-6` 전방차분으로 `C` 를 구한다. J2 상태 `state_gp` 를 고정한 채 `F+ h·dF` 로 3방향 미분 — 비용 3× `pk2_voigt` + 비대칭 보정. JAX 경로는 autodiff로 정확하나 numpy 폴백은 부정확·비대칭 보정 필요. Abaqus는 해석적 일관 접선.
- **`_ALPHA_MAX=0.05` tanh 클램프:** 큰 굽힘에서 `α` 포화는 수렴을 돕나 물리적 EAS 변형을 인위 제한. Simo & Armero (1992) 는 `α` 제한 없이 `K_αα` 정규화로 처리.

**설계안:**

1. EAS-4 유지하되 numpy 경로의 FD 접선을 **JAX autodiff 래퍼 재사용** 으로 교체(가능하면) 또는 해석적 `C` 로 교체.
2. 장래 박판 고정밀을 위해 `Q4_EAS_ANS` 변종 신설 — `γ_xy` 에 Bathe-Dvorkin 4점 ANS 보간 추가.
3. `_ALPHA_MAX` 를 제거하고 `K_αα` Tikhonov `1e-10·tr(K)/4` 정규화만으로 처리 — JAX `tanh` 는 fallback 유지.

### 3.3 Q4_COROTATIONAL — `q4_corotational_jax.py`

**이론:** 요소 강체회전 `R` 를 edge vector 평균으로 추출(`e1=(v12+v43)/|·|, R=R_curr·R_refᵀ`), `u_local = Rᵀ·x_curr - X_ref`, `F_local = I+∇u_local` 로 대변형 회전을 제거해 작은 변형률 영역서 재료법칙 적용. Wriggers (2008) Ch.4.

**현 구현:**

- J2와 결합된 `compute_corotational_j2_contributions_jax` 는 `F_local` 로 `tangent_voigt_jax` 호출, `K_local=Σ B_Lᵀ C B_L w`, `K_global=T8 K_local T8ᵀ` — **dT8/du 항 제거(modified Newton)**.
- 문서·코드가 컴파일 비용(480초) 때문에 full autodiff를 포기했다고 명시 — 타당한 엔지니어링 판단.

**비판:**

- **dT8/du 누락은 일관성 손실:** `f_global=T8·f_local(u_local(R(u_global)))` 의 정확한 접선은 `K_T = T8 K_local T8ᵀ + (∂T8/∂u)·f_local` . 둘째 항이 없으면 Newton 2차 수렴 상실 — 관찰된 1→2회 반복 증가 원인. 90° 접힘에선 1회→4–7회로 악화 가능.
- **SRI 결합 (`Q4_COROTATIONAL_SRI`):** `J0/J` 보정이 없는 plain corot + SRI 체적. 그러나 `R` 추출이 reference `coords_init` 기준이라 요소가 크게 찌그러지면 `R` 추정 오차. Abaqus `*ORIENTATION` 은 polar decomposition `R= F·U⁻¹` 로 더 정확.
- **현재 `DEFAULT_CONFIG` 가 `pet_element_type="Q4_COROTATIONAL_SRI"` :** `Q4_EAS` 가 shear locking을 완전히 제거(AGENTS.md §4.1)하는데 굳이 corot+SRI로 간 것은 `Q4_EAS` 의 FD 접선 불안정 때문으로 보임. 그러나 corot도 modified Newton이라 속도 이득이 상쇄.

**설계안:**

1. **회전 접선 보완 (선택):** `∂T8/∂u` 를 해석적으로 추가 — `∂R/∂u` 는 edge vector 미분으로 closed-form. 비용은 `K_local` 대비 10% 이내, 수렴 2차 회복.
2. **권장 PET 요소 재평가:** `Q4_EAS` 를 기본으로 복귀(`Q4_COROTATIONAL_SRI` 는 비교 옵션으로 유지). EAS가 이미 shear+volumetric 모두에서 AR 독립성을 증명.
3. **Polar decomposition 옵션:** `R` 추출을 `scipy.linalg.polar(F_avg)` 로 교체한 `Q4_COROTATIONAL_POLAR` 변종을 벤치.

### 3.4 Q4_VISCO_SIMO — `q4_visco_simo_fs_jax.py`

**이론 (최첨단):** Flory 분해 `F=J^{1/3} F̄, det F̄=1, J=det F`, `Ī1=J^{-2/3} I1`, `S_iso=2W1 J^{-2/3}(I - I1/3 C⁻¹)` (Holzapfel 2000 Eq.6.88-6.91) 로 `S_iso=0 @ F=I` 보장 + `S_vol=κ lnJ C⁻¹` (Simo & Hughes 1998). `W1` 은 neohookean/yeoh/arruda 플러그. Simo (1987) overstress `h_i(t+Δt)=β_i h_i(t)+g_i γ_i ΔS_iso`, `β=exp(-Δt/τ*), γ=(1-β)τ*/Δt`, `S_eff=S_vol+g∞S_iso+Σh_i`. 접선은 `jax.jacobian` at frozen history — 정확, `γ` 가 자동 반영.

**현 구현 강점:** F-bar `F̄=F√(J0/J)` 로 체적 잠금 방지(de Souza Neto 1996), JAX autodiff로 `W1` 별 분기 없이 정확한 `K` — **Abaqus PRF와 동급 이상**(Abaqus는 동일 Simo 프레임워크).

**약점:**

- **체적 에너지 불일치:** `_simo_pk2` 는 `U=κ/2 (lnJ)²` , `ArrudaBoyce` 는 `U=K/2 (J-1)²` . `simo_fs_args` 가 `K` 를 `κ` 로 넘기지만 `W` 형태가 달라 `S_vol` 이 요소별로 다름 — PSA vs PET 체적 응답이 다른 `U` 로 계산.
- **F-bar vs EAS 중복:** visco 요소는 F-bar만, EAS는 미포함. 얇은 PSA(0.03mm, ν=0.49)도 굽힘 시 shear locking 가능. `Q4_VISCO_SIMO` 에 EAS를 결합한 `Q4_VISCO_SIMO_EAS` 가 이론적으로 필요하나 미구현.
- **WLF 등온 고정:** `temperature=20°C` 상수, `wlf_shift` 는 있으나 folding 중 온도 상승(접힘 마찰열) 미모델.

**설계안:**

1. 체적 `U` 를 `lnJ` 형태로 통일 — `ArrudaBoyce` 의 `K/2(J-1)²` 를 `κ/2(lnJ)²` 로 교체하거나, `K` 전달 시 보정 `κ=K·J` .
2. `Q4_VISCO_SIMO` 에 EAS-4 결합 — `F_enh` 를 `F̄` 에 더하는 형태.
3. PRF 다중네트워크: 현재 1-term Prony(`g=0.2, τ=3.33`)를 2-term(`g=[0.2,0.1], τ=[0.5,10]`) 로 확장해 단기 크리프+장기 완화 동시 포착.

### 3.5 Q4_UP / LinearViscoelastic — `q4_up_jax.py`, `linear_viscoelastic.py`

**이론:** Q1P0 mean-dilatation hybrid(Simo & Hughes Ch.7) — 변위+압력 `p` 2-필드, `p` 는 요소 상수. PSA ν=0.49 체적 잠금에 필수.

**현 코드 혼선:** `FoldModelConfig.psa_element_type="Q4_UP"` 이나 `dynamic.py` L1134-1135에서 `Q4_UP+ViscoelasticMaterial` 은 `Q4_VISCO_SIMO` F-bar 커널로 라우팅된다는 주석이 있다. 즉 **라벨은 Q4_UP, 실제는 F-bar**. 진짜 Q1P0(`q4_up_jax.py`)는 `LinearViscoelastic` 전용으로만 분기. `ViscoelasticMaterial(ArrudaBoyce)` 는 F-bar, `LinearViscoelastic` 은 Q1P0 — 두 Visco가 다른 요소로 가는 설계는 혼란.

**설계안:** 라벨 정리 — `Q4_UP` 은 Q1P0, `Q4_VISCO_SIMO` 는 F-bar+Simo로 엄격 분리. PSA 권장 경로는 `Q4_VISCO_SIMO` (유한변형 정확)로 일원화하고, `LinearViscoelastic` 은 소변형 검증 전용으로 문서화.

---

## 4. 재료 모델 리뷰

### 4.1 J2Plasticity — `plastic.py` / `plastic_jax.py`

- Simo (1992) multiplicative `F=F_e·F_p, det F_p=1`, Hencky `ε_e=½ ln b_e`, spectral return map, `σ_y=σ_y0+H·eqps` 선형경화 — **정석**. `b_e_tr` 스펙트럼 분해→시도 Kirchhoff→방사형 복귀→`F_p⁻¹` 갱신 흐름 정확.
- Batch `pk2_voigt_batch` 의 `bad` 요소 정규화(영 응력+등방 접선)는 line search/cutback 복구를 가능케 하는 **실전적** 처리 — Abaqus `*PLASTIC` 의 `RTOL` 과 유사.
- **격차:** 경화 테이블 보간은 있으나 `σ_y0=80, H=400` 고정값은 PET 실측과 괴리. Abaqus는 `*PLASTIC, HARDENING=COMBINED` kinematic+isotropic, 변형률속도 의존 Johnson-Cook까지. 피로/크랙 예측엔 부족.

### 4.2 Viscoelastic — `viscoelastic.py`

- Prony+WLF + 편차/체적 분리 `S_vol=J·p·C⁻¹, p=(-μ+λ lnJ)/J` 는 Simo & Hughes 10.3 정확 구현. `_extract_lam_mu` 가 NeoHookean/Arruda/K 하이브리드를 `μ=0, λ=K` 로 통일해 `p=K lnJ/J` 로 환원 — 영리한 파라미터 브리지.
- Batch의 `bad` 정규화, `det2` 해석적 역, `lnJ` 안전 — **수치 안정성 우수**.
- **격차:** `h_state=(M+1,3,3)` 전체 텐서 저장은 메모리 과다. Abaqus는 대칭 6성분만. 14층×~2000요소×4GP×42double≈ 5MB 수준으로 현재는 괜찮으나 3D 확장 시 병목.

### 4.3 Arruda-Boyce — `arruda_boyce.py`

- 5항 `W_dev=μ Σ C_k/λ_m^{2(k-1)}(Ī1^k-3^k)` + `K/2(J-1)²` — 8-chain 정확. `λ_m=3.0` 가정치는 AGENTS.md가 이미 지적한 측정 공백 — **가장 큰 재료 불확도**. `K=0.8333` MPa (E=0.05 MPa, ν=0.49) 는 PSA 실측 대비 10× 연화(이전 0.5→0.05) — 의도적 연화.

---

## 5. Abaqus 대비 속도·정확도 격차 종합 및 설계안

### 5.1 속도가 Abaqus보다 느린 이유 — 프로파일 기반

`dev_log/perf_assembly_batch_neohookean_20260726.md` 와 `AGENTS.md §4.11` 이 이미 규명:

1. **Assembly 58%** — prescribed plate 480요소 + `lil` 산란 + per-element Python 루프 일부 잔존
2. **PARDISO 85%→해결됐으나 factorize 반복** — singleton으로 63% 개선했으나 여전히 매 Newton factorize
3. **JAX recompilation** — per-pid 캐시로 20분→초 해결, 그러나 `in_axes=(0,0,0,None,0)` 등 세분화 시 재컴파일 위험 잔존
4. **Dense condensation** — `K_red.toarray()` O(n²) 메모리

**Abaqus 초월 로드맵 (속도):**

| 순위 | 개선 | 예상 효과 | 난이도 |
|---|---|---|---|
| P0 | Prescribed 요소 스킵 + COO 일괄 | 30–50% wall 감소 | 하 |
| P0 | Numba F-bar/EAS 전체 커버(현재 JAX만) — `q4_visco_simo_fs_jax` Numba 포팅 | 2× microbench, 전체 30% | 중 |
| P1 | PARDISO symbolic 재사용 + equilibration 캐시 | 15–20% | 하 |
| P1 | Assembly→CSR 직접, `lil` 제거 | 10% | 하 |
| P2 | Iterative AMG (pyamg) + PARDISO 하이브리드 | 대화면(n>50k)에서 3× | 중 |
| P2 | GPU JAX (jit→pmap) — `jax` 이미 GPU 준비 | 클러스터서 5× 가능 | 중 |
| P3 | Matrix-free Newton-Krylov (K 없이 `J·v` 만) | 초대형서 메모리 10× 절감 | 상 |

### 5.2 정확도가 Abaqus보다 낮은 지점

- **Tie penalty 오차:** `k=1e4` 에서 gap 1e-5mm 수준이나, 층간 전단 81µm 대비 12% 오차 가능성. Abaqus MPC는 gap 0.
- **SRI 전단 잠금 잔류:** `Q4_COROTATIONAL_SRI` 의 AR 30에서 316× 과강성(AGENTS.md §4.1) — EAS가 해결하나 현재 기본이 SRI이므로 PET 일부 층에서 과대 굽힘 모멘트.
- **점탄성 1-term 한계:** 단일 τ=3.33s는 0.1–10s 스펙트럼을 못 잡음. Abaqus 3-term Prony가 표준.
- **검증 공백:** 대변형 elastica 외에 접힘 크리즈(crease) 곡률, interlayer slip 벤치가 없어 정확도 주장 불가.

**정확도 로드맵:**

| 순위 | 개선 | 효과 | 난이도 |
|---|---|---|---|
| P0 | PET 기본 `Q4_COROTATIONAL_SRI` → `Q4_EAS` 복귀 | 굽힘 모멘트 오차 수%→0.3% | 하 |
| P0 | Tie Augmented Lagrangian 2회 | gap 1e-5→1e-9 | 하 |
| P1 | 2-term Prony + `lambda_m` 측정 캘리브 | 크리프 예측 오차 50%→10% | 중 |
| P1 | 체적 `U` 통일(`lnJ` 계) | PSA/PET 체적 일관성 | 하 |
| P2 | Mortar tie + 재투영 | 곡면 접촉 정확도 | 중 |
| P2 | EAS-7/ANS + Q1P0 진짜 hybrid | 박판+비압축 동시 정확 | 상 |

---

## 6. 구체적 설계 스케치 — 구현 전 합의 필요

### 6.1 Solver 인터페이스 개편안 (breaking 최소화)

```python
# 현재
DynamicSolver(mesh, material, rho=1e-9, mode="quasistatic", ...)

# 제안 — 하위호환 유지, mode 확장
DynamicSolver(
    mesh, material,
    mode="visco",           # "static"|"visco"|"transient"|"generalized-alpha"
    rho=1e-9,               # visco면 0이어도 ok
    alpha_rho_inf=0.8,      # Generalized-α spectral radius
    stabilization=AbaqusViscousStabilization(c=2e-4),  # None이면 off
    tie_al_iters=2,         # Augmented Lagrangian 반복
    skip_fully_prescribed=True,
)
```

### 6.2 요소 라벨 정리안

| 라벨 | 의미 | 재료 페어링 | 상태 |
|---|---|---|---|
| `Q4` | B-bar SRI (현 `q4.py`) | Steel plate | 유지 |
| `Q4_EAS` | EAS-4 + TL + J2 | PET 권장 | 유지, FD→autodiff 교체 |
| `Q4_COROTATIONAL` | corot + J2 (modified Newton) | PET 대안 | 유지, dT/du 옵션 추가 |
| `Q4_COROTATIONAL_SRI` | corot + SRI | PET (현 기본, 비권장) | deprecate 예정, `Q4_EAS` 로 리다이렉트 경고 |
| `Q4_VISCO_SIMO` | Flory F-bar + Simo overstress (JAX autodiff) | PSA 권장 | 유지, 2-term 확장 |
| `Q4_UP` | 진짜 Q1P0 hybrid u-p | PSA 소변형 검증 전용 | `LinearViscoelastic` 전용으로 문서화, `ViscoelasticMaterial` 과 분리 |

### 6.3 검증 확장안

`verification/benchmarks.py` 에 추가:

- **elastica pure moment (대변형 굽힘):** 해석해 `M=EI·κ` vs corot/EAS 곡률 오차 — Abaqus NAFEMS 3D7 변형
- **volumetric locking patch (ν=0.499):** Q4 vs Q4_UP vs F-bar 압력 오실레이션 비교
- **interlayer shear (PSA):** `check_interlayer_shear.py` 의 81µm book-page를 해석적 shear lag 해와 비교 — 현 코드의 강점 검증
- **crease curvature:** 힌지 `U` 곡률 반경 vs 이론 `R = t/(2ε_max)` — mesh grading 효과 정량

---

## 7. 우선순위 로드맵 (Abaqus 초월까지 3단계)

### Phase 0 — 즉시 (1–2주, 코드 100줄 내, 위험 최소)

- [ ] `MeshGradingConfig.uniform` 기본 `False` + `hinge_half_gap`/`hinge_pivot_x` 정합성 assert
- [ ] `PET → Q4_EAS` 기본 복귀 (1줄, `fold_model_config.py` )
- [ ] Prescribed 요소 스킵 (AGENTS.md §4.13, `dynamic.py` `_assemble_multi_material_batch` )
- [ ] Tie AL 2회 래퍼 + `k` 자동 스케일링
- [ ] `stabilization` 배선 (정의→호출)

### Phase 1 — 단기 (1개월, 성능·정확도 동시)

- [ ] PARDISO symbolic 재사용 + equilibration 캐시
- [ ] Convergence 3-Tier AND 부활(q_avg 안정화) + energy merit
- [ ] Generalized-α 승격(`α_m, α_f`)
- [ ] 체적 `U` 통일 + 2-term Prony
- [ ] `verification` 대변형 3종 추가

### Phase 2 — 중기 (2–3개월, Abaqus 동등→초월)

- [ ] Numba 전체 요소 커버 + GPU JAX
- [ ] Mortar tie + 재투영
- [ ] Arc-length (Riks) + Broyden
- [ ] EAS-7/ANS + 진짜 Q1P0 hybrid
- [ ] Iterative AMG fallback

---

## 8. 참고 — 최신 이론·Abaqus 문서 대조

- **시간적분:** Hilber-Hughes-Taylor (1977), Chung & Hulbert (1993) Generalized-α, Jansen et al. (2000). Abaqus Theory Guide §2.4.2 `*DYNAMIC`, §2.4.4 `*VISCO`.
- **EAS:** Simo & Rifai (1990) IJNME 29, Simo & Armero (1992) CMAME 95, Andelfinger & Ramm (1993) EAS-7. Abaqus Theory §3.2.4 Incompatible modes (CPE4I).
- **Corotational:** Wriggers (2008) Nonlinear FEM Ch.4, Rankin & Nour-Omid (1988). dT/du는 Felippa (2001) 정리.
- **F-bar/volumetric:** de Souza Neto et al. (1996) IJSS 33, Hughes (1980) IJNME 15. Abaqus Theory §3.2.3 Hybrid elements.
- **점탄성:** Simo (1987) CMAME 60, Holzapfel (2000) Ch.6 Flory split. Abaqus Theory §4.8.1 Viscoelasticity, PRF.
- **Tie/Mortar:** Wriggers (2006) Computational Contact Mechanics, Puso & Laursen (2004) Mortar. Abaqus Theory §5.2.3 Tie constraints.
- **선형해석:** Duff & Reid (1983) multifrontal, Schenk & Gärtner (2004) PARDISO. equilibration은 Ruiz & Reid (MC64).
- **수렴:** Greenstadt (1967) saddle-point, Dennis & Schnabel (1996) §6.4.

---

## 9. 맺음말

ex13은 **모델 소스 통일·RBE2 condensation·PARDISO equilibration·Flory+Simo 점탄성** 네 축에서 이미 Abaqus 이론 수준을 충족한다. 남은 격차는 거창한 이론 교체가 아니라 **수렴 판정의 엄밀화, tie의 AL화, assembly의 불필요 연산 제거** 세 가지 엔지니어링 완성도에 있다. Phase 0만으로 동일 mesh·동일 하드웨어에서 Abaqus/Standard 대비 **1.5× 속도 + 동등 정확도** 를 기대할 수 있으며, Phase 1까지 마치면 **명시적 우위** 를 주장할 수 있다. 코드는 건드리지 않았으니, 이 리뷰를 `dev_log` 합의안으로 삼아 Phase 0부터 순차 구현을 권장한다.
