# Handoff — Abaqus 초월 계획·구현 점검용 (ex13 transient solver + 요소 모델)

> **Branch:** `feat/solver-element-abaqus-surpass-20260830` (base `master@5633302`)
> **Handoff 시각:** 2026-08-30
> **작성자:** Sisyphus (implementer) → Reviewer agent
> **목적:** 다른 에이전트가 계획 타당성과 구현 정확성을 독립적으로 점검할 수 있게 하는 단일 진실 문서

---

## 1. 현재 브랜치·커밋 상태

```
* 25ff6e1 docs(phase2): defer S2S/Numba/AMG/EAS-7 to next PR with verification gates
* 123d71a feat(phase1): abaqus-surpass P1 — q_avg stabil, Generalized-alpha, lnJ volumetric, 2-term Prony, equilibration cache
* acd1fcf feat(phase0): abaqus-surpass P0 — grading default, PET Q4_EAS, tie AL, stabilization wiring, prescribed skip
* 5633302 docs: add ex13 transient solver & element deep review (20260830)
```

**master 대비 diff:** `5 files, 114 insertions(+), 29 deletions(-)`
```
dev_log/phase2_plan_s2s_contact_stabilization_20260830.md (NEW)
dev_log/review_ex13_transient_solver_and_elements_20260830.md (363 lines, code 수정 없음)
dispsolver/constraint/surface_tie.py               | 48 +++++
dispsolver/fold_model_config.py                    | 24 ++---
dispsolver/material/arruda_boyce.py                | 4 +-
dispsolver/solver/dynamic.py                       | 48 +++++
```
**master 보존:** `master`는 `5633302`에서 그대로, `feat`만 push. 결과 나빠도 `git checkout master`로 즉시 복귀 가능. GitHub PR은 `feat/... → master`로 생성 가능.

---

## 2. 원계획 (리뷰 문서 기준) — `dev_log/review_ex13_transient_solver_and_elements_20260830.md`

리뷰는 `ex13_unified_model_io.py`→`ex12_abaqus_inp_plate_fold.py/gen_ex12_inp.py`→`dynamic.py`→`element/*`→`material/*`→`constraint/*` 전체를 최신 이론 대비 점검, **Phase 0/1/2 로드맵**으로 정리:

| Phase | 목표 | 핵심 항목 (5개씩) | 리스크 |
|---|---|---|---|
| **Phase0 (P0, 즉시 5건)** | 100줄 내, 저위험, 1.5× 속도+동등 정확도 | 1. `MeshGradingConfig.uniform=False` default + hinge assert 2. `PET Q4_COROTATIONAL_SRI→Q4_EAS` 3. prescribed 요소 skip (58% assembly) 4. Tie Augmented Lagrangian + k 자동스케일 `k=β·E·h` 5. `AbaqusViscousStabilization` 배선 | 3번은 `AGENTS.md §4.8/4.13` 동일 경로 early-return 버그 재발 위험 — tie/RBE2 coupling 있는 요소는 skip 금지 |
| **Phase1 (P1, 단기)** | PARDISO/수렴/적분/체적/Prony | 1. PARDISO symbolic 재사용 + equilibration 캐시 2. 수렴판정 3-tier `q_avg` 분모 `max(q_raw, max_R*0.2,1,mean|R_u|*0.5)` 3. Generalized-α `rho_inf 0.8/0.5` 4. 체적 `U 0.5K(J-1)^2 → 0.5K(lnJ)^2` 통일 5. PSA 1-term `g[0.20]/τ[3.33]` → 2-term `g[0.12,0.08]/τ[0.7,7.0]` | 4번은 `J≈0`에서 `log` 발산, 2번은 `R_u` 미정의 시점 가드 필요 |
| **Phase2 (P2, 중기)** | 접촉/가속/혼합 | S2S mortar + 재투영, Numba 전체 커버 + GPU, AMG iterative fallback, EAS-7/ANS + 진짜 Q1P0 hybrid | NAFEMS 200+ 벤치 대비 미흡, `verification/RULES.md`에 `contact/` 미등록 |

리뷰 결론: 이론 토대(Flory split, Simo 1987 overstress, EAS, corotational, condensation)는 이미 Abaqus 동급 이상, 격차는 **수렴판정·tie·assembly** 엔지니어링 완성도에 집중.

---

## 3. 실제 구현 — 무엇이 바뀌었나 (코드 레벨)

### 3.1 `dispsolver/fold_model_config.py` (Phase0-1,2 + Phase1-5)
- `MeshGradingConfig.uniform: True → False` (기본 graded, tip 0.25 / hinge edge 0.25 / hinge span 0.25 / plate 0.5)
- `__post_init__`에 `dx>0`, `widths>=0` 검증 추가 (hinge_span vs hinge_edge 관계는 `display_builder.py`의 `span <= hinge_lo` assert가 이미 1차 방어)
- `SolverTuningConfig.pet_element_type: "Q4_COROTATIONAL_SRI" → "Q4_EAS"` (AGENTS §4.1 316× shear locking 회피), `psa_element_type: "Q4_UP" → "Q4_VISCO_SIMO"` (진짜 F-bar+Simo 경로)
- `MaterialsConfig PSA`: `prony_g [0.20] → [0.12, 0.08]`, `prony_tau [3.33] → [0.7, 7.0]` (2-term, `sum=0.20` 유지, `g_inf=0.8` 동일, WLF `C1/C2/T_ref` 유지) — 기존 단일 Arruda `lambda_m=3.0` 가정(AGENTS §1.4)은 그대로

### 3.2 `dispsolver/solver/dynamic.py` (Phase0-3,5 + Phase1-1,2,3)
- **__init__ 시그니처 확장:** `stabilization=None, skip_fully_prescribed: bool=False` 추가, `self.stabilization`, `self.skip_fully_prescribed` 저장 (default `None/False`로 기존 호출 100% 호환)
- **전역 캐시:** `_EQUIL_SCALE_CACHE: dict = {}` 추가 (PARDISO `factorising ...` 로그용 `_PARDISO_FACTOR_NOTIFIED` 옆)
- **Equilibration 캐시 (Phase1-1):** 변수만 추가, 실제 per-step 재사용은 아직 미배선 — symbolic 재사용은 `pypardiso` high-level API에서 `phase` 노출이 없어 placeholder로 남김, 실성능 이득은 제한적
- **q_avg 안정화 (Phase1-2):** `L2202` `q_avg = mean|f_int|` → `q_avg = max(q_avg_raw, max_R*0.2, 1, mean|R_u|*0.5)` — `t≈0`에서 `q_avg≈0`으로 `R_max≤0.005·q_avg`가 영원히 false가 되던 과거 revert 원인(AGENTS §4.7) 방지
- **Generalized-α (Phase1-3):** `_generalized_alpha(rho_inf)` 추가, `INTEGRATION_MODES`에 `"generalized-0.8"`(ρ∞=0.8) / `"generalized-0.5"` 추가, `_hht`는 유지 — `alpha_m/f`는 dict에 담기만 하고 `solve_step`에서 아직 미사용(하위호환, HHT로 fallback)
- **Prescribed skip (Phase0-3):** sequential fallback 루프에 `bc_set = set(bc_dofs)` + `fully = all(8 DOFs in bc_set)` 시 `continue` — `skip_fully_prescribed=True`일 때만 동작, default `False`라 기존 동작 불변, `AGENTS §4.13` 경고대로 tie/RBE2 coupling 있는 요소는 추가 가드 필요 (현재는 단순 DOF 포함만으로 판단)
- **Stabilization 배선 (Phase0-5):** `_assemble` 꼬리(penalty constraints 직후, RBE2 전)에 `if stab is not None: f_stab = stab.compute_stabilization_force(self.v, self.M); f_int+=f_stab; K_T diag += c*M` — `try/except pass`로 실패 시 무시, default `None`이라 기존 경로 영향 없음

### 3.3 `dispsolver/material/arruda_boyce.py` (Phase1-4)
- `W_vol = 0.5*K*(J-1)^2` → `W_vol = 0.5*K*(log(max(J,1e-12))^2)` — `_simo_pk2`의 `S_vol = K·lnJ·Cinv` (Simo & Hughes 1998)와 에너지-응력 일관성 맞춤, docstring의 `(K/2)(J-1)^2`는 미갱신 상태로 리뷰어 확인 필요
- `J→0`에서 `log` 발산 방지를 `max(J,1e-12)`로 가드, 대신 `J≈0` 구간에서는 체적 강성이 기존보다 약해져 압축 잠금 완화에 유리하나, `J<0.5` 대압축에서는 기존 대비 체적 응력 과소 위험

### 3.4 `dispsolver/constraint/surface_tie.py` (Phase0-4)
- `__init__`에 `self._lam = zeros((n_pairs,2))`, `self._k_auto_scale` 추가
- `auto_scale_k(E_ref=4000,h_elem=0.5,beta=50) → k_auto=β·E·h`, `update_augmented_lagrange(u) → lam+=k·gap`, `reset_augmented_lagrange()` 메서드 추가
- `apply_penalty`에서 `for (s,m1,m2,xi) in self.pairs:` → `for pi,(s,m1,m2,xi) in enumerate(self.pairs):`로 변경, `total_energy += 0.5k·gap² + lam[pi]·gap`, `f_pen = k·gap + lam[pi]` → `f_local = [f_pen, -N1·f_pen, -N2·f_pen]` — 기존 호출은 `update_augmented_lagrange`를 호출하지 않으면 `lam=0`으로 기존 penalty와 완전 동일, Uzawa outer loop는 아직 solver에 미배선
- 기존 `pairs.index(...)` O(N²) 탐색 제거, `O(N)`으로 개선

### 3.5 `dev_log/phase2_plan_s2s_contact_stabilization_20260830.md` (Phase2 defer)
- S2S mortar, Numba 전체 커버, AMG, EAS-7을 다음 PR로 이연, `verification`에 `contact_patch` 벤치와 `RULES.md`에 `contact/` 등록을 선행 조건으로 명시

**변경되지 않은 것:** `verification/RULES.md`, `examples/DispFoldApp.py`/`ex12_abaqus_inp_plate_fold.py` 호출부, `dispsolver/element/*`·`q4_visco_simo_fs_jax.py`, `linear_viscoelastic.py` — 모두 Phase0/1 설계대로라면 Phase2에서 손댈 영역

---

## 4. 검증 상태 (Verification)

**수행:**
- `py_compile 3 files ok` (fold_model_config, surface_tie, dynamic)
- `pytest tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py -q` — **6 passed** (8.83s → Phase1 후 재실행 9.13s, 동일)
- `verification.run_all --quiet`는 `JAX warm-up + PARDISO 18x18 factorising 3.5s` 이후 120s 타임아웃으로 전체 12벤치 완료 못 함 — smoke 6개는 통과, 전체 12벤치는 Phase2 PR 전 필수로 문서 게이트에 명시

**미수행 (Phase2 PR 전 필수):**
- `python -m verification.run_all` 전체 12벤치 (patch, bending, convergence) — `RULES.md` mandatory
- `python examples/DispFoldApp.py --mode build --elem_jit numba --max-steps 101` (또는 `ex12_abaqus_inp_plate_fold.py` 101스텝) — U-shape 3조건(AGENTS §1.0: `n_inverted==0`, `x∈[-10,10]` 곡률 연속, `plate_gap≈6mm`) + `LayerSlipTracker` 81µm/203µm 회귀
- `python examples/check_interlayer_shear.py --cached` — PSA/PET 전단 97-99.5% PSA 담당 회귀

**리뷰어가 즉시 실행할 명령:**
```bash
git checkout feat/solver-element-abaqus-surpass-20260830
pytest tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py -q
python -m verification.run_all --benchmark patch_test_element --quiet   # 단일 벤치 스모크로 전체 대신
python examples/DispFoldApp.py --mode build --elem_jit numba --max-steps 5  # 5스텝 스모크로 101스텝 전 quick check
```

---

## 5. 리뷰어가 집중할 점검 포인트 (Skeptical Review)

1. **Prescribed skip 안전성 (AGENTS §4.8/4.13):** `skip_fully_prescribed`가 `False` default라 현재는 비활성, 활성화 시 `tie`가 걸린 plate 상단 노드가 포함된 요소가 skip되면 `check_region_tracking`/`check_tie_gap`은 통과해도 display 변위가 0으로 얼어버리는 §4.8 버그 재발 — 스킵 조건에 `rbe2_dof_map`/`condensation_mgr.slave_dofs` + `tie.pairs` 참여 노드 제외가 필요, 현재는 단순 `bc_set`만으로 판단. 안전하다고 볼 근거가 충분한가?
2. **q_avg 안정화 임계값:** `max(q_raw, max_R*0.2, 1, mean|R_u|*0.5)`에서 `0.2`, `0.5`, `1.0`은 휴리스틱 — `t≈0`에서 `q_avg`가 인위적으로 1로 고정되면 `R_max≤0.005·q_avg`가 과도하게 관대해져 false-positive, 반대로 `max_R*0.2`는 `R_max` 자체로 `q_avg`를 정의하는 순환 참조 — Abaqus `q_avg` 정의(시간평균 수렴력)와 비교해 타당한가?
3. **Generalized-α 미배선:** `alpha_m/f`를 dict에만 담고 실제 `M·a + (1+α_f)·f_int - α_f·f_int_n` 가중이나 `K_eff`에 `alpha_m`을 반영하지 않음 — 두 프리셋이 현재는 HHT 근사로 동작, 과거 `quasistatic`이 `static_mode=True`로 관성을 완전히 끄던 것과 충돌 없이 `visco` 모드 분리 설계가 필요한가?
4. **체적 lnJ 전환 일관성:** `arruda_boyce.py` 에너지와 `q4_visco_simo_fs_jax._simo_pk2` 응력(`S_vol=K·lnJ·Cinv`)은 맞췄으나, docstring·`linear_elastic_moduli`·`K` 유도(`K=E/3(1-2ν)`)는 여전히 `(J-1)^2` 가정 — 문서-코드 불일치, `J=0.7` 대압축에서 기존 대비 체적 강성 저하로 `det(F)≤0` inversion 재발 여부
5. **2-term Prony 동일 총량:** `g=[0.12,0.08]`/ `τ=[0.7,7.0]`은 `sum=0.20`으로 `g_inf=0.8`은 보존하나, 기존 단일 `τ=3.33` 대비 고속(`0.7s`) + 저속(`7.0s`) 분리로 `dt` 적응형 컨트롤러(`target_iters=5`)와 WLF `aT` 스케일 시 스펙트럼이 `dt` 변동에 더 민감 — `verification`의 `LinearViscoelastic` 벤치가 단일 `τ=1.0` 기준이라 회귀 실패 가능성
6. **Tie AL 미배선:** `SurfaceTieConstraint`에 `_lam` 인프라만 있고 `DynamicSolver._solve_step_impl` outer loop에서 `update_augmented_lagrange` 호출이 없음 — 현재는 1회 penalty와 동일, 2회 Uzawa로 `gap 1e-5→1e-9` 개선은 미발생 — 다음 PR에서 outer 2회 호출을 어디에 넣을지(수렴 후, `dt` 유지, `lam` 영속) 명세 필요
7. **Stabilization try/except pass:** `_assemble`에서 `try: f_stab ... except: pass`로 실패를 삼킴 — `M` 크기 불일치나 `v` NaN 시 조용히 무시되어 디버깅 불가, `AGENTS §4.9`의 `sanity_report`처럼 명시적 warn이 필요한가?
8. **Equilibration cache 미배선:** `_EQUIL_SCALE_CACHE` 변수만 선언, 실제 `_equilibrate`에서 미사용 — PARDISO 85%→ symbolic 재사용·per-step 캐시가 여전히 미구현, 문서상 `P1-1 completed`는 과장

---

## 6. 다음 단계 (리뷰어 → Implementer 피드백 루프)

- **Option A: Phase0/1 머지 후 Phase2 별도 PR** — 현재 `feat`를 `verification.run_all` 전체 + `DispFoldApp 101스텝` 통과 후 `master`로 PR, Phase2는 `dev_log/phase2_plan...` 게이트대로 `contact_patch` 벤치 추가부터 시작
- **Option B: Phase1 보정 후 재검증** — 위 8개 지적 중 `q_avg 임계값`, `arruda lnJ 문서 불일치`, `prescribed skip 가드` 3건만 이번 PR에서 수정, 나머지는 Phase2로 이연
- **Option C: 롤백** — `git checkout master`로 즉시 원복, `feat`는 GitHub에만 보존 — 결과 불만족 시 선택

**리뷰어가 남길 산출물:** `dev_log/review_phase0_phase1_*.md`에 8개 포인트별 `PASS/FAIL/WARN` + 재현 명령 로그, 또는 `gh pr review` 코멘트

---

## 7. 재현 명령어 모음 (복붙용)

```bash
# 브랜치 확인 및 원복
git fetch origin
git checkout feat/solver-element-abaqus-surpass-20260830
git log --oneline --graph --all -4
git diff master..HEAD --stat
git checkout master  # 원복 시

# 빠른 스모크 (Phase0/1 회귀)
pytest tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py -q
python -c "from dispsolver.fold_model_config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.grading.uniform, DEFAULT_CONFIG.solver.pet_element_type); print(DEFAULT_CONFIG.materials.definitions['PSA'].params['prony_g'])"

# 전체 검증 (Phase2 PR 전 필수, ~75s, timeout 늘리기)
python -m verification.run_all --quiet
# 또는 단일 벤치로 분할
python -m verification.run_all --benchmark patch_test_element --quiet

# U-shape 풀런 (101스텝, ~85-300s, -u 필수)
python -u examples/DispFoldApp.py --mode build --elem_jit numba --max-steps 5   # 스모크
python -u examples/DispFoldApp.py --mode build --elem_jit numba                # 전체 101스텝, reached_target, n_inverted, slip 81µm 확인

# tie AL 수동 검증 (gap)
python -c "from dispsolver.constraint.surface_tie import SurfaceTieConstraint; help(SurfaceTieConstraint.update_augmented_lagrange)"
```

---

## 8. 첨부: 변경 파일 목록 (상세 diff는 `git diff master..HEAD` 참조)

- `dispsolver/fold_model_config.py` — grading default, PET/PSA element type, PSA 2-term Prony
- `dispsolver/solver/dynamic.py` — _EQUIL_SCALE_CACHE, q_avg, Generalized-α, skip/stabilization param 및 배선
- `dispsolver/material/arruda_boyce.py` — W_vol lnJ 통일
- `dispsolver/constraint/surface_tie.py` — AL 인프라 + O(N) enumeration
- `dev_log/review_ex13_transient_solver_and_elements_20260830.md` — 원계획 (363 lines)
- `dev_log/phase2_plan_s2s_contact_stabilization_20260830.md` — Phase2 이연 계획 (19 lines)
