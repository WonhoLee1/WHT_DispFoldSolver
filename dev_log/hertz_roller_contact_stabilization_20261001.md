# Hertz 롤러 접촉 벤치마크 안정화 보고서 (2026-10-01)

대상: `benchmark_inp/phase2_methodology/01_hertz_roller_plate_3d.inp`
(롤러–평판 3D 접촉, μ=0.2, INDENT → SLIDE 2단계)
출발점: `contact_stabilization_plan.md` (Fix A/B/C 적용 후에도 수렴 실패 상태)

## 1. 결론

- INDENT와 SLIDE 두 step 모두 컷백 0으로 완주하며, CLI summary는 `converged: true`임.
- 원인은 솔버가 아니라 **deck 결함**이었음. roller C3D8I 요소 240개가 전부 좌수계(음체적)였음.
- 계획서의 Fix A/B/C는 셋 다 불필요했으므로 모두 원복함.

## 2. 근본 원인

- `gen_01_hertz_roller_plate_3d.py`의 roller 절점 순서가 (θ, r, z)임.
  - 롤러 하단에서는 +θ ≈ +x, +r ≈ −y 방향이므로 이 순서는 좌수계가 됨.
  - 그 결과 det J가 −0.092 ~ −0.116으로 나옴. plate 600개는 +0.125로 정상임.
- 음체적 요소는 강성이 음정부호가 됨. 접촉으로 plate와 결합하면 전체 시스템이 부정부호가 되어 Newton이 수렴할 수 없음.
  - 그래서 dt를 1e-5까지 줄여도 소용없었음.
- 결정적 측정은 y방향 절점 컴플라이언스 (K⁻¹)ᵢᵢ임.
  - roller 절점: −1.95e-5 mm/N (접촉이 없어도 음수)
  - plate 절점: +2.07e-5 mm/N
- 솔버가 음체적 초기 요소를 에러 없이 받아들였기 때문에, 결함이 수렴 문제로 위장됨.

## 3. 계획서 Fix A/B/C 판정

구 deck에서 A/B 측정한 결과:

| 구성 | INDENT 정체 지점 | SLIDE 정체 지점 |
|---|---|---|
| Fix A+B+C | t=0.0486 | t=0.0100 |
| Fix B+C만 (A 롤백) | t=0.1702 | t=0.0400 |

- **Fix A** (frozen 절점 단방향 스프링): 오히려 악화시킴. freeze가 제거하던 kink를 다시 들여옴.
- **Fix B** (SDI 후 β=0.5), **Fix C** (r_tol 5%→8%, du 기준 완화):
  - 수정된 deck에서는 이 수정 없이도 완주함.
  - Fix C는 모든 접촉 해석의 수렴 기준을 완화하는 변경이므로 원복함.
- 계획서의 전제 `k_contact = 1e8`도 틀렸음.
  - deck에 penalty 지정이 없어서 기본값 10·k_ref = **2.015e6 N/mm**가 사용됨.

## 4. 변경 사항

| 파일 | 내용 |
|---|---|
| `benchmark_inp/phase2_methodology/gen_01_hertz_roller_plate_3d.py` | roller 연결성의 k/k+1 층을 교환하고 deck을 재생성함. det J는 0.088~0.120이 됨. S5(외면 r=10)와 키워드 라인은 그대로임 |
| `dispsolver/solver3d/dynamic3d.py` | `_check_initial_hex_orientation()` 추가: 8절점 hex의 2×2×2 Gauss점 중 det J ≤ 0이 있으면 `ValueError` (Abaqus `ErrElemVolSmallNegZero`에 대응) |
| 〃 | `_is_symmetric()` 추가: K가 비대칭이면 PARDISO를 LU(mtype=11)로 전환함 |
| 〃 | Fix B/C를 계획 이전 값으로 원복함 |
| `dispsolver/solver/pardiso_manager.py` | mtype이 바뀔 때 이전 PARDISO 핸들의 `free_memory` 호출 (native 메모리 누수 방지) |
| `dispsolver/constraint3d/surface_contact3d_deformable.py` | 법선 부분의 정확한 접선을 기본값으로 함 (`consistent_tangent=True`) |
| 〃 | 마찰 접선은 기존 대칭 Ktt를 유지하고, 정확한 마찰 접선은 `consistent_friction_tangent` 옵션으로만 둠 |
| 〃 | Fix A를 원복하고 측정 근거를 주석으로 남김 |
| `tests/test_3d_negative_volume_guard.py` | 7개 테스트 |
| `tests/test_deformable_contact_consistent_tangent.py` | 7개 테스트 |
| `dev_log/phase2_methodology_benchmark_audit_20260929.md` | §3.3 작성, 표에 01번 "통과" 표시 |

### 4.1 비대칭 K를 Cholesky로 풀던 결함

- `PardisoNonlinearSolver(mtype="auto")`는 대칭 여부를 모를 때 mtype=2를 씀. 이 경우 상삼각만 사용하는 Cholesky가 됨.
- 비대칭 K에 이를 적용하면 **예외 없이 해가 818% 틀림**. LU로 풀면 2e-10임.
- reviewer 측정 결과:
  - C3D8, C3D8I, C3D8R, C3D8H, SOLID_SHELL은 대칭(약 3e-16)이라 계속 Cholesky를 씀.
  - **C3D8_FBAR는 17~21% 비대칭이라 이제 LU를 씀.** 이전에는 조용히 틀린 행렬로 풀리고 있었음.

### 4.2 접촉 접선

- 법선 부분에서 추가로 미분한 항:
  - 법선 회전 dn/du
  - gap의 (xs − xm)·dn/du 항
- FD 대비 오차 (단위 테스트 fixture 기준):

| 분기 | 기존 블록 | 정확한 블록 |
|---|---|---|
| 무마찰 | 3.6% | 1.1e-10 |
| stick | 3.2% | 1.0e-10 |
| slip | 41% | 1.2e-10 |

- 정확한 마찰 접선은 FD 검증을 통과했지만 SLIDE를 t=0.213에서 발산시킴. 활성 집합이 iter마다 6→2→10으로 요동침. 그래서 opt-in으로만 둠.
- reviewer가 모든 f_mag 분기(frozen bilateral, AL, chatter gate, Nonlinear/Soft law)에서 법선 블록을 FD와 비교함. 결과는 1e-10~2e-10임.
- 이 deck에서 기본값 변경의 효과는 제한적임:
  - SLIDE 최종 잔차: 85 → 32 N
  - 엄격 기준(0.5%) 사용 시: 91 vs 94 iter
  - 대가: 모든 변형체 접촉 모델이 LU를 쓰게 됨 (메모리·시간 증가)

## 5. 최종 결과

실행 명령: `python -u -m dispsolver.cli.run_inp benchmark_inp/phase2_methodology/01_hertz_roller_plate_3d.inp --dim 3 --elem-jit numba`

| 항목 | INDENT | SLIDE |
|---|---|---|
| 완주 / 컷백 | t=1.0, 22 inc, 0 | t=1.0, 22 inc, 0 |
| Newton iter/inc | 3 | 3 |
| 수락 분기 | 전부 residual | 전부 residual |
| 최악 수락 잔차 (r/‖반력‖) | 0.37% | 9.2% (Inc 3, stick→slip 전이) |
| RF (roller top) | (−30.5, −12827.1, 0.8) N | (2163.4, −11864.8, 0.3) N |
| 총 접촉력 | 12823.6 N (RF_y와 일치) | 11898.1 N |
| 최대 침투 | 1.304e-3 mm = p_max/k (2627.7 N / 2.015e6), 압입량의 2.6% | 1.210e-3 mm |
| 마찰 상태 | stick 6/6 | **slip 6/6** |
| 로컬 마찰비 Σ‖f_t‖/Σp | — | **0.199995** (= μ) |
| \|RF_x\|/\|RF_y\| | — | **0.1823** (\|·−0.2\| = 0.0177 < 0.02) |

- 전역 반력비(0.1823)가 μ보다 작은 이유: 접촉 법선이 약 1° 기울어졌기 때문임 (Σf_n,x = 222 N).
  - slave 절점이 master 절점 위에 정확히 겹친 채 한쪽 face에만 페어링된 이산화 효과임.
  - 로컬 좌표계에서는 Coulomb 법칙이 정확히 성립함.
- 최대 침투는 penalty 입력값의 함수이며 솔버 정확도 지표가 아님. 더 줄이려면 deck에서 penalty를 지정해야 함.
- reviewer가 엄격 기준으로 재검증함 (f_tol 0.5%, correction 분기 사실상 차단):
  - 두 step 모두 컷백 0으로 완주, SLIDE 최종 잔차 9.4 N.
  - 따라서 보고한 답은 느슨한 수락 기준이 만든 결과가 아니라 실제 평형임.

## 6. 회귀 검증

- **신규 테스트**: 14/14 통과.
- **전체 `tests/`**: 908 passed, 10 failed, 19 skipped, 26 xfailed.
  - 10개 실패는 모두 이번 변경(guard, LU 전환, 일관 접선)을 몽키패치로 끈 상태에서도 동일하게 실패함. 따라서 **기존 실패**임:
    - `test_3d_c3d10m` integration
    - `test_3d_c3d6` integration
    - `test_abaqus_contact_api` two_cubes
    - `test_deformable_soft_nonlinear_contact` AL
    - `test_surface_to_surface_contact` AL two-block
    - `test_concrete_dp` tension softening
    - `test_field_output` C3D8 / C3D8_FBAR stress recovery
    - `test_traction` ×2
  - 처음 5개(접촉 계열)는 세션 시작 상태 코드(Fix A/B/C 적용)로도 실패함을 따로 확인함.
  - `test_nodal_transform.py::test_abaqus_deck_transform_end_to_end`는 numba 조립 중 네이티브 access violation으로 크래시함. 이번 변경을 꺼도 똑같이 크래시하므로 제외하고 실행함.
- **`verification.run_all`**: 13/14 통과. 실패한 Gulati 2-point bending은 2D `DynamicSolver` 경로라 이번 변경과 무관함.
- **deslop 이후 Hertz CLI 재실행**: 두 step 완주, 컷백 0, `converged: true`.

## 7. 남은 위험 / 후속 과제

1. **3D solver 수락 기준이 느슨함** (이번 작업 이전부터 있던 문제)
   - `_solve_step_with_hooks` / `solve_step`의 수락 조건: `rel_r < 5%`에서 분모가 반복 중 최대 f_int임. 또는 `rel_f < 5%`, `rel_du < 0.01`.
   - 이 때문에 반력의 약 9% 잔차로도 증분이 수락됨.
   - 조용히 틀린 답이 나올 위험이 가장 큰 곳임. AGENTS §4.7의 경고를 감안해 따로 판단해야 함.
2. **F-bar 모델의 LU 전환은 벤치마크로 확인하지 않았음**
   - `benchmark_element/benchmark_3d_mechanics.py`를 다시 실행해 볼 것.
3. **음체적 검사는 8절점 hex만 대상임**
   - C3D4, C3D10, C3D6, C3D20의 방향 오류는 여전히 검출되지 않음.
4. **`DeformableSurfaceContactConstraint3D`의 `tangential_stiffness_ratio` 기본값이 HEAD 대비 1.0 → 0.1로 바뀌어 있음**
   - 이번 세션 이전부터 있던 uncommitted 변경이라 건드리지 않음. 의도된 변경인지 확인 필요함.
5. **SLIDE의 Newton 수렴이 선형임** (iter당 약 0.67배 감소)
   - 대칭 마찰 블록을 쓰기 때문임.
   - 정확한 마찰 접선은 수렴을 불안정하게 만듦. 개선하려면 stick/slip 전이 처리(SDI 포함 여부 등)부터 검토해야 함.
6. **커밋하지 않았음**
   - `benchmark_inp/phase2_methodology/`는 untracked 상태임.

## 8. 교훈

- 작은 탄성 접촉 모델이 수렴하지 않으면, 허용오차나 접선을 조정하기 전에 **요소 방향(det J)과 절점 컴플라이언스 부호**부터 확인할 것.
- Newton 로그의 `act=N` 카운트만 봐서는 활성 집합 멤버가 바뀌는 것을 알 수 없음.
- 증거 없는 "Fix"(문헌 인용만 있는 A/B/C)는 A/B 측정 전까지 채택하지 말 것.
