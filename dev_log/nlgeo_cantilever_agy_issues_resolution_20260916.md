# NLGEO Cantilever: agy 잔류 이슈 3건 해소 + SOLID_SHELL 분리 입증 (2026-09-15/16)

## 1. agy 로그 3건 판정

| agy 보고 | 재현 결과 | 판정 |
|---|---|---|
| C3D8I_CR `rows_topo is None` | 현 트리에서 정상 수렴 (uy=7.647085, cutbacks=1). `DynamicSolver3D.__init__:120`은 무조건 `_setup_numba_topology()` 호출, `rows_topo=None`은 빈 메쉬 때만. 벤치마크는 호출마다 새 solver 생성 — 재사용 가설 기각. | **STALE** — 구 코드 상태 로그, 현 트리 재현 불가 |
| C3D8_CR uy=1.388 (75% 오차) | coarse 75.0% → fine(nx=20) 23.7%, cutbacks 0. 메쉬 정제 수렴. k=1 정품 CR 커널. | **진품 전단잠김**, 버그 아님 |
| SOLID_SHELL 실행 중 | coarse uy=7.647085135940194, C3D8I_CR와 7e-14 일치 | 아래 §2 |

## 2. SOLID_SHELL ≡ C3D8I_CR (brick에서 1e-13) — 예상된 동등, 디스패치 버그 아님

- `solid_shell_numba.py`는 독립 구현 (ANS tying + EAS-9 응축) 확인. EAS 복사 아님.
- 직육면체에서 MITC tying은 표준 전단과 정확히 일치 → 동일 EAS-9와 비트일치 당연.
- C3D8==C3D8_CR 동일값도 동일 사유: NLGEOM에서 둘 다 k=1 동일 커널로 승격 (dynamic3d.py:270-275, 설계 의도).
- 기존 YZ 패치 5.19e-6 honest note가 실제 분리기록이었음.

## 3. 왜곡 분리실험 (`benchmark_element/run_distorted_nlgeo_separation.py`, 신규)

사다리꼴 왜곡 (sin envelope, root/tip 평면 유지, 진폭 0.15h/0.30h/0.45h):

| 진폭 | C3D8I_CR uy | SOLID_SHELL uy | 분리 |
|---|---|---|---|
| clean | 7.647085 | 7.647085 | 7e-14 |
| x1 | 7.316147 | 7.316147 | 6.1e-07 |
| x2 | 6.837669 | 6.837667 | ~2e-06 |
| x3 | 6.287587 | 6.287581 | 5.6e-06 |

분리량 왜곡에 단조 증가 (×3 → ×9) → 정식화 신호, 노이즈 아님. **커널 분리 입증, 디스패치 정상.**
부수 관측: 왜곡이 EAS계열을 이론쪽으로 연화 (37.7→13.2%), C3D8_CR은 추가 잠김 (75→83%).

## 5. 이론해 정밀화 + uy/ux 라벨 스왑 수정 (2026-09-16 후속)

- `BisshoppDruckerElastica`를 특이 적분(`quad` 1/sqrt) → 후방 IVP shooting (`solve_ivp` DOP853, rtol=1e-13) + 사다리꼴 적분으로 교체. `IntegrationWarning` 소거 (원인: θ_tip=81.96°로 π/2 근접, 1/sqrt(cos θ_tip) 증폭 + `max(d,1e-30)` 평탄부).
- 구 이론값(5.552312/8.107242)과 신 이론값 소수점 6자리 일치 — 구 수치는 맞았고 경고만 문제였음.
- **라벨 스왑 발견**: `solve_at_load` 반환순서 (θ, x_short, y_def)인데 `run_nlgeo_solid_shell_vs_best.py:35,128`과 `run_distorted_nlgeo_separation.py:91`이 2번째를 uy로 취함. 참값은 횡처짐 **uy=8.107241**, 축단축 **ux=5.552312**. §3 포함 기존 모든 err%는 단축 기준이었음 — 수정 후:
  - clean C3D8I_CR/SOLID_SHELL: 37.7% → **5.68%** / C3D8_CR: 75.0% → **82.9%**
  - x1 왜곡 C3D8I_CR/SOLID_SHELL: 31.8% → **9.76%** / C3D8_CR: 78.5% → **85.3%**
## 6. 2-turn 롤업 비교 (`run_moment_rollup_comparison.py`, 신규)

회전처방 4π, nx=20, 40스텝. 성공=팁 원점 복귀 오차:

| 요소 | iters 평균/최대 | nonconv | wall | tip 복귀오차 |
|---|---|---|---|---|
| C3D8I_CR | 13.25 / 25 | 10/40 | 46.1s | 0 |
| SOLID_SHELL | 13.25 / 25 | 10/40 | 43.0s | 0 |
| C3D8_CR | 8.15 / 17 | 0/40 | 4.8s | 0 |

- 셋 다 2-turn 형상 완수. SHELL≡EAS 동일 수렴경로 (brick 동등 재확인).
- **SHELL이 더 빠르지 않음** — C3D8_CR이 수렴-rate·wall 모두 우위 (EAS α-응축이 대회전에서 뉴턴 rate 깎음).
- 단, 이 테스트는 형상 처방이라 정확도 변별 없음. 진짜 정확도(반력 모멘트)는 `benchmark_pure_moment_rollup.py`형 하중처방에서 가려야 — SHELL 미실행.
## 7. 180° 모멘트 롤업 SHELL 편입 (suite_3d + ROLLUP_ONLY 필터)

- NLGEOM 자동 CR 확인: `SOLID_SHELL` → k=14 → `assemble_mesh_solid_shell_cr_numba` (dynamic3d.py:286-289, 506-510). 180° 수렴 자체가 증거 (소변형 k=12로는 불가).
- 결과 (25스텝, 5층 PET-PSA, 이형 이론 M≈0.2317):

| 조합 | 도달각 | M_root (오차) | circularity | slip | iters | wall |
|---|---|---|---|---|---|---|
| SS+SS | 180° | 0.23064 (-0.45%) | 1.205% | 209.9µm | 180 | 74s |
| SS+C3D8H | 180° | 0.23066 (-0.45%) | 1.204% | 209.9µm | 180 | 38s |

- AGENTS §2.4 3D-Opt1 (+0.2307, -0.4%, 209.9µm)와 4자리 일치 — **최강자 동점**. PSA 비압축 우려(SS u-p 없음)는 수치상 미발현.
- wall 차이는 컴파일 순서 아티팩트 (첫 케이스가 SS 컴파일 부담). iters 동일 180.
- 관찰 (SHELL 무관, 기존 지표): M_tip -0.239 vs M_root +0.231 (3.6% 불균형), U_strain 0.00 mJ — 반력/에너지 지표 점검 필요.
## 8. 반력 3.6%·에너지 0 별도점검 결과 (C:\...\Temp\opencode\audit_balance.py, 오프라인)

- U 0.00 원인 2겹: (a) `compute_strain_energy` ½uᵀf + `max(0,.)` — NLGEOM에서 부호 뒤집혀 clamp, (b) 벤치마크 `*1000`은 SI 가정인데 모델은 mm/MPa (N·mm=mJ 그대로). 둘 다 수정 → 증분일 Trapezoid 적산으로 교체 (2D/3D 러너), U=0.407 mJ (이론 ½Mθ=0.362, +12%).
- M 불균형은 수렴판정 탓 아님: tol 1e-8 재실행 시 iters·M 6자리 동일 → 판정 tol-둔감.
- 캐시 최종상태 오프라인 재조립 감사: ΣF≈1e-15 (힘평형 완벽) vs ΣM≈4.4e-3 (M의 1.9%, 0 아님), 절점평균 |f|=0.20N, 팁축반력 −8.8e-3 vs 루트 −1.2e-4 (75배 비대칭, 내부가 상쇄).
- 해석: 5-way OR 수락기준이 잔류 0.2N급에서 조기수락 (q_avg 분모 희석, AGENTS §4.7 경고 실측). M_root −0.45% 일치는 이 수준 오차 안에서는 운 포함 — 지표 정밀도는 ~1-4%, 0.4% 아님.
- AND-판정 재도입은 §4.7에서 dt붕괴로 회귀된 전적 — 손대지 말고 현 판정+오차대로 읽을 것.
## 9. JSON 전체 갱신 + SMP NT=1/2/4 (2026-09-16)

- `benchmark_pure_moment_rollup.json` 정식 갱신 (2D 5건+3D 7건, 수정 지표). 2D-Opt1 M=−0.230658·U=0.4072 (AGENTS −0.2307 일치).
- SMP wall(s), 200요소·25스텝:

| case | NT1 | NT2 | NT4 |
|---|---|---|---|
| 3D-CR/Base | 25/34 | 33/33 | 27/32 |
| 3D-Opt1 | 59 | 66 | 60 |
| 3D-Opt2-H/Opt2 | 49/32 | 54/33 | 56/31 |
| 3D-SHELL/SHELLOpt | 66/36 | 64/35 | 63/34 |

- 스케일링 없음 (NT2가 오히려 느림). 1500 DOF에서 PARDISO 직렬분해 지배, 조립병렬 무의미. SMP 이득은 대형모델에서만.
- 관찰: 2D-CPE6M 180° 도달하나 M −0.309(+34%)·U 0.07 — modified 삼각형 별도 점검 대상.
## 10. 병렬 방향 (2026-09-16, 사용자 결정): Hybrid MPI+SMP

- SMP 단독은 1500 DOF에서 효과 0 (PARDISO 직렬분해 지배) — 실측 NT1/2/4 평탄.
- 향후 Hybrid: MPI 영역분할(랭크간) + 랭크내 numba 스레드. 전제: mpi4py+MS-MPI, 분산 선형솔버(MUMPS/반복법+전처리)가 본체 — 조립병렬은 쉬운 쪽.
- 당장 유효한 병렬은 케이스 단위 프로세스 병렬 (스위트 wall 1/N).
## 11. Anderson 변별 벤치마크 3b (순수 Newton 킹크, line search 없음)

- guarded B3 토이는 line search가 구제해서 변별 불가 (둘 다 9회). 가드 제거 시 STD 2-cycle 무한진동(60회 미수렴) vs AA 36회 수렴 — `run_chattering_pure_newton` 정식 편입.
- 단, 실전 solve_step은 항상 line search → 메커니즘 프로브이지 실전 비교 아님. 실전 이득은 aggressive 캔틸레버 −25% 반복이 전부.
## 12. 교정 1단계: 참잔류 계측 (판정 불변, 읽기전용)

- `DynamicSolver3D._record_acceptance` 추가: 수락 3분기(residual/correction/fallthrough)에서 수락상태 f로 참 자유잔류 norm 기록 (`last_accept`). assemble 호출 추가 없음 (기존 반환값 재활용).
- 롤업 SHELLOpt 실측: 25스텝 전부 residual분기, worst r=20.3N (r_0≈4000N의 0.5% — Abaqus식 5e-3 그대로). 저울 눈금 수치 확정.
- 회귀: mechanics 게이트 14통과, C3D8I 굽힘 1실패는 stash 대조로 기존 실패 확인 (리포트도 0.49 LOCKED — 테스트 임계치 stale).
## 13. 교정 2단계: 반력평형 게이트 (사후, solve 불변)

- 단면분할 전달식 게이트는 폐기: 팁 단면 유사반력(−8.8e-3, 루트의 75배, 내부가 상쇄)을 25mm 암으로 증폭해 100% 오경보. 전역장(전절점 ΣF·ΣM, 루트중심) 게이트로 교체, 임계 5% (solver 5e-3 스케일 정합).
- 실측: Global F 0.000% 전 케이스, Global M 0.0~1.9% → SHELL·Opt1·Opt2·Base·2D 전원 OK.
- **게이트가 진짜 결함 검출**: 2D-CPE6M Global M 90.4% FAIL (M +34%·U 0.07 corroborate). §9 관찰을 게이트가 확정.
## 14. AA 실전(접촉) 비교: Hertz — AA 열위 (negative evidence)

- `benchmark_3d_contact.py`에 `use_anderson`·iters·wall 배선 추가. uz=−0.06, 40스텝:
- STD: 125회·153.2s·40/40호출(컷백 0) vs AA(m=4): 166회·202.6s·동일 물리(b=4.0).
- 떨림 자체가 없어서 소거 비교 불가. 활성집합 이동 시 AA 외삽이 stale → safeguard 기각+여분 조립으로 역효과.
- 결론: AA는 만능 아님. 활성집합 이동 문제에선 OFF가 정답. 용법: aggressive 평활문제 ON, 접촉 OFF.
## 15. Handoff (세션 종료, 커밋 미실행)

- 커밋·푸시 미실행 (계획만 출력, §COMMIT PLAN). 다음 세션 첫 작업: 위 계획대로 6커밋 후 push.
- 미커밋 범위(내 작업): benchmark_nlgeo_cantilever.py, benchmark_pure_moment_rollup.py, benchmark_3d_contact.py, run_nlgeo_solid_shell_vs_best.py, run_distorted_nlgeo_separation.py, run_moment_rollup_comparison.py, benchmark_anderson_solver.py(3b 등), results JSON/npz, 본 로그.
- 손대지 말 것: GEMINI.md, mechanics_patches.py, cr_wrapper_3d.py, solver2d/dynamic2d.py, element3d/__init__.py, figures, 타 dev_log — 병렬 에이전트 작업물. dynamic3d.py 내 편집은 a761bf4에 이미 포함됨.
- 중단 작업: Hertz stab 4조합 실험 (STD+STAB 1회 시작 후 중단, 결과 없음).
- 다음 후보: CPE6M 추적 / AA 실전 추가 / NAFEMS / 커밋 정리.
- 분리량(6.1e-07→5.6e-06 스케일링)은 이론값과 무관하게 유지 — §3 결론 불변.
