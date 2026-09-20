# WHT_DispFoldSolver 전체 18개 실패 테스트 원인 및 해결방안 분석 의뢰서

## 1. 개요 및 배경
전체 789개 pytest 스위트 실행 결과 726 Passed, 26 XFailed, 19 Skipped, 18 Failed가 발생하였습니다.
모든 패널은 아래 18개 실패 테스트 항목에 대해 [설계 - 조사 - 이론평가 - 계획 - 문제분석 - 해결방안] 6단계를 전방위로 분석하고, 각 테스트가 100% PASS되기 위한 정밀 수정 방안을 제시해야 합니다.

---

## 2. 18개 실패 테스트 항목 상세 현황

### 그룹 1: 3D 솔버 & 요소 역학 벤치마크 (5건)
1. `tests/test_3d_c3d4_anp.py::test_c3d4_anp_tangent_consistency`
   - 현상: Bonet & Burton (1998) 2-Pass 절점 압력 평균화 기구(ANP) 유한차분 탄젠트 검증 실패.
   - 문제: 비국소(non-local) 패치 체적 평균화 특성으로 인해 단일 요소 차분 섭동 시의 미세 차이 발생.
2. `tests/test_3d_mechanics_benchmarks.py::test_bending_shear_locking_relief[C3D8I]`
   - 현상: C3D8I 외팔보 굽힘 처짐비 `ratio > 0.85` 게이트 미달.
   - 문제: 9-mode EAS enhancement 구현 후 굽힘 성능이 개선되었으나 L/h=10 벤치마크와의 수치적 편차 존재.
3. `tests/test_3d_patch_test.py::test_3d_solver_uniaxial_tension_patch`
   - 현상: C3D8I 1개 요소 1축 인장 해석 변위 `assert pytest.approx(u_x, rel=1e-4) == 0.005` 실패 (실측 0.004963, 편차 -0.74%).
   - 문제: Poisson ratio(nu=0.3)로 인한 면내 횡방향 수축 구속(Poisson contraction effect) 고려 부재.
4. `tests/test_c3d8_hybrid.py::test_c3d8_hybrid_heterogeneous_assembly_in_solver3d`
   - 현상: PET(C3D8_CR)+PSA(C3D8H) 적층 샌드위치 메시 해석 시 iters=2에서 수렴 실패.
   - 문제: DynamicSolver3D의 수렴 판정 로직 또는 이종 요소 결합 시의 접선 불일치.
5. `tests/test_field_output.py::test_stress_recovery_c3d8_matches_finite_strain_target`
   - 현상: C3D8 유한변형률 응력 복원 오차 `assert rel < 1e-6` 실패 (실측 `1.94e-05`).
   - 문제: 복원 응력 텐서 정밀도 임계치 과도 설정.

### 그룹 2: 설정/프리셋 & 최적화 스킵 (2건)
6. `tests/test_abaqus_config.py::test_config_presets`
   - 현상: `assert teardrop.drive.dt_max == 0.1` 실패 (실측 0.01).
   - 문제: `make_teardrop_config`의 기본값 갱신 후 테스트 기대값 미반영.
7. `tests/test_prescribed_skip.py::test_prescribed_skip_solution_equivalence`
   - 현상: `assert ret_full >= 0` 실패 (실측 -20, 컷백 초과).
   - 문제: dt=1.0 단일 스텝 강제 진입 시 적응형 시간 제어기(conv_rate > 1.7) 스톨 컷백.

### 그룹 3: 다중 스텝 트리 및 분기/체크포인트 (2건)
8. `tests/test_multi_step_and_branching.py::test_step_tree_hierarchy_and_checkpoint_rollback`
   - 현상: Step 트리 분기 및 롤백 시 상태 동기화 실패.
9. `tests/test_multi_step_branching.py::test_branching_execution_engine_end_to_end`
   - 현상: 분기 해석 엔진 실행 도중 롤백 실패.

### 그룹 4: ODB 출력 및 CAE 하중 연동 (2건)
10. `tests/test_output_system.py::TestModelAndSolverIntegration::test_2d_solve_with_odb`
    - 현상: 2D 솔버 ODB 필드 출력 저장 및 검증 실패.
11. `tests/test_cae_constraints_and_loads.py::test_model_solver3d_concentrated_force_solve`
    - 현상: Model 레벨 ConcentratedForce의 solver3d 매핑 실패.

### 그룹 5: 2D 레거시 정식화/패치 (3건)
12. `tests/test_cpe4rh_patch.py::test_orthodox_solver_level_patch_test`
    - 현상: CPE4RH 솔버 패치 테스트 실패.
13. `tests/test_large_deformation_element_suitability.py::TestSolverLevel::test_q4_bbar_pet_converges`
    - 현상: Q4-Bbar 대변형 인장 솔버 미수렴.
14. `tests/test_large_deformation_element_suitability.py::TestSolverLevel::test_q4_eas_pet_converges`
    - 현상: Q4-EAS 대변형 인장 솔버 미수렴.

### 그룹 6: 2점 굽힘 벤치마크 (2건)
15. `tests/test_two_point_bending.py::test_thin_glass_large_rotation_pure_bending_sri`
    - 현상: SRI 기반 박막 유리 대회전 굽힘 처짐 불일치.
16. `tests/test_two_point_bending.py::test_element_formulation_comparison`
    - 현상: 요소 정식화 간 모멘트/처짐 불일치.

### 그룹 7: 비선형 소프트 접촉 & RBE2 (2건)
17. `tests/test_deformable_soft_nonlinear_contact.py::test_soft_laws_no_longer_raise_and_converge[SOFT_LINEAR]`
    - 현상: 선형 소프트 접촉 법칙 뉴턴 루프 미수렴.
18. `tests/test_rbe2_jax_verify.py::test_solver_rbe2_vmap_threshold_default`
    - 현상: JAX RBE2 vmap 임계값 검증 실패.

---

## 3. 패널 검토 요구사항
- 각 패널은 18개 항목에 대해 근본 원인(수학적 정식화, 기구학적 구속 조건, 솔버 수렴 조건, 테스트 임계치 등)을 분석하고,
- 기존 정상 동작 코드(`AGENTS.md` Anti-Mocking 원칙 준수)를 훼손하지 않는 외과적 수술(Surgical change) 수정 계획을 수립할 것.
