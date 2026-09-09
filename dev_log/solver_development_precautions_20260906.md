# 비선형 유한요소 솔버 개발 필수 준수 수칙 (Solver Development Guidelines)

> **문서 생성일**: 2026-09-06  
> **적용 대상**: `WHT_DispFoldSolver` 모든 개발자 및 LLM 에이전트 (Claude Code, Gemini, Antigravity)  
> **목적**: 유한요소 수치해석 기본 원칙 위반 방지, 수치적 회귀(Regression) 예방 및 불완전한 구현 억제

---

## 1. 개요 및 반성 (Reflection)

과거 `DynamicSolver` 및 구속조건 구현 중 발생했던 **플레이트 회전각 비대칭/꼬임 오차** 및 **Surface Tie 구속 이탈** 문제는 FEA 수치해석의 가장 기초적인 기본 원칙(Essential BC 강제, 대회전 재투영, 전역 방정식 조립 검증)을 충실히 지키지 않은 불완전한 구현에서 비롯되었습니다.

동일한 오판과 임기응변식 수정이 재발하지 않도록, 아래 5대 솔버 개발 수칙을 **영구 지침**으로 제정하고 엄격히 준수합니다.

---

## 2. 솔버 개발 5대 절대 수칙 (5 Mandatory Solver Rules)

### 📌 규칙 1. 필수 경계조건(Essential BC)의 엄격한 직접 강제 (No Weak Penalty Springs for Drive BCs)
- **원칙**: 마스터 노드 회전각, 변위 구동 등 물리적으로 규정된 경계조건(Dirichlet BC)은 **Exact Condensation 또는 잔차 직접 강제($R_{\text{ext}} = u_{\text{target}} - u_{\text{actual}}$)**를 통해 강제해야 한다.
- **금지 사항**: 반작용 모멘트/하중 스케일에 맞추지 않은 임의의 고정 페널티 스프링($K_\theta$)을 사용하여 경계조건을 유도하는 것은 엄격히 금지한다. (구동 모멘트에 의해 스프링이 꼬이는 수십 도의 수치 오차가 발생함)
- **페널티 사용 조건**: 페널티법 적용이 불가피한 경우, 반드시 전역 강성행렬 대각 성분 최고값 기준 최소 $10^8$배 이상의 동적 스케일링($K_{\text{penalty}} \ge 10^8 \cdot \max |K_{ii}|$)을 보장해야 한다.

---

### 📌 규칙 2. 전역 방정식 조립 루프 완전성 검증 (Complete Assembly Audit)
- **원칙**: 모델 내 생성되거나 `DynamicSolver`에 등록된 모든 구속조건(Penalty Constraints, Tie, MPC)은 **`_compute_R_total()` 및 `_solve_step_impl()`의 전역 잔차(\(R\)) 및 강성행렬(\(K\)) 조립 루프에 100% 연결**되어 있어야 한다.
- **금지 사항**: solver `__init__`에서 객체 참조만 보존하고 조립 함수 내 `apply_penalty()` 호출을 누락하는 등의 불완전한 연결을 금지한다.
- **검증 절차**: 구속조건 추가/수정 시, 단원 테스트 뿐만 아니라 실제 조립 루프에서 해당 구속조건의 힘/강성이 전역 잔차 벡터 및 행렬에 가산되는지 디버그 프린트 또는 린트로 직접 확인한다.

---

### 📌 규칙 3. 대변형/대회전(Updated Lagrangian) Frame Re-projection 필수 적용
- **원칙**: Updated Lagrangian 모드(`ul_large_rotation_mode = True`) 또는 10° 이상의 대회전이 일어나는 해석에서는 **매 Newton-Raphson assembly 단계 시작 전 `SurfaceTieConstraint.reproject_deformed(u)`를 반드시 호출**하여 현재 변형 좌표계($x = X + u$) 기준 법선/접선 및 투영 파라미터를 업데이트해야 한다.
- **금지 사항**: $t=0$ 초기 위치($X$)에서 계산된 투영 좌표/가중치를 대회전 해석 전체 동안 고정 사용하는 행위를 금지한다. (판재가 회전함에 따라 접촉/타이 구속력을 완전히 상실하고 기하학적 분리가 발생함)

---

### 📌 규칙 4. 수치 수렴(Convergence Ratio)에 착시된 물리적 타당성 오판 금지
- **원칙**: Newton-Raphson의 `Disp.Ratio < tol` 수렴 성공 판정은 **"수치 잔차가 줄어들었다"**는 의미일 뿐, **"물리적 해석 결과가 올바르다"**는 증명이 아니다. (구속조건 조립이 누락되어도 노드 자유도가 독립적으로 움직여 수렴할 수 있음)
- **필수 모니터링**:
  - 모든 예제 및 검증 코드에는 `[MONITOR]` 실시간 모니터링 로그를 심어 **Target vs Actual 각도 오차가 $0.00^\circ$인지**, **Tie Node 유효 결합률이 100%인지**, **자유단 Y 변위가 대칭을 이루는지** 물리적 메트릭을 실시간 검증해야 한다.
  - 해석 완료 후 반드시 `solver._check_mesh_quality()`로 메쉬 뒤집힘(`n_inverted == 0`) 및 물리적 형상(U자형 루프)을 시각적으로 최종 확인한다.

---

### 📌 규칙 5. 코드 변경 후 표준 검증 프로토콜(Verification Suite) 실행 의무
- **원칙**: `dispsolver/` 하위의 solver, element, constraint, material, mesh 코드를 수정한 후에는 **반드시 아래 검증 스크립트를 실행하여 기존 기능이 훼손되지 않았음을 입증(A/B Test)** 해야 한다.
  1. `pytest tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py -q`
  2. `python -m verification.run_all` (9개 물리 벤치마크 검증)

---

## 3. 이력 관리 및 이슈 트래커 참조

- **이슈 기록 문서**: `dev_log/rbe2_tie_fundamental_lessons_20260906.md`
- **프로젝트 공식 규칙**: `AGENTS.md` (§4.13 FEA Fundamental Rules)
