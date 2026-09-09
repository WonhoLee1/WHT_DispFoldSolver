# RBE2 & Surface Tie Implementation Lessons & Principles (2026-09-06)

## 1. 개요 및 반성 (Overview & Reflection)

`ex12_abaqus_inp_plate_fold.py` 실행 과정에서 발생했던 **플레이트 비대칭 회전각 오차(Left -116° vs Right -33°)** 및 **Surface Tie 구속 이탈** 현상은 비선형 유한요소 해석(FEA)의 가장 기본인 **필수 경계조건(Essential BC) 강제**와 **대변형 서페이스 재투영(Re-projection)** 원칙을 제대로 준수하지 않은 불완전한 구현으로 인해 발생했습니다.

이 문서에서는 이 실수의 근본적 원인을 명확히 기록하고, 향후 동일한 수치적 오류가 재발하지 않도록 준수해야 할 유한요소 솔버 개발 수칙을 명시합니다.

---

## 2. 근본 원인 분석 (Root Cause Analysis)

### ① RBE2 마스터 회전각 페널티 강성 오류 (Essential BC Violations)
- **발생한 결함**: 마스터 노드 회전 구동 시 임의의 작은 고정 페널티 강성($K_\theta$)을 적용함.
- **수학적 원인**: 다층 디스플레이 패널이 굽혀지면서 발생하는 반작용 모멘트($M_{\text{reaction}} \sim 10^8 \text{ N}\cdot\text{mm}$)가 페널티 강성($K_\theta$)을 압도하여, 페널티 평형식 $\theta_{\text{actual}} - \theta_{\text{target}} = \frac{M_{\text{reaction}}}{K_\theta}$ 에 의해 페널티 스프링 자체가 수십 도 이상 꼬이는 오차가 발생함.
- **기본 원칙 위반**: 필수 경계조건(Essential BC)을 수치 스케일링 없는 약한 가상 스프링으로 수동 제어하려 했던 것은 유한요소 수치해석 기본 원리의 명백한 오류임.

### ② Surface Tie 초기 좌표 고정 오류 (Large Rotation Frame Updates)
- **발생한 결함**: 대변형/대회전(Updated Lagrangian) 모드에서 Surface Tie 투영 파라미터($\xi$)를 $t=0$ 초기 위치로 고정 사용함.
- **수학적 원인**: 90° 이상 판재가 회전함에 따라 마스터 요소의 법선/접선 벡터가 공간상에서 완전히 회전하였으나, 고정 투영 좌표로 인해 회전 후 구속력을 상실하고 서페이스가 분리됨.
- **기본 원칙 위반**: 대회전 환경에서 변형 배치(Current Deformed Configuration $x_k = X + u_k$) 기준 매 Step 재투영(Re-projection)을 수행해야 하는 비선형 접촉/타이 알고리즘 기본 준수 실패.

---

## 3. 재발 방지를 위한 필수 솔버 준수 수칙 (Solver Development Rules)

1. **경계조건(BC) 강제 수칙**:
   - 강체 마스터 노드 및 필수 경계조건(Dirichlet BC)은 페널티 수치에 의존하지 않고, **Exact Dirichlet Condensation 또는 $R_{\text{ext}} = u_{\text{target}} - u_{\text{actual}}$ 잔차 직접 강제**를 통해서만 적용한다.
   - 페널티법을 사용하는 경우 반드시 전역 강성 대각 성분 기준 $10^8$배 이상의 동적 스케일링($K_{\text{penalty}} \ge 10^8 \cdot \max K_{ii}$)을 적용해야 한다.

2. **대변형/대회전 Surface Tie 수칙**:
   - Updated Lagrangian 모드 (`ul_large_rotation_mode = True`) 실행 시, 매 Newton-Raphson assembly 단계 시작 전 반드시 `SurfaceTieConstraint.reproject_deformed(u)`를 호출하여 **현재 변형 좌표계 상에서 투영 좌표를 재산출**해야 한다.

3. **검증 모니터링 수칙**:
   - 모든 대변형 구동 예제에서는 솔버 실행 중 **Target vs Actual 각도 오차가 $10^{-4\circ}$ 이내**인지, **Tie Node 구속 유지 개수가 100%인지** 실시간 대칭성 로그(`[MONITOR]`)를 통해 엄격히 자동 검증한다.

---

## 4. 결론

본 레슨을 프로젝트 전역 이슈 트래커 및 개발 지침으로 고정 기록하여, 향후 어떠한 물리 모델 확장 시에도 기본 유한요소 알고리즘이 훼손되거나 임기응변식 코드가 작성되지 않도록 엄격히 관리합니다.
