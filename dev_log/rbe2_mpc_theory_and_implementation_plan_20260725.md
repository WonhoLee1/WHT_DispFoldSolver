# 📚 RBE2 / MPC 강체 구동 유한요소 해석의 이론적 근거 및 고도화 계획서

**작성일**: 2026-07-25  
**목적**: 과거 RBE2 / MPC 구동 시 발생한 수치적 스톨(Stall) 및 요소 뒤집힘(Element Inversion)의 근본 수학적 원인을 규명하고, Abaqus/Standard 수준의 수치적 안정성을 보장하는 **RBE2 Kinematic Elimination (자유도 응축) + Co-rotational Q4 FEA 통합 엔진**의 이론적 근거와 구체적 구현 방안을 정립한다.

---

## 1. 과거 RBE2 / MPC 수치 실패의 근본 원인 분석

### ① 라그랑주 승수법 (Lagrange Multiplier)의 KKT 부정치 (Indefinite Saddle-Point) 문제
- RBE2/MPC 다점 제약 조건 C * u - d = 0을 라그랑주 승수 λ로 유한요소 시스템에 추가할 때, KKT 확장 강성 행렬은 다음과 같습니다:
  [ K  C^T ] [ du ]   [ f_ext - f_int ]
  [ C   0  ] [ λ  ] = [       0       ]
- **문제점**: 오른쪽 아래 0 블록(Zero Diagonal)으로 인해 시스템 강성 행렬이 **양의 정치성(Positive Definiteness)을 상실하고 Indefinite Saddle-Point**가 됩니다.
- 일반적인 Direct Sparse/Cholesky 솔버나 iterative CG 솔버는 이 zero diagonal에서 조건수(Condition Number)가 폭발하며, 뉴턴-랩슨 이터레이션 1~2회 만에 수렴 불능 및 Cutback 스톨을 유발합니다.

### ② 0.0167mm 박막 계면의 굽힘-전단 락킹 (Bending-Shear Locking)
- 하우징 플레이트 강체 구역(x < -10mm, x > 10mm)과 자유 힌지 구역(-10mm <= x <= 10mm)의 경계선(x = ±10mm)에서 연속체 Q4 요소가 급격한 대변형 회전을 받을 때, Cauchy-Green 선형 변형률 오차로 인해 가우스 포인트에서 det(F) <= 0 오진이 발생합니다.

---

## 2. 해결을 위한 이론적 근거 (Theoretical Foundation)

### [원리 1] RBE2 Kinematic Elimination (자유도 소거 및 강성 응축)
- 라그랑주 승수법 대신, 슬레이브 자유도 u_s를 마스터 회전 및 변위 u_m에 관한 관계식으로 직접 소거하는 **변환 행렬 T**를 구성합니다:
  u_full = T * u_reduced
  K_condensed = T^T * K_full * T
  f_condensed = T^T * (f_ext - f_int)
- **수학적 이점**: KKT 0 대각항이 완전히 축소 소거되어, 응축된 강성 행렬 K_condensed는 **100% 대칭 양의 정치 행렬(Symmetric Positive Definiteness, SPD)**이 됩니다. Pardiso / Sparse Cholesky 솔버가 수치적 안정성을 100% 유지하며 수렴합니다.

### [원리 2] Co-rotational Local Frame Isolation (동회전 국소 좌표계)
- 4절점 Q4 요소의 대변형 계산 시, 요소의 강체 회전 텐서 R_e를 추출한 후 국소 좌표계에서 순수 변형률(Pure Deformation Strain)만 연산합니다:
  F_e = R_e * U_e  (det(R_e) = 1.0)
  ε_local = ln(U_e)
- **수학적 이점**: 아무리 큰 대회전(-90° / +90°)이 가해지더라도 det(U_e) > 0 이 수학적으로 엄격히 유지되어 요소 뒤집힘(Element Inversion)이 근본 차단됩니다.

### [원리 3] Abaqus Automatic Viscous Stabilization (점성 안정화)
- 힌지 중앙부의 국소 좌굴 및 좌/우 날개 접힘 시의 국소 이완 현상을 방지하기 위해 점성 감쇠 행렬 C_stab = c_stab * M을 추가하고 에너지 소산 비율을 E_stab / E_strain < 0.5%로 엄격 제어합니다.

---

## 3. 단계별 실행 로드맵 (Step-by-Step Execution Plan)

1. **Step 1 (이론 검증 모듈 정립)**: `dispsolver/constraint/rbe2_condensed.py` 모듈에서 RBE2 마스터-슬레이브 변환 행렬 T의 변형률 비가반성 및 Pure SPD 응축 검증.
2. **Step 2 (Co-rotational Q4 FEA 솔버 결합)**: `dispsolver/solver/dynamic.py`에 Co-rotational 요소를 조립하여 det(U_e) > 0 및 Pure SPD 강성 조립.
3. **Step 3 (실제 RBE2 FEA 시뮬레이션 완주)**: `examples/ex03_rbe2_fea_folding.py` 구동으로 Left -90°, Right +90° 완전 수직 접힘 및 정중앙 하향 U자 처짐 FEA 해 산출.
4. **Step 4 (1x2 GIF 애니메이션 & 물리 데이터 산출)**: 고정 좌표계에서 유효 응력 및 변형률 GIF 생성 및 검증.
