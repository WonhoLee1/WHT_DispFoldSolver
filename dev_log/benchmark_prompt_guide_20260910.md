# AI 에이전트 수치 해석 & 벤치마크 프롬프트 가이드 (Prompting Guide for FEA Benchmarks)
**작성일**: 2026-09-10  
**적용 대상**: FEA Solver, 유한요소 제형, 비선형 구조해석 벤치마크 및 검증 작업  

---

## 1. 벤치마크 프롬프트 작성의 4대 핵심 원칙

### ① 이론적 기준값(Analytical Reference) 및 허용 오차 명시
- 단순히 "잘 되는지 테스트해줘"라고 요청하면 비선형 수렴 여부나 물리적 타당성을 검증하지 않고 종료될 위험이 있습니다.
- 논문이나 표준 아바쿠스(Abaqus) 가이드 문서의 **정확한 수치 목표**와 **허용 오차 범위(예: `<0.5%`)**를 명시하여 AI가 스스로 루프를 돌며 정밀도를 검증하도록 지시하세요.
- *예시*: `Bisshopp & Drucker (1945) 이론해 v_tip = 8.03 m 대비 오차 < 0.5% 달성`

### ② 물성치 / 단위계 / 하중 인가 방식 구체화
- 물성치($E, \nu$), 면적/단면적, 하중의 크기 및 분포 인가 방식(포인트 하중 vs 면 구속분배), 구속조건(BC)을 명확히 전달하여 파싱 오류나 강도 계산 왜곡을 방지하세요.

### ③ 비선형/대변형(NLGEOM) 및 락킹 방지 제형 명시
- 소변형 선형 해석인지 대변형 비선형(Total/Updated Lagrangian) 해석인지를 명확히 구분하세요.
- 비압축성 근사 영역($\nu \to 0.5$)이나 휨 변형 시 전단/체적 락킹을 방지하는 특수 제형(`EAS 9-mode`, `F-bar`) 적용을 명시하세요.

### ④ 수치적 검증 프로토콜 강제 (Empirical Verification Protocol)
- 코드 수정에 그치지 않고 **실제 스크립트 실행(`python -u`)과 로그 검시**를 의무화하세요.
- 이상 수치(발산, 오차 폭발 등) 발생 시 예외 처리나 임시값 반환으로 덮지 말고, **근본 원인(Root Cause Analysis)**을 파헤치도록 강제하세요.

---

## 2. 복사해서 바로 쓰는 프롬프트 표준 템플릿

```text
[요소 제형 및 해석 엔진 벤치마크 수행 요청]

1. 대상 제형 및 해석 조건:
   - 요소 제형: 3D Total Lagrangian C3D8I (9-mode EAS + F-bar) 및 C3D8_FBAR
   - 비선형 옵션: NLGEOM = True (Finite Strain, Green-Lagrange Strain E, PK2 Stress S)
   - 재질 모델: 미세가압/비압축성 성질 (E = 1.0 MPa, nu = 0.49995 또는 E = 100 MPa, nu = 0.3)

2. 수행할 벤치마크 및 검증 기준 (Success Criteria):
   ① Cook's Membrane Benchmark (Near-Incompressible In-Plane Shear):
      - 경계/하중: X=0 고정, X=48 면 전단 하중 Fy 인가
      - 목표 정밀도: 선형/비선형 이론 처짐량 대비 오차 < 0.5%
   ② Geometrically Nonlinear Cantilever Deflection (Elastica Benchmark):
      - 외팔보 L = 10m, 하중 P = 269.35 N
      - 목표 정밀도: Bisshopp & Drucker (1945) 이론해 v_tip = 8.03 m 대비 오차 < 0.5%

3. 작업 및 검증 지침:
   - 기존 모듈과의 독립성을 100% 유지할 것 (2D / 3D 분리).
   - 스크립트를 실제 실행(`python -u ...`)하여 수렴 여부 및 처짐량을 수치로 직접 확인할 것.
   - 잔차 노름(Residual Norm) 평가 시 Penalty BC 영향이 없도록 자유 자유도(Free DOFs) 기준으로 평가할 것.
   - 이상 수치(발산, 100% 오차 등) 발생 시 표면적 임시 처방을 금지하고, Stiffness/B-matrix/Line search 등의 근본 원인을 진단하여 해결할 것.
   - 최종 결과를 요소 비교 표(Benchmark Matrix) 및 dev_log/ 문서로 상세히 기록할 것.
```

---

## 3. 주요 문제 발생 시 대응 프롬프트 문구 모음

| 발생 문제 (Symptom) | 권장 조치 프롬프트 문구 |
|---|---|
| **처짐량이 0에 가깝거나 강도가 너무 큼** | `DynamicSolver의 __init__ 재질 파라미터 매핑과 E, nu 수치가 default값으로 덮어씌워지지 않았는지 점검하고 실제 E값 반영 여부를 확인하라.` |
| **Line Search / Newton 회차가 25회 돌며 수렴 안 됨** | `Penalty BC 노름이 r_free 노름을 오염시키고 있지 않은지 확인하고, 잔차 최소화 scaling s를 선택하는 Best-Step Line Search를 적용하라.` |
| **Windows 파이썬 실행 시 무한 대기/먹통** | `Numba OpenMP thread deadlock 가능성이 있으니 @njit(parallel=True) 대신 @njit(fastmath=True) 직렬 JIT 커널로 전환하여 테스트하라.` |
| **1-Element Thick 휨 변형 시 락킹 발생** | `C3D8I 9-mode EAS 모드 변형률 matrix B_tilde와 static condensation K_uu - K_ua*inv(K_aa)*K_ua^T 가 정상 작동하는지 점검하라.` |
