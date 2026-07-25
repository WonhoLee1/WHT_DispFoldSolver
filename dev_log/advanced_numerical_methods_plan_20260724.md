# WHT_DispFoldSolver — Abaqus 최신 버전 수준의 수렴 성능 및 정확도 확보 계획서

**작성일**: 2026-07-24  
**문서 상태**: Abaqus Parity 목표 설정 및 초고급 수치해석 아키텍처 확정  
**최종 목표**: **Abaqus/Standard 최신 버전 수준의 수렴 속도, 견고성(Robustness) 및 수치 정확도(Accuracy) 100% 확보**  
**대상 문제**: 디스플레이 다층 패널(PET/PSA/CoverWindow 등 7층 복합체) 90° 폴딩 대변형·접촉·점탄성·소성 수치 해석  

---

## 1. Abaqus/Standard 수준 수렴성 & 정확도 핵심 목표 (Target Metrics)

Abaqus/Standard 최신 버전과 동등한 수렴 패러다임과 수치 정확도를 확보하기 위해 다음 4대 핵심 기준을 적용합니다.

```
========================================================================================
                  Abaqus/Standard Level Convergence & Accuracy Parity
========================================================================================
 [1. Abaqus Standard Controls]   -> [2. Exact Algorithmic Tangent] -> [3. Mortar/Nitsche Contact]
  R_max <= 0.005 * q_avg            Pure Quadratic Newton            C1-Continuous Smooth Contact
  c_max <= 0.01 * du_max            No Finite Difference             Stabilized Penalty/Augmented
----------------------------------------------------------------------------------------
 [4. Master-Slave MPC]          -> [5. Automatic Energy Damping]
  Kinematic Elimination             Abaqus Adaptive Stabilization 
  Zero-Diagonal KKT Elimination     Energy Dissipation Fraction < 0.5%
========================================================================================
```

| 검증 항목 | 기존 WHT_DispFoldSolver 현황 | Abaqus/Standard 목표 수준 |
|---|---|---|
| **수렴 속도 (NR Iterations)** | 10~18회 반복 시 발산 또는 Cutback | **증분당 평균 3~6회 NR 반복 내 수렴** ($I_N \le 8$) |
| **시간 증분 제어 (Step Control)** | dt: $10^{-2} \to 10^{-6}$ (스톨 현상) | **Cutback 0~2회 이내로 90° 폴딩 완주** |
| **수치 정확도 (Accuracy)** | 요소 락킹 및 비일관 접선으로 응력 오차 발생 | **Abaqus C3D8I / C3D8R 대비 변위·응력 오차 < 0.1%** |
| **제약조건 처치 (MPC/RBE2)** | Penalty/Lagrange 혼용 (KKT 조건수 $10^{16}$) | **Master-Slave Kinematic Condensation (SPD 행렬)** |
| **접촉 해석 (Contact)** | Node-to-Segment 분선형 Penalty | **Mortar/Segment-to-Segment C1-Smooth Contact** |

---

## 2. Abaqus 동등 수준 수치해석 아키텍처 상세

### 2.1. Abaqus 수렴 제어 규격 (Abaqus Default Solution Controls)
Abaqus/Standard의 잔차 및 변위 수정량 수렴 판정식을 엄격히 도입합니다:
1. **잔차 제어 (Residual Criterion)**:
   $$R_{\max} = \max_i |R_i| \le q_{\tau} \cdot \bar{q}$$
   - 여기서 $\bar{q}$는 구조물 전체 절점력/모멘트의 공간적 평균값(Spatial Average Force)이며, Abaqus 표준 허용치 $q_{\tau} = 0.005$ (0.5%)를 적용합니다.
2. **변위 수정량 제어 (Correction Criterion)**:
   $$c_{\max} = \max_i |\Delta u_i| \le r_{\tau} \cdot \Delta u_{\max}$$
   - $r_{\tau} = 0.01$ (1%)를 적용하여 잔차뿐만 아니라 구하고자 하는 운동학적 변위의 변화량까지 완전 검증합니다.
3. **자동 증분 조절 법칙 (Automatic Time Stepping Rules)**:
   - $I_N \le 4$: 다음 증분 크기 $dt_{next} = 1.5 \times dt$ (증가)
   - $4 < I_N \le 8$: 증분 유지 ($dt_{next} = dt$)
   - $8 < I_N \le 16$: 라인서치 가동 및 다음 증분 축소
   - $I_N > 16$: 증분 즉시 Cutback ($dt_{next} = 0.25 \times dt$) 및 상태 복원

### 2.2. Abaqus 수치 점성 감쇠 및 자동 안정화 (Automatic Viscous Stabilization)
- **개념**: 힌지 부근의 극심한 대변형, 주름(Wrinkling), 국소 좌굴 구역에서 순간적 가짜 수치 발산을 막기 위해 Abaqus의 `*STATIC, STABILIZE` 기술을 도입합니다.
- **감쇠력 추가**:
  $$F_{stab} = c \cdot M^* \cdot v, \quad c = s \cdot \rho L v_{char}$$
- **에너지 소산 비율 제어 (Dissipated Energy Fraction)**:
  - 감쇠에 의해 소산된 에너지 $E_{stab}$가 구조물 전체 변형 에너지 $E_{strain}$의 **0.5% 이하**가 되도록 감쇠 계수 $c$를 자동 스케일링하여 **정확도를 훼손하지 않으면서 완전 수렴**을 보장합니다.

### 2.3. Abaqus 수준의 Exact Algorithmic Consistent Tangent
- 유한차분(FD)을 100% 제거하고, Abaqus UMAT 수치 정밀도와 동일한 **JAX Exact Hessian**을 결합합니다.
- **Simo-Taylor J2 소성 return-mapping 일관 접선**:
  $$\mathbb{C}^{ep} = \mathbb{C}^{el} - \frac{2\mu \bar{\gamma}}{\|s\|} \mathbb{I}_{dev} + 2\mu \left( \frac{\bar{\gamma}}{\|s\|} - \frac{1}{3\mu + H} \right) n \otimes n$$
- **Simo/Holzapfel 대변형 점탄성 이력 미분** 및 **Flory 등체적/체적 분할 대변형 초탄성**의 Exact JAX Autodiff.

### 2.4. Kinematic Master-Slave Constraints (RBE2/Tie)
- Abaqus의 MPC(Multi-Point Constraint) 처리 방식과 동일하게, KKT 라그랑지 승수 행렬 대신 **변환 매트릭스 소거법(Kinematic Condensation)**을 적용합니다.
  $$u_{slave} = C u_{master}$$
  $$K_{condensed} = T^T K_{full} T$$
- 이를 통해 Saddle-point 제약에 의한 KKT 조건수 폭발을 제거하고 **Pure SPD (Symmetric Positive Definite)** 행렬 구조로 PARDISO 풀이 속도 및 정밀도를 극대화합니다.

### 2.5. Mortar/Nitsche 기반 C1-Smooth Contact Formulation
- Abaqus의 Surface-to-Surface Contact 패러다임 적용:
  - Penalty 기법의 스티프니스 수치 조건수 악화 방지를 위한 **Augmented Lagrangian Loop** 결합.
  - 접촉 면간 Gap 미분 연속성(C1-Continuity)을 확보하여 절점 투과(Penetration) 및 척력 진동 방지.

---

## 3. Abaqus Parity 달성을 위한 정밀 실행 로드맵 (Detailed Roadmap)

```
========================================================================================
                      Abaqus Parity 5단계 구체적 개발 일정
========================================================================================
 [Phase 1] Master-Slave Condensation  : RBE2/Tie 라그랑지 승수 제거 및 SPD 강성 변환
 [Phase 2] Abaqus Convergence Engine  : R_max, c_max, Spatial Average Force, Auto Stepping
 [Phase 3] Automatic Viscous Damping  : Stabilization Energy Fraction < 0.5% 자동 제어
 [Phase 4] JAX Exact Tangent Complete : FD 0%, Simo J2 + Simo Visco Exact Autodiff
 [Phase 5] Abaqus Cross-Verification  : Abaqus/Standard 2024 대조 검증 (오차 < 0.1%)
========================================================================================
```

### Phase 1: Kinematic Master-Slave Elimination (Day 1-2) 🔴 **CRITICAL**
- `dispsolver/constraint/rbe2_condensed.py` 구현
- RBE2/Tie 변환 매트릭스 $T$ 생성 및 $K_{condensed} = T^T K T$ 조립
- **검증**: KKT 대각 0 소멸 및 PARDISO 풀이 시간 80% 단축 확인

### Phase 2: Abaqus Standard Solution Control Engine (Day 3-4) 🔴 **CRITICAL**
- `dispsolver/solver/dynamic.py`에 Abaqus 수렴 엔진 탑재
- Spatial Average Force $\bar{q}$ 및 $R_{\max} \le 0.005 \bar{q}$, $c_{\max} \le 0.01 \Delta u_{\max}$ 수렴 판정식 구현
- Automatic Time Stepping ($I_N \le 4 \implies dt \uparrow$, $I_N > 8 \implies dt \downarrow$)

### Phase 3: Automatic Viscous Stabilization (Day 5-6) 🟠 **HIGH**
- `dispsolver/solver/stabilization.py` 추가
- Local damping factor $c$ 조절 및 $E_{stab} / E_{strain} < 0.5\%$ 모니터링 모듈 탑재

### Phase 4: Full Exact Algorithmic Tangent (Day 7-8) 🟠 **HIGH**
- JAX 유한차분 전면 폐지, `jax.jacobian` 및 `jax.hessian`을 통한 EAS/Viscoelastic/Plasticity 완전 미분
- 증분당 Pure 2차 Newton 수렴 ($I_N \le 4$) 달성

### Phase 5: Abaqus/Standard Benchmarking & Verification (Day 9-10) 🟢 **MEDIUM**
- `examples/ex03_display_fold.py` 90° 완주 및 Abaqus 레퍼런스 결과와의 오차 대조
- 변위, 응력, 모멘트-회전각 곡선 오차 **< 0.1%** 검증 완료

---

## 4. 검증 및 게이트 기준 (Gate Criteria)

1. **Abaqus 수렴 파리티 (Convergence Parity)**:
   - 90° 디스플레이 폴딩 수치 해석 전 구간 증분당 평균 NR 반복 **3~5회**.
   - 수치 스톨/Cutback 횟수 **0회**.
2. **Abaqus 정확도 파리티 (Accuracy Parity)**:
   - 힌지 구간 맥스 스트레스/변형률 Abaqus/Standard 대조 오차 **0.1% 이내**.
   - 에너지 소산 비율 $E_{stab} / E_{strain} < 0.5\%$ 유지.

---

*Updated & Persisted in `./dev_log/advanced_numerical_methods_plan_20260724.md`*
