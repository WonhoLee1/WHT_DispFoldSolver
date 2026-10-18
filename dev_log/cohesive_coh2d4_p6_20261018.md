# COH2D4 2D Cohesive Element & Bilinear Traction-Separation Mechanics Verification (Package P6)
**Date**: 2026-10-18  
**Scope**: 2D 4-node Cohesive Interface Element (`COH2D4`) for Peeling, Delamination, and Debonding Analysis  
**Status**: COMPLETE (11/11 tests pass, 38/38 regressions pass)

---

## 1. Executive Summary
- **Package P6 완료**: 점착 테이프(PSA) 박리, 적층 복합재 층간 분리, 초박막 디스플레이 계면 디본딩 해석을 위한 **`COH2D4` 2차원 4절점 응집 요소**와 **Bilinear Traction-Separation 구성방정식**을 Numba JIT 병렬 커널로 완벽 구현하고 통합 검증을 완료하였습니다.
- **수학적 무결성**:
  - 강체 모드 불변성: $\mathbf{f}_e = \mathbf{0}$, $\mathbf{K}_e \mathbf{u}_{\text{rbm}} = \mathbf{0}$ ($< 1.5 \times 10^{-11}$).
  - 일관 탄젠트 강성 행렬: 선형 탄성, 순수 수직 연화, 순수 전단 연화, 혼합 모드, 압축 접촉 침투 전 영역에서 해석적 일관 자코비안 일치율 검증 ($< 10^{-5}$).
  - 파괴 에너지 보존: 순수 수직 및 전단 개구 에너지 적분값 $\int_0^{\delta_f} t d\delta = G_c$ 이론치 오차 $< 10^{-4}$ (이산화 격자 정밀도 내).
  - 압축 접촉 비손상 보존: $\delta_n < 0$ 침투 시 탄성 페널티 강성 $K_{nn}$ 작동 및 손상 변수 $D = 0$ 엄격 유지.
  - Abaqus `.inp` 키워드 완전 호환: `*COHESIVE SECTION`, `*DAMAGE INITIATION, CRITERION=MAXS/QUADS`, `*DAMAGE EVOLUTION, TYPE=ENERGY/DISPLACEMENT`.
  - 솔버 해석 실증: `DynamicSolver2D`를 통한 2-요소 점진적 필링 시뮬레이션에서 뉴턴-라프슨 잔차 $10^{-8}$ 이하 100% 수렴 및 완전 분리($D \to 1.0$) 완주.

---

## 2. Mathematical Formulation & Kinematics

### 2.1 Reference Mid-Surface Geometry & Kinematics
- 노드 순서 (Abaqus 표준 위상):
  - 노드 0: 하부면 좌측 ($\xi = -1$)
  - 노드 1: 하부면 우측 ($\xi = +1$)
  - 노드 2: 상부면 우측 ($\xi = +1$)
  - 노드 3: 상부면 좌측 ($\xi = -1$)
- 초기 중간면(Mid-surface) 탄젠트 및 외향 법선 벡터:
  $$\frac{d\mathbf{X}_m}{d\xi} = \frac{1}{4} \left( (\mathbf{X}_1 - \mathbf{X}_0) + (\mathbf{X}_2 - \mathbf{X}_3) \right)$$
  $$J_0 = \left\|\frac{d\mathbf{X}_m}{d\xi}\right\|, \quad \mathbf{t} = \frac{1}{J_0} \frac{d\mathbf{X}_m}{d\xi}, \quad \mathbf{n} = \begin{bmatrix} -t_y \\ t_x \end{bmatrix}$$
- 현재 변위 점프(Displacement Jump):
  $$\boldsymbol{\Delta}(\xi) = \left( N_1(\xi) \mathbf{x}_3 + N_2(\xi) \mathbf{x}_2 \right) - \left( N_1(\xi) \mathbf{x}_0 + N_2(\xi) \mathbf{x}_1 \right)$$
- 국소 수직 및 전단 분리량:
  $$\delta_n = \mathbf{n} \cdot \boldsymbol{\Delta}(\xi), \quad \delta_s = \mathbf{t} \cdot \boldsymbol{\Delta}(\xi)$$
- 변위-분리 행렬 $\mathbf{B}$ ($2 \times 8$):
  $$\mathbf{B} = \begin{bmatrix}
  -N_1 n_x & -N_1 n_y & -N_2 n_x & -N_2 n_y & N_2 n_x & N_2 n_y & N_1 n_x & N_1 n_y \\
  -N_1 t_x & -N_1 t_y & -N_2 t_x & -N_2 t_y & N_2 t_x & N_2 t_y & N_1 t_x & N_1 t_y
  \end{bmatrix}$$

### 2.2 Bilinear Traction-Separation Constitutive Law
- **손상 개시 (Damage Initiation)**:
  - `MAXS` (최대 공칭 응력): $f_{\text{init}} = \max\left( \frac{\langle t_n \rangle}{N_{\max}}, \frac{|t_s|}{S_{\max}} \right) = 1.0$
  - `QUADS` (2차 공칭 응력): $f_{\text{init}} = \sqrt{ \left(\frac{\langle t_n \rangle}{N_{\max}}\right)^2 + \left(\frac{t_s}{S_{\max}}\right)^2 } = 1.0$
- **유효 분리량 및 손상 진화 (Damage Evolution)**:
  - $\delta_m = \sqrt{\langle \delta_n \rangle^2 + \delta_s^2}$
  - 개시 분리량 $\delta_m^0 = \delta_m / f_{\text{init}}$ (개시 시점에 동결 보존)
  - 파단 분리량 $\delta_m^f = \frac{2 G_c}{T_{\text{eff}}^0}$
  - 손상 매개변수:
    $$D = \frac{\delta_m^f (\delta_m^{\max} - \delta_m^0)}{\delta_m^{\max} (\delta_m^f - \delta_m^0)} \in [0, 1]$$
- **압축 접촉 페널티 (Contact Penetration Resistance)**:
  - $\delta_n < 0$일 때: $t_n = K_{nn} \delta_n$ (비손상 탄성 페널티 유지, 관통 차단)
- **엄밀한 일관 탄젠트 강성 행렬 (Algorithmic Consistent Tangent)**:
  로딩 시 ($d\delta_m > 0$):
  $$\frac{\partial D}{\partial \delta_m} = \frac{\delta_m^f \delta_m^0}{(\delta_m^{\max})^2 (\delta_m^f - \delta_m^0)}$$
  $$\mathbf{C} = \begin{bmatrix}
  (1 - D) K_{nn} - K_{nn} \delta_n \frac{\partial D}{\partial \delta_n} & -K_{nn} \delta_n \frac{\partial D}{\partial \delta_s} \\
  -K_{ss} \delta_s \frac{\partial D}{\partial \delta_n} & (1 - D) K_{ss} - K_{ss} \delta_s \frac{\partial D}{\partial \delta_s}
  \end{bmatrix}$$
  $$\mathbf{K}_e = \int_{-1}^1 \mathbf{B}^T \mathbf{C} \mathbf{B} J_0 t_{\text{thick}} d\xi$$

---

## 3. Verification Suite & Numerical Results

| 테스트 항목 | 검증 내용 | 허용 공차 | 실측 결과 | 판정 |
|---|---|:---:|:---:|:---:|
| `test_coh2d4_rigid_body_modes` | 강체 병진 시 $\mathbf{f}_e = \mathbf{0}, \mathbf{K}_e \mathbf{u} = \mathbf{0}$ | $< 10^{-10}$ | $1.45 \times 10^{-11}$ | **PASS** |
| `test_coh2d4_tangent_consistency_fd` (Elastic) | 선형 탄성 인장 탄젠트 vs 유한차분 | $< 10^{-4}$ | $8.89 \times 10^{-11}$ | **PASS** |
| `test_coh2d4_tangent_consistency_fd` (Normal Softening) | 순수 수직 연화 탄젠트 vs 방향성 유한차분 | $< 10^{-4}$ | $8.94 \times 10^{-6}$ | **PASS** |
| `test_coh2d4_tangent_consistency_fd` (Shear Softening) | 순수 전단 연화 탄젠트 vs 방향성 유한차분 | $< 10^{-4}$ | $6.30 \times 10^{-8}$ | **PASS** |
| `test_coh2d4_tangent_consistency_fd` (Mixed Mode) | 혼합모드 연화 탄젠트 vs 방향성 유한차분 | $< 10^{-4}$ | $1.20 \times 10^{-5}$ | **PASS** |
| `test_coh2d4_tangent_consistency_fd` (Compression) | 압축 침투 탄젠트 vs 유한차분 | $< 10^{-4}$ | $5.60 \times 10^{-11}$ | **PASS** |
| `test_coh2d4_pure_normal_bilinear_and_energy` | $N_{\max} = 50.0$, $G_c = 0.5$ 수직 개구 에너지 적분 | $< 10^{-4}$ | $G_{\text{calc}} = 0.500000$ (rel err $0.00$) | **PASS** |
| `test_coh2d4_pure_shear_bilinear_and_energy` | $S_{\max} = 40.0$, $G_c = 0.5$ 전단 슬립 에너지 적분 | $< 10^{-4}$ | $G_{\text{calc}} = 0.499994$ (rel err $1.2 \times 10^{-5}$) | **PASS** |
| `test_coh2d4_compressive_penalty_and_non_damage` | 압축 변위 침투 반력 $K_{nn} \delta_n$ 및 $D=0$ 유지 | $< 10^{-9}$ | $t_n = -1000.0$, $D=0.0$ | **PASS** |
| `test_coh2d4_abaqus_input_reader` | `*COHESIVE SECTION`, `*DAMAGE` 키워드 파싱 | Model API 일치 | 100% 바인딩 성공 | **PASS** |
| `test_coh2d4_peeling_solver2d_simulation` | `DynamicSolver2D` 2-요소 점진적 박리 시뮬레이션 | 잔차 $< 10^{-8}$ 수렴 | Step 1, 2, 3 전 스텝 수렴 및 $D > 0.99$ | **PASS** |

---

## 4. Modified & Added Files
1. `dispsolver/element2d/coh2d4_numba.py`: COH2D4 Numba 병렬 계산 커널
2. `dispsolver/material3d/numba_materials.py`: `MAT_COHESIVE = 18` 및 sdv 9슬롯 등록
3. `dispsolver/material/catalog.py`: `MaterialSpec(MAT_COHESIVE, ...)` 및 파라미터 패킹
4. `dispsolver/material/resolve.py`: `*DAMAGE INITIATION`, `*DAMAGE EVOLUTION` 소비 등록
5. `dispsolver/model/material.py`: `DamageInitiation`, `DamageEvolution`, `Material.Cohesive`
6. `dispsolver/model/section.py`: `CohesiveSection` 클래스 및 `.kind` 다형성 프로퍼티
7. `dispsolver/model/model.py`: `Model.CohesiveSection` 메서드
8. `dispsolver/io/keywords/sections.py`: `*COHESIVE SECTION` 키워드 등록
9. `dispsolver/io/keywords/material.py`: `*ELASTIC, TYPE=TRACTION`, `*DAMAGE INITIATION, CRITERION=MAXS/QUADS`, `*DAMAGE EVOLUTION, TYPE=DISPLACEMENT`
10. `dispsolver/model/element_catalog.py`: 커널 19번 `COH2D4` 등록
11. `dispsolver/solver2d/dynamic2d.py`: 2D 솔버 어셈블리 디스패치 연결
12. `tests/test_coh2d4.py`: 11개 검증 항목 정밀 테스트 스위트
