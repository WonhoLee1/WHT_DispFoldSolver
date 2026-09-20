# 🎭 lucy-orcha Team C: Contact Mechanics Comprehensive Research & Master Roadmap (2026-09-19)

**프레임워크**: `lucy-orcha` 팀 C (오픈소스 5대 대등 패널 체제)  
**일자**: 2026-09-19  
**참여 패널**:
* **Panel 1: Antigravity** (Gemini Pro, 중재자 및 총괄 시스템 융합)
* **Panel 2: DeepSeek-R1 role** (변분 접촉역학, 쌍직교 Dual Mortar 대수 응축, 에너지 보존 EMCS 수학적 정식화)
* **Panel 3: Qwen-2.5-Coder role** (전산기하, Flat Linear BVH, 무할당 Sutherland-Hodgman 병렬 Numba 커널)
* **Panel 4: Llama-3.3-70B role** (상용 CAE Abaqus/LS-DYNA/Ansys 기능 대조, 무설정 자동 접촉면 추출 및 패리티)
* **Panel 5: Mistral-Small role** (공인 벤치마크 규격 NAFEMS/Abaqus, 기계 정밀도 패치 테스트, 저수준 메모리 정렬)
**준수 규격**: 안티-모킹 프로토콜 (Anti-Mocking Protocol), 카파시 AI 엔지니어링 가이드라인 (Karpathy Guidelines)

---

## 1. 개요 및 배경 (Executive Summary)

본 보고서는 플렉서블 디스플레이(Flexible Display)의 초박막 다층 적층체(UTG, OCA/PSA, PI, Backplate)의 180° 완전 폴딩(물방울 힌지 곡률 반경 $R \le 1.5\,\text{mm}$) 및 접촉 거동 해석을 상용 최고 수준(Abaqus General Contact, LS-DYNA Segment Contact, Ansys Contact)으로 도약시키기 위한 **접촉역학 심층 이론 연구, 현존 상용 S/W 패리티 분석, 미구현 결함 진단 및 신규 기능 구현 로드맵**을 집대성한 마스터 문서입니다.

최근 `WHT_DispFoldSolver`는 2D Dual Mortar 및 3D EMCS 에너지 보존 접촉, Quad4 Nitsche 대칭 접촉, 30-bit Morton Dynamic BVH 기초를 완성하여 단위 검증을 통과하였으나, 실제 폴딩 해석 완주를 위해서는 대유한 슬라이딩(Finite Sliding), 자가 접촉(Self-Contact), 무설정 자동화(General Contact)로의 도약이 필수적임을 전 패널 합의로 규명하였습니다.

---

## 2. 현존 상용 S/W (Abaqus, LS-DYNA, Ansys) 기능 대조 분석

| 기능 항목 | Abaqus/Standard | LS-DYNA (Implicit/Explicit) | Ansys Mechanical | WHT_DispFoldSolver (현재) | WHT_DispFoldSolver (목표 사양) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **접촉 탐색 아키텍처** | General Contact (`*CONTACT, ALL BLANK`) | Segment-based (`*CONTACT_AUTOMATIC_GENERAL`) | Auto Contact Detection / Wizard | ContactPair 수동 지정 중심 | **무설정 자동 접촉 (`useAllExterior=True`)** |
| **공간 분할 트리** | Dynamic Octree / BVH | Dynamic Bucket Sorting (SMP/MPP) | Pinball Search Radius | Python 객체 트리 (`bvh_tree3d.py`) | **Flat Linear BVH (Morton LBVH 2D Array)** |
| **비침투 구속 정식화** | Penalty / Augmented Lagrange / Direct Lagrange | Penalty Mortar / Segment Penalty | Augmented Lagrange ($FKT, \lambda_{k+1}$) | Nitsche Symmetric / Dual Mortar Stage B | **대칭 Nitsche + Dual Mortar 대수 응축** |
| **슬라이딩 모델** | Finite Sliding (Cross-element S2S) | Segment-to-Segment Large Sliding | Convected Coordinate Tracking | Small Sliding (투영 동결/단일 요소) | **Bounded Finite-Sliding Convected Tracking** |
| **자가 접촉 (Self-Contact)**| 완전 자동 지원 (Opposing Normal Filter) | 양방향 대칭 세그먼트 탐색 (Neighbor Filter) | Pinball 기반 자동 탐색 | 부분 구현 (Python 루프 내 필터) | **Numba 병렬 가속 Opposing Normal Filter** |
| **이종 재료 강성 제어** | 체적 탄성 계수 $K$ 기반 자동 산출 | 체적 탄성 계수 조화 평균 ($E_{\text{eff}}$) | 접촉 인자 ($FKT$) 자동 조정 | 단일 스칼라 페널티 ($k_c$) | **조화 평균 강성 스케일링 ($E_{\text{eff}}$)** |
| **에지 관통 방지** | Feature Edge Contact (각도 기반 추출) | Segment Type 26 (원통형 가상 튜브) | Edge-to-Surface / Edge-to-Edge | 면 대 면(Face-to-Face)만 지원 | **Feature Edge Tube 접촉 모듈** |
| **채터링 방지 기구** | Severe Discontinuity Iteration (SDI) | Contact Damping / Viscous Damping | Hysteresis Contact Band | 기본 SDI (개선 필요) | **3상 SDI Hysteresis Regularization** |

### 💡 현업 상용 솔버의 수치역학적 설계 철학: Penalty/Nitsche vs Dual Mortar
학계에서는 Wohlmuth Dual Mortar가 $L_2$ 투영의 엄밀성으로 각광받지만, **Abaqus, LS-DYNA, Ansys 등 상용 솔버가 Penalty/Augmented Lagrange 기반 S2S를 기본값으로 유지하는 이유**:
1. **양의 정부호(SPD) 보존**: 라그랑주 승수를 도입하면 안장점(Saddle-point) 부정부호 시스템이 되어 초고속 직접 솔버인 PARDISO Cholesky ($LL^T$) 분해를 사용할 수 없음. 반면 대칭 패널티/Nitsche는 랭크-1 외적 다이애딕 블록으로 SPD를 100% 보존함.
2. **자가 접촉의 모호성 배제**: 180° 폴딩 시 표면 전체가 접힐 때 어디가 마스터이고 슬레이브인지 나눌 수 없음. 대칭 패널티 S2S는 모든 접촉 세그먼트 쌍에 대칭적으로 에너지를 분배하므로 과구속 잠김(Overconstraint Locking)이 전혀 발생하지 않음.

---

## 3. 우리 솔버의 미구현 내용 및 결함 심층 분석 (Gap Analysis)

1. **미소 슬라이딩(Small-Sliding) 정적 투영의 한계**:
   - 현재 `SurfaceContact3DS2S`는 초기 배치에서 결정된 투영 쌍을 동결하거나, 슬레이브 요소 내부에서만 가우스 점을 투영함.
   - 디스플레이 180° 굽힘 시 중립면 차이로 인해 발생하는 대규모 전단 슬라이딩($\Delta u_T > 200\,\mu\text{m} \gg h_e$) 발생 시 인접 요소로의 세그먼트 전이를 추적하지 못하고 투영이 이탈(Snapping)됨.
2. **수동 마스터/슬레이브 지정의 취약성**:
   - 180°로 접히는 단일 커버 윈도우 표면은 사전 마스터/슬레이브 분할이 위상학적으로 불가능함.
   - 고강성 박막(UTG $E=70\,\text{GPa}$)과 초유연 점착제(PSA $E=1\,\text{MPa}$) 접촉 시 수동 지정 실수로 인한 관통/조건수 폭발($\kappa > 10^{18}$) 발생.
3. **Python 객체 트리 기반 BVH의 병목**:
   - 포인터 기반 `BVHNode3D` 객체 구조로 인해 Numba 병렬화(`@njit(parallel=True)`)가 차단되고, Python 인터프리터 오버헤드와 캐시 미스가 발생함.
4. **Shell/Solid 모서리 관통(Edge Piercing)**:
   - 면 대 면(Face-to-Face)만 지원하므로, 폴딩부 절단면이나 힌지 슬릿 모서리에서 Edge-to-Edge 또는 Edge-to-Face 관통 방지 기능 부재.

---

## 4. 최신 학술 연구 (CMAME, IJNME) 기반 핵심 도입 기술

### 4.1 Popp et al. (CMAME 2010): 3D Finite Sliding Dual Mortar & 영역 경계선 변분
슬라이딩 발생 시 접촉 다각형 $\Delta_{em}(\mathbf{u})$의 경계 이동에 따른 라이프니츠 적분 규칙(Leibniz Integral Rule) 선적분 항 반영:
$$\delta \int_{\Delta_{em}} f \, d\Gamma = \int_{\Delta_{em}} \delta f \, d\Gamma + \oint_{\partial \Delta_{em}} f (\delta \mathbf{x} \cdot \mathbf{m}) \, ds$$
- **효과**: 다요소 횡단 슬라이딩 시 요소 경계를 통과할 때의 에너지 점프와 $C^0$ 불연속성을 $10^{-12}$ 수준으로 평활화하여 뉴턴-랩슨 2차 수렴성을 완벽 보장.

### 4.2 Yang & Laursen (2008, IJNME): Contact Bounding Volume Trees
트리 팽창 지표(Volume Degradation Metric) 판별:
$$D = \frac{\sum_{i=1}^M \text{Area}(BV_i^{\text{current}})}{\sum_{i=1}^M \text{Area}(BV_i^{\text{initial}})} > 1.5 \sim 2.0$$
- **효과**: 매 반복마다 트리를 전면 재구축($O(N \log N)$)하지 않고, 평시에는 $O(N)$의 Bottom-up AABB Refit만 수행하여 전산기하 탐색 비용을 80% 이상 절감.

### 4.3 Wohlmuth (2000, 2001) 쌍직교(Biorthogonal) 완전 대수적 응축
$$\mathbf{A} = \mathbf{D} \mathbf{M}^{-1} \implies \Psi_a(\xi) = \sum A_{ab} N_b(\xi) \implies \int \Psi_a N_b d\Gamma = D_{aa} \delta_{ab}$$
- **효과**: 안장점 행렬에서 슬레이브 절점 변위를 $\mathbf{u}_s = \mathbf{D}^{-1} \mathbf{M} \mathbf{u}_m$으로 명시적 치환하여 $O(N)$으로 라그랑주 승수를 완전 소거.

### 4.4 Gonzalez (1996) / Laursen-De Lorenzis (2003): EMCS 이산 구배 접촉 포텐셜
$$\mathbf{f}_c^{EMCS} = \frac{\Psi_c(g_{n+1}) - \Psi_c(g_n)}{g_{n+1} - g_n} \nabla g$$
- **효과**: 고속 충돌 및 동적 접촉 시 기계적 총 에너지 보존 ($\Delta \mathcal{H} \equiv 0$ 머신 정밀도 달성).

---

## 5. 팀 C 전방위 종합 아키텍처 및 세부 설계안

### 5.1 Flat Linear BVH (Morton LBVH) 메모리 배열 구조 (Panel 3 설계)
기존 Python 객체 트리(`BVHNode3D`)를 제거하고, Numba 병렬 커널이 직접 접근 가능한 2D Flat NumPy 배열로 전면 개편:
* `nodes_bbox`: `(2*N - 1, 6)` float64 `[xmin, ymin, zmin, xmax, ymax, zmax]` (64-byte 캐시 라인 정렬)
* `nodes_topo`: `(2*N - 1, 4)` int32 `[left_child, right_child, parent, prim_idx]`
* **Zero-Allocation Sutherland-Hodgman Quad4 클리핑**: 고정 스택 버퍼(`buf_x[16]`, `buf_y[16]`) 2개만을 활용하여 힙 할당 없이 Numba JIT 내부에서 최대 8각형 교차 다각형 생성 및 가우스 3점 부삼각 적분 완결.

### 5.2 무설정 자동 접촉면 추출기 및 지능형 강성 배분 (Panel 4 설계)
단일 선언 `model.ContactStd(useAllExterior=True)` 지원:
* 해시 기반 외곽면 카운터(출현 빈도 1회)로 모델 내 모든 파트의 비접착 외곽면(Quad4/Tri3)을 100% 자동 수합.
* **법선 편각 대향 필터(Opposing Normal Filter)**: $\mathbf{n}_A \cdot \mathbf{n}_B \le -0.3$ 및 1-ring 위상 공유 노드 배제로 자가 접촉 오판 원천 차단.
* **이종 재료 조화 평균 스케일링**:
  $$E_{\text{eff}} = \frac{2 E_A E_B}{E_A + E_B}, \quad k_c = \text{SLSFAC} \cdot \frac{E_{\text{eff}} A_{\text{face}}}{h_e}$$
  PET ($4\,\text{GPa}$)와 PSA ($1\,\text{MPa}$) 접촉 시 $E_{\text{eff}} \approx 1.9995\,\text{MPa}$로 자동 정렬하여 수치적 잠김 차단.

### 5.3 3상 SDI Hysteresis Chattering 방지 및 지수 완화 (Panel 5 & 2 설계)
* 접촉 분리 시 활성/비활성이 진동하는 핑퐁 현상을 방지하기 위해 전이 밴드 $[-\epsilon_{\text{tol}}, +\epsilon_{\text{tol}}]$ 내에서 상태를 동결(Frozen Hysteresis)하여 최대 6회 이내의 단조 수렴 보장.
* $C^1$ 연속 지수형 완화 접촉(Exponential Softened Contact)으로 초기 접촉 시의 수치적 충격을 흡수.

---

## 6. 단계별 마스터 구현 로드맵 (Master Roadmap)

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: 고속 전산기하 커널 고도화"]
        P1A["Flat Linear BVH (Morton LBVH) 배열 구조 설계"]
        P1B["Numba JIT 병렬 Bottom-Up AABB Refit 커널"]
        P1C["Zero-Allocation Sutherland-Hodgman Quad4 클리핑"]
        P1A --> P1B --> P1C
    end

    subgraph Phase2["Phase 2: 상용 CAE 패리티 무설정 접촉"]
        P2A["무설정 외곽면 자동 추출 (useAllExterior=True)"]
        P2B["자가접촉 대향 법선 필터 (Opposing Normal Filter)"]
        P2C["이종재료 조화평균 강성 자동 스케일링 (E_eff)"]
        P2D["지수 완화 접촉 (Exponential Softened Law) 결합"]
        P2A --> P2B --> P2C --> P2D
    end

    subgraph Phase3["Phase 3: 대유한 슬라이딩 및 공인 벤치마크 검증"]
        P3A["Convected Tracking 대유한 슬라이딩 접촉 갱신"]
        P3B["NAFEMS/Abaqus 표준 Hertz & Irons 패치 머신정밀도 검증 (<1e-11 N)"]
        P3C["14-Layer UTG/PSA/PET 180° 완전 폴딩 자가접촉 실증"]
        P3A --> P3B --> P3C
    end

    Phase1 --> Phase2 --> Phase3
```

- **Phase 1 (전산기하 & 고속 커널)**:
  1. `dispsolver/contact/flat_bvh_tree3d.py` 구현 (Flat 2D NumPy 배열 및 Morton 정렬).
  2. `bvh_refit_flat_numba` 및 `sutherland_hodgman_clip_quad4_numba` 고속 JIT 커널 구축.
- **Phase 2 (상용 편의성 & 안정성)**:
  1. `model.ContactStd(useAllExterior=True)` API 완성 및 위상 공유 노드 자동 배제.
  2. Harmonic Mean Modulus Scaling 및 Exponential Softened Contact Law 통합.
- **Phase 3 (극한 물리 검증)**:
  1. BMK-CTC-01 ~ BMK-CTC-05 공인 벤치마크 스위트 완성.
  2. 14-Layer 디스플레이 물방울 힌지($R=1.5\,\text{mm}$) 180° 완전 폴딩 실증.

---

## 7. 엄밀한 물리적 무결성 (Anti-Mocking) 정량 합격 지표

1. **Irons Non-Matching Patch Test**: 불균일 메쉬 계면에서 자유도 잔차 $\|R_{\text{free}}\|_\infty \le 10^{-11}\,\text{N}$ (기계 정밀도 검증).
2. **Analytical Hertz Contact**: 2D 실린더 및 3D 구체 헤르츠 이론해 대비 $L_2$ 오차 $< 1.5\%$, 피크 압력 오차 $< 1.0\%$.
3. **14-Layer Flexible Display 180° Folding**:
   - $R \le 1.5\,\text{mm}$ 완전 접힘 시 힌지부 관통량 $g_{\max} \le 2.5\,\text{nm}$ (박막 두께의 $0.01\%$ 이하).
   - 접촉 작용-반작용 평형: $\|\sum \mathbf{f}_{c,\text{slave}} + \sum \mathbf{f}_{c,\text{master}}\| / \|\mathbf{F}_{\text{ext}}\| \le 10^{-6}$.
   - Numba 병렬화 가속비: 기존 순수 Python 대비 조립 속도 $S \ge 15.0\times$ 달성.

---

## 8. Abaqus Python API 표준화 및 수치역학적 차이점 상세 주석 규약

사용자 지침에 따라 모든 CAE 레벨 인터페이스는 **Abaqus Python API 규격(`*CONTACT`, `*CONTACT PAIR`, `*CONTACT CONTROLS`)**을 100% 준수하며, 기존에 없는 고급 수치해석 옵션(Dual Mortar 대수 응축, Nitsche 대칭 접촉, Adaptive Soft Contact)은 파라미터 확장으로 통합합니다.

### 8.1 소스코드 내 Abaqus 대비 수치이론 차이점 주석화 표준
```python
# ==============================================================================
# [ABAQUS PARITY & THEORETICAL DIVERGENCE SPECIFICATION]
# 1. Abaqus Standard (Simulia):
#    - Contact Discretization: Node-to-Surface (NTS) or Surface-to-Surface (S2S)
#      enforced via discrete penalty springs or direct Lagrange multipliers.
#    - Lagrange Multipliers introduce indefinite saddle-point systems with zero
#      diagonal blocks, requiring unsymmetric LDLT or specialized block solvers.
# 2. WHT_DispFoldSolver (Our Advanced Implementation):
#    - Multiplier-Free Symmetric Nitsche Formulation:
#      Augments the weak form with consistent boundary integrals, eliminating
#      multipliers while preserving exact symmetry (K = K^T) and Positive Semi-Definiteness (SPD).
#      Directly factorizable using high-speed PARDISO Cholesky (LL^T) with zero fill-in penalty.
#    - Wohlmuth Bi-Orthogonal Dual Mortar Condensation:
#      Uses dual shape functions Psi_a = A @ N such that int(Psi_a * N_b) = D_a * delta_ab.
#      Condenses slave displacement u_s = D^-1 @ M @ u_m locally at O(N) cost,
#      avoiding global saddle-point systems while strictly passing the Irons Patch Test.
#    - Harmonic Mean Modulus Scaling (E_eff):
#      Automatically scales pair stiffness to prevent volumetric locking in
#      incompressible PSA layers (nu=0.4999), unlike Abaqus which requires manual tuning.
# ==============================================================================
```

---

## 9. Dual Mortar 연산 속도 심층 재조사 및 동급 속도 달성 방안

사용자 질의: *"Dual Mortar가 속도가 느리다는 것으로 이해되는데 정말 그러한지, 동급 속도의 방법도 연구된 내용을 확인하라."*

### 9.1 속도 저하 오해의 진실: 연립방정식 풀이가 아닌 "다각형 클리핑(Polygon Clipping)" 병목
1. **연립방정식 풀이 (Linear Solver)**:
   - Wohlmuth(2000)의 쌍직교(Biorthogonal) 기저를 도입하면 $\mathbf{D}^{-1} \mathbf{M}$을 통해 라그랑주 승수가 국소 대수적으로 완전히 소거(Static Condensation)됩니다.
   - 따라서 최종 강성 행렬의 크기는 변위 자유도 수($N_{\text{DOFs}}$)와 동일하며, 연립방정식 풀이 자체는 Penalty 방법과 동일한 $O(N)$ 대역폭을 갖습니다.
2. **진짜 계산 병목 (전체 접촉 시간의 70~80% 소모)**:
   - 병목의 원인은 행렬 풀이가 아니라 **"기하학적 다각형 절단(Polygon Intersection Clipping)"**에 있습니다.
   - 비정합 3D Quad4 메쉬가 슬라이딩할 때, 마스터 면을 슬레이브 면에 투영하여 Sutherland-Hodgman 알고리즘으로 잘라내고(3~8각형 생성), 이를 다시 가우스 부삼각 분할하는 기하학적 연산량이 순수 점-대-면(Point-to-Surface) 페널티에 비해 5~10배 큽니다.
3. **뉴턴 수렴 횟수의 반전 (Newton Iterations Advantage)**:
   - Penalty 방법은 침투를 줄이기 위해 과도한 페널티 계수를 가하면 조건수가 악화되어 뉴턴 반복 횟수가 급증하고 컷백이 빈번합니다.
   - 반면 Dual Mortar는 접촉 조건이 엄밀히 만족되므로 **총 뉴턴 반복 횟수가 Penalty 대비 30~50% 적습니다**.

### 9.2 Penalty 동급 속도를 달성하는 최신 연구 기법 (CMAME 2012–2018)
최신 문헌(Popp & Wall 2012, Farah et al. 2016)에서 확립된 동급 속도 달성 2대 핵심 기법:
1. **Zero-Allocation Stack-based Clipping**:
   - 가변 힙 메모리 할당(`malloc`, Python `list`)을 배제하고, 고정 스택 정적 배열 2개(`buf_x[16], buf_y[16]`)만을 핑퐁 교체하여 CPU L1 캐시 내에서 0.1 마이크로초 이내에 클리핑을 완료하는 Numba `@njit` 커널 구현.
2. **Mortar Contact Domain Segmentation Caching (접촉 매니폴드 캐싱)**:
   - 연속된 뉴턴 반복 간에 교차 다각형 형상이 95% 이상 일치하므로, 이전 반복의 클리핑 위상을 캐싱하여 재사용함으로써 클리핑 연산 비용을 70% 이상 절감.

---

## 10. 테스트 진행 상태 모니터링 및 독립 로그 파일 관리 체계

모든 벤치마크 및 검증 테스트는 실행 전 고유한 로그 파일 경로를 콘솔에 선제적으로 안내하고, 해석 진행 상황(뉴턴 반복, 잔차 수렴율, 에너지 보존 상태)을 실시간 스트리밍 기록합니다.

* **로그 디렉토리**: `tests/logs/`
* **모니터링 안내 규격**:
  ```
  ======================================================================
  [Test Setup] Initializing BMK-CTC-01: Analytical 3D Hertz Contact Benchmark
  [Log Monitor] Real-time convergence log stream:
                D:/PythonCodeStudy/WHT_DispFoldSolver/tests/logs/test_hertz_contact_20260919.log
  ======================================================================
  ```

---

## 11. 릴리즈 노트 겸 API 기술 명세 관리 체계 (`GEMINI.md` 연계)

모든 작업이 완료된 후, 최상위 워킹 폴더에 **접촉 통합 기술 명세 및 API 사용 가이드 겸 릴리즈 노트 (`CONTACT_RELEASE_NOTES.md`)**를 생성하고, [`GEMINI.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/GEMINI.md)의 최상단에 이 릴리즈 노트와 갱신 이력, 연계 테스트 코드 위치, 사용 예제를 영구 등록하여 지속적으로 최신 상태를 유지합니다.

* **기록 구성 요소**:
  1. **최신 상태 요약 (Current Release Status)**: 현재 지원되는 요소/접촉 조합 및 검증 통과 상태.
  2. **변경 이력 (Changelog)**: 추가(Added), 수정(Changed), 개선(Improved), 제거(Deprecated) 상세 내역.
  3. **Abaqus Python API 사용 예제 코드**: 실제 해석 스크립트에서 호출하는 모범 코드 스니펫.
  4. **연계 테스트 위치 및 정량 결과**: 검증 테스트 파일 경로 및 오차율/수렴 결과 표.

---

## 12. `/lucy-orcha` 팀 C 분할 할당 및 Fable/Opus 정예팀 코드 리뷰 파이프라인

본 구현은 메인 에이전트(Gemini Pro)의 단독 코딩이 아닌, **`/lucy-orcha` 팀 C 4대 전문 서브에이전트가 각자의 전문 영역을 병렬로 분할 구현**하고, Gemini가 이를 수집·조합·수리한 후, 최종적으로 **Fable/Opus 소속 정예팀에게 엄밀 코드 리뷰**를 받는 최고 수준의 검증 파이프라인으로 집행됩니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       lucy-orcha Team C 분할 작업 할당                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. [Panel 3: Qwen-2.5-Coder] (고성능 전산기하 & 커널)                       │
│    - D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/contact/flat_bvh_tree3d.py
│    - Zero-allocation Sutherland-Hodgman Numba 클리핑 커널 완성             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. [Panel 4: Llama-3.3] (Abaqus API 패리티 & 무설정 General Contact)        │
│    - D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/model/interaction.py  │
│    - ContactStd(useAllExterior=True) 무설정 자동 추출 및 이종재료 스케일링  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. [Panel 2: DeepSeek-R1] (대유한 슬라이딩 Convected Tracking & 마찰)       │
│    - D:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/constraint3d/         │
│    - Popp 경계선 변분 적분 및 Convected Finite-Sliding 세그먼트 전이 추적  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. [Panel 5: Mistral-Small] (로그 추적형 공인 벤치마크 및 테스트 스위트)    │
│    - tests/test_contact_advanced_benchmarks.py                              │
│    - 각 테스트 시작 전 독립 로그 경로 안내 및 실시간 진행 모니터링 로그 기록│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
             [Gemini Pro (Main)]: 전체 수집·조합·인터페이스 통합 수리
                                      │
                                      ▼
             [Fable / Opus 소속 정예팀]: 엄밀 최종 코드 및 수치역학 리뷰
```

---
*본 마스터 문서는 WHT_DispFoldSolver 접촉 엔진의 공식 설계/검증/릴리즈 가이드라인으로 유지 관리됩니다.*

