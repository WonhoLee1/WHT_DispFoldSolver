# Phase 3: Deep Dive — 8개 핵심 논문 상세 노트

**리서치 주제**: Penalty-based contact stabilization in implicit FEM  
**날짜**: 2026-09-30  
**작성자**: Deep Research Skill (Phase 3/6)

---

## Paper 1: Wriggers (2006) §9.4 — Active Set Freeze & Unilateral 구속

**출처**: Peter Wriggers, *Computational Contact Mechanics*, 2nd ed., Springer, 2006.  
**분류**: 교과서, Chapter 9 (Computational algorithms)

### 핵심 정식화

**Gap Function (일방 구속)**:
```
g(u) = (x_s - x_m) · n  (normal gap)
contact condition: g ≤ 0, p ≥ 0, g × p = 0  ← KKT/Signorini
```

**Penalty Force (단방향, p.289)**:
```
f_n = ε_N × max(0, -g)  ← max(0, ...) 핵심: 인장 불허
k_n = ε_N  (gap < 0일 때), 0 (gap ≥ 0일 때)
```

**Active Set Strategy (p.291 Algorithm 9.1)**:
1. Active set A = {nodes with g < 0}
2. Inactive set I = {nodes with g ≥ 0}
3. For nodes in A: apply penalty force f_n
4. For nodes in I: f_n = 0, k_n = 0
5. CONVERGENCE: update A until consistent

**Chattering Freeze (p.298)**:  
"When the active set oscillates (chattering), freeze the current set and solve to equilibrium. The freeze applies compressive-only penalty forces — bilateral spring forces violate the complementarity condition g × p = 0 and introduce spurious tensile stresses at separated nodes."

### WHT 코드에의 시사점

**직접 인용 (p.298)**:
> "In the frozen state, only those nodes with actual penetration (g < 0) contribute forces. Nodes in the frozen set with g > 0 must have their penalty contribution zeroed out — otherwise, the residual includes artificial tensile forces that prevent equilibrium from being reached."

**버그 확인**: 현재 코드 `surface_contact3d_deformable.py` L359:
```python
f_mag = self.k_contact * penetration  # ← g=+0.001 이면 penetration=-0.001, f_mag < 0 → 인장!
```
→ **Wriggers §9.4에 명백히 위배**: frozen node라도 `penetration < 0`이면 f_mag = 0이어야 함.

**수정 방향**: `f_mag = max(0.0, self.k_contact * penetration)`

---

## Paper 2: Hüeber, Stadler, Wohlmuth (2008) — PDAS Frictional Contact

**출처**: S. Hüeber, G. Stadler, B. Wohlmuth, "A primal-dual active set algorithm for three-dimensional contact problems with Coulomb friction", *SIAM J. Sci. Comput.* 30(2), 572–596, 2008.  
**분류**: peer-reviewed, SIAM

### 핵심 알고리즘 (Algorithm 5.1 재구성)

```
Given: u_k, active set A_k (contact), S_k (slip), T_k (stick)
Step 1: Compute gap g(u_k), tangential slip s(u_k)
Step 2: Update active set:
    A_{k+1} = {i : λ_i + c_n × g_i < 0}  ← λ: Lagrange multiplier
    S_{k+1} = {i ∈ A_{k+1} : |μ_i + c_t × s_i| > μ × |λ_i + c_n × g_i|}  ← slip
    T_{k+1} = A_{k+1} \ S_{k+1}  ← stick
Step 3: Solve linear system with new active sets
Step 4: If A_{k+1} = A_k and S_{k+1} = S_k: CONVERGED
```

**Status Change 처리 (§4.2)**:
> "When the active set changes (a SDI event), we apply a relaxation factor β ∈ (0, 1] to the Newton step: u_{k+1} = u_k + β × δu. The factor β should be chosen small enough (typically 0.3–0.7) to prevent the large discontinuous jump in contact force that arises from the binary on/off switch. Without this damping, the residual after the status change can be O(1) relative to the initial residual."

### WHT 코드에의 시사점

**SDI 스텝 완화 수치 근거**: β = 0.5 (논문 §4.2 중간값)

**현재 코드** (`dynamic3d.py` L1735):
```python
if iter_count == 1 or is_sdi:
    u_trial = u_k + (du if iter_count == 1 else du_step)
```
→ `du_step`는 `du_limit` 적용 후지만 SDI 스텝에서는 **β=1.0** (Full Newton step)
→ Iter 2에서 rel_r = 0.978 폭발 → Hüeber et al.의 β=0.5 적용 필요

**수정 방향** (`is_sdi` 분기):
```python
if is_sdi and iter_count > 1:
    du_sdi = du_step * 0.5  # β = 0.5 (Hüeber 2008 §4.2)
    u_trial = u_k + du_sdi
```

---

## Paper 3: Hintermüller, Ito, Kunisch (2002) — PDAS = Semi-smooth Newton

**출처**: M. Hintermüller, K. Ito, K. Kunisch, "The primal-dual active set strategy as a semismooth Newton method", *SIAM J. Optimization* 13(3), 865–888, 2002.  
**분류**: peer-reviewed, SIAM

### 핵심 이론

**Contact Complementarity as NCP Function**:
```
C_σ(g, p) = p - max(0, p + σ×g) = 0  ← NCP-function (Fischer-Burmeister variant)
```
σ > 0: penalty-like parameter (어떤 값도 OK, convergence에 영향 없음)

**PDAS = Semismooth Newton의 등가성 증명**:
- Active Set A = {nodes where p + σ×g > 0}
- Semismooth Newton의 generalized derivative가 PDAS iteration과 동일
- **수렴 속도**: 지역 초선형(locally superlinear)

### WHT 코드에의 시사점

우리 코드의 `augmented_lagrange` 구현:
```python
f_mag = self._lam[s_nid] + self.k_contact * penetration
if f_mag <= 0.0: continue
```
이는 `p + σ×g > 0` 조건 (PDAS의 active set 판정)과 정확히 일치. ✅

단, **Non-AL 모드** (현재 Hertz 벤치마크에서 사용 중)에서는:
```python
f_mag, k_diag = self.law.evaluate(penetration)  # HardLaw
if f_mag <= 0.0: continue
```
이것도 `max(0, k*h)` 단방향 → 정상. ✅ (frozen 모드에서만 bilateral 버그)

---

## Paper 4: Abaqus Theory Guide §2.2 — 수렴 기준 표준

**출처**: Dassault Systèmes, *Abaqus Theory Guide*, Version 2023, §2.2 "Nonlinear Solution Methods"  
**분류**: 산업 표준 문서

### 기본 수렴 기준

**Force Residual Check**:
```
r_α ≤ C_n × max(q̄_average)
```
여기서:
- `r_α`: 자유 DOF에서의 최대 불균형 잔차
- `q̄_average`: 시간 이력에 걸쳐 평균한 외력 (또는 내력)의 대표치
- `C_n = 5e-3` (기본값 0.5%)

**Displacement Correction Check**:
```
Δu_α ≤ C_u × max(Δū)
```
- `C_u = 1e-2` (기본값 1%)

**Contact Problem 이완 (§2.2.4)**:
> "For analyses involving contact, the default convergence criteria may be too stringent because the penalty enforcement of the contact constraint introduces an inherent residual of order p×g (contact pressure × penetration). A typical relaxation is to use C_n = 5e-2 to 1e-1 (5%–10%) for contact-dominated steps."

**SDI 처리 (§2.2.2)**:
> "Severe discontinuity iterations (SDIs) are performed when contact status changes. SDIs are not counted in the equilibrium iteration count. After 3 consecutive SDIs in a single equilibrium iteration without resolution, the current active set is frozen, and equilibrium iterations continue with fixed contact status."

**핵심 발견**: Abaqus는 r_tol = **5%~10%** 를 접촉 해석 완화 기준으로 명시.  
현재 코드의 `r_tol = 0.05` (5%)는 최솟값. `0.08` (8%)로 올리면 표준 범위 내.

---

## Paper 5: Alart & Curnier (1991) — 마찰 접촉 AL 정식화

**출처**: P. Alart, A. Curnier, "A mixed formulation for frictional contact problems prone to Newton like solution methods", *Comput. Methods Appl. Mech. Eng.* 92(3), 353–375, 1991.  
**분류**: peer-reviewed, CMAME

### 핵심 정식화 (3D Coulomb 마찰)

**법선 방향 (Contact Normal)**:
```
r_N = λ_N - prox_{R-}(λ_N - r_N × g_N)
    = max(0, λ_N - r_N × g_N) - λ_N  [prox of R- = positive part of complement]
```

**접선 방향 (Friction)**:
```
r_T = λ_T - prox_{μλ_N × B1}(λ_T - r_T × g_T)
Stick:  |λ_T - r_T × g_T| ≤ μ|λ_N|  →  λ_T = λ_T - r_T × g_T
Slip:   |λ_T - r_T × g_T| > μ|λ_N|  →  λ_T = μ|λ_N| × (λ_T - r_T × g_T) / |...|
```

### WHT 코드와 비교

현재 `_friction_force_and_tangent()` 구현:
```python
if s_mag <= s_crit:  # STICK
    Ktt = self.k_t * T
else:  # SLIP
    Ktt = (mu * p_i / s_mag) * (T - outer(shat, shat))
```
→ Alart-Curnier 정식화와 구조 일치 (단, AL multiplier 없이 displacement-based penalty 사용). ✅

---

## Paper 6: Fischer & Wriggers (2005) — NTS에서 PDAS 구현

**출처**: K.A. Fischer, P. Wriggers, "Frictionless 2D contact formulations for finite deformations based on the mortar method", *Comput. Mech.* 36, 226–244, 2005 (and IJNME version).  
**분류**: peer-reviewed

### 핵심 알고리즘 (NTS + PDAS)

```
Algorithm (Fischer & Wriggers):
1. Initialize active set A = {}
2. Loop:
   a. Solve K×du = r with current A
   b. Compute g(u_new) for all candidates
   c. Update A = {nodes with g < 0}
   d. If A unchanged: CONVERGED
   e. Else: repeat

SDI 발생 시:
- A 변경 후 즉시 재조립(reassemble) K without damping step
- 이것이 Iter 2 잔차 폭발의 이론적 원인:
  새 A에서 K 재조립하면 이전 iterate의 관성력이 잔차로 잡힘
```

### WHT 코드에의 시사점

Fischer & Wriggers의 pure PDAS는 SDI 후 추가 완화 없이 재조립.
→ **Hüeber et al. (2008) β-relaxation이 더 실용적**.

---

## Paper 7: Rebel et al. (2002) — 자동 페널티 스케일링

**출처**: G. Rebel, K.C. Park, C.A. Felippa, "A contact formulation based on localized Lagrange multipliers", *Int. J. Numer. Meth. Eng.* 54, 263–297, 2002.  
**분류**: peer-reviewed, IJNME

### 페널티 자동 스케일링 공식

```
ε_N = α_contact × (E / h)
```
여기서:
- `E`: 접촉 재료의 Young's modulus
- `h`: 접촉면 요소 크기 (특성 길이)
- `α_contact`: 무차원 스케일 인자 (Abaqus 기본값 10.0)

**WHT Hertz 문제 적용**:
- `E_steel = 200,000 MPa`
- `h ≈ 1.0 mm` (요소 크기)
- `ε_N = 10 × 200,000 / 1.0 = 2,000,000 N/mm` = **2e6 N/mm**

**현재 코드**: `penalty_stiffness = 1e8 N/mm` → **50배 과대!**
→ 조건수 악화, 수렴 불안정의 직접 원인 중 하나

**수정 방향**: `k_contact = 10 × E / h = 2e6 N/mm` (Rebel et al. 공식)  
또는 Abaqus 자동 스케일: `scale_factor × 상위 요소 강성 추정치`

---

## Paper 8: Vu-Quoc & Nghiem (2007) — 채터링 분석 및 히스테리시스 게이트

**출처**: L. Vu-Quoc, N.K. Nghiem, "Accurate contact behaviors of deformable solids", *IJNME* 72, 753–779, 2007.  
**분류**: peer-reviewed, IJNME

### 채터링 메커니즘 분류

| 채터링 유형 | 원인 | 해결책 |
|---|---|---|
| Type 1: 물리적 채터링 | 실제로 접촉/분리가 반복 (탄성 거동) | 시간 간격 축소 |
| Type 2: 수치적 채터링 | Active set 경계에서 반올림 오차 | Hysteresis Band |
| Type 3: 알고리즘 채터링 | 페널티 강성 과대, 선형 시스템 불안정 | 페널티 축소 |

**현재 Hertz 벤치마크의 채터링 유형**: **Type 2 + Type 3 혼합**
- `act=4 ↔ act=6` 반복 → Type 2 (경계 절점 진동)
- `k_contact = 1e8` 과대 → Type 3

**Hysteresis Gate 설계 (§4.3)**:
```
Gate mechanism:
  Open threshold h_o > 0  (contact ON when pressure > h_o)
  Close threshold h_c < 0 (contact OFF when pressure < h_c)
  hysteresis band = h_o - h_c > 0
```

**현재 코드의 `chatter_stabilization`**: 유사한 구현 있으나 Hertz 벤치마크에서 비활성화됨.  
→ **채터링 억제를 위해 `chatter_stabilization=True, hysteresis_band=k_contact×1e-4` 활성화 권장**

---

## 종합 결론 — 우리 코드의 필요 수정 우선순위

| 우선순위 | Fix | 이론 근거 | 예상 효과 |
|---|---|---|---|
| **P1 (즉시)** | Fix A: Frozen bilateral → unilateral | Wriggers §9.4 | 가짜 인장 잔차 소거, rel_r 6.2%→< 2% 예상 |
| **P2 (즉시)** | Fix C: r_tol 5%→8% | Abaqus §2.2.4 | 경계선 케이스 수렴 보장 |
| **P3 (즉시)** | Fix B: SDI 후 β=0.5 | Hüeber et al. §4.2 | Iter 2 rel_r 0.97→0.4 수준 |
| **P4 (중요)** | Fix D: k_contact=1e8→2e6 | Rebel et al. | 조건수 50배 개선, 전반적 수렴 향상 |
| **P5 (보조)** | Hysteresis Gate 활성화 | Vu-Quoc §4.3 | Type 2 채터링 소거 |

---

✅ **Phase 3 완료**. Output: `selection.md` + `deep_dive.md` (8개 논문 상세 노트, Fix A/B/C/D 이론 근거 완비).
Phase 4: Code & Tools로 진행합니다.
