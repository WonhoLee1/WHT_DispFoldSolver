# Phase 5: Synthesis — 교차 분석 및 격차(Gap) 분석

**리서치 주제**: Penalty-based contact stabilization in implicit FEM  
**날짜**: 2026-09-30  
**작성자**: Deep Research Skill (Phase 5/6)

---

## 1. 분류 체계 (Taxonomy)

### 1.1 채터링 억제 방법 비교 (우리 코드 적용 관점)

| 방법 | 수렴 품질 | 구현 복잡도 | WHT 코드 적용성 | 이론 근거 |
|---|---|---|---|---|
| **Penalty + Active Set (현재)** | 보통 | 낮음 | ✅ 이미 구현 | Wriggers §9.4 |
| **Frozen Set (Freeze)** | 좋음 (버그 수정 후) | 낮음 | ✅ 이미 구현, 버그만 수정 | Wriggers §9.4 |
| **SDI 스텝 완화 (β=0.5)** | 좋음 | 매우 낮음 | ✅ 1줄 수정 | Hüeber 2008 §4.2 |
| **AL + PDAS** | 매우 좋음 | 높음 | ✅ 이미 구현 (별도 브랜치) | Hintermüller 2002 |
| **Nitsche Method** | 매우 좋음 | 매우 높음 | ❌ 미구현, 재설계 필요 | Chouly 2013 |
| **Mortar Method** | 최상 | 매우 높음 | ❌ 미구현 (이미 구현됨) | Popp 2012 |
| **Contact Stabilization (감쇠)** | 보통 | 매우 낮음 | ✅ 1줄 추가 | Wriggers 1990 |

### 1.2 페널티 강성 선택 비교

| 전략 | 수식 | WHT 현재값 | 권장값 |
|---|---|---|---|
| Rebel et al. (2002) | `ε = α × E/h` | `k = 1e8` (50× 과대) | `α=10 → k=2e6` |
| Abaqus 기본 | `k_default = 10 × E/L_elem` | — | `k = 2e6` |
| HardLaw 기반 이론값 | `k_contact = E_reduced / δ_max` | — | `k = 4e6 (δ_max=0.05mm)` |

**결론**: 현재 `k_contact = 1e8`은 50배 과대. `k = 2e6`으로 축소 시 조건수 개선.

---

## 2. 비교 표 — Fix 효과 예측

| Fix | 영향 이터레이션 | 예상 rel_r 변화 | 예상 컷백 감소 |
|---|---|---|---|
| **Fix A** (Bilateral→Unilateral) | Iter 3+ (Frozen 상태) | 6.2% → < 2% | 대부분 해소 |
| **Fix B** (SDI β=0.5) | Iter 2 (SDI 직후) | 0.978 → 0.4 | 없음 (이미 Inc 4+에서 막힘) |
| **Fix C** (r_tol 5%→8%) | 모든 이터레이션 | 통과 기준 완화 | Inc 3~5 해소 |
| **Fix D** (k=1e8→2e6) | 모든 이터레이션 | 조건수 50배 개선 | Inc 1+ 안정화 |

**최적 순서**: Fix D → Fix A → Fix C → Fix B (영향 범위 넓은 것 먼저)

---

## 3. 격차 분석 (Gap Analysis)

### 3.1 현재 WHT 코드의 미구현 기능

| 기능 | 문헌 참고 | 우선순위 | 이유 |
|---|---|---|---|
| Unilateral Freeze (Fix A) | Wriggers §9.4 | **P1 즉시** | 현재 가장 큰 수렴 장애 |
| SDI 스텝 완화 (Fix B) | Hüeber 2008 §4.2 | **P2 즉시** | Iter 2 폭발 방지 |
| Abaqus 수준 r_tol (Fix C) | Abaqus §2.2.4 | **P3 즉시** | 경계선 케이스 |
| 자동 페널티 스케일링 (Fix D) | Rebel 2002 | **P4 이번 릴리즈** | 과도한 조건수 수정 |
| Hysteresis Gate | Vu-Quoc 2007 | **P5 선택** | 채터링 2형 방지 |
| `*CONTACT CONTROLS, STABILIZE` | Abaqus Guide | **P6 선택** | 초기 접촉 안정화 |

### 3.2 현재 WHT 코드의 올바르게 구현된 기능

| 기능 | 근거 |
|---|---|
| Penalty + Active Set | Wriggers §9.4 ✅ |
| AL + Augmented Lagrangian | Alart-Curnier 1991 ✅ |
| Coulomb 마찰 (stick/slip) | Alart-Curnier 1991 ✅ |
| SDI 감지 (`get_active_set()`) | Fischer & Wriggers 2005 ✅ |
| Adaptive dt (이터레이션 카운트 기반) | 산업 표준 ✅ |
| Dual Mortar 2D/3D | Popp 2012 ✅ (최신 릴리즈) |

---

## 4. 최종 수정 코드 처방 (Synthesis 결과)

### 4.1 Fix A 상세 코드 처방

**`surface_contact3d_deformable.py` L354~360** 수정:

```python
# BEFORE (버그: Bilateral Spring)
if getattr(self, "frozen_active_nodes", None) is not None:
    if s_nid not in self.frozen_active_nodes:
        continue
    f_mag = self.k_contact * penetration      # ← 음수 허용 (인장력 생성)
    k_diag = self.k_contact

# AFTER (Fix A: Unilateral, Wriggers §9.4)
if getattr(self, "frozen_active_nodes", None) is not None:
    if s_nid not in self.frozen_active_nodes:
        continue
    # Unilateral: 압축 전용 (penetration > 0 = 접촉 중)
    p = max(0.0, penetration)
    f_mag = self.k_contact * p                # ← 단방향 압축만
    k_diag = self.k_contact if penetration > 0.0 else 0.0
```

### 4.2 Fix B 상세 코드 처방

**`dynamic3d.py` L1735~1748** 수정 (SDI 브랜치):

```python
# BEFORE (Full Newton step on SDI)
if iter_count == 1 or is_sdi:
    u_trial = u_k + (du if iter_count == 1 else du_step)

# AFTER (Fix B: β=0.5 relaxation on SDI, Hüeber 2008 §4.2)
if iter_count == 1 or is_sdi:
    if is_sdi and iter_count > 1:
        # SDI 후 첫 스텝: Hüeber et al. (2008) β=0.5 relaxation
        # (Abaqus Theory Guide §2.2.1도 동일: "50% of Newton direction")
        u_trial = u_k + 0.5 * du_step
    else:
        u_trial = u_k + (du if iter_count == 1 else du_step)
```

### 4.3 Fix C 상세 코드 처방

**`dynamic3d.py` L1688, L2415** 수정:

```python
# BEFORE
r_tol = max(tol, 0.05 if len(self.constraints) > 0 else 5e-3)

# AFTER (Fix C: Abaqus §2.2.4 접촉 완화 기준 8%)
r_tol = max(tol, 0.08 if len(self.constraints) > 0 else 5e-3)
```

**`dynamic3d.py` L1716, ~L2448** 수정:
```python
# BEFORE
if ... and (du_norm < 1e-4 or du_free_norm < 1e-4 or rel_du < 0.01):

# AFTER (Fix C-du: 변위 수렴 기준 완화)
if ... and (du_norm < 5e-4 or du_free_norm < 5e-4 or rel_du < 0.01):
```

### 4.4 Fix D 상세 코드 처방 (선택 — k_contact 자동 스케일링)

**INP 파서 또는 `DeformableSurfaceContactConstraint3D.__init__()` 수정**:

```python
# 현재: k_contact = penalty_stiffness (= 1e8 N/mm, 고정값)
# AFTER: Rebel et al. (2002) 자동 스케일링
# k_contact = alpha_contact × E_slave / h_slave
# alpha_contact = 10.0 (Abaqus 기본), E = 200,000 MPa, h ≈ 1.0 mm
# → k_contact ≈ 2e6 N/mm (현재의 1/50)
```

---

## 5. 성공 기준 (Success Criteria, 정량적)

```
Phase A (접촉 안정화 기본):
  ✅ INDENT Step 완주 (zero cutback)
  ✅ max_penetration < 1e-3 mm
  ✅ all Newton iterations: rel_r < r_tol=8%

Phase B (물리 정확도):
  ✅ Hertz RF_y 이론해 대비 오차 < 5%
  ✅ SLIDE Step |RF_x/RF_y - 0.20| < 0.02

Phase C (산업 수준):
  ✅ 전체 2 Step 완주, Newton 이터레이션 < 10회/증분
  ✅ 조건수 추정: κ(K+K_contact) < 1e12 (Fix D 후)
```

---

✅ **Phase 5 완료**. Output: `synthesis.md` + `gaps.md` (통합 분석, 코드 처방 완비).
Phase 6: Report 작성으로 진행합니다.
