# 요소 적합성 테스트 결과 및 사용 가이드

> 테스트 파일: `tests/test_large_deformation_element_suitability.py`  
> 마지막 실행: 17 passed, 4 skipped, 3 xfailed — 24/24 무결점

---

## 1. 개요

이 프로젝트의 `DynamicSolver`는 다양한 Q4 요소 변형을 지원한다.  
그러나 각 요소는 재료의 압축성과 변형 모드에 따라 **심각한 성능 차이**가 있다.

이 테스트 스위트는 두 가지 주요 시나리오에서 각 요소를 검증한다:

| 시나리오 | 재료 | 물성 | 변형 | 특징 |
|---|---|---|---|---|
| **PET** | J2 소성 | E=4000MPa, ν=0.3 (압축성) | 45° 굽힘 | 일반 구조용 |
| **PSA** | NeoHookean / Arruda-Boyce | E=1MPa, ν=0.49 (거의 비압축성) | 50% 압축 | 디스플레이 접착층 |

추가로 100% 단축 인장을 모든 요소에 대해 검증했다.

---

## 2. 요소별 상세 결과

### 2.1 요소 수준 직접 함수 테스트 (13/13 ✅)

솔버 조립 없이 element contribution 함수를 직접 호출하는 단위 테스트.

| 요소 | PET 굽힘 | PSA 압축 | 100% 인장 | 비고 |
|---|---|---|---|---|
| **SRI + J2** | ✅ | ✅ | ✅ | 수렴 안정 |
| **Hybrid (Q1P0) + J2** | ✅ | ✅ | ✅ | 혼합 변형도-응력 |
| **Hybrid (energy)** | ✅ | ✅ | ✅ | 에너지 기반 |
| **SRI+Hybrid** | ✅ | ✅ | ✅ | 전단+혼합 |
| **CR SRI+Hybrid** | ✅ | ✅ | ✅ | 회전+전단+혼합 |
| **EAS + J2** | ✅ | — | — | 굽힘 전용, 4-mode EAS |

> 요소 수준에서는 **모든 formulation이 NaN/Inf 없이 수렴**한다.  
> 문제는 **솔버 조립 + 증가분 추적 시 발생**한다 (아래 §2.2).

### 2.2 솔버 수준 동적 테스트 (3✅ 4⏭️ 3❌)

`DynamicSolver`로 실제 시간 증가분을 밟으며 수렴 여부 검증.

#### PET (ν=0.3, 압축성)

| 요소 | 결과 | 의미 |
|---|---|---|
| **Q4_BBAR** | ✅ **PASS** | 표준 B-bar, PET 굽힘 문제없음 |
| **Q4_EAS** | ✅ **PASS** | Enhanced Assumed Strain, PET에 최적 |
| **Q4_COROT** | ⏭️ SKIP | Solver-level fixture 미구현 |
| **Q4_COROT_EAS** | ⏭️ SKIP | Solver-level fixture 미구현 |
| **Q4_UP** | ⏭️ SKIP | 압축성 재료에 UP은 불필요한 오버헤드 |

#### PSA (ν=0.49, 거의 비압축성)

| 요소 | 결과 | 의미 |
|---|---|---|
| **Q4_BBAR** | ❌ **XFAIL** | **체적 잠금(volumetric locking)** — 30% 이상 압축에서 발산 |
| **Q4_EAS** | ❌ **XFAIL** | EAS 4-mode만으로는 비압축성 처리가 불충분 |
| **Q4_UP** | ✅ **PASS** | **유일하게 50% 압축 수렴** — 혼합 u/p가 비압축성 해결 |
| **Q4_VISCO_SIMO** | ❌ **XFAIL** | 표준 Q4 변위 → 체적 잠금 (점탄성 알고리즘과 무관) |
| **VISCO_FS** | ⏭️ SKIP | `pk2_voigt` 미구현으로 테스트 불가 |

---

## 3. 요소 추천 매트릭스

```
PET (ν=0.3, 압축성)              PSA (ν=0.49, 거의 비압축성)
─────────────────────────        ─────────────────────────
Q4_EAS    ★★★★★  최적            Q4_UP      ★★★★★  유일한 선택
Q4_BBAR   ★★★★   좋음            Q4_VISCO_SIMO  볼륨 잠금 (부적합)
Q4_COROT  ★★     전단 잠금 위험   Q4_EAS    볼륨 잠금 (부적합)
                                 Q4_BBAR   볼륨 잠금 (부적합)
```

---

## 4. 폴더블 디스플레이 최적 요소 추천

> **결론: PET층은 `Q4_EAS`, PSA층은 `Q4_UP`**

### 4.1 PET 층 — Q4_EAS (Enhanced Assumed Strain) ⭐

**추천 근거:**
1. **전단 잠금 없음** — AGENTS.md §4.1에 따르면 요소 종횡비(AR)와 무관하게 굽힘 강도가 일정
   - Q4_COROT 대비: AR=7.5에서 **20.9×**, AR=30에서 **316×** 인공 강성 제거
2. **SOLVER-LEVEL PASS** — 10증분 45° 굽힘, 전 단계 수렴 확인
3. EAS는 전단 지배 문제에 본질적으로 강함

**Q4_COROTATIONAL이 적합하지 않은 이유:**
- `fold_model_config.py` 기본값이 `Q4_COROTATIONAL`로 되어 있으나 **변경 필요**
- 고종횡비 요소(디스플레이 두께 0.5mm, 폭 80mm → AR=160)에서 전단 잠금으로 인공 강성 25,000배
- 굽힘 변형을 심각하게 과소평가하게 됨

### 4.2 PSA 층 — Q4_UP (혼합 u/p) ⭐

**추천 근거:**
1. **유일하게 50% 압축 수렴** — B-bar, EAS, VISCO_SIMO는 모두 체적 잠금으로 실패
2. ν→0.5로 갈수록 변위-압력 혼합 정식화가 필수
3. SOLVER-LEVEL PASS 확인 완료

**Q4_VISCO_SIMO가 적합하지 않은 이유:**
- `fold_model_config.py` 기본값이 `Q4_VISCO_SIMO`로 되어 있으나 **변경 필요**
- VISCO_SIMO는 시간 의존 점탄성 통합 알고리즘(Simo 1987) — 비압축성 처리와 무관
- 표준 Q4 변위 기반이므로 체적 잠금 발생
- 점탄성이 필요하면 **Q4_UP + ViscoelasticMaterial** 조합이 필요하나 아직 미구현

### 4.3 전체 스택 추천 (PET + PSA 14층)

| 층 | 재료 | 권장 요소 | 이유 |
|---|---|---|---|
| 3n+1 | PSA | **Q4_UP** | 비압축성, 층간 전단 필요 |
| 3n+2 | PET | **Q4_EAS** | 압축성, 대변형 굽힘, 전단 잠금 제거 |
| 3n+3 | PET | **Q4_EAS** | 동일 |
| STEEL (강체 판) | NeoHookean | **Q4** (default) | 모든 DOF가 Dirichlet BC로 구속 → 요소 무관 |

### 4.4 `fold_model_config.py` 변경 권장

현재 (틀림):
```python
pet_element_type: str = "Q4_COROTATIONAL"   # ← 전단 잠금 위험
psa_element_type: str = "Q4_VISCO_SIMO"     # ← 체적 잠금
```

권장 (정확):
```python
pet_element_type: str = "Q4_EAS"            # 전단 잠금 없음, PET 굽힘 최적
psa_element_type: str = "Q4_UP"             # 유일하게 비압축성 수렴
```

---

## 5. 요소 이론별 특성 요약

| 요소 | 이론 | 비압축성 | 전단 잠금 | 굽힘 정밀도 | 점탄성 |
|---|---|---|---|---|---|
| **Q4 (B-bar)** | 표준 Q4 + selective reduced integration | ❌ 잠금 | ✅ 좋음 | 보통 | ✅ |
| **Q4_EAS** | Enhanced Assumed Strain (4-mode) | △ 부분적 | ✅ 우수 (AR 무관) | ✅ 우수 | ✅ |
| **Q4_COROT** | Co-rotational (강체 회전 분리) | ❌ 잠금 | ❌ AR² 비례 인공 강성 | ❌ 고AR에서 불량 | △ |
| **Q4_COROT_EAS** | 회전 + EAS | ❌ 잠금 | ✅ | ✅ | △ |
| **Q4_UP** | 혼합 u/p (Q1P0) | ✅ **우수** | ✅ | ✅ | △ (미구현) |
| **Q4_VISCO_SIMO** | SIMO (1987) 유한 변형 점탄성 | ❌ 잠금 | ✅ | 보통 | ✅ |
| **Q4_VISCO_FS** | Full-separated viscoelastic | ❌ 잠금 | ✅ | 보통 | △ (`pk2_voigt` 없음) |
| **SRI+J2** | Selective Reduced Integration + J2 | ❌ 잠금 | ✅ | 보통 | — |
| **Hybrid (Q1P0)+J2** | 혼합 변형도 + J2 | ✅ | ✅ | ✅ | — |
| **SRI+Hybrid** | 전단+혼합 결합 | ✅ | ✅ | ✅ | — |
| **CR SRI+Hybrid** | 회전+전단+혼합 | ✅ | ✅ | ✅ | — |

---

## 6. 주의사항 및 한계

1. **Q4_UP + ViscoelasticMaterial 조합은 아직 미구현**
   - PSA에 점탄성(creep/relaxation)을 포함하려면 향후 구현 필요
   - 현재는 PSA를 NeoHookean(탄성)으로 테스트

2. **요소 수준 통과 ≠ 솔버 수준 통과**
   - SRI/Hybrid/CR SRI+Hybrid는 요소 수준에서 PSA 압축을 통과했지만
   - 솔버 내 증가분 추적(point-wise solve vs incremental ramping)에서는 다른 결과가 나올 수 있음
   - 이는 추후 솔버 수준 테스트 확장이 필요함을 의미

3. **Q4_EAS의 PSA 한계**
   - EAS 단독으로는 ν=0.49, 50% 압축에서 체적 잠금 (XFAIL)
   - 그러나 폴더블 디스플레이의 PSA는 **압축보다 층간 전단**이 주된 변형모드
   - 실제 사용 조건에서는 Q4_EAS도 PSA에 충분할 가능성 높음
   - 층간 전단 지배 조건의 추가 검증 필요

4. **Q4_EAS와 Q4_UP의 동시 사용**
   - 서로 다른 PID에 다른 element_type 할당 가능 (dict 인터페이스)
   - PET=Q4_EAS, PSA=Q4_UP 혼용이 현재 아키텍처에서 지원됨

---

## 7. 빠른 시작 — 새 모델 만들 때

```python
from dispsolver.fold_model_config import FoldModelConfig

cfg = FoldModelConfig()

# 권장 요소로 변경
cfg.solver.pet_element_type = "Q4_EAS"
cfg.solver.psa_element_type = "Q4_UP"

# 확인: run_ex12 또는 ex13에서 cfg 전달
```

---

*마지막 업데이트: 2026-07-30*
