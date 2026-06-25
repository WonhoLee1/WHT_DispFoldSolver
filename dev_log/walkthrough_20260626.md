# Walkthrough: ex03 디스플레이 폴딩 수렴성 개선 — 요소 기술·일관 접선·대변형 점탄성 (2026-06-26)

`ex03_display_fold`의 90° 폴딩 해석이 증분당 회전 ~0.027°에서만 수렴하던 문제를, **근본 원인(요소 락킹 + 비일관 접선)** 을 차례로 제거해 해결한 기록. 부수적으로 JAX 구현의 치명적 NaN 결함 2건을 발견·수정하고, 시간적분 모드 프리셋과 **base 플러그인 대변형 점탄성 요소**를 추가함.

---

## 0. 문제 진단 — "수치 튜닝이 아니라 모델이 원인"

세 가지 처방(힌지존 x‑세분 / `max_step` / dt 축소)이 **모두 동일한 벽**(증분당 회전 ~0.027°, load factor 0.0003)에 부딪힘. 메쉬·스텝제어·보정제한에 전부 불변 → 병목은 수치 튜닝이 아니라 **모델 본질**임을 확정.

원인 두 가지:
1. **PET(J2 소성)의 굽힘 전단 락킹** — 얇은 요소(높이 0.0167mm, 종횡비 30:1)가 바닥행 강체구동에 뒤집힘(element inversion).
2. **재료 해석 접선의 비일관성** — J2 재료가 돌려주는 `C_v`가 자기 응력의 실제 `dS/dE`와 불일치(전단항 C[2,2]=2692 vs 실제 674) → Newton 준선형 수렴.

---

## 1. EAS 요소 (PET 굽힘 락킹 제거)

**Simo & Rifai (1990)** Enhanced Assumed Strain — 향상모드 `M(ξ)·α`를 추가하고 요소내 정적응축. `dispsolver/element/q4_eas.py`(numpy) → `q4_eas_jax.py`(JAX).

- 검증: patch test 정확(α=0), 일관 접선 FD 대조 **2e‑6**.
- **핵심 우회**: 재료 해석접선이 비일관이므로, 접선을 **재료 응력의 유한차분(FD)** 으로 구성(`dF = F⁻ᵀ dE`) → 비일관 접선 버그와 무관하게 2차 수렴 보장.
- 효과: 증분 한계 16배↑ (0.0003 → 0.005, 상승 지속).

---

## 2. JAX 구현 점검 — 치명적 NaN 결함 2건 발견·수정

JAX 포팅(`plastic_jax.py`, `q4_eas_jax.py`)은 autodiff 일관접선 + vmap으로 방향은 옳았으나, **단위 테스트가 분리된 변형(uniaxial/shear)만 검증해 실전 케이스를 놓침**.

### 2.1. `jnp.linalg.eigh` autodiff NaN (미변형 F=I)
고유벡터 미분의 `1/(λᵢ−λⱼ)` 항이 **고유값 중복 시 NaN**. ex03 초기엔 전 요소 F=I(세 고유값 모두 1) → 첫 조립에서 전 PET 접선 NaN → 즉사.
- **수정**: `tangent_voigt_jax`의 autodiff를 **응력의 중심 FD**로 교체. 응력은 F=I에서도 유한하므로 NaN 회피.

### 2.2. `inv(F)` 인버전 가드 누락
`tangent_voigt_jax`의 `FinvT = inv(F)`가 라인서치 중 뒤집힌 시험 F에서 폭발 → NaN → LS.alpha 0.12로 감쇠·정체.
- **수정**: `det(F) <= 1e-8`이면 등방 정규화 접선 반환(응력 0). LS.alpha 1.0 회복.

두 수정 후: 증분 한계 0.0003 → **0.005+ 상승**, LS.alpha=1.0, 폴드 진행.

---

## 3. 대변형 PSA 점탄성 — 소변형 요소 교체

PSA(`Q4_UP`)가 **소변형 요소**(ε=Bu, 기하강성 없음)인데 90° 대회전에 사용됨 → 회전을 변형으로 오인(가짜 응력) + 비일관 접선 → 선형 수렴 잔존.

`dispsolver/element/q4_visco_hybrid_fs_jax.py` 신규: **Green‑Lagrange 변형 + F‑bar + autodiff 접선**.
- 점탄성은 고유분해가 없어 autodiff가 **NaN 안전** → 일관접선 공짜.
- 검증: 강체회전 30° → |f_int|=1e‑14, 접선 FD 대조 1.4e‑9.

---

## 4. 시간적분 모드 프리셋 (HHT‑α)

관성은 이 폴드에서 **가짜**(진짜 시간의존성은 점탄성 tau=1s≈T_total). 질량항 `(1/βdt²)M`은 LM 댐핑과 동등한 정규화이나 가짜 고주파 전이를 유발. **HHT‑α(Hilber–Hughes–Taylor 1977)** 로 고주파만 감쇠.

`DynamicSolver`에 `mode` 프리셋 추가 (β,γ를 α에 일관 연동: β=(1−α)²/4, γ=½−α):

| 모드 | α | β | γ | 성격 |
|---|---|---|---|---|
| `transient` | 0 | 0.25 | 0.50 | 순수 동역학 |
| `moderate-1` | −0.05 | 0.276 | 0.55 | 가벼운 감쇠 |
| `moderate-2` | −0.15 | 0.331 | 0.65 | 강한 감쇠, static‑like |
| `quasistatic` | — | — | — | 관성 제거(`static_mode`) |

`DynamicSolver(..., mode="moderate-2")` 한 줄로 일관 설정. ex03_optimized에 적용.

---

## 5. 완전한 대변형 점탄성 — base 플러그인 + Flory 분할

`q4_visco_hybrid_fs_jax`는 "선형 Prony를 Green‑Lagrange에 얹은" 근사. 완전한 **Simo (1987)** 대변형 점탄성으로 격상: `dispsolver/element/q4_visco_simo_fs_jax.py`.

### 5.1. 발견 — 압력 분할의 F=I 가짜 기준선
기존 `ViscoelasticMaterial`의 `S_vol = J·p·C⁻¹` 압력 분할은 F=I에서 `S_dev = μ·I ≠ 0` → 점탄성이 이를 완화하며 `S_eff ≈ −μΣgᵢ(1−γᵢ)I` 가짜 응력(초기 검증에서 |f|=5.5e‑2).

### 5.2. 해법 — Flory 등체적/체적 분할 (base 무관)
**Flory (1961)** F̄=J^(−1/3)F. 등체적 PK2 (Holzapfel 2000):
```
S_iso = 2 W1 J^(−2/3) ( I − (1/3) I1 C⁻¹ ),   W1 = dW_iso/dĪ1
```
→ F=I에서 `I − (I1/3)C⁻¹ = 0` 이므로 **S_iso = 0**. 가짜 기준선 소멸.

### 5.3. base 플러그인 — Yeoh/Arruda‑Boyce 동일 프레임
점탄성·Prony 점화식은 base 무관, **W1만 교체**:
- neo‑Hookean: `W1 = μ/2`
- **Yeoh (1993)**: `W1 = c1 + 2c2 x + 3c3 x²`, x=Ī1−3
- **Arruda‑Boyce (1993)**: `W1 = μ Σ i cᵢ/λm^(2i−2) Ī1^(i−1)`

체적: `S_vol = κ lnJ C⁻¹`. 과응력: `h_i = β_i h_i + g_i γ_i ΔS_iso`, `S_eff = S_vol + g_∞ S_iso + Σh_i`. 접선은 동결 이력에서 autodiff(γ_i 정합 — 기존 재료의 β_i 근사 접선 결함도 자동 해소).

### 5.4. 검증 (세 base 동일)
| base | F=I \|f\| | 강체회전 \|f\| | 접선 FD err |
|---|---|---|---|
| neo‑Hookean | 0.00 | 2e‑14 | 3.15e‑9 |
| Yeoh | 0.00 | 2e‑14 | 3.15e‑9 |
| Arruda‑Boyce | 0.00 | 2e‑14 | 3.15e‑9 |

---

## 6. 기술 출처

- Simo & Rifai (1990) — EAS 요소. IJNME 29(8).
- Simo (1992) — 유한변형 J2 multiplicative 소성. CMAME 99.
- Simo (1987), Simo & Hughes (1998 §10) — 대변형 점탄성 과응력.
- Hilber, Hughes & Taylor (1977) — HHT‑α 시간적분.
- Flory (1961) — 등체적/체적 분할. Holzapfel (2000) — 등체적 PK2.
- Yeoh (1993), Arruda & Boyce (1993) — 초탄성 base.
- de Souza Neto et al. (1996) — F‑bar 락킹 제어.
- Belytschko, Liu & Moran (2000) — TL B_L / 기하강성.

---

## 7. 남은 작업

1. `q4_visco_simo_fs_jax` 솔버 라우팅 통합 (PSA → `ViscoelasticMaterial` + base 선택).
2. 네 시간적분 모드 비교 실험 (수렴 반복수·완주 여부 정량화).
3. 후반 self‑contact 단계: 페널티 평활화/증강 라그랑지안 검토.
4. RBE2 안장점 → 변환법(master‑slave 응축) 검토.
