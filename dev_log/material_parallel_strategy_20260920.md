# 재료 잔여 개발 병렬 전략 (2026-09-20)

## 원칙

1. 공유 파일 수정 금지. 모든 산출물은 NEW 파일로만 (충돌 원천 차단).
2. 에이전트 1개 = 목표 1개 + 산출물 1세트. 검증(pytest)까지 포함.
3. 큰 3덩어리(damage·concrete / 주파수 점탄성 / 이방 초탄성)는 설계문서 먼저, 구현 나중.
4. 메인 스레드는 산출물 직접 열어 난수/더미 검사 + 독립 재실행 후에만 완료 인정.

## Wave 0 (진행 중)

| 담당 | 목표 | 산출물 |
|:---|:---|:---|
| A | annealtemp | `material/anneal.py` + `test_annealtemp.py` |
| B | FIELD ef1 쌍 | `benchmark_field_ef1.py` + `test_field_ef1.py` |
| C | Ogden/Mooney | `material/ogden.py` + `material/mooney_rivlin.py` + 테스트 |

## Wave 1 (신규 투입)

| 담당 | 목표 | 산출물 | 의존성 |
|:---|:---|:---|:---|
| D | Neuber/Glinka 후처리 | `material/neuber_glinka.py` + `test_plasticity_corrections.py` | 없음 (독립) |
| E | Ogden/AB/VdW 열팽창 확장 | `material/expansion_hyper.py` + `test_expansion_hyper.py` | C의 Ogden식 읽기만 (복사 금지, import도 금지 — 독립 구현 후 C완료시 대조) |
| F | Hill48 소성 | `material/plasticity_hill.py` + `test_plasticity_hill.py` | `plastic_hardening.py` 읽기만 |

## Wave 2 (Wave 0/1 완료 후)

- G: 크리프 쌍곡사인+Anand (`material/creep_extended.py` ✅ 2026-09-20 직접구현 10/10).
- Wave3: damage ✅ (`damage_initiation.py` + MAT13 + API/packing + `test_damage_ductile_jc.py` 10/10),
  주파수점탄성 ✅, 이방초탄성 ✅, Rayleigh ✅ — Wave0~3 구현분 전부 닫힘.
- H: viscous 2층 기실패 해소 + damping 설계 → 기실패 재현 안 됨 2026-09-20 (visco/damping 전수 71 passed, 0 failed). Rayleigh HHT 구현 ✅ (`set_rayleigh_damping`, `test_rayleigh_damping.py` 4/4).
- I~K: 3대 설계문서 (damage·주파수점탄성·이방초탄성) — 구현 아님, htm 정답→정식화→게이트
  (✅ 2026-09-20 직접작성 4건: design_damage_ductile_jc, design_frequency_viscoelasticity,
  design_anisotropic_hyperelastic + design_rayleigh_damping).

## Wave 3

- 설계문서 기반 대공사 3건 구현. 별도 승인 후 착수.

## DoD (에이전트 공통)

- 새 테스트 전부 PASS + 기존 관련 스위트 회귀 PASS.
- inp 또는 해석해 대조 수치 보고 (허위 수치 금지).
- 커밋 금지. 산출물은 지정 경로만.
