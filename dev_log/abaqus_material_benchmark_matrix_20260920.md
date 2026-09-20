# Abaqus Material 검증 전체 매트릭스 (2026-09-20 통합)

소스: `dev_log/material_verification_requirements_20260919.md`(요구 23종),
`dev_log/abaqus26_material_models_analysis_20260919.md`(설계),
`dev_log/material_implementation_walkthrough_20260919.md`(70/70),
`dev_log/area_analysis_material_20260919.md`(81/81 living).
상태는 코드·테스트 대조로 재확정. 근거 없는 PASS 없음.

## 전체 리스트 (23 in-scope + 3 제외)

| # | Abaqus 검증 파일 | 상태 | 통과 근거 / 갭 |
|:---|:---|:---:|:---|
| 1 | elastic | △ | NH·AB·Yeoh+선형만 (`test_hyperelastic` 16). Ogden·Mooney·poly·이방·다공 없음 |
| 2 | viscoelastic | △ | Prony+WLF 등방 (`test_viscoelastic` 28). 주파수영역·hyperfoam·이방 없음 |
| 3 | mullinseffect | ✅ | `mullins_damage.py` + 테스트 5 (Ogden-Roxburgh η, 재재하 포락선 복귀) |
| 4 | hysteretic | ✅ | `bergstrom_boyce.py` + 테스트 5 (B망 점근완화 → A 수렴) |
| 5 | tempelastic | ✅ | `temp_elastic_ef1` 오차 0.010% PASS (`benchmark_abaqus_material_inp.py` 1/4) |
| 6 | fieldelastic | △ | 2D 테이블 인프라만. FIELD plumbing 미완 (마스터플랜: T-first, FIELD 훅) |
| 7 | viscohyper | △ | Standard형만. Explicit 검증 케이스 없음 |
| 8 | nonlinviscohyper | ✅(core) | BB 2-네트워크 완료. Explicit·망변형 미검증 |
| 9 | rateindepplast | ✅(core) | Mises 등방+경화3종+JC+CS+테이블 (`test_plastic_hardening` 5, radioss/tabular 8). Hill·D-P·Cap·Foam·다공 없음 |
| 10 | viscocylinder | △ | Prony+시간적분. 자동증분은 solver 영역 |
| 11 | ratedepplastexp | ✅ | JC+CS, `johnson_cook_dynamic` PASS (4/4) |
| 12 | annealtemp | ✗ | 코드 0건 (grep 확인). T-인프라 위 소규모로 가능 — S1 계획 대비 결번 |
| 13 | tempplastic | ✅ | `temp_plastic_mises` PASS (고온 유동응력 저하 assert) |
| 14 | fieldplastic | △ | FIELD 훅 수준. T-인프라+소성 결합 미완 |
| 15 | johnsoncook | ✅ | `johnson_cook_dynamic` PASS + 기준상태 축퇴 1e-12 + 완전용융 σ=0 |
| 16 | damage | ✗ | P4 별도 대공사 |
| 17 | creep | ✅ | 거듭제곱 크리프 음함수 리턴매핑+일관탄젠트 (`test_creep_power_law` 4) |
| 18 | viscous | △ | `linear_viscoelastic.py` 존재하나 `test_linear_visco_hybrid` 2 실패 = 베이스라인 기실패(135/10 중) |
| 19 | concrete | ✗ | 최후순위 |
| 20 | concretedamaged | ✗ | 최후순위 |
| 21 | damping | ✗ | 동적 solver 영역과 묶음 |
| 22 | simpleexpansion | ✅ | `thermal_expansion_simple` 구속반력 210.0000N 오차 0.0000% + 자유팽창 3.000μm 정밀일치 |
| 23 | plasticity-corrections | ✗ | Neuber/Glinka 후처리, 솔버 불필요 — 언제든 소규모 |
| 24–26 | 전도율·비열·간극전도 | — | 범위 제외 (열요소·과도해석 영역) |

## 테스트·벤치 증거 묶음

- 단위/결합: 81 passed (`test_dependence_table` 4, `test_thermal_mechanical_coupling` 5,
  열팽창 2D/3D 각 2, `test_plastic_hardening` 5, Mullins/BB 5, Law36 4, tabular 4,
  hyperelastic 16, viscoelastic 28, creep 4, bimetal 2). 당일 70→81 성장, 81이 최신.
- Abaqus parity: `benchmark_abaqus_material_inp.py` 4/4 PASS (위 #5·11·13·22).
- 구조 연계: 바이메탈 Timoshenko 2D 0.110%·3D 0.208%, 14층 디스플레이 열롤업 0.11% PASS.

## 항목별 벤치/테스트 코드 매핑

| # | 테스트 코드 (`tests/`) | 벤치 코드 (`benchmark_element/`) | 구현 모듈 (`dispsolver/material/`) |
|:---|:---|:---|:---|
| 1 | `test_hyperelastic.py` | — | `neohookean.py` `yeoh.py` `arruda_boyce.py` |
| 2 | `test_viscoelastic.py`, `bench_visco_jax.py` | — | `viscoelastic.py` `linear_viscoelastic.py` |
| 3 | `test_mullins_and_bergstrom_boyce.py` | — | `mullins_damage.py` |
| 4 | `test_mullins_and_bergstrom_boyce.py` | — | `bergstrom_boyce.py` |
| 5 | `test_dependence_table.py` | `benchmark_abaqus_material_inp.py::run_benchmark_temp_elastic_ef1` | `dependence_table.py` |
| 6 | — (인프라만) | — | `dependence_table.py` (2D 테이블) |
| 7 | `test_viscoelastic.py` (Standard 범위) | — | `viscoelastic.py` |
| 8 | `test_mullins_and_bergstrom_boyce.py` (BB망) | — | `bergstrom_boyce.py` |
| 9 | `test_plastic_hardening.py`, `test_radioss_law36_smoothing_and_tail.py`, `test_tabular_plasticity_radioss.py` | — | `plastic_hardening.py` |
| 10 | `test_viscoelastic.py` (Prony 시간적분) | — | `viscoelastic.py` |
| 11 | `test_plastic_hardening.py` (JC/CS) | `benchmark_abaqus_material_inp.py::run_benchmark_johnson_cook_dynamic` | `plastic_hardening.py` |
| 12 | 없음 | 없음 | 없음 |
| 13 | — | `benchmark_abaqus_material_inp.py::run_benchmark_temp_plastic_mises` | `dependence_table.py` + `plastic_hardening.py` |
| 14 | 없음 (훅) | 없음 | — |
| 15 | `test_plastic_hardening.py` | `benchmark_abaqus_material_inp.py::run_benchmark_johnson_cook_dynamic` | `plastic_hardening.py` |
| 16 | 없음 | 없음 | 없음 |
| 17 | `test_creep_power_law.py` | — | `creep_numba.py` |
| 18 | `test_linear_visco_hybrid.py` (2 실패) | — | `linear_viscoelastic.py` |
| 19 | 없음 | 없음 | 없음 |
| 20 | 없음 | 없음 | 없음 |
| 21 | 없음 | 없음 | 없음 |
| 22 | `test_thermal_mechanical_coupling.py`, `test_solver_thermal_expansion_2d.py`, `test_solver_thermal_expansion_3d.py`, `test_bimetallic_strip.py` | `benchmark_abaqus_material_inp.py::run_benchmark_thermal_expansion_simple`, `benchmark_bimetallic_strip.py`, `benchmark_display_thermal_rollup.py` | `thermal_expansion.py` `solver_thermal_adapter.py` |
| 23 | 없음 | 없음 | 없음 |

## 정직한 주의 4건

1. "100% PARITY"는 4케이스 한정. 23종 전체 parity 아님.
2. `test_linear_visco_hybrid` 2 실패는 기실패 베이스라인. viscous △ 사유.
3. annealtemp는 S1 계획에 있고 코드 없음. 다음 소규모 작업 1순위.
4. FIELD 2종(#6·#14)은 훅만. T-인프라 검증 후 FIELD 매트릭스 2배 작업 필요.

## "100여종"/"180여종" 보고와의 정합 (2026-09-20 실측)

- `SIMACAEVERRefMap\simaver-c-*.htm` 전체 = **413개** (검증 매뉴얼 전 챕터, 재료 외 포함).
- 재료 파일 27종의 `Results and discussion` 섹션 합 = **106개**. 제외 3종 6개 빼면
  **in-scope 서브케이스 = 정확히 100개** ("100여종" 보고 = 이것).
- **"180여종" 보고 = 100 + 재료-인접 81 = 181.** 인접 내역 실측:
  - 사용자 서브루틴 37 (U계 21: uhard·uhyper_stretch·umatht·umatuhyper5·uinter·ucreepnetwork·utrsnetwork·uexpan·uanisohyper,
    V계 16: vuhard·vumat·vumatht·vucreepnetwork·vusdfld·vugens·vuel·vuinter)
  - import-* 12 (hyper·plast·visco·uniaxial·hfoam·matchange 각 2)
  - 열-구조 인접 21 (tempfield 8·temppulse·tempdepfilm·constrthermalexp·cylshellthermalstress·rigidbodytemp 5)
  - 인장/파괴 11 (tensile·cracking·cracking-ts 각 2, crackdebonding 5)
- 요구서 27종에 없는 재료 본체 파일도 실존: `ratedepplaststd`(6), `damagefrm`(10),
  `eosmaterial`(7), `damageaniso`·`gurson`·`drucker`·`captests`·`barlatplasticity`(각 2).
  전부 합산 시 220개. 다음 요구서 개정 시 편입 여부 결정 필요.

## 다음 순서 (요구서 P-순서 유지)

P0 잔여: annealtemp → FIELD plumbing (#6·#14) → #22 범위확장(Ogden/AB/VdW 팽창).
P1 잔여: Hill·D-P 등 비금속 소성 (수요 시). P2 잔여: Ogden/Mooney 초탄성, Explicit 케이스.
P3: Neuber/Glinka 후처리 (독립). P4: damage·concrete (별도 제안서).
