# 설계 I: 연성/JC 손상 개시·진전·요소삭제 (Wave3 구현용)

## 1. htm 정답 (SIMULIA 2024LE, `simaver-c-damage.htm`)
- 개시 패밀리: Ductile, Johnson-Cook (ductile 특수형), Hosford-Coulomb, Shear.
- 재료결합: Mises / Hill / Drucker-Prager 소성, EOS+JC (Explicit).
- 별도: FLD / FLSD / MSFLD / M-K (성형한계, 판재 전용 — 본 프로젝트 범위 외, Wave3 제외).
- 진전 + 요소삭제 (element deletion) 포함.
- 대표 덱: `damage_ductile_mises.inp`, `damage_jc_mises.inp`, `damage_ductile_hill.inp`.

## 2. 현행 (repo)
- `numba_materials.py:1155-1170`: LAW36 tabular `eps_fail` 한계 + `is_failed` 플래그(SDV[7]) + soften 0.01. sudden-death만, 개시누적/삼축의존/에너지진전 없음.
- `plastic_hardening.py`: JC 경화(`evaluate_johnson_cook_yield`)는 있음, JC 손상은 없음.
- Mullins (`mullins_damage.py`)는 별개 클래스(고무 연화) — 혼동 금지.

## 3. 정식화 (구현 시)
- 개시 (둘 다 무응력·무접선 변경 — 수렴위험 0, 출력/게이트 전용):
  - Ductile: `ω_D = ∫ dε̄_p / ε̄_D(η, ε̇_p)`, `ε̄_D` 삼축응력 η 테이블 보간 (`dependence_table` 1D/2D 재사용).
  - JC: `ε̄_D = [D1 + D2·exp(D3·η)]·[1 + D4·ln(ε̇*)]·[1 + D5·T*]`.
- 진전 (연화 — 점성정칙화 필요):
  - 변위 기반이 아닌 에너지 기반: `D = ∫ L·dε̄_p / ū_f`, `ū_f = G_f/σ_y0` (메시독립).
  - `σ = (1-D)·σ̃`, 접선 `(1-D)·C̃ - ∂D/∂ε⊗σ̃`.
  - `D→1` 시 기존 `is_failed` 패턴 재사용 (soften + 플래그), 삭제는 강성 잔류 + 후처리 제외로 (실제 행/열 삭제는 Wave3 범위 외).
- SDV: J2 7 + ω(1) + D(1) + flag(1) = 10. `get_default_sdv_count` + 신규 `MAT_*` 상수.

## 4. 게이트 (구현 후)
- `damage_ductile_mises` 단축: ω=1 시점이 해석해 `ε̄_D(η=1/3)`와 일치 (1e-6).
- JC D1-D5 중 D2=D3=D4=D5=0 퇴화 → 상수 한계변형률과 일치.
- 진전: G_f 일정 시 요소크기 2배 → 하중-변위 곡선 동일 (메시독립).
- C8/C10 계약 + `verification.run_all` 9/9(현행 기준) + pytest 베이스라인.
