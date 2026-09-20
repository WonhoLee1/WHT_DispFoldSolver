# 설계 J: 주파수영역 점탄성 (Wave3 구현용 — 신규 물성상수 없음)

## 1. htm 정답 (`simaver-c-viscoelastic.htm`)
- 시간↔주파수 데이터 변환 등가성이 검증 대상 자체: `mvhcdo2sr2` (주파수 데이터에서 교정한 Prony로 시간해석),
  `mvhcdo2ssd` (Prony에서 유도한 주파수 데이터로 정상상태 동해석). 양방향 일치.
- 주파수 직접지정 (`mvhcdo2ss2`, `mvhcdo2zzz` 테이블)도 있으나 1차 범위는 Prony→주파수 평가.
- 공식계(formula), 보간, 이방성 테이블 변형 (`frq_visco_formula_ortho_*`, `frq_visco_tabular_aniso_*`)은 2차.

## 2. 현행 (repo)
- 시간영역 Prony M≤6 (`viscoelastic.py`, `numba_materials.py` MAT_HYPER_VISCOELASTIC, SDV 42).
- 주파수 평가 함수 없음. DMA sweep 없음.

## 3. 정식화 (구현 시)
- Prony→복소탄성률 (정확 변환, 신규 파라미터 0):
  `G'(ω) = G0·(1-Σg_i) + G0·Σ g_i·(ωτ_i)²/(1+(ωτ_i)²)`,
  `G''(ω) = G0·Σ g_i·ωτ_i/(1+(ωτ_i)²)`, `tanδ = G''/G'`.
  체적부 동일형 (k_i). WLF 이동인자 적용 시 환산주파수 `ω·aT`.
- 위치: `material/frequency_viscoelastic.py` 신규 (순수함수, Numba 불필요 — 후처리/설계용).
- 용도: PSA 적층 감쇠 설계 (주파수별 손실계수), 시간↔주파수 교차검증.

## 4. 게이트 (구현 후)
- ω→0: `G'→G0(1-Σg)` (장기탄성률), ω→∞: `G'→G0` (순간탄성률) — 해석해 일치 1e-12.
- 단항 Prony 해석해 대조 (Debye 피크: `ωτ=1`에서 `G''` 최대).
- 시간영역 완화곡선 수치 Laplace 역변환 대조 (1%).
