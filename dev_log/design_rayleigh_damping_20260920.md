# 설계: Rayleigh 감쇠 (Wave2-H 잔여, Wave3 구현용)

## 1. htm 정답 (`simaver-c-damping.htm`)
- 검증: 봉 파동전파 에너지 이력 (`damp3d`, `dampax`, `dampbeam*`, `dampmembrane*`,
  `damppe`/`dampps` + `*hyper` 변형).帯域 제한 감쇠 (`*_band`)도 있으나 1차는 고전 Rayleigh.
- 관측량: 운동/변형 에너지 균등분할, 자유단 반사 상쇄, 감쇠손실 에너지 ALLVD.

## 2. 현행 (repo)
- `DynamicSolver3D`: 질량행렬 조립 존재 (l.1137), `elem_density` + 재료 density 경로 존재.
- `stabilization.py`: 점성 안정화 (`F=c·M·v`, 에너지분율 게이트) — 감쇠 아님, 혼동 금지.
- 감쇠행렬 조립 없음. 2D 동역학 감쇠 없음.

## 3. 정식화 (구현 시)
- `C = α·M + β·K` (Newmark/Implicit 동역학 스텝 내). `α, β ← (ξ1, ξ2, ω1, ω2)` 2-주파수 맞춤
  또는 단일 ξ·ω 대표값. 재료별 배분은 2차 (전역 단일 쌍 먼저).
- 위치: 질량조립 옆 (`dynamic3d.py` l.1137 인근) + 시간적분기 점성항. `α=β=0` 기본 → 기존 거동 bit-identical.
- 주의: 접선에 `β·K` 기여 시 Newton 접선에 포함 (속도-무관 정적해석에는 영향 0).

## 4. 게이트 (구현 후)
- SDOF 해석해: 대수감쇠율 `δ = 2πξ/√(1-ξ²)` (1e-6).
- 외팔보 자유진동 포락선 vs Rayleigh 예측 (1%).
- `α=β=0` 시 기존 `verification.run_all` bit-identical.
- 에너지 감사: 감쇠손실 = 초기-현재(운동+변형) 에너지 (htm ALLVD parity 개념).
