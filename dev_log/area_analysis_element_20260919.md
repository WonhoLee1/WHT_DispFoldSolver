# 영역별 분석: Element (living doc, 초판 2026-09-19)

본 문서는 단일 소스 `GEMINI.md`의 영역 인덱스에서 참조된다. 수치 주장은 실측 기반만 기재.

## 1. 2D 12원소 (`dispsolver/element2d/`, solver2d)
- Rank 100% (RBM 3, spurious 0), 탄젠트 1e-14, Irons 패치 1e-19~1e-20.
- `CPE4IH` 및 `CPE4IH_CR` (4-모드 Simo-Rifai EAS + 3-DOF 선형 혼합 압력 $(p_0, p_\xi, p_\eta)$, 2026-10-19 완료): 2단계 Schur 보완 정적 응축으로 $\nu=0.4999$에서도 $O(K)-O(K)$ 상쇄 없이 전단 잠김과 체적 잠김을 동시 소거(단일 층 세장 보 순수 굽힘 오차 $1.36 \times 10^{-9}$, 왜곡 Irons 패치 $< 10^{-12}$, 회전 기준틀 $0^\circ\sim 90^\circ$ 불변성 $< 10^{-10}$). JAX `CPE4IH` 프로토타입의 안장점 정규화 결함 수정으로 `tests/element_contract/test_contract.py` C1–C9 전 항목 Green 달성. 또한 `cpe4i_numba.py`의 `_voigt_transform_2d` 내 $\mathbf{J}_0^{-1}$ vs $\mathbf{J}_0^{-\mathsf{T}}$ 전치 결함을 교정 완료.
- `COH2D4` 4절점 코히시브 계면 요소 (2026-10-18 P6 완료): 강체 모드 불변성 1.45e-11, 중간면 참조 좌표계 기반 선형 B-행렬로 기하 강성 왜곡 원천 차단, 방향성 유한차분 탄젠트 오차 < 1.2e-5, 순수 수직/전단 파괴 에너지 보존 오차 < 1.2e-5, 압축 침투 저항 및 DynamicSolver2D 다중 스텝 박리 100% 수렴 실증 (tests/test_coh2d4.py 11/11 Green).
- 생산 조합: `2D-Opt1` (`CPE4I + CPE4H`), `2D-Opt1-IH` (`CPE4I + CPE4IH`).
- `tests/element_contract/` C1–C11이 표준 게이트. 회전-참조 스위프 없는 요소 검증 인용 금지 (§4.14).

## 2. 3D 12원소 (`dispsolver/element3d/`, solver3d)
- 12원소 전부 in-house + Abaqus 공식 패치双重 통과 (2026-09-13, 독립 재검증).
- `C3D8IH` 및 `C3D8IH_CR` (9-모드 Simo-Armero EAS + 4-DOF 선형 혼합 압력 $(p_0, p_\xi, p_\eta, p_\zeta)$, 2026-10-19 완료): 6개 강체 모드 영공간 $< 10^{-12}$ 및 17개 편차 변형 모드 $O(G)$ + 1개 체적 팽창 모드 $O(K)$ 분리 확인, 3D 왜곡 8요소 Irons 패치($\nu=0.4999$) $< 10^{-12}$, 비압축 세장 보 굽힘 이론해 일치 오차 $< 10^{-8}$ (`tests/test_cpe4ih_c3d8ih.py` 10/10 Green).
- 잔류 한계 3건 (실사용 범위 밖, 신규 증거 없으면 추적 중단):
  `C3D8_FBAR` nu~0.4999 불안정대, `C3D8H` 굽힘 0.55 vs 0.60 bar,
  `C3D4_ANP` 평균화 무효과(해당 형상), `C3D8_CR`≡`C3D8H` 비트동일 미해명.

## 3. 판정 기록 (세션 실측)
- **CPE6M 롤업** (dev_log §16): "180° 도달"은 오수락. 요소레벨 무죄 4건
  (강체회전 1e-11, Tri1/Tri2 프레임 0.0007° 일치, 중첩회전 불변, 선형강성 −9%).
  Newton 경로 유죄 (~140° 이탈). 해당 행 NON-CONVERGED.
- **MITC4+ 전단 스케일** (2026-09-19, 수정·검증 완료): 전단 B 회전항 텐서/공학 관례 혼합 →
  조립 굽힘 정확히 4배 뻣뻣. 0.5 제거 4줄로 Scordelis-Lo 16x16 오차 **0.28%** (재측정),
  평판 수렴 0.9997, 6RBM 유지. 수정 파일 untracked — 커밋 대기.

## 4. Open (미해결 — 인용 금지)
- [ ] probe2의 CPE4I_CR 6.8× 비등방: Opt1 실전 −0.4%와 모순 → 프로브 아티팩트 의심. 규명 전 인용 금지.
- [ ] J2 NumPy 탄젠트 86% 오차 (수렴속도만 영향, 기각치 아님 — 미수정).
- [ ] MITC3+ Lee-Bathe edge tying 교체 (보류 중).
