# 영역별 분석: Constraint (living doc, 초판 2026-09-19)

본 문서는 단일 소스 `GEMINI.md`의 영역 인덱스에서 참조된다. 수치 주장은 실측 기반만 기재.

## 1. SurfaceTie (penalty node-to-segment)
- `dispsolver/constraint/surface_tie.py`. display-plate 결합용.
- AGENTS.md §4.8: batch 조립 경로에서 tie/RBE2 꼬리가 통째로 스킵되던 silent 버그 수정됨.
  신규 조립 분기는 `return` 대신 fallthrough — 재발 금지.
- AGENTS.md §4.12: tie 강성 올리면 팁 특이점 악화. 경계면 역전 시 메시 grading이 정답.
- 현재 상태: ex12 기준 `max_gap ≈ 1e-5 mm` (타이트).

## 2. RBE2 (exact condensation vs penalty hinge)
- `rbe2_condensed.py`: 운동학적 응축, SPD 유지 — 플레이트 구동의 정답 경로.
- `RBE2HingeElement` (penalty+AL): 플레이트 구동에 이미 실패 확정 (§4.10). 신규 구동에 사용 금지.
- `skip_fully_prescribed` (§4.13): 구현済·기본 OFF. 켜려면 sanity_report 동반.

## 3. Contact constraints (3D)
- `SurfaceContactConstraint3D` / `DeformableSurfaceContactConstraint3D`:
  stick-slip 마찰 구현済. `GeneralContact3D`: mu/k_t/앵커/반환사상 구현済(untracked).
- 측정 (2026-09-19, 현 트리): 릴리즈 지정 9파일 스위트 **58/58 통과** (48.6s, 재측정 확인).
  단, 순서 의존성 결함 발견: `test_contact_friction.py` → `test_abaqus_contact_api.py`
  순서로 실행하면 후자에서 2건 실패 (`Model.ContactPair` 미노출처럼 보임 +
  슬립 미수렴). 역순이면 29/29 통과, 단독도 각자 통과. → 테스트 격리 결함
  (전역 상태 오염 추정), 제품 코드 결함 아님. 수정 대기.
- Dual mortar Stage 5 (Sutherland-Hodgman + Wohlmuth 쌍직교): 구현済, 릴리즈 로그 참조.
- 3D Hertz 롤러–평판 압입 & 마찰 슬라이딩 벤치마크 (2026-10-01):
  - `01_hertz_roller_plate_3d.inp` 완주: INDENT 22inc 0컷백, SLIDE 22inc 0컷백, 총 44inc, 59.54s.
  - 마찰 슬라이딩 로컬 마찰비 $\Sigma \|f_t\| / \Sigma p = 0.199995$ (처방값 $\mu=0.20$과 0.0025% 정합).
  - 법선 일관 접선(`consistent_tangent=True`) 도입: FD 대비 $10^{-10}$ 기계 정밀도 (`tests/test_deformable_contact_consistent_tangent.py` 7/7 통과).
- 3D 볼트 프리로드 & 파이프 플랜지 가스켓 밀봉 접촉 벤치마크 (Deck #06, 2026-10-03):
  - `06_bolt_preload_gasket_flange_3d.inp` 완주: Step 1 (볼트 프리로드) 및 Step 2 (파이프 내압 $5.0\,\text{MPa}$) 2개 스텝 완벽 완주.
  - 정적 작용-반작용 1:1 완벽 평형: Step 1 $RF_{\text{bolt}} = -2,750.22\,\text{N}$ vs $RF_{\text{gask}} = +2,747.34\,\text{N}$ (오차 < 0.10%), Step 2 $RF_{\text{bolt}} = -2,743.96\,\text{N}$ vs $RF_{\text{gask}} = +2,748.90\,\text{N}$ (오차 < 0.18%).
  - Wohlmuth Dual Basis 변환 복원: `finite_sliding_mortar3d.py`에서 DUAL 모타 타입 조건 판정 버그(`self.mortar_type in ("STANDARD", "DUAL")` -> `== "STANDARD"`)를 수정하여 진정한 Wohlmuth dual basis $A_f = \text{diag}(D_f) M_f^{-1}$ 수치 계산을 정상 복원함.

## 4. Open (미해결)
- [ ] 테스트 순서 의존성 격리 수정 (friction → abaqus_api 오염원 특정)
- [ ] `Model.ContactPair` 노출 여부는 오염 제거 후 재확인 (단독·정순 실행에서는 통과)
- [ ] `DeformableSurfaceContactConstraint3D`의 `tangential_stiffness_ratio` 기본값(1.0 vs 0.1) 의도성 확인

