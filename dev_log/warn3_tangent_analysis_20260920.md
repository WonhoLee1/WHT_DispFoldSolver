# 3D WARN 3종 탄젠트 분석 및 ANP 수정 (2026-09-20)

기준: `benchmark_3d_elements.py` central-FD(h=1e-6), PASS < 1e-5.
출발점: `dev_log/benchmark_3d_elements_20260920.md` — C3D8_CR 1.03e-03, C3D8H 6.47e-04, C3D4_ANP 1.20e-03.

## 결과

| 요소 | 수정 전 | 수정 후 | 판정 |
|---|---|---|---|
| C3D4_ANP | 1.20e-03 | 2.19e-12 | PASS (커널 수정) |
| C3D8_CR | 1.03e-03 | 동일 | WARN 유지 (면제 권고) |
| C3D8H | 6.47e-04 | 동일 | WARN 유지 (면제 권고) |

## ANP 수정 (완료, 검증됨)

- 위치: `dispsolver/element3d/c3d4_anp_numba.py` `compute_c3d4_anp_element_matrices`
- 원인: `c_e = J_bar_e/detF` 가 u 의존인데 K가 B_bar 고정. FD는 u 섭동(J_bar 고정)이라
  `dB_bar/du` 항 2개 누락. O(strain) 상대오차 → 1e-3 변형에서 1.2e-3.
- 수정: rank-1 보정 2항 추가
  `K += V0·(-(c_e/3))·[B_vol⊗g·tr(s) + (B_bar^T·Crowsum)⊗g·q]`,
  `g = d(ln detF)/du`, `q = B_vol·u`. 상수 C에 대해 exact.
- 범위: 선형/custom 탄성 exact. 비선형 UMAT의 dC/du, F_bar 선형화는 범위 밖(주석에 명시).
- 증거: 단위셀 FD 2.19e-12, `tests/test_3d_c3d4_anp.py` 4 passed, 벤치 재측정 PASS.

## CR/H WARN 원인 (수정 없이 분류 완료)

- 공통: 두 커널 모두 co-rotational (`K_global = T(R)·K_local·T(R)^T`), `dR/du` 미선형화.
  변형 스케일 스윕 1e-4/1e-3/1e-2 → 오차 10배씩 선형 증가 (CR 1.65e-4/1.65e-3/1.67e-2,
  H 1.16e-4/1.16e-3/1.17e-2). 미분-누락 클래스 확정. 잔차는 정상 (단축인장 C3D8 대비 1.7e-12).
- 미커밋 `_add_geometric_stiffness_3d`는 초기응력항만 보충, `dR/du`는 여전히 없음.
- H 고유 체인(mean-dilatation)은 프레임항에 가려 분리 불가. 프레임 수정 후 재측정 필요.
- 권고: 2D §4.4 선례대로 modified-Newton 면제 문서화. 완전 수정은 3D 스핀-기하강성으로 대공사.
- 미커밋 diff 부수 관찰(수정 안 함): wrapper `Any` 미import(lint F821, agy diff 유래),
  443행 NumbaPerformanceWarning(agy diff 유래).

## §2.2 "bit-identical 1e-12" 이상 설명됨 — 디스패치 정상

- `dynamic3d.py:476-477`: nlgeom=True에서 plain `C3D8` → k=1(C3D8_CR 커널)로 승격이 설계.
  C3D8==C3D8_CR 일치는 당연. C3D8H(k=2)도 소변형에서 B-bar≈mean-dilatation,
  Newton 타이트 수렴 시 동일 해. 버그 아님. §2.2 해당 문장은 해소됨으로 갱신 요망.
