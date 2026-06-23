# Walkthrough: 수치 강건화 · 멀티코어 · 하이브리드 점탄성 · 시간의존 하중 (2026-06-24)

상용 SW(Abaqus) 수준의 안정성·확장성을 목표로, 배치 조립 정확성 검증부터 시작해 퇴화요소 강건화, 멀티코어 정확도, Q1P0 점탄성 하이브리드 요소(+WLF TTS), 재료별 요소/두께 설정, 시간의존 하중(Amplitude)까지 구현한 내용을 정리합니다. 결과적으로 디스플레이 폴딩 예제([ex03_display_fold.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/examples/ex03_display_fold.py))가 **180° 완전 접힘(Load Factor 1.0)** 까지 완주합니다. 전체 **105개 단위 테스트 통과**.

---

## 1. 배치 조립 접선 일치 검증 및 수정 (수렴성 회복)

대규모 메쉬 가속을 위한 배치(vectorized) 조립이 순차 경로 대비 **불일치 접선(inconsistent tangent)** 을 만들어 Newton 수렴을 망가뜨리는 문제를 발견·수정했습니다.

* **수치 비교로 원인 규명**: 동일 입력에 대해 순차 vs 배치 응력/접선을 직접 비교.
  - J2 소성 배치: 이미 기계정밀도 일치(dC≈2e-10) — 문제 없음.
  - 점탄성 배치: 두 버그 — (a) 순차는 **numpy 해석적 NeoHookean**, 배치는 **JAX autodiff**를 써서 C[2,2] 전단 규약이 **정확히 2배** 차이, (b) NeoHookean numpy ↔ JAX 자체가 불일치.
* **수정**: [neohookean.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/material/neohookean.py)에 `pk2_tensor_batch`/`tangent_voigt_batch`(numpy 해석식 벡터화)를 추가하고, [viscoelastic.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/material/viscoelastic.py)의 점탄성 배치가 이를 사용하도록 전환(JAX 제거). 결과 **순차와 기계정밀도(dC≈4e-13) 일치**.
* **`fast_assembly` 토글**: 검증된 배치 가속(기본 True)과 순차 참조(False)를 선택. 배치/순차가 동일 Newton 반복을 생성함이 보장됨.

---

## 2. 퇴화 요소 유한 정규화 (NaN 크래시 근본 제거)

직전 스텝이 저장한 소성/기하 상태가 한 요소를 역전(det F→0)시키면, 그 base point 조립이 NaN을 뱉어 K_T 전체를 오염 → du NaN → 컷백 불가 → FATAL로 이어지는 구조적 결함을 해결.

* **NaN 주입 → 유한 정규화**: 퇴화/비유한 요소를 **영응력 + 안정 등방 접선**으로 대체. 전역 시스템이 유한하게 유지되어 Newton+라인서치+컷백이 요소를 역전에서 빼냅니다(상용코드 표준).
* **예외 없는 해석적 역행렬**: `np.linalg.inv`가 배치 중 한 행이라도 수치적 특이면 전체 예외를 던지므로, 블록대각 구조를 이용한 2×2 해석적 역행렬로 대체.
* 적용: [plastic.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/material/plastic.py)·[viscoelastic.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/material/viscoelastic.py) 배치 경로.
* **효과**: ex03 도달 Load Factor **0.0017(FATAL) → 0.154** (약 90배), 크래시 완전 제거.

> 참고: 잔차-노름 Armijo 라인서치도 시도했으나, **부정부호 KKT 시스템에서 잔차가 단조감소하지 않아** 2차 수렴이 깨짐(테스트 7개 실패)을 확인하고, 풀스텝 Newton + NaN-가드 라인서치로 원복.

---

## 3. 멀티코어 수렴성 해결 (반복 정련, Iterative Refinement)

Lagrange 승수 제약이 만드는 **부정부호 saddle-point KKT**를 멀티스레드 PARDISO가 스레드 의존 라운딩으로 덜 정확하게 풀어 수렴이 저하되는 현상.

* **해법**: 직접 솔브 후 잔차 `r = b − J·x`를 같은 인수분해로 재해 `x += solve(r)` (`_solve_linear_system`). 모든 코어를 유지하면서 기계정밀도 복원. Abaqus 직접솔버와 동일 기법.
* **검증**: N_CORES=4 + 반복 정련으로 ex03 **100% 완주** → 멀티코어가 수렴을 저하시키지 않음 실증.

---

## 4. 선형 점탄성 재료 + WLF TTS

이력(history)·소산(dissipation)은 단일 에너지 포텐셜로 표현 불가하므로 응력 기반으로 구현.

* **`LinearViscoelastic`** ([linear_viscoelastic.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/material/linear_viscoelastic.py)): 소변형 Prony 편차 완화 + 탄성 체적. 알고리즘 전단계수 `G_alg = G_∞ + Σ G_i γ_i`.
* **WLF 시간-온도 중첩(TTS)**: `τ_i* = τ_i · a_T(T)`로 온도 의존 완화. 순간(G0)↔완화(G_∞) 거동, 온도 시프트 검증.

---

## 5. Q1P0 점탄성 하이브리드 요소 (체적 락킹 완화)

준비압축 PSA 접착층의 힌지부 체적 락킹이 LF 0.154 수렴 벽의 원인이었음.

* **`q4_visco_hybrid`** ([q4_visco_hybrid.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element/q4_visco_hybrid.py)): 소변형 **mean-dilatation(B-bar) Q1P0** 요소. 체적은 평균팽창, 편차는 Prony 완화. 소변형이라 요소 역전 문제도 원천 없음.
* **효과**: 이 요소를 PSA층에 적용하자 **수렴 벽(LF 0.154) 돌파 → 180° 완전 접힘 완주**.

---

## 6. 재료(pid)별 요소 타입 · 섹션 두께

* **pid별 `element_type`**: `element_type={0:"Q4", 1:"Q4_UP"}` dict 지원 → PET(J2)=Q4 배치, PSA(점탄성)=Q1P0 하이브리드 혼용. 배치 조립기에 하이브리드 그룹 분기 추가.
* **`section_thickness`** (면외 깊이): 스칼라(균일) 또는 `{pid: t}` dict로 **적층마다 다른 깊이** 지정. 요소 내력·강성·질량을 선형 스케일(모든 조립 경로 + lumped mass 적용). 공유절점도 합산으로 정상.

---

## 7. 접촉 깊이 · RBE2

* **Contact**: `PenaltyContactConstraint(depth=...)` — 접촉력 = 압력×면적(∝깊이)이므로 penalty 강성을 깊이로 스케일. ([contact_jax.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/constraint/contact_jax.py))
* **RBE2**: Lagrange 강체 운동학 제약이라 두께 스케일 대상 없음 — 반력이 자동 균형(이미 일관).

---

## 8. 시간 의존 하중 (Abaqus *AMPLITUDE)

* **`Amplitude`** ([amplitude.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/load/amplitude.py)): 시간-값 테이블 + 보간(`linear`=RAMP, `smooth`=SMOOTH STEP, `step`). 범위 밖 끝점 hold.
* **solver 연동**: `set_prescribed_dofs(dofs, base, amplitudes=...)` — DOF별 amplitude(None=상수). 유효값 = `base × amplitude(time)`. solver가 `self.time`을 추적하여 **수렴 성공 시에만 전진**, 컷백/롤백 시 `save/restore_state`로 복원.
* **ex03 적용**: 힌지 회전을 단일 Amplitude(linear ramp)로 한 번만 설정 → 루프가 BC를 매 스텝 재계산하지 않음. 하중 경로·완주 결과 동일.

---

## 9. ex03 디스플레이 폴딩 최종 결과

* 7층 적층(PET 4 + PSA 3), 힌지 RBE2, 자기접촉, 적응 시간증분 + 컷백.
* **PET(J2 소성)** Q4 B-bar 배치 + **PSA(선형 점탄성)** Q1P0 하이브리드 혼용.
* 재료: PET `E=4000, σy=80, H=620`(100% 소성변형 시 700MPa), PSA `E=10, ν=0.49` 선형 점탄성.
* **결과: Load Factor 1.0000 (180° 완전 접힘) 전체 완주**, 멀티코어(4코어) 정련 적용.

---

## 10. 테스트 / 커밋

* **전체 105개 단위 테스트 통과** (신규: 점탄성 하이브리드 6, 섹션두께 2, 접촉깊이 1, Amplitude 6 등).
* 커밋: `4a27bd5`(강건화·멀티코어·하이브리드) → `28efcff`(섹션두께·접촉깊이·Amplitude) → `c3ad938`(ex03 Amplitude 적용). origin/master 푸시 완료.

---

## 11. 남은 개선 과제 (ex03 기준)

1. **두께방향 메쉬 해상도**: 층당 요소 1개 + 종횡비 10:1 → 굽힘 곡률 과소 해상. 층당 2~4 요소 필요.
2. **결과 가시화**: 현재 최종 변위만 export. **스텝별 프레임 + 응력/eqps/점탄성 필드** 출력 필요(진행 예정).
3. **공학 지표 추출**: 최대 응력, 최대 소성변형, 힌지 반력모멘트, 최소 굽힘반경/접촉 gap.
4. **섹션 깊이 적용**: `section_thickness` 기능을 ex03에 실제 반영(현재 기본 1.0).
5. **준정적 vs 동적**: `rho=1000` 관성항이 준정적 폴딩에 인위적 — 정적 모드 또는 rho 축소 검토.
6. **점탄성 하이브리드 배치화**: 현재 순차 루프(스텝당 수초) — 가속 여지.
7. **WLF/온도 시연**: 기능은 있으나 ex03은 등온.
