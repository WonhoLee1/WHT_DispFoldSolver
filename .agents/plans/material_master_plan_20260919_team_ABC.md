# 재료 구현 마스터플랜 (lucy-orcha A+B+C 융합, 2026-09-19)

소스: `dev_log/material_verification_requirements_20260919.md` + R1 9패널 brief.
R1 14중 9완료·5타임아웃 (A1·A4·B2·B3·B4, 14병렬 과부하 stall — 동일 프롬프트 재발사 금지).
부족 각도는 R4에서 실측 근거로 직접 메움 (§7).

## R2 교차비판 해소안 (9 brief 대조)

1. 테이블 모듈명: B1 `dependence_table.py` vs C2 `field_table.py`
   → `dependence_table.py` 채택 (Abaqus 용어 DEPENDENCIES). 순수함수 `eval_table`.
2. 탄젠트: A2·B1 frozen-field 일치 (등온 ∂S/∂E, dS/dT 제외, predefined-field 관례) → 채택·문서화.
3. 범위: C1 축소안 (T-first, FIELD 훅만) vs A3·B1 전체 → 단계화 채택.
   T 검증 후 FIELD (검증행렬 2배 방지).
4. 선행게이트 (C5): J2 탄젠트 수정 + 디스패치 무결성 + C8/C10 베이스라인을 P0 전에 → S-1로 채택.
5. 임계치: 테이블 단위 1e-12 (C3) / 솔버 수준 응력 1% (A3) / FD-Jacobian C8 1e-5 → 층별 분리 채택.
6. P4 주차 (B5·C1·C5 일치, C4 문서 스캐폴드만) → master에 커밋 항목으로 넣지 않음.
7. 어닐링은 상향횡단 시에만 리셋 (B1) → Newton 루프 내 반복 소거 방지.
8. 팽창분할은 어댑터 래퍼 (B1·C2 일치) → 커널 무수정.
9. 속도/온도 인자는 params 구조체 경유, solver 신규 인수 금지 (B5) → jit 캐시 보존.
10. 누락 각도 대체: B4 외부참조 → 온디스크 htm 27종으로 대체 가능 (C:\SIMULIA\…\SIMACAEVERRefMap).
    A4 API 전체매핑·B2 터치포인트맵·A1 정준파라미터·B3 BB상세는 S-1 산출물로 이관 (§7 Open).

## S-1 선행게이트 (P0 진입 조건)

- `tests/element_contract/test_material_tangent.py` 통과 (J2 86% 탄젠트 수정).
- 디스패치 무결성: 신규 FIELD 경로에 dispatch-reject 테스트 (C10식 JAX≡Numba + T-zero-field identity).
- C8/C10 베이스라인 테이블 기록 (이후 회귀 비교용).

## S0 공통계약 + 러너

- 커널 인터페이스 동결: `pk2_voigt(F, state, params(T,FIELD)) + tangent_voigt + batch`.
- 상태벡터 고정폭+마스크 (가변길이 금지 — JAX 재컴파일 방지).
- 단위요소 러너 1종 (CPE4R/C3D8R, dict-form `{pid:name}` 조립 경유 — 커널 직접 호출 금지).
  재료블록 분리 (`temp_elastic_ef1.inp` 패턴). RED-first.

## S1 최소P0 (T-first)

1. `dependence_table.py` + 단위테스트 (노드 exact, 중간점 1e-12, clamp).
2. `*TEMPERATURE` predefined plumbing (절점→GP 형상함수 보간, 균일장 우선).
3. `tempelastic` 등방 + `tempplastic` Mises등방 + `simpleexpansion` 등방 + `annealtemp`.
4. 검증 4케이스 (인장+전단 × T상수/T램프) vs inp, 응력 1%.
5. DoD: `verification/run_all` 9/9 + ex12 온도폴드 1회 end-to-end.

## S2 P1 (경화3종 + JC)

- 선형이동 → 다중선형 → 복합 → JC 준정적 → JC 속도/온도항 → ratedep.
- Bauschinger 역항복 2σy0−σfwd (1e-6), JC는 T=Tref·ε̇=ε̇0에서 기저경화로 축퇴 (1e-9).
- JC는 rate-frozen 반일치 탄젠트 (문서화). eigh NaN 가드 필수 (§4.3).

## S3 P2 (조건부, PSA 게이트)

- 착수 조건: P0 온도폴드에서 PSA 잔류변형이 실측과 어긋남 확인 시에만 (C1 게이트).
- 순서: Ogden/Mooney → Mullins → Bergström-Boyce (역전 금지).
- Mullins 재장전 불연속은 물리 → C8 비평활 예상으로 문서화.

## S4 P3/P4

- P3: Neuber/Glinka 후처리만 anytime. 크리프·2층·감쇠는 solver 인터페이스 문서 후.
- P4 (damage/concrete): master 제외, 별도 제안서만. 재진입에 승인 필요.

## 검증 게이트

- G0: `tests/test_material_<family>.py` + `verification/run_all` 9/9.
- G1: 전체 pytest vs 135/10 베이스라인. 신규 실패 = block.
- G2: C8/C10 touched elements. 빨간 위 초록 금지 (§4.17).

## §7 Open (R1 누락분 이관)

- A1 정준파라미터표 (Ogden/Mooney/JC/Anand) — htm + B4대체문서에서 추출.
- B3 BB 2-네트워크 유량식 상세 — `nonlinviscohyper` htm + `viscnet_*` inp.
- A4 MaterialDef 확장 매핑표 — factory.py 기준 초안 작성.
- B2 터치포인트맵 — `_assemble_*` 호출점 목록화.

벤치마크 일괄실행은 구현+점검 완료 후 (요구서 §5 지시 유지).
