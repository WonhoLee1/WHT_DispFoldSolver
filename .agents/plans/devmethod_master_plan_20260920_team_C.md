# 개발방법 현대성 토론 — Master Plan (C팀 융합, 2026-09-20)

> B팀 마스터플랜(`abaqus_inp_reader_master_plan_20260920_team_B.md`) 위 개발방법 논쟁 5패널(C1 파싱 / C2 스키마 / C3 확장성 / C4 실행검증 / C5 비판자) R1→R2→R3 융합.
> 판정 기조: 고정 인터프리터 유지, P0-1 원자PR 보호, grandfather gate. PEG/pydantic/515일괄 전면안 전부 P0 부적격.

## 1. 합의 (전원 일치)

1. 현 렉서 전근대적이나 P0 수술로 버팀 가능. P1에서 Lark로 렉서만 교체 (C1).
2. pydantic 전면 기각. 표준lib 강화(B) 기본, 경계 국소 허용만. 행단위 모델 생성 금지 (C2+C5).
3. 현 registry 확장불가. 단일진입점 모양만, 처분·스코프·크롤연결 미비 (C3).
4. 고정 인터프리터 유지. 코드생성 실행경로 신설 금지. Model→.py 덤프는 디버그용만 (C4).
5. 개수목표 gate 금지. 판정=deck집합×zero-hit×PASS (D5).
6. 이중파이프라인 즉시삭제 금지. 정본선언+동결, G0 게이트는 P0-3 (C5).
7. 전면 즉시STRICT 금지. 신규STRICT+기존WARN유예(기한부) (C2+C5).
8. material/벤치팀 경계: shape+통과만. `to_numba_props` 변환식 금지.

## 2. 쟁점 판정

| # | 쟁점 | 판정 |
|---|---|---|
| D1 | P0-1 원자 범위 | **C5 최소안 7건 + C1 F2/에러corpus수집 포함으로 확정**: ①전이+MATERIAL-exit+검사+등급A단언 ②STEP 진입/복귀 ③TEMP 수리 ④RIGID NSET ⑤TIE 전행 ⑥INSTANCE 스코프 ⑦unsupported+warnings 반환+F2(선두`**`만, 수락deck 무변경 실증済)+에러corpus 목록수집. F3(blanket末尾`,`병합) 제외 — ELEMENT 8절점 오접합 위험, G1에서 핸들러-인식 규칙과 함께 설계 |
| D2 | G0 배치 | **P0-3**. P0-1은 수락deck A/B로 대체. 에러케이스 동등 포함은 수집 P0-1, 강제 P0-3 |
| D3 | STRICT 전환 | **단계적**: 신규 즉시STRICT+C6-3종테스트 머지조건, 기존 WARN+원장(전환기한)+은폐경로 grep첨부. 전역 try/except 신규경로 삭제, 기존경로 원장후 삭제 |
| D4 | Spec 규모 | **코어4 확정**: disposition/support/scope(parent+sub+exits 묶음)/canonical. params_schema·products/type/level/datalines/slug는 백로그 원장 컬럼으로 격리(P1). `description` 이상 문서필드 코드 추가 금지 |
| D5 | keywords/ 분리 | **P1**, G0 단일정본 선언 후. 조건 3원자: 유일소유+충돌거부+레거시 동결. P0-1 PR에 혼입 금지 |
| D6 | G7 시점 | **분할**: G7a(수락집합 golden v0)+G7b(패리티+팩토리주입 5줄) = P0. 풀corpus15·golden확대·퍼즈·CI경유화 = P1/P2. C5 이연안 중 팩토리주입만 P0-0 뼈대로 관철 — DI프레임 아닌 5줄 주입이므로 원자팽창 아님 |
| D7 | Lark 범위 | **P1 렉서 한정** + `tokenize()` 시그니처 동결 + G0/G7 공용게이트. P0 신규의존성 반입 금지, 불가시 stdlib 폴백 |
| D8 | C2 3결정 | ①정규경로 `AbaqusInputReader→Model` 확정(결정 전 신규키워드 동결) ②Spec 코어4+원장격리 ③신규STRICT+기존유예 |

## 3. P0-1 원자PR (확정, 증감 불가)

전이+MATERIAL-exit+검사+단언 / STEP 진입복귀 / TEMP / RIGID NSET / TIE 전행 / INSTANCE 스코프 / 반환포함+F2+에러corpus수집. 머지조건: grandfather(4테스트+ex12 deck 녹색) + 수락deck×guard-hit 0×종단PASS.
명시적 제외: Lark·pydantic·515일괄·분리·Spec+10·G0머지금지·strict즉시반전·corpus/golden/퍼즈/CI·overlay·OP완전의미론·base.py전역수정.

## 4. 순서표

P0-0(렉서유지+뼈대, 팩토리주입 5줄 포함) → P0-1(원자7+F2) → P0-2(이식6+신규6, FRICTION 보관+미연결) → P0-3(T-then-R 합성점+G0) → P0b직후 G1(3단어1행+F6+F3 handlers-aware+32종회귀) → P1(Lark·분리·corpus/G3-G5·overlay·OP/진폭·strict반전) → P2(Spec문서·렉서잔여·퍼즈/CI) → 범위외(열-전기/Explicit·element이슈·to_numba_props).

## 5. 진입점 강제 순서 (C2)

context → params → handler → on_error. 핸들러 내부 개별 get+return 은폐는 신규 키워드 린트 대상. `LENIENT_DEFAULT` 신설 금지.
