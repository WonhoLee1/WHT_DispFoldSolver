# Abaqus Input Reader 체계 확립 — Master Plan (B팀 융합, 2026-09-20)

> Round1→Round2→Round3 전량 융합. 5패널(B1 taxonomy / B2 assembly-instance / B3 executor-binding / B4 coverage-index / B5 robustness-critic) 합의+쟁점판정 포함.
> 정본 문서: 로컬 `C:\SIMULIA\Documentation\2024LE\English\SIMACAEKEYRefMap\` (r-페이지 594, m-링크 593+anchor중복1, Std 515 — B4R3 재현 확인). `desktop-whl:4040`은 셸만 반환 → 미사용.
> 기존 구현: `dispsolver/io/abaqus_reader.py` 741줄(정본 32종+별칭9=raw41), `keyword_registry.py`, `abaqus_lexer.py`, `abaqus_parser.py`(레거시), `dispsolver/model/` 일급객체, `tests/test_abaqus_reader_and_executor.py` 4PASS, `dev_log/abaqus_reader_and_executor_20260919.md`.
> 범위: reader 체계 확립 + executor 연결. material 수치식(탄성/소성/초탄성/점탄성 구성식·계수·검증)은 별도 에이전트 영역 — 여기선 키 shape+통과 테스트만 (B3-C4).

## 1. 합의 사항 (5패널 전원 일치, 코드 재확인済)

1. `parse_blocks`(`abaqus_reader.py:94-113`)가 `valid_contexts`를 검사하지 않음 — 6종 컨텍스트는 장식 필드 (B1R2/B2R2/B5-F8 일치).
2. `*TEMPERATURE`는 항상-실패+항상-은폐: `step.py:79` Dict vs `:647` `.append` → AttributeError → `:100-105` blanket try/except가 warning으로 삼킴 (B3R1→B1R2/B5 수용).
3. `*RIGID BODY, NSET=` 미수신: `:505-507`은 elset/tie nset/pin nset만, plain `NSET=` 탈락. `gen_ex12_inp.py:188-189` 주력 덱 형식과 불일치 (B3R1→B2R2 수용).
4. `execute_inp`(`:709-729`)는 첫 non-Initial 1스텝 `solve_step` 1회 — 멀티스텝 침묵 폐기, Step dt/amplitude/load/interaction 무시 (B3R1→B1R2 P0-0 승격).
5. `*AMPLITUDE` 사슬 3중 단절: 핸들러 없음 + `STEP, AMPLITUDE=` 무시 + BC/Load `AMPLITUDE=` 미파싱 (B1/B3 일치, 최우선).
6. `*TIE` 첫행 절단: `:488-498` 첫 유효행 후 break, 멀티행 소실 (B5-F11→B3R2 Stage0 수용).
7. 이중 파이프라인 병존: `abaqus_parser.py`+`model_builder.py`(solver객체 직접조립) vs `abaqus_reader.py`+`Model`(팩토리만). 기본 element값 발산(CPE4 vs C3D8), NSET GENERATE·HYPERELASTIC 폴백 상이 (B5-F5→전원 수용). 정본=레지스트리 경로, 레거시 동결.
8. 이중 flatten: `model_builder._flatten_parts`(2D·elset-drop TODO `:365-369`·회전중심 없음) vs `assembly.build_solver_system`(3D·qualified+bare-alias). 정본=assembly (B2→전원).
9. `Transform3D.apply`(`model/base.py:55-77`)는 R→T 고정. Abaqus 정본은 translation BEFORE rotation (`simakey-r-instance.htm`). identity instance에선 차이 0이라 미발현 (B2→B1/B3 수용, 수정위치는 쟁점 §2-2).
10. `RE_KEYWORD`(`abaqus_lexer.py:33`) 최대 2토큰. B4R3 실증: `*CONTACT CONTROLS ASSIGNMENT, TYPE=FACTOR` → keyword 오판+params 소실 동시 (B4→B1 수용·악성 상향).
11. `*FRICTION` 핸들러 0건 — dev_log "지원" 표기는 허위 (B1R1 지적→B3 확인). 현 3실덱에 없어 무음유실 아님.
12. `isdigit` 필터(`abaqus_reader.py:258,287`, `abaqus_parser.py:183-191`)가 비숫자 토큰 침묵 탈락 — 한정자·범위 NSET 사망 (B5-F6→전원).
13. QUAD4 폴백(`model_builder.py:428`)·RIGID 침묵 return(`abaqus_parser.py:451`)은 §4.8 silent-fallthrough 재현 (B5→전원).
14. `create_solver2d`(`model/model.py:961-1017`)에 constraints/interactions attach 없음, 3D(`:891-915`)와 비대칭 (B3→B2 수용).
15. `*STEP` 핸들러가 `context=STEP` 대입 자체 없음, `*MATERIAL` 진입 후 탈출 규칙 없음, `*END STEP`은 pass (B2-E1/E2→B1/B3 수용).
16. INSTANCE-context NSET→`get_active_part()` 오배선 (`:261-265`, assembly 구간서 `current_part=None` → PART_DEFAULT 오염) (B2-E3→B3 수용).
17. `**` 인라인 절단(`:98-100` 전행 적용), 가짜 `-` 연속행(`:133-146`, 주석은 `,` 규칙), `RE_KEYWORD.finditer` 무차별 블록절단(`:108`, `*TRANSLATION` 오인식) (B5-F2/F3/F4→전원).
18. `OP=` reader 내 0건 — 멀티스텝 BC 의미론 전멸 (B5-F10→B3 수용).
19. `verification/abaqus_official_benchmarks.py`는 inp를 읽지 않음 — reader 게이트 대용 불가 (B5-F15).

## 2. 쟁점 6 판정 (융합 결정)

| # | 쟁점 | 판정 | 이유 |
|---|---|---|---|
| D1 | 가드 수위 (B1 삼원 vs B2 2등급 vs B5 전면strict) | **B1 삼원+ B5 하드게이트 adopted**: 불법-문맥→raise / 합법-미구현→warn+원장 / IGNORED→명시침묵. P0 수락 deck집합 내에선 warn 1건도 gate 실패(=그 범위 전면strict). `strict=False` 명시 호출에서만 관용, 기본 strict | 침묵-파손(TEMPERATURE型)은 원장이, 구조붕괴는 raise가 포획 — 적이 다름. B1R3 V-A 증명(전이 없는 가드는 기존 4테스트 전멸) 따라 전이+MATERIAL-exit+가드를 1PR 원자로 묶음 |
| D2 | T-before-R 수정위치 (B1 base.py vs B2 flatten합성점) | **B2안 채택**: `x'=R(a→b,θ)(x+t)` 합성을 flatten 유일적용점(`assembly.py:196-199`)에. `base.py` 전역변경 금지 | B1안 영향반경 큼(수동 체인·2D in-plane 분기). B3-C6 신설로 정본 1곳 고정, reader/executor 독자해석 금지. 픽스처(t≠0且θ≠0)로 검증 |
| D3 | NSET→STEP 허용 (B1 양시론 vs B2 불허) | **2단계**: P0=현 오배송 차단+warn원장(파트셋 혼입 금지), P1=Step-scoped overlay 구현(`step.py` 확장 비용 명시). STEP 내 정의 raise 아님 | 합법 키워드 raise는 정본배반, 파트셋 혼입은 오염 — 제3경로. B1R3 V-B(Step에 셋 필드 없음) 근거로 P1 비용 명시 |
| D4 | lexer 3단어 (B4 G1전제 vs B2 기각) | **B4 실증 수용, 범위 한정**: P0b 6종은 1-2단어라 P0b 무수정 진행. P0b 완료 직후 G1로 `RE_KEYWORD` 3단어 허용 1행+32종 회귀. 연속행/`**` 의미론은 P0 deck 요구시에만 | B4R3 악성 실증(오판+소실)이 기각 반증. B2 합리핵(도금금지)은 P0b 비블로킹으로 흡수 |
| D5 | P0 범위 (B1집중 vs B4 38+6 vs B5 치명5) | **B1집중+P0a/b 구조 adopted**: P0a 유지32+이식6, P0b 신규6, P0-0 렉서토큰화+멀티스텝. B5 치명5는 수락집합에 자동포함(별도협상 불요). 개수목표 기각 — 판정은 deck집합×zero-hit×PASS | B4R3 종료조건(a)ex12/gen_ex12 100% (b)P0a/b+P1 94종 등록을 gate로 명문화. B5 축소안의 하드거부 5건은 D1 gate에 흡수 |
| D6 | 엄격 vs 호환 (B5 strict-core vs B3 호환게이트) | **엄격-전방/호환-후방**: 기존 4테스트+gen_ex12 deck은 grandfather gate(녹색 유지). 신규 위반 raise. 호환 shim은 파서 밖 별도 레이어, 코어 오염 금지 | B1R3 V-A로 기존deck 녹색 유지 증명 가능. B3 strict모드(`execute_inp(strict=True)`서만 실패)안은 shim 위치로 흡수 |

## 3. 목표 아키텍처

```
.inp → abaqus_lexer(tokenize: L1 토큰화 게이트)
  → GLOBAL_KEYWORD_REGISTRY(handler; KeywordContext 강제; 처분 3항)
  → Model 일급객체만 생성 (Part/Instance/Assembly/Material/Section/Constraint/Step/BC/Load/Amplitude)
  → Model.build_solver_system() → assembly.flatten (x'=R(x+t), qualified+bare-alias)
  → create_solver(3D/2D 자동) → MultiStepExecutor.run_step_tree → DynamicSolver.solve
```

- 분류어: B1 G0-G7 (G0 선언/관리, G1 공간이산화, G2 표면/영역, G3 재료, G4 단면, G5 구속/상호작용, G6 히스토리-절차, G7 히스토리-하중) + L1 토큰화 게이트 선행.
- 상태머신 (B2 v3): GLOBAL→PART→GLOBAL, GLOBAL→ASSEMBLY→GLOBAL, ASSEMBLY→INSTANCE→ASSEMBLY, GLOBAL→MATERIAL→(비-sub 첫키워드)→GLOBAL (sub=ELASTIC/EXPANSION/PLASTIC/CREEP/HYPERELASTIC/VISCOELASTIC/DENSITY), GLOBAL→STEP→GLOBAL (`*STEP` 진입대입+`*END STEP` 복귀 확정). 가드: 등급A hard-reject(중첩PART/ASSEMBLY밖INSTANCE/END부정합/2번째ASSEMBLY/빈명INSTANCE/a==b) + 등급B warn-accept(미정의PART→빈Part, bare first-wins+warn, 혼합덱 병존, IMPORT, 불완전line2). flat R1-R7: PART밖 형상키워드→PART_DEFAULT lazy 생성, 종료시 INST_DEFAULT 1개, transform identity, qualified+bare 등록(ASSEMBLY 우선·충돌warn·union).
- 스코프: PART→Part, INSTANCE→합성명 `INST__SET` 격리(정식 `Instance.sets` 보류), ASSEMBLY→Assembly, GLOBAL/flat→PART_DEFAULT. NSET STEP 불허(P0), overlay P1.
- 규약 (B3 C1-C6): C1 Model팩토리만. C2 문자열참조·지연해석. C3 미연결은 `미연결` 표기 의무("지원" 금지). C4 벤치팀 경계(키 shape+통과만). C5 solver 무변경. C6 INSTANCE 합성정본 1곳.

## 4. 단계 계획 (Stage→P 매핑)

- **P0-0 (렉서+멀티스텝 기반)**: `RE_KEYWORD` 토큰화 유지+별칭표, `execute_inp` v2=`read_inp`→`create_solver`(3D/2D)→`MultiStepExecutor` (기존 단일스텝은 `smoke_` 별칭 강등). B4 G1(3단어 허용)은 P0b 직후.
- **P0-1 (가드+치명수리, 원자PR)**: 전이규칙+MATERIAL-exit+처분3항+검사(`if context not in spec.valid_contexts`)+TEMPERATURE(`add_predefined_field`) + RIGID `NSET=`→tieNset + TIE 전행순회 + INSTANCE 스코프 전환 + `unsupported+warnings`를 `execute_inp` 반환 포함. Grandfather gate(4테스트+ex12 deck 녹색) 머지조건. 수락식: 수락deck집합×guard-hit 0×종단PASS.
- **P0-2 (이식 6+신규)**: MPC/DLOAD/AMPLITUDE사슬(INITIAL/DENSITY/TRS 포함)+ORIENTATION/TRANSFORM/COUPLING/KINEMATIC/EQUATION/INCLUDE. TRS는 shape만. FRICTION은 파싱-보관+미연결 라벨(허위지원 표기 정정).
- **P0-3 (수학+정본)**: T-then-R flatten 합성 + 이중flatten 정본선언(구 `_flatten_parts` 삭제, 호출자 0건 확인 후). 회전-병진 혼합 픽스처.
- **P1**: EQUATION/COUPLING 잔여, INCLUDE展开, VISCO 등 절차, 접촉기초(CONTACT/SURFACE BEHAVIOR/FRICTION/CONTROLS/STABILIZATION/OUTPUT), 재료·단면(HYPERFOAM/MULLINS/DAMAGE/BEAM/SHELL-GENERAL/MEMBRANE/REBAR/DISTRIBUTION). OP= 보관+전파문서, B10/B11 amplitude 전달.
- **P2**: KeywordSpec 문서필드(`products/type/level/support`) + 정합검사 테스트, STEP-overlay, 렉서 의미론 잔여. 열-전기 재료·Explicit 전용은 범위 외.
- **검증 게이트 G0-G6 (B5 v3)**: G0 단일정본(양파서 동등성, 실패시 머지금지→구 deprecate). G1 lexer 상태기계(선두`**`만 주석/末尾`,`병합/등록키워드만 경계, G0직후·G2이전). G2 하드거부(INCLUDE·isdigit→토큰분류기·QUAD4폴백삭제·RIGID침묵삭제·발산). G3 roundtrip(gen_ex12+2D/3D, 멀티행 보존). G4 공식벤치(.inp 체크인+허용오차). G5 BC 의미론(OP·진폭·면매핑 덤프). G6 침묵탐지(`unsupported/skipped` boş assert+`warnings→error`).

## 5. 크롤 인계 (B4 최종)

- 7-튜플 `(KEYWORD, products, type, level, params[], datalines_variants[], slug)`, 예상 593행. 링크 정규식 `href\s*=\s*"(simakey-r-[^"]+)"[^>]*>\*([^<]+)<` (class 조건 금지). `ProductIndex-sb.htm` 제외. Std 필터 `\*([A-Z][A-Z \-/]+)</a>` (515 재현). `params[]`에 Required/Optional 귀속 태그 포함 → G2 판정표 입력. `elastic` 10분기 등 조건부 데이터행은 핸들러 내 `params["type"]` 분기 템플릿으로 표준화.
- 순서: P0a이식(6) → P0b(6) → G1 → P1-S1 → P1-S2 → 종료조건 판정.

## 6. material 벤치마크 팀과 경계

- B팀 소유: 키 shape+통과 테스트(DENSITY 값전달, TRS `extra_params["trs"]` shape+roundtrip, CREEP 파서 존재·실행 미연결 명시).
- 벤치팀 소유: 구성식·계수·수치검증. `to_numba_props` 값 변환식 작성 금지.

## 7. 잔여 리스크

1. `**` 인라인 절단 변경은 기존deck 거동 변경 — A/B 필요 (B1R2).
2. B4 C6/CPE4H 계열·Numba/JAX force 1e-3 divergence 등 element 이슈는 본 플랜 범위 밖 — 인용 금지.
3. Std515 카운트는 재현 스크립트(§5)로만 인용, 카운트 달성치를 gate로 사용 금지.
