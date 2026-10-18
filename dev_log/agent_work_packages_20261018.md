# 에이전트 작업 패키지 P1~P6 (워크트리별 병렬 진행용)

**작성일**: 2026-10-18
**근거 문서**: `dev_log/abaqus_feature_gap_dispfold_20261018.md` (목록 번호는 이 문서의 번호)
**사용법**: 패키지마다 워크트리와 브랜치를 하나씩 만들고, "공통 지침"과 해당 패키지 프롬프트를 에이전트에게 함께 전달한다.

권장 워크트리 이름: `feat/p1-reader-keywords`, `feat/p2-output-vars`, `feat/p3-procedures`, `feat/p4-dispfold-parity`, `feat/p5-kinematic-hardening`, `feat/p6-cohesive-coh2d4`.

## 통합자 사전 작업 (에이전트 시작 전)

1. 메인 트리의 미커밋 변경(`example_disp_fold/*.py`, `examples/ex13_3d_display_plate_fold.py`, `LESSONS_LEARNED.md`, `dev_log/*`, `my_plan.md` 등)을 커밋하거나 정리한다. 워크트리는 HEAD 기준이라 이 변경이 빠지면 P4가 나중에 충돌한다.
2. 위 목록 문서와 이 문서도 같이 커밋한다. 에이전트가 읽어야 한다.
3. 워크트리는 정리된 HEAD에서 만든다.

---

## 공통 지침 (모든 에이전트에게 전달)

작업 전에 반드시 읽을 것: 루트 `AGENTS.md` (특히 §3 검증 절차, §3.2 hand-off 규칙, §4.14~4.17 수치 함정), `GEMINI.md`의 "릴리즈 및 영역별 분석 인덱스".

규칙:
- 자기 워크트리와 브랜치에서만 작업하고 커밋한다. 공유 브랜치에 직접 커밋하지 않는다.
- 소유 파일 밖은 수정하지 않는다. 밖의 수정이 필요하면 hand-off의 "통합자 요청" 항목에 적는다.
- `git stash`, `git add -A`, 자기가 바꾸지 않은 파일에 대한 `git checkout -- <file>` 금지. 파일은 이름으로 stage한다.
- 공유 append 파일은 수정하지 않는다: `LESSONS_LEARNED.md`, `release_notes_*.md`, `GEMINI.md`, `AGENTS.md`. 교훈과 릴리즈 노트 문구는 hand-off 메시지에 적는다.
- 시작하자마자 자기 워크트리에서 `pytest tests/ -q`를 돌려 **실패 목록을 기록**한다. AGENTS.md의 "135 passed, 10 failed" 기준선은 오래되었으니 신뢰하지 말고, 이 실측 목록을 A/B 기준으로 쓴다.
- 커밋 전에 `python -m verification.run_all --changed`를 실행한다.
- 실행은 `python -u`로 한다. 긴 해석은 로그로 보내고 `-u` 없이는 멈춘 것처럼 보인다 (AGENTS.md §4.6).
- 수렴 성공은 물리적 정확성의 증거가 아니다 (AGENTS.md §4.9). 수치 검증은 해석해나 기준값과 비교한다.
- 자리표시자, `test.skip`, 스텁 테스트, 미구현 분기를 남기지 않는다. 못 한 것은 hand-off에 못 했다고 적는다.
- Abaqus 입력에서 생략된 파라미터는 사내 임의값이 아니라 Abaqus Keywords Guide 기본값을 따른다.

hand-off 형식 (작업 끝에 반드시 제출):
1. 브랜치와 커밋 해시
2. 변경 파일 전체 목록 (추가/수정/삭제 구분)
3. 실행한 테스트와 통과/실패 수 (시작 시 기록한 실패 목록과의 차이 포함)
4. 동작 변화 (기존 사용자에게 달라지는 점)
5. 통합자 요청 (소유 밖 수정이 필요한 부분)
6. 새 교훈과 릴리즈 노트 초안 (통합자가 `LESSONS_LEARNED.md`, `release_notes_*.md`에 반영)
7. 미완료와 미확인 항목

---

## P1. 리더 키워드 추가

**브랜치**: `feat/p1-reader-keywords`
**목표**: 아래 키워드를 `.inp` 리더에 등록하고, 이미 있는 `Model` 객체에 연결한다.
- `*CONTROLS` → `Model.SolutionControl`
- `*CONTACT CONTROLS` → `Model.ContactControl`
- `*CONTACT STABILIZATION` → `Model.ContactStabilization`
- `*SPRING`, `*DASHPOT` → 요소 `SPRING1`/`SPRING2`

**소유 파일**: `dispsolver/io/keywords/` 아래의 **새 파일**(예: `solution.py`, `springs.py`), 키워드 등록부(필요한 최소 수정), 새 테스트 파일.
**수정 금지**: `io/keywords/steps.py`, `io/keywords/material.py`, `dispsolver/model/` (P3·P5 소유). `Model` 쪽에 파라미터가 부족하면 hand-off의 통합자 요청에 적는다.

**작업 순서**:
1. `dispsolver/io/keyword_base.py`와 기존 키워드 구현(`io/keywords/p0_2.py`, `interaction.py`)을 읽고 등록 방식(`@keyword`, `Schema`, `Param`, `support=` 등급)을 따른다.
2. Abaqus Keywords Guide의 각 키워드 파라미터와 데이터 라인 형식을 확인한다. 지원하지 않는 파라미터는 조용히 버리지 말고 기존 방식대로 경고 원장에 올리거나 거부한다 (`supported=`).
3. 각 키워드를 대응하는 `Model` 메서드에 연결한다.
4. 키워드별 소형 덱 테스트를 만든다. 읽은 뒤 `Model`에 값이 들어갔는지, `execute_inp`에서 실제로 반영되는지(예: `*CONTROLS`가 수렴 기준에 영향) 확인한다.

**완료 기준**:
- 각 키워드가 `GLOBAL_KEYWORD_REGISTRY`에 등록되고 `release_notes_keyword.md`의 정본 개수 표기와 맞는 후보 문구가 hand-off에 있다.
- 새 테스트 통과. `tests/inp_corpus` 및 기존 리더 테스트(`test_abaqus_reader_and_executor.py` 등) 무회귀.
- 파싱만 되고 실행에 반영되지 않는 키워드는 "PARTIAL"로 명시한다.

---

## P2. 출력 변수 추가

**브랜치**: `feat/p2-output-vars`
**목표**: 아래 변수를 `output/`에 추가한다.
- 에너지 히스토리: `ALLAE`(인공 변형 에너지), `ALLSD`(안정화 소산), `ALLFD`(마찰 소산)
- 접촉 필드: `CSHEAR`, `COPEN`, `CSTATUS` (기존 `CPRESS`, `CSLIP` 옆에)
- 응력 파생: `MISES`, 최대/최소 주응력(`MAXP`, `MINP`)

**소유 파일**: `dispsolver/output/` 전체, 새 테스트.
**수정 금지**: 솔버 파일 (`solver*/`, `constraint*/`, `contact/`). 필요한 데이터가 솔버에서 노출되지 않으면 어떤 값이 어디서 노출돼야 하는지 hand-off의 통합자 요청에 적는다. 그 변수는 "데이터 미제공"으로 표시하고 가짜 값을 만들지 않는다.

**작업 순서**:
1. `output/variables.py`, `recovery2d.py`, `recovery3d.py`, `manager.py`, `requests.py`를 읽고 변수 등록과 계산 경로를 파악한다. 이미 파생 계산이 다른 곳에 있는지 먼저 grep한다 (`MISES`, `principal` 등).
2. 응력 파생(`MISES`, `MAXP`, `MINP`)은 recovery 결과 S에서 계산한다. 2D 평면 변형률에서는 면외 응력 성분을 포함해 Mises를 계산해야 한다.
3. 접촉 변수는 접촉 구속 객체가 이미 가진 값에서 가져온다. 에너지 변수는 솔버가 노출하는 값이 있을 때만 연결한다.
4. 변수 이름은 Abaqus 정의를 따른다. 별칭과 요청 검증(`FieldOutputRequest`/`HistoryOutputRequest`가 새 이름을 받는지)을 확인한다.

**완료 기준**:
- 변수별 수치 검증: 단축 인장 요소에서 `MISES`가 해석해와 일치, `MAXP`/`MINP`가 응력 텐서의 고유값과 일치(1e-12), 2D와 3D 모두.
- 접촉·에너지 변수는 데이터가 있는 경우 1개 이상의 실제 모델에서 비영 값을 확인.
- 기존 출력 테스트 무회귀.

---

## P3. 해석 절차: STABILIZE 포팅 + 점탄성 시간 의미 조사

**브랜치**: `feat/p3-procedures`
**목표**:
1. `*STATIC, STABILIZE` (안정화 계수와 에너지 비율 두 형식)를 solver2d/3d에 구현하고 리더 `Static` 파라미터에 추가.
2. STATIC 스텝에서 증분 시간 `dt`가 점탄성(Prony/WLF)에 물리 시간으로 쓰이는지 **측정**하고, 결과에 따라 `*VISCO` 스텝 설계안을 작성.

**소유 파일**: `dispsolver/io/keywords/steps.py`, `dispsolver/model/step.py`, `dispsolver/solver2d/dynamic2d.py`, `dispsolver/solver3d/dynamic3d.py`, `dispsolver/solver/step_executor.py`, 새 테스트.
**수정 금지**: 접촉·요소·재료 코드, `output/`.

**작업 순서**:
1. 레거시 `dispsolver/solver/stabilization.py`(`AbaqusViscousStabilization`)와 `dynamic.py`의 사용처를 읽는다. 수식(`F_stab = c·M_diag·v`, 에너지 비율 0.5% 클램핑)을 Abaqus Theory Guide 정의와 대조한다.
2. solver2d/3d의 Newton 루프와 SDI 경로에서 안정화 힘/강성을 삽입할 위치를 정한다. AGENTS.md §4.8의 교훈: 조립 경로에서 조기 `return`으로 공통 꼬리를 건너뛰는 일이 없게 한다. 3D에는 실행 경로가 둘(`solve_step`과 `_solve_step_with_hooks`)이라는 `area_analysis_solver_20260919.md`의 지적을 반영해 둘 다 처리한다.
3. 리더 `Static` 파라미터를 추가하고 `StaticStep(**kwargs)`로 전달한다. Abaqus 기본값(Keywords Guide)을 따른다.
4. 점탄성 시간 의미 조사: 단일 PSA 요소로 일정 변형률을 가하고 `dt`를 바꿔 응력 이완이 물리 시간 `t`에 대해 Prony 해석해와 맞는지 확인한다. STATIC과 DYNAMIC 모두. 결과를 dev_log 형식의 보고서(`dev_log/viscoelastic_time_semantics_<날짜>.md`)로 저장한다. 구현은 하지 말고 설계안(필요한 변경점)까지만 쓴다.

**완료 기준**:
- 안정화 on/off 비교: 모든 에너지 비율이 설정값 이하로 유지되고 `ALLSD/ALLIE`가 계산됨(출력 변수 노출은 P2와 별개이므로 솔버 내부 값으로 검증).
- stabilize off일 때 기존 결과와 비트 동일.
- 기존 ex12 계열과 `verification.run_all --changed` 무회귀.
- 점탄성 시간 의미 보고서 존재.

---

## P4. 프레임워크 패리티: exporter + 다단계 접힘

**브랜치**: `feat/p4-dispfold-parity`
**목표**:
1. `example_disp_fold/abaqus_api_builder.py`가 실제 해석과 같은 덱을 내보내게 한다: `*RIGID BODY`(판 RP), 회전 BC(UR3) + `*AMPLITUDE`, `*TIE`, `*SURFACE`/`*SURFACE INTERACTION`/`*CONTACT PAIR`(필요 시), `*OUTPUT`.
2. 내보낸 덱을 `execute_inp`로 실행해 손루프(`ex1_dispfold_teardrop.py`) 결과와 대조.
3. 다단계 접힘(접기 → 유지 → 펴기)을 `SimulationCaseSpec`과 실행 경로에 추가.

**소유 파일**: `example_disp_fold/` 전체, 새 테스트.
**수정 금지**: `dispsolver/` 전체.
**선행 확인 (가장 먼저)**: 2D 솔버가 RP의 UR3 처리를 실제로 하는가. 작은 덱(RP 하나에 강체 요소 하나, UR3 처방)으로 확인한다. 동작하지 않으면 고치지 말고 재현 덱과 증상을 hand-off의 통합자 요청에 적고, 가능한 범위(예: 3번 항목의 exporter 형식 정리, 다단계 구조)를 계속 진행한다.

**작업 순서**:
1. 선행 확인.
2. `generate_inp_string`에 누락된 키워드를 추가한다. Abaqus 형식을 지키고 `validate_inp_roundtrip`을 노드/요소 수 비교에서 키워드 수준 비교(구속, BC, 재료)로 확장한다.
3. 손루프와 내보낸 덱 실행 결과의 반력 모멘트(`M_L`, `M_R`)와 최종 형상을 2D에서 비교한다. 3D는 2D가 맞은 뒤에 한다.
4. 다단계: 손루프는 단일 스텝 구조이므로, 스텝 목록을 받는 구조로 바꾸되 기존 단일 스텝 호출의 결과는 비트 동일하게 유지한다.

**완료 기준**:
- 2D teardrop 프리셋에서 내보낸 덱의 실행 결과가 손루프와 허용 오차 내 일치(허용 오차는 근거와 함께 hand-off에 명시). 불일치 시 원인 분리 보고.
- 다단계 접힘 스모크 테스트(`max_steps` 사용) 통과.
- AGENTS.md §1.0의 성공 판정 3항목(요소 반전 0, 매끄러운 곡률, 판 평행)을 접힘 결과에서 확인하되, 이번 변경이 이를 바꾸지 않았음을 보인다.

---

## P5. Kinematic/Combined hardening 연결

**브랜치**: `feat/p5-kinematic-hardening`
**목표**: `*PLASTIC, HARDENING=KINEMATIC|COMBINED`와 `*CYCLIC HARDENING`을 리더에서 받아 2D J2 경로에서 실제로 쓰이게 한다.

**소유 파일**: `dispsolver/material/plastic_hardening.py`, `dispsolver/io/keywords/material.py`의 PLASTIC 관련 부분, 2D J2 재료 커널 파일, `tests/element_contract/`의 추가분, 새 테스트.
**수정 금지**: `output/`, `steps.py`, `solver2d/dynamic2d.py`의 절차 로직. 재료 커널과 디스패치에 필요한 최소 수정만 허용하고 hand-off에 명시.

**작업 순서**:
1. `plastic_hardening.py`를 읽고 지원 범위(선형 kinematic, combined, Armstrong-Frederick/Chaboche 다중 backstress)와 현재 테스트 여부를 확인한다. 어떤 솔버도 참조하지 않는 상태임이 이미 확인되었으니, 연결 지점(상태변수 배치, 접선 강성, 유한변형 처리)을 설계한다.
2. 이 코드베이스의 J2는 유한변형 곱셈 소성(Hencky 로그변형률)이다 (AGENTS.md §2). 커널이 이 틀과 맞는지 확인하고, 안 맞으면 어떻게 맞출지 hand-off에 적는다. 소변형 전제의 커널을 유한변형 경로에 그대로 붙이지 않는다.
3. 상태변수(backstress)를 상태 커밋·컷백 롤백 경로에 통합한다. 컷백 시 상태 오염이 없어야 한다 (트랜잭션 방식 커밋, `area_analysis_solver` 참조).
4. 리더: `HARDENING` 파라미터의 `supported` 확대, `*CYCLIC HARDENING` 추가.
5. 일관 접선이 맞는지 유한차분으로 검증한다 (AGENTS.md §4.15의 표준 검증: 접선이 잔차의 야코비안인가).

**완료 기준**:
- 1요소 사이클 인장-압축에서 Bauschinger 효과가 해석해(선형 kinematic의 닫힌 해)와 일치.
- `HARDENING=ISOTROPIC` 결과가 변경 전과 비트 동일.
- 접선이 유한차분 야코비안과 일치. `tests/element_contract` 무회귀.
- 지원하지 않는 형식(예: 3D 경로)은 명시적으로 거부하고 hand-off에 적는다.

---

## P6. Cohesive 요소 COH2D4 (완료: 2026-10-18, tests/test_coh2d4.py 11/11 GREEN)

**브랜치**: `feat/p6-cohesive-coh2d4`
**목표**: 2D 4절점 점착 요소 `COH2D4`와 `*COHESIVE SECTION`, traction-separation 거동(선형 탄성 시작 + 손상 개시/진전).
**상태**: **COMPLETED (완료)** — `tests/test_coh2d4.py` 11개 전 검증 항목 100% 통과 (강체 모드 1.45e-11, 탄젠트 < 1.2e-5, 파괴 에너지 보존 < 1.2e-5, 압축 침투 차단, Abaqus 키워드, DynamicSolver2D 다중 스텝 박리 수렴 실증). 상세: `dev_log/cohesive_coh2d4_p6_20261018.md`.

**소유 파일**: 새 파일들 (예: `dispsolver/element2d/coh2d4_*.py`, 새 테스트). `dispsolver/model/element_catalog.py`와 `dispsolver/model/assembly.py`는 **등록 한 줄 수준**만 허용하고, 수정한 줄을 hand-off에 명시.
**수정 금지**: 기존 요소 커널, 솔버 루프.

**작업 순서**:
1. Abaqus Analysis User's Guide의 cohesive element 정식화(두께 방향 정의, 정상/전단 분리, 퇴화 처리)와 `*DAMAGE INITIATION`/`*DAMAGE EVOLUTION` 형식을 확인한다. 리더에 이미 있는 연속체 손상 키워드와 어떻게 구분되는지 정한다.
2. 요소 커널을 Numba로 작성한다 (프로젝트 방침: 신규 재료·요소 R&D는 JAX 허용, 운영은 Numba). 일관 접선을 갖추고 AGENTS.md §4.14~4.17 표준(회전된 참조 구성에서 검증, 접선=잔차 야코비안)을 적용한다.
3. 요소 contract 스위트에 등록해 C1~C11 중 적용 가능한 것을 통과시킨다. 적용되지 않는 contract는 이유와 함께 `_XFAIL` 규칙(측정값과 원인 명시)에 따라 기록한다.
4. 손상 없는 순수 탄성 구간부터 구현·검증하고, 그 다음 손상 진전을 추가한다.

**완료 기준**:
- DCB(이중 외팔보) 박리 벤치마크에서 임계 하중 또는 하중-변위 곡선이 해석해와 일치.
- 콘택트 스위트와 기존 요소 테스트 무회귀.
- 가능한 범위(구현 완료/부분/미구현)를 hand-off에 정직하게 구분.

---

## 통합 순서 제안

P1 → P2 → P3 → P5 → P4 → P6.
- P1·P2는 새 파일 위주라 충돌 위험이 가장 낮다.
- P3은 `steps.py`와 solver 파일을 만지므로 P1 뒤에 합친다.
- P4는 P2·P3의 결과를 활용하면 좋지만 막히지는 않는다.
- 통합자는 각 hand-off마다 AGENTS.md §3.2의 절차를 따른다: 현재 HEAD로 rebase/cherry-pick, 삭제된 줄 검토(`git diff HEAD~1 -- <file> | grep '^-'`), 대상 테스트와 `tests/inp_corpus`와 `verification.run_all --changed` 실행, hand-off된 파일만 커밋. 에이전트가 제출한 교훈과 릴리즈 노트 초안은 이때 통합자가 `LESSONS_LEARNED.md`, `release_notes_*.md`에 반영한다.
