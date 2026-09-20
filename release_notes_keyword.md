# 🚀 WHT_DispFoldSolver: Abaqus Keyword Reader & Executor Release Notes

> **최초 작성일**: 2026-09-20
> **담당 팀**: `/lucy-orcha` B팀 (체계 설계) + C팀 (개발방법 개선)
> **준수 표준**: SIMACAEKEYRefMap 키워드 양식, Anti-Mocking Protocol, Karpathy AI Engineering Guidelines
> **상태**: **IN PROGRESS — P0/P1/P2 완료 (47/47 Tests Passed, 2026-09-20).
> 명시 중단 키워드는 아래 §8 로드맵에 따라 구현을 이어간다.**

---

## 📌 목차
1. [개요 및 핵심 아키텍처](#1-개요-및-핵심-아키텍처)
2. [공개 API 및 사용 예제](#2-공개-api-및-사용-예제)
3. [상용 CAE (Abaqus)와의 이론적 차이점 및 우월성](#3-상용-cae-abaqus와의-이론적-차이점-및-우월성)
4. [속도 병목 분석 및 달성 기법](#4-속도-병목-분석-및-달성-기법)
5. [공인 벤치마크 검증 지표](#5-공인-벤치마크-검증-지표)
6. [변경 이력 및 주요 구현 파일 색인](#6-변경-이력-및-주요-구현-파일-색인)
7. [독립 재측정 기록](#7-독립-재측정-기록)
8. [명시 중단 키워드 구현 로드맵](#8-명시-중단-키워드-구현-로드맵)

---

## 1. 개요 및 핵심 아키텍처

Abaqus `.inp` 파일을 내부 `Model` 일급 객체로 읽고 실행하는 고정 인터프리터 파이프라인이다. 코드 생성 방식이 아니라 선언형 키워드를 데이터로 해석하므로, 미지원 키워드는 조용히 버려지지 않고 경고 원장 또는 명시 중단 중 하나로 처리된다.

```
                    [.inp text]
                         │
              [Lexer: abaqus_lexer]
     토큰화 게이트 (3단어 키워드·따옴표 콤마·末尾콤마 병합)
                         │
     [Registry: GLOBAL_KEYWORD_REGISTRY]  ← 유일 진입점
     keywords/{scope,mesh,material,interaction,steps,p0_2}.py
     처분 3항: RAISE(불법 문맥) / WARN+원장(합법 미구현) / IGNORED(명시 무시)
                         │
            [Model 일급 객체 + Step 생명주기]
     Part/Assembly/Instance · OP=NEW/MOD 전파 · 진폭/하중/구속
                         │
        [Executor: execute_inp / MultiStepExecutor]
     하중 없는 단계는 smoke 경로(반복 횟수 보존),
     하중·상호작용 단계는 전체 경로로 자동 전환
```

등록 키워드 48종 정본(+별칭 58종 raw). 설계 문서는
`.agents/plans/abaqus_inp_reader_master_plan_20260920_team_B.md`와
`.agents/plans/devmethod_master_plan_20260920_team_C.md`이다.

---

## 2. 공개 API 및 사용 예제

```python
from dispsolver.io import read_inp, execute_inp, execute_inp_smoke

# 읽기: Model 일급 객체 (파싱만, 해석 없음)
model = read_inp("cook_2d.inp", dim=2)

# 실행: 하중 유무에 따라 smoke/전체 경로 자동 선택
res = execute_inp("cook_2d.inp", dim=2, verbose=False)
print(res["converged"], res["unsupported_keywords"], res["warnings"])

# 단일 단계 스모크 (반복 횟수 포함)
res = execute_inp_smoke("cube.inp", dim=3, verbose=False)

# 엄격 모드 (기본 True): 불법 문맥은 예외, 관용 모드는 strict=False
model = read_inp("legacy.inp", dim=2, strict=False)
```

미지원 키워드는 `res["unsupported_keywords"]`와 `res["warnings"]`에
집계되며, 내부 착지가 없는 키워드(`ORIENTATION`, `EQUATION` 등)는
`_AbortParse`로 파싱을 중단한다. 착지 없는 채로 계속 풀면 다른 모델을
푸는 셈이 되기 때문이다.

---

## 3. 상용 CAE (Abaqus)와의 이론적 차이점 및 우월성

| 항목 | Abaqus | 본 구현 | 근거 |
|---|---|---|---|
| `*INSTANCE` 합성 순서 | 병진 후 회전 `x'=R(x+t)` | 동일 (flatten 1곳 합성) | 혼합 픽스처 `(1,0)+t(1,0),90도 → (0,2)` 통과 |
| 불법 문맥 키워드 | 입력 오류 | 구조 키워드는 예외, 데이터는 경고+건너뜀 (STRICT 기본) | `*INSTANCE` ASSEMBLY 밖 기각 프로브 |
| 미지원 키워드 | 경고 후 계속 | 동일 + 원장 보고, 착지 없음은 중단 | 전수 인벤토리 ERROR 0건 |
| 진폭 사슬 | `AMPLITUDE`→BC/하중 이름 해석 | 동일 (문자열 보관, 해석은 실행 시점) | BC/CLOAD 진폭 전달 프로브 |
| `OP=NEW/MOD` | 단계 전파 규칙 | 동일 (Step 생명주기 상태와 정합) | 2단계 전파 덤프 테스트 |

---

## 4. 속도 병목 분석 및 달성 기법

- 렉서는 I/O 대비 무시 수준이며, 11MB `cook_3d.inp`(절점 101,693개) 파싱이
  수 초 내 완료된다. 별도 병목 최적화는 수행하지 않았다.
- `keywords/` 분리로 핸들러 등록 경로가 단일화되었고, 레거시 2벌 파서와의
  발산은 G0 동등성 테스트로 차단한다.

---

## 5. 공인 벤치마크 검증 지표

| 덱 출처 | 규모 | 판정 |
|---|---|---|
| 공식 fetch 19종 (`benchmark_inp/official/`) | `cook_3d` 최대 101,693절점/85,242요소 | 13종 정상, 6종 명시 중단 |
| 검증 아카이브 전수 1,859종 (`benchmark_inp/verification/`) | 단일 요소 위주 | 정상 1,377종, 명시 중단 482종, 오류 0건 |
| 재료 검증 26종 페이지 참조분 (중복 포함 1,465건) | 상동 | 정상 944건, 명시 중단 481건, 오류 0건, 누락 0건 |
| 자생 등가 11종 (`benchmark_inp/*.inp`) | 단위 기하 | 11종 정상 |

미지원 상위 키워드는 재료 팀 백로그와 일치한다 (`RATE DEPENDENT`,
`DRUCKER PRAGER`, `CONCRETE DAMAGED PLASTICITY` 계열). 리더는 이를
정확히 보고하고 수치 구현에는 손대지 않는다.

테스트 게이트: `test_abaqus_reader_and_executor.py`(12),
`test_3d_inp_parser.py`(1), `test_abaqus_official_inp.py`(11),
`test_abaqus_official_fetch.py`(20), `test_abaqus_verification_inp.py`(3).

---

## 6. 변경 이력 및 주요 구현 파일 색인

- P0-0: `execute_inp` 다단계 디스패처, `smoke` 별칭, 솔버 팩토리 주입
  (`io/abaqus_reader.py`, `solver/step_executor.py`)
- P0-1: 상태 전이, `MATERIAL` 종료 규칙, 2등급 가드, `TEMPERATURE` 수정,
  `RIGID BODY NSET` 수신, `TIE` 전행, 인스턴스 스코프 격리, 미지원 보고
- P0-2: `AMPLITUDE`/`DENSITY`/`TRS`/`DLOAD`/`MPC`/초기조건 이식,
  `COUPLING` 계열, `FRICTION`, 착지 없음 4종 중단
- P0-3: `T-then-R` 합성점, 이중 평탄화 정본 선언, G0 동등성 게이트
- P1: `OP=` 의미론, STEP 오버레이, `keywords/` 분리, STRICT 기본값,
  G3/G4/G5 게이트, Lark 기각(표준 라이브러리 방식 채택)
- P2: 렉서 G1 업그레이드, `KeywordSpec` 문서 필드와 지원 등급 게이트,
  G6 무경고 게이트
- 공식덱 발견 수정 3건: 연속 `*MATERIAL` 암시 종료,
  `DISTRIBUTING/ KINEMATIC COUPLING` 복합 키워드, STEP 내 진폭·접촉 등록

주요 파일: `dispsolver/io/abaqus_reader.py`,
`dispsolver/io/keyword_base.py`, `dispsolver/io/keywords/*.py`,
`dispsolver/io/keyword_registry.py`, `dispsolver/io/abaqus_lexer.py`,
`dispsolver/model/model.py`, `dispsolver/model/step.py`,
`dispsolver/model/assembly.py`, `dispsolver/model/instance.py`,
`dispsolver/solver/step_executor.py`.

---

## 8. 명시 중단 키워드 구현 로드맵 (계획, 미구현)

아래 키워드는 현재 `_AbortParse`로 파싱을 중단한다. 이는 임시 회피가
아니라 구현 예정 목록이며, 각 항목은 착지 설계와 함께 배치로 들어간다.

| 키워드 | 중단 규모 (전수 기준) | 착지 설계 | 배치 |
|---|---|---|---|
| `ORIENTATION` | 약 163건 | `section.py`의 방향 필드에 기록 후 재료 호출에 전달 | 다음 배치 |
| `EQUATION` | 약 145건 | 선형 다점 구속 신규 모델 클래스 필요 | 다음 배치 |
| `TRANSFORM` | 소수 | 절점 변환 저장 슬롯 신설 필요 | 다음 배치 |
| `INCLUDE` | 16건 | 렉서 단계 파일 삽입과 경로·순환 가드 | 다음 배치 |
| `INITIAL CONDITIONS` 변형 (`FIELD`, `RATIO`, `HARDENING`, `PLASTIC STRAIN` 등) | 약 80건 | `PredefinedField`형 보관 후 솔버 소비 연결 | 다음 배치 |

원칙은 기존과 같다. 파싱과 보관은 리더에서 처리하고, 실행 연결은
솔버에 대상이 있을 때만 하며, 없을 경우 미연결로 명시한다.

---

## 7. 독립 재측정 기록

- 2026-09-20, `pytest tests/test_abaqus_reader_and_executor.py
  tests/test_3d_inp_parser.py tests/test_abaqus_official_inp.py
  tests/test_abaqus_official_fetch.py tests/test_abaqus_verification_inp.py -q`:
  **47 passed** (worktree `feat/inp-reader-p0`, 메인 venv).
- 2026-09-20, `benchmark_inp/verification/inventory_all.py` 전수:
  **OK=1377, ABORT=482, ERROR=0**.
- 2026-09-20, 등록 현황 직접 조회: **정본 48종, raw 58종**.
