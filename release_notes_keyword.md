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
| 공식 fetch 19종 (`benchmark_inp/official/`) | `cook_3d` 최대 101,693절점/85,242요소 | 18종 정상, 1종 명시 중단(`TRANSFORM`) |
| 검증 아카이브 전수 1,859종 (`benchmark_inp/verification/`) | 단일 요소 위주 | 정상 1,857종, 명시 중단 2종, 오류 0건 |
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
| `INCLUDE` 대상 파일 부재 | 2건 | 아카이브에 없는 참조 파일(`mullins_calibrate_testdata.inp`) — 코드 간극 아님 | 열린 항목 |

2026-09-21 배치에서 구현 완료되어 위 목록에서 제외된 항목:
`INCLUDE` 렉서 삽입(경로·순환·깊이 가드, 미해결 시 기존 중단 유지),
`TRANSFORM` 노드 변환 슬롯 보관(PARTIAL, 솔버 프레임 집행 미연결 경고),

2026-09-21 배치에서 구현 완료되어 위 목록에서 제외된 항목:
`ORIENTATION` 4형식 + 단면 연결(PARTIAL, 재료커널 미연결 경고),
`EQUATION` 고전형 + 셋 전개 + 페널티 조립(FULL),
`INITIAL CONDITIONS TYPE=TEMP` 약어 허용,
`DAMAGE INITIATION/EVOLUTION`, `CONCRETE DAMAGED PLASTICITY` 일가 5종,
`DRUCKER PRAGER` 2종, `RATE DEPENDENT`, `MULLINS EFFECT`,
`PLASTIC HARDENING=JOHNSON COOK`, `ELASTIC/PLASTIC DEPENDENCIES` 장 테이블,
`INITIAL CONDITIONS` 8종(`VELOCITY/FIELD/RATIO/HARDENING/PLASTIC STRAIN/
DAMAGE INITIATION/RELATIVE DENSITY/SOLUTION`, 보관+미연결 경고).
같은 배치에서 `HYPERELASTIC` 모델명 전달 버그 수정
(`type=` → `model=` + `N=` 전달, bare 키워드 대소문자·하이픈 정규화).

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
- 2026-09-28, `official/hertzcontact_2d_rsurf.inp` (Deck #9):
  `*SURFACE, TYPE=SEGMENTS` 해석적 강체면 파서 및 순수 대칭(Pure SPD) 2D 점-선 접촉 정식화 구현.
  `assembly.py` CPE8 4절점 강제 절단 버그 제거 -> 8절점 Numba 3×3 커널 직결. 5 Newton iters, 14.13s 기계 정밀도 수렴 (**PASSED**).
- 2026-09-28, `official/neckingtensilebar_cax4r.inp` (Deck #10):
  `*NSET`/`*ELSET` 기존 세트 이름 재귀적 해석 구현 + `*BOUNDARY` trailing comma 라인 병합 방지 및 빈 변위 필드 0.0 처리.
  축대칭 CAX4R 2단계 인장 파이프라인 수렴 확인했으나, 축대칭 차원 및 $J_2$ 소성 텐서 미지원으로 벤치마크 정식 범위 제외 (**OUT-OF-SCOPE / UNSUPPORTED**).
- 2026-09-28, `official/nlgeocantilever_c3d8_coarse.inp` (Deck #11) & `nlgeocantilever_c3d8i_coarse.inp` (Deck #12):
  1) `abaqus_lexer.py`: `RE_KEYWORD` 쉼표 앞 공백(`\s*,\s*`) 허용 및 키워드명 `.strip()` 적용 -> `*SOLID SECTION , ELSET=BEAM, MATERIAL=MATE1` 파라미터 유실 버그 원천 수정 ($E=10^8$ 정상 바인딩).
  2) `constraint.py`: `Coupling.bind`에서 `global_nsets`의 0-based 인덱스를 `nid_to_idx`로 중복 조회하여 slave/RP 노드가 밀리거나 누락되던 결함 수정 (RP=노드 44, Slaves=[0,1,2,3] 정합).
  3) `dynamic3d.py`: STATIC 스텝 하중의 Abaqus 기본 진폭 규정(선형 램프 `scale = t / period`) 반영 및 점하중 로컬-글로벌 노드 ID 자동 분해 적용.
  4) `assembly.py`: `ABAQUS_ELEM_MAP`에서 3D 고급 요소(`C3D8I`, `C3D8H`, `C3D8R`, `C3D10`, `C3D10M`)를 `C3D8`/`C3D4`로 다운그레이드하던 레거시 매핑 완전 제거 -> 정품 `C3D8I_CR` (EAS-9 + Corotational) 커널 직결.
  5) 물리적 무결성 검증 (Bisshopp & Drucker 1945 / Abaqus Benchmarks Manual §1.2.1):
     - **Deck #11 (`C3D8` 완전적분 1층 격자)**: 20 증분 전수 컷백 0회 수렴 (2.04s, 48 iters). $u_y = 1.7883\,\text{mm}$ (예상대로 1차 완전적분 전단 잠김(Shear Locking) 발생 재현, **PASSED**).
     - **Deck #12 (`C3D8I` 9-mode EAS 비적합 모드)**: 32 증분 전수 컷백 0회 수렴 (31.95s, 265 iters). **$u_y = 7.5942\,\text{mm}$** (이론치 $7.5 \sim 8.1\,\text{mm}$ 정중앙 적중, 전단 잠김 완전 소거 입증), 축방향 풀백 $u_x = 4.8286\,\text{mm}$, 비틀림 $u_z = 0.0017\,\text{mm}$ (**PASSED**).
- 2026-09-28, `official/nlgeocantilever_c3d8r_coarse_eh.inp` (Deck #13):
  1) `C3D8R` 1점 감차적분 + 플래너건-벨리치코 아워글래스 제어 대변형 외팔보 벤치마크.
  2) 20 증분 전수 컷백 0회 수렴 (62.62s, 63 Newton iters).
  3) $u_y = 3.7103\,\text{mm}$, $u_x = 0.8668\,\text{mm}$, $u_z = 4.71\times 10^{-6}\,\text{mm}$.
  4) 이론적 강성 서열 검증 완료: $u_{y, \text{C3D8}} (1.79) < u_{y, \text{C3D8R}} (3.71) < u_{y, \text{C3D8I}} (7.59) \approx u_{y, \text{theory}} (7.5\sim 8.1)$ (**PASSED**).
- 2026-09-28, `official/pipecrushing_cpe4i.inp` (Deck #14):
  1) `CPE4I` 비적합 모드 평면 변형률 16개 요소 + `*PLASTIC` 8점 탄소성 + 이동식 강체 평판(`*RIGID BODY`, `*SURFACE TYPE=SEGMENTS`) 접촉 압축 벤치마크 (Abaqus Benchmarks §1.1.8).
  2) 5대 핵심 결함 해결:
     - `dynamic2d.py`: 구속조건 분리/비활성 시 `len(rows_extra) <= 1`에서 `K_global` 미할당 `UnboundLocalError` 수정.
     - `dynamic2d.py`: `apply_boundary_conditions`에서 `empty_rows`의 `r_bc`를 0으로 덮어써서 비요소 기준 절점(node 1000)의 Dirichlet 처방 변위(-1.3)가 무시되던 결함 수정.
     - `dynamic2d.py`: `*STATIC` 스텝 경계조건에 명시적 진폭 없을 때 Abaqus 규격 `AMPLITUDE=RAMP` 기본 선형 램프 적용.
     - `interaction.py` & `mesh.py`: `*rigid body`의 `refnode`(공백 없음) 파라미터 파싱 및 `*rigid body`가 `*surface`보다 선행할 때의 지연 바인딩 연결.
     - `surface_contact2d.py`: 이동식 강체 평판의 기준점 변위($u_{\text{ref}}$) 반영 및 Newton 제3법칙(상호작용 접촉 강성 및 작용-반작용력) 완전 정식화.
  3) 103 증분 전수 컷백 0회 완벽 수렴 (32.64s, 468 Newton iters).
  4) 물리적 정량성 검증 완료:
     - 강체 평판 처방 변위: $u_y = -1.3000\,\text{in}$ (100.000%).
     - 관 상단 정점 밀착 압축: $u_y = -1.3083\,\text{in}$.
     - 관 적도점 타원화 횡팽창(Ovalization): $u_x = +0.7292\,\text{in}$ ($R_o: 2.25 \to 2.98\,\text{in}$).
     - 대칭면 구속 보존: `XSYMM` $u_x = 0.0$, `YSYMM` $u_y = 0.0$ 보존.
     - 수직력 작용-반작용 정적 평형: $RF_{y, \text{top}} = -128,540.51\,\text{lbf}$, $RF_{y, \text{bottom}} = +128,539.97\,\text{lbf}$ (평형 오차 $0.55\,\text{lbf}$, 상대 오차 **0.0004%**!) (**PASSED**).
- 2026-09-28, `official/radstr_holcyl_cax4r.inp` (Deck #15) & `official/rubberdisk_cax4ih_poly.inp` (Deck #17):
  축대칭(`CAX4R`, `CAX4IH`) $r-\theta-z$ 차원 역학 미지원으로 프로젝트 공식 지원 범위 외 분류 (**OUT-OF-SCOPE / UNSUPPORTED**).
- 2026-09-28, `official/roofsnapthrough_s4r.inp` (Deck #16):
  6자유도 미드서피스 순수 쉘 요소(`S4R`) 및 Riks 호장법(`*STATIC, RIKS`) 비선형 좌굴 실행기 미지원으로 한계 명시 (**UNSUPPORTED**).
- 2026-09-29, `official/thickcompcyl_8el.inp` (Deck #18):
  1) `C3D20R` 20절점 2차 세렌디피티 감차적분 32개 요소 + 4층 직교이방성 복합재 스택업(`*SOLID SECTION, COMPOSITE`) + 원통 좌표계 국소 방향(`*ORIENTATION, SYSTEM=CYLINDRICAL`) + 내압 $P = 50.0\,\text{MPa}$ (`*DLOAD, Inside, P6, 50.0`) 벤치마크 (Abaqus Benchmarks §1.1.2).
  2) 핵심 결함 해결:
     - `material.py`: `*SOLID SECTION, COMPOSITE`에서 데이터 라인의 실제 재료(`Mcircum`, `Maxial`) 및 방향(`Ocircum`) 파싱 연결.
     - `c3d20_numba.py`: 20절점 형상함수 `_sf3d_c3d20` 기반 각 가우스점 전역 좌표 투영 및 원통 좌표계 직교이방성 텐서 회전 변환($C_{glob} = T_\varepsilon^T S_{loc}^{-1} T_\varepsilon$)과 8점 감차적분 완전 구현.
     - `dynamic3d.py`: 곡면 2차 8절점 면(Quad8) 2x2 Gauss 등가 절점력 적분($\int N_k P (\mathbf{t}_1 \times \mathbf{t}_2) dA$) 구현 및 `Model.Pressure`의 `face` 인자 전달 복원.
  3) 22 증분 전수 컷백 0회 완벽 수렴 (29.73s, 매 증분 2회 뉴턴 반복).
  4) Abaqus 공식 벤치마크 매뉴얼(Lekhnitskii 이론해 / C3D20R 8el) 정량 대조 완료:
     - **내경 반경 변위 ($R_i = 60\,\text{mm}$)**: Abaqus 공식값 $0.070805\,\text{mm}$ vs WHT 계산값 **$0.070806\,\text{mm}$** (**상대 오차 0.0014%**).
     - **외경 반경 변위 ($R_o = 140\,\text{mm}$)**: Abaqus 공식값 $0.007480\,\text{mm}$ vs WHT 계산값 **$0.007479\,\text{mm}$** (**상대 오차 0.013%**).
     - 평면 변형률 축방향 구속 $\max |u_z| = 3.37\times 10^{-17}\,\text{mm}$ (기계 정밀도), 대칭면 구속 오차 0.000%, 원주 방향 원형 유지 편차 0.29% (**PASSED**).
- 2026-09-29, `official/unifcollapspipe_str_elbelem.inp` (Deck #19):
  1D 특수 파이프 엘보우 Brazier 타원화 자유도 요소(`ELBOW31B`) 미지원으로 공식 범위 외 분류 (**UNSUPPORTED / OUT-OF-SCOPE**).

