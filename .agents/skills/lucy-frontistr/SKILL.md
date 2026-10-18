---
name: lucy-frontistr
description: FrontISTR 오픈소스 대규모 병렬 유한요소(FEM) 구조/열/동적/모드/비선형(재료비선형, 대변형, 접촉) 해석 실행, 입력 파일(.dat, .msh, .cnt) 작성, AI LLM 지원 워크플로우(frontistr-fabrica), 소스코드(FrontISTR-master) 분석/확장, 한국어 전체 매뉴얼(docs_ko) 조회 및 SMP/MPI 실행을 통합 지원하는 Antigravity·Claude 겸용 전역 스킬.
---

# 🛠️ lucy-frontistr: FrontISTR 통합 해석·개발·AI Fabrica 스킬 (Antigravity & Claude 공용)

`lucy-frontistr`는 오픈 소스 대규모 병렬 유한요소법(FEM) 구조 해석 소프트웨어인 **FrontISTR (v6.0)**의 **해석 실행(SMP/MPI)**, **입력 파일(`.dat`, `.msh`, `.cnt`) 작성**, **AI LLM 워크플로우(`frontistr-fabrica-main`)**, **소스코드(`FrontISTR-master`) 분석/수정**, 그리고 **한국어 공식 매뉴얼 전체 크롤링 아카이브(`docs_ko`) 참조**를 통합 지원하는 스킬입니다.

---

## 📂 1. 접근 권한 부여 폴더 및 핵심 경로 (Authorized Directories)

아래의 모든 폴더 및 실행 파일은 Antigravity와 Claude Code 모두에서 읽기/쓰기/실행 권한이 부여되어 있습니다:

| 구분 | 절대 경로 | 주요 내용 및 역할 |
|:---|:---|:---|
| **FrontISTR 루트** | `D:\PythonCodeStudy\FrontISTR` | FrontISTR 전체 환경 루트 폴더 |
| **한국어 공식 설명서 (크롤링 MD)** | `D:\PythonCodeStudy\FrontISTR\docs_ko` | https://docs.frontistr.com/ko/ 전체 205개 페이지 크롤링 `.md` 및 `consolidated/` 8대 통합 매뉴얼 |
| **AI LLM 지원 (Fabrica)** | `D:\PythonCodeStudy\FrontISTR\frontistr-fabrica-main` | AI 에이전트용 프롬프트(`prompts/`), 리포트 템플릿(`reports/`), 가이드(`guide_*.md`), 환경 프로파일(`fistr_env.md`), 도구(`tools/`) |
| **MPI 하이브리드 실행 파일** | `D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.hybrid_msmpi_openblas` | MS-MPI + OpenMP + OpenBLAS + MUMPS + METIS + ML 빌드 바이너리 (`fistr1.exe`, `hecmw_part1.exe`, `mpiexec.exe`, `msmpi.dll`) |
| **SMP 멀티스레드 실행 파일** | `D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.thread_openblas` | OpenMP + OpenBLAS + MUMPS + METIS + ML 빌드 바이너리 (`fistr1.exe`, `hecmw_part1.exe`, `hecmw_vis1.exe`, `neu2fstr.exe`) |
| **FrontISTR 마스터 소스코드** | `D:\PythonCodeStudy\FrontISTR\FrontISTR-master` | 솔버 본체(`fistr1/`), 미들웨어(`hecmw1/`), 회귀 테스트(`tests/`), 20개 튜토리얼(`tutorial/`) |

---

## 🚀 2. FrontISTR 실행 방법 (Windows PowerShell 7 기준)

### 2.1 SMP (단일 프로세스 / OpenMP 멀티스레드) 실행
영역 분할 없이 입력 파일(`hecmw_ctrl.dat`, `.msh`, `.cnt`)이 위치한 작업 디렉토리에서 즉시 실행합니다.

```powershell
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$FISTR_SMP = "D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.thread_openblas\fistr1.exe"

# 작업 디렉토리에서 OpenMP 4스레드로 실행 (초회 실행 시 반드시 로그 파일로 리다이렉트)
Push-Location "<작업_디렉토리>"
& $FISTR_SMP -t 4 > out_fistr.log 2>&1
Pop-Location
```

### 2.2 MPI 병렬 및 Hybrid (MPI × OpenMP) 실행
사전에 `hecmw_part_ctrl.dat`를 작성하고 `hecmw_part1.exe`로 메시 영역 분할을 수행한 뒤, 동봉된 `mpiexec.exe`로 병렬 실행합니다.

```powershell
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$MPI_DIR = "D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.hybrid_msmpi_openblas"
$PART_BIN = "$MPI_DIR\hecmw_part1.exe"
$MPIEXEC  = "$MPI_DIR\mpiexec.exe"
$FISTR_MPI = "$MPI_DIR\fistr1.exe"

Push-Location "<작업_디렉토리>"
# 1단계: 메시 영역 분할 (hecmw_part_ctrl.dat 참조)
& $PART_BIN > out_part.log 2>&1

# 2단계: MPI 4프로세스 × OpenMP 2스레드 하이브리드 병렬 실행
& $MPIEXEC -np 4 $FISTR_MPI -t 2 > out_fistr.log 2>&1
Pop-Location
```

### 2.3 명령줄 옵션 및 정상 종료(성공) 판정 규칙
- 주요 옵션:
  - `-h`: 도움말 표시
  - `-v`: 버전 및 빌드 정보(`MPI`, `OpenMP`, `--with-mumps`, `--with-metis`, `--with-ml` 등) 표시
  - `-t <n>`: OpenMP 스레드 수 지정
  - `-c <path>`: 전체 제어 파일 지정 (기본값: `./hecmw_ctrl.dat`)
  - `--debug`: 로그 레벨을 DEBUG로 설정
- **[필수] 성공 판정 기준**:
  - `fistr1.exe`는 에러가 발생해도 종료 코드(`$LASTEXITCODE`)가 `0`으로 끝나는 경우가 존재합니다.
  - 따라서 종료 코드만으로 성공을 단정하지 말고, 반드시 표준 출력(`out_fistr.log`)의 마지막에 **`FrontISTR Completed !!`** 문구가 출력되었는지 확인해야 합니다.

---

## 📋 3. 입출력 파일 구조 및 핵심 키워드 요약

### 3.1 입력 파일 4종
| 파일명 | 역할 | 한국어 매뉴얼 참조 경로 (`D:\PythonCodeStudy\FrontISTR\docs_ko\`) |
|:---|:---|:---|
| `hecmw_ctrl.dat` | 전체 제어 (메시·제어·결과·시각화 파일 연결) | `keyword_reference/dat/index.md` (`consolidated/03_keyword_reference_dat_msh_part.md`) |
| `<model>.msh` | 메시 데이터 (절점 `!NODE`, 요소 `!ELEMENT`, 그룹 `!NGROUP`/`!EGROUP`/`!SGROUP`, 단면 `!SECTION`) | `keyword_reference/msh/index.md` (`consolidated/03_keyword_reference_dat_msh_part.md`) |
| `<model>.cnt` | 해석 제어 데이터 (해석종류 `!SOLUTION`, 경계/하중, 재료물성, 접촉, 스텝, 솔버, 출력) | `keyword_reference/cnt/index.md` (`consolidated/04_keyword_reference_cnt.md`) |
| `hecmw_part_ctrl.dat` | 영역 분할 제어 (MPI 병렬 실행 시만 사용, `!PARTITION`) | `keyword_reference/part/index.md` (`consolidated/03_keyword_reference_dat_msh_part.md`) |

### 3.2 출력 파일 4종
| 출력 파일 | 내용 및 확인 방법 |
|:---|:---|
| `0.log` (병렬: `0.log`~`n.log`) | 물리량(변위 `//U1~U3`, 미제스 응력 `//SMS` 등)의 최대·최소 요약 (`Global Summary @Node`, `Global Summary @Element`) |
| `FSTR.msg`, `FSTR.sta` | 해석 진행 메시지(`FSTR.msg`) 및 스텝/서브스텝/뉴턴 반복 수렴 이력(`FSTR.sta`) |
| `<header>.res.<rank>.<step>` | 수치 결과 파일 (`!WRITE,RESULT` 지정 시 출력) |
| `<header>_vis_psf.*.pvtu` / `.vtu` | ParaView 시각화 데이터 (`!WRITE,VISUAL` 및 `!output_type = BIN_VTK` 지정 시 출력) |

### 3.3 주요 요소 타입 및 정식화 선택 (`functions_reference/02_element.md`)
- **솔리드 요소**:
  - `341` (4절점 1차 사면체): `!SECTION`에서 `FORM341=FI` (기본) 또는 `FORM341=SELECTIVE_ESNS` (전단/체적 잠김 완화 평활화 FEM, MPI 병렬 시 `hecmw_part_ctrl.dat`에 `DEPTH=2` 필수)
  - `342` (10절점 2차 사면체)
  - `361` (8절점 1차 육면체): `!SECTION`에서 `FORM361=IC` (소변형 기본, 비적합 굽힘 보정), `FBAR` (대변형 기본, 체적 잠김 보정), `BBAR` (소변형 체적 보정), `UP` (비압축성 u-p 혼합 정식화), `FI` (완전 적분)
  - `362` (20절점 2차 육면체)
- **구조 요소 및 자유도 혼재**:
  - 보 요소 `611` (`FORM611=EULER` / `TIMOSHENKO`), 셸 요소 `731` (MITC3), `741` (MITC4), `743` (MITC9) — 6자유도 절점
  - 솔리드 요소(3자유도)와 직접 공유 절점으로 연결할 때는 자유도 혼재용 요소 `641` (보), `761` (MITC3), `781` (MITC4) 사용 (`tutorial/20_shell_solid_mix` 참조)

---

## 🤖 4. AI LLM 지원 워크플로우 (`frontistr-fabrica-main`)

`D:\PythonCodeStudy\FrontISTR\frontistr-fabrica-main`은 AI 에이전트가 FrontISTR의 해석·개발·디버깅·문서화를 체계적으로 수행하기 위한 **리포트 구동형(Report-Driven) 공방(Fabrica)**입니다.

### 4.1 핵심 참조 문서
1. `README.md` & `AGENTS.md`: 전체 폴더 지도 및 상시 준수 규칙
2. `prompt_dispatch.md`: 사용자의 자연어 요청을 12개 전문 프롬프트 템플릿(`prompts/*.md`)으로 매핑하는 디스패처
3. `guide_run_frontistr.md`: 실행 방법, 입출력 파일, `ctest` 회귀 테스트(`serial`, `openmp`, `mpi`, `hybrid`), 참조해 생성(`create_reference.sh`) 가이드
4. `guide_code_frontistr.md`: `fistr1/` 및 `hecmw1/` 소스 코드 아키텍처 및 병렬/소유권 코딩 규칙
5. `guide_feedback.md`: 작업 리포트의 피드백 3요소(무엇이 일어났는가 / 왜인가 / 어떻게 해야 하는가) 기록 규칙
6. `fistr_env.md`: 현재 Windows 워크스테이션의 SMP/MPI 실행 파일 경로가 기입된 검증 완료 환경 프로파일

### 4.2 태스크별 프롬프트(`prompts/`) 및 리포트(`reports/`) 매핑
| 작업 유형 | 프롬프트 템플릿 | 리포트 템플릿 | 주요 용도 |
|:---|:---|:---|:---|
| **해석 실행** | `prompts/run.md` | `reports/run.md` | 메시 확인, `.cnt`/`hecmw_ctrl.dat` 작성, 솔버 실행, 결과/시각화 정리 |
| **기능 추가** | `prompts/enhance.md` | `reports/enhance.md` | 신규 키워드/재료/요소/알고리즘 설계, 구현, 단위/타당성 테스트 |
| **버그 수정·디버깅** | `prompts/debug.md` | `reports/debug.md` | 오류/발산 재현, 원인 가설 수립 및 검증, 회귀 테스트 추가 |
| **코드·문서 조사** | `prompts/investigate.md` | — | 코드 변경 없이 수식·구현 위치·호출 흐름·알고리즘 분석 |
| **도구 제작** | `prompts/maketool.md` | `reports/maketool.md` | 전처리/후처리 자동화 파이썬 스크립트 제작 |
| **문서 작성** | `prompts/document.md` | — | 매뉴얼 집필 및 코드 정합성 검증 |
| **코드 리뷰** | `prompts/review.md` | `reports/review.md` | 브랜치/MR 차이점 분석, 수식/구현 검증, 리포트 검증 |
| **환경 구축** | `prompts/build.md` | `fistr_env_template.md` | 빌드 설정 및 `fistr_env.md` 생성 |

### 4.3 Fabrica 핵심 작업 수칙 (`prompts/_common.md`)
- 모든 산출물(작성된 프롬프트, 작업 리포트, 해석 실행 서브폴더 `run/`, 임시 폴더 `tmp/`)은 `work_<YYYYMMDD>_<짧은_영문_슬러그>/` 폴더 아래에 통합 관리합니다.
- 해석 조건(하중, 구속 방향, 재료 물성치 등)에서 미확정 사항이 있으면 추측으로 결정하지 않고 사용자에게 먼저 확인합니다.
- 리포트 검증 시 `python "D:\PythonCodeStudy\FrontISTR\frontistr-fabrica-main\tools\verify_report.py" --brief <리포트_경로>`를 활용할 수 있습니다.

---

## 🧬 5. 소스코드(`FrontISTR-master`) 구조 및 개발 가이드

경로: `D:\PythonCodeStudy\FrontISTR\FrontISTR-master`

### 5.1 디렉토리 아키텍처
- `fistr1/src/` (솔버 본체, Fortran 90):
  - `main/`: 엔트리 포인트
  - `common/`: 제어 파일 파서(`fstr_ctrl_*.f90`), 셋업(`fstr_setup.f90`)
  - `analysis/static/`, `analysis/dynamic/`, `analysis/heat/`, `analysis/interaction/` (접촉 `fstr_contact*.f90`), `analysis/precheck/`
  - `lib/`: 요소 라이브러리(`static_LIB_*.f90` — `3dIC`, `C3D8_Bbar`, `Fbar`, `UP`, `C3D4_selectiveESNS`, `shell_mitc4` 등), 데이터 구조(`m_fstr.F90`), 출력(`m_out.f90`)
  - `lib/physics/`: 구성칙(탄성, 탄소성, 초탄성, 점탄성, 크리프)
  - `lib/contact/`: 접촉 정의 및 탐색(`fstr_contact_def.F90`, `bucket_search.f90`)
  - `lib/user/`: 사용자 재료/하중 서브루틴 인터페이스 (`umat.f`, `uyield.f90`, `uload.f`, `uelastic.f90`)
- `hecmw1/src/` (미들웨어 기반, C & Fortran):
  - `common/`: 메시 데이터 구조, I/O(HEC, Abaqus `.inp`, Nastran), MPI 통신, 리스타트
  - `solver/`: 선형 솔버(반복법 `CG`, `BiCGSTAB`, `GMRES`, 직접법 `MUMPS`, `MKL PARDISO`, `DIRECTmf`, 전처리 `AMG/SAAMG`, `MPC`)
  - `visualizer/`: VTK 시각화 출력 (`hecmw_fstr_output_vtk.c`)
- `tutorial/`: `01_elastic_hinge` ~ `20_shell_solid_mix` (즉시 실행 가능한 20개 표준 예제)
- `tests/`: `analysis/`, `lib/`, `solver/`, `with_mkl/`, `with_mumps/`, `with_ml/` 회귀 테스트 데이터셋

### 5.2 `fistr1` × `hecmw1` 병렬 코딩 철칙 (`guide_code_frontistr.md`)
- `fistr1` 내부에 MPI 함수를 직접 호출하지 않고, 통신은 반드시 `hecmw1` 함수를 호출합니다:
  - 절점값 랭크 간 합산: `call hecmw_assemble_R(hecMESH, v, n, ndof)`
  - 중첩 영역(Sleeve) 최신화: `call hecmw_update_R(hecMESH, v, n, ndof)`
  - 전 랭크 스칼라 집약: `call hecmw_allreduce_R1(hecMESH, v, hecmw_max)`
- 소유권 판정은 새로운 래퍼 함수를 만들지 말고 기존 정형을 인라인으로 사용합니다:
  - 내부 절점 루프(이중 계산 방지): `do i = 1, hecMESH%nn_internal`
  - 자기 랭크 소유 요소 판정: `ID_area = hecMESH%elem_ID(i*2)` 후 `if( ID_area == hecMESH%my_rank )`

---

## 📚 6. 한국어 공식 매뉴얼 (`docs_ko`) 및 내장 레퍼런스 활용

`https://docs.frontistr.com/ko/` 전체(205개 문서)가 크롤링되어 다음 두 곳에 정리되어 있습니다:
1. **카테고리별 8대 통합 문서** (`D:\PythonCodeStudy\FrontISTR\docs_ko\consolidated\` 및 본 스킬 폴더 내 `references/`):
   - `00_introduction_install_build.md`: 소개, 주요 기능, AI 연계, 설치 및 CMake/GPU 빌드 가이드
   - `01_execution_guide.md`: 해석 흐름, 직렬/병렬 실행 절차, 데이터 변환(`neu2fstr`, `abaqus2fstr`), 튜토리얼 01 상세
   - `02_functions_reference.md`: 해석 종류, 요소 라이브러리, 재료 모델, 경계/하중, MPC, 접촉, 솔버, 스텝, 출력/리스타트, 영역 분할, 오류 메시지
   - `03_keyword_reference_dat_msh_part.md`: 입력 규칙, `hecmw_ctrl.dat`, `.msh` 키워드 전체, `hecmw_part_ctrl.dat`
   - `04_keyword_reference_cnt.md`: `.cnt` 해석 제어 키워드 81개 전체 사전 (`!BOUNDARY`, `!CONTACT`, `!ELASTIC`, `!HYPERELASTIC`, `!PLASTIC`, `!SOLVER`, `!STEP`, `!WRITE` 등)
   - `05_theory_reference.md`: 선형/비선형 정적, 동적, 고유치, 열전도, 구성칙, 접촉(Augmented/Standard Lagrangian), 선형 솔버, 병렬화 이론 수식
   - `06_examples_showcase_vv.md`: 예제집(`tutorial_01`~`17`), 응용 사례집, V&V 검증 결과
   - `07_coding_guide_and_release_notes.md`: 요소 번호 체계, 사용자 서브루틴(`umat`, `uyield`, `uload`, `uelastic`), 최신 릴리즈 노트(v6.0)
2. **전체 문서 색인**: `D:\PythonCodeStudy\FrontISTR\docs_ko\README.md`

---

## ⚡ 7. 내장 헬퍼 스크립트 (`scripts/fistr_helper.py`) 사용법

스킬에 동봉된 `scripts/fistr_helper.py`를 통해 **문서 검색**, **SMP/MPI 솔버 실행 및 완료 검증**, **`0.log` 요약 추출**, **튜토리얼 즉시 테스트**를 수행할 수 있습니다.

```powershell
$HELPER = "C:\Users\GOODMAN\.gemini\config\skills\lucy-frontistr\scripts\fistr_helper.py"

# 1. 한국어 매뉴얼 키워드 검색 (205개 문서 대상)
python $HELPER --search "!CONTACT_ALGO"
python $HELPER --search "FORM361"

# 2. SMP 모드로 작업 폴더 내 FrontISTR 해석 실행 및 결과 요약 자동 출력
python $HELPER --run "<작업_디렉토리>" --mode smp --threads 4

# 3. MPI 하이브리드 모드로 영역 분할(hecmw_part1) + 병렬 실행(mpiexec) + 결과 요약
python $HELPER --run "<작업_디렉토리>" --mode mpi --np 4 --threads 2

# 4. 기존 실행 폴더의 0.log 및 FSTR.sta 결과 요약 파싱
python $HELPER --summary "<작업_디렉토리>"
```

---

## 🧠 8. FrontISTR 비선형 대변형(`NLSTATIC`) 해석 실전 노하우 (소스코드 검증 완료)

1. **2D 평면변형률(Plane Strain) 대변형 해석 시 `TYPE=361` 1층 돌출 필수**:
   - `fstr_CreateMatrix.f90` (L371) 및 `static_LIB_2d.f90` (`UPDATE_C2`)에서 FrontISTR의 2D 평면 요소(`231, 232, 241, 242`)는 미소 변형(`INFINITESIMAL`)만 지원하며 기하 비선형 진입 시 중단(`stop`)됩니다.
   - 2D 평면변형률 대변형(예: 180° 폴딩) 해석은 반드시 2D Quad4 격자를 $Z$ 방향으로 1개 층 돌출시킨 `TYPE=361` 육면체로 모델링하고 모든 절점에 $u_z = 0$ 구속(`!BOUNDARY` `Z_FIXED, 3, 3, 0.0`)을 부여해야 합니다.
2. **유한 회전 경계조건 (`!BOUNDARY, ROT_CENTER=<NGRP>`)**:
   - `fstr_AddBC.f90`의 `get_rot_displacement()`는 `ROT_CENTER`로 지정된 피벗 절점의 좌표를 기준으로 Rodrigues 유한 회전 행렬을 계산하여 종속 절점군에 정확한 원호 궤적 변위 증분을 부과합니다.
   - 본체와 분리된 고정 참조 요소(`EL_PIVOT_REF`, 전 절점 1~3 구속)의 절점을 피벗 그룹(`PIVOT_L`, `PIVOT_R`)으로 지정하고, 종속 절점군에 병진 자유도(`1, 2, 0.0`)와 회전 자유도(`3, 3, <rad>`)를 함께 선언합니다.
3. **대변형 전단($\gamma > 100\%$)을 겪는 초연성 점착제(PSA)는 `!HYPERELASTIC, TYPE=NEOHOOKE` 필수**:
   - Total Lagrangian `!ELASTIC` (Saint-Venant Kirchhoff) 사용 시 대변형 단순 전단에서 Green-Lagrange 수직 변형률 $E_{22} = \frac{1}{2}\gamma^2$로 인한 비물리적 Poynting 압축 붕괴가 발생합니다.
   - 등적 불변량 $\bar{I}_1 = J^{-2/3}I_1$을 사용하는 `!HYPERELASTIC, TYPE=NEOHOOKE` ($C_{10} = \frac{\mu}{2}$, $D_1 = \frac{2}{K}$)를 적용하면 $200\%$ 이상의 전단에서도 붕괴 없이 수렴합니다.
4. **Total Lagrangian 2차 뉴턴 수렴을 위한 `FORM361=FI` / `BBAR` 선택**:
   - `static_LIB_3dIC.f90` (`FORM361=IC`)는 내부 자유도 응축 시 이전 반복 잔차 `-qf_a`가 생략되어 있고, `static_LIB_Fbar.f90` (`FORM361=FBAR`)는 `TOTALLAG` 초기응력 스케일링 계수 2가 생략되어 대변형 시 선형 수렴으로 저하됩니다.
   - 대변형 `TOTALLAG` 해석에서 빠른 2차 수렴(서브스텝당 3~4회 반복)이 필요할 때는 일관 접선 강성이 정확히 구현된 `!SECTION, SECNUM=<n>, FORM361=FI` (또는 `BBAR`)와 `!SOLVER, METHOD=MUMPS` 조합을 권장합니다.

