아래의 내용을 스킬화한다.
skill 명은 lucy-frontistr 이다.
antigravity, claude 모두 사용할 수 있도록 한다.
FrontISTR을 사용하기 위한 안내이다.

# FrontISTR
아래의 폴더들은 접근 권한을 부여
FrontISTR는 오픈 소스 대규모 병렬 유한요소법(FEM) 구조 해석 소프트웨어입니다. 재료 비선형, 대변형, 접촉을 포함한 비선형 변형 및 응력 해석뿐만 아니라 모드 해석, 주파수 응답 해석, 열전도 해석 등 산업 및 학술 실무에 필요한 해석 기능을 제공합니다. 

## 설명서
https://docs.frontistr.com/ko/
이 페이지 전체를 크롤링하여 md 파일로 정리하여 참조하여도 좋겠다.

## AI LLM 지원 
readme.md 뿐만 아니라 각종 md 파일을 참고하여 작업한다.
D:\PythonCodeStudy\FrontISTR\frontistr-fabrica-main
주로 여기 내용을 잘 파악하면 전반적인 사용이 가능할 것

## MPI 실행 파일
D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.hybrid_msmpi_openblas

## SMP 실행 파일
D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.thread_openblas

## 소스코드
D:\PythonCodeStudy\FrontISTR\FrontISTR-master



###

  현재 세션(Conversation) ID는 다음과 같습니다:

  0a537f5d-4984-46aa-9170-20a9979342e3

  • 아티팩트 및 세션 로그 경로: C:\Users\GOODMAN\.gemini\antigravity-cli\brain\0a537f5d-4984-46aa-9170-20a9979342e3




  현재 세션(Conversation) ID는 다음과 같습니다:

    6e0be6af-fd3f-4b54-b349-ef02946cf0c2

  • 아티팩트 디렉터리: 6e0be6af-fd3f-4b54-b349-ef02946cf0c2
  • Handoff 문서: handoff_20261019.md / handoff_20261019.md





### 

# 너는 FEM S/W 전문가, 수치해석 전문가, 수학-공학자

# 목적
- FISTR의 해석 시간, 성능을 확인 하기 위한 테스트

# 할일
- 2D folding 테스트
  - 요소 크기를 50% 낮춰가면서 4번 분할
  - 100%, 50%, 25%, 12.5%, 6.25%
  - 각 요소 크기에 따른 해석 속도 기록 및 정리
  - 해속 속도 및 하중-응력-변형률 비교
- 3D folding  테스트
  - 요소 크기를 100%, 50%, 25% 비교, 해석 속도 및 하중-응력-변형률 비교
  - 요소 크기 100% 기준으로 깊이 방향으로 요소수 1개, 10개, 20개 속도 비교
  - 요소 크기 50% 기준으로 깊이 방향으로 요소수 1개, 10개, 20개 속도 비교
- 동일 요소 크기 조건으로 wht 솔버와 비교 (속도, 하중, 응력, 변현율, 형상 등)
  
# 하지말 것
- 할일 이외의 것을 하고 싶으면 제안부터하고 자율 진행 X


## 1. 연구 목적 (Objective)
14-Layer Full-Stack (`teardrop` preset, $630\,\mu\text{m}$) 모델의 가이드 플레이트 20° 회전 상태($\theta = 20^\circ$)에서
FrontISTR(MUMPS 직접 솔버, 공인 Ground Truth)과 WHT Solver(Intel oneMKL PARDISO)의 수치해석 결과를
1:1 정밀 대조하여, WHT Solver의 기구학 커널, 재료 구성방정식, 내력 적분 및 모멘트 계산 루틴의
수치적 오류를 격리·진단하고 정확도 개선 방안을 도출한다.

## 2. 기하학 및 해석 조건 (Kinematics & Boundary Conditions)
- 모델: 14-Layer Full-Stack (7 PSA + 6 PET + 1 UTG Glass + Hinge Tape Void)
- 하중 조건: 좌/우 가이드 플레이트 각 $\theta = \pm 20.0^\circ$ 회전 (Smoothstep 가속, 5~10 substeps)
- 메쉬 기준: 2D Plane Strain (단위 폭 $1.0\,\text{mm}$) 및 3D ($W_z = 1.0\,\text{mm}, 10.0\,\text{mm}$)
- 재료 물성 동등화 (Parity Alignment):
  * PET (6개 층): Linear Elastic, $E = 4,000\,\text{MPa}$, $\nu = 0.30$
  * PSA (7개 층): Hyperelastic Neo-Hookean, $C_{10} = 8.3389\times 10^{-3}\,\text{MPa}$, $D_1 = 0.24\,\text{MPa}^{-1}$
  * UTG Glass (최상단): Linear Elastic, $E = 70,000\,\text{MPa}$, $\nu = 0.22$
  * Guide Plate (좌우 구동단): Linear Elastic, $E = 200,000\,\text{MPa}$, $\nu = 0.30$

## 3. 물리량 정규화 표준 (Unit Depth / Width Normalization)
2D 평면변형률 및 3D 모델의 폭($W_z$) 차이에 무관하게 직접 비교할 수 있도록
모멘트와 반력을 단위 깊이 길이(Unit Depth)로 정규화한다:
- 단위 폭 모멘트 밀도: $\bar{M}_z = \frac{M_z}{W_z} \quad [\text{N}\cdot\text{mm} / \text{mm} = \text{N}]$
- 단위 폭 수평 반력 밀도: $\bar{F}_x = \frac{F_x}{W_z} \quad [\text{N} / \text{mm}]$
- 단위 폭 수직 반력 밀도: $\bar{F}_y = \frac{F_y}{W_z} \quad [\text{N} / \text{mm}]$

## 4. 20° 시점 4대 핵심 정량 비교 지표 (4 Verification Pillars)
1. 구동 모멘트 및 반력 밀도 ($\bar{M}_z, \bar{F}_x, \bar{F}_y$):
   - 좌/우 구동단 모멘트의 크기 및 작용-반작용 대칭성 오차 ($|\bar{M}_L - \bar{M}_R| / \bar{M}_{\text{avg}} \times 100\%$)
   - FrontISTR 기준 WHT Solver 모멘트 상대 오차 ($\epsilon_M = \frac{\bar{M}_{\text{WHT}} - \bar{M}_{\text{FISTR}}}{\bar{M}_{\text{FISTR}}} \times 100\%$)
2. 층별 응력장 (Stress Field):
   - 최외곽 PET 인장/압축 주응력 $\sigma_{11}$ 및 Von Mises 응력 $\sigma_{\text{vM}}$
   - PSA 점착층 전단응력 $\tau_{12}$ 및 정수압 $p = -\frac{1}{3}\operatorname{tr}(\boldsymbol{\sigma})$
3. 층별 변형률장 (Strain Field):
   - PET 굽힘 변형률 $\varepsilon_{11}$ 및 PSA 대변형 전단변형률 $\gamma_{12}$
   - 힌지 영역 층간 미끄러짐(Slip) 변위 분포
4. 변형 형상 프로파일 (Deformed Silhouette):
   - 힌지 루프 곡률 반경 및 최하단 처짐/융기 변위 $u_y(x)$ 일치도

## 5. WHT Solver 오류 진단 및 개선 분석 체크리스트 (Root-Cause Diagnosis)
- [ ] 진단 1 (재료 모델): PET의 항복 발생 여부 (탄소성 모델 활성화로 인한 조기 항복 강성 저하 확인)
- [ ] 진단 2 (내력/모멘트 적분계): `compute_section_reactions`의 모멘트 팔 $\mathbf{r} = \mathbf{x} - \mathbf{x}_c$가 변형된 현재 좌표계 $\mathbf{x} = \mathbf{X} + \mathbf{u}$를 정확히
반영하는지 점검
- [ ] 진단 3 (요소 기구학): Incompatible Modes(`C3D8I`/`CPE4I`) vs 1점 감차적분(`C3D8R`) vs B-bar(`C3D8`) 간의 전단/체적 잠김 여부 판정
- [ ] 진단 4 (경계조건 구속): 회전 가이드 플레이트 구속 DOF 매핑 및 피벗 회전 변위 전달의 기구학적 무결성 확인
──────


### 1. 작성된 영구 문서 링크

• 상세 기술 계획서: fistr_element_strategy_and_conforming_benchmark_plan_20261020.md
    • 6대 FrontISTR 접근법 비교 총괄표 및 최종안 선정 근거
    • 단위 깊이(nz = 1) 3D 테스트 모델의 기구학 및 절점 일치 토폴로지
    • 정량적 검증 목표 (이론 정해 모멘트 0.2317 N · mm 오차 < 5% 판정)
• 누적 교훈 문서 (SSOT): LESSONS_LEARNED.md:720-730 (Section 16.18 신설 반영)
──────
### 2. FrontISTR 테스트 작업 요청 프롬프트 (Prompt for Implementation)

아래 프롬프트를 사용하여 해당 벤치마크 러너 구현 작업을 바로 지시하실 수 있습니다:

  FrontISTR(fistr1.exe) 환경에서 박막 적층 굽힘의 전단 잠김과 체적 잠김을 동시에 극복하는
  "단위 깊이(nz=1) 2차-1차 적합 메쉬(Conforming Mesh) 벤치마크" 스크립트를 구현하라.

  ### 1. 대상 모델 및 레이어 스택업 (총 두께 130 um)
  - 모델 길이 L = 5.0 mm, 단위 폭 W = 1.0 mm (Z in [0, 1])
  - Top Layer: PET (t = 0.05 mm, E = 4000 MPa, nu = 0.3), FrontISTR 요소 `362` (20절점 2차 육면체, 1개 층)
  - Mid Layer: PSA (t = 0.03 mm, E = 2.0 MPa, nu = 0.499), FrontISTR 요소 `361` with `FORM361=UP` (1차 u-P, 1개 층)
  - Bottom Layer: PET (t = 0.05 mm, E = 4000 MPa, nu = 0.3), FrontISTR 요소 `362` (20절점 2차 육면체, 1개 층)

  ### 2. 메쉬 적합 세분화(Conforming Subdivision) 구현 규칙 (Hanging Node 0개 원칙)
  - PET 층(2차 요소)은 X방향 길이 분할수 Nx_pet = 20, Z방향 폭 분할수 Nz_pet = 1 (2차이므로 면당 8개 절점 생성)
  - PSA 층(1차 요소)은 X방향 분할수 Nx_psa = 40 (2배), Z방향 폭 분할수 Nz_psa = 2 (2배)로 분할
  - PET 아랫면의 모서리 중간 노드(Mid-side node)와 코너 노드가 PSA 윗면의 1차 요소 코너 노드와
    정확히 동일한 (X, Z) 좌표를 갖도록 절점 번호(Node ID)를 1:1로 공유(Shared Nodes)할 것.
  - TIE 접촉이나 MPC 구속을 일체 쓰지 말고, 단일 HEC-MW 그리드 토폴로지로 직접 조립할 것.

  ### 3. FrontISTR 제어 파일 (.cnt) 설정
  - `!SOLUTION, TYPE=NLSTATIC`
  - `!PARAM, NLGEOM=ON`
  - PSA 섹션에 `!SECTION_CONTROL, SECNUM=2, FORM361=UP` 지정 (체적 잠김 차단)
  - 경계 조건: 고정단(X=0, 완전 고정), 구동단(X=L, 20도 순수 굽힘 강제 회전 변위 처방)

  ### 4. 검증 및 출력
  - `example_disp_fold/run_fistr_conforming_benchmark.py`에 러너를 구현하고 독립 실행 검증.
  - 고정단 반력 모멘트(M_root)를 적분 계산하여 이론 정해(M_theory = 0.2317 N*mm)와 비교 출력할 것.
  - 기존 361 FI 결과(5.4 N*mm, 93.3% 잠김) 대비 전단 잠김 소거 여부 및 뉴턴 반복 수렴 속도를 정량적으로 보고할 것.

원하실 때 바로 위 프롬프트를 주시면, 스크립트 작성 및 FrontISTR 연동 해석과 이론치 검증을 바로 착수하겠습니다.



──────
  ### master 브랜치 통합 상태 확인 요약

  • 현재 master HEAD: 5654958 (origin/master 대비 17 커밋 ahead)
      1. beb23f3 ~ 0028e3f: P1~P6 패키지 (*ELSET GENERATE, 출력 변수, 점탄성, 2D/3D Fold-Hold-Unfold, *DLOAD, COH2D4) 통합 완료
      2. 0a858e8: 본 세션의 CPE4IH (2D) & C3D8IH (3D) 이원 혼합 요소 구현 및 카탈로그/솔버 배선 통합 완료
      3. 5654958: 외부 워크트리(wip/cpe4rh-on-head)에서 작업된 CPE4RH (HOURGLASS=ENHANCED|STIFFNESS) 커밋이 0a858e8 위로 Fast-Forward 통합
      완료
  • 미통합 워크트리 참고: D:/PythonCodeStudy/wht_p7 (feat/p7-restart-orientation, e3d3d2c)은 아직 별도 워크트리에 남아 있습니다.




opencode new key
oc_sk_7b253043b420_0ZCmEMsba4hp8p4bsc4rEDfUvlgknPjg


claude -r --allow-dangerously-skip-permissions

agy -c --dangerously-skip-permissions

# contact, D2026-09-19
모든 interface는 abaqus python api 기준으로 구성하고 기존에 없는 옵션은 추가하자. 소스코드에는 충분한 주석을 담도록 한다.
특히, 이론적인 측면에서 abaqus와 차이가 있을 것 아닌가. 그런 부분도 담는다.
dual mortar가 속도가 느리다는 것으로 이해가 되는데 정말 그런지 재조사하고 동급 속도의 방법도 연구된 내용을 확인하자.
각 합격기준 평가로 test에서는 그 진행 상태를 모니터링 할 수 있도록 각각에 대한 log 파일 관리하고 테스트 시작 전에 해당
로그 파일 경로를 안내한다.
모든 작업이 완료된 후에 일반적으로 적절한 api 접촉 예제와 우리의 주요 기술을 담은 md 파일을 만들자. 이 파일은 계속해서
contact 관련 작업 후에 업데이트하고 최신으로 유지되도록 한다. 이 업데이트 관련 작업은 워킹폴더의 gemini md 파일에
기록한다. 지금은 contact 개선 코드이므로 이 부분만 언급하지만 차후에는 다른 부분들도 이런 방식으로 최신의 내용으로 관리할
것이다. 일종의 릴리즈 노트이면서 추가-삭제-수정-개선 등의 상세한 이력을 기록하고 젤 앞에는 현재의 최신 상태를 기록한다.
그리고 각각의 기능 사용 코드를 예제로도 남기고 그것과 연계된 test 코드의 위치와 결과도 남긴다.
이번 구현은 gemini가 모두 진행하지 않고 /lucy-orcha의 팀 C의 에이전트들이 분할 할당받아 구현을 진행하고 이를 수집 조합을
gemini에서 통합하고 수리하는 방식으로 진행하겠다. 마지막에 코드 리뷰를 통해 어긋난 것을 다시 /lucy-orcha 정예팀에
요청하여 검사 받기로 한다. fable, opus 소속팀


# my_plan

$env:PYTHONPATH="examples;.;$env:PYTHONPATH"; python -u examples/ex13_unified_model_io.py --mode roundtrip --elem_jit jax --viewer  --threads 6 --preset moderate-4

# Fetching Sample Input Files
http://desktop-whl:4040/English/?show=SIMACAEEXCRefMap/simaexc-c-fetchproc.htm

# Querying the Keyword/Problem Database
http://desktop-whl:4040/English/?show=SIMACAEEXCRefMap/simaexc-c-keywordproc.htm

# 벤치마크 
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-ov.htm



http://desktop-whl:4040/English/?show=SIMACAETHERefMap/simathe-c-solids.htm
http://desktop-whl:4040/English/?show=SIMACAETHERefMap/simathe-c-solidform.htm
http://desktop-whl:4040/English/?show=SIMACAETHERefMap/simathe-c-hybridincompress.htm
http://desktop-whl:4040/English/?show=SIMACAETHERefMap/simathe-c-solidisoquadhex.htm
http://desktop-whl:4040/English/?show=SIMACAETHERefMap/simathe-c-incompatible.htm
http://desktop-whl:4040/English/?show=SIMACAETHERefMap/simathe-c-tritetwedge.htm

http://desktop-whl:4040/English/?show=SIMACAESUBRefMap/simasub-c-uelmat.htm

http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-basicelemover.htm

http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-singleelemeig.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-elempatcheseig.htm
http://desktop-whl:4040/English/SIMACAEVERRefMap/simaver-c-3dpatch.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-nlgeocantilever.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-caxasaxa2pointbending.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-cookmembrane.htm




http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-beamgap.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-thickcompcyl.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-rubberdisk.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-elasticsheet.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-neckingtensilebar.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-concreteslump.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-hertzcontact.htm
http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-c-pipecrushing.htm



# static benchmark
❯ 다음으로는 static 해석에 대한 벤치마크를 준비해야 할 것.
  http://desktop-whl:4040/English/?show=SIMACAEBMKRefMap/simabmk-m-StaticStressdisplacementAnalysis-sb.htm

  Beam/gap example
  Analysis of an anisotropic layered plate
  Composite shells in cylindrical bending
  Thick composite cylinder subjected to internal pressure
  Uniform collapse of straight and curved pipe segments
  Snap-through of a shallow, cylindrical roof under a point load
  Pressurized rubber disc
  Uniaxial stretching of an elastic sheet with a circular hole
  Necking of a round tensile bar
  Concrete slump test
  The Hertz contact problem
  Crushing of a pipe
  Radial stretching of a cylinder


#



# material verification
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-elastic.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-viscoelastic.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-mullinseffect.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-hysteretic.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-tempelastic.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-fieldelastic.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-viscohyper.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-nonlinviscohyper.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-rateindepplast.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-viscocylinder.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-ratedepplastexp.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-annealtemp.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-tempplastic.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-fieldplastic.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-johnsoncook.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-damage.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-creep.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-viscous.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-concrete.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-concretedamaged.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-damping.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-simpleexpansion.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-plasticity-corrections.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-fieldvardependentconductivity.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-conductivityspecificheat.htm
http://desktop-whl:4040/English/?show=SIMACAEVERRefMap/simaver-c-gapconductance.htm

위 링크는 모두 로컬 서버의 링크이며 모두 로컬 c 드라이브 내에 있는 htm 파일들이다.
내부의 내용을 파악하면, abaqus 기준의 현재의 요구되는 기능들을 정리할 수 있다.
재료 관련의 요구되는 기능들이니 이를 파악하자.
열 전달과 관련된 것은 우선 제외하고 추후로 우선 순위를 조절한다.
대신 온도에 의한 의존된 필드 정보를 사용하는 것은 대상에 포함한다.
왜냐면 온도를 직접적으로 부여하는 경우는 현재 고려 대상이다.

포함된 inp파일의 예제를 읽어 동일한 벤치마크를 수행할 수 있도록 준비한다.
벤치마크 진행은 요구된 기능의 구현과 점검을 모두 다한 후가 될 것이다.


# abaqus input file reader and executor
## 개요
- abaqus input file을 완벽하게 읽을 수 있는 reader 개발
## 방법
- 현재 abaqus 키워드와 대응되는 기능을 중심으로 해당 keyword들에 대해서 지원할 수 있게 설계 우선 진행 
- 참조 문서
  - http://desktop-whl:4040/English/?show=SIMACAEKEYRefMap/simakey-c-std-prodidx.htm
  - 로컬 C 드라이브에 해당 htm 문서 있음
  - htm 링크를 타고 가면 각 키워드의 문법 및 내용이 있다. 
  - SIMACAEKEYRefMap/simakey-m-A-sb.htm
    - A 는 A ~ Z 이면,
    - P,Q는 PQ, simakey-PQ-sb.htm
    - W,X,Y,Z는 WXYZ로 simakey-WXYZ-sb.htm
- Instances, Assembly 구조도 고려
- non-assembly 구조도 지원 가능
- 키워드 구조를 분석하다보면 추가로 요구하는 내부 코드가 있을 것. 이에 대해서도 설계한다. 
- 모든 키워드를 현재 다 구현하기는 어려울 것
- 대신 키워드의 구조를 이해하면, 양식을 체계화 하여 인덱스하기 좋은 체계적은 방법을 추출할 수 있을 것
  - http://desktop-whl:4040/English/?show=SIMACAEKEYRefMap/simakey-c-ov.htm
  - http://desktop-whl:4040/English/?show=SIMACAEMODRefMap/simamod-c-model.htm
  - http://desktop-whl:4040/English/?show=SIMACAEMODRefMap/simamod-m-SpatialModeling-sb.htm
- 즉, 체계적으로 abaqus input 파일을 읽을 수 있는 reader 개발
- 내부 데이터이 구조에 abaqus input 의 내용을 담을 수 있고
- 다음 우리의 기능을 활용할 수 있게 하자.



Kratos 폴더 내에 보면 agent md 파일과 agent 폴더가 있다. 내용을 보고 이해하자.
