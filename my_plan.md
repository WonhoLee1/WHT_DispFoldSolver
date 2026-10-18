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
