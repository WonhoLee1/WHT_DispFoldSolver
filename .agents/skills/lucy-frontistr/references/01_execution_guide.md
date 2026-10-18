# FrontISTR 실행 가이드 및 입출력 튜토리얼

> 크롤링 출처: `https://docs.frontistr.com/ko/` (수록 페이지 수: 8개)

## 수록 문서 목차

- `execution_guide/data_conversion.md` — 데이터 변환 도구
- `execution_guide/overview/01_flow.md` — 해석 흐름
- `execution_guide/procedure/01_serial.md` — 직렬 해석
- `execution_guide/procedure/02_parallel.md` — 병렬 해석
- `execution_guide/tutorial_01/01_overview.md` — 선형 정적 해석 (탄성)
- `execution_guide/tutorial_01/02_dat.md` — 전체 제어 데이터hecmw_ctrl.dat
- `execution_guide/tutorial_01/03_msh.md` — 메시 데이터
- `execution_guide/tutorial_01/04_cnt.md` — 해석 제어 데이터hinge.cnt

---


<!-- ========================================== -->
<!-- FILE: execution_guide/data_conversion.md | TITLE: 데이터 변환 도구 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/execution_guide/data_conversion.html -->

# 데이터 변환 도구

 미정


---


<!-- ========================================== -->
<!-- FILE: execution_guide/overview/01_flow.md | TITLE: 해석 흐름 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/execution_guide/overview/01_flow.html -->

# 해석 흐름

 FrontISTR 해석은 입력 파일을 준비하고 솔버 `fistr1`을 실행한 뒤 출력된 파일을 확인하는 흐름으로 진행된다. 이 페이지에서는 전체 흐름과 취급하는 입력·출력 파일의 종류, 순차·병렬 실행 방법을 정리하여 설명한다. 각 단계의 자세한 절차는 실행 가이드의 해당 페이지를 참조한다.

 
## 전체 흐름

 순차 계산과 병렬 계산은 분할과 결과 병합의 유무가 다르다. 기본 흐름은 다음과 같다.

 
1. FrontISTR 본체와 주변 프로그램을 입수하여 실행할 수 있는 상태로 만든다. → [설치](../../install_guide/binary_install.html)
2. 지정된 형식에 따라 입력 파일(전체 제어 데이터 `hecmw_ctrl.dat`, 메시 데이터, 해석 제어 데이터)을 준비한다.
3. 병렬 계산을 수행하는 경우 메시 분할 프로그램 `hecmw_part1`로 단일 영역 메시를 분산 영역 메시로 분할한다. 순차 계산에서는 이 절차가 필요하지 않다.
4. `fistr1`을 실행한다.
5. 출력된 로그·결과·시각화 파일을 확인한다. 병렬 계산에서 결과를 하나의 파일로 집약하려는 경우 필요에 따라 결과를 병합한다.

 순차 계산의 구체적인 절차는 [순차 처리에 의한 해석](../procedure/01_serial.html), 병렬 계산의 구체적인 절차는 [병렬 처리에 의한 해석](../procedure/02_parallel.html)을 참조한다.

 
## 입력·출력 파일

 ![FrontISTR 입출력 파일](/execution_guide/image/analysis01_01.png)

  그림 해석의 입출력 파일  
### 입력 파일

 FrontISTR는 다음 3종류의 입력 파일을 읽는다.

 
- **전체 제어 데이터**(`hecmw_ctrl.dat`): 메시 데이터·해석 제어 데이터의 입력 파일명과 결과·시각화 데이터의 출력 파일명을 지정한다.
- **메시 데이터**: 유한요소 메시, 재료 데이터, 섹션 데이터, 그룹 데이터를 정의한다. 단일 CPU로 해석하는 경우 단일 영역 메시를 사용하고, 여러 CPU에서 병렬 실행하는 경우 `hecmw_part1`로 분할한 분산 영역 메시를 사용한다.
- **해석 제어 데이터**: 해석 종류, 경계 조건, 하중, 솔버 제어, 시각화 제어를 정의한다.

 이들은 모두 텍스트 파일이며 에디터로 작성·편집할 수 있다. 각 파일에 기재하는 내용의 형식은 [입력 규칙](../../keyword_reference/input_rules.html)을 참조한다.

 
### 출력 파일

 FrontISTR를 실행하면 해석 내용과 해석 제어 데이터의 지정에 따라 다음 파일이 출력된다.

 
- **메시지·로그 파일**: 실행 진행 상황과 변위·변형률·응력의 최대/최소값, 모드 해석 결과 등의 요약이 출력된다.
- **해석 결과 파일**: 해석 제어 데이터에서 `!WRITE,RESULT`를 지정한 경우 절점·요소별 물리량이 출력된다. 후처리나 다른 해석에서 읽어들이는 데 사용한다.
- **시각화 파일**: `!WRITE,VISUAL`을 지정한 경우 시각화 도구용 형식으로 출력된다.

 출력되는 파일의 명명 규칙과 확인 방법은 [순차 처리에 의한 해석](../procedure/01_serial.html)·[병렬 처리에 의한 해석](../procedure/02_parallel.html)에, 출력 가능한 물리량과 형식의 자세한 내용은 [출력·리스타트](../../functions_reference/09_output_restart.html)에 설명되어 있다.

 
## 실행 방법

 `fistr1`(Windows 버전은 `fistr1.exe`)을 PATH가 설정된 디렉터리 또는 입력 파일이 있는 디렉터리에 둔다. 입력 파일이 있는 디렉터리에서 다음을 실행하면 순차 계산이 시작된다.

 
```text
fistr1
```

 `hecmw_ctrl.dat`에 기술한 메시 데이터·해석 제어 데이터가 읽히고 지정한 결과·시각화 파일이 출력된다.

 
## 병렬 실행

 병렬 계산에서는 미리 MPI에 대응하는 `fistr1`을 준비하고 `hecmw_part1`로 분할한 분산 영역 메시를 입력한다. 실행은 MPI 실행 환경을 따른다. 4개 영역으로 실행하는 예를 보인다.

 
```text
mpirun -np 4 fistr1
```

 Windows 버전 배포 바이너리에는 MPI 런타임이 포함되어 있어 별도로 MPI를 설치하지 않고 병렬 실행할 수 있다. 자세한 내용은 [바이너리 패키지를 통한 설치](../../install_guide/binary_install.html)를 참조한다. 영역 분할의 개념은 [분할·병렬 계산](../../functions_reference/10_partition.html)에 설명되어 있다.

 
> Ver.3.x까지는 병렬 접촉 해석의 경우에만 단일 영역 메시를 입력해야 했지만, Ver.5.x 이후에는 병렬 접촉 해석에서도 다른 해석과 마찬가지로 분산 영역 메시를 입력하는 방식으로 통일되었다.

 
## 관련 항목

 
- [순차 처리에 의한 해석](../procedure/01_serial.html) — 순차 실행 절차와 출력 파일 확인.
- [병렬 처리에 의한 해석](../procedure/02_parallel.html) — 분할부터 병렬 실행까지의 절차.
- [입력 규칙](../../keyword_reference/input_rules.html) — 입력 파일 공통 형식 규칙.
- [출력·리스타트](../../functions_reference/09_output_restart.html) — 출력 가능한 물리량과 리스타트 구조.
- [분할·병렬 계산](../../functions_reference/10_partition.html) — 병렬 계산을 위한 분할.


---


<!-- ========================================== -->
<!-- FILE: execution_guide/procedure/01_serial.md | TITLE: 직렬 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/execution_guide/procedure/01_serial.html -->

# 직렬 해석

 
## 실행 흐름

 그림 3.1.2에 FrontISTR를 사용하여 단일 프로세서에서 직렬 처리할 때의 실행 흐름을 나타냅니다.

 그림 3.1.2 직렬 실행 흐름

 
## 준비할 입력 파일

 
### (1) 전체 제어 데이터(dat 확장자)

 이 파일은 메시 데이터와 해석 제어 데이터의 입력 파일 및 해석 결과 출력 파일을 지정합니다. 파일명은 hecmw_ctrl.dat로 고정됩니다.

 전체 제어 데이터의 예를 아래에 나타냅니다. 이 예에서 FrontISTR는 단일 영역 메시 데이터 model.msh와 해석 제어 데이터 model.cnt를 읽고 해석 결과 데이터 model.res.0.1을 출력합니다. 또한 hecmw_vis는 단일 영역 메시 데이터 model.msh와 해석 결과 데이터 model.res.0.1을 읽고 출력 지정에 따라 model_vis_psf.0000.(확장자)을 출력합니다. 자세한 내용은 [키워드 참조(전체 제어 데이터)](../../keyword_reference/dat/index.html)를 참조하십시오.

 
### (2) 단일 영역 메시 데이터(msh 확장자)

 이 파일은 해석 모델의 전체 메시 구성, 재료 데이터, 해석 제어 데이터에서 사용하는 그룹 데이터 등을 정의합니다. 자세한 내용은 [키워드 참조(메시 데이터)](../../keyword_reference/msh/index.html)를 참조하십시오.

 
### (3) 해석 제어 데이터(cnt 확장자)

 이 파일은 해석 유형, 변위 경계 조건, 하중 경계 조건 등을 정의합니다. 또한 솔버 제어 데이터와 비주얼라이저 제어 데이터를 지정합니다. 해석 제어 데이터의 예는 3장에 제시되어 있습니다. 자세한 내용은 [키워드 참조(해석 제어 데이터)](../../keyword_reference/cnt/index.html)를 참조하십시오.

 
### (4) 시각화 제어 데이터(ini 확장자)

 이 파일은 hecmw_vis의 제어 데이터를 지정합니다. 기본 파일명은 hecmw_vis.ini입니다. 시각화 제어 데이터의 예를 아래에 나타냅니다. 이 예에서는 MicroAVS용 비정형 격자 데이터(inp 확장자)를 출력합니다. 자세한 내용은 키워드 참조의 [`!visual`](../../keyword_reference/cnt/visualization/visual.ja.md)과 [`!OUTPUT_VIS`](../../keyword_reference/cnt/output_vis.ja.md)를 참조하십시오.

 
## 실행 절차

 입력 파일이 있는 디렉터리에서 다음 명령으로 FrontISTR를 실행합니다.

 
```text
fistr1
```

 시각화는 두 가지 방법으로 수행할 수 있습니다. 하나는 해석 제어 데이터에 다음을 지정하여 FrontISTR의 후처리로 실행하는 방법입니다.

 
```text
!WRITE, VISUAL
```

 이렇게 하면 시각화가 자동으로 실행됩니다. 이 경우 시각화 제어 데이터가 해석 제어 데이터에 포함되어 있어야 합니다.

 FrontISTR 실행이 끝난 뒤 시각화를 수행하려면 먼저 해석 제어 데이터에 다음을 지정합니다.

 
```text
!WRITE, RESULT
```

 그런 다음 FrontISTR를 실행합니다.

 FrontISTR가 끝나면 입력 파일과 해석 결과 파일이 있는 디렉터리에서 다음 명령으로 hecmw_vis를 실행합니다.

 
```text
hecmw_vis1
```

 
## 출력 파일 설명

 
### (1) 해석 결과 메시지 파일(msg 확장자)

 이 파일에는 FrontISTR 해석 진행 상황 등의 메시지가 기록됩니다. 실행마다 하나의 파일이 생성되며 파일명은 FSTR.msg로 고정됩니다.

 
### (2) 해석 결과 로그 파일(log 확장자)

 이 파일에는 FrontISTR의 각 절점과 요소에서의 물리량 해석 결과가 기록됩니다. 물리량의 최대/최소값과 모드 해석 결과도 포함됩니다. 동적 해석에서는 모든 스텝의 해석 결과가 이 파일에 출력됩니다. 실행마다 하나의 파일이 생성되며 파일명은 0.log로 고정됩니다.

 
### (3) 해석 결과 파일(확장자 없음)

 이 파일은 !WRITE, RESULT 옵션을 지정했을 때 출력됩니다.

 이 파일에는 FrontISTR의 각 절점과 요소에서의 물리량 해석 결과가 기록됩니다. 스텝마다 파일이 생성되며 전체 제어 데이터에서 지정한 파일 헤더를 사용하여 다음과 같이 이름이 정해집니다.

 명명 규칙: (!RESULT로 지정한 파일 헤더).0.(스텝 번호)

 예: model.res.0.1

 
### (4) 해석 결과 비트맵 파일(bmp 확장자)

 이 파일은 시각화 제어 데이터에서 지정했을 때 출력됩니다.

 이 파일에는 시각화 처리로 생성된 비트맵 데이터가 들어 있습니다. 파일명은 전체 제어 데이터에서 지정한 파일 헤더를 사용해 정해집니다. 명명 규칙의 자세한 내용은 hecmw1 문서(0803_001f_hecmw_PC_cluster_201_vis.pdf)를 참조하십시오.

 
### (5) 해석 결과 비정형 격자 데이터 파일(inp 확장자)

 이 파일은 시각화 제어 데이터에서 지정했을 때 출력됩니다.

 이 파일은 REVOCAP_PrePost, MicroAVS 등의 후처리에 사용할 수 있습니다. 파일명은 전체 제어 데이터에서 지정한 파일 헤더를 사용하여 다음과 같이 정해집니다.

 명명 규칙: (!RESULT로 지정한 파일 헤더)_psf.(스텝 번호).inp

 예: model_vis_psf.0000.inp

 
### (6) 해석 결과 뉴트럴 파일(neu 확장자)

 이 파일은 시각화 제어 데이터에서 지정했을 때 출력됩니다.

 이 파일은 Femap을 사용한 후처리에 사용할 수 있습니다. 파일명은 전체 제어 데이터에서 지정한 파일 헤더를 사용하여 다음과 같이 정해집니다.

 명명 규칙: (!RESULT로 지정한 파일 헤더)_psf.(스텝 번호).neu

 예: model_vis_psf.0000.neu

 주의: FSTR.dbg도 출력되지만 디버깅용이므로 일반적으로 참조할 필요가 없습니다.


---


<!-- ========================================== -->
<!-- FILE: execution_guide/procedure/02_parallel.md | TITLE: 병렬 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/execution_guide/procedure/02_parallel.html -->

# 병렬 해석

 
## 실행 흐름

 그림 3.1.3은 FrontISTR를 사용하여 다중 프로세서 병렬 처리를 수행할 때의 실행 흐름을 나타냅니다.

 그림 3.1.3 병렬 처리 실행 흐름

 
## 준비할 입력 파일

 
### (1) 전체 제어 데이터(dat 확장자)

 이 파일은 메시 데이터와 해석 제어 데이터의 입력 파일 및 해석 결과 출력 파일을 지정합니다. 파일명은 hecmw_ctrl.dat로 고정됩니다.

 전체 제어 데이터의 예를 아래에 나타냅니다. 이 예에서는 먼저 hecmw_part가 단일 영역 메시 데이터 model.msh를 읽고 분산 영역 메시 데이터 model_8.0부터 n까지를 출력합니다. FrontISTR는 분산 영역 메시 데이터 model_8.0부터 n까지와 해석 제어 데이터 model.cnt를 읽고 해석 결과 데이터 model.res.0부터 n.1까지를 출력합니다. 또한 hecmw_vis는 분산 영역 메시 데이터 model_8.0부터 n까지와 해석 결과 데이터 model.res.0부터 n.1까지를 읽고 지정한 출력에 대응하는 model_vis_psf.0000.(확장자)을 출력합니다. 자세한 내용은 [키워드 참조(전체 제어 데이터)](../../keyword_reference/dat/index.html)를 참조하십시오.

 
### (2) 단일 영역 메시 데이터(msh 확장자)

 이 파일은 해석 대상의 전체 메시 구성, 재료 데이터, 해석 제어 데이터에서 사용하는 그룹 데이터 등을 정의합니다. 자세한 내용은 [키워드 참조(메시 데이터)](../../keyword_reference/msh/index.html)를 참조하십시오.

 
### (3) 해석 제어 데이터(cnt 확장자)

 이 파일은 해석 유형, 변위 경계 조건, 하중 경계 조건 등을 정의합니다. 또한 솔버 제어 데이터와 비주얼라이저 제어 데이터를 지정합니다. 해석 제어 데이터의 예는 3장에 제시되어 있습니다. 자세한 내용은 [키워드 참조(해석 제어 데이터)](../../keyword_reference/cnt/index.html)를 참조하십시오.

 
### (4) 분할 유틸리티 제어 데이터(dat 확장자)

 이 파일은 hecmw_part의 제어 데이터를 지정합니다. 파일명은 hecmw_part_ctrl.dat로 고정됩니다. 분할 유틸리티 제어 데이터의 예를 아래에 나타냅니다. 이 예에서는 PMETIS 분할 방법을 사용하여 하나의 영역을 8개 영역으로 분할합니다. 또한 분할 후의 메시를 표시하기 위한 model_8.inp 파일도 출력됩니다. 자세한 내용은 hecmw1 문서(0803_001x_hecmw_part_201_users.pdf)를 참조하십시오.

 
### (5) 시각화 제어 데이터(ini 확장자)

 이 파일은 hecmw_vis의 제어 데이터를 지정합니다. 기본 파일명은 hecmw_vis.ini입니다. 시각화 제어 데이터의 예를 아래에 나타냅니다. 이 예에서는 MicroAVS용 비정형 격자 데이터(inp 확장자)를 출력합니다. 자세한 내용은 키워드 참조의 [`!visual`](../../keyword_reference/cnt/visualization/visual.ja.md)과 [`!OUTPUT_VIS`](../../keyword_reference/cnt/output_vis.ja.md)를 참조하십시오.

 
## 실행 방법

 입력 파일이 있는 디렉터리에서 다음 명령줄로 hecmw_part를 실행합니다.

 
```text
hecmw_part1
```

 입력 파일이 있는 디렉터리에서 다음 명령줄로 FrontISTR를 실행합니다. MPI 프로세스를 기동하는 방법은 실행 환경에 맞게 조정해야 합니다.

 
```text
mpirun -np 8 fistr1
```

 시각화는 두 가지 방법으로 실행할 수 있습니다. 하나는 FrontISTR의 후처리로 실행하는 방법입니다. 이 경우 해석 제어 데이터에

 !WRITE, VISUAL

 을 지정하면 시각화가 자동으로 실행됩니다. 이 경우 시각화 제어 데이터가 해석 제어 데이터에 포함되어 있어야 합니다.

 FrontISTR 실행이 끝난 후 시각화를 수행하려면 먼저 해석 제어 데이터에

 !WRITE, RESULT

 를 지정하고 FrontISTR를 실행합니다.

 FrontISTR가 끝나면 입력 파일과 해석 결과 파일이 있는 디렉터리에서 다음 명령줄로 hecmw_vis를 실행합니다. MPI 프로세스를 기동하는 방법은 실행 환경에 맞게 조정해야 합니다.

 
```text
mpirun -np 8 hecmw_vis1
```

 
## 출력 파일 설명

 
### (1) 분할 유틸리티 로그 파일(log 확장자)

 이 파일에는 hecmw_part 처리 진행 상황 등의 메시지가 기록됩니다. 파일명은 hecmw_part.log로 고정됩니다.

 
### (2) 분산 영역 메시 파일(확장자 없음)

 이 파일에는 영역 분할된 메시 구성, 재료 데이터, 해석 제어 데이터에서 사용하는 그룹 데이터 등이 들어 있습니다. 분산 영역마다 하나의 파일이 생성되며 전체 제어 데이터에서 지정한 파일 헤더를 사용하여 다음과 같이 이름이 정해집니다.

 명명 규칙: (!MESH로 지정한 파일 헤더).(분산 영역 번호)

 예: model_8.0부터 model_8.7까지

 
### (3) 영역 분할 메시 표시 파일(inp 확장자)

 이 파일에는 영역 분할된 메시를 표시하기 위한 비정형 격자 데이터가 들어 있습니다. MicroAVS 등의 소프트웨어로 표시할 수 있습니다.

 
### (4) 해석 결과 메시지 파일(msg 확장자)

 이 파일에는 FrontISTR 해석 진행 상황 등의 메시지가 기록됩니다. 실행마다 하나의 파일이 생성되며 파일명은 FSTR.msg로 고정됩니다.

 
### (5) 해석 결과 로그 파일(log 확장자)

 이 파일에는 FrontISTR의 각 절점과 요소에서의 물리량 해석 결과가 기록됩니다. 물리량의 최대/최소값과 모드 해석 결과도 포함됩니다. 동적 해석에서는 모든 스텝의 해석 결과가 이 파일에 출력됩니다. 분산 영역마다 하나의 파일이 생성되며 파일명은 n.log(n은 분산 영역 번호)로 고정됩니다.

 
### (6) 해석 결과 파일(확장자 없음)

 이 파일은 !WRITE, RESULT 옵션을 지정했을 때 출력됩니다.

 이 파일에는 FrontISTR의 각 절점과 요소에서의 물리량 해석 결과가 기록됩니다. 각 분산 영역과 각 스텝마다 파일이 생성되며 전체 제어 데이터에서 지정한 파일 헤더를 사용하여 다음과 같이 이름이 정해집니다.

 명명 규칙: (!RESULT로 지정한 파일 헤더).(분산 영역 번호).(스텝 번호)

 예: model_8.res.0.1부터 model_8.res.7.1까지

 
### (7) 해석 결과 비트맵 파일(bmp 확장자)

 이 파일은 시각화 제어 데이터에서 지정했을 때 출력됩니다.

 이 파일에는 시각화 처리로 생성된 비트맵 데이터가 들어 있습니다. 파일명은 전체 제어 데이터에서 지정한 파일 헤더를 사용해 정해집니다. 명명 규칙의 자세한 내용은 hecmw1 문서(0803_001f_hecmw_PC_cluster_201_vis.pdf)를 참조하십시오.

 
### (8) 해석 결과 비정형 격자 데이터 파일(inp 확장자)

 이 파일은 시각화 제어 데이터에서 지정했을 때 출력됩니다.

 이 파일은 REVOCAP_PrePost, MicroAVS 등의 후처리에 사용할 수 있습니다. 파일명은 전체 제어 데이터에서 지정한 파일 헤더를 사용하여 다음과 같이 정해집니다.

 명명 규칙: (!RESULT로 지정한 파일 헤더)_psf.(스텝 번호).inp

 예: model_vis_psf.0000.inp

 
### (9) 해석 결과 뉴트럴 파일(neu 확장자)

 이 파일은 시각화 제어 데이터에서 지정했을 때 출력됩니다.

 이 파일은 Femap을 사용한 후처리에 사용할 수 있습니다. 파일명은 전체 제어 데이터에서 지정한 파일 헤더를 사용하여 다음과 같이 정해집니다.

 명명 규칙: (!RESULT로 지정한 파일 헤더)_psf.(스텝 번호).neu

 예: model_vis_psf.0000.neu

 주의: FSTR.dbg.0부터 n 파일도 출력되지만 디버깅용이므로 일반적으로 참조할 필요가 없습니다.


---


<!-- ========================================== -->
<!-- FILE: execution_guide/tutorial_01/01_overview.md | TITLE: 선형 정적 해석 (탄성) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/execution_guide/tutorial_01/01_overview.html -->

# 선형 정적 해석 (탄성)

 이 해석에서는 `tutorial/01_elastic_hinge`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 힌지 부품입니다. 형상은 그림 4.1.1에, 메시 데이터는 그림 4.1.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 선형 정적 해석 | !SOLUTION,TYPE=STATIC |  |
| 절점 수 | 84,056 |  |  |
| 요소 수 | 49,871 |  |  |
| 요소 유형 | 10절점 2차 사면체 요소 | !ELEMENT,TYPE=342 | [요소 라이브러리](../../functions_reference/02_element.html) |
| 재료명 | STEEL | !MATERIAL,NAME=STEEL | [재료 데이터](../../functions_reference/03_material.html) |
| 재료 특성 | ELASTIC | !ELASTIC |  |
| 경계 조건 | 구속, 집중 하중 |  |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 ![힌지 부품 형상](/execution_guide/tutorial_01/image/tutorial01_01.png)

  그림 4.1.1 힌지 부품 형상  ![힌지 부품 메시 데이터](/execution_guide/tutorial_01/image/tutorial01_02.png)

  그림 4.1.2 힌지 부품 메시 데이터  
## 해석 내용

 FrontISTR 소스 코드를 압축 해제하고 이 예제의 디렉터리로 이동하여 해석에 필요한 다음 파일이

 
| 파일명 | 종류 | 역할 |
| --- | --- | --- |
| `hecmw_ctrl.dat` | 전체 제어 데이터 | 메시 데이터 및 해석 제어 데이터의 입력·출력 파일을 지정합니다. |
| `hinge.cnt` | 해석 제어 데이터 | 해석 종류, 변위 경계 조건, 집중 하중 등을 정의하고 솔버 및 비주얼라이저의 제어도 지정합니다. |
| `hinge.msh` | 메시 데이터 | 유한요소 메시와 재료 및 단면 데이터를 정의합니다. |

 있는지 확인합니다.

 
```text
tar xvf FrontISTR.tar.gz
cd FrontISTR/tutorial/01_elastic_hinge
ls
```

 실행 결과:

 
```text
hecmw_ctrl.dat  hinge.cnt  hinge.msh
```

 그림 4.1.1에 나타낸 구속면의 변위를 구속하고 하중면에 집중 하중을 부여하여 응력 해석을 수행합니다.

 
## 해석 절차

 FrontISTR 명령 `fistr1`을 실행합니다.

 
```text
fistr1 -t 4
```

 (4스레드로 실행합니다.)

 
```text
##################################################################
#                         FrontISTR
#
##################################################################
---
version:    5.1.0
git_hash:   acab000c8c633b7b9d596424769e14363f720841
build:
  date:     2020-10-05T07:39:55Z
  MPI:      enabled
  OpenMP:   enabled
  option:   "-p --with-tools --with-refiner --with-metis --with-mumps --with-lapack --with-ml --with-mkl "
  HECMW_METIS_VER: 5
execute:
  date:       2020-10-07T10:01:16+0900
  processes:  1
  threads:    4
  cores:      4
  host:
    0: flow-p06
---
...
 Step control not defined! Using default step=1
 fstr_setup: OK
 Start visualize PSF 1 at timestep 0

 loading step=    1
 sub_step= 1,   current_time=  0.0000E+00, time_inc=  0.1000E+01
 loading_factor=    0.0000000   1.0000000
### 3x3 BLOCK CG, SSOR, 1
      1    1.903375E+00
      2    1.974378E+00
      3    2.534627E+00
...
...
   2967    1.080216E-08
   2968    1.004317E-08
   2969    9.375729E-09
### Relative residual = 9.39429E-09

### summary of linear solver
      2969 iterations      9.394286E-09
    set-up time      :     1.953022E-01
    solver time      :     5.704201E+01
    solver/comm time :     5.145826E-01
    solver/matvec    :     2.306329E+01
    solver/precond   :     2.632665E+01
    solver/1 iter    :     1.921253E-02
    work ratio (%)   :     9.909789E+01

 Start visualize PSF 1 at timestep 1
### FSTR_SOLVE_NLGEOM FINISHED!

 ====================================
    TOTAL TIME (sec) :     59.99
           pre (sec) :      0.71
         solve (sec) :     59.29
 ====================================
 FrontISTR Completed !!
```

 `FrontISTR Completed !!`가 표시되면 해석이 완료된 것입니다.

 
## 해석 결과

 해석이 완료되면 몇 개의 파일이 새로 생성됩니다.

 
```text
ls
```

 실행 결과:

 
```text
0.log       hecmw_ctrl.dat  hinge.res.0.0            hinge_vis_psf.0001
FSTR.dbg.0  hecmw_vis.ini   hinge.res.0.1            hinge_vis_psf.0001.pvtu
FSTR.msg    hinge.cnt       hinge_vis_psf.0000
FSTR.sta    hinge.msh       hinge_vis_psf.0000.pvtu
```

 `*.res.*` 파일은 결과 데이터이며 FrontISTR의 해석 결과가 저장되어 있습니다. REVOCAP_PrePost 등으로 표시할 수 있습니다.

 `*_vis_*` 파일은 시각화 데이터이며 범용 시각화 소프트웨어로 표시할 수 있습니다. 이 예제에서는 VTK 형식으로 출력하므로 ParaView 등을 사용하여 표시합니다.

 REVOCAP_PrePost로 작성한 미제스 응력 컨투어도를 그림 4.1.3에 나타냅니다. 또한 해석 결과의 수치 데이터로 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![미제스 응력 해석 결과](/execution_guide/tutorial_01/image/tutorial01_03.png)

  그림 4.1.3 미제스 응력 해석 결과  
### 해석 결과 로그 `0.log`

  
```text
fstr_setup: OK
#### Result step=     0
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1  0.0000E+00         1
 //U2    0.0000E+00         1  0.0000E+00         1
 //U3    0.0000E+00         1  0.0000E+00         1
 //E11   0.0000E+00         1  0.0000E+00         1
 //E22   0.0000E+00         1  0.0000E+00         1
 //E33   0.0000E+00         1  0.0000E+00         1
 //E12   0.0000E+00         1  0.0000E+00         1
 //E23   0.0000E+00         1  0.0000E+00         1
 //E31   0.0000E+00         1  0.0000E+00         1
 //S11   0.0000E+00         1  0.0000E+00         1
 //S22   0.0000E+00         1  0.0000E+00         1
 //S33   0.0000E+00         1  0.0000E+00         1
 //S12   0.0000E+00         1  0.0000E+00         1
 //S23   0.0000E+00         1  0.0000E+00         1
 //S31   0.0000E+00         1  0.0000E+00         1
 //SMS   0.0000E+00         1  0.0000E+00         1
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   0.0000E+00         1  0.0000E+00         1
 //E22   0.0000E+00         1  0.0000E+00         1
 //E33   0.0000E+00         1  0.0000E+00         1
 //E12   0.0000E+00         1  0.0000E+00         1
 //E23   0.0000E+00         1  0.0000E+00         1
 //E31   0.0000E+00         1  0.0000E+00         1
 //S11   0.0000E+00         1  0.0000E+00         1
 //S22   0.0000E+00         1  0.0000E+00         1
 //S33   0.0000E+00         1  0.0000E+00         1
 //S12   0.0000E+00         1  0.0000E+00         1
 //S23   0.0000E+00         1  0.0000E+00         1
 //S31   0.0000E+00         1  0.0000E+00         1
 //SMS   0.0000E+00         1  0.0000E+00         1
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1  0.0000E+00         1
 //U2    0.0000E+00         1  0.0000E+00         1
 //U3    0.0000E+00         1  0.0000E+00         1
 //E11   0.0000E+00         1  0.0000E+00         1
 //E22   0.0000E+00         1  0.0000E+00         1
 //E33   0.0000E+00         1  0.0000E+00         1
 //E12   0.0000E+00         1  0.0000E+00         1
 //E23   0.0000E+00         1  0.0000E+00         1
 //E31   0.0000E+00         1  0.0000E+00         1
 //S11   0.0000E+00         1  0.0000E+00         1
 //S22   0.0000E+00         1  0.0000E+00         1
 //S33   0.0000E+00         1  0.0000E+00         1
 //S12   0.0000E+00         1  0.0000E+00         1
 //S23   0.0000E+00         1  0.0000E+00         1
 //S31   0.0000E+00         1  0.0000E+00         1
 //SMS   0.0000E+00         1  0.0000E+00         1
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   0.0000E+00         1  0.0000E+00         1
 //E22   0.0000E+00         1  0.0000E+00         1
 //E33   0.0000E+00         1  0.0000E+00         1
 //E12   0.0000E+00         1  0.0000E+00         1
 //E23   0.0000E+00         1  0.0000E+00         1
 //E31   0.0000E+00         1  0.0000E+00         1
 //S11   0.0000E+00         1  0.0000E+00         1
 //S22   0.0000E+00         1  0.0000E+00         1
 //S33   0.0000E+00         1  0.0000E+00         1
 //S12   0.0000E+00         1  0.0000E+00         1
 //S23   0.0000E+00         1  0.0000E+00         1
 //S31   0.0000E+00         1  0.0000E+00         1
 //SMS   0.0000E+00         1  0.0000E+00         1
#### Result step=     1
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    3.9115E-02     82452 -7.1083E-04     65233
 //U2    7.4504E-05       354 -5.8813E-04       696
 //U3    5.9493E-04        84 -5.8751E-03     61080
 //E11   1.3777E-03       130 -1.3653E-03     77625
 //E22   4.9199E-04        61 -5.4370E-04       102
 //E33   6.8634E-04     51036 -6.1176E-04     30070
 //E12   7.1556E-04     27808 -6.8093E-04     27863
 //E23   5.3666E-04        56 -5.4347E-04        82
 //E31   7.2396E-04     36168 -9.6621E-04       130
 //S11   3.8626E+02       130 -3.6387E+02     28580
 //S22   1.6628E+02       130 -1.5743E+02     28580
 //S33   1.6502E+02     30033 -1.5643E+02     28580
 //S12   5.7795E+01     27808 -5.4998E+01     27863
 //S23   4.3345E+01        56 -4.3896E+01        82
 //S31   5.8474E+01     36168 -7.8040E+01       130
 //SMS   2.8195E+02     77625  1.2755E-02     75112
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   1.0731E-03     10485 -1.2123E-03     41779
 //E22   3.9143E-04     33536 -4.1389E-04     22892
 //E33   5.9415E-04     44563 -5.0497E-04     47965
 //E12   5.3264E-04      9163 -5.0405E-04      9161
 //E23   3.9226E-04     33024 -4.1464E-04     23465
 //E31   5.7633E-04     43142 -4.8019E-04      9571
 //S11   2.7231E+02      9180 -2.9763E+02     41779
 //S22   1.0792E+02      9180 -1.0656E+02     41779
 //S33   1.3921E+02     44569 -1.1431E+02     47974
 //S12   4.3021E+01      9163 -4.0712E+01      9161
 //S23   3.1683E+01     33024 -3.3490E+01     23465
 //S31   4.6550E+01     43142 -3.8785E+01      9571
 //SMS   2.4057E+02     41779  3.1383E-02     38687
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    3.9115E-02     82452 -7.1083E-04     65233
 //U2    7.4504E-05       354 -5.8813E-04       696
 //U3    5.9493E-04        84 -5.8751E-03     61080
 //E11   1.3777E-03       130 -1.3653E-03     77625
 //E22   4.9199E-04        61 -5.4370E-04       102
 //E33   6.8634E-04     51036 -6.1176E-04     30070
 //E12   7.1556E-04     27808 -6.8093E-04     27863
 //E23   5.3666E-04        56 -5.4347E-04        82
 //E31   7.2396E-04     36168 -9.6621E-04       130
 //S11   3.8626E+02       130 -3.6387E+02     28580
 //S22   1.6628E+02       130 -1.5743E+02     28580
 //S33   1.6502E+02     30033 -1.5643E+02     28580
 //S12   5.7795E+01     27808 -5.4998E+01     27863
 //S23   4.3345E+01        56 -4.3896E+01        82
 //S31   5.8474E+01     36168 -7.8040E+01       130
 //SMS   2.8195E+02     77625  1.2755E-02     75112
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   1.0731E-03     10485 -1.2123E-03     41779
 //E22   3.9143E-04     33536 -4.1389E-04     22892
 //E33   5.9415E-04     44563 -5.0497E-04     47965
 //E12   5.3264E-04      9163 -5.0405E-04      9161
 //E23   3.9226E-04     33024 -4.1464E-04     23465
 //E31   5.7633E-04     43142 -4.8019E-04      9571
 //S11   2.7231E+02      9180 -2.9763E+02     41779
 //S22   1.0792E+02      9180 -1.0656E+02     41779
 //S33   1.3921E+02     44569 -1.1431E+02     47974
 //S12   4.3021E+01      9163 -4.0712E+01      9161
 //S23   3.1683E+01     33024 -3.3490E+01     23465
 //S31   4.6550E+01     43142 -3.8785E+01      9571
 //SMS   2.4057E+02     41779  3.1383E-02     38687
```


---


<!-- ========================================== -->
<!-- FILE: execution_guide/tutorial_01/02_dat.md | TITLE: 전체 제어 데이터hecmw_ctrl.dat -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/execution_guide/tutorial_01/02_dat.html -->

# 전체 제어 데이터 `hecmw_ctrl.dat`

 메시 데이터와 해석 제어 데이터의 입출력 파일을 지정합니다.

 
```text
#
# 솔버용
#
!MESH, NAME=fstrMSH, TYPE=HECMW-ENTIRE
 hinge.msh
!CONTROL, NAME=fstrCNT
 hinge.cnt
!RESULT, NAME=fstrRES, IO=OUT
 hinge.res
!RESULT, NAME=vis_out, IO=OUT
 hinge_vis
```

 
- [!MESH](../../keyword_reference/dat/mesh.html): 단일 영역 메시 데이터 파일 지정
- [!CONTROL](../../keyword_reference/dat/control.html): 해석 제어 데이터 지정
- [!RESULT](../../keyword_reference/dat/result.html): 결과 데이터 및 시각화 데이터 지정


---


<!-- ========================================== -->
<!-- FILE: execution_guide/tutorial_01/03_msh.md | TITLE: 메시 데이터 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/execution_guide/tutorial_01/03_msh.html -->

# 메시 데이터

 유한요소 메시와 그 재료 및 단면 데이터를 정의합니다. (일부 예만 표시)

 
```text
!HEADER
 HECMW_Msh File generated by REVOCAP
!NODE
       1,    -1.22042,     2.23355,     1.65220
       2,    -1.27050,    -3.10529,     1.59209
...
!ELEMENT, TYPE=342
       1,    1157,    3549,    3321,    3739,   12629,   12627,   12626,   12628,   12631,   12630
       2,    8207,    3321,    3549,    3739,   12629,   12633,   12632,   12634,   12630,   12631
...
!MATERIAL, NAME=STEEL, ITEM=2
!ITEM=1, SUBITEM=2
210000.0, 0.3
!ITEM=2, SUBITEM=1
7.85e-6
!SECTION, TYPE=SOLID, EGRP=Solid0, MATERIAL=STEEL
!EGROUP, EGRP=Solid0
1
2
...
!END
```

 
- [!HEADER](../../keyword_reference/msh/header.html): 헤더 지정
- [!NODE](../../keyword_reference/msh/node.html): 절점 정의
- [!ELEMENT](../../keyword_reference/msh/element.html): 요소 정의
- [!MATERIAL](../../keyword_reference/msh/material.html): 재료 정의
- [!SECTION](../../keyword_reference/msh/section.html): 단면 정의
- [!EGROUP](../../keyword_reference/msh/egroup.html): 요소 그룹 정의
- [!END](../../keyword_reference/msh/end.html): 메시 데이터의 끝 지정


---


<!-- ========================================== -->
<!-- FILE: execution_guide/tutorial_01/04_cnt.md | TITLE: 해석 제어 데이터hinge.cnt -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/execution_guide/tutorial_01/04_cnt.html -->

# 해석 제어 데이터 `hinge.cnt`

 해석 종류, 변위 경계조건, 집중하중 등을 정의하고 솔버와 비주얼라이저의 제어도 지정합니다.

 
```text
#  FISTR 제어 파일
## 해석 제어
!VERSION
 3
!SOLUTION, TYPE=STATIC
!WRITE,RESULT              
!WRITE,VISUAL
## 솔버 제어
### 경계 조건
!BOUNDARY
 BND0, 1, 3, 0.000000      
!BOUNDARY
 BND1, 1, 3, 0.000000      
!CLOAD
 CL0, 1, 0.01000           
### 재료
!MATERIAL, NAME=STEEL
!ELASTIC                   
 210000.0, 0.3
!DENSITY
 7.85e-6
### 솔버 설정
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=YES,TIMELOG=YES
 10000, 1
 1.0e-08, 1.0, 0.0
## 후처리 제어
!VISUAL,method=PSR
!surface_num=1
!surface 1                  
!output_type=VTK
!END
```

 
- [!VERSION](../../keyword_reference/cnt/version.html): 파일 형식의 버전을 지정
- [!SOLUTION](../../keyword_reference/cnt/solution.html): 해석 종류를 지정
- [!WRITE,RESULT](../../keyword_reference/cnt/write.html): 결과 데이터 출력을 지정
- [!WRITE,VISUAL](../../keyword_reference/cnt/write.html): 시각화 데이터 출력을 지정
- [!BOUNDARY](../../keyword_reference/cnt/boundary.html): Dirichlet 경계조건 설정
- [!CLOAD](../../keyword_reference/cnt/cload.html): 집중하중 설정
- [!MATERIAL](../../keyword_reference/cnt/material.html): 재료 물성 지정
- [!ELASTIC](../../keyword_reference/cnt/elastic.html): 탄성 물질 정의
- [!DENSITY](../../keyword_reference/cnt/density.html): 질량 밀도 정의
- [!SOLVER](../../keyword_reference/cnt/solver.html): 솔버 제어
- [!visual](../../keyword_reference/cnt/visualization/visual.html): 시각화 기법 지정
- [!surface_num](../../keyword_reference/cnt/visualization/surface_num.html): 하나의 서피스 렌더링 내 서피스 수
- [!surface](../../keyword_reference/cnt/visualization/surface_num.html): 서피스 내용 지정
- [!output_type](../../keyword_reference/cnt/visualization/output_type.html): 시각화 파일 형식 지정
- [!END](../../keyword_reference/cnt/end.html): 해석 제어 데이터의 끝을 나타냄


---

