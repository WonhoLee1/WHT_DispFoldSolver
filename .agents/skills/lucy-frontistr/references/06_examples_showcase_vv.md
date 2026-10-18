# FrontISTR 예제집(1~17), 응용 사례집(Showcase), 검증 및 타당성 확인(V&V)

> 크롤링 출처: `https://docs.frontistr.com/ko/` (수록 페이지 수: 25개)

## 수록 문서 목차

- `examples/index.md` — 예제 모음
- `examples/tutorial_01.md` — 선형 정적 해석 (탄성)
- `examples/tutorial_02.md` — 선형 정적 해석 (탄성, 병렬)
- `examples/tutorial_03.md` — 비선형 정적 해석 (초탄성 1)
- `examples/tutorial_04.md` — 비선형 정적 해석 (초탄성, 2부)
- `examples/tutorial_05.md` — 비선형 정적 해석 (탄소성 1)
- `examples/tutorial_06.md` — 비선형 정적 해석(탄소성, 2부)
- `examples/tutorial_07.md` — 비선형 정적 해석 (점탄성)
- `examples/tutorial_08.md` — 정적 해석(크리프)
- `examples/tutorial_09.md` — 접촉 해석 (1부)
- `examples/tutorial_10.md` — 접촉 해석(2부)
- `examples/tutorial_11.md` — 접촉 해석 (3부)
- `examples/tutorial_12.md` — 선형 동적 해석
- `examples/tutorial_13.md` — 비선형 동적 해석
- `examples/tutorial_14.md` — 비선형 접촉 동적 해석
- `examples/tutorial_15.md` — 모드 해석
- `examples/tutorial_16.md` — 열전도 해석
- `examples/tutorial_17.md` — 주파수 응답 해석
- `showcase/index.md` — 사례집
- `showcase/mold.md` — 대규모 금형 모델의 정적 해석
- `verification_and_validation/01_simple.md` — 단순 형상 모델에 의한 검증
- `verification_and_validation/02_static.md` — 탄성 정적 해석용 실제 모델 예제
- `verification_and_validation/03_dynamic.md` — 선형 동적 해석용 실제 모델 예제
- `verification_and_validation/04_eigen.md` — 모드 해석용 실제 모델 예제
- `verification_and_validation/05_heat.md` — 열전도 해석의 실모델 예

---


<!-- ========================================== -->
<!-- FILE: examples/index.md | TITLE: 예제 모음 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/index.html -->

# 예제 모음

 
- 튜토리얼:
  - [정적 해석 (탄성)](tutorial_01.html)
  - [정적 해석 (탄성, 병렬)](tutorial_02.html)
  - [정적 해석 (초탄성 1)](tutorial_03.html)
  - [정적 해석 (초탄성 2)](tutorial_04.html)
  - [정적 해석 (탄소성 1)](tutorial_05.html)
  - [정적 해석 (탄소성 2)](tutorial_06.html)
  - [정적 해석 (점탄성)](tutorial_07.html)
  - [정적 해석 (크리프)](tutorial_08.html)
  - [접촉 해석 (1)](tutorial_09.html)
  - [접촉 해석 (2)](tutorial_10.html)
  - [접촉 해석 (3)](tutorial_11.html)
  - [선형 동적 해석](tutorial_12.html)
  - [비선형 동적 해석](tutorial_13.html)
  - [비선형 접촉 동적 해석](tutorial_14.html)
  - [모드 해석](tutorial_15.html)
  - [열전도 해석](tutorial_16.html)
  - [주파수 응답 해석](tutorial_17.html)


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_01.md | TITLE: 선형 정적 해석 (탄성) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_01.html -->

# 선형 정적 해석 (탄성)

 이 해석에는 `tutorial/01_elastic_hinge`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 힌지 부품이며, 형상은 그림 4.1.1에, 메시 데이터는 그림 4.1.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 선형 정적 해석 | !SOLUTION,TYPE=STATIC |  |
| 절점 수 | 84,056 |  |  |
| 요소 수 | 49,871 |  |  |
| 요소 유형 | 10절점 2차 사면체 요소 | !ELEMENT,TYPE=342 | [요소 라이브러리](../functions_reference/02_element.html) |
| 재료명 | STEEL | !MATERIAL,NAME=STEEL | [재료 데이터](../functions_reference/03_material.html) |
| 재료 특성 | ELASTIC | !ELASTIC |  |
| 경계 조건 | 구속, 집중 하중 |  |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 ![힌지 부품의 형상](/examples/image/tutorial01_01.png)

  그림 4.1.1 힌지 부품의 형상  ![힌지 부품의 메시 데이터](/examples/image/tutorial01_02.png)

  그림 4.1.2 힌지 부품의 메시 데이터  
## 해석 내용

 FrontISTR 소스 코드를 압축 해제하고 이 예제의 디렉터리로 이동하여 해석에 필요한 다음 파일이

 
| 파일명 | 종류 | 역할 |
| --- | --- | --- |
| `hecmw_ctrl.dat` | 전체 제어 데이터 | 메시 데이터와 해석 제어 데이터의 입출력 파일을 지정합니다 |
| `hinge.cnt` | 해석 제어 데이터 | 해석 종류, 변위 경계 조건, 집중 하중 등을 정의하고 솔버와 시각화 도구의 제어도 지정합니다 |
| `hinge.msh` | 메시 데이터 | 유한 요소 메시를 정의하고 그 재료 데이터와 단면 데이터를 정의합니다 |

 있는지 확인하십시오.

 
```text
tar xvf FrontISTR.tar.gz
cd FrontISTR/tutorial/01_elastic_hinge
ls
```

 실행 결과:

 
```text
hecmw_ctrl.dat  hinge.cnt  hinge.msh
```

 그림 4.1.1에 표시된 구속면의 변위를 구속하고 하중면에 집중 하중을 가하는 응력 해석을 수행합니다. 전체 제어 데이터와 해석 제어 데이터를 아래에 나타냅니다.

 
### 전체 제어 데이터 `hecmw_ctrl.dat`

 메시 데이터와 해석 제어 데이터의 입출력 파일을 지정합니다.

 
```text
#
# for solver
#
!MESH, NAME=fstrMSH, TYPE=HECMW-ENTIRE # 단일 메시 데이터 지정
 hinge.msh
!CONTROL, NAME=fstrCNT                 # 해석 제어 데이터 지정
 hinge.cnt
!RESULT, NAME=fstrRES, IO=OUT          # 결과 데이터 지정
 hinge.res
!RESULT, NAME=vis_out, IO=OUT          # 시각화 데이터 지정
 hinge_vis
```

 
### 해석 제어 데이터 `hinge.cnt`

 해석 종류, 변위 경계 조건, 집중 하중 등을 정의하고 솔버와 시각화 도구의 제어도 지정합니다.

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION                   # 파일 형식 버전 지정
 3
!SOLUTION, TYPE=STATIC     # 해석 종류 지정
!WRITE,RESULT              # 결과 데이터 출력 지정
!WRITE,VISUAL              # 시각화 데이터 출력 지정
## Solver Control
### Boundary Condition
!BOUNDARY
 BND0, 1, 3, 0.000000      # 구속면 1 지정
!BOUNDARY
 BND1, 1, 3, 0.000000      # 구속면 2 지정
!CLOAD
 CL0, 1, 0.01000           # 하중면 지정
### Material
!MATERIAL, NAME=STEEL      # 재료 물성 지정
!ELASTIC                   # 탄성 재료 정의
 210000.0, 0.3
!DENSITY                   # 질량 밀도 정의
 7.85e-6
### Solver Setting
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=YES,TIMELOG=YES  # 솔버 제어
 10000, 1
 1.0e-08, 1.0, 0.0
## Post Control
!VISUAL,method=PSR          # 시각화 방법 지정
!surface_num=1              # 하나의 서피스 렌더링 내 서피스 수
!surface 1                  # 서피스 내용 지정
!output_type=VTK            # 시각화 파일 형식 지정
!END                        # 해석 제어 데이터의 끝을 나타냄
```

 
### 메시 데이터

 유한 요소 메시를 정의하고 그 재료 데이터와 단면 데이터를 정의합니다.

 (일부만 발췌)

 
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

 
## 해석 절차

 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
fistr1 -t 4
```

 (4개 스레드로 실행)

 
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

 `*.res.*`는 결과 데이터이며 FrontISTR의 해석 결과가 저장되어 있습니다. REVOCAP_PrePost 등으로 표시할 수 있습니다.

 `*_vis_*`는 시각화 데이터이며 범용 시각화 소프트웨어로 표시할 수 있습니다. 이 예제에서는 VTK 형식으로 출력하므로 ParaView 등을 사용하여 표시합니다.

 REVOCAP_PrePost로 von Mises 응력 등고선도를 작성하여 그림 4.1.3에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![von Mises 응력 해석 결과](/examples/image/tutorial01_03.png)

  그림 4.1.3 von Mises 응력 해석 결과  
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
<!-- FILE: examples/tutorial_02.md | TITLE: 선형 정적 해석 (탄성, 병렬) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_02.html -->

# 선형 정적 해석 (탄성, 병렬)

 정적 해석(탄성)을 4병렬로 수행하려면 `tutorial/02_elastic_hinge_parallel`의 데이터를 사용합니다.

 
## 해석 대상

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 선형 정적 해석 | !SOLUTION,TYPE=STATIC |  |
| 절점 수 | 84,056 |  |  |
| 요소 수 | 49,871 |  |  |
| 요소 유형 | 10절점 2차 사면체 요소 | !ELEMENT,TYPE=342 | [요소 라이브러리](../functions_reference/02_element.html) |
| 재료명 | STEEL | !MATERIAL,NAME=STEEL | [재료 데이터](../functions_reference/03_material.html) |
| 재료 특성 | ELASTIC | !ELASTIC |  |
| 경계 조건 | 구속, 집중 하중 |  |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 ![각 절점의 계산 영역](/examples/image/tutorial02_01.png)

  그림 4.2.1 각 절점의 계산 영역  
## 해석 내용

 FrontISTR 코드를 압축 해제하고 이 예제의 디렉터리로 이동한 뒤, 해석에 필요한 파일이

 
| 파일명 | 종류 | 역할 |
| --- | --- | --- |
| hecmw_ctrl.dat | 전체 제어 데이터 |  |
| hinge.cnt | 해석 제어 데이터 |  |
| hinge.msh | 메시 데이터 |  |
| hecmw_part_ctrl.dat | 메시 분할 제어 데이터 | `hecmw_part1`로 메시 데이터를 분할할 때의 제어 데이터 |

 있는지 확인하십시오.

 
```text
tar xvf FrontISTR.tar.gz
cd FrontISTR/tutorial/02_elastic_hinge_parallel
ls
```

 실행 결과:

 
```text
hecmw_ctrl.dat  hecmw_part_ctrl.dat  hinge.cnt  hinge.msh
```

 구속면의 변위를 구속하고 하중면에 집중 하중을 가하는 응력 해석을 수행합니다.

 전체 제어 데이터, 해석 제어 데이터 및 메시 분할 제어 데이터를 아래에 나타냅니다.

 
### 전체 제어 데이터 `hecmw_ctrl.dat`

 
```text
#
# 분할기용
#
!MESH, NAME=part_in,TYPE=HECMW-ENTIRE # hecmw_part1로 분할할 원본 메시 데이터
 hinge.msh
!MESH, NAME=part_out,TYPE=HECMW-DIST  # hecmw_part1로 분할한 후의 파일명
 hinge_4
#
# 솔버용
#
!MESH, NAME=fstrMSH, TYPE=HECMW-DIST  # 분할된 메시 데이터를 지정
 hinge_4
!CONTROL, NAME=fstrCNT                # 해석 제어 데이터를 지정
 hinge.cnt
!RESULT, NAME=fstrRES, IO=OUT         # 결과 데이터를 지정
 hinge.res
!RESULT, NAME=vis_out, IO=OUT         # 시각화 데이터를 지정
 hinge_vis
```

 
### 해석 제어 데이터 `hinge.cnt`

 
```text
#  FISTR 제어 파일
## 해석 제어
!VERSION                   # 파일 형식 버전을 지정
 3
!SOLUTION, TYPE=STATIC     # 해석 종류를 지정
!WRITE,RESULT              # 결과 데이터 출력을 지정
!WRITE,VISUAL              # 시각화 데이터 출력을 지정
## 솔버 제어
### 경계 조건
!BOUNDARY
 BND0, 1, 3, 0.000000      # 구속면 1을 지정
!BOUNDARY
 BND1, 1, 3, 0.000000      # 구속면 2를 지정
!CLOAD
 CL0, 1, 0.01000           # 하중면을 지정
### 재료
!MATERIAL, NAME=STEEL      # 재료 물성을 지정
!ELASTIC                   # 탄성 재료를 정의
 210000.0, 0.3
!DENSITY                   # 질량 밀도를 정의
 7.85e-6
### 솔버 설정
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=YES,TIMELOG=YES  # 솔버 제어
 10000, 2
 1.0e-08, 1.0, 0.0
## 후처리 제어
!VISUAL,method=PSR         # 시각화 방법을 지정
!surface_num=1             # 하나의 표면 렌더링에 포함되는 표면 수
!surface 1                 # 표면 내용을 지정
!output_type=VTK           # 시각화 파일 형식을 지정
!END                       # 해석 제어 데이터의 끝을 나타냄
```

 
### 메시 분할 제어 데이터 `hecmw_part_ctrl.dat`

 
```text
!PARTITION,TYPE=NODE-BASED,METHOD=PMETIS,DOMAIN=4,UCD=part.inp
```

 
## 해석 절차

 MPI로 FrontISTR를 실행하기 위해 먼저 메시 데이터 `hinge.msh`를 4개 영역으로 분할합니다.

 
```text
hecmw_part1
```

 실행 결과:

 
```text
Oct 07 11:04:52 Info: Reading mesh file...
Oct 07 11:04:52 Info: Starting domain decomposition...
Oct 07 11:04:52 TH(0/8) Info: Creating local mesh for domain #0 ...
Oct 07 11:04:52 TH(2/8) Info: Creating local mesh for domain #1 ...
Oct 07 11:04:52 TH(6/8) Info: Creating local mesh for domain #2 ...
Oct 07 11:04:52 TH(7/8) Info: Creating local mesh for domain #3 ...
Oct 07 11:04:52 Info: Domain decomposition done
```

 새로 hinge_4.x와 part.inp라는 파일이 생성됩니다.

 
```text
ls
```

 실행 결과:

 
```text
hecmw_ctrl.dat  hecmw_part_ctrl.dat  hinge.msh  hinge_4.1  hinge_4.3
hecmw_part.log  hinge.cnt            hinge_4.0  hinge_4.2  part.inp
```

 이어서 MPI를 사용하여 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
mpirun -np 4 fistr1 -t 1
```

 (MPI 4병렬, OpenMP 스레드 1개로 실행)

 
```text
##################################################################
#                         FrontISTR                              #
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
  date:       2020-10-07T11:07:21+0900
  processes:  4
  threads:    1
  cores:      4
  host:
    0: flow-p06
    1: flow-p06
    2: flow-p06
    3: flow-p06
---
...
 Step control not defined! Using default step=1
 fstr_setup: OK
 Start visualize PSF 1 at timestep 0

 loading step=    1
 sub_step= 1,   current_time=  0.0000E+00, time_inc=  0.1000E+01
 loading_factor=    0.0000000   1.0000000
### 3x3 BLOCK CG, SSOR, 2
      1    2.183567E+00
      2    2.423900E+00
      3    2.939117E+00
...
   2084    1.158654E-08
   2085    1.032414E-08
   2086    9.436273E-09
### Relative residual = 9.43589E-09

### summary of linear solver
      2086 iterations      9.435886E-09
    set-up time      :     4.695220E-02
    solver time      :     7.103976E+01
    solver/comm time :     1.929294E+01
    solver/matvec    :     1.544405E+01
    solver/precond   :     3.243278E+01
    solver/1 iter    :     3.405549E-02
    work ratio (%)   :     7.284205E+01

 Start visualize PSF 1 at timestep 1
### FSTR_SOLVE_NLGEOM FINISHED!

 ====================================
    TOTAL TIME (sec) :     72.42
           pre (sec) :      0.29
         solve (sec) :     72.13
 ====================================
 FrontISTR Completed !!
```

 `FrontISTR Completed !!`가 표시되면 해석이 완료된 것입니다.

 
## 해석 결과

 해석이 완료되면 여러 파일이 새로 생성됩니다.

 
```text
ls
```

 실행 결과:

 
```text
0.log       FSTR.dbg.3           hinge.cnt      hinge.res.2.1  hinge_vis_psf.0000
1.log       FSTR.msg             hinge.msh      hinge.res.3.0  hinge_vis_psf.0000.pvtu
2.log       FSTR.sta             hinge.res.0.0  hinge.res.3.1  hinge_vis_psf.0001
3.log       hecmw_ctrl.dat       hinge.res.0.1  hinge_4.0      hinge_vis_psf.0001.pvtu
FSTR.dbg.0  hecmw_part.log       hinge.res.1.0  hinge_4.1      part.inp
FSTR.dbg.1  hecmw_part_ctrl.dat  hinge.res.1.1  hinge_4.2
FSTR.dbg.2  hecmw_vis.ini        hinge.res.2.0  hinge_4.3
```

 `*.res.*`는 결과 데이터라고 하며 FrontISTR의 각 MPI 노드별 해석 결과가 저장됩니다. REVOCAP_PrePost 등으로 표시할 수 있습니다.

 `*_vis_*`는 시각화 데이터라고 하며 범용 시각화 소프트웨어로 표시할 수 있습니다. 이번 예에서는 VTK 형식으로 출력하므로 ParaView 등을 사용해 표시합니다.

 ![미제스 응력 해석 결과(VTK로 표시)](/examples/image/tutorial02_02.png)

  그림 4.2.2 미제스 응력 해석 결과(VTK로 표시)  
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
 //U1    0.0000E+00        14  0.0000E+00        14
 //U2    0.0000E+00        14  0.0000E+00        14
 //U3    0.0000E+00        14  0.0000E+00        14
 //E11   0.0000E+00        14  0.0000E+00        14
 //E22   0.0000E+00        14  0.0000E+00        14
 //E33   0.0000E+00        14  0.0000E+00        14
 //E12   0.0000E+00        14  0.0000E+00        14
 //E23   0.0000E+00        14  0.0000E+00        14
 //E31   0.0000E+00        14  0.0000E+00        14
 //S11   0.0000E+00        14  0.0000E+00        14
 //S22   0.0000E+00        14  0.0000E+00        14
 //S33   0.0000E+00        14  0.0000E+00        14
 //S12   0.0000E+00        14  0.0000E+00        14
 //S23   0.0000E+00        14  0.0000E+00        14
 //S31   0.0000E+00        14  0.0000E+00        14
 //SMS   0.0000E+00        14  0.0000E+00        14
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   0.0000E+00        14  0.0000E+00        14
 //E22   0.0000E+00        14  0.0000E+00        14
 //E33   0.0000E+00        14  0.0000E+00        14
 //E12   0.0000E+00        14  0.0000E+00        14
 //E23   0.0000E+00        14  0.0000E+00        14
 //E31   0.0000E+00        14  0.0000E+00        14
 //S11   0.0000E+00        14  0.0000E+00        14
 //S22   0.0000E+00        14  0.0000E+00        14
 //S33   0.0000E+00        14  0.0000E+00        14
 //S12   0.0000E+00        14  0.0000E+00        14
 //S23   0.0000E+00        14  0.0000E+00        14
 //S31   0.0000E+00        14  0.0000E+00        14
 //SMS   0.0000E+00        14  0.0000E+00        14
#### Result step=     1
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    7.4906E-03     50978 -6.9368E-04     51989
 //U2    6.1150E-05      2426 -1.2781E-04     40202
 //U3    1.8720E-04      2712 -5.5024E-03     67858
 //E11   1.3024E-03     28591 -1.2844E-03     28580
 //E22   2.1102E-04     24593 -2.0594E-04       205
 //E33   6.8634E-04     51036 -5.9603E-04     67794
 //E12   5.2186E-04       209 -5.4559E-04     28641
 //E23   2.5417E-04     28521 -3.2956E-04       187
 //E31   7.2396E-04     36168 -9.5847E-04     28591
 //S11   3.6844E+02     28591 -3.6387E+02     28580
 //S22   1.5841E+02     28591 -1.5743E+02     28580
 //S33   1.5959E+02     51036 -1.5643E+02     28580
 //S12   4.2150E+01       209 -4.4067E+01     28641
 //S23   2.0529E+01     28521 -2.6618E+01       187
 //S31   5.8474E+01     36168 -7.7415E+01     28591
 //SMS   2.5134E+02     28591  3.5157E-02     12726
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   1.0115E-03      9959 -9.7870E-04      9942
 //E22   1.5457E-04     13694 -1.4639E-04      7189
 //E33   5.9415E-04     44563 -5.0497E-04     47965
 //E12   3.5513E-04      9579 -3.9083E-04      9600
 //E23   2.0608E-04      9563 -2.3480E-04      9589
 //E31   5.7633E-04     43142 -4.8019E-04      9571
 //S11   2.6038E+02      9573 -2.5605E+02      9571
 //S22   1.0144E+02      9573 -1.0100E+02      9571
 //S33   1.3921E+02     44569 -1.1431E+02     47974
 //S12   2.8683E+01      9579 -3.1567E+01      9600
 //S23   1.6645E+01      9563 -1.8964E+01      9589
 //S31   4.6550E+01     43142 -3.8785E+01      9571
 //SMS   2.0350E+02      9959  3.4602E-02        28
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
<!-- FILE: examples/tutorial_03.md | TITLE: 비선형 정적 해석 (초탄성 1) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_03.html -->

# 비선형 정적 해석 (초탄성 1)

 본 해석에는 `tutorial/03_hyperelastic_cylinder`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 원형 봉의 1/8 모델이며, 형상은 그림 4.3.1에, 메시 데이터는 그림 4.3.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 정적 해석(초탄성) | !SOLUTION,TYPE=NLSTATIC |  |
| 절점 수 | 629 |  |  |
| 요소 수 | 432 |  |  |
| 요소 유형 | 8절점 1차 육면체 요소 | !ELEMENT,TYPE=361 |  |
| 재료명 | MAT1 | !MATERIAL,NAME=MAT1 |  |
| 재료 특성 | HYPERELASTIC | !HYPERELASTIC,TYPE=MOONEY-RIVLIN |  |
| 경계 조건 | 구속, 강제 변위 |  |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 ![원형 봉(1/8 모델)의 형상](/examples/image/tutorial03_01.png)

  그림 4.3.1 원형 봉(1/8 모델)의 형상  ![원형 봉(1/8 모델)의 메시 데이터](/examples/image/tutorial03_02.png)

  그림 4.3.2 원형 봉(1/8 모델)의 메시 데이터  
## 해석 내용

 원형 봉에 축 방향 인장 변위를 부여하는 응력 해석을 수행합니다. 초탄성 재료 구성식에는 Mooney-Rivlin 모델을 사용합니다. 해석 제어 데이터를 아래에 나타냅니다.

 
### 해석 제어 데이터 `cylinder.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!SOLUTION, TYPE=NLSTATIC
!WRITE,RESULT
!WRITE,VISUAL
## Solver Control
### Boundary Condition
!BOUNDARY, GRPID=1
 LOADS, 3, 3, -7.0
 FIX,   3, 3, 0.0
 XSYMM, 1, 1, 0.0
 YSYMM, 2, 2, 0.0
### STEP
!STEP, SUBSTEPS=5, CONVERG=1.0e-5
 BOUNDARY, 1
### Material
!MATERIAL, NAME=MAT1
!HYPERELASTIC, TYPE=MOONEY-RIVLIN
 0.1486, 0.4849, 0.0789
### Solver Setting
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=YES,TIMELOG=YES
 10000, 1
 1.0e-8, 1.0, 0.0
## Post Control
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 결과

 5번째 서브스텝의 해석 결과에 대해, REVOCAP_PrePost로 미제스 응력 등고선을 추가한 변형도를 작성하여 그림 4.3.3에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![변형 및 미제스 응력 해석 결과](/examples/image/tutorial03_03.png)

  그림 4.3.3 변형 및 미제스 응력 해석 결과  
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

 ...

#### Result step=     5
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1 -6.7543E-01        90
 //U2    0.0000E+00         1 -6.7543E-01        89
 //U3    0.0000E+00         1 -7.0000E+00        38
 //E11  -1.0781E-01        38 -1.1448E-01         7
 //E22  -1.0781E-01        50 -1.1448E-01        13
 //E33   2.3916E-01         7  2.2715E-01        50
 //E12   8.9609E-04        53 -9.2501E-04        10
 //E23   1.0074E-07        38 -2.6232E-03        86
 //E31   1.0074E-07        50 -2.6232E-03        93
 //S11   2.7292E-03        14 -2.6468E-03        49
 //S22   2.7292E-03         6 -2.6468E-03        39
 //S33   8.5422E-01         7  8.1021E-01        50
 //S12   1.2861E-03        53 -1.3373E-03        10
 //S23   1.4289E-07        38 -3.1695E-03        86
 //S31   1.4289E-07        50 -3.1695E-03        93
 //SMS   8.5307E-01         7  8.1134E-01        50
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11  -1.0817E-01       759 -1.1410E-01       354
 //E22  -1.0817E-01       768 -1.1410E-01       363
 //E33   2.3863E-01       354  2.2766E-01       759
 //E12   7.4078E-04       766 -7.5734E-04       360
 //E23  -2.5376E-05       753 -2.4790E-03       471
 //E31  -2.5376E-05       749 -2.4790E-03       462
 //S11   2.8994E-03       363 -2.8080E-03       768
 //S22   2.8994E-03       354 -2.8080E-03       759
 //S33   8.5295E-01       354  8.1143E-01       759
 //S12   1.0634E-03       766 -1.0947E-03       360
 //S23  -3.0682E-05       753 -2.9954E-03       471
 //S31  -3.0682E-05       749 -2.9954E-03       462
 //SMS   8.5118E-01       354  8.1315E-01       759
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1 -6.7543E-01        90
 //U2    0.0000E+00         1 -6.7543E-01        89
 //U3    0.0000E+00         1 -7.0000E+00        38
 //E11  -1.0781E-01        38 -1.1448E-01         7
 //E22  -1.0781E-01        50 -1.1448E-01        13
 //E33   2.3916E-01         7  2.2715E-01        50
 //E12   8.9609E-04        53 -9.2501E-04        10
 //E23   1.0074E-07        38 -2.6232E-03        86
 //E31   1.0074E-07        50 -2.6232E-03        93
 //S11   2.7292E-03        14 -2.6468E-03        49
 //S22   2.7292E-03         6 -2.6468E-03        39
 //S33   8.5422E-01         7  8.1021E-01        50
 //S12   1.2861E-03        53 -1.3373E-03        10
 //S23   1.4289E-07        38 -3.1695E-03        86
 //S31   1.4289E-07        50 -3.1695E-03        93
 //SMS   8.5307E-01         7  8.1134E-01        50
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11  -1.0817E-01       759 -1.1410E-01       354
 //E22  -1.0817E-01       768 -1.1410E-01       363
 //E33   2.3863E-01       354  2.2766E-01       759
 //E12   7.4078E-04       766 -7.5734E-04       360
 //E23  -2.5376E-05       753 -2.4790E-03       471
 //E31  -2.5376E-05       749 -2.4790E-03       462
 //S11   2.8994E-03       363 -2.8080E-03       768
 //S22   2.8994E-03       354 -2.8080E-03       759
 //S33   8.5295E-01       354  8.1143E-01       759
 //S12   1.0634E-03       766 -1.0947E-03       360
 //S23  -3.0682E-05       753 -2.9954E-03       471
 //S31  -3.0682E-05       749 -2.9954E-03       462
 //SMS   8.5118E-01       354  8.1315E-01       759
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_04.md | TITLE: 비선형 정적 해석 (초탄성, 2부) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_04.html -->

# 비선형 정적 해석 (초탄성, 2부)

 이 해석에는 `tutorial/04_hyperelastic_spring`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 스프링이며, 형상은 그림 4.4.1에, 메시 데이터는 그림 4.4.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 정적 해석(초탄성) | !SOLUTION,TYPE=NLSTATIC |  |
| 절점 수 | 78,771 |  |  |
| 요소 수 | 46,454 |  |  |
| 요소 유형 | 10절점 2차 사면체 요소 | !ELEMENT,TYPE=342 |  |
| 재료명 | MAT1 | !MATERIAL,NAME=MAT1 |  |
| 재료 특성 | HYPERELASTIC | !HYPERELASTIC,TYPE=ARRUDA-BOYCE |  |
| 경계 조건 | 구속/강제 변위 |  |  |
| 행렬 해법 | CG/AMG | !SOLVER,METHOD=CG,PRECOND=5 |  |

 ![스프링 형상](/examples/image/tutorial04_01.png)

  그림 4.4.1 스프링 형상  ![스프링 메시 데이터](/examples/image/tutorial04_02.png)

  그림 4.4.2 스프링 메시 데이터  
## 해석 내용

 그림 4.4.1에 표시된 구속면의 변위를 구속하고 변위 부여면에 변위를 부여하는 응력 해석을 수행합니다. 초탄성 재료의 구성 모델에는 Arruda-Boyce 모델을 사용합니다. 해석 제어 데이터는 다음과 같습니다.

 
### 해석 제어 데이터 `spring.cnt`

 
```text
#  FISTR 제어 파일
## 해석 제어
!VERSION
 3
!SOLUTION, TYPE=NLSTATIC
!WRITE,RESULT
!WRITE,VISUAL
## 솔버 제어
### 경계 조건
!BOUNDARY, GRPID=1
 LOADS, 2, 2, -5.0
 FIX,   1, 3, 0.0
### 스텝
!STEP, SUBSTEPS=1, CONVERG=1.0e-5
 BOUNDARY, 1
### 재료
!MATERIAL, NAME=MAT1
!HYPERELASTIC, TYPE=ARRUDA-BOYCE
 0.71, 1.7029, 0.1408
### 솔버 설정
!SOLVER,METHOD=CG,PRECOND=5,ITERLOG=NO,TIMELOG=YES
 10000, 1
 1.0e-5, 1.0, 0.0
## 후처리 제어
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
cd FrontISTR/tutorial/04_hyperelastic_spring
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 변위 등고선을 추가한 변형도를 REVOCAP_PrePost로 작성하여 그림 4.4.3에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![변형 및 변위 해석 결과](/examples/image/tutorial04_03.png)

  그림 4.4.3 변형 및 변위 해석 결과  
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
 //U1    2.6070E-01     42179 -2.4651E-01     44163
 //U2    2.0567E-02     14512 -5.0036E+00     44753
 //U3    6.6559E-02     15443 -6.6862E-01     44664
 //E11   4.1026E-03      1015 -4.5809E-03      3429
 //E22   2.3030E-03     42626 -1.5152E-03     44761
 //E33   4.1949E-03     23553 -5.0169E-03     27938
 //E12   1.3348E-02     55149 -1.3381E-02     56829
 //E23   2.7881E-02     48353 -1.9534E-02     48322
 //E31   1.0861E-02     47938 -8.8803E-03      9493
 //S11   8.4247E-03      8264 -9.1017E-03      8501
 //S22   1.0769E-02     56772 -6.3921E-03     56787
 //S33   1.3881E-02     56764 -1.2778E-02     42632
 //S12   1.2396E-02     55149 -1.2434E-02     56829
 //S23   2.5880E-02     48353 -1.8138E-02     48322
 //S31   1.0081E-02     47938 -8.2435E-03      9493
 //SMS   4.9638E-02     48353  2.8482E-04     66578
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   3.9156E-03     34074 -4.3403E-03     16932
 //E22   1.8391E-03     23282 -1.2721E-03     56220
 //E33   3.9490E-03     45925 -4.4530E-03     34145
 //E12   1.2578E-02     45680 -1.1729E-02     19492
 //E23   2.0534E-02     40739 -1.3512E-02     24172
 //E31   9.4162E-03     32786 -7.6149E-03     44971
 //S11   7.7882E-03     25880 -8.2932E-03     45925
 //S22   6.8884E-03     23275 -6.1373E-03     49821
 //S33   9.9242E-03     59974 -9.5475E-03     31626
 //S12   1.1679E-02     45680 -1.0893E-02     19492
 //S23   1.9071E-02     40739 -1.2534E-02     24172
 //S31   8.7376E-03     32786 -7.0660E-03     44971
 //SMS   3.5586E-02     40739  4.1161E-04     40537
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    2.6070E-01     42179 -2.4651E-01     44163
 //U2    2.0567E-02     14512 -5.0036E+00     44753
 //U3    6.6559E-02     15443 -6.6862E-01     44664
 //E11   4.1026E-03      1015 -4.5809E-03      3429
 //E22   2.3030E-03     42626 -1.5152E-03     44761
 //E33   4.1949E-03     23553 -5.0169E-03     27938
 //E12   1.3348E-02     55149 -1.3381E-02     56829
 //E23   2.7881E-02     48353 -1.9534E-02     48322
 //E31   1.0861E-02     47938 -8.8803E-03      9493
 //S11   8.4247E-03      8264 -9.1017E-03      8501
 //S22   1.0769E-02     56772 -6.3921E-03     56787
 //S33   1.3881E-02     56764 -1.2778E-02     42632
 //S12   1.2396E-02     55149 -1.2434E-02     56829
 //S23   2.5880E-02     48353 -1.8138E-02     48322
 //S31   1.0081E-02     47938 -8.2435E-03      9493
 //SMS   4.9638E-02     48353  2.8482E-04     66578
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   3.9156E-03     34074 -4.3403E-03     16932
 //E22   1.8391E-03     23282 -1.2721E-03     56220
 //E33   3.9490E-03     45925 -4.4530E-03     34145
 //E12   1.2578E-02     45680 -1.1729E-02     19492
 //E23   2.0534E-02     40739 -1.3512E-02     24172
 //E31   9.4162E-03     32786 -7.6149E-03     44971
 //S11   7.7882E-03     25880 -8.2932E-03     45925
 //S22   6.8884E-03     23275 -6.1373E-03     49821
 //S33   9.9242E-03     59974 -9.5475E-03     31626
 //S12   1.1679E-02     45680 -1.0893E-02     19492
 //S23   1.9071E-02     40739 -1.2534E-02     24172
 //S31   8.7376E-03     32786 -7.0660E-03     44971
 //SMS   3.5586E-02     40739  4.1161E-04     40537
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_05.md | TITLE: 비선형 정적 해석 (탄소성 1) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_05.html -->

# 비선형 정적 해석 (탄소성 1)

 본 해석에는 `tutorial/05_plastic_cylinder`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 [비선형 정적 해석(초탄성 1)](tutorial_03.html)과 동일한 원형 봉 1/8 모델입니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 정적 해석(탄소성) | !SOLUTION,TYPE=NLSTATIC |  |
| 절점 수 | 629 |  |  |
| 요소 수 | 432 |  |  |
| 요소 유형 | 8절점 1차 육면체 요소 | !ELEMENT,TYPE=361 |  |
| 재료명 | MAT1 | !MATERIAL,NAME=MAT1 |  |
| 재료 특성 | ELASTIC,PLASTIC | !ELASTIC !PLASTIC |  |
| 경계 조건 | 구속, 강제 변위 |  |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 
## 해석 내용

 소성 변형에 의한 원형 봉의 네킹 현상을 해석합니다. 항복 함수에는 미제스 모델을 사용합니다. 해석 제어 데이터를 아래에 나타냅니다.

 
### 해석 제어 데이터 `necking.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!SOLUTION, TYPE=NLSTATIC
!WRITE,RESULT,FREQUENCY=10
!WRITE,VISUAL,FREQUENCY=10
## Solver Control
### Boundary Condition
!BOUNDARY, GRPID=1
 LOADS, 3, 3, -7.0
 FIX,   3, 3, 0.0
 XSYMM, 1, 1, 0.0
 YSYMM, 2, 2, 0.0
### STEP
!STEP, SUBSTEPS=40, CONVERG=1.0e-3
 BOUNDARY, 1
### Material
!MATERIAL, NAME=MAT1
!ELASTIC
 206900.0, 0.29
!PLASTIC, YIELD=MISES, HARDEN=MULTILINEAR
 450.0, 0.0
 608.0, 0.05
 679.0, 0.1
 732.0, 0.2
 752.0, 0.3
 766.0, 0.4
 780.0, 0.5
### Output
!OUTPUT_VIS
 NSTRAIN, ON
!OUTPUT_RES
 ISTRESS, ON
### Solver Setting
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=NO,TIMELOG=YES
 2000, 1
 1.0e-8, 1.0, 0.0
## Post Control
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
cd FrontISTR/tutorial/05_plastic_cylinder
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 35번째 서브스텝의 해석 결과에 대해, REVOCAP_PrePost로 미제스 응력 등고선을 추가한 변형도를 작성하여 그림 4.5.1에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![변형 및 미제스 응력 해석 결과](/examples/image/tutorial05_01.png)

  그림 4.5.1 변형 및 미제스 응력 해석 결과  
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

 ...

#### Result step=    40
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1 -3.4606E+00         7
 //U2    0.0000E+00         1 -3.4606E+00        13
 //U3    0.0000E+00         1 -7.0000E+00        38
 //E11  -3.8922E-02        38 -6.4666E-01        14
 //E22  -3.8922E-02        50 -6.4666E-01         6
 //E33   1.2914E+00         1  7.8682E-02        44
 //E12   2.5910E-01        10 -1.1813E-02       387
 //E23   9.7554E-03        91 -1.1753E+00        13
 //E31   9.7554E-03        88 -1.1753E+00         7
 //S11   2.6632E+02         1 -3.6916E+02       192
 //S22   2.6632E+02         1 -3.6916E+02       192
 //S33   1.0471E+03         1 -4.1469E+01       191
 //S12   7.2338E-01        53 -1.3479E+02       387
 //S23   5.9172E+01       191 -2.5727E+02        89
 //S31   5.9172E+01       191 -2.5727E+02        90
 //SMS   7.9063E+02        13  2.4706E+02       190
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11  -3.9027E-02       750 -5.5060E-01       345
 //E22  -3.9027E-02       750 -5.5060E-01       345
 //E33   1.1045E+00       345  7.8722E-02       750
 //E12   1.7067E-01       361 -1.6134E-02       414
 //E23  -2.0595E-05       756 -9.6480E-01       363
 //E31  -2.0595E-05       748 -9.6480E-01       354
 //S11   2.6751E+02       345 -3.9312E+02       399
 //S22   2.6751E+02       345 -3.9312E+02       399
 //S33   1.0447E+03       345 -4.7069E+01       426
 //S12   1.4870E+01       361 -1.4324E+02       442
 //S23   4.0405E+01       478 -2.8018E+02       393
 //S31   4.0405E+01       486 -2.8018E+02       380
 //SMS   7.7935E+02       345  2.3268E+02       480
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1 -3.4606E+00         7
 //U2    0.0000E+00         1 -3.4606E+00        13
 //U3    0.0000E+00         1 -7.0000E+00        38
 //E11  -3.8922E-02        38 -6.4666E-01        14
 //E22  -3.8922E-02        50 -6.4666E-01         6
 //E33   1.2914E+00         1  7.8682E-02        44
 //E12   2.5910E-01        10 -1.1813E-02       387
 //E23   9.7554E-03        91 -1.1753E+00        13
 //E31   9.7554E-03        88 -1.1753E+00         7
 //S11   2.6632E+02         1 -3.6916E+02       192
 //S22   2.6632E+02         1 -3.6916E+02       192
 //S33   1.0471E+03         1 -4.1469E+01       191
 //S12   7.2338E-01        53 -1.3479E+02       387
 //S23   5.9172E+01       191 -2.5727E+02        89
 //S31   5.9172E+01       191 -2.5727E+02        90
 //SMS   7.9063E+02        13  2.4706E+02       190
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11  -3.9027E-02       750 -5.5060E-01       345
 //E22  -3.9027E-02       750 -5.5060E-01       345
 //E33   1.1045E+00       345  7.8722E-02       750
 //E12   1.7067E-01       361 -1.6134E-02       414
 //E23  -2.0595E-05       756 -9.6480E-01       363
 //E31  -2.0595E-05       748 -9.6480E-01       354
 //S11   2.6751E+02       345 -3.9312E+02       399
 //S22   2.6751E+02       345 -3.9312E+02       399
 //S33   1.0447E+03       345 -4.7069E+01       426
 //S12   1.4870E+01       361 -1.4324E+02       442
 //S23   4.0405E+01       478 -2.8018E+02       393
 //S31   4.0405E+01       486 -2.8018E+02       380
 //SMS   7.7935E+02       345  2.3268E+02       480
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_06.md | TITLE: 비선형 정적 해석(탄소성, 2부) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_06.html -->

# 비선형 정적 해석(탄소성, 2부)

 이 해석에는 `tutorial/06_plastic_can`의 데이터를 사용합니다.

 
## 해석 대상

 해석 모델은 용기의 1/2 모델입니다. 형상은 그림 4.6.1에, 메시 데이터는 그림 4.6.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 정적 해석(탄소성) | !SOLUTION,TYPE=NLSTATIC |  |
| 절점 수 | 14,119 |  |  |
| 요소 수 | 7,236 |  |  |
| 요소 유형 | 10절점 2차 사면체 요소 | !ELEMENT,TYPE=342 |  |
| 재료명 | M1 | !MATERIAL,NAME=M1 |  |
| 재료 특성 | ELASTIC PLASTIC | !ELASTIC !PLASTIC,YIELD=DRUCKER-PRAGER |  |
| 경계 조건 | 구속, 분포 하중 | !DLOAD |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 ![용기 형상](/examples/image/tutorial06_01.png)

  그림 4.6.1 용기 형상  ![용기 메시 데이터](/examples/image/tutorial06_02.png)

  그림 4.6.2 용기 메시 데이터  
## 해석 내용

 그림 4.6.1에 표시된 구속면의 변위를 구속하고 용기 내부를 하중면으로 하여 분포 하중을 적용하는 응력 해석을 수행합니다. 항복 함수에는 Drucker-Prager 모델을 사용합니다. 해석 제어 데이터는 아래와 같습니다.

 
### 해석 제어 데이터 `can.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!SOLUTION, TYPE=NLSTATIC
!WRITE,RESULT
!WRITE,VISUAL
## Solver Control
### Boundary Condition
!BOUNDARY, GRPID=1
 BND0, 3, 3, 0.000000
!BOUNDARY, GRPID=1
 BND1, 1, 1, 0.000000
 BND1, 2, 2, 0.000000
 BND1, 3, 3, 0.000000
!DLOAD, GRPID=1
 DL0, S, 1.0
!DLOAD, GRPID=1
 DL1, S, 1.0
!DLOAD, GRPID=1
 DL2, S, 0.5
### STEP
!STEP, SUBSTEPS=10, CONVERG=1.0e-5
 BOUNDARY, 1
 LOAD, 1
### Material
!MATERIAL, NAME=M1
!ELASTIC
 24000.0, 0.2
!PLASTIC, YIELD=DRUCKER-PRAGER
 500.0, 20.0, 0.0
### Solver Setting
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=NO,TIMELOG=YES
 20000, 1
 1.0e-8, 1.0, 0.0
## Post Control
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 명령 `fistr1`을 실행합니다.

 
```text
cd FrontISTR/tutorial/06_plastic_can
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 10번째 서브스텝의 해석 결과에 대해 REVOCAP_PrePost를 사용하여 von Mises 응력 등고선을 포함한 변형 형상도를 작성하고 그림 4.6.3에 나타냅니다. 변형 배율은 30입니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![해석 결과: 변형 및 von Mises 응력](/examples/image/tutorial06_03.png)

  그림 4.6.3 해석 결과: 변형 및 von Mises 응력  
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

 ...

#### Result step=    10
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    1.6169E+00      1600 -1.6123E+00     11901
 //U2    1.9281E+01      6877 -4.5296E-01      7096
 //U3    1.6087E+00      7016 -1.5107E+00      6934
 //E11   9.9108E-04     11242 -6.5834E-04      1404
 //E22   1.5025E-03     13972 -5.4245E-04      2367
 //E33   9.8320E-04      6833 -6.4744E-04      7000
 //E12   1.6822E-03      2698 -1.7175E-03     11906
 //E23   1.7082E-03      6749 -1.4469E-03     13509
 //E31   1.2090E-03     12475 -1.1181E-03     11342
 //S11   2.7781E+01      1086 -1.9439E+01      2363
 //S22   3.7876E+01     13972 -1.4556E+01      2367
 //S33   2.7335E+01      1086 -1.9745E+01     13082
 //S12   1.6817E+01      2698 -1.7170E+01     11906
 //S23   1.7077E+01      6749 -1.4463E+01     13509
 //S31   1.2086E+01     12475 -1.1178E+01     11342
 //SMS   3.7448E+01      2834  2.7593E-04      7333
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   5.8557E-04      6613 -4.0802E-04      6416
 //E22   1.1391E-03      7144 -3.1424E-04      6613
 //E33   6.9233E-04      6130 -4.5897E-04      6222
 //E12   1.1076E-03       165 -1.1101E-03      5196
 //E23   1.4338E-03      6215 -9.2107E-04      6176
 //E31   9.6618E-04      5618 -7.4939E-04      4704
 //S11   1.7994E+01      6613 -1.2254E+01      6224
 //S22   2.7863E+01      7144 -7.3017E+00        24
 //S33   1.9725E+01      6613 -1.3385E+01      6222
 //S12   1.1072E+01       165 -1.1097E+01      5196
 //S23   1.4332E+01      6215 -9.2080E+00      6176
 //S31   9.6603E+00      5618 -7.4918E+00      4704
 //SMS   3.3151E+01      5619  3.3306E-04      2665
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    1.6169E+00      1600 -1.6123E+00     11901
 //U2    1.9281E+01      6877 -4.5296E-01      7096
 //U3    1.6087E+00      7016 -1.5107E+00      6934
 //E11   9.9108E-04     11242 -6.5834E-04      1404
 //E22   1.5025E-03     13972 -5.4245E-04      2367
 //E33   9.8320E-04      6833 -6.4744E-04      7000
 //E12   1.6822E-03      2698 -1.7175E-03     11906
 //E23   1.7082E-03      6749 -1.4469E-03     13509
 //E31   1.2090E-03     12475 -1.1181E-03     11342
 //S11   2.7781E+01      1086 -1.9439E+01      2363
 //S22   3.7876E+01     13972 -1.4556E+01      2367
 //S33   2.7335E+01      1086 -1.9745E+01     13082
 //S12   1.6817E+01      2698 -1.7170E+01     11906
 //S23   1.7077E+01      6749 -1.4463E+01     13509
 //S31   1.2086E+01     12475 -1.1178E+01     11342
 //SMS   3.7448E+01      2834  2.7593E-04      7333
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   5.8557E-04      6613 -4.0802E-04      6416
 //E22   1.1391E-03      7144 -3.1424E-04      6613
 //E33   6.9233E-04      6130 -4.5897E-04      6222
 //E12   1.1076E-03       165 -1.1101E-03      5196
 //E23   1.4338E-03      6215 -9.2107E-04      6176
 //E31   9.6618E-04      5618 -7.4939E-04      4704
 //S11   1.7994E+01      6613 -1.2254E+01      6224
 //S22   2.7863E+01      7144 -7.3017E+00        24
 //S33   1.9725E+01      6613 -1.3385E+01      6222
 //S12   1.1072E+01       165 -1.1097E+01      5196
 //S23   1.4332E+01      6215 -9.2080E+00      6176
 //S31   9.6603E+00      5618 -7.4918E+00      4704
 //SMS   3.3151E+01      5619  3.3306E-04      2665
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_07.md | TITLE: 비선형 정적 해석 (점탄성) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_07.html -->

# 비선형 정적 해석 (점탄성)

 이 해석에는 `tutorial/07_viscoelastic_cylinder`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 [비선형 정적 해석(초탄성, 1부)](tutorial_03.html)과 동일한 원형 봉의 1/8 모델입니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 정적 해석(탄소성) | !SOLUTION,TYPE=NLSTATIC |  |
| 절점 수 | 629 |  |  |
| 요소 수 | 432 |  |  |
| 요소 유형 | 8절점 1차 육면체 요소 | !ELEMENT,TYPE=361 |  |
| 재료명 | MAT1 | !MATERIAL,NAME=MAT1 |  |
| 재료 특성 | ELASTIC,VISCOELASTIC | !ELASTIC !VISCOELASTIC |  |
| 경계 조건 | 구속, 강제 변위 |  |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 
## 해석 내용

 원형 봉에 축 방향 인장 변위를 부여하여 응력 완화 해석을 수행합니다. 해석 제어 데이터는 다음과 같습니다.

 
### 해석 제어 데이터 `cylinder.cnt`

 
```text
#  FISTR 제어 파일
## 해석 제어
!VERSION
 3
!SOLUTION, TYPE=NLSTATIC
!WRITE,VISUAL
!WRITE,RESULT
## 솔버 제어
### 경계 조건
!BOUNDARY, GRPID=1
 LOADS, 3, 3, -7.0
 FIX,   3, 3, 0.0
 XSYMM, 1, 1, 0.0
 YSYMM, 2, 2, 0.0
### 스텝
!STEP, TYPE=VISCO, CONVERG=1.0e-5
 0.2, 2.0
 BOUNDARY, 1
### 재료
!MATERIAL, NAME=MAT1
!ELASTIC
 206900.0, 0.29
!VISCOELASTIC
 0.5, 1.0
### 솔버 설정
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=YES,TIMELOG=YES
 10000, 1
 1.0e-8, 1.0, 0.0
## 후처리 제어
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
cd FrontISTR/tutorial/07_viscoelastic_cylinder
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 미제스 응력 등고선을 추가한 변형도를 REVOCAP_PrePost로 작성하여 그림 4.7.1에 나타냅니다. 2초 후(10번째 스텝)의 해석 결과입니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![변형 및 미제스 응력 해석 결과](/examples/image/tutorial07_01.png)

  그림 4.7.1 변형 및 미제스 응력 해석 결과  
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

 ...

#### Result step=    10
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1 -7.4530E-01        91
 //U2    0.0000E+00         1 -7.4530E-01        88
 //U3    0.0000E+00         1 -7.0000E+00        38
 //E11  -1.2136E-01        38 -1.2719E-01         7
 //E22  -1.2136E-01        50 -1.2719E-01        13
 //E33   2.3670E-01         7  2.2951E-01        50
 //E12   1.2005E-03        53 -1.2295E-03        10
 //E23   1.2987E-04        72 -2.5210E-03        83
 //E31   1.2987E-04        64 -2.5210E-03        96
 //S11   1.3771E+02        13 -1.3366E+02        50
 //S22   1.3771E+02         7 -1.3366E+02        38
 //S33   6.3193E+04         7  5.9532E+04        50
 //S12   3.5416E+01        53 -3.5955E+01        10
 //S23   1.0704E+01        35 -2.1669E+02        83
 //S31   1.0704E+01        31 -2.1669E+02        96
 //SMS   6.3093E+04         7  5.9628E+04        50
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11  -1.2167E-01       759 -1.2687E-01       354
 //E22  -1.2167E-01       768 -1.2687E-01       363
 //E33   2.3619E-01       354  2.3001E-01       768
 //E12   1.1247E-03       765 -1.1492E-03       360
 //E23  -8.9560E-06       758 -2.4072E-03       552
 //E31  -8.9560E-06       771 -2.4072E-03       543
 //S11   1.0437E+02       363 -1.0216E+02       768
 //S22   1.0437E+02       354 -1.0216E+02       759
 //S33   6.2914E+04       354  5.9795E+04       768
 //S12   3.3111E+01       765 -3.3459E+01       360
 //S23  -7.6469E-01       758 -2.0669E+02       552
 //S31  -7.6469E-01       771 -2.0669E+02       543
 //SMS   6.2843E+04       354  5.9864E+04       768
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1 -7.4530E-01        91
 //U2    0.0000E+00         1 -7.4530E-01        88
 //U3    0.0000E+00         1 -7.0000E+00        38
 //E11  -1.2136E-01        38 -1.2719E-01         7
 //E22  -1.2136E-01        50 -1.2719E-01        13
 //E33   2.3670E-01         7  2.2951E-01        50
 //E12   1.2005E-03        53 -1.2295E-03        10
 //E23   1.2987E-04        72 -2.5210E-03        83
 //E31   1.2987E-04        64 -2.5210E-03        96
 //S11   1.3771E+02        13 -1.3366E+02        50
 //S22   1.3771E+02         7 -1.3366E+02        38
 //S33   6.3193E+04         7  5.9532E+04        50
 //S12   3.5416E+01        53 -3.5955E+01        10
 //S23   1.0704E+01        35 -2.1669E+02        83
 //S31   1.0704E+01        31 -2.1669E+02        96
 //SMS   6.3093E+04         7  5.9628E+04        50
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11  -1.2167E-01       759 -1.2687E-01       354
 //E22  -1.2167E-01       768 -1.2687E-01       363
 //E33   2.3619E-01       354  2.3001E-01       768
 //E12   1.1247E-03       765 -1.1492E-03       360
 //E23  -8.9560E-06       758 -2.4072E-03       552
 //E31  -8.9560E-06       771 -2.4072E-03       543
 //S11   1.0437E+02       363 -1.0216E+02       768
 //S22   1.0437E+02       354 -1.0216E+02       759
 //S33   6.2914E+04       354  5.9795E+04       768
 //S12   3.3111E+01       765 -3.3459E+01       360
 //S23  -7.6469E-01       758 -2.0669E+02       552
 //S31  -7.6469E-01       771 -2.0669E+02       543
 //SMS   6.2843E+04       354  5.9864E+04       768
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_08.md | TITLE: 정적 해석(크리프) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_08.html -->

# 정적 해석(크리프)

 본 해석에는 `tutorial/08_creep_cylinder`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 4.3절의 [비선형 정적 해석(초탄성 1)](tutorial_03.html)과 동일한 원형 봉 1/8 모델입니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 정적 해석(탄소성) | !SOLUTION,TYPE=NLSTATIC |  |
| 절점 수 | 629 |  |  |
| 요소 수 | 432 |  |  |
| 요소 유형 | 8절점 1차 육면체 요소 | !ELEMENT,TYPE=361 |  |
| 재료명 | MAT1 | !MATERIAL,NAME=MAT1 |  |
| 재료 특성 | ELASTIC,CREEP | !ELASTIC !CREEP,TYPE=NORTON |  |
| 경계 조건 | 구속, 강제 변위 |  |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 
## 해석 내용

 원형 봉에 축 방향 인장 변위를 부여하여 크리프 거동 해석을 수행합니다. 해석 제어 데이터를 아래에 나타냅니다.

 
### 해석 제어 데이터 `cylinder.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!SOLUTION, TYPE=NLSTATIC
!WRITE,RESULT
!WRITE,VISUAL
## Solver Control
### Boundary Condition
!BOUNDARY, GRPID=1
 LOADS, 3, 3, -7.0
 FIX,   3, 3, 0.0
 XSYMM, 1, 1, 0.0
 YSYMM, 2, 2, 0.0
### STEP
!STEP, SUBSTEPS=5, CONVERG=1.0e-5
 BOUNDARY, 1
### Material
!MATERIAL, NAME=MAT1
!ELASTIC
 206900.0, 0.29
!CREEP, TYPE=NORTON
 1.e-10, 5.0, 0.0
### Solver Setting
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=YES,TIMELOG=YES
 10000, 1
 1.0e-8, 1.0, 0.0
## Post Control
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
cd FrontISTR/tutorial/08_creep_cylinder
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 5번째 서브스텝의 해석 결과에 대해, REVOCAP_PrePost로 미제스 응력 등고선을 추가한 변형도를 작성하여 그림 4.8.1에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![변형 및 미제스 응력 해석 결과](/examples/image/tutorial08_01.png)

  그림 4.8.1 변형 및 미제스 응력 해석 결과  
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

 ...

#### Result step=     5
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1 -4.1863E-01        91
 //U2    0.0000E+00         1 -4.1863E-01        88
 //U3    0.0000E+00         1 -7.0000E+00        38
 //E11  -6.5748E-02        38 -6.9459E-02         7
 //E22  -6.5748E-02        50 -6.9459E-02        13
 //E33   2.3884E-01        13  2.2745E-01        50
 //E12   5.5591E-04        53 -5.7135E-04        10
 //E23   1.0510E-04        72 -2.0805E-03        84
 //E31   1.0510E-04        64 -2.0805E-03        95
 //S11   1.1232E+02        14 -1.1026E+02        49
 //S22   1.1232E+02         6 -1.1026E+02        39
 //S33   4.7484E+04         7  4.5271E+04        50
 //S12   4.2781E+01        53 -4.3872E+01        10
 //S23   7.8190E+00        72 -1.6257E+02        84
 //S31   7.8190E+00        64 -1.6257E+02        95
 //SMS   4.7448E+04         7  4.5307E+04        50
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11  -6.6010E-02       759 -6.9186E-02       354
 //E22  -6.6010E-02       768 -6.9186E-02       363
 //E33   2.3840E-01       363  2.2788E-01       768
 //E12   4.9263E-04       766 -5.0184E-04       360
 //E23  -1.6179E-05       753 -1.9977E-03       498
 //E31  -1.6179E-05       749 -1.9977E-03       489
 //S11   1.0954E+02       363 -1.0744E+02       768
 //S22   1.0954E+02       354 -1.0744E+02       759
 //S33   4.7416E+04       354  4.5337E+04       759
 //S12   3.7890E+01       766 -3.8501E+01       360
 //S23  -1.3249E+00       753 -1.5632E+02       498
 //S31  -1.3250E+00       749 -1.5632E+02       489
 //SMS   4.7346E+04       363  4.5406E+04       759
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    0.0000E+00         1 -4.1863E-01        91
 //U2    0.0000E+00         1 -4.1863E-01        88
 //U3    0.0000E+00         1 -7.0000E+00        38
 //E11  -6.5748E-02        38 -6.9459E-02         7
 //E22  -6.5748E-02        50 -6.9459E-02        13
 //E33   2.3884E-01        13  2.2745E-01        50
 //E12   5.5591E-04        53 -5.7135E-04        10
 //E23   1.0510E-04        72 -2.0805E-03        84
 //E31   1.0510E-04        64 -2.0805E-03        95
 //S11   1.1232E+02        14 -1.1026E+02        49
 //S22   1.1232E+02         6 -1.1026E+02        39
 //S33   4.7484E+04         7  4.5271E+04        50
 //S12   4.2781E+01        53 -4.3872E+01        10
 //S23   7.8190E+00        72 -1.6257E+02        84
 //S31   7.8190E+00        64 -1.6257E+02        95
 //SMS   4.7448E+04         7  4.5307E+04        50
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11  -6.6010E-02       759 -6.9186E-02       354
 //E22  -6.6010E-02       768 -6.9186E-02       363
 //E33   2.3840E-01       363  2.2788E-01       768
 //E12   4.9263E-04       766 -5.0184E-04       360
 //E23  -1.6179E-05       753 -1.9977E-03       498
 //E31  -1.6179E-05       749 -1.9977E-03       489
 //S11   1.0954E+02       363 -1.0744E+02       768
 //S22   1.0954E+02       354 -1.0744E+02       759
 //S33   4.7416E+04       354  4.5337E+04       759
 //S12   3.7890E+01       766 -3.8501E+01       360
 //S23  -1.3249E+00       753 -1.5632E+02       498
 //S31  -1.3250E+00       749 -1.5632E+02       489
 //SMS   4.7346E+04       363  4.5406E+04       759
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_09.md | TITLE: 접촉 해석 (1부) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_09.html -->

# 접촉 해석 (1부)

 이 해석에는 `tutorial/09_contact_hertz`의 데이터를 사용합니다.

 
## 해석 대상

 해석은 Hertz 접촉 문제이며, 해석 대상의 형상은 그림 4.9.1에, 메시 데이터는 그림 4.9.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 정적 해석(탄소성, 접촉) | !SOLUTION,TYPE=NLSTATIC !CONTACT |  |
| 절점 수 | 408 |  |  |
| 요소 수 | 168 |  |  |
| 요소 유형 | 8절점 1차 육면체 요소 | !ELEMENT,TYPE=361 |  |
| 재료명 | MAT1 | !MATERIAL,NAME=MAT1 |  |
| 재료 특성 | ELASTIC | !ELASTIC |  |
| 경계 조건 | 구속, 강제 변위 |  |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 ![해석 대상의 형상](/examples/image/tutorial09_01.png)

  그림 4.9.1 해석 대상의 형상  ![해석 대상의 메시 데이터](/examples/image/tutorial09_02.png)

  그림 4.9.2 해석 대상의 메시 데이터  
## 해석 내용

 원판 1/4 모델의 윗면에 압축 방향의 강제 변위를 부여하는 접촉 해석을 증강 라그랑주 승수법으로 수행합니다. 해석 제어 데이터는 다음과 같습니다.

 
### 해석 제어 데이터 `cgs3.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!SOLUTION, TYPE=NLSTATIC
!WRITE,RESULT
!WRITE,VISUAL
## Solver Control
### Boundary Condition
!BOUNDARY, GRPID=1
 ALL, 3, 3, 0.0
 BOTTOM, 2, 2, 0.0
 CENTER, 1, 1, 0.0
 UPPER,  2, 2, -0.306
!CONTACT_ALGO, TYPE=ALAGRANGE
!CONTACT, GRPID=1
 CP1, 0.0
### STEP
!STEP, SUBSTEPS=5, CONVERG=1.0e-5
 BOUNDARY, 1
 CONTACT,  1
### Material
!MATERIAL, NAME=MAT1
!ELASTIC
 1100.0, 0.0
### Solver Setting
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=YES,TIMELOG=YES
 1000, 1
 1.0e-10, 1.0, 0.0
## Post Control
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
cd FrontISTR/tutorial/09_contact_hertz
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 5번째 서브스텝의 해석 결과에 대해 y 방향 변위 등고선을 추가한 변형도를 REVOCAP_PrePost로 작성하여 그림 4.9.3에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![변형 및 y 방향 변위 해석 결과](/examples/image/tutorial09_03.png)

  그림 4.9.3 변형 및 y 방향 변위 해석 결과  
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

 ...

#### Result step=     5
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    1.3079E-02        31 -4.1442E-02        54
 //U2    4.9597E-03      1008 -3.0604E-01        69
 //U3    0.0000E+00         1  0.0000E+00         1
 //E11   7.1171E-03      1049 -4.7627E-02         1
 //E22   2.0644E-03      1012 -5.3428E-02        50
 //E33   9.3301E-03        50 -4.0103E-03        54
 //E12   4.3231E-02      1000 -2.7666E-02        54
 //E23   1.3028E-12      1046 -1.2515E-12         2
 //E31   3.5660E-13        48 -2.1028E-12        51
 //S11   7.9455E+00      1049 -4.9987E+01         1
 //S22   2.2819E+00      1012 -5.3282E+01        50
 //S33   1.1359E+01        50 -4.4937E+00        54
 //S12   2.0802E+01      1000 -1.4668E+01        54
 //S23   6.7682E-10      1046 -6.3666E-10         2
 //S31   2.0235E-10        48 -1.1497E-09        51
 //SMS   6.0944E+01      1000  1.0198E-01        69
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   4.6311E-03      1024 -3.5443E-02         8
 //E22   1.7091E-03      1031 -5.4262E-02         8
 //E33   1.5903E-04         7 -1.6631E-05       101
 //E12   1.8590E-02      1009 -2.6073E-02        14
 //E23   1.5204E-13      1000 -4.1521E-13        16
 //E31   1.3902E-13         5 -8.2391E-13         7
 //S11   5.2516E+00      1024 -3.8099E+01         8
 //S22   1.8872E+00      1031 -5.5262E+01         8
 //S33   2.2824E-01         7 -1.8263E-02       101
 //S12   9.8516E+00      1009 -1.3352E+01        14
 //S23   7.8767E-11      1000 -2.1075E-10        16
 //S31   7.7553E-11         5 -4.4780E-10         7
 //SMS   4.9470E+01         8  2.4786E-01      1007
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    1.3079E-02        31 -4.1442E-02        54
 //U2    4.9597E-03      1008 -3.0604E-01        69
 //U3    0.0000E+00         1  0.0000E+00         1
 //E11   7.1171E-03      1049 -4.7627E-02         1
 //E22   2.0644E-03      1012 -5.3428E-02        50
 //E33   9.3301E-03        50 -4.0103E-03        54
 //E12   4.3231E-02      1000 -2.7666E-02        54
 //E23   1.3028E-12      1046 -1.2515E-12         2
 //E31   3.5660E-13        48 -2.1028E-12        51
 //S11   7.9455E+00      1049 -4.9987E+01         1
 //S22   2.2819E+00      1012 -5.3282E+01        50
 //S33   1.1359E+01        50 -4.4937E+00        54
 //S12   2.0802E+01      1000 -1.4668E+01        54
 //S23   6.7682E-10      1046 -6.3666E-10         2
 //S31   2.0235E-10        48 -1.1497E-09        51
 //SMS   6.0944E+01      1000  1.0198E-01        69
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   4.6311E-03      1024 -3.5443E-02         8
 //E22   1.7091E-03      1031 -5.4262E-02         8
 //E33   1.5903E-04         7 -1.6631E-05       101
 //E12   1.8590E-02      1009 -2.6073E-02        14
 //E23   1.5204E-13      1000 -4.1521E-13        16
 //E31   1.3902E-13         5 -8.2391E-13         7
 //S11   5.2516E+00      1024 -3.8099E+01         8
 //S22   1.8872E+00      1031 -5.5262E+01         8
 //S33   2.2824E-01         7 -1.8263E-02       101
 //S12   9.8516E+00      1009 -1.3352E+01        14
 //S23   7.8767E-11      1000 -2.1075E-10        16
 //S31   7.7553E-11         5 -4.4780E-10         7
 //SMS   4.9470E+01         8  2.4786E-01      1007
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_10.md | TITLE: 접촉 해석(2부) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_10.html -->

# 접촉 해석(2부)

 이 해석에는 `tutorial/10_contact_2tubes`의 데이터를 사용합니다.

 
## 해석 대상

 원통 압입 문제입니다. 해석 대상의 형상은 그림 4.10.1에, 메시 데이터는 그림 4.10.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 정적 해석(탄성, 접촉) | !SOLUTION,TYPE=NLSTATIC !CONTACT |  |
| 접촉점 수 | 4,000 |  |  |
| 요소 수 | 2,888 |  |  |
| 요소 유형 | 8절점 1차 육면체 요소 | !ELEMENT,TYPE=361 |  |
| 재료명 | M1 | !MATERIAL,NAME=M1 |  |
| 재료 특성 | ELASTIC | !ELASTIC |  |
| 경계 조건 | 구속, 강제 변위 |  |  |
| 행렬 해법 | 직접법 | !SOLVER,METHOD=MUMPS |  |

 ![해석 모델 형상](/examples/image/tutorial10_01.png)

  그림 4.10.1 해석 모델 형상  ![해석 모델 메시 데이터](/examples/image/tutorial10_02.png)

  그림 4.10.2 해석 모델 메시 데이터  
## 해석 내용

 그림 4.10.1에 표시된 강제 변위 면에 압입 방향의 강제 변위를 부여하고 라그랑주 승수법을 사용하여 접촉 해석을 수행합니다. 해석 제어 데이터는 아래와 같습니다.

 
### 해석 제어 데이터 `2tubes.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!SOLUTION, TYPE=NLSTATIC
!WRITE,RESULT
!WRITE,VISUAL
## Solver Control
### Boundary Condition
!BOUNDARY, GRPID=1
  X0, 1, 3, 0.0
  Y0, 2, 2, 0.0
  Z0, 3, 3, 0.0
!BOUNDARY, GRPID=2
  X1, 1, 1, 0.0
!BOUNDARY, GRPID=3
  X1, 1, 1, -1.0
!CONTACT_ALGO, TYPE=SLAGRANGE
!CONTACT, GRPID=1, INTERACTION=FSLID, NPENALTY=1.0e+2
  CP1, 0.0, 1.0e+5
### STEP
!STEP, SUBSTEPS=4, CONVERG=1.0e-5
 BOUNDARY, 1
 BOUNDARY, 3
 CONTACT, 1
### Material
!MATERIAL, NAME=M1
!ELASTIC
 2.1e+5, 0.3
### Solver Setting
!SOLVER,METHOD=MUMPS
## Post Control
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
cd FrontISTR/tutorial/10_contact_2tubes
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 4번째 서브스텝의 해석 결과에 대해 REVOCAP_PrePost를 사용하여 von Mises 응력 등고선을 포함한 변형 형상도를 작성하고 그림 4.10.3에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![해석 결과: 변형 및 von Mises 응력](/examples/image/tutorial10_03.png)

  그림 4.10.3 해석 결과: 변형 및 von Mises 응력  
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

 ...

#### Result step=     4
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    8.7136E-04        32 -1.0021E+00      2006
 //U2    8.7633E-03       104 -7.0546E-03      2006
 //U3    8.7634E-03         4 -7.0546E-03      1901
 //E11   7.5985E-04      1901 -4.1250E-04       105
 //E22   9.8385E-04         2 -9.3128E-04      2058
 //E33   9.8386E-04       102 -9.3120E-04      3843
 //E12   5.3544E-04       133 -2.8361E-04       278
 //E23   1.2509E-03      1901 -1.4195E-03         4
 //E31   5.3554E-04        33 -2.8366E-04      1678
 //S11   7.7172E+01       103 -8.9880E+01       101
 //S22   2.0158E+02         2 -2.2918E+02      1905
 //S33   2.0159E+02       102 -2.2920E+02      2010
 //S12   4.3210E+01       133 -2.2913E+01       278
 //S23   1.0057E+02      1901 -1.1446E+02         4
 //S31   4.3219E+01        33 -2.2918E+01      1678
 //SMS   3.0051E+02      1901  3.1670E+00      2454
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   4.3715E-04      2809 -3.1965E-04      1293
 //E22   8.8407E-04        73 -7.9864E-04      1369
 //E33   8.8408E-04      1365 -7.9867E-04      2809
 //E12   5.3226E-04      1324 -1.9554E-04       188
 //E23   1.0836E-03      2149 -1.0974E-03       684
 //E31   5.3223E-04        32 -1.9559E-04      1176
 //S11   6.6688E+01      1296 -7.9313E+01      1293
 //S22   1.8533E+02        73 -2.1547E+02      1369
 //S33   1.8533E+02      1365 -2.1548E+02      2809
 //S12   4.2951E+01      1324 -1.5800E+01       188
 //S23   8.7338E+01      2149 -8.8751E+01       684
 //S31   4.2949E+01        32 -1.5804E+01      1176
 //SMS   1.9064E+02      1369  3.2782E+00      2126
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    8.7136E-04        32 -1.0021E+00      2006
 //U2    8.7633E-03       104 -7.0546E-03      2006
 //U3    8.7634E-03         4 -7.0546E-03      1901
 //E11   7.5985E-04      1901 -4.1250E-04       105
 //E22   9.8385E-04         2 -9.3128E-04      2058
 //E33   9.8386E-04       102 -9.3120E-04      3843
 //E12   5.3544E-04       133 -2.8361E-04       278
 //E23   1.2509E-03      1901 -1.4195E-03         4
 //E31   5.3554E-04        33 -2.8366E-04      1678
 //S11   7.7172E+01       103 -8.9880E+01       101
 //S22   2.0158E+02         2 -2.2918E+02      1905
 //S33   2.0159E+02       102 -2.2920E+02      2010
 //S12   4.3210E+01       133 -2.2913E+01       278
 //S23   1.0057E+02      1901 -1.1446E+02         4
 //S31   4.3219E+01        33 -2.2918E+01      1678
 //SMS   3.0051E+02      1901  3.1670E+00      2454
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   4.3715E-04      2809 -3.1965E-04      1293
 //E22   8.8407E-04        73 -7.9864E-04      1369
 //E33   8.8408E-04      1365 -7.9867E-04      2809
 //E12   5.3226E-04      1324 -1.9554E-04       188
 //E23   1.0836E-03      2149 -1.0974E-03       684
 //E31   5.3223E-04        32 -1.9559E-04      1176
 //S11   6.6688E+01      1296 -7.9313E+01      1293
 //S22   1.8533E+02        73 -2.1547E+02      1369
 //S33   1.8533E+02      1365 -2.1548E+02      2809
 //S12   4.2951E+01      1324 -1.5800E+01       188
 //S23   8.7338E+01      2149 -8.8751E+01       684
 //S31   4.2949E+01        32 -1.5804E+01      1176
 //SMS   1.9064E+02      1369  3.2782E+00      2126
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_11.md | TITLE: 접촉 해석 (3부) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_11.html -->

# 접촉 해석 (3부)

 이 해석에는 `tutorial/11_contact_2beam`의 데이터를 사용합니다.

 
## 해석 대상

 해석은 두 보의 접촉 문제이며, 해석 모델의 개요를 그림 4.11.1에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 정적 해석(탄소성, 접촉) | !SOLUTION,TYPE=NLSTATIC !CONTACT |  |
| 절점 수 | 252 |  |  |
| 요소 수 | 80 |  |  |
| 요소 유형 | 8절점 1차 육면체 요소 | !ELEMENT,TYPE=361 |  |
| 재료명 | M1 | !MATERIAL,NAME=M1 |  |
| 재료 특성 | ELASTIC, PLASTIC | !ELASTIC !PLASTIC |  |
| 경계 조건 | 구속, 강제 변위 |  |  |
| 행렬 해법 | 직접법 | !SOLVER,METHOD=MUMPS |  |

 ![해석 모델 개요](/examples/image/tutorial11_01.png)

  그림 4.11.1 해석 모델 개요  
## 해석 내용

 위쪽 보의 끝면에 강제 변위를 부여하는 접촉 해석을 라그랑주 승수법으로 수행합니다. 해석 제어 데이터는 다음과 같습니다.

 
### 해석 제어 데이터 `2beams.cnt`

 
```text
!!
!! Control File for FISTR
!!
!VERSION
 3
!SOLUTION, TYPE=NLSTATIC
!WRITE,RESULT
!WRITE,VISUAL
!BOUNDARY, GRPID=1
  ng1, 1, 3, 0.0
  ng2, 1, 3, 0.0
  ng3, 3, 3, -3.0
!CONTACT_ALGO, TYPE=SLAGRANGE
!CONTACT, GRPID=1, INTERACTION=FSLID
  CP1, 0.0, 1.0e+5
!STEP, SUBSTEPS=100, CONVERG=1.0e-4, MAXITER=1000
 BOUNDARY, 1
 CONTACT, 1
!MATERIAL, NAME=M1
!ELASTIC
 2.1e+5, 0.3
!PLASTIC,YIELD=MISES
 500.0, 10.0
!SOLVER,METHOD=MUMPS
## Post Control
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
cd FrontISTR/tutorial/11_contact_2beam
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 100번째 서브스텝의 해석 결과에 대해 미제스 응력 등고선을 추가한 변형도를 REVOCAP_PrePost로 작성하여 그림 4.11.2에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![변형 및 미제스 응력 해석 결과](/examples/image/tutorial11_02.png)

  그림 4.11.2 변형 및 미제스 응력 해석 결과  
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

 ...

#### Result step=   100
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    1.4052E-01       196 -6.1114E-01         6
 //U2    4.5562E-02        11 -4.5562E-02       195
 //U3    0.0000E+00         1 -3.0000E+00         8
 //E11   1.5391E-01        11 -1.2825E-01        49
 //E22   5.8868E-02        49 -7.4361E-02       195
 //E33   7.2825E-02       152 -8.2406E-02       132
 //E12   8.9678E-02         7 -8.9678E-02       192
 //E23   9.9327E-02        11 -9.9327E-02       195
 //E31   8.0649E-02       153 -1.3281E-01         7
 //S11   5.7908E+02       132 -6.3560E+02        89
 //S22   1.3056E+02         3 -1.2707E+02        10
 //S33   1.5273E+02         3 -1.4227E+02         1
 //S12   1.4313E+02         3 -1.4313E+02       129
 //S23   1.8051E+02       109 -1.8051E+02       172
 //S31   1.5818E+02       153 -2.4710E+02       194
 //SMS   6.2325E+02       152  8.3543E+00       128
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   8.2835E-02        43 -5.4976E-02         2
 //E22   1.7707E-02        44 -2.6416E-02        43
 //E33   3.7575E-02         2 -5.6705E-02        22
 //E12   1.0513E-11         2 -2.5044E-11        22
 //E23   6.2706E-12         2 -1.8208E-11        42
 //E31   2.2838E-02         2 -6.5650E-02        43
 //S11   5.3959E+02        21 -4.8101E+02         1
 //S22   1.9112E+02        21 -2.2398E+02         1
 //S33   4.5634E+01        25 -6.1476E+01         1
 //S12   3.4435E-07         2 -2.4339E-07        42
 //S23   5.4042E-07         1 -6.6410E-07        22
 //S31   1.0158E+02         4 -1.3326E+02        24
 //SMS   4.6388E+02        22  1.0584E+01        20
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    1.4052E-01       196 -6.1114E-01         6
 //U2    4.5562E-02        11 -4.5562E-02       195
 //U3    0.0000E+00         1 -3.0000E+00         8
 //E11   1.5391E-01        11 -1.2825E-01        49
 //E22   5.8868E-02        49 -7.4361E-02       195
 //E33   7.2825E-02       152 -8.2406E-02       132
 //E12   8.9678E-02         7 -8.9678E-02       192
 //E23   9.9327E-02        11 -9.9327E-02       195
 //E31   8.0649E-02       153 -1.3281E-01         7
 //S11   5.7908E+02       132 -6.3560E+02        89
 //S22   1.3056E+02         3 -1.2707E+02        10
 //S33   1.5273E+02         3 -1.4227E+02         1
 //S12   1.4313E+02         3 -1.4313E+02       129
 //S23   1.8051E+02       109 -1.8051E+02       172
 //S31   1.5818E+02       153 -2.4710E+02       194
 //SMS   6.2325E+02       152  8.3543E+00       128
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   8.2835E-02        43 -5.4976E-02         2
 //E22   1.7707E-02        44 -2.6416E-02        43
 //E33   3.7575E-02         2 -5.6705E-02        22
 //E12   1.0513E-11         2 -2.5044E-11        22
 //E23   6.2706E-12         2 -1.8208E-11        42
 //E31   2.2838E-02         2 -6.5650E-02        43
 //S11   5.3959E+02        21 -4.8101E+02         1
 //S22   1.9112E+02        21 -2.2398E+02         1
 //S33   4.5634E+01        25 -6.1476E+01         1
 //S12   3.4435E-07         2 -2.4339E-07        42
 //S23   5.4042E-07         1 -6.6410E-07        22
 //S31   1.0158E+02         4 -1.3326E+02        24
 //SMS   4.6388E+02        22  1.0584E+01        20
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_12.md | TITLE: 선형 동적 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_12.html -->

# 선형 동적 해석

 이 해석에는 `tutorial/12_dynamic_beam`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 외팔보이며, 형상은 그림 4.12.1에, 메시 데이터는 그림 4.12.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 선형 동적 해석 | !SOLUTION,TYPE=DYNAMIC !DYNAMIC,TYPE=LINEAR |  |
| 절점 수 | 525 |  |  |
| 요소 수 | 240 |  |  |
| 요소 유형 | 10절점 2차 사면체 요소 | !ELEMENT,TYPE=342 |  |
| 재료명 | M1 | !MATERIAL,NAME=M1 |  |
| 경계 조건 | 구속, 집중 하중 | !CLOAD |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 ![외팔보의 형상](/examples/image/tutorial12_01.png)

  그림 4.12.1 외팔보의 형상  ![외팔보의 메시 데이터](/examples/image/tutorial12_02.png)

  그림 4.12.2 외팔보의 메시 데이터  
## 해석 내용

 그림 4.12.1에 나타낸 구속면의 변위를 구속하고 하중 절점에 집중 하중을 적용한 후 선형 동적 해석을 수행합니다. 해석 제어 데이터는 아래와 같습니다.

 
### 해석 제어 데이터 `beam.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!WRITE,LOG,FREQUENCY=5000
!WRITE,RESULT,FREQUENCY=5000
!SOLUTION, TYPE=DYNAMIC
!DYNAMIC, TYPE=LINEAR
 11 , 1
 0.0, 1.0, 500000, 1.0000e-8
 0.5, 0.25
 1, 1, 0.0, 0.0
 100000, 3121, 500
 1, 1, 1, 1, 1, 1
## Solver Control
### Boundary Condition
!BOUNDARY, AMP=AMP1
 FIX, 1, 3, 0.0
!CLOAD, AMP=AMP1
 CL1,    3,    -1.0
### Material
# define in mesh file
### Solver Setting
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=NO,TIMELOG=NO
 10000, 1
 1.0e-06, 1.0, 0.0
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 fistr1을 실행합니다.

 
```text
cd FrontISTR/tutorial/12_dynamic_beam
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 해석 제어 데이터에서 지정한 모니터링 절점(하중 절점, 절점 번호 3121)의 변위 시계열을 Microsoft Excel로 작성하여 그림 4.12.3에 나타냅니다. 또한 해석 결과의 수치 데이터로서 모니터링 절점 변위 출력 파일(dyna_disp_3121.txt)의 일부를 아래에 나타냅니다.

 ![모니터링 절점 변위의 시계열](/examples/image/tutorial12_03.png)

  그림 4.12.3 모니터링 절점 변위의 시계열  
### 모니터링 절점의 변위 `dyna_disp_3121.txt`

 
```text
0  0.0000E+000      3121  0.0000E+000  0.0000E+000  0.0000E+000
       500  5.0000E-006      3121  5.3301E-005 -2.6682E-005 -1.5646E-002
      1000  1.0000E-005      3121  4.0790E-005 -1.0696E-006 -4.4118E-002
      1500  1.5000E-005      3121  9.1017E-005  5.7542E-005 -8.1017E-002
      2000  2.0000E-005      3121  1.8944E-005  5.6499E-005 -1.2358E-001
      2500  2.5000E-005      3121  3.4535E-005  6.1147E-005 -1.7787E-001
      3000  3.0000E-005      3121  3.0248E-005  1.6211E-004 -2.2844E-001
      3500  3.5000E-005      3121  4.2434E-005  1.1706E-004 -2.7330E-001
      4000  4.0000E-005      3121 -2.0130E-005  1.2298E-004 -3.2436E-001
      4500  4.5000E-005      3121  4.1976E-005 -4.2753E-005 -3.8902E-001
      5000  5.0000E-005      3121  5.6526E-005  1.2043E-004 -4.6494E-001
      5500  5.5000E-005      3121  1.9195E-005  8.8901E-006 -5.4673E-001
      6000  6.0000E-005      3121  3.9722E-005 -8.0492E-005 -6.4665E-001
      6500  6.5000E-005      3121  9.0688E-005 -1.9603E-004 -7.5697E-001
      7000  7.0000E-005      3121  3.8175E-005  1.3406E-004 -8.6961E-001
      7500  7.5000E-005      3121 -2.1776E-005  2.9617E-004 -9.6952E-001
      8000  8.0000E-005      3121 -1.6732E-005  2.0223E-004 -1.0672E+000
      8500  8.5000E-005      3121  1.0129E-004  4.9717E-004 -1.1583E+000
      9000  9.0000E-005      3121  4.4797E-005  6.6073E-004 -1.2421E+000
      9500  9.5000E-005      3121 -5.5023E-007  7.2865E-004 -1.3154E+000
     10000  1.0000E-004      3121  4.6793E-005  3.6134E-004 -1.3947E+000
```

 
### 해석 결과 로그 `0.log`

  
```text
fstr_setup: OK
#### Result step=  5000
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    3.5119E-02      5021 -3.5178E-02      1221
 //U2    1.1456E-03      1203 -1.1672E-03      1003
 //U3    0.0000E+00      1001 -4.6494E-01      3121
 //V1    1.2518E+03   1001249 -1.4015E+03   1000542
 //V2    3.8803E+02      5021 -3.3289E+02   1001074
 //V3    4.7101E+01   1000007 -1.5435E+04   1001066
 //A1    6.7723E+09      5021 -4.1214E+09      5011
 //A2    4.9616E+09      3219 -1.2304E+10      1221
 //A3    9.4377E+09      1221 -5.7155E+09      5121
 //E11   8.0049E-03      5001 -7.9582E-03   1000367
 //E22   2.8641E-03      1203 -2.2671E-03      5103
 //E33   2.8794E-03      1203 -2.6589E-03   1001105
 //E12   2.6224E-03      1201 -2.3782E-03      1001
 //E23   8.3475E-04   1000713 -1.0763E-03   1000353
 //E31   5.8002E-04   1000750 -2.3236E-03   1000368
 //S11   3.7805E+01   1000738 -4.1247E+01      1201
 //S22   1.4126E+01   1000738 -1.7677E+01      1201
 //S33   9.6734E+00   1001098 -1.7677E+01      1201
 //S12   4.0344E+00      1201 -3.6588E+00      1001
 //S23   1.2842E+00   1000713 -1.6558E+00   1000353
 //S31   8.9234E-01   1000750 -3.5747E+00   1000368
 //SMS   3.2995E+01      1203  1.8529E-01   1000240
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   5.9346E-03       126 -5.9635E-03        61
 //E22   1.6102E-03        68 -1.6304E-03       125
 //E33   1.8989E-03        61 -1.9798E-03       126
 //E12   8.3076E-04        62 -7.8339E-04         3
 //E23   5.7501E-04       119 -6.2415E-04       178
 //E31   2.5209E-04        35 -1.3096E-03         2
 //S11   2.4565E+01       126 -2.6193E+01         2
 //S22   5.2809E+00       183 -5.7364E+00         2
 //S33   5.5451E+00       183 -5.7987E+00         2
 //S12   1.2781E+00        62 -1.2052E+00         3
 //S23   8.8463E-01       119 -9.6023E-01       178
 //S31   3.8783E-01        35 -2.0148E+00         2
 //SMS   2.3676E+01        61  1.7835E+00       179
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    3.5119E-02      5021 -3.5178E-02      1221
 //U2    1.1456E-03      1203 -1.1672E-03      1003
 //U3    0.0000E+00      1001 -4.6494E-01      3121
 //V1    1.2518E+03   1001249 -1.4015E+03   1000542
 //V2    3.8803E+02      5021 -3.3289E+02   1001074
 //V3    4.7101E+01   1000007 -1.5435E+04   1001066
 //A1    6.7723E+09      5021 -4.1214E+09      5011
 //A2    4.9616E+09      3219 -1.2304E+10      1221
 //A3    9.4377E+09      1221 -5.7155E+09      5121
 //E11   8.0049E-03      5001 -7.9582E-03   1000367
 //E22   2.8641E-03      1203 -2.2671E-03      5103
 //E33   2.8794E-03      1203 -2.6589E-03   1001105
 //E12   2.6224E-03      1201 -2.3782E-03      1001
 //E23   8.3475E-04   1000713 -1.0763E-03   1000353
 //E31   5.8002E-04   1000750 -2.3236E-03   1000368
 //S11   3.7805E+01   1000738 -4.1247E+01      1201
 //S22   1.4126E+01   1000738 -1.7677E+01      1201
 //S33   9.6734E+00   1001098 -1.7677E+01      1201
 //S12   4.0344E+00      1201 -3.6588E+00      1001
 //S23   1.2842E+00   1000713 -1.6558E+00   1000353
 //S31   8.9234E-01   1000750 -3.5747E+00   1000368
 //SMS   3.2995E+01      1203  1.8529E-01   1000240
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   5.9346E-03       126 -5.9635E-03        61
 //E22   1.6102E-03        68 -1.6304E-03       125
 //E33   1.8989E-03        61 -1.9798E-03       126
 //E12   8.3076E-04        62 -7.8339E-04         3
 //E23   5.7501E-04       119 -6.2415E-04       178
 //E31   2.5209E-04        35 -1.3096E-03         2
 //S11   2.4565E+01       126 -2.6193E+01         2
 //S22   5.2809E+00       183 -5.7364E+00         2
 //S33   5.5451E+00       183 -5.7987E+00         2
 //S12   1.2781E+00        62 -1.2052E+00         3
 //S23   8.8463E-01       119 -9.6023E-01       178
 //S31   3.8783E-01        35 -2.0148E+00         2
 //SMS   2.3676E+01        61  1.7835E+00       179

 ...

#### Result step=500000
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    5.2758E-02      5221 -5.2989E-02      1021
 //U2    1.0426E-03   1000750 -7.2090E-02      1021
 //U3    0.0000E+00      1001 -6.5930E-01      3121
 //V1    1.6455E+03      5217 -1.6058E+03      1019
 //V2    8.7881E+02   1000132 -2.2448E+03      1221
 //V3    1.9455E+02   1000003 -1.7874E+04      1221
 //A1    5.7327E+09      5221 -4.0019E+09      1203
 //A2    7.0089E+09      3221 -4.3280E+09   1000713
 //A3    4.9140E+09      1211 -8.4702E+09      5021
 //E11   1.0808E-02   1001104 -1.0904E-02      1003
 //E22   3.4008E-03      1203 -3.2574E-03      5203
 //E33   3.6321E-03      1103 -3.5530E-03   1001105
 //E12   3.4760E-03      1201 -2.9256E-03      1001
 //E23   9.1779E-04   1000017 -1.3462E-03      1001
 //E31   6.4324E-04   1000750 -2.5756E-03      1101
 //S11   5.0236E+01      5201 -5.2624E+01      1001
 //S22   1.8815E+01   1001098 -2.2485E+01   1000008
 //S33   1.2848E+01   1001098 -2.2485E+01   1000008
 //S12   5.3477E+00      1201 -4.5009E+00      1001
 //S23   1.4120E+00   1000017 -2.0711E+00      1001
 //S31   9.8961E-01   1000750 -3.9625E+00      1101
 //SMS   4.4276E+01      1003  6.5308E-01   1000690
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   8.0789E-03       186 -8.0188E-03      1001
 //E22   2.1847E-03         8 -2.1148E-03       192
 //E33   2.5029E-03      1001 -2.6704E-03       186
 //E12   1.1765E-03        62 -1.0144E-03         3
 //E23   4.9965E-04         6 -5.1495E-04        66
 //E31  -1.2336E-05       125 -1.4721E-03         2
 //S11   3.3432E+01       186 -3.4438E+01         2
 //S22   7.0191E+00       183 -7.4671E+00         2
 //S33   7.6014E+00       183 -7.7370E+00         2
 //S12   1.8099E+00        62 -1.5605E+00         3
 //S23   7.6870E-01         6 -7.9223E-01        66
 //S31  -1.8979E-02       125 -2.2647E+00         2
 //SMS   3.1742E+01       186  7.0883E-01       114
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    5.2758E-02      5221 -5.2989E-02      1021
 //U2    1.0426E-03   1000750 -7.2090E-02      1021
 //U3    0.0000E+00      1001 -6.5930E-01      3121
 //V1    1.6455E+03      5217 -1.6058E+03      1019
 //V2    8.7881E+02   1000132 -2.2448E+03      1221
 //V3    1.9455E+02   1000003 -1.7874E+04      1221
 //A1    5.7327E+09      5221 -4.0019E+09      1203
 //A2    7.0089E+09      3221 -4.3280E+09   1000713
 //A3    4.9140E+09      1211 -8.4702E+09      5021
 //E11   1.0808E-02   1001104 -1.0904E-02      1003
 //E22   3.4008E-03      1203 -3.2574E-03      5203
 //E33   3.6321E-03      1103 -3.5530E-03   1001105
 //E12   3.4760E-03      1201 -2.9256E-03      1001
 //E23   9.1779E-04   1000017 -1.3462E-03      1001
 //E31   6.4324E-04   1000750 -2.5756E-03      1101
 //S11   5.0236E+01      5201 -5.2624E+01      1001
 //S22   1.8815E+01   1001098 -2.2485E+01   1000008
 //S33   1.2848E+01   1001098 -2.2485E+01   1000008
 //S12   5.3477E+00      1201 -4.5009E+00      1001
 //S23   1.4120E+00   1000017 -2.0711E+00      1001
 //S31   9.8961E-01   1000750 -3.9625E+00      1101
 //SMS   4.4276E+01      1003  6.5308E-01   1000690
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   8.0789E-03       186 -8.0188E-03      1001
 //E22   2.1847E-03         8 -2.1148E-03       192
 //E33   2.5029E-03      1001 -2.6704E-03       186
 //E12   1.1765E-03        62 -1.0144E-03         3
 //E23   4.9965E-04         6 -5.1495E-04        66
 //E31  -1.2336E-05       125 -1.4721E-03         2
 //S11   3.3432E+01       186 -3.4438E+01         2
 //S22   7.0191E+00       183 -7.4671E+00         2
 //S33   7.6014E+00       183 -7.7370E+00         2
 //S12   1.8099E+00        62 -1.5605E+00         3
 //S23   7.6870E-01         6 -7.9223E-01        66
 //S31  -1.8979E-02       125 -2.2647E+00         2
 //SMS   3.1742E+01       186  7.0883E-01       114
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_13.md | TITLE: 비선형 동적 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_13.html -->

# 비선형 동적 해석

 본 해석에는 `tutorial/13_dynamic_beam_nonlinear`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 앞 절의 [선형 동적 해석](tutorial_12.html)과 동일한 외팔보입니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 동적 해석 | !SOLUTION,TYPE=DYNAMIC !DYNAMIC,TYPE=NONLINEAR |  |
| 절점 수 | 525 |  |  |
| 요소 수 | 240 |  |  |
| 요소 유형 | 10절점 2차 사면체 요소 | !ELEMENT,TYPE=342 |  |
| 재료명 | M1 | !MATERIAL,NAME=M1 |  |
| 경계 조건 | 구속, 집중 하중 | !CLOAD |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 
## 해석 내용

 그림 4.12.1에 나타낸 구속면의 변위를 구속하고 하중 절점에 집중 하중을 부하한 후의 비선형 동적 해석을 수행합니다. 해석 제어 데이터를 아래에 나타냅니다.

 ![외팔보의 형상](/examples/image/tutorial12_01.png)

  그림 4.12.1 외팔보의 형상  
### 해석 제어 데이터 `beam.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!WRITE,RESULT,FREQUENCY=100
!SOLUTION, TYPE=DYNAMIC
!DYNAMIC, TYPE=NONLINEAR
 1 , 1
 0.0, 0.1, 100000, 1.0000e-8
 0.5, 0.25
 1, 1, 0.0, 0.0
 1000, 3121, 100
 1, 1, 1, 1, 1, 1
## Solver Control
### Boundary Condition
!BOUNDARY, GRPID=1, AMP=AMP1
 FIX, 1, 3, 0.0
!CLOAD, GRPID=1, AMP=AMP1
 CL1,    3,    -1.0
### STEP
!STEP, CONVERG=1.0e-3
 BOUNDARY, 1
 LOAD,     1
### Material
!DENSITY
 1.0e-8
!HYPERELASTIC, TYPE=NEOHOOKE
 1000.0, 0.00005
### Solver Setting
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=NO,TIMELOG=NO
 10000, 1
 1.0e-06, 1.0, 0.0
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 `fistr1`을 실행합니다.

 
```text
cd FrontISTR/tutorial/13_dynamic_beam_nonlinear
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 해석 제어 데이터에서 지정한 모니터링 절점(하중 절점, 절점 번호3121)의 변위 시계열을 Microsoft Excel로 작성하여 그림 4.13.1에 나타냅니다. 또한 해석 결과의 수치 데이터로서 모니터링 절점의 변위 출력 파일(dyna_disp_3121.txt)의 일부를 아래에 나타냅니다.

 ![모니터링 절점의 변위 시계열](/examples/image/tutorial13_01.png)

  그림 4.13.1 모니터링 절점의 변위 시계열  
### 모니터링 절점의 변위 `dyna_disp_3121.txt`

 
```text
0  0.0000E+000      3121  0.0000E+000  0.0000E+000  0.0000E+000
       100  1.0000E-006      3121  7.6885E-005 -7.3733E-005 -6.0988E-004
       200  2.0000E-006      3121  3.3089E-005 -7.5879E-006 -8.2481E-004
       300  3.0000E-006      3121  8.9272E-005 -5.6180E-005 -1.2550E-003
       400  4.0000E-006      3121  5.8434E-005 -2.9113E-005 -1.9326E-003
       500  5.0000E-006      3121  3.3598E-005 -3.7069E-005 -2.6955E-003
       600  6.0000E-006      3121  9.2438E-005 -2.9415E-005 -3.4297E-003
       700  7.0000E-006      3121  4.4742E-005 -1.9064E-005 -4.2128E-003
       800  8.0000E-006      3121  4.2702E-005 -3.7315E-005 -5.2563E-003
       900  9.0000E-006      3121  7.9468E-005  4.8283E-006 -6.1239E-003
      1000  1.0000E-005      3121  2.5902E-005 -3.1393E-005 -7.1463E-003
      1100  1.1000E-005      3121  6.9365E-005 -1.1486E-005 -8.3515E-003
```

 
### 해석 결과 로그 `0.log`

  
```text
fstr_setup: OK
#### Result step=     1
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    2.7619E-10      3121 -1.4312E-10   1001067
 //U2    2.6490E-10   1000353 -5.5027E-10      3121
 //U3    2.8724E-10      5121 -1.8462E-07      3121
 //V1    5.5239E-02      3121 -2.8624E-02   1001067
 //V2    5.2980E-02   1000353 -1.1005E-01      3121
 //V3    5.7448E-02      5121 -3.6925E+01      3121
 //A1    1.1048E+07      3121 -5.7248E+06   1001067
 //A2    1.0596E+07   1000353 -2.2011E+07      3121
 //A3    1.1490E+07      5121 -7.3849E+09      3121
 //E11   4.3811E-10   1000347 -3.4878E-10   1000709
 //E22   2.1254E-09   1000352 -2.1246E-09   1001427
 //E33   3.6982E-07   1001067 -3.6953E-07   1000347
 //E12   1.1329E-09   1000709 -1.1050E-09   1000352
 //E23   3.6823E-07   1000713 -3.6823E-07   1000353
 //E31   9.1796E-08   1001031 -2.7687E-07      3121
 //S11   1.4231E-02   1001067 -1.4214E-02   1000347
 //S22   1.4226E-02   1001067 -1.4210E-02   1000347
 //S33   1.5711E-02   1001067 -1.5694E-02   1000347
 //S12   2.2266E-06   1000709 -2.2148E-06   1000352
 //S23   7.3645E-04   1000713 -7.3646E-04   1000353
 //S31   1.8358E-04   1001031 -5.5379E-04      3121
 //SMS   1.7350E-03   1000347  3.4921E-43      1201
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   6.5755E-11       176 -1.7785E-10       178
 //E22   7.2706E-10       178 -7.2236E-10       119
 //E33   1.0581E-09       239 -8.9407E-10        58
 //E12   9.6411E-10       119 -4.1741E-10        60
 //E23   2.1837E-09       119 -2.2050E-09       178
 //E31   2.5589E-10       176 -1.0737E-09       119
 //S11   5.7325E-05       178 -6.8164E-05       119
 //S22   6.0945E-05       178 -7.0539E-05       119
 //S33   6.1843E-05       178 -7.1200E-05       119
 //S12   1.9176E-06       119 -8.3014E-07        60
 //S23   4.3640E-06       119 -4.4100E-06       178
 //S31   5.1153E-07       176 -2.1545E-06       119
 //SMS   9.4730E-06       119  1.0387E-41         2
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    2.7619E-10      3121 -1.4312E-10   1001067
 //U2    2.6490E-10   1000353 -5.5027E-10      3121
 //U3    2.8724E-10      5121 -1.8462E-07      3121
 //V1    5.5239E-02      3121 -2.8624E-02   1001067
 //V2    5.2980E-02   1000353 -1.1005E-01      3121
 //V3    5.7448E-02      5121 -3.6925E+01      3121
 //A1    1.1048E+07      3121 -5.7248E+06   1001067
 //A2    1.0596E+07   1000353 -2.2011E+07      3121
 //A3    1.1490E+07      5121 -7.3849E+09      3121
 //E11   4.3811E-10   1000347 -3.4878E-10   1000709
 //E22   2.1254E-09   1000352 -2.1246E-09   1001427
 //E33   3.6982E-07   1001067 -3.6953E-07   1000347
 //E12   1.1329E-09   1000709 -1.1050E-09   1000352
 //E23   3.6823E-07   1000713 -3.6823E-07   1000353
 //E31   9.1796E-08   1001031 -2.7687E-07      3121
 //S11   1.4231E-02   1001067 -1.4214E-02   1000347
 //S22   1.4226E-02   1001067 -1.4210E-02   1000347
 //S33   1.5711E-02   1001067 -1.5694E-02   1000347
 //S12   2.2266E-06   1000709 -2.2148E-06   1000352
 //S23   7.3645E-04   1000713 -7.3646E-04   1000353
 //S31   1.8358E-04   1001031 -5.5379E-04      3121
 //SMS   1.7350E-03   1000347  3.4921E-43      1201
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   6.5755E-11       176 -1.7785E-10       178
 //E22   7.2706E-10       178 -7.2236E-10       119
 //E33   1.0581E-09       239 -8.9407E-10        58
 //E12   9.6411E-10       119 -4.1741E-10        60
 //E23   2.1837E-09       119 -2.2050E-09       178
 //E31   2.5589E-10       176 -1.0737E-09       119
 //S11   5.7325E-05       178 -6.8164E-05       119
 //S22   6.0945E-05       178 -7.0539E-05       119
 //S33   6.1843E-05       178 -7.1200E-05       119
 //S12   1.9176E-06       119 -8.3014E-07        60
 //S23   4.3640E-06       119 -4.4100E-06       178
 //S31   5.1153E-07       176 -2.1545E-06       119
 //SMS   9.4730E-06       119  1.0387E-41         2

 ...

#### Result step=500000
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    5.2758E-02      5221 -5.2989E-02      1021
 //U2    1.0426E-03   1000750 -7.2090E-02      1021
 //U3    0.0000E+00      1001 -6.5930E-01      3121
 //V1    1.6455E+03      5217 -1.6058E+03      1019
 //V2    8.7881E+02   1000132 -2.2448E+03      1221
 //V3    1.9455E+02   1000003 -1.7874E+04      1221
 //A1    5.7327E+09      5221 -4.0019E+09      1203
 //A2    7.0089E+09      3221 -4.3280E+09   1000713
 //A3    4.9140E+09      1211 -8.4702E+09      5021
 //E11   1.0808E-02   1001104 -1.0904E-02      1003
 //E22   3.4008E-03      1203 -3.2574E-03      5203
 //E33   3.6321E-03      1103 -3.5530E-03   1001105
 //E12   3.4760E-03      1201 -2.9256E-03      1001
 //E23   9.1779E-04   1000017 -1.3462E-03      1001
 //E31   6.4324E-04   1000750 -2.5756E-03      1101
 //S11   5.0236E+01      5201 -5.2624E+01      1001
 //S22   1.8815E+01   1001098 -2.2485E+01   1000008
 //S33   1.2848E+01   1001098 -2.2485E+01   1000008
 //S12   5.3477E+00      1201 -4.5009E+00      1001
 //S23   1.4120E+00   1000017 -2.0711E+00      1001
 //S31   9.8961E-01   1000750 -3.9625E+00      1101
 //SMS   4.4276E+01      1003  6.5308E-01   1000690
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   8.0789E-03       186 -8.0188E-03      1001
 //E22   2.1847E-03         8 -2.1148E-03       192
 //E33   2.5029E-03      1001 -2.6704E-03       186
 //E12   1.1765E-03        62 -1.0144E-03         3
 //E23   4.9965E-04         6 -5.1495E-04        66
 //E31  -1.2336E-05       125 -1.4721E-03         2
 //S11   3.3432E+01       186 -3.4438E+01         2
 //S22   7.0191E+00       183 -7.4671E+00         2
 //S33   7.6014E+00       183 -7.7370E+00         2
 //S12   1.8099E+00        62 -1.5605E+00         3
 //S23   7.6870E-01         6 -7.9223E-01        66
 //S31  -1.8979E-02       125 -2.2647E+00         2
 //SMS   3.1742E+01       186  7.0883E-01       114
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    5.2758E-02      5221 -5.2989E-02      1021
 //U2    1.0426E-03   1000750 -7.2090E-02      1021
 //U3    0.0000E+00      1001 -6.5930E-01      3121
 //V1    1.6455E+03      5217 -1.6058E+03      1019
 //V2    8.7881E+02   1000132 -2.2448E+03      1221
 //V3    1.9455E+02   1000003 -1.7874E+04      1221
 //A1    5.7327E+09      5221 -4.0019E+09      1203
 //A2    7.0089E+09      3221 -4.3280E+09   1000713
 //A3    4.9140E+09      1211 -8.4702E+09      5021
 //E11   1.0808E-02   1001104 -1.0904E-02      1003
 //E22   3.4008E-03      1203 -3.2574E-03      5203
 //E33   3.6321E-03      1103 -3.5530E-03   1001105
 //E12   3.4760E-03      1201 -2.9256E-03      1001
 //E23   9.1779E-04   1000017 -1.3462E-03      1001
 //E31   6.4324E-04   1000750 -2.5756E-03      1101
 //S11   5.0236E+01      5201 -5.2624E+01      1001
 //S22   1.8815E+01   1001098 -2.2485E+01   1000008
 //S33   1.2848E+01   1001098 -2.2485E+01   1000008
 //S12   5.3477E+00      1201 -4.5009E+00      1001
 //S23   1.4120E+00   1000017 -2.0711E+00      1001
 //S31   9.8961E-01   1000750 -3.9625E+00      1101
 //SMS   4.4276E+01      1003  6.5308E-01   1000690
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   8.0789E-03       186 -8.0188E-03      1001
 //E22   2.1847E-03         8 -2.1148E-03       192
 //E33   2.5029E-03      1001 -2.6704E-03       186
 //E12   1.1765E-03        62 -1.0144E-03         3
 //E23   4.9965E-04         6 -5.1495E-04        66
 //E31  -1.2336E-05       125 -1.4721E-03         2
 //S11   3.3432E+01       186 -3.4438E+01         2
 //S22   7.0191E+00       183 -7.4671E+00         2
 //S33   7.6014E+00       183 -7.7370E+00         2
 //S12   1.8099E+00        62 -1.5605E+00         3
 //S23   7.6870E-01         6 -7.9223E-01        66
 //S31  -1.8979E-02       125 -2.2647E+00         2
 //SMS   3.1742E+01       186  7.0883E-01       114
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_14.md | TITLE: 비선형 접촉 동적 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_14.html -->

# 비선형 접촉 동적 해석

 본 해석에는 `tutorial/14_dynamic_plate_contact`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 바닥면으로 각재가 낙하할 때의 충격 해석이며, 형상은 그림 4.14.1에, 메시 데이터는 그림 4.14.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 비선형 접촉 동적 해석 | !SOLUTION,TYPE=DYNAMIC !DYNAMIC,TYPE=NONLINEAR !CONTACT |  |
| 절점 수 | 10,712 |  |  |
| 요소 수 | 8,232 |  |  |
| 요소 유형 | 8절점 육면체 요소 | !ELEMENT,TYPE=361 |  |
| 재료명 | M1, M2 | !MATERIAL,NAME=M1 !MATERIAL,NAME=M2 |  |
| 재료 특성 | ELASTIC, PLASTIC | !ELASTIC !PLASTIC |  |
| 경계 조건 | 구속, 초기 속도 | !VELOCITY,TYPE=INITIAL |  |
| 행렬 해법 | 직접법 | !SOLVER,METHOD=MUMPS |  |

 ![바닥면과 각재의 형상](/examples/image/tutorial14_01.png)

  그림 4.14.1 바닥면과 각재의 형상  ![바닥면과 각재의 메시 데이터](/examples/image/tutorial14_02.png)

  그림 4.14.2 바닥면과 각재의 메시 데이터  
## 해석 내용

 해석 대상 각재에 초기 속도4427mm/s를 설정하고 접촉 동적 해석을 수행합니다. 해석 제어 데이터를 아래에 나타냅니다.

 
### 해석 제어 데이터 `plateToGround.cnt`

 
```text
!!
!! Control File for FISTR
!!
!VERSION
 3
!WRITE,LOG,FREQUENCY=10
!WRITE,RESULT,FREQUENCY=10
!SOLUTION, TYPE=DYNAMIC
!DYNAMIC, TYPE=NONLINEAR
 1 , 1
 0.0, 1.0, 200, 1.0000e-8
 0.65, 0.330625
 1, 1, 0.0, 0.0
 20, 2621, 1
 1, 1, 1, 1, 1, 1
!BOUNDARY, GRPID = 1
  bottom, 1, 3, 0.0
!VELOCITY, TYPE = INITIAL
  plate, 3, 3, -4427.0
!CONTACT_ALGO, TYPE=SLAGRANGE
!CONTACT, GRPID=1, INTERACTION=FSLID
  CP1, 0.0, 1.0e+5
!STEP, CONVERG=1.0e-5, ITMAX=100
 BOUNDARY, 1
 CONTACT, 1
!MATERIAL, NAME = M1
!ELASTIC
2.00000e+5, 0.3
!PLASTIC
 1.0e+8, 0.0
!MATERIAL, NAME = M2
!ELASTIC
 1.16992e+5, 0.3
!PLASTIC
 70.0, 0.0
!SOLVER,METHOD=MUMPS
!END
```

 
## 해석 결과

 낙하 충격 시의 미제스 응력 등고선을 그림 4.14.3에 나타냅니다. 또한 해석 결과의 수치 데이터로서 모니터링 절점의 에너지 출력 파일(dyna_energy.txt)의 일부를 아래에 나타냅니다.

 ![낙하 충격 시의 미제스 응력](/examples/image/tutorial14_03.png)

  그림 4.14.3 낙하 충격 시의 미제스 응력  
### 모니터링 절점의 에너지 `dyna_energy.txt`

 
```text
time step     time      kinetic energy   strain energy   total energy
         0  0.0000E+000     9.7816E-003     0.0000E+000     9.7816E-003
         1  1.0000E-008     9.7756E-003     4.9470E-006     9.7806E-003
         2  2.0000E-008     9.7654E-003     1.4636E-005     9.7800E-003
         3  3.0000E-008     9.7566E-003     2.2609E-005     9.7792E-003
         4  4.0000E-008     9.7505E-003     3.7965E-005     9.7884E-003
         5  5.0000E-008     9.7425E-003     6.4932E-005     9.8074E-003
         6  6.0000E-008     9.7214E-003     8.4571E-005     9.8060E-003
         7  7.0000E-008     9.7139E-003     9.0613E-005     9.8045E-003
         8  8.0000E-008     9.7184E-003     1.0958E-004     9.8280E-003
         9  9.0000E-008     9.7175E-003     1.5717E-004     9.8747E-003
        10  1.0000E-007     9.6909E-003     1.7998E-004     9.8709E-003
        11  1.1000E-007     9.6917E-003     1.9733E-004     9.8890E-003
        12  1.2000E-007     9.7137E-003     2.2403E-004     9.9377E-003
        13  1.3000E-007     9.6813E-003     2.4397E-004     9.9253E-003
```

 
### 해석 결과 로그 `0.log`

  
```text
fstr_setup: OK
#### Result step=    10
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    7.6111E-05      2780 -1.3733E-04      2807
 //U2    3.5826E-04      2645 -3.5826E-04      2597
 //U3    7.3650E-07      2599 -4.8426E-04      2785
 //V1    1.0743E+03      2780 -2.1388E+03      2785
 //V2    4.1975E+03      2645 -4.1975E+03      2597
 //V3    3.4753E+02      2632 -5.2027E+03      2785
 //A1    1.8419E+10      2797 -1.7512E+10      2790
 //A2    1.6413E+10      2645 -1.6413E+10      2597
 //A3    2.6159E+11      2632 -7.1824E+11      2598
 //E11   5.5643E-03      2798 -3.1738E-03      2785
 //E22   9.6885E-03      2598 -1.4502E-04      2780
 //E33   2.6801E-03      2785 -1.5067E-02      2598
 //E12   5.7146E-03      2778 -5.7146E-03      2782
 //E23   8.8189E-03      2597 -8.8189E-03      2645
 //E31   9.6709E-03      2785 -8.9273E-03      2780
 //S11   2.3553E+01      2890 -6.9294E+01      2621
 //S22   3.9504E+01      2797 -6.4266E+01      2621
 //S33   1.5165E+01      2827 -1.3217E+02      2621
 //S12   1.9751E+01      2778 -1.9751E+01      2782
 //S23   1.9616E+01      2838 -1.9616E+01      2842
 //S31   3.7803E+01      2868 -3.4973E+01      2867
 //SMS   7.7288E+01      2840  1.4896E-13     10709
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   3.3315E-03      2188 -3.5688E-03      2198
 //E22   4.2263E-03      2185 -1.6011E-05      1817
 //E33   2.8970E-03      2198 -7.3958E-03      2185
 //E12   4.8924E-03      2186 -4.8924E-03      2192
 //E23   3.4853E-03      2205 -3.4853E-03      2208
 //E31   1.1270E-02      2187 -9.6668E-03      2188
 //S11   2.6052E+01      2241 -7.3461E+01      2189
 //S22   5.6465E+01      2194 -7.8034E+01      2189
 //S33   7.5395E+01      2194 -1.1968E+02      2189
 //S12   1.3438E+01      2186 -1.3438E+01      2192
 //S23   1.4576E+01      2205 -1.4576E+01      2208
 //S31   3.5356E+01      2217 -3.3931E+01      2220
 //SMS   6.8689E+01      2233  1.4239E-13      8230
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    7.6111E-05      2780 -1.3733E-04      2807
 //U2    3.5826E-04      2645 -3.5826E-04      2597
 //U3    7.3650E-07      2599 -4.8426E-04      2785
 //V1    1.0743E+03      2780 -2.1388E+03      2785
 //V2    4.1975E+03      2645 -4.1975E+03      2597
 //V3    3.4753E+02      2632 -5.2027E+03      2785
 //A1    1.8419E+10      2797 -1.7512E+10      2790
 //A2    1.6413E+10      2645 -1.6413E+10      2597
 //A3    2.6159E+11      2632 -7.1824E+11      2598
 //E11   5.5643E-03      2798 -3.1738E-03      2785
 //E22   9.6885E-03      2598 -1.4502E-04      2780
 //E33   2.6801E-03      2785 -1.5067E-02      2598
 //E12   5.7146E-03      2778 -5.7146E-03      2782
 //E23   8.8189E-03      2597 -8.8189E-03      2645
 //E31   9.6709E-03      2785 -8.9273E-03      2780
 //S11   2.3553E+01      2890 -6.9294E+01      2621
 //S22   3.9504E+01      2797 -6.4266E+01      2621
 //S33   1.5165E+01      2827 -1.3217E+02      2621
 //S12   1.9751E+01      2778 -1.9751E+01      2782
 //S23   1.9616E+01      2838 -1.9616E+01      2842
 //S31   3.7803E+01      2868 -3.4973E+01      2867
 //SMS   7.7288E+01      2840  1.4896E-13     10709
  ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   3.3315E-03      2188 -3.5688E-03      2198
 //E22   4.2263E-03      2185 -1.6011E-05      1817
 //E33   2.8970E-03      2198 -7.3958E-03      2185
 //E12   4.8924E-03      2186 -4.8924E-03      2192
 //E23   3.4853E-03      2205 -3.4853E-03      2208
 //E31   1.1270E-02      2187 -9.6668E-03      2188
 //S11   2.6052E+01      2241 -7.3461E+01      2189
 //S22   5.6465E+01      2194 -7.8034E+01      2189
 //S33   7.5395E+01      2194 -1.1968E+02      2189
 //S12   1.3438E+01      2186 -1.3438E+01      2192
 //S23   1.4576E+01      2205 -1.4576E+01      2208
 //S31   3.5356E+01      2217 -3.3931E+01      2220
 //SMS   6.8689E+01      2233  1.4239E-13      8230

 ...

#### Result step=   200
 ##### Local Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    1.0996E-03      2818 -2.9048E-03      2803
 //U2    8.2012E-03      2645 -8.2012E-03      2597
 //U3    1.1800E-06      2465 -8.5895E-03      5373
 //V1    8.9566E+02     10712 -2.1228E+03      2831
 //V2    3.4246E+03      2645 -3.4246E+03      2597
 //V3    3.2884E+02      2794 -4.0108E+03      5377
 //A1    2.1058E+09      9165 -1.2760E+09      4757
 //A2    7.5110E+08      2769 -7.5110E+08      2478
 //A3    4.4278E+09      8975 -1.0747E+09      2465
 //E11   5.4925E-02      2838 -4.1160E-02      2788
 //E22   2.2537E-01      2621 -4.2604E-04      2921
 //E33   1.4576E-02      2815 -2.7463E-01      2621
 //E12   9.7058E-02      2789 -9.7058E-02      2791
 //E23   7.4272E-02      2779 -7.4272E-02      2781
 //E31   1.3776E-01      2785 -1.5350E-01      2780
 //S11   4.2677E+01      3113 -6.9474E+01      2621
 //S22   4.7637E+01      2815 -4.2010E+01      2921
 //S33   1.6261E+01      2853 -1.1496E+02      2780
 //S12   2.2732E+01      2830 -2.2732E+01      2826
 //S23   4.0309E+01      2597 -4.0309E+01      2645
 //S31   3.8697E+01      2913 -3.7722E+01      3054
 //SMS   9.0611E+01      2597  6.2561E-02      9662
 ##### Local Summary @Element :Max/IdMax/Min/IdMin####
 //E11   5.4784E-02      2209 -4.6698E-02      2201
 //E22   1.8822E-01      2188 -7.6612E-05      2323
 //E33   2.2781E-02      2202 -2.2608E-01      2188
 //E12   1.0327E-01      2193 -1.0327E-01      2196
 //E23   4.0800E-02      2210 -4.0800E-02      2211
 //E31   1.3391E-01      2189 -1.7023E-01      2194
 //S11   4.3368E+01      2434 -7.5148E+01      2188
 //S22   6.8745E+01      2203 -4.7855E+01      2048
 //S33   7.0360E+01      2202 -1.1351E+02      2188
 //S12   1.9354E+01      2193 -1.9354E+01      2196
 //S23   1.4194E+01      1796 -1.4194E+01      1754
 //S31   3.8915E+01      2367 -3.3190E+01      2298
 //SMS   6.9701E+01      2187  3.8349E-02      7504
 ##### Global Summary @Node    :Max/IdMax/Min/IdMin####
 //U1    1.0996E-03      2818 -2.9048E-03      2803
 //U2    8.2012E-03      2645 -8.2012E-03      2597
 //U3    1.1800E-06      2465 -8.5895E-03      5373
 //V1    8.9566E+02     10712 -2.1228E+03      2831
 //V2    3.4246E+03      2645 -3.4246E+03      2597
 //V3    3.2884E+02      2794 -4.0108E+03      5377
 //A1    2.1058E+09      9165 -1.2760E+09      4757
 //A2    7.5110E+08      2769 -7.5110E+08      2478
 //A3    4.4278E+09      8975 -1.0747E+09      2465
 //E11   5.4925E-02      2838 -4.1160E-02      2788
 //E22   2.2537E-01      2621 -4.2604E-04      2921
 //E33   1.4576E-02      2815 -2.7463E-01      2621
 //E12   9.7058E-02      2789 -9.7058E-02      2791
 //E23   7.4272E-02      2779 -7.4272E-02      2781
 //E31   1.3776E-01      2785 -1.5350E-01      2780
 //S11   4.2677E+01      3113 -6.9474E+01      2621
 //S22   4.7637E+01      2815 -4.2010E+01      2921
 //S33   1.6261E+01      2853 -1.1496E+02      2780
 //S12   2.2732E+01      2830 -2.2732E+01      2826
 //S23   4.0309E+01      2597 -4.0309E+01      2645
 //S31   3.8697E+01      2913 -3.7722E+01      3054
 //SMS   9.0611E+01      2597  6.2561E-02      9662
 ##### Global Summary @Element :Max/IdMax/Min/IdMin####
 //E11   5.4784E-02      2209 -4.6698E-02      2201
 //E22   1.8822E-01      2188 -7.6612E-05      2323
 //E33   2.2781E-02      2202 -2.2608E-01      2188
 //E12   1.0327E-01      2193 -1.0327E-01      2196
 //E23   4.0800E-02      2210 -4.0800E-02      2211
 //E31   1.3391E-01      2189 -1.7023E-01      2194
 //S11   4.3368E+01      2434 -7.5148E+01      2188
 //S22   6.8745E+01      2203 -4.7855E+01      2048
 //S33   7.0360E+01      2202 -1.1351E+02      2188
 //S12   1.9354E+01      2193 -1.9354E+01      2196
 //S23   1.4194E+01      1796 -1.4194E+01      1754
 //S31   3.8915E+01      2367 -3.3190E+01      2298
 //SMS   6.9701E+01      2187  3.8349E-02      7504
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_15.md | TITLE: 모드 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_15.html -->

# 모드 해석

 이 해석에는 `tutorial/15_eigen_spring`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 앞 절의 [비선형 정적 해석(초탄성 2)](tutorial_04.html)과 동일한 스프링입니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 모드 해석 | !SOLUTION, TYPE=EIGEN |  |
| 절점 수 | 78,771 |  |  |
| 요소 수 | 46,454 |  |  |
| 요소 유형 | 10절점 2차 사면체 요소 | !ELEMENT,TYPE=342 |  |
| 재료명 | MAT1 | !MATERIAL,NAME=MAT1 |  |
| 경계 조건 | 구속 |  |  |
| 행렬 해법 | 직접법 | !SOLVER,METHOD=MUMPS |  |

 
## 해석 내용

 그림 4.4.1에 표시된 구속면의 변위를 구속하고 5차 모드까지 모드 해석을 수행합니다. 해석 제어 데이터를 아래에 나타냅니다.

 ![스프링 형상](/examples/image/tutorial04_01.png)

  그림 4.4.1 스프링 형상  
### 해석 제어 데이터 `spring.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!SOLUTION, TYPE=EIGEN
!EIGEN
 5, 1.0E-8, 60
!WRITE,RESULT
!WRITE,VISUAL
## Solver Control
### Boundary Condition
!BOUNDARY
 XFIX, 1, 1, 0.0
 YFIX, 2, 2, 0.0
 ZFIX, 3, 3, 0.0
### Material
# define in mesh file
### Solver Setting
!SOLVER,METHOD=MUMPS
## Post Control
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 fistr1을 실행합니다.

 
```text
cd FrontISTR/tutorial/15_eigen_spring
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 해석 결과 데이터 파일 spring.res.0.3을 사용하여 3차 진동 모드(스프링의 y 방향 압축·신장)를 REVOCAP_PrePost로 작성하고 그림 4.15.1에 나타냅니다. 변형 배율은 10으로 설정합니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일에 출력된 고유진동수 목록을 아래에 나타냅니다.

 ![스프링의 3차 진동 모드](/examples/image/tutorial15_01.png)

  그림 4.15.1 스프링의 3차 진동 모드  
### 해석 결과 로그 `0.log`

 
```text
********************************
*RESULT OF EIGEN VALUE ANALYSIS*
********************************

### NUMBER OF ITERATIONS =       20
### TOTAL MASS =   3.4184E-06

       PERIOD     FREQUENCY  PARTICIPATION FACTOR             EFFECTIVE MASS
  NO.  [Sec]      [HZ]       X          Y          Z          X          Y          Z
  ---  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------
    1  2.245E-03  4.454E+02  1.029E+00 -8.894E-02 -7.052E-01  1.301E-06  9.718E-09  6.109E-07
    2  2.239E-03  4.465E+02  6.969E-01  1.075E-01  1.011E+00  6.129E-07  1.460E-08  1.289E-06
    3  1.100E-03  9.087E+02  4.862E-03  1.236E+00 -7.917E-02  4.007E-11  2.591E-06  1.062E-08
    4  1.014E-03  9.858E+02 -2.965E-02  3.385E-01 -6.782E-03  9.823E-10  1.280E-07  5.138E-11
    5  5.525E-04  1.810E+03 -5.867E-01 -5.386E-02 -7.659E-01  2.838E-07  2.391E-09  4.837E-07
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_16.md | TITLE: 열전도 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_16.html -->

# 열전도 해석

 이 해석에는 `tutorial/16_heat_block`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 구멍이 있는 블록이며, 형상은 그림 4.16.1에, 메시 데이터는 그림 4.16.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 열전도 해석 | !SOLUTION,TYPE=HEAT |  |
| 절점 수 | 37,386 |  |  |
| 요소 수 | 32,160 |  |  |
| 요소 유형 | 8절점 1차 육면체 요소 | !ELEMENT,TYPE=361 |  |
| 재료명 | AL | !MATERIAL,NAME=AL |  |
| 경계 조건 | 규정 온도 | !FIXTEMP |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 ![구멍이 있는 블록의 형상](/examples/image/tutorial16_01.png)

  그림 4.16.1 구멍이 있는 블록의 형상  ![구멍이 있는 블록의 메시 데이터](/examples/image/tutorial16_02.png)

  그림 4.16.2 구멍이 있는 블록의 메시 데이터  
## 해석 내용

 해석 대상의 원통형 내면에 열원을 부여하는 정상 열전도 해석을 수행합니다. 해석 제어 데이터를 아래에 나타냅니다.

 
### 해석 제어 데이터 `block.cnt`

 
```text
#  Control File for FISTR
## Analysis Control
!VERSION
 3
!SOLUTION, TYPE=HEAT
!HEAT
 0.0
!WRITE,RESULT
!WRITE,VISUAL
## Solver Control
### Boundary Condition
!FIXTEMP
 FTMPC,  100.0
 FTMPS1, 20.0
 FTMPS2, 20.0
 FTMPS3, 20.0
 FTMPS4, 20.0
### Material
# define in mesh file
### Solver Setting
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=YES,TIMELOG=YES
 100, 1
 1.0e-8, 1.0, 0.0
## Post Control
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 FrontISTR 실행 명령 fistr1을 실행합니다.

 
```text
cd FrontISTR/tutorial/16_heat_block
fistr1 -t 4
```

 (4개 스레드로 실행)

 
## 해석 결과

 REVOCAP_PrePost로 작성한 온도 등고선도를 그림 4.16.3에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![온도 해석 결과](/examples/image/tutorial16_03.png)

  그림 4.16.3 온도 해석 결과  
### 해석 결과 로그 파일 `0.log`

 
```text
fstr_setup: OK

 ISTEP =     1
 Time  =     0.000
 Maximum Temperature :   100.000
 Maximum Node No.    :         9
 Minimum Temperature :    20.000
 Minimum Node No.    :        85
 Maximum Temperature(global) :   100.000
 Minimum Temperature(global) :    20.000
```


---


<!-- ========================================== -->
<!-- FILE: examples/tutorial_17.md | TITLE: 주파수 응답 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/examples/tutorial_17.html -->

# 주파수 응답 해석

 이 해석에는 `tutorial/17_freq_beam`의 데이터를 사용합니다.

 
## 해석 대상

 해석 대상은 외팔보이며, 형상은 그림 4.17.1에, 메시 데이터는 그림 4.17.2에 나타냅니다.

 
| 항목 | 내용 | 비고 | 참조 |
| --- | --- | --- | --- |
| 해석 종류 | 주파수 응답 해석 | !SOLUTION,TYPE=EIGEN !SOLUTION,TYPE=DYNAMIC |  |
| 절점 수 | 55 |  |  |
| 요소 수 | 126 |  |  |
| 요소 유형 | 4절점 1차 사면체 요소 | !ELEMENT,TYPE=341 |  |
| 재료명 | Material-1 | !MATERIAL,NAME=Material-1 |  |
| 경계 조건 | 구속, 집중 하중, 고유값 | !EIGENREAD |  |
| 행렬 해법 | CG/SSOR | !SOLVER,METHOD=CG,PRECOND=1 |  |

 ![외팔보의 형상](/examples/image/tutorial17_01.png)

  그림 4.17.1 외팔보의 형상  ![외팔보의 메시 데이터](/examples/image/tutorial17_02.png)

  그림 4.17.2 외팔보의 메시 데이터  
## 해석 내용

 해석 대상 외팔보의 한쪽 끝을 완전히 구속하고 반대쪽 끝의 두 절점에 집중 하중을 가하여 주파수 응답 해석을 수행합니다.

 절차로서 동일한 경계 조건에서 10차 모드까지 모드 해석을 수행한 후, 5차 모드까지의 고유값과 고유벡터를 사용하여 해석합니다. 주파수 응답 해석용 해석 제어 데이터를 아래에 나타냅니다.

 
### 해석 제어 데이터 `beam_eigen.cnt`

 
```text
#  Control File for FISTR
!VERSION
 3
!WRITE,RESULT
!WRITE,VISUAL
!SOLUTION, TYPE=EIGEN
!EIGEN
 10, 1.0E-8, 60
!BOUNDARY
_PickedSet4, 1, 3, 0.0
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=NO,TIMELOG=YES
 10000, 1
 1.0e-8, 1.0, 0.0
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
### 해석 제어 데이터 `beam_freq.cnt`

 
```text
#  Control File for FISTR
!VERSION
 3
!WRITE,RESULT
!WRITE,VISUAL
!SOLUTION, TYPE=DYNAMIC
!DYNAMIC
 11 , 2
 14000, 16000, 20, 15000.0
 0.0, 6.6e-5
 1, 1, 0.0, 7.2E-7
 10, 2, 1
 1, 1, 1, 1, 1, 1
!EIGENREAD
 eigen_0.log
 1, 5
!BOUNDARY
_PickedSet4, 1, 3, 0.0
!FLOAD, LOAD CASE=2
_PickedSet5, 2, 1.
!FLOAD, LOAD CASE=2
_PickedSet6, 2, 1.
!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=NO,TIMELOG=YES
 10000, 1
 1.0e-8, 1.0, 0.0
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
!END
```

 
## 해석 절차

 먼저 모드 해석용 전체 제어 데이터 `hecmw_ctrl_eigen.dat`의 이름을 `hecmw_ctrl.dat`로 변경하고 모드 해석을 수행합니다.

 다음으로 주파수 응답 해석용 전체 제어 데이터 `hecmw_ctrl_freq.dat`의 이름을 `hecmw_ctrl.dat`로 변경하고, 모드 해석의 해석 결과 로그 파일 `0.log`를 `eigen_0.log`로 변경한 뒤(주파수 응답 해석용 해석 제어 데이터에서 지정됨) 주파수 응답 해석을 수행합니다.

 
```text
cp hecmw_ctrl_eigen.dat hecmw_ctrl.dat
fistr1 -t 4
mv 0.log eigen_0.log
cp hecmw_ctrl_freq.dat hecmw_ctrl.dat
fistr1 -t 4
```

 
## 해석 결과

 해석 제어 데이터에서 지정한 모니터링 절점(절점 번호 1)의 주파수와 변위 진폭의 관계를 Microsoft Excel로 작성하여 그림 4.17.3에 나타냅니다. 또한 해석 결과의 수치 데이터로서 해석 결과 로그 파일의 일부를 아래에 나타냅니다.

 ![모니터링 절점의 주파수와 변위 진폭의 관계](/examples/image/tutorial17_03.png)

  그림 4.17.3 모니터링 절점의 주파수와 변위 진폭의 관계  
### 해석 결과 로그 `0.log` (주파수 응답 해석)

  
```text
fstr_setup: OK
 Rayleigh alpha:   0.0000000000000000     
 Rayleigh beta:   7.1999999999999999E-007
 read from=eigen_0.log
 start mode=           1
 end mode=           5
 start frequency:   14000.000000000000     
 end frequency:   16000.000000000000     
 number of the sampling points          20
 monitor nodeid=           1
   14100.000000000000      [Hz] :    8.3935554529723885E-002
   14100.000000000000      [Hz] :            1 .res
   14200.000000000000      [Hz] :    9.1211083509607632E-002
   14200.000000000000      [Hz] :            2 .res
   14300.000000000000      [Hz] :    9.9579777896922961E-002
   14300.000000000000      [Hz] :            3 .res
   14400.000000000000      [Hz] :   0.10914967594967491     
   14400.000000000000      [Hz] :            4 .res
   14500.000000000000      [Hz] :   0.11992223203326918     
   14500.000000000000      [Hz] :            5 .res
   14600.000000000000      [Hz] :   0.13164981801806747     
   14600.000000000000      [Hz] :            6 .res
   14700.000000000000      [Hz] :   0.14360931008440975     
   14700.000000000000      [Hz] :            7 .res
   14800.000000000000      [Hz] :   0.15436500205940235     
   14800.000000000000      [Hz] :            8 .res
   14900.000000000000      [Hz] :   0.16180768408076251     
   14900.000000000000      [Hz] :            9 .res
   15000.000000000000      [Hz] :   0.16388019610373711     
   15000.000000000000      [Hz] :           10 .res
   15100.000000000000      [Hz] :   0.15982110598747551     
   15100.000000000000      [Hz] :           11 .res
   15200.000000000000      [Hz] :   0.15074650286398145     
   15200.000000000000      [Hz] :           12 .res
   15300.000000000000      [Hz] :   0.13885370598993371     
   15300.000000000000      [Hz] :           13 .res
   15400.000000000000      [Hz] :   0.12618976409021948     
   15400.000000000000      [Hz] :           14 .res
   15500.000000000000      [Hz] :   0.11405716994112736     
   15500.000000000000      [Hz] :           15 .res
   15600.000000000000      [Hz] :   0.10306231010139058     
   15600.000000000000      [Hz] :           16 .res
   15700.000000000000      [Hz] :    9.3374567545990342E-002
   15700.000000000000      [Hz] :           17 .res
   15800.000000000000      [Hz] :    8.4945897112663621E-002
   15800.000000000000      [Hz] :           18 .res
   15900.000000000000      [Hz] :    7.7641947016103510E-002
   15900.000000000000      [Hz] :           19 .res
   16000.000000000000      [Hz] :    7.1307642422355627E-002
   16000.000000000000      [Hz] :           20 .res
 start time:   0.0000000000000000     
 end time:   6.6000000000000005E-005
 frequency:   15000.000000000000     
 node id:           1
 num disp:          10
 time=   0.0000000000000000       :            1 .res
 time=   0.0000000000000000       :            1 .vis
 time=   6.6000000000000003E-006  :            2 .res
 time=   6.6000000000000003E-006  :            2 .vis
 time=   1.3200000000000001E-005  :            3 .res
 time=   1.3200000000000001E-005  :            3 .vis
 time=   1.9800000000000000E-005  :            4 .res
 time=   1.9800000000000000E-005  :            4 .vis
 time=   2.6400000000000001E-005  :            5 .res
 time=   2.6400000000000001E-005  :            5 .vis
 time=   3.3000000000000003E-005  :            6 .res
 time=   3.3000000000000003E-005  :            6 .vis
 time=   3.9600000000000000E-005  :            7 .res
 time=   3.9600000000000000E-005  :            7 .vis
 time=   4.6200000000000005E-005  :            8 .res
 time=   4.6200000000000005E-005  :            8 .vis
 time=   5.2800000000000003E-005  :            9 .res
 time=   5.2800000000000003E-005  :            9 .vis
 time=   5.9400000000000000E-005  :           10 .res
 time=   5.9400000000000000E-005  :           10 .vis
```


---


<!-- ========================================== -->
<!-- FILE: showcase/index.md | TITLE: 사례집 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/showcase/index.html -->

# 사례집

 실제 형상의 모델을 FrontISTR로 해석한 사례를 소개한다. 각 페이지에서는 하나의 모델을 다루며, 메시 규모, 해석 조건 및 계산 환경의 차이를 해당 페이지 안에서 비교한다.

 
| 사례 | 주요 내용 | 규모 |
| --- | --- | --- |
| [금형](mold.html) | 메시 규모에 따른 변위·응력의 변화와 대규모 병렬 실행 | 약 100만~약 2억 1천만 절점 |

 게재된 수치는 해당 조건에서 계산한 기록이며 정답값이나 합격·불합격 판정 기준이 아니다. 이론해 또는 다른 소프트웨어와의 비교를 통한 검증은 [V&V](../verification_and_validation/01_simple.html)를 참조한다.


---


<!-- ========================================== -->
<!-- FILE: showcase/mold.md | TITLE: 대규모 금형 모델의 정적 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/showcase/mold.html -->

# 대규모 금형 모델의 정적 해석

 금형 어셈블리에 균일 분포 압력을 가하는 선형 정적 해석 사례이다. 2차 사면체 요소를 사용하며 약 100만 절점부터 약 2.1억 절점까지 4종류의 메시가 있다.

 가장 큰 Mold_200mil은 13노드 208프로세스로 실행한 실적이 있으며, 실무에서 사용되는 형상의 모델을 어느 규모까지 다룰 수 있는지를 보여 주는 사례로 활용할 수 있다. 4종류는 형상·재료·경계 조건이 같고 메시의 세밀함만 다르므로, 규모가 정확도와 계산 자원에 미치는 영향도 비교할 수 있다.

 ![금형 모델의 전체 형상, 하부 형상 및 단면](/showcase/image/mold_overview.jpg)

  그림 1. 금형 어셈블리: 전체 부품, 하부 부품 및 절단면.  
## 해석 모델

 4종류의 모델은 형상·재료·경계 조건이 같고 메시의 세밀함이 다르다.

 
| 항목 | 내용 |
| --- | --- |
| 형상·연결 | 금형 어셈블리. 부품 간은 공유 절점으로 연결 |
| 요소 | 2차 사면체 요소 |
| 재료 | Structural Steel, 선형 탄성체. 영률 200 GPa, 포아송비 0.3 |
| 하중 | 상면에 균일 분포 압력 1 MPa(면 그룹 `PRESSURE1`) |
| 구속 | 하면의 병진 3성분을 고정(절점 그룹 `CONSTRAINT1`) |

 ![금형의 하중·구속 위치와 제품면의 요소면 그룹](/showcase/image/mold_modeling_specification_en.png)

  그림 2. 압력 하중, 변위 구속 및 제품 표면 그룹.  
## 해석 결과

 
### 메시 규모에 따른 변위와 응력

 
| 모델명 | Z 변위의 최솟값 (mm) | 최대 Mises 응력·절점값 (MPa) | 최대 Mises 응력·요소값 (MPa) |
| --- | --- | --- | --- |
| Mold_001mil | -0.2527 | 118.2 | 75.9 |
| Mold_011mil | -0.2638 | 130.7 | 101.3 |
| Mold_055mil | -0.2659 | 189.9 | 156.5 |
| Mold_200mil | -0.2666 | 250.4 | 213.1 |

 메시를 세밀하게 할수록 Z 변위 최솟값의 변화 폭은 작아지며, 마지막 두 규모의 차이는 표에 표시된 값으로 0.0007 mm이다. 한편 최대 Mises 응력은 절점값과 요소값 모두 계속 증가한다. 응력의 수렴 여부나 특이점의 유무를 판단하려면 최댓값의 위치와 주변 형상·경계 조건을 조사해야 한다.

 
| Mold_001mil | Mold_011mil |
| --- | --- |
| ![Mises 응력 확대도: Mold_001mil](/showcase/image/mold_stress_mises_detail_001mil.jpg) | ![Mises 응력 확대도: Mold_011mil](/showcase/image/mold_stress_mises_detail_011mil.jpg) |
| **Mold_055mil** | **Mold_200mil** |
| ![Mises 응력 확대도: Mold_055mil](/showcase/image/mold_stress_mises_detail_055mil.jpg) | ![Mises 응력 확대도: Mold_200mil](/showcase/image/mold_stress_mises_detail_200mil.jpg) |

  그림 3. 4가지 모델 규모의 메시와 절점 von Mises 응력(MPa).  그림 3은 절점 외삽값(`NodalMISES`)을 표시하며, 범례는 모든 그림에서 0~50 MPa로 고정되어 있다. 요소의 모서리선에서 메시의 차이를 비교할 수 있다.

 
### 대규모 실행 기록

 2020년 7월에 FrontISTR 5.1.0을 사용하여 CG법, 수렴 임계값 `1.0E-06`, 최대 반복 횟수 100,000회로 계산했다. 실행 환경은 PC 클러스터이며, 각 노드에 Xeon E5-2670(2.6 GHz, 8코어)을 2기, 메모리 128 GB를 탑재하고 InfiniBand FDR로 연결했다. 병렬 방식은 Flat MPI(1프로세스 1스레드)이다.

 
| 모델명 | MPI 프로세스 수(노드 수) | 전처리기 | 반복 횟수 | 솔버 시간 (s) |
| --- | --- | --- | --- | --- |
| Mold_001mil | 32(2) | AMG(ML) | 611 | 80.16 |
| Mold_011mil | 64(4) | AMG(ML) | 854 | 782.65 |
| Mold_055mil | 128(8) | AMG(ML) | 778 | 2,015.05 |
| Mold_200mil | 208(13) | 대각 스케일링 | 92,626 | 96,969.34 |

 시간은 실행 로그의 `solve` 값이다. Mold_200mil은 AMG를 사용하면 13노드에서도 메모리가 부족했기 때문에 전처리기를 대각 스케일링으로 변경했으며, 반복 횟수는 92,626회, 계산 시간은 약 27시간이다.

 
## 해석 데이터

 
| 모델명 | 절점 수 | 요소 수 | 압축 용량 | 다운로드 |
| --- | --- | --- | --- | --- |
| Mold_001mil | 1,015,778 | 638,041 | 35.6 MB | [Mold_001mil.tar.gz](https://gitlab.com/api/v4/projects/21515176/packages/generic/Mold_001mil/1.0.0/Mold_001mil.tar.gz) |
| Mold_011mil | 11,036,922 | 7,617,345 | 461 MB | [Mold_011mil.tar.gz](https://gitlab.com/api/v4/projects/21515176/packages/generic/Mold_011mil/1.0.0/Mold_011mil.tar.gz) |
| Mold_055mil | 55,508,692 | 39,271,921 | 2.41 GB | [Mold_055mil.tar.gz](https://gitlab.com/api/v4/projects/21515176/packages/generic/Mold_055mil/1.0.0/Mold_055mil.tar.gz) |
| Mold_200mil | 209,893,102 | 151,044,460 | 9.58 GB | 5개 분할(아래 참조) |

 데이터 버전은 `1.0.0`이다.

 
### Mold_200mil 가져오기

 Mold_200mil은 파일당 한도(5 GB)를 초과하므로 2 GiB씩 5개로 분할하여 등록되어 있다. 모든 파일을 동일한 디렉터리에 내려받은 다음 결합한다.

 
```sh
for i in 00 01 02 03 04; do
  curl -fLO "https://gitlab.com/api/v4/projects/21515176/packages/generic/Mold_200mil/1.0.0/Mold_200mil.tar.gz.part$i"
done
cat Mold_200mil.tar.gz.part?? > Mold_200mil.tar.gz
```

 결합 후 SHA-256은 `c67d5b8990302c84103de7e8a2848449f0035f7813cd122fb443e206f38b8606`이다.

 
## 참고 문헌

 
## 협력

 
- CAD 및 입력 데이터 작성: TechnoStar
- FrontISTR로 변환 및 병렬 해석: 야마구치 씨, 이나가키 씨


---


<!-- ========================================== -->
<!-- FILE: verification_and_validation/01_simple.md | TITLE: 단순 형상 모델에 의한 검증 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/verification_and_validation/01_simple.html -->

# 단순 형상 모델에 의한 검증

 
## 탄성 정적 해석

 이 검증에서는 캔틸레버 보를 그림 9.1.1과 같이 메시 분할한 모델을 대상으로 하였다. 검증 조건은 그림 9.1.2에 나타낸 하중 조건을 변경한 exA~exG의 7개 조건에 대해 해석하였다. 또한 exG는 exA와 동일한 하중 조건에서 직접법 솔버를 사용한 검증 사례이다.

 표 9.1.1~표 9.1.7에 각 하중 조건의 검증 결과를 나타낸다.

 ![캔틸레버 보의 메시 분할 예(육면체 요소)](/verification_and_validation/image/example01_01.png) 그림 9.1.1 캔틸레버 보의 메시 분할 예(육면체 요소)

 
|  |  |
| --- | --- |
| ![](/verification_and_validation/image/example01_02.png) | (a) exA,G: 집중 하중 |
| ![](/verification_and_validation/image/example01_03.png) | (b) exD: 중력 |
| ![](/verification_and_validation/image/example01_04.png) | (c) exB: 면 분포 하중 |
| ![](/verification_and_validation/image/example01_05.png) | (d) exE: 원심력 |
| ![](/verification_and_validation/image/example01_06.png) | (e) exC: 체적 하중 |
| ![](/verification_and_validation/image/example01_07.png) | (f) exF: 열 하중 |

 
| 항목 | 값 |
| --- | --- |
| 영률 | \(E = 4000.0\ \mathrm{kgf/mm^2}\) |
| 길이 | \(L = 10.0\ \mathrm{mm}\) |
| 포아송비 | \(\nu = 0.3\) |
| 단면적 | \(A = 1.0\ \mathrm{mm^2}\) |
| 질량 밀도 | \(\rho = 8.0102 \times 10^{-10}\ \mathrm{kgf}\;\mathrm{s}^2/\mathrm{mm}^4\) |
| 단면 2차 모멘트 | \(I = 1.0/12.0\ \mathrm{mm^4}\) |
| 중력 가속도 | \(g = 9800.0\ \mathrm{mm/s^2}\) |
| 선열팽창계수 | \(\alpha = 1.0 \times 10^{-5}\) |

 그림 9.1.2 캔틸레버 보 모델의 검증 조건

 표 9.1.1 exA: 집중 하중 문제의 검증 결과

 
| 사례명 | 요소 수 |  | 예측값: \(\delta_{max}= -1.000\) |  | 비고 |
| --- | --- | --- | --- | --- | --- |
|  |  | NASTRAN | 상용 소프트웨어 | FrontISTR |  |
| A231 | 40 | -0.338 | -0.371 | -0.371 | 33절점 / 평면 응력 문제 |
| A232 | 40 | -0.942 | -1.002 | -1.002 | 105절점 / 평면 응력 문제 |
| A241 | 20 | -0.720 | -0.711 | -0.711 | 33절점 / 평면 응력 문제 |
| A242 | 20 | -0.910 | -1.002 | -1.002 | 85절점 / 평면 응력 문제 |
| A341 | 240 | -0.384 | -0.384 | -0.386 | 99절점 |
| A342 | 240 | -0.990 | -0.990 | -0.999 | 525절점 |
| A351 | 80 | -0.353 | -0.355 | -0.351 | 99절점 |
| A352 | 80 | -0.993 | -0.993 | -0.992 | 381절점 |
| A361 | 40 | -0.954 | -0.985 | -0.984 | 99절점 |
| A362 | 40 | -0.994 | -0.993 | -0.993 | 220절점 |
| A731 | 40 | - | - | -0.991 | 33절점 / 직접법 |
| A741 | 20 | - | - | -0.996 | 33절점 / 직접법 |

 표 9.1.2 exB: 면 분포 하중 문제의 검증 결과

 
| 사례명 | 요소 수 |  | 예측값: \(\delta_{max}= -3.750\) |  | 비고 |
| --- | --- | --- | --- | --- | --- |
|  |  | NASTRAN | 상용 소프트웨어 | FrontISTR |  |
| B231 | 40 | -1.281 | -1.403 | -1.403 | 33절점 / 평면 응력 문제 |
| B232 | 40 | -3.579 | -3.763 | -3.763 | 105절점 / 평면 응력 문제 |
| B241 | 20 | -3.198 | -2.680 | -2.680 | 33절점 / 평면 응력 문제 |
| B242 | 20 | -3.426 | -3.765 | -3.765 | 85절점 / 평면 응력 문제 |
| B341 | 240 | -1.088 | -1.449 | -1.454 | 99절점 |
| B342 | 240 | -3.704 | -3.704 | -3.748 | 525절점 |
| B351 | 80 | -3.547 | -1.338 | -1.325 | 99절점 |
| B352 | 80 | -0.3717 | -3.716 | -3.713 | 381절점 |
| B361 | 40 | -3.557 | -3.691 | -3.688 | 99절점 |
| B362 | 40 | -3.726 | -3.717 | -3.717 | 220절점 |
| B731 | 40 | - | - | -3.722 | 33절점 / 직접법 |
| B741 | 20 | - | - | -3.743 | 33절점 / 직접법 |

 표 9.1.3 exC: 체적 하중 문제의 검증 결과

 
| 사례명 | 요소 수 |  | 예측값: \(\delta_{max} = -2.944 \times 10^{-5}\) |  | 비고 |
| --- | --- | --- | --- | --- | --- |
|  |  | NASTRAN | 상용 소프트웨어 | FrontISTR |  |
| C231 | 40 | - | -1.101e-5 | -1.101e-5 | 33절점 / 평면 응력 문제 |
| C232 | 40 | - | -2.951e-5 | -2.951e-5 | 105절점 / 평면 응력 문제 |
| C241 | 20 | - | -2.102e-5 | -2.102e-5 | 33절점 / 평면 응력 문제 |
| C242 | 20 | - | -2.953e-5 | -2.953e-5 | 85절점 / 평면 응력 문제 |
| C341 | 240 | - | -1.136e-5 | -1.140e-5 | 99절점 |
| C342 | 240 | - | -2.905e-5 | -2.937e-5 | 525절점 |
| C351 | 80 | - | -1.050e-5 | -1.039e-5 | 99절점 |
| C352 | 80 | - | -2.914e-5 | -2.911e-5 | 381절점 |
| C361 | 40 | - | -2.895e-5 | -2.893e-5 | 99절점 |
| C362 | 40 | - | -2.915e-5 | -2.915e-5 | 220절점 |
| C731 | 40 | - | - | -2.922e-5 | 33절점 / 직접법 |
| C741 | 20 | - | - | -2.938e-5 | 33절점 / 직접법 |

 표 9.1.4 exD: 중력 문제의 검증 결과

 
| 사례명 | 요소 수 |  | 예측값: \(\delta_{max} = -2.944 \times 10^{-5}\) |  | 비고 |
| --- | --- | --- | --- | --- | --- |
|  |  | NASTRAN | 상용 소프트웨어 | FrontISTR |  |
| D231 | 40 | -1.101e-5 | -1.101e-5 | -1.101e-5 | 33절점 / 평면 응력 문제 |
| D232 | 40 | -2.805e-5 | -2.951e-5 | -2.951e-5 | 105절점 / 평면 응력 문제 |
| D241 | 20 | -2.508e-5 | -2.102e-5 | -2.102e-5 | 33절점 / 평면 응력 문제 |
| D242 | 20 | -2.684e-5 | -2.953e-5 | -2.953e-5 | 85절점 / 평면 응력 문제 |
| D341 | 240 | -1.172e-5 | -1.136e-5 | -1.140e-5 | 99절점 |
| D342 | 240 | -2.906e-5 | -2.905e-5 | -2.937e-5 | 525절점 |
| D351 | 80 | -1.046e-5 | -1.050e-5 | -1.039e-5 | 99절점 |
| D352 | 80 | -2.917e-5 | -2.914e-5 | -2.911e-5 | 381절점 |
| D361 | 40 | -2.800e-5 | -2.895e-5 | -2.893e-5 | 99절점 |
| D362 | 40 | -2.919e-5 | -2.915e-5 | -2.915e-5 | 220절점 |
| D731 | 40 | - | - | -2.922e-5 | 33절점 / 직접법 |
| D741 | 20 | - | - | -2.938e-5 | 33절점 / 직접법 |

 표 9.1.5 exE: 원심력 문제의 검증 결과

 
| 사례명 | 요소 수 |  | 예측값: \(\delta_{max} = 2.635 \times 10^{-3}\) |  | 비고 |
| --- | --- | --- | --- | --- | --- |
|  |  | NASTRAN | 상용 소프트웨어 | FrontISTR |  |
| E231 | 40 | 2.410e-3 | 2.616e-3 | 2.650e-3 | 33절점 / 평면 응력 문제 |
| E232 | 40 | 2.447e-3 | 2.627e-3 | 2.628e-3 | 105절점 / 평면 응력 문제 |
| E241 | 20 | 2.386e-3 | 2.622e-3 | 2.624e-3 | 33절점 / 평면 응력 문제 |
| E242 | 20 | 2.387e-3 | 2.627e-3 | 2.629e-3 | 85절점 / 평면 응력 문제 |
| E341 | 240 | 2.708e-3 | 2.579e-3 | 2.625e-3 | 99절점 |
| E342 | 240 | 2.639e-3 | 2.614e-3 | 2.638e-3 | 525절점 |
| E351 | 80 | 2.642e-3 | 2.598e-3 | 2.625e-3 | 99절점 |
| E352 | 80 | 2.664e-3 | 2.617e-3 | 2.616e-3 | 381절점 |
| E361 | 40 | 2.611e-3 | 2.603e-3 | 2.603e-3 | 99절점 |
| E362 | 40 | 2.623e-3 | 2.616e-3 | 2.616e-3 | 220절점 |
| E731 | 40 | - | - | 2.619e-3 | 33절점 / 직접법 |
| E741 | 20 | - | - | 2.622e-3 | 33절점 / 직접법 |

 표 9.1.6 exF: 열응력 하중 문제의 검증 결과

 
| 사례명 | 요소 수 |  | 예측값: \(\delta_{max} = 1.000 \times 10^{-2}\) |  | 비고 |
| --- | --- | --- | --- | --- | --- |
|  |  | NASTRAN | 상용 소프트웨어 | FrontISTR |  |
| F231 | 40 | - | 1.016e-2 | 1.007e-2 | 33절점 / 평면 응력 문제 |
| F232 | 40 | - | 1.007e-2 | 1.007e-2 | 105절점 / 평면 응력 문제 |
| F241 | 20 | - | 1.010e-2 | 1.010e-2 | 33절점 / 평면 응력 문제 |
| F242 | 20 | - | 1.006e-2 | 1.006e-2 | 85절점 / 평면 응력 문제 |
| F341 | 240 | - | 1.047e-2 | 1.083e-2 | 99절점 |
| F342 | 240 | - | 1.018e-2 | 1.022e-2 | 525절점 |
| F351 | 80 | - | 1.031e-2 | 1.062e-2 | 99절점 |
| F352 | 80 | - | 1.015e-2 | 1.017e-2 | 381절점 |
| F361 | 40 | - | 1.026e-2 | 1.026e-2 | 99절점 |
| F362 | 40 | - | 1.016e-2 | 1.016e-2 | 220절점 |

 표 9.1.7 exG: 직접법의 검증 결과(집중 하중 문제)

 
| 사례명 | 요소 수 |  | 예측값: δmax= -1.000 |  | 비고 |
| --- | --- | --- | --- | --- | --- |
|  |  | NASTRAN | 상용 소프트웨어 | FrontISTR |  |
| G231 | 40 | -0.338 | -0.371 | -0.371 | 33절점 / 평면 응력 문제 |
| G232 | 40 | -0.942 | -1.002 | -1.002 | 105절점 / 평면 응력 문제 |
| G241 | 20 | -0.720 | -0.711 | -0.711 | 33절점 / 평면 응력 문제 |
| G242 | 20 | -0.910 | -1.002 | -1.002 | 85절점 / 평면 응력 문제 |
| G341 | 240 | -0.384 | -0.384 | -0.386 | 99절점 |
| G342 | 240 | -0.990 | -0.990 | -0.999 | 525절점 |
| G351 | 80 | -0.353 | -0.355 | -0.351 | 99절점 |
| G352 | 80 | -0.993 | -0.993 | -0.992 | 381절점 |
| G361 | 40 | -0.954 | -0.985 | -0.984 | 99절점 |
| G362 | 40 | -0.994 | -0.993 | -0.993 | 220절점 |
| G731 | 40 | - | - | -0.991 | 33절점 / 직접법 |
| G741 | 20 | - | - | -0.996 | 33절점 / 직접법 |

 
## 비선형 정적 해석

 
### (2-1) exnl1: 기하학적 비선형 해석

 검증 사례 exI의 검증 모델은 검증 사례 exA~G의 모델과 동일하다. 그림 9.1.3에 검증 모델의 개념도를 나타낸다. 이 모델에 대해 기하학적 비선형 해석을 수행한다. 또한 검증 결과를 표 9.1.8에 나타낸다.

 비선형 계산은 기준 하중 \(P = 1.0\ \mathrm{kgf}\)에 대해, 하중 증분을 \(0.1P\)로 하여 10개 하중 스텝으로 수행한다.

 ![검증 모델](/verification_and_validation/image/example01_08.png) 그림 9.1.3 검증 모델

 표 9.1.8 exI: 검증 결과(최대 처짐량 이력)

 
| 사례명 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 선형 해 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I231 | - | - | - | - | - | - | - | - | - | - | - |
| I232 | - | - | - | - | - | - | - | - | - | - | - |
| I241 | - | - | - | - | - | - | - | - | - | - | - |
| I242 | - | - | - | - | - | - | - | - | - | - | - |
| I341 | 0.039 | 0.077 | 0.116 | 0.154 | 0.193 | 0.232 | 0.270 | 0.309 | 0.348 | 0.386 | 0.386 |
| I342 | 0.099 | 0.200 | 0.300 | 0.400 | 0.499 | 0.599 | 0.698 | 0.797 | 0.896 | 0.995 | 0.999 |
| I351 | 0.035 | 0.070 | 0.105 | 0.141 | 0.176 | 0.211 | 0.246 | 0.281 | 0.316 | 0.351 | 0.351 |
| I352 | 0.099 | 0.198 | 0.298 | 0.397 | 0.496 | 0.595 | 0.693 | 0.792 | 0.890 | 0.987 | 0.992 |
| I361 | 0.070 | 0.139 | 0.209 | 0.278 | 0.348 | 0.417 | 0.487 | 0.556 | 0.625 | 0.694 | 0.984 |
| I362 | 0.099 | 0.197 | 0.298 | 0.397 | 0.496 | 0.595 | 0.694 | 0.793 | 0.891 | 0.988 | 0.993 |

 
### (2-2) exnl2: 탄소성 변형 해석

 본 검증 문제는 NAFEMS(National Agency for Finite Element Methods and Standards (U.K.))의 Test NL1을 참고하였으며, 기하학적 비선형 및 여러 경화 법칙을 포함하여 탄소성 변형 해석을 수행하였다. 그림 9.1.4에 해석 모델을 나타낸다.

 ![탄소성 변형 해석 모델](/verification_and_validation/image/example01_09.png)

 그림 9.1.4 탄소성 변형 해석 모델

 (1) 검증 조건

 
| 항목 | 값 |
| --- | --- |
| 재료 | Mises 탄소성 재료 |
| 영률 | \(E = 250\ \mathrm{GPa}\) |
| 포아송비 | \(\nu=0.25\) |
| 초기 항복 응력 | \(5\ \mathrm{MPa}\) |
| 초기 항복 변형률 | \(0.25\times10^{-4}\) |
| 등방 경화 계수 | \(H_i = 0\) 또는 \(62.5\ \mathrm{GPa}\) |

 (2) 경계 조건

 
| 항목 | 경계 조건 | 값 |
| --- | --- | --- |
| 스텝1 | 절점 2와 3에 강제 변위 | \(u_x = 0.2500031251 * 10^{-4}\) |
| 스텝2 | 절점 2와 3에 강제 변위 | \(u_x = 0.25000937518 * 10^{-4}\) |
| 스텝3 | 절점 3과 4에 강제 변위 | \(u_y = 0.2500031251 * 10^{-4}\) |
| 스텝4 | 절점 3과 4에 강제 변위 | \(u_y = 0.25000937518 * 10^{-4}\) |
| 스텝5 | 절점 2와 3에 강제 변위 | \(u_x = -0.25000937518 * 10^{-4}\) |
| 스텝6 | 절점 2와 3에 강제 변위 | \(u_x = -0.2500031251 * 10^{-4}\) |
| 스텝7 | 절점 3과 4에 강제 변위 | \(u_y = -0.25000937518 * 10^{-4}\) |
| 스텝8 | 절점 3과 4에 강제 변위 | \(u_y = -0.2500031251 * 10^{-4}\) |

 여기에 표시하지 않은 절점은 모두 완전 구속된다. 이 문제의 이론해는 다음과 같다.

 
| 변형률 (\(\times10^{-4}\))<br>[\(\varepsilon_x\), \(\varepsilon_y\), \(\varepsilon_z\)] | 등가 응력 (MPa)<br>[\(H_i=0\ H_k=0\); \(H_i=62.5\ H_k=0\)] |
| --- | --- |
| 0.25, 0, 0 | 5.0; 5.0 |
| 0.50, 0, 0 | 5.0; 5.862 |
| 0.50, 0.25, 0 | 5.0; 5.482 |
| 0.50, 0.50, 0 | 5.0; 6.362 |
| 0.25, 0.50, 0 | 5.0; 6.640 |
| 0, 0.50, 0 | 5.0; 7.322 |
| 0, 0.25, 0 | 3.917; 4.230 |
| 0, 0, 0 | 5.0; 5.673 |

 이에 대한 계산 결과는 다음과 같다.

 
| 변형률 (\(\times10^{-4}\))<br>[\(\varepsilon_x\), \(\varepsilon_y\), \(\varepsilon_z\)] | 등가 응력 (MPa)<br>[\(H_i=0\ H_k=0\); \(H_i=62.5\ H_k=0\)] |
| --- | --- |
| \(\varepsilon_{x}\) | \(\varepsilon_{y}\) |
| 0.25, 0, 0 | 5.0 (0.0%); 5.0 (0.0%) |
| 0.50, 0, 0 | 5.0 (0.0%); 5.862 (0.0%) |
| 0.50, 0.25, 0 | 5.0 (0.0%); 5.482 (0.0%) |
| 0.50, 0.50, 0 | 5.0 (0.0%); 6.362 (-0.05%) |
| 0.25, 0.50, 0 | 5.0 (0.0%); 6.640 (-0.21%) |
| 0, 0.50, 0 | 5.0 (0.0%); 7.322 (-0.34%) |
| 0, 0.25, 0 | 3.824 (-2.4%); 4.230 (-2.70%) |
| 0, 0, 0 | 5.0 (0.0%); 5.673 (5.673 (-2.50%) |

 
## 접촉 해석(1)

 본 검증 문제는 National Agency for Finite Element Methods and Standards (U.K.)의 접촉 패치 테스트 문제 CGS-4를 참고하였으며, 마찰이 있는 유한 미끄럼 접촉 문제 기능을 테스트하는 것이다. 그림 9.1.5에 해석 모델을 나타낸다.

 ![접촉 해석 모델](/verification_and_validation/image/example01_10.png)

 그림 9.1.5 접촉 해석 모델

 이 문제의 평형 조건은 다음과 같다.

 \[ F \cos \alpha - G \sin \alpha = \pm f_{c} \] 고착 마찰 단계에서 마찰력은

 \[ f_{c} = E_t \Delta u \] 이며, 미끄럼 마찰 단계에서는

 \[ f_{c} = \mu(G \cos \alpha + F \sin \alpha) \] 따라서 위의 관계가 성립한다.

 계산 결과와 해석해의 비교는 다음과 같다.

 
| \(\mu\) | \(F/G\) 해석해 | \(F/G\) 계산 결과 |
| --- | --- | --- |
| 0.0 | 0.1 | 0.1 |
| 0.1 | 0.202 | 0.202 |
| 0.2 | 0.306 | 0.306 |
| 0.3 | 0.412 | 0.412 |

 
## 접촉 해석(2): Hertz 접촉 문제

 본 검증에서는 무한 길이 원주와 무한 평면의 Hertz 접촉 문제를 해석하였다. 원주의 반지름은 \(R = 8\ \mathrm{mm}\)이며, 변형체의 영률 \(E\)와 포아송비 \(\mu\)는 각각 \(1100\ \mathrm{MPa}\)와 \(0.0\)이다. 또한 접촉 면적은 원주의 반지름에 비해 충분히 작다고 가정하고 문제의 대칭성도 고려하여 원주의 4분의 1 모델로 해석하였다.

 ![Hertz 접촉 문제 해석 모델](/verification_and_validation/image/example01_11.png) 그림 9.1.6 Hertz 접촉 문제 해석 모델

 
### (1) 접촉 반경의 검증 결과

 접촉 반경을 계산하는 이론식은 다음과 같다.

 \[ a = \sqrt{\frac{4FR}{\pi E^{*}}} \] 여기서,

 \[ E^{*} = E/2(1 - \mu^{2}) \] 따라서 위 식을 얻는다. 본 계산에서는 압력 \(F=100\)일 때 접촉 반경 \(a=1.36\)이 된다.

 그림 9.1.7에는 접촉점의 등가 절점력을 나타낸다. 이 절점력 분포를 외삽하여 접촉 반경을 구한다.

 ![접촉점의 등가 절점력 분포](/verification_and_validation/image/example01_12.png)

 그림 9.1.7 접촉점의 등가 절점력 분포

 
### (2) 최대 전단 응력의 검증 결과

 이론해에서는 접촉 위치

 \[ z = 0.78a \] 에서 최대 전단 응력이

 \[ \tau_{\max} = 0.30\sqrt{\frac{\text{FE}^{*}}{\pi R}} \] 이 값이 이론적인 최대 전단 응력이다. 본 계산 조건에서는

 \[ \tau_{\max} = 14.2 \] 이는 본 계산 조건에서의 이론값이다. 이에 대해,

 \[ \tau_{\max} = 15.6 \] 이에 대해 계산에서는 위의 결과를 얻었다.

 ![전단 응력 분포(최대값=15.6](/verification_and_validation/image/example01_13.png) 그림 9.1.8 전단 응력 분포(최대값=15.6)

 
## (3) 모드 해석

 검증 사례 exJ~K의 검증 모델은 검증 사례 exA~G의 모델과 동일하다. 그림 9.1.9에 검증 모델의 개념도를 나타낸다. 이 모델에 대해 모드 해석을 수행한다. 1차부터 3차까지의 고유진동수를 구한다. 또한 exJ에서는 반복법 솔버를, exK에서는 직접법 솔버를 사용한다. 또한 검증 결과를 표 9.1.9~표 9.1.12에 나타낸다.

 ![검증 모델](/verification_and_validation/image/example01_14.png) 그림 9.1.9 검증 모델

 캔틸레버 보의 고유진동수는 다음 식으로 구한다.

 1차 고유진동수

 \[ n_1 = \frac{1.875^2}{2 \pi l^2} \sqrt{ \frac{gEI}{\omega} } \] 2차 고유진동수

 \[ n_2 = \frac{4.694^2}{2 \pi l^2} \sqrt{ \frac{gEI}{\omega} } \] 3차 고유진동수

 \[ n_3 = \frac{7.855^2}{2 \pi l^2} \sqrt{ \frac{gEI}{\omega} } \] 검증 모델의 특성값은 다음 표와 같다.

 
| 항목 | 값 |
| --- | --- |
| \(I\) | \(1.0/12.0\ \mathrm{mm}^4\) |
| \(E\) | \(4000.0\ \mathrm{kgf/mm^2}\) |
| \(l\) | \(10.0\ \mathrm{mm}\) |
| \(\omega\) | \(7.85 \times 10^{-6}\ \mathrm{kgf/mm^3}\) |
| \(g\) | \(9800.0\ \mathrm{mm/s^2}\) |

 따라서 3차까지의 고유진동수는 다음과 같다.

 
| 모드 번호 | 값 |
| --- | --- |
| \(n_1\) | 3.609e3 |
| \(n_2\) | 2.262e4 |
| \(n_3\) | 6.335e4 |

 표 9.1.9 exJ: 반복법의 검증 결과(1차 고유진동수)

 
| 사례명 | 요소 수 | 예측값: n1=3.609e3 |  | 비고 |
| --- | --- | --- | --- | --- |
|  |  | NASTRAN | FrontISTR |  |
| J231 | 40 | 5.861e3 | 5.861e3 | 33절점 / 평면 응력 문제 |
| J232 | 40 | 3.596e3 | 3.593e3 | 105절점 / 평면 응력 문제 |
| J241 | 20 | 3.586e3 | 4.245e3 | 33절점 / 평면 응력 문제 |
| J242 | 20 | 3.590e3 | 3.587e3 | 85절점 / 평면 응력 문제 |
| J341 | 240 | 5.442e3 | 5.429e3 | 99절점 |
| J342 | 240 | 3.621e3 | 3.595e3 | 525절점 |
| J351 | 80 | 3.695e3 | 4.298e3 | 99절점 |
| J352 | 80 | 3.610e3 | 3.609e3 | 381절점 |
| J361 | 40 | 3.679e3 | 3.619e3 | 99절점 |
| J362 | 40 | 3.611e3 | 3.606e3 | 220절점 |

 표 9.1.10 exJ: 반복법의 검증 결과(2차 고유진동수)

 
| 사례명 | 요소 수 | 예측값: n2=2.262e4 |  | 비고 |
| --- | --- | --- | --- | --- |
|  |  | NASTRAN | FrontISTR |  |
| J231 | 40 | 3.350e4 | 3.351e4 | 33절점 / 평면 응력 문제 |
| J232 | 40 | 2.163e4 | 2.156e4 | 105절점 / 평면 응력 문제 |
| J241 | 20 | 2.149e4 | 2.516e4 | 33절점 / 평면 응력 문제 |
| J242 | 20 | 2.149e4 | 2.143e4 | 85절점 / 평면 응력 문제 |
| J341 | 240 | 3.145e4 | 3.138e4 | 99절점 |
| J342 | 240 | 2.171e4 | 2.155e4 | 525절점 |
| J351 | 80 | 2.208e4 | 2.546e4 | 99절점 |
| J352 | 80 | 2.156e4 | 2.149e4 | 381절점 |
| J361 | 40 | 2.202e4 | 2.168e4 | 99절점 |
| J362 | 40 | 2.154e4 | 2.144e4 | 220절점 |

 주) 3차원 모델에서는 1차와 2차 모드가 중근이므로, 3차 계산 고유진동수를 표에서는 2차 고유진동수로 기재하였다.

 표 9.1.11 exK: 직접법의 검증 결과(1차 고유진동수)

 
| 사례명 | 요소 수 | 예측값: n1=3.609e3 |  | 비고 |
| --- | --- | --- | --- | --- |
|  |  | NASTRAN | FrontISTR |  |
| J231 | 40 | 5.861e3 | 5.861e3 | 33절점 / 평면 응력 문제 |
| J232 | 40 | 3.596e3 | 3.593e3 | 105절점 / 평면 응력 문제 |
| J241 | 20 | 3.586e3 | 4.245e3 | 33절점 / 평면 응력 문제 |
| J242 | 20 | 3.590e3 | 3.587e3 | 85절점 / 평면 응력 문제 |
| J341 | 240 | 5.442e3 | 5.429e3 | 99절점 |
| J342 | 240 | 3.621e3 | 3.595e3 | 525절점 |
| J351 | 80 | 3.695e3 | 4.298e3 | 99절점 |
| J352 | 80 | 3.610e3 | 3.609e3 | 381절점 |
| J361 | 40 | 3.679e3 | 3.619e3 | 99절점 |
| J362 | 40 | 3.611e3 | 3.606e3 | 220절점 |
| J731 | 40 | - | 3.606e3 | 220절점 |
| J741 | 20 | - | 3.594e3 | 220절점 |

 표 9.1.12 exK: 직접법의 검증 결과(2차 고유진동수)

 
| 사례명 | 요소 수 | 예측값: n2=2.262e4 |  | 비고 |
| --- | --- | --- | --- | --- |
|  |  | NASTRAN | FrontISTR |  |
| J231 | 40 | 3.350e4 | 3.351e4 | 33절점 / 평면 응력 문제 |
| J232 | 40 | 2.163e4 | 2.156e4 | 105절점 / 평면 응력 문제 |
| J241 | 20 | 2.149e4 | 2.516e4 | 33절점 / 평면 응력 문제 |
| J242 | 20 | 2.149e4 | 2.143e4 | 85절점 / 평면 응력 문제 |
| J341 | 240 | 3.145e4 | 3.138e4 | 99절점 |
| J342 | 240 | 2.171e4 | 2.155e4 | 525절점 |
| J351 | 80 | 2.208e4 | 2.546e4 | 99절점 |
| J352 | 80 | 2.156e4 | 2.149e4 | 381절점 |
| J361 | 40 | 2.202e4 | 2.168e4 | 99절점 |
| J362 | 40 | 2.154e4 | 2.144e4 | 220절점 |
| J731 | 40 | - | 2.156e4 | 220절점 |
| J741 | 20 | - | 2.153e4 | 220절점 |

 주) 3차원 모델에서는 1차와 2차 모드가 중근이므로, 3차 계산 고유진동수를 표에서는 2차 고유진동수로 기재하였다.

 
## (4) 열전도 해석

 정상 열전도 해석의 공통 조건을 그림 9.1.10에 나타낸다. 검증 사례 exM~exT의 개별 조건을 그림 9.1.11에 나타낸다. 메시 분할은 exA와 동등한 것을 사용한다.

 표 9.1.13~표 9.1.20에 각 사례의 검증 결과인 온도 분포를 나타낸다.

 ![열전도 해석](/verification_and_validation/image/example01_15.png)

 
|  |  |
| --- | --- |
| AB 사이 길이 | \(L = 10.0\ \mathrm{m}\) |
| 단면적 | \(A = 1.0\ \mathrm{m^2}\) |

 열전도율의 온도 의존성

 
| 열전도율 \(\lambda\) (W/(m·K)) | 온도 (°C) |
| --- | --- |
| 50.0 | 0.0 |
| 35.0 | 500.0 |
| 20.0 | 1000.0 |

 그림 9.1.10 정상 열전도 해석의 검증 조건

 
|  |  |
| --- | --- |
| exM:선형 재료 |  |
| exN:지정 온도 문제 | ![](/verification_and_validation/image/example01_16.png) |
| exO:집중 열유속 문제 | ![](/verification_and_validation/image/example01_17.png) |
| exP:분포 열유속 문제 | ![](/verification_and_validation/image/example01_18.png) |
| exQ:대류 열전달 문제 | ![](/verification_and_validation/image/example01_19.png) |
| exR:복사 열전달 문제 | ![](/verification_and_validation/image/example01_20.png) |
| exS:체적 발열 문제 | ![](/verification_and_validation/image/example01_21.png) |
| exT:내부 갭 문제 | ![](/verification_and_validation/image/example01_22.png) |

 그림 9.1.11 검증 사례별 해석 조건

 표 9.1.13 exM: 선형 재료에 의한 정상 계산의 검증 결과

 
| 사례명 | 요소 유형 | 요소/절점 수 | A단에서의 거리(m) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | A단 | 2.0 | 4.0 | 6.0 | 8,0 | B단 |
| M361A | 361 | 40/33 | 0.0 | 100.0 | 200.0 | 300.0 | 400.0 | 500.0 |
| M361B | 361 | 40/105 | 0.0 | 100.0 | 200.0 | 300.0 | 400.0 | 500.0 |
| M361C | 361 | 20/33 | 0.0 | 100.0 | 200.0 | 300.0 | 400.0 | 500.0 |
| M361D | 361 | 20/85 | 0.0 | 100.0 | 200.0 | 300.0 | 400.0 | 500.0 |
| M361E | 361 | 240/99 | 0.0 | 100.0 | 200.0 | 300.0 | 400.0 | 500.0 |
| M361F | 361 | 24/525 | 0.0 | 100.0 | 200.0 | 300.0 | 400.0 | 500.0 |
| M361G | 361 | 80/99 | 0.0 | 100.0 | 200.0 | 300.0 | 400.0 | 500.0 |

 표 9.1.14 exN: 지정 온도 문제의 검증 결과

 
| 사례명 | 요소 유형 | 요소/절점 수 | A단에서의 거리(m) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | A단 | 2.0 | 4.0 | 6.0 | 8,0 | B단 |
| 상용 소프트웨어 | 361 | 40/99 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |
| N231 | 231 | 40/33 | 0.0 | 87.2 | 179.5 | 278.0 | 384.1 | 500.0 |
| N232 | 232 | 40/105 | 0.0 | 86.0 | 178.3 | 276.8 | 382.9 | 500.0 |
| N241 | 241 | 20/33 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |
| N242 | 242 | 20/85 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |
| N341 | 341 | 240/99 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |
| N342 | 342 | 24/525 | 0.0 | 87.9 | 179.9 | 278.0 | 383.6 | 500.0 |
| N351 | 351 | 80/99 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |
| N352 | 352 | 80/381 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |
| N361 | 361 | 40/99 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |
| N362 | 362 | 40/330 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |
| N731 | 731 | 40/33 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |
| N741 | 741 | 20/33 | 0.0 | 87.3 | 179.7 | 278.2 | 384.3 | 500.0 |

 표 9.1.15 exO: 집중 열유속 문제의 검증 결과

 
| 사례명 | 요소 유형 | 요소/절점 수 | A단에서의 거리(m) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | A단 | 2.0 | 4.0 | 6.0 | 8,0 | B단 |
| 상용 소프트웨어 | 361 | 40/99 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| O231 | 231 | 40/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| O232 | 232 | 40/105 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| O241 | 241 | 20/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| O242 | 242 | 20/85 | 0.0 | 103.2 | 213.7 | 333.4 | 465.2 | 618.0 |
| O341 | 341 | 240/99 | - | - | - | - | - | - |
| O342 | 342 | 24/525 | 0.0 | 104.4 | 214.9 | 334.7 | 466.3 | 614.6 |
| O351 | 351 | 80/99 | - | - | - | - | - | - |
| O352 | 352 | 80/381 | 0.0 | 103.2 | 213.7 | 333.3 | 465.0 | 624.2 |
| O361 | 361 | 40/99 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| O362 | 362 | 40/330 | 0.0 | 103.2 | 213.7 | 333.4 | 465.5 | 623.5 |
| O731 | 731 | 40/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.5 |
| O741 | 741 | 20/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |

 표 9.1.16 exP: 분포 열유속 문제의 검증 결과

 
| 사례명 | 요소 유형 | 요소/절점 수 | A단에서의 거리(m) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | A단 | 2.0 | 4.0 | 6.0 | 8,0 | B단 |
| 상용 소프트웨어 | 361 | 40/99 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| P231 | 231 | 40/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| P232 | 232 | 40/105 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| P241 | 241 | 20/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| P242 | 242 | 20/85 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| P341 | 341 | 240/99 | - | - | - | - | - | - |
| P342 | 342 | 24/525 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| P351 | 351 | 80/99 | - | - | - | - | - | - |
| P352 | 352 | 80/381 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| P361 | 361 | 40/99 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| P362 | 362 | 40/330 | 0.0 | 103.2 | 213.7 | 333.4 | 465.5 | 612.6 |
| P731 | 731 | 40/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.5 |
| P741 | 741 | 20/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |

 표 9.1.17 exQ: 대류 열전달 문제의 검증 결과

 
| 사례명 | 요소 유형 | 요소/절점 수 | A단에서의 거리(m) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | A단 | 2.0 | 4.0 | 6.0 | 8,0 | B단 |
| 상용 소프트웨어 | 361 | 40/99 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q231 | 231 | 40/33 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q232 | 232 | 40/105 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q241 | 241 | 20/33 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q242 | 242 | 20/85 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q341 | 341 | 240/99 | - | - | - | - | - | - |
| Q342 | 342 | 24/525 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q351 | 351 | 80/99 | - | - | - | - | - | - |
| Q352 | 352 | 80/381 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q361 | 361 | 40/99 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q362 | 362 | 40/330 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q731 | 731 | 40/33 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |
| Q741 | 741 | 20/33 | 0.0 | 89.2 | 183.8 | 284.8 | 393.9 | 513.2 |

 표 9.1.18 exR: 복사 열전달 문제의 검증 결과

 
| 사례명 | 요소 유형 | 요소/절점 수 | A단에서의 거리(m) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | A단 | 2.0 | 4.0 | 6.0 | 8,0 | B단 |
| 상용 소프트웨어 | 361 | 40/99 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R231 | 231 | 40/33 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R232 | 232 | 40/105 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R241 | 241 | 20/33 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R242 | 242 | 20/85 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R341 | 341 | 240/99 | - | - | - | - | - | - |
| R342 | 342 | 24/525 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R351 | 351 | 80/99 | - | - | - | - | - | - |
| R352 | 352 | 80/381 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R361 | 361 | 40/99 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R362 | 362 | 40/330 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R731 | 731 | 40/33 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |
| R741 | 741 | 20/33 | 0.0 | 89.5 | 184.4 | 285.8 | 395.3 | 515.2 |

 표 9.1.19 exS: 체적 발열 문제의 검증 결과

 
| 사례명 | 요소 유형 | 요소/절점 수 | A단에서의 거리(m) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | A단 | 2.0 | 4.0 | 6.0 | 8,0 | B단 |
| 상용 소프트웨어 | 361 | 40/99 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S231 | 231 | 40/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S232 | 232 | 40/105 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S241 | 241 | 20/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S242 | 242 | 20/85 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S341 | 341 | 240/99 | - | - | - | - | - | - |
| S342 | 342 | 24/525 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S351 | 351 | 80/99 | - | - | - | - | - | - |
| S352 | 352 | 80/381 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S361 | 361 | 40/99 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S362 | 362 | 40/330 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S731 | 731 | 40/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |
| S741 | 741 | 20/33 | 0.0 | 103.2 | 213.7 | 333.3 | 464.8 | 612.6 |

 표 9.1.20 exT: 내부 갭 문제의 검증 결과

 
| 사례명 | 요소 유형 | 요소/절점 수 | A단에서의 거리(m) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | A단 | 2.0 | 4.0 | 6.0 | 8,0 | B단 |
| 상용 소프트웨어 | 361 | 40/99 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S231 | 231 | 40/33 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S232 | 232 | 40/105 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S241 | 241 | 20/33 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S242 | 242 | 20/85 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S341 | 341 | 240/99 | - | - | - | - | - | - |
| S342 | 342 | 24/525 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S351 | 351 | 80/99 | - | - | - | - | - | - |
| S352 | 352 | 80/381 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S361 | 361 | 40/99 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S362 | 362 | 40/330 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S731 | 731 | 40/33 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |
| S741 | 741 | 20/33 | 0.0 | 88.6 | 182.4 | 282.6 | 387.7 | 500.0 |

 
## 선형 동적 해석

 exW에서는 (1)항과 같은 캔틸레버 보를 대상으로 선형 동적 해석을 수행하였다. 그림 9.1.12에 검증 조건을 나타낸다. 여기서는 동일한 메시 분할에 대해 시간 증분이 결과에 미치는 영향을 검증하였다. 동적 해석 방법으로 암시적 해법과 명시적 해법을 모두 사용하고 요소 유형은 361 및 342를 사용하였다. 검증 결과를 표 9.1.22 및 그림 9.1.13~그림 9.1.15에 나타낸다.

 ![해석 모델](/verification_and_validation/image/example01_23.png) 해석 모델

 ![외력 F의 시간 이력](/verification_and_validation/image/example01_24.png) 외력 F의 시간 이력

 가진점 변위의 이론해:

 \[ F(t)=F_0 I(t) \] 여기서,

 \[ \(F_0:\ \text{상수 벡터}\) \] \[ I(t)= \begin{cases} 0, t < 0 \\\ 1, 0 \leq t \end{cases} \] \[ u(t) = \frac{F_0 l^3}{EI} \sum^{\infty}_{i=1} \frac{1-\cos{\omega_i t}}{{\lambda_i}^4} \left\lbrace \cosh{\lambda_i}-\cos{\lambda_i}-\frac{\cosh{\lambda_i} + \cos{\lambda_i}}{\sin{\lambda_i}+\sin{\lambda_i}} (\sinh{\lambda_i} - \sin{\lambda_i}) \right\rbrace^2 \] 그림 9.1.12 선형 동적 해석의 검증 조건

 검증 조건:

 
|  |  |  |
| --- | --- | --- |
| 길이 | \(L\) | \(10.0\ \mathrm{mm}\) |
| 단면 폭 | \(a\) | \(1.0\ \mathrm{mm}\) |
| 단면 높이 | \(b\) | \(1.0\ \mathrm{mm}\) |
| 영률 | \(E\) | \(4000.0\ \mathrm{kgf/mm^2}\) |
| 포아송비 | \(\nu\) | \(0.3\) |
| 밀도 | \(\rho\) | \(1.0 \times 10^{-9}\ \mathrm{kgf}\;\mathrm{s}^2/\mathrm{mm}^4\) |
| 중력 가속도 | \(g\) | \(9800.0\ \mathrm{mm/s^2}\) |
| 외력 | \(F_0\) | \(1.0\ \mathrm{kgf}\) |

 
|  |  |
| --- | --- |
| 요소 | 6면체 1차 요소 |
| 4면체 2차 요소 |  |
| 해법 | 암시적 해법 |
| Newmark-\(\beta\)법의 매개변수 \(\gamma\) | 1/2 |
| Newmark-\(\beta\)법의 매개변수 \(\beta\) | 1/4 |
| 명시적 해법 |  |
| 감쇠 | 없음 |

 표 9.1.21 선형 동적 해석의 검증 조건 (계속)

 
| 사례명 | 요소 유형 | 절점 수 | 요소 수 | 해법 | 시간 증분 \(\Delta t\) (s) |
| --- | --- | --- | --- | --- | --- |
| W361_c0_im_m2_t1 | 361 | 99 | 40 | 암시적 해법 | 1.0E-06 |
| W361_c0_im_m2_t2 | 361 | 99 | 40 | 암시적 해법 | 1.0E-05 |
| W361_c0_im_m2_t3 | 361 | 99 | 40 | 암시적 해법 | 1.0E-04 |
| W361_c0_ex_m2_t1 | 361 | 99 | 40 | 암시적 해법 | 1.0E-08 |
| W361_c0_ex_m2_t2 | 361 | 99 | 40 | 암시적 해법 | 1.0E-07 |
| W361_c0_ex_m2_t3 | 361 | 99 | 40 | 암시적 해법 | 1.0E-06 |
| W342_c0_im_m2_t1 | 342 | 525 | 240 | 명시적 해법 | 1.0E-06 |
| W342_c0_im_m2_t2 | 342 | 525 | 240 | 명시적 해법 | 1.0E-05 |
| W342_c0_im_m2_t3 | 342 | 525 | 240 | 명시적 해법 | 1.0E-04 |
| W342_c0_ex_m2_t1 | 342 | 525 | 240 | 명시적 해법 | 1.0E-08 |
| W342_c0_ex_m2_t2 | 342 | 525 | 240 | 명시적 해법 | 5.0E-08 |
| W342_c0_ex_m2_t3 | 342 | 525 | 240 | 명시적 해법 | 1.0E-07 |

 표 9.1.22 exW: 캔틸레버 보를 대상으로 한 선형 동적 해석의 검증 결과

 
| 사례명 | 요소 유형 | 절점 수 | 요소 수 | 해법 | 시각 \(t = 0.002\ \mathrm{s}\)에서의 z 방향 변위 (mm) |  |
| --- | --- | --- | --- | --- | --- | --- |
| W361_c0_im_m2_t1 | 361 | 99 | 40 | 암시적 해법 | 1.9753 | 1.9302 |
| W361_c0_im_m2_t2 | 361 | 99 | 40 | 암시적 해법 | 1.9753 | 1.8686 |
| W361_c0_im_m2_t3 | 361 | 99 | 40 | 암시적 해법 | 1.9753 | 0.3794 |
| W361_c0_ex_m2_t1 | 361 | 99 | 40 | 암시적 해법 | 1.9753 | 1.9302 |
| W361_c0_ex_m2_t2 | 361 | 99 | 40 | 암시적 해법 | 1.9753 | 1.9247 |
| W361_c0_ex_m2_t3 | 361 | 99 | 40 | 암시적 해법 | 1.9753 | 발산 |
| W342_c0_im_m2_t1 | 342 | 525 | 240 | 명시적 해법 | 1.9753 | 1.9431 |
| W342_c0_im_m2_t2 | 342 | 525 | 240 | 명시적 해법 | 1.9753 | 1.8719 |
| W342_c0_im_m2_t3 | 342 | 525 | 240 | 명시적 해법 | 1.9753 | 0.3873 |
| W342_c0_ex_m2_t1 | 342 | 525 | 240 | 명시적 해법 | 1.9753 | 1.9359 |
| W342_c0_ex_m2_t2 | 342 | 525 | 240 | 명시적 해법 | 1.9753 | 1.9358 |
| W342_c0_ex_m2_t3 | 342 | 525 | 240 | 명시적 해법 | 1.9753 | 발산 |

 ![캔틸레버 보의 변형도 및 등가 응력 분포](/verification_and_validation/image/example01_25.png) 그림 9.1.13 캔틸레버 보의 변형도 및 등가 응력 분포

 ![(a) 요소 유형 361:암시적 해법](/verification_and_validation/image/example01_26.png) (a) 요소 유형 361:암시적 해법

 ![(b) 요소 유형 361:명시적 해법](/verification_and_validation/image/example01_27.png) (b) 요소 유형 361:명시적 해법

 그림 9.1.14 가진점 변위 \(u_z\)의 시간 이력

 ![(a) 요소 유형 342:암시적 해법](/verification_and_validation/image/example01_28.png) (a) 요소 유형 342:암시적 해법

 ![(b) 요소 유형 342:명시적 해법](/verification_and_validation/image/example01_29.png) (b) 요소 유형 342:명시적 해법

 그림 9.1.15 가진점 변위 \(u_z\)의 시간 이력

 
## 주파수 응답 해석

 본 검증에서는 캔틸레버 보를 대상으로 주파수 응답 해석을 수행하고, 범용 상용 소프트웨어의 결과와 비교하여 검증하였다. 해석 모델과 검증 조건을 나타낸다.

 ![](/verification_and_validation/image/example01_30.png) ![](/verification_and_validation/image/example01_31.png)

 해석 조건:

 
|  |  |  |
| --- | --- | --- |
| 영률 | \(E\) | \(210000\ \mathrm{N/mm^2}\) |
| 포아송비 | \(\nu\) | \(0.3\) |
| 밀도 | \(\rho\) | \(7.89 \times 10^{-9}\ \mathrm{t/mm^3}\) |
| 중력 가속도 | \(g\) | \(9800.0\ \mathrm{mm/s^2}\) |
| 하중 | \(F_0\) | \(1.0\ \mathrm{N}\) |
| Rayleigh 감쇠 매개변수 | \(R_m\) | \(0.0\) |
| Rayleigh 감쇠 매개변수 | \(R_k\) | \(7.2E-07\) |

 그림 9.1.16 해석 모델(1차 사면체 요소: 요소 126개, 절점 55개)

 모드 해석에서 얻은 5차까지의 고유진동수와 가진점의 주파수 응답을 나타낸다.

 
| 모드 | FrontISTR | 상용 소프트웨어 |
| --- | --- | --- |
| 1 | 14952 | 14952 |
| 2 | 15002 | 15003 |
| 3 | 84604 | 84539 |
| 4 | 84771 | 84697 |
| 5 | 127054 | 126852 |

 ![가진점의 변위 진폭의 주파수 의존성](/verification_and_validation/image/example01_32.png) 그림 9.1.17 가진점의 변위 진폭의 주파수 의존성


---


<!-- ========================================== -->
<!-- FILE: verification_and_validation/02_static.md | TITLE: 탄성 정적 해석용 실제 모델 예제 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/verification_and_validation/02_static.html -->

# 탄성 정적 해석용 실제 모델 예제

 
## 해석 모델

 탄성 정적 해석을 대상으로 한 실제 모델 검증 예제의 목록을 표 9.2.1에 나타낸다. 또한 모델의 형상(일부 제외)을 그림 9.2.1~그림 9.2.5에 나타낸다. 요소 유형 731 및 741의 예제를 실행하려면 별도의 직접법 솔버가 필요하다.

 표 9.2.1 탄성 정적 해석용 실제 모델 검증 예제

 
| 케이스명 | 요소 유형 | 검증 모델 | 절점 수 | 자유도 수 |
| --- | --- | --- | --- | --- |
| EX01A | 342 | 커넥팅 로드(10만 절점) | 94,074 | 282,222 |
| EX01B | 342 | 커넥팅 로드(33만 절점) | 331,142 | 993,426 |
| EX02 | 361 | 구멍이 있는 블록 | 37,386 | 112,158 |
| EX03 | 342 | 터빈 블레이드 | 10,095 | 30,285 |
| EX04 | 741 | 원통 셸 | 10,100 | 60,600 |
| EX05A | 731 | 와인잔(성긴 메시) | 7,240 | 43,440 |
| EX05B | 731 | 와인잔(중간 메시) | 48,803 | 292,818 |
| EX05C | 731 | 와인잔(미세 메시) | 100,602 | 603,612 |

 ![커넥팅 로드(EX01A)](/verification_and_validation/image/example02_01.png)

 그림 9.2.1 커넥팅 로드(EX01A)

 ![구멍이 있는 블록(EX02)](/verification_and_validation/image/example02_02.png)

 그림 9.2.2 구멍이 있는 블록(EX02)

 ![터빈 블레이드(EX03, EX06)](/verification_and_validation/image/example02_03.png)

 그림 9.2.3 터빈 블레이드(EX03, EX06)

 ![원통 셸(EX04, EX09)](/verification_and_validation/image/example02_04.png)

 그림 9.2.4 원통 셸(EX04, EX09)

 ![와인잔(EX05, EX10A)](/verification_and_validation/image/example02_05.png)

 그림 9.2.5 와인잔(EX05, EX10A)

 
## 해석 결과

 
### 해석 결과 예

 해석 결과의 예를 그림 9.2.6~그림 9.2.9에 나타낸다.

 ![EX01A 해석 결과(Mises 응력과 변형 형상도(10배))](/verification_and_validation/image/example02_06.png)

 그림 9.2.6 EX01A 해석 결과(Mises 응력과 변형 형상도(10배))

 ![EX02 해석 결과(Mises 응력과 변형 형상도(100배))](/verification_and_validation/image/example02_07.png)

 그림 9.2.7 EX02 해석 결과(Mises 응력과 변형 형상도(100배))

 ![EX03 해석 결과(변형 형상도(10배))](/verification_and_validation/image/example02_08.png)

 그림 9.2.8 EX03 해석 결과(변형 형상도(10배))

 ![EX04 해석 결과(변형 형상도(100배))](/verification_and_validation/image/example02_09.png)

 그림 9.2.9 EX04 해석 결과(변형 형상도(100배))

 
### 검증 예제 EX02에 의한 해석 성능 검증 결과

 검증 예제 EX02 구멍이 있는 블록 모델과 동등한 모델을 사용하여 범용 상용 소프트웨어로 해석을 수행했다. FrontISTR와 응력 성분의 최대값 및 최소값을 비교한 결과를 그림 9.2.10에 나타낸다. 이 그림에서 응력 성분이 매우 잘 일치함을 알 수 있다.

 다음으로 응력 분포에 대한 영역 분할의 영향을 조사한 결과를 나타낸다. 메시 분할은 RCB법으로 수행하여 X, Y, Z 각 축 방향으로 2분할하고 전체를 8개 영역으로 분할했다. 그림 9.2.11에 메시 분할 상태를 나타낸다. 그림 9.2.12에는 단일 영역과 8개 영역으로 메시를 분할한 해석 결과의 응력 분포를 나타낸다.

 ![EX02에서 범용 소프트웨어와의 응력 성분 비교](/verification_and_validation/image/example02_10.png)

 그림 9.2.10 EX02에서 범용 소프트웨어와의 응력 성분 비교

 ![RCB법으로 EX02 메시를 8개 영역으로 분할한 결과](/verification_and_validation/image/example02_11.png)

 그림 9.2.11 RCB법으로 EX02 메시를 8개 영역으로 분할한 결과

 ![영역 분할에 따른 Mises 응력 분포의 차이](/verification_and_validation/image/example02_12.png)

 그림 9.2.12 영역 분할에 따른 Mises 응력 분포의 차이

 그림 9.2.12에서 양자의 차이는 보이지 않으며 완전히 일치함을 알 수 있다.

 다음으로 사용한 HEC-MW 솔버 설정에 따른 실행 시간을 비교한 결과를 표 9.2.2에 나타낸다. 또한 그림 9.2.13에 해를 구할 때까지의 수렴 이력을 나타낸다.

 표 9.2.2 HEC-MW 솔버에 의한 실행 시간 비교

 
| 솔버 | 실행 시간 (s) |
| --- | --- |
| CGI | 38.79 |
| CGscale | 52.75 |
| BCGS | 60.79 |
| CG8 | 6.65 |

 ![HEC-MW 솔버에 의한 수렴 이력 비교(수렴 판정 임계값: 1.0×10⁻⁸)](/verification_and_validation/image/example02_14.png)

 그림 9.2.13 HEC-MW 솔버에 의한 수렴 이력 비교(수렴 판정 임계값: \(1.0 \times 10^{-8}\))

 
### 검증 예제 EX01A에 의한 계산 시간 비교

 검증 예제 EX01A(커넥팅 로드)를 사용하여 영역 분할에 따른 계산 속도의 가속률을 검증했다. 계산에는 Xeon 2.8 GHz 프로세서를 탑재한 24노드 클러스터 계산기를 사용했다. 결과를 그림 9.2.14에 나타낸다. 이 그림에서 계산 속도는 영역 수에 비례하여 빨라짐을 알 수 있다.

 또한 계산 시간의 계산기 환경에 따른 차이도 조사했다. 결과를 표 9.2.3에 나타낸다.

 ![영역 분할에 의한 가속 효과](/verification_and_validation/image/example02_15.png)

 그림 9.2.14 영역 분할에 의한 가속 효과

 표 9.2.3 계산 시간의 계산기별 비교(1CPU)

 
| CPU | 주파수 (GHz) | 운영체제 | CPU 시간 (s) | 솔버 시간 (s) |
| --- | --- | --- | --- | --- |
| Xeon | 2.8 | Linux | 850 | 817 |
| Pentium III | 0.866 | Win2000 | 2008 | 1980 |
| Pentium M | 0.760 | WinXP | 1096 | 1070 |
| Pentium 4 | 2.0 | WinXP | 802 | 785 |
| Pentium 4 | 2.8 | WinXP | 738 | 718 |
| Celeron | 0.700 | Win2000 | 2252 | 2215 |
| Pentium 4 | 2.4 | WinXP | 830 | 804 |


---


<!-- ========================================== -->
<!-- FILE: verification_and_validation/03_dynamic.md | TITLE: 선형 동적 해석용 실제 모델 예제 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/verification_and_validation/03_dynamic.html -->

# 선형 동적 해석용 실제 모델 예제

 
## 해석 모델

 선형 동적 해석에서는 실제 모델로 그림9.3.1에 나타낸 기계 부품을 대상으로 한다. 이 실제 모델에 대해 하중 조건과 감쇠계수가 서로 다른 4개 사례의 검증 예제를 설정한다. 그 목록을 표9.3.1에 나타낸다.

 표 9.3.1 선형 동적 해석용 실제 모델 검증 예제

 
| 사례명 | 요소 유형 | 검증 모델 | 하중 조건 | 감쇠 조건 | 절점 수 | 자유도 수 |
| --- | --- | --- | --- | --- | --- | --- |
| EX31A | 342 | 맞물림 모델 | 스텝 하중 \(F_0\) | 없음 | 15,214 | 45,642 |
| EX31B | 342 |  | 스텝 하중 \(F_0\) | 있음 | 15,214 | 45,642 |
| EX31C | 342 |  | 사각파 펄스 \(F_1\) | 없음 | 15,214 | 45,642 |
| EX31D | 342 |  | 사각파 펄스 \(F_1\) | 있음 | 15,214 | 45,642 |

 ![맞물림 모델](/verification_and_validation/image/example05_01.png)

 그림 9.3.1 맞물림 모델

 
## 해석 결과

 해석 결과의 예를 그림9.3.2~그림9.3.3에 나타낸다.

 ![가진점 변위](/verification_and_validation/image/example05_02.png)

 그림 9.3.2 가진점 변위 \(u_z\)의 시간 이력

 ![변형도 및 등가 응력 분포(변형 배율 5.0): EX31C](/verification_and_validation/image/example05_03.png)

 그림 9.3.3 변형도 및 등가 응력 분포(변형 배율 5.0): EX31C


---


<!-- ========================================== -->
<!-- FILE: verification_and_validation/04_eigen.md | TITLE: 모드 해석용 실제 모델 예제 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/verification_and_validation/04_eigen.html -->

# 모드 해석용 실제 모델 예제

 
## 해석 모델

 모드 해석용 실제 모델 검증 예제를 표 9.4.1에 나타냅니다. 이 중 EX07(터빈 로터)과 EX08(스프링)의 모델 형상을 그림 9.4.1과 그림 9.4.2에 나타냅니다. 그 밖의 모델 형상은 앞서 제시한 해당 탄성 정적 해석 검증 예제와 동일합니다. 요소 종류 731 및 741을 사용하는 예제를 실행하려면 별도의 직접 솔버가 필요합니다.

 표 9.4.1 모드 해석용 실제 모델 검증 예제

 
| 케이스명 | 요소 유형 | 검증 모델 | 절점 수 | 자유도 수 |
| --- | --- | --- | --- | --- |
| EX06 | 342 | 터빈 블레이드 | 10,095 | 30,285 |
| EX07 | 361 | 터빈 로터 | 127,440 | 382,320 |
| EX08 | 342 | 스프링 | 78,771 | 236,313 |
| EX09 | 741 | 원통 셸 | 10,100 | 60,600 |
| EX10A | 731 | 와인 글라스(거친 격자) | 7,240 | 43,440 |
| EX10B | 731 | 와인 글라스(중간 격자) | 48,803 | 292,818 |

 ![터빈 로터 (EX07)](/verification_and_validation/image/example03_01.png)

 그림 9.4.1 터빈 로터(EX07)

 ![스프링 (EX08)](/verification_and_validation/image/example03_02.png)

 그림 9.4.2 스프링(EX08)

 
## 해석 결과

 모드 형상과 고유진동수를 아래에 나타냅니다.

 
### EX06 터빈 블레이드

 
|  |  |
| --- | --- |
| ![(a) 1차 모드 (1170 kHz)](/verification_and_validation/image/example03_03.png) | ![(b) 2차 모드 (3250 kHz)](/verification_and_validation/image/example03_04.png) |
| (a) 1차 모드 (1170 kHz) | (b) 2차 모드 (3250 kHz) |
| ![(c) 3차 모드 (4130 kHz)](/verification_and_validation/image/example03_05.png) | ![(d) 4차 모드 (4140 kHz)](/verification_and_validation/image/example03_06.png) |
| (c) 3차 모드 (4130 kHz) | (d) 4차 모드 (4140 kHz) |
| ![(e) 5차 모드 (8210 kHz)](/verification_and_validation/image/example03_07.png) |  |
| (e) 5차 모드 (8210 kHz) |  |

 그림 9.4.3 EX06 터빈 블레이드의 모드 형상

 
### EX07 터빈 로터

 
|  |  |
| --- | --- |
| ![(a) 1차 모드 (1067 Hz)](/verification_and_validation/image/example03_08.png) | ![(b) 2차 모드 (1072 Hz)](/verification_and_validation/image/example03_09.png) |
| (a) 1차 모드 (1067 Hz) | (b) 2차 모드 (1072 Hz) |
|  |  |
| ![(c) 3차 모드 (1072 Hz)](/verification_and_validation/image/example03_10.png) | ![(d) 4차 모드 (1093 Hz)](/verification_and_validation/image/example03_11.png) |
| (c) 3차 모드 (1072 Hz) | (d) 4차 모드 (1093 Hz) |
|  |  |
| ![(e) 5차 모드 (1093 Hz)](/verification_and_validation/image/example03_12.png) |  |
| (e) 5차 모드 (1093 Hz) |  |
|  |  |

 그림 9.4.4 EX07 터빈 로터의 모드 형상

 
### EX08 스프링

 
|  |  |  |
| --- | --- | --- |
| 1차 모드 (445 Hz) | 2차 모드 (446 Hz) | 3차 모드 (908 Hz) |
| ![1차 모드 (445 Hz)](/verification_and_validation/image/example03_13.png) | ![2차 모드 (446 Hz)](/verification_and_validation/image/example03_14.png) | ![3차 모드 (908 Hz)](/verification_and_validation/image/example03_15.png) |
|  |  |  |
| 4차 모드 (985 Hz) | 5차 모드 (1819 Hz) |  |
| ![4차 모드 (985 Hz)](/verification_and_validation/image/example03_16.png) | ![5차 모드 (1819 Hz)](/verification_and_validation/image/example03_17.png) |  |

 
|  | FrontISTR | NASTRAN |
| --- | --- | --- |
| 1 | 445 Hz | 445 Hz |
| 2 | 446 Hz | 446 Hz |
| 3 | 908 Hz | 908 Hz |
| 4 | 985 Hz | 986 Hz |
| 5 | 1810 Hz | 1810 Hz |

 그림 9.4.5 EX08 스프링의 모드 형상

 
### EX09 원통 셸

 ![(a) 1차 및 2차 모드 (109 Hz)](/verification_and_validation/image/example03_18.png)

 (a) 1차 및 2차 모드 (109 Hz)

 ![(b) 3차 및 4차 모드 (570 Hz)](/verification_and_validation/image/example03_19.png)

 (b) 3차 및 4차 모드 (570 Hz)

 ![(c) 5차 모드 (615 Hz)](/verification_and_validation/image/example03_20.png)

 (c) 5차 모드 (615 Hz)

 그림 9.4.6 EX09 원통 셸의 모드 형상

 
### EX10A 와인 글라스

 
|  |  |  |
| --- | --- | --- |
| ![(a) 1차 모드 (101 Hz)](/verification_and_validation/image/example03_21.png) | ![(b) 2차 모드 (102 Hz)](/verification_and_validation/image/example03_22.png) | ![(c) 3차 모드 (354 Hz)](/verification_and_validation/image/example03_23.png) |
| (a) 1차 모드 (101 Hz) | (b) 2차 모드 (102 Hz) | (c) 3차 모드 (354 Hz) |
|  |  |  |
| ![(d) 4차 모드 (1335 Hz)](/verification_and_validation/image/example03_24.png) | ![(e) 5차 모드 (1336 Hz)](/verification_and_validation/image/example03_25.png) |  |
| (d) 4차 모드 (1335 Hz) | (e) 5차 모드 (1336 Hz) |  |

 그림 9.4.7 EX10A 와인 글라스의 모드 형상


---


<!-- ========================================== -->
<!-- FILE: verification_and_validation/05_heat.md | TITLE: 열전도 해석의 실모델 예 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/verification_and_validation/05_heat.html -->

# 열전도 해석의 실모델 예

 
## 해석 모델

 열전도 해석의 실모델로 사용후핵연료 수송 용기를 사용합니다. 이 실모델에 대해 메시 분해능이 서로 다른 세 가지 검증 예를 정의합니다. 목록은 표 9.5.1에 나타냅니다. 모델 형상은 그림 9.5.1부터 그림 9.5.4에도 나타냅니다.

 표 9.5.1 열전도 해석 실모델 검증 예

 
| 케이스명 | 요소 유형 | 검증 모델 | 절점 수 | 자유도 수 |
| --- | --- | --- | --- | --- |
| EX21A | 361 | 사용후핵연료 수송 용기 | 88,938 | 79,920 |
| EX21B | 361 |  | 309,941 | 289,800 |
| EX21C | 361 |  | 1,205,765 | 1,159,200 |

 ![사용후핵연료 수송 용기](/verification_and_validation/image/example04_01.png)

 그림 9.5.1 사용후핵연료 수송 용기

 ![사용후핵연료 수송 용기의 치수](/verification_and_validation/image/example04_02.png)

 그림 9.5.2 사용후핵연료 수송 용기의 치수

 ![모델 개략도](/verification_and_validation/image/example04_03.png)

 그림 9.5.3 모델 개략도

 ![모델 경계 조건 및 메시 분할 (EX21A)](/verification_and_validation/image/example04_04.png)

 그림 9.5.4 모델 경계 조건 및 메시 분할 (EX21A)

 
## 해석 결과

 해석 결과의 예를 그림 9.5.5부터 그림 9.5.7에 나타냅니다.

 ![온도 분포](/verification_and_validation/image/example04_05.png)

 그림 9.5.5 온도 분포

 ![분산 모델](/verification_and_validation/image/example04_06.png)

 그림 9.5.6 분산 모델

 ![분산 처리에 의한 속도 향상](/verification_and_validation/image/example04_07.png)

 그림 9.5.7 분산 처리에 의한 속도 향상


---

