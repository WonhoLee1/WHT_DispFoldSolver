# FrontISTR 키워드 레퍼런스 — 해석제어(.cnt) 전체 키워드 사전

> 크롤링 출처: `https://docs.frontistr.com/ko/` (수록 페이지 수: 81개)

## 수록 문서 목차

- `keyword_reference/cnt/acceleration.md` — !ACCELERATION
- `keyword_reference/cnt/autoinc_param.md` — !AUTOINC_PARAM
- `keyword_reference/cnt/boundary.md` — !BOUNDARY
- `keyword_reference/cnt/cflux.md` — !CFLUX
- `keyword_reference/cnt/cload.md` — !CLOAD
- `keyword_reference/cnt/contact.md` — !CONTACT
- `keyword_reference/cnt/contact_algo.md` — !CONTACT_ALGO
- `keyword_reference/cnt/contact_param.md` — !CONTACT_PARAM
- `keyword_reference/cnt/couple.md` — !COUPLE
- `keyword_reference/cnt/creep.md` — !CREEP
- `keyword_reference/cnt/damping.md` — !DAMPING
- `keyword_reference/cnt/dashpot_a.md` — !DASHPOT_A
- `keyword_reference/cnt/dashpot_d.md` — !DASHPOT_D
- `keyword_reference/cnt/density.md` — !DENSITY
- `keyword_reference/cnt/dflux.md` — !DFLUX
- `keyword_reference/cnt/dload.md` — !DLOAD
- `keyword_reference/cnt/dynamic.md` — !DYNAMIC
- `keyword_reference/cnt/echo.md` — !ECHO
- `keyword_reference/cnt/eigen.md` — !EIGEN
- `keyword_reference/cnt/eigenread.md` — !EIGENREAD
- `keyword_reference/cnt/elastic.md` — !ELASTIC
- `keyword_reference/cnt/element_activation.md` — !ELEMENT_ACTIVATION
- `keyword_reference/cnt/embed.md` — !EMBED
- `keyword_reference/cnt/end.md` — !END
- `keyword_reference/cnt/expansion_coeff.md` — !EXPANSION_COEFF
- `keyword_reference/cnt/film.md` — !FILM
- `keyword_reference/cnt/fixtemp.md` — !FIXTEMP
- `keyword_reference/cnt/fload.md` — !FLOAD
- `keyword_reference/cnt/fluid.md` — !FLUID
- `keyword_reference/cnt/heat.md` — !HEAT
- `keyword_reference/cnt/hyperelastic.md` — !HYPERELASTIC
- `keyword_reference/cnt/index.md` — 해석 제어 데이터 (cnt)
- `keyword_reference/cnt/initial_condition.md` — !INITIAL_CONDITION
- `keyword_reference/cnt/material.md` — !MATERIAL
- `keyword_reference/cnt/method.md` — !METHOD
- `keyword_reference/cnt/orientation.md` — !ORIENTATION
- `keyword_reference/cnt/output_res.md` — !OUTPUT_RES
- `keyword_reference/cnt/output_sstype.md` — !OUTPUT_SSTYPE
- `keyword_reference/cnt/output_vis.md` — !OUTPUT_VIS
- `keyword_reference/cnt/plastic.md` — !PLASTIC
- `keyword_reference/cnt/radiate.md` — !RADIATE
- `keyword_reference/cnt/reftemp.md` — !REFTEMP
- `keyword_reference/cnt/restart.md` — !RESTART
- `keyword_reference/cnt/section.md` — !SECTION
- `keyword_reference/cnt/sfilm.md` — !SFILM
- `keyword_reference/cnt/sflux.md` — !SFLUX
- `keyword_reference/cnt/solution.md` — !SOLUTION
- `keyword_reference/cnt/solver.md` — !SOLVER
- `keyword_reference/cnt/spring.md` — !SPRING
- `keyword_reference/cnt/spring_a.md` — !SPRING_A
- `keyword_reference/cnt/spring_d.md` — !SPRING_D
- `keyword_reference/cnt/sradiate.md` — !SRADIATE
- `keyword_reference/cnt/static.md` — !STATIC
- `keyword_reference/cnt/step.md` — !STEP
- `keyword_reference/cnt/temperature.md` — !TEMPERATURE
- `keyword_reference/cnt/time_points.md` — !TIME_POINTS
- `keyword_reference/cnt/trs.md` — !TRS
- `keyword_reference/cnt/uload.md` — !ULOAD
- `keyword_reference/cnt/user_material.md` — !USER_MATERIAL
- `keyword_reference/cnt/velocity.md` — !VELOCITY
- `keyword_reference/cnt/version.md` — !VERSION
- `keyword_reference/cnt/viscoelastic.md` — !VISCOELASTIC
- `keyword_reference/cnt/visualization/ambient_coef.md` — !ambient_coef
- `keyword_reference/cnt/visualization/color_comp_name.md` — !color_comp_name
- `keyword_reference/cnt/visualization/color_mapping_bar_on.md` — !color_mapping_bar_on
- `keyword_reference/cnt/visualization/data_comp_name.md` — !data_comp_name
- `keyword_reference/cnt/visualization/deform_scale.md` — !deform_scale
- `keyword_reference/cnt/visualization/display_method.md` — !display_method
- `keyword_reference/cnt/visualization/font_size.md` — !font_size
- `keyword_reference/cnt/visualization/index.md` — 시각화용
- `keyword_reference/cnt/visualization/initial_style.md` — !initial_style
- `keyword_reference/cnt/visualization/isoline_number.md` — !isoline_number
- `keyword_reference/cnt/visualization/output_type.md` — !output_type
- `keyword_reference/cnt/visualization/surface_num.md` — !surface_num
- `keyword_reference/cnt/visualization/viewpoint.md` — !viewpoint
- `keyword_reference/cnt/visualization/visual.md` — !VISUAL
- `keyword_reference/cnt/visualization/x_resolution.md` — !x_resolution
- `keyword_reference/cnt/weld_line.md` — !WELD_LINE
- `keyword_reference/cnt/write.md` — !WRITE
- `keyword_reference/cnt/write_result.md` — !WRITE,RESULT
- `keyword_reference/cnt/write_visual.md` — !WRITE,VISUAL

---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/acceleration.md | TITLE: !ACCELERATION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/acceleration.html -->

# !ACCELERATION

 가속도 경계 조건의 정의

 
## 매개변수

 
```text
TYPE = INITIAL (초기 가속도 경계 조건)
     = TRANSIT (AMPLITUDE로 지정한 시간 이력 가속도 경계 조건: 기본값)
GRPID = 그룹 ID
AMP  = 시간 함수명 (!AMPLITUDE로 지정)
       !AMPLITUDE에서 시간 t와 계수 f(t)의 관계를 지정한다.
       아래 Value에 계수 f(t)를 곱한 값이 해당 시각의 구속값이 된다
       (지정하지 않은 경우: 시간과 계수의 관계는 f(t) = 1.0이 된다).
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NODE_ID, DOF_idS, DOF_idE, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_ID | I/C | 절점 번호 또는 절점 그룹명 |
| DOF_idS | I | 구속 자유도의 시작 번호 |
| DOF_idE | I | 구속 자유도의 종료 번호 |
| Value | R | 구속값(기본값: 0) |

 
## 사용 예

 
```text
!ACCELERATION, TYPE=TRANSIT, AMP=AMP1
  1, 1, 3, 0.0
  ALL, 3, 3
  ※ 구속값은 0.0
!ACCELERATION, TYPE=INITIAL
  1, 3, 3, 1.0
  2, 3, 3, 1.0
  3, 3, 3, 1.0
```

 
- 가속도 경계 조건의 경우 변위 경계 조건과 달리 여러 자유도를 한꺼번에 정의할 수 없으므로, DOF_idS와 DOF_idE는 동일한 번호여야 한다.
- `TYPE`이 `INITIAL`인 경우 `AMP`는 무효가 된다.
- `GRPID`는 `!STEP`의 `BOUNDARY` 행에서 참조하며, 가속도 경계 조건을 스텝별로 활성화하기 위해 사용한다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/autoinc_param.md | TITLE: !AUTOINC_PARAM -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/autoinc_param.html -->

# !AUTOINC_PARAM

 자동 증분 매개변수의 지정.

 
## 매개변수

 
| 매개변수명 | 유형 | 설명 |
| --- | --- | --- |
| NAME | C | 자동 증분 매개변수명(필수) |

 ** 두 번째 행 **

 감소 조건 및 그때의 시간 증분 감소율을 지정한다.

 
```text
(두 번째 행)　RS, NS_MAX, NS_SUM, NS_CONT, N_S
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| RS | R | 시간 증분 감소율(기본값: 0.5) |
| NS_MAX | I | Newton법 최대 반복 횟수의 임계값(기본값: 20) |
| NS_SUM | I | Newton법 총 반복 횟수의 임계값(기본값: 100) |
| NS_CONT | I | 접촉 반복 횟수의 임계값(기본값: 10) |
| N_S | I | 감소 조건이 성립할 때까지의 서브스텝 수(기본값: 2) |

 ** 세 번째 행 **

 증가 조건 및 그때의 시간 증분 증가율을 지정한다.

 
```text
(세 번째 행)　RL, NL_MAX, NL_SUM, NL_CONT, N_L
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| RL | R | 시간 증분 증가율(기본값: 1.75) |
| NL_MAX | I | Newton법 최대 반복 횟수의 임계값(기본값: 10) |
| NL_SUM | I | Newton법 총 반복 횟수의 임계값(기본값: 50) |
| NL_CONT | I | 접촉 반복 횟수의 임계값(기본값: 5) |
| N_L | I | 증가 조건이 성립할 때까지의 서브스텝 수(기본값: 2) |

 ** 4번째 행 **

 cutback을 설정한다.

 
```text
(4번째 행)　RC, N_C
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| RC | R | cutback 시 시간 증분 감소율(기본값: 0.25) |
| N_C | I | 연속 cutback 횟수의 허용 상한(기본값: 5) |

 
## 사용 예

 기본 설정과 동일하게 설정하는 경우

 
```text
!AUTOINC_PARAM, NAME=AP1
    0.5, 20, 100, 10, 2
    1.75, 10,  50,  5, 2
    0.25,  5
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/boundary.md | TITLE: !BOUNDARY -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/boundary.html -->

# !BOUNDARY

 변위 경계 조건 정의

 
## 매개변수

 
```text
GRPID      = 그룹 ID
AMP        = 시간 함수명 (!AMPLITUDE로 지정)
ROT_CENTER = 회전 변위 구속의 중심 절점 번호 또는 절점 집합명.
             지정한 경우 해당 !BOUNDARY는 회전 변위 구속으로 인식된다.
TOTAL        지정한 경우 변위량은 초기 배치로부터의 총량으로 취급된다 (기본값은 스텝 시작 시 배치로부터의 상대량)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NODE_ID, DOF_idS, DOF_idE, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_ID | I/C | 절점 번호 또는 절점 그룹명 |
| DOF_idS | I | 구속 자유도의 시작 번호 |
| DOF_idE | I | 구속 자유도의 종료 번호 |
| Value | R | 구속값(기본값: 0) |

 
## 사용 예

 
```text
!BOUNDARY, GRPID=1
  1, 1, 3, 0.0
  ALL, 3, 3,
```

 ※ 구속값은 0.0

 
```text
!BOUNDARY, ROT_CENTER=7, GRPID=1
  ROT_NODES, 1, 1, 3.141
  ROT_NODES, 2, 2, -4.188
```

 `ROT_NODES`에 대해 절점 7을 중심으로 크기 \(||(3.141, -4.188)|| = 5.233[rad]\), 회전축 \((3/5, -4/5, 0)\)의 회전을 가한다.

 `ROT_CENTER`에 의한 회전은 병진 3자유도에 대한 변위 구속이며, 셸 요소에 대한 4, 5, 6 자유도 구속과는 다르다.

 `ROT_CENTER`를 지정한 경우

 
- `ROT_CENTER`에 지정하는 절점 집합은 하나의 절점으로만 이루어진 집합으로 한다.
- 하나의 `!BOUNDARY` 데이터 블록 안에서는 `NODE_ID`를 모두 동일하게 한다.
- 지정하지 않은 자유도는 모두 0으로 구속된다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/cflux.md | TITLE: !CFLUX -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/cflux.html -->

# !CFLUX

 절점에 적용하는 집중 열유속의 정의

 
## 매개변수

 
```text
AMP = 열유속 이력 테이블명(!AMPLITUDE에서 지정)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NODE_GRP_NAME, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_GRP_NAME | C/I | 절점 그룹명 또는 절점 번호 |
| Value | R | 열유속 값 |

 
## 사용 예

 
```text
!CFLUX
  ALL, 1.0E-3
!CFLUX, AMP=FLUX1
  ALL, 1.0
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/cload.md | TITLE: !CLOAD -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/cload.html -->

# !CLOAD

 집중 하중을 정의합니다.

 
## 매개변수

 
```text
GRPID      = 그룹 ID
AMP        = 시간 함수명(!AMPLITUDE로 지정)
ROT_CENTER = 회전 중심 절점 번호 또는 절점 그룹명. 지정하면 !CLOAD가 토크 하중으로 인식됩니다.
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NODE_ID, DOF_id, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_ID | I/C | 절점 ID 또는 절점 그룹명 |
| DOF_id | I | 자유도 번호 |
| Value | R | 하중 값 |

 
## 사용 예

 
```text
!CLOAD, GRPID=1
  1, 1, 1.0e3
  ALL, 3, 10.0
!CLOAD, ROT_CENTER=7, GRPID=1
  TORQUE_NODES, 1, 3
  TORQUE_NODES, 3, -4
```

 절점 7을 중심으로 `ROT_NODES`에 크기 \(||(3, 0, -4)|| = 5\), 회전축 \((3/5, 0, -4/5)\)인 토크 하중을 적용합니다.

 `ROT_CENTER`로 지정한 토크 하중은 내부적으로 지정한 토크와 등가인 절점 하중으로 적용되며, 셸 요소의 자유도 4, 5, 6에 적용하는 절점 하중과는 다릅니다.

 `ROT_CENTER`를 지정한 경우

 
- `ROT_CENTER`로 지정한 절점 그룹에는 정확히 하나의 절점만 포함되어야 합니다.
- 하나의 `!CLOAD` 데이터 블록 내에서는 모든 `NODE_ID` 값이 동일해야 합니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/contact.md | TITLE: !CONTACT -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/contact.html -->

# !CONTACT

 접촉 조건 정의

 
## 매개변수

 
```text
GRPID       = 경계 조건 그룹 ID
INTERACTION = SSLID(미소 미끄럼 접촉·기본값) / FSLID(유한 미끄럼 접촉) / TIED(결합 구속)
NTOL        = 접촉 법선 방향 수렴 임계값(기본값: 1.e-5)
TTOL        = 접촉 접선 방향 수렴 임계값(기본값: 1.e-3)
NPENALTY    = 접촉 법선 방향 페널티(기본값: 강성 행렬×1.e3)
TPENALTY    = 접촉 접선 방향 페널티(기본값: 1.e3)
DAMP_ALPHA  = 접촉 감쇠 계수(무차원. 기본값: 0.0 = 감쇠 없음)
DAMP_GACT   = 접촉 감쇠의 작용 거리(길이 단위. 기본값: 0.0 = 감쇠 없음)
CONTACTPARAM = 접촉 판정 매개변수 세트 이름(!CONTACT_PARAM, NAME으로 지정)
```

 ** 두 번째 행부터 **

 
### `INTERACTION = SSLID, FSLID`인 경우

 
```text
(두 번째 행) PAIR_NAME, fcoef, factor
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| PAIR_NAME | C | 접촉 쌍 이름(`!CONTACT PAIR`에서 정의) |
| fcoef | R | 마찰 계수(기본값: 0.0) |
| factor | R | 마찰의 페널티 강성 |

 
### `INTERACTION = TIED`인 경우

 
```text
(두 번째 행) PAIR_NAME
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| PAIR_NAME | C | 접촉 쌍 이름(`!CONTACT PAIR`에서 정의) |

 
## 사용 예

 
```text
!CONTACT_ALGO, TYPE=SLAGRANGE
!CONTACT, GRPID=1, INTERACTION=FSLID
  CP1, 0.1, 1.0e+5
```

 접촉 감쇠를 함께 사용하는 경우

 
```text
!CONTACT_ALGO, TYPE=SLAGRANGE
!CONTACT, GRPID=1, INTERACTION=FSLID, DAMP_ALPHA=1.0e-2, DAMP_GACT=2.0e-2
  CP1, 0.0, 5.0e+2
```

 
## 비고

 
- CONTACTPARAM을 지정하는 경우 참조되는 !CONTACT_PARAM은 !CONTACT 카드보다 앞에 정의되어 있어야 한다. 또한 이 매개변수를 생략하면 기본 접촉 판정 매개변수 세트가 사용된다.
- 접촉 감쇠는 `DAMP_ALPHA`와 `DAMP_GACT`가 모두 양수일 때에만 활성화된다. 둘 중 하나만 지정해도 작동하지 않는다
- 접촉 감쇠의 기능 개요와 감쇠 계수 및 작용 거리의 설정 방법은 [기능 참조: 접촉 및 임베딩](../../functions_reference/05_contact.html)을 참조한다


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/contact_algo.md | TITLE: !CONTACT_ALGO -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/contact_algo.html -->

# !CONTACT_ALGO

 접촉 해석 알고리즘 지정

 
## 매개변수

 
```text
TYPE = SLAGRANGE (Lagrange 승수법)
       ALAGRANGE (증강 라그랑주 승수법)
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/contact_param.md | TITLE: !CONTACT_PARAM -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/contact_param.html -->

# !CONTACT_PARAM

 접촉 판정 매개변수 지정.

 
## 매개변수

 
| 매개변수명 | 유형 | 설명 |
| --- | --- | --- |
| NAME | C | 접촉 판정 매개변수명(필수) |

 ** 두 번째 행 **

 면내 방법의 각종 클리어런스를 지정한다.

 
```text
(두 번째 행)　CLEARANCE, CLR_SAME_ELEM, CLR_DIFFLPOS, CLR_CAL_NORM
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| CLEARANCE | R | 일반 클리어런스(기본값: 1e-2) |
| CLR_SAME_ELEM | R | 이미 접촉 중인 요소에 대한 클리어런스(너무 많이 움직이는 경우 완화)(기본값: 5e-3) |
| CLR_DIFFLPOS | R | 요소 내에서 서로 다른 점으로 간주하는 클리어런스(진동하는 경우 완화)(기본값: 1e-2) |
| CLR_CAL_NORM | R | 법선을 계산할 때 에지 위로 간주하는 클리어런스(기본값: 1e-4) |

 ** 세 번째 행 **

 면외 방향의 각종 클리어런스 등을 지정한다.

 
```text
(세 번째 행)　DISTCLR_INIT, DISTCLR_FREE, DISTCLR_NOCHECK, TENSILE_FORCE
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| DISTCLR_INIT | R | 최초 판정에 사용하는 면외 방향 클리어런스(기본값: 1e-2) |
| DISTCLR_FREE | R | 비접촉 절점 판정에 사용하는 면외 방향 클리어런스(약간 관통한 뒤 접촉으로 간주)(기본값: -1e-6) |
| DISTCLR_NOCHECK | R | 이미 접촉 중인 절점에 대해 거리 방향 판정을 건너뛸 때의 면외 방향 클리어런스(박리는 인장력으로 판정하므로 큰 값을 지정하여 접촉 상태를 유지)(기본값: 1.0) |
| TENSILE_FORCE | R | 박리로 판정하는 인장력(기본값: -1e-2) |

 
## 사용 예

 기본 설정과 동일하게 설정하는 경우

 
```text
!CONTACT_PARAM, NAME=CPARAM1
    1.0e-2,  5.0e-3,  1.0e-2,  1.0e-4
    1.0e-2, -1.0e-6,  1.0,    -1.0e-2
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/couple.md | TITLE: !COUPLE -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/couple.html -->

# !COUPLE

 연성 면을 정의합니다(연성 해석에서만 사용).

 
## 매개변수

 
```text
TYPE =    1: 단방향 연성          (FrontISTR가 데이터 수신부터 시작)
          2: 단방향 연성          (FrontISTR가 데이터 송신부터 시작)
          3: 스태거드 양방향 연성 (FrontISTR가 데이터 수신부터 시작)
          4: 스태거드 양방향 연성 (FrontISTR가 데이터 송신부터 시작)
          5: 반복 분할 양방향 연성  (FrontISTR가 데이터 수신부터 시작)
          6: 반복 분할 양방향 연성  (FrontISTR가 데이터 송신부터 시작)
ISTEP =   스텝 수
          해석 시작부터 여기서 지정한 스텝 수까지 유체력에 0에서 1로 증가하는 선형 함수를 곱하여 증가시키고,
          그 이후에는 입력 유체력을 그대로 적용합니다.
WINDOW => 0 : 윈도 함수(*)를 곱한 유체력을 적용
```

 (*) \(\frac{1}{2}(1 - \cos\frac{2\pi i}{N}),(i: 현재 스텝, N: 현재 해석의 총 스텝 수)\)

 ** 두 번째 행부터 **

 
```text
(두 번째 행) COUPLING_SURFACE_ID
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| SURFACE_ID | C | 표면 그룹명 |

 
## 사용 예

 
```text
!COUPLE , TYPE=1
  SCOUPLE1
  SCOUPLE2
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/creep.md | TITLE: !CREEP -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/creep.html -->

# !CREEP

 크리프 재료 정의

 `!CREEP`를 정의할 때는 같은 `!MATERIAL` 안에서 `!ELASTIC`도 함께 정의해야 한다.

 
## 매개변수

 
```text
TYPE = NORTON (기본값)
DEPENDENCIES = 0 (기본값) / 1
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) A, n, m, Temperature
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| A | R | 재료 계수 |
| n | R | 재료 계수 |
| m | R | 재료 계수 |
| Temperature | R | 온도(`DEPENDENCIES=1`일 때 필요) |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/damping.md | TITLE: !DAMPING -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/damping.html -->

# !DAMPING

 요소별로 작용하는 Rayleigh 감쇠 계수를 정의한다. `!MATERIAL` 카드에 이어서 지정한다.

 재료 \(i\)에 속하는 요소의 감쇠 행렬 \(C_i\)는 해당 재료의 질량 행렬 \(M_i\)와 접선 강성 행렬 \(K_i\)를 사용하여

 \[ C_i = R_m M_i + R_k K_i \] 로 평가되어 전역 감쇠 행렬에 조립된다. `!DYNAMIC` 카드에서 지정하는 전역 Rayleigh 감쇠 계수(`ray_m`, `ray_k`)는 `!DAMPING`으로 해당 재료에 값이 지정된 경우 그 재료에 대해서는 무시되며, `!DAMPING`의 값이 우선한다.

 암시적 동적 해석(`!SOLUTION, TYPE=DYNAMIC` + Newmark-β)에서만 유효하다. 정적 해석, 모드 해석, 주파수 응답 해석 및 명시적 동적 해석에서는 참조되지 않는다.

 
## 매개변수

 
```text
TYPE = RAYLEIGH : Rayleigh 감쇠(기본값, 현재는 이 모델만 지원)
```

 
| 매개변수명 | 유형 | 설명 |
| --- | --- | --- |
| TYPE | P | 감쇠 모델의 종류. 생략 가능. 현재는 `RAYLEIGH`만 지원(기본값) |

 ** 두 번째 행 **

 
```text
(두 번째 행) Rm, Rk
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| Rm | R | 질량 행렬에 곱하는 비례 계수 \(R_m\)(기본값: 0.0) |
| Rk | R | 강성 행렬에 곱하는 비례 계수 \(R_k\)(기본값: 0.0) |

 
## 사용 예

 
```text
!MATERIAL, NAME=M1
!ELASTIC
 210000.0, 0.3
!DENSITY
 7.85e-9
!DAMPING, TYPE=RAYLEIGH
 0.5, 1.0e-3
```

 
## 관련 항목

 
- [`!DYNAMIC`](dynamic.ja.md): 전역 Rayleigh 감쇠 계수 지정
- [`!MATERIAL`](material.ja.md): 재료 카드의 일반 형식


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/dashpot_a.md | TITLE: !DASHPOT_A -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/dashpot_a.html -->

# !DASHPOT_A

 축방향 댐퍼의 정의.

 2절점 커넥터 요소(요소 타입 511)가 연결하는 두 절점 사이에, 그 두 절점을 잇는 축 방향으로 작용하는 댐퍼를 정의한다. 댐퍼는 감쇠 행렬에만 기여하므로, 암시적 동적 해석(`!SOLUTION, TYPE=DYNAMIC`이며 [`!DYNAMIC`](dynamic.ja.md)의 `idx_eqa=1`)에서 사용한다. 정적 해석에서는 효과가 없다. 특정 자유도끼리 연결하는 댐퍼를 정의하려면 [`!DASHPOT_D`](dashpot_d.ja.md)를 사용한다.

 
## 메시 데이터 측 정의

 `!DASHPOT_A`는 재료 특성으로 감쇠 계수를 지정하는 키워드이며, 이 키워드만으로는 댐퍼가 생성되지 않는다. 메시 데이터(msh) 측에 다음 두 항목을 정의해야 한다.

 
- 댐퍼가 연결하는 두 절점을 연결 정보로 갖는 요소 타입 511의 요소(→ [`!ELEMENT`](../msh/element.ja.md))
- 해당 요소 그룹에 대한 `TYPE=INTERFACE` 섹션(→ [`!SECTION`](../msh/section.ja.md)). `MATERIAL`에는 이 키워드로 감쇠 계수를 정의한 재료명을 지정한다

 솔리드 요소와 커넥터 요소를 함께 사용하는 경우에는 요소 그룹을 분리하고 각각에 `!SECTION`을 정의한다.

 
## 매개변수

 없음

 ** 두 번째 행 **

 축방향 댐퍼 계수를 지정한다.

 
```text
(두 번째 행) DAMPING_COEFF
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| DAMPING_COEFF | R | 감쇠 계수 |

 
## 사용 예

 절점 1과 절점 5를 연결하는 축방향 댐퍼(감쇠 계수 100)를 솔리드 요소(요소 그룹 `ESLD`)와 함께 정의하는 경우

 msh

 
```text
!ELEMENT, TYPE=511, EGRP=EDAMP
3, 1, 5
!SECTION, TYPE=SOLID,     EGRP=ESLD,  MATERIAL=MATERIAL1
1.0
!SECTION, TYPE=INTERFACE, EGRP=EDAMP, MATERIAL=MATERIAL2
1.0
```

 cnt

 
```text
!SOLUTION, TYPE=DYNAMIC, NONLINEAR
!MATERIAL, NAME=MATERIAL2
!DASHPOT_A
100
```

 
## 비고

 
- 댐퍼는 감쇠 행렬에만 기여한다. 정적 해석(`!SOLUTION, TYPE=STATIC`)에서는 강성 행렬에 기여하지 않으므로 효과가 없다.
- 감쇠 계수가 정의된 재료는 커넥터 요소 전용 재료로 취급된다. 동일한 재료명에 탄성 물성을 부여하여 솔리드 요소와 공유할 수 없으므로 솔리드 요소와는 다른 재료명을 사용한다.
- 축방향 댐퍼는 축 방향의 한 성분에만 감쇠를 갖는다. 스프링과 마찬가지로 축에 직교하는 방향은 다른 요소나 경계 조건으로 지지해야 한다.

 
## 관련 항목

 
- [`!DASHPOT_D`](dashpot_d.ja.md)
- [`!SPRING_A`](spring_a.ja.md)
- [`!DYNAMIC`](dynamic.ja.md)
- [`!ELEMENT`](../msh/element.ja.md)
- [`!SECTION`](../msh/section.ja.md)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/dashpot_d.md | TITLE: !DASHPOT_D -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/dashpot_d.html -->

# !DASHPOT_D

 자유도 지정 대시포트의 정의.

 2절점 커넥터 요소(요소 타입 511)가 연결하는 두 절점에 대해, 절점 1의 자유도 `DOF1`과 절점 2의 자유도 `DOF2`를 연결하는 대시포트를 정의한다. 대시포트의 방향은 자유도 번호로 정해지며 모델의 변형을 따라가지 않는다. 대시포트는 감쇠 행렬에만 기여하므로, 암시적 동적 해석(`!SOLUTION, TYPE=DYNAMIC`이며 [`!DYNAMIC`](dynamic.ja.md)의 `idx_eqa=1`)에서 사용한다. 정적 해석에서는 효과가 없다. 두 절점을 연결하는 축 방향 대시포트를 정의하려면 [`!DASHPOT_A`](dashpot_a.ja.md)를 사용한다.

 여기서 절점 1과 절점 2는 `!ELEMENT`의 연결 정보에 지정한 순서에서 각각 첫 번째와 두 번째 절점이다.

 
## 메시 데이터 측 정의

 `!DASHPOT_D`는 재료 속성으로 감쇠 계수를 지정하는 키워드이며, 이 키워드만으로는 대시포트가 존재하지 않는다. 메시 데이터(msh) 측에 다음 두 가지를 정의해야 한다.

 
- 대시포트가 연결하는 두 절점을 연결 정보로 갖는 요소 타입 511의 요소(→ [`!ELEMENT`](../msh/element.ja.md))
- 해당 요소 그룹에 대한 `TYPE=INTERFACE` 섹션(→ [`!SECTION`](../msh/section.ja.md)). `MATERIAL`에는 이 키워드로 감쇠 계수를 정의한 재료명을 지정한다

 솔리드 요소와 커넥터 요소를 혼재시키는 경우에는 요소 그룹을 나누고 각각에 `!SECTION`을 정의한다.

 
## 매개변수

 없음

 ** 두 번째 행부터 **

 자유도 지정 대시포트의 계수를 지정한다. 자유도 조합마다 한 줄씩 작성하며, 여러 줄을 나열하여 여러 자유도 조합에 대시포트를 정의할 수 있다.

 
```text
(두 번째 행) DOF1, DOF2, DAMPING_COEFF
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| DOF1 | I | 절점 1의 자유도 번호(1,2,3) |
| DOF2 | I | 절점 2의 자유도 번호(1,2,3) |
| DAMPING_COEFF | R | 감쇠 계수 |

 
## 사용 예

 절점 1과 절점 5를 연결하는 대시포트를 솔리드 요소(요소 그룹 `ESLD`)와 혼재하여 정의하고, x·y·z 각 방향에 서로 다른 감쇠 계수를 설정하는 경우

 msh

 
```text
!ELEMENT, TYPE=511, EGRP=EDAMP
3, 1, 5
!SECTION, TYPE=SOLID,     EGRP=ESLD,  MATERIAL=MATERIAL1
1.0
!SECTION, TYPE=INTERFACE, EGRP=EDAMP, MATERIAL=MATERIAL2
1.0
```

 cnt

 
```text
!SOLUTION, TYPE=DYNAMIC, NONLINEAR
!MATERIAL, NAME=MATERIAL2
!DASHPOT_D
1, 1, 100.0
2, 2, 300.0
3, 3, 400.0
```

 
## 비고

 
- 대시포트는 감쇠 행렬에만 기여한다. 정적 해석(`!SOLUTION, TYPE=STATIC`)에서는 강성 행렬에 기여하지 않으므로 효과가 없다.
- `DOF1`과 `DOF2`에 서로 다른 자유도 번호를 지정하면 절점 1의 한 방향과 절점 2의 다른 방향을 연결하는 대시포트가 된다. 같은 방향의 대시포트로 하려면 `DOF1`과 `DOF2`에 같은 번호를 지정한다.
- 감쇠 계수를 정의한 재료는 커넥터 요소 전용 재료로 취급된다. 같은 재료명에 탄성 물성을 부여하여 솔리드 요소와 공용할 수 없으므로 솔리드 요소와는 다른 재료명을 사용한다.

 
## 관련 항목

 
- [`!DASHPOT_A`](dashpot_a.ja.md)
- [`!SPRING_D`](spring_d.ja.md)
- [`!DYNAMIC`](dynamic.ja.md)
- [`!ELEMENT`](../msh/element.ja.md)
- [`!SECTION`](../msh/section.ja.md)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/density.md | TITLE: !DENSITY -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/density.html -->

# !DENSITY

 질량 밀도의 정의

 
## 매개변수

 
```text
DEPENDENCIES = 종속 변수의 수(미구현)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) density
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| density | R | 질량 밀도 |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/dflux.md | TITLE: !DFLUX -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/dflux.html -->

# !DFLUX

 요소 표면에 부여하는 분포 열유속 및 내부 발열 정의

 
## 매개변수

 
```text
AMP = 열유속 이력 테이블 이름(!AMPLITUDE에서 지정)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) ELEMENT_GRP_NAME, LOAD_type, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| ELEMENT_GRP_NAME | C/I | 요소 그룹명 또는 요소 번호 |
| LOAD_type | C | 하중 유형 번호 |
| Value | R | 열유속 값 |

 
## 사용 예

 
```text
!DFLUX
  ALL, S1, 1.0
!DFLUX, AMP=FLUX2
  ALL, S0, 1.0
```

 
### 하중 매개변수

 
| 하중 유형 번호 | 작용면 | 매개변수 |
| --- | --- | --- |
| BF | 요소 전체 | 발열량 |
| S1 | 제1면 | 열유속 값 |
| S2 | 제2면 | 열유속 값 |
| S3 | 제3면 | 열유속 값 |
| S4 | 제4면 | 열유속 값 |
| S5 | 제5면 | 열유속 값 |
| S6 | 제6면 | 열유속 값 |
| S0 | 셸 면 | 열유속 값 |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/dload.md | TITLE: !DLOAD -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/dload.html -->

# !DLOAD

 분포 하중 정의

 
## 매개변수

 
```text
GRPID = 그룹 ID
AMP = 시간 함수 이름(!AMPLITUDE로 지정)
FOLLOW = YES(기본값) / NO
         (압력 하중의 방향을 변형에 추종시킬지 여부를 지정, 유한 변형 해석에서 유효)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) ID_NAME, LOAD_type, param1, param2,…
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| ID_NAME | I/C | 면 그룹명, 요소 그룹명 또는 요소 번호 |
| LOAD_type | C | 하중 유형 번호 |
| param* | R | 하중 매개변수(아래 참조) |

 
### 하중 매개변수

 
| 하중 유형 번호 | 종류 | 매개변수 수 | 매개변수 순서와 의미 |
| --- | --- | --- | --- |
| S | 면 그룹으로 지정한 면에 대한 압력 | 1 | 압력값 |
| P0 | 셸 요소에 대한 압력 | 1 | 압력값 |
| PX | 셸 요소 X방향 압력 | 1 | 압력값 |
| PY | 셸 요소 Y방향 압력 | 1 | 압력값 |
| PZ | 셸 요소 Z방향 압력 | 1 | 압력값 |
| P1 | 제1면에 대한 압력 | 1 | 압력값 |
| P2 | 제2면에 대한 압력 | 1 | 압력값 |
| P3 | 제3면에 대한 압력 | 1 | 압력값 |
| P4 | 제4면에 대한 압력 | 1 | 압력값 |
| P5 | 제5면에 대한 압력 | 1 | 압력값 |
| P6 | 제6면에 대한 압력 | 1 | 압력값 |
| BX | X방향 체적력 | 1 | 체적력 값 |
| BY | Y방향 체적력 | 1 | 체적력 값 |
| BZ | Z방향 체적력 | 1 | 체적력 값 |
| GRAV | 중력 | 4 | 중력 가속도의 크기, 중력 방향의 X성분(방향 코사인), Y성분, Z성분 |
| CENT | 원심력 | 7 | 각속도, 회전축 위 점의 X좌표, Y좌표, Z좌표, 회전축 방향의 X성분, Y성분, Z성분 |

 
## 사용 예

 
```text
!DLOAD, GRPID=1
  1, P1, 1.0
  ALL, BX, 1.0
  ALL, GRAV, 9.8, 0.0, 0.0, -1.0
  ALL, CENT, 188.495, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/dynamic.md | TITLE: !DYNAMIC -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/dynamic.html -->

# !DYNAMIC

 동적 해석 제어

 
## 매개변수

 
```text
TYPE = LINEAR    : 선형 동적 해석
       NONLINEAR : 비선형 동적 해석
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) idx_eqa, idx_resp
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| idx_eqa | I | 운동방정식 해법(직접 시간 적분)(기본값: 1)<br>1: 암시적 방법(Newmark-β법)<br>11: 명시적 방법(중앙 차분법) |
| idx_resp | I | 해석 유형(기본값: 1)<br>1: 시간이력 응답 해석<br>2: 주파수 응답 해석 |

 
### `idx_resp=1`인 경우(시간이력 응답 해석)

 
```text
(세 번째 행) t_start , t_end , n_step, t_delta
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| t_start | R | 해석 시작 시간(기본값: 0.0), 미사용 |
| t_end | R | 해석 종료 시간(기본값: 1.0), 미사용 |
| n_step | I | `!STEP`을 생략한 경우 또는 하위 호환용 기본 서브스텝 수(기본값: 1) |
| t_delta | R | `!STEP`을 생략한 경우 또는 하위 호환용 기본 시간 증분(기본값: 1.0) |

 
```text
(4번째 행) gamma , beta
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| gamma | R | Newmark-β법의 매개변수 γ(기본값: 0.5) |
| beta | R | Newmark-β법의 매개변수 β(기본값: 0.25) |

 
```text
(5번째 행) idx_mas ,idx_dmp , ray_m ,ray_k
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| idx_mas | I | 질량 행렬 유형(기본값: 1)<br>1: 집중 질량 행렬<br>2: 일관 질량 행렬 |
| idx_dmp | I | 1: Rayleigh 감쇠(기본값: 1) |
| ray_m | R | Rayleigh 감쇠 매개변수 Rm(기본값: 0.0) |
| ray_k | R | Rayleigh 감쇠 매개변수 Rk(기본값: 0.0) |

 
```text
(6번째 행) nout, node_monit_1, nout_monit
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| nout | I | 미사용 |
| node_monit_1 | I | 모니터링 절점 번호(전역) 또는 절점 그룹 이름 |
| nout_monit | I | 모니터링 결과 출력 간격(기본값: 1) |

 참고: 이 줄에서 지정한 모니터링 절점의 정보는 전역 절점 번호를 *NID*로 사용하여 출력됩니다. 변위는 `<dyna_disp_NID.txt>`에 스텝 번호, 서브스텝 번호, 현재 시간, *NID*, u1, u2, u3 순서로 출력됩니다. 속도와 가속도는 각각 동일한 순서로 `<dyna_velo_NID.txt>`와 `<dyna_acce_NID.txt>`에 출력됩니다. 절점 변형률은 `<dyna_strain_NID.txt>`에 스텝 번호, 서브스텝 번호, 현재 시간, *NID*, e11, e22, e33, e12, e23, e13 순서로 출력됩니다. 절점 응력은 `<dyna_stress_NID.txt>`에 스텝 번호, 서브스텝 번호, 현재 시간, *NID*, s11, s22, s33, s12, s23, s13, smises 순서로 출력됩니다. 절점 그룹을 사용하여 여러 모니터링 절점을 지정한 경우 위의 각 파일은 절점마다 별도로 출력됩니다. 또한 이 출력을 지정하면 전체 해석 모델의 운동 에너지, 변형 에너지 및 총 에너지가 `<dyna_energy.txt>`에 스텝 번호, 서브스텝 번호, 현재 시간, 운동 에너지, 변형 에너지, 총 에너지 순서로 출력됩니다.

 
## 비고

 
- 시간이력 동적 해석에서는 여러 개의 `!STEP` 정의를 사용하여 각 스텝별로 경계 조건 및 하중 그룹을 전환할 수 있습니다.
- 고정 증분 시간 제어에서는 `!STEP`을 사용하여 두 번째 행에 `SUBSTEPS` 및 `DTIME, ETIME`을 명시적으로 지정하는 것을 권장합니다. `n_step` 및 `t_delta`는 `!STEP`을 생략한 단일 스텝 해석의 기본값 또는 하위 호환을 위해 사용됩니다.
- `!STEP, INC_TYPE=AUTO`를 지정하면 자동 증분은 `!STEP`의 두 번째 행에 지정한 초기 시간 증분, 스텝 시간, 최소 시간 증분 및 최대 시간 증분을 사용합니다. 자동 증분 매개변수는 `!AUTOINC_PARAM`으로 지정할 수 있습니다.

 
```text
(7번째 행) iout_list(1), iout_list(2), iout_list(3), iout_list(4), iout_list(5), iout_list(6)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| iout_list(1) | I | 변위 출력 설정(기본값: 0)<br>0: 출력하지 않음<br>1: 출력 |
| iout_list(2) | I | 속도 출력 설정(기본값: 0)<br>0: 출력하지 않음<br>1: 출력 |
| iout_list(3) | I | 가속도 출력 설정(기본값: 0)<br>0: 출력하지 않음<br>1: 출력 |
| iout_list(4) | I | 반력 출력 설정(기본값: 0)<br>0: 출력하지 않음<br>1: 출력 |
| iout_list(5) | I | 변형률 출력 설정(기본값: 0)<br>0: 출력하지 않음(요소 기준 또는 절점 기준)<br>1: 출력<br>2: 출력(절점 기준)<br>3: 출력(요소 기준) |
| iout_list(6) | I | 응력 출력 설정(기본값: 0)<br>0: 출력하지 않음(요소 기준 또는 절점 기준)<br>1: 출력<br>2: 출력(절점 기준)<br>3: 출력(요소 기준) |

 
## 사용 예

 
```text
!DYNAMIC, TYPE=NONLINEAR
  1 , 1
  0.0, 1.0, 500, 1.0000e-5
  0.5, 0.25
  1, 1, 0.0, 0.0
  100, 55, 1
  0, 0, 0, 0, 0, 0
```

 
### `idx_resp=2`인 경우(주파수 응답 해석)

 
```text
(세 번째 행) f_start, f_end, n_freq, f_disp
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| f_start | R | 하한 주파수 |
| f_end | R | 상한 주파수 |
| n_freq | I | 응답 계산점 수 |
| f_disp | R | 변위를 측정하는 주파수 |

 
```text
(4번째 행) t_start, t_end
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| t_start | R | 실시간에서의 진동 시작 시간 |
| t_end | R | 실시간에서의 진동 종료 시간 |

 
```text
(5번째 행) idx_mas, idx_dmp, ray_m ,ray_k
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| idx_mas | I | 질량 행렬 유형(1(집중 질량 행렬)로 고정) |
| idx_dmp | I | 1: Rayleigh 감쇠 |
| ray_m | R | Rayleigh 감쇠 매개변수 Rm(기본값: 0.0) |
| ray_k | R | Rayleigh 감쇠 매개변수 Rk(기본값: 0.0) |

 
```text
(6번째 행) nout, vistype, nodeout
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| nout | I | 시간 영역의 샘플 수 |
| vistype | I | 시각화 데이터 출력 설정(1: 모드 공간, 2: 물리 공간) |
| nodeout | I | 주파수 영역의 모니터링 절점 ID |

 
```text
(7번째 행) iout_list(1), iout_list(2), iout_list(3), iout_list(4), iout_list(5), iout_list(6)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| iout_list(1) | I | 출력 제어: 변위(1: 출력, 0: 출력하지 않음) |
| iout_list(2) | I | 출력 제어: 속도(1: 출력, 0: 출력하지 않음) |
| iout_list(3) | I | 출력 제어: 가속도(1: 출력, 0: 출력하지 않음) |
| iout_list(4) | I | 무시 |
| iout_list(5) | I | 무시 |
| iout_list(6) | I | 무시 |

 
## 사용 예

 
```text
!DYNAMIC
  11 , 2
  14000, 16000, 20, 15000.0
  0.0, 6.6e-5
  1, 1, 0.0, 7.2E-7
  10, 2, 1
  1, 1, 1, 1, 1, 1
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/echo.md | TITLE: !ECHO -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/echo.html -->

# !ECHO

 절점 데이터, 요소 데이터 및 재료 데이터를 로그 파일에 출력한다.

 
## 매개변수

 없음


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/eigen.md | TITLE: !EIGEN -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/eigen.html -->

# !EIGEN

 모드 해석의 매개변수 설정

 
## 매개변수

 없음

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NGET, LCZTOL, LCZMAX, SIGMA
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NSET | I | 고유값 수 |
| LCZTOL | R | 허용 오차 (기본값: 1.0e-8) |
| LCZMAX | I | 최대 반복 횟수 (기본값: 60) |
| SIGMA | R | 시프트량\(\sigma\) (기본값: 0.01) |

 
## 사용 예

 
```text
!EIGEN
  3, 1.0e-10, 40, 0.01
```

 `SIGMA`는 일반적으로 기본값 `0.01`에서 변경할 필요가 없다. 다만 프리-프리 조건(=구속 조건이 없는 모드 해석)을 수행할 때는 변경해야 할 가능성이 있다. 프리-프리 조건의 모드 해석을 수행했으며 모드 해석의 고유값 해에 0Hz가 나타난 경우, `SIGMA`를 `0.01`보다 큰 값(예: `1.0`)으로 설정해 보면 해결될 가능성이 있다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/eigenread.md | TITLE: !EIGENREAD -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/eigenread.html -->

# !EIGENREAD

 주파수 응답 해석에 사용할 고유값·고유모드 지정

 
## 매개변수

 없음

 ** 두 번째 행부터 **

 
```text
(두 번째 행) eigenlog_filename
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| eigenlog_filename | C | 모드 해석의 로그 파일명 |

 
```text
(세 번째 행) start_mode, end_mode
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| start_mode | I | 모드 해석에 사용할 지정 모드 시작점 |
| end_mode | I | 모드 해석에 사용할 지정 모드 끝점 |

 
## 사용 예

 
```text
!EIGENREAD
  eigen_0.log
  1, 5
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/elastic.md | TITLE: !ELASTIC -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/elastic.html -->

# !ELASTIC

 탄성 재료의 정의

 
## 매개변수

 
```text
TYPE = ISOTROPIC (기본값) / ORTHOTROPIC / USER
DEPENDENCIES = 0 (기본값) / 1
INFINITESIMAL    지정한 경우 미소변형 가정이 적용된다
CAUCHY           지정하면 갱신 라그랑주법을 적용한다
```

 ** 두 번째 행부터 **

 
### `TYPE = ISOTROPIC`인 경우

 (두 번째 행) YOUNGS, POISSON, Temperature

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| YOUNGS | R | 영률 |
| POISSON | R | 포아송비 |
| Temperature | R | 온도(`DEPENDENCIES`=1일 때 필요) |

 
### `TYPE= ORTHOTROPIC`인 경우

 
> **주의:** `TYPE=ORTHOTROPIC`을 사용할 때는 대응하는 `!SECTION`에 반드시 `ORIENTATION=` 매개변수로 국소 좌표계를 지정해야 한다. `ORIENTATION=`을 생략하면 솔버가 실행 중 비정상 종료한다.

 ```text
(두 번째 행) E1, E2, E3, ν12, ν13, ν23, G12, G13, G23, Temperature
```

 [ \begin{bmatrix} \varepsilon_{11} \\ \varepsilon_{22} \\ \varepsilon_{33} \\ 2\varepsilon_{12} \\ 2\varepsilon_{23} \\ 2\varepsilon_{31} \end{bmatrix} = \begin{bmatrix} 1/E_1 & -\nu_{12}/E_1 & -\nu_{13}/E_1 & 0 & 0 & 0 \\ & 1/E_2 & -\nu_{23}/E_2 & 0 & 0 & 0 \\ & & 1/E_3 & 0 & 0 & 0 \\ & & & 1/G_{12} & 0 & 0 \\ & 대칭 & & & 1/G_{23} & 0 \\ & & & & & 1/G_{31} \end{bmatrix} \begin{bmatrix} \sigma_{11} \\ \sigma_{22} \\ \sigma_{33} \\ \sigma_{12} \\ \sigma_{23} \\ \sigma_{31} \end{bmatrix} ]

 
### `TYPE = USER`인 경우

 
```text
(두 번째 행-10번째 행) v1, v2, v3, v4, v5, v6, v7, v8, v9, v10
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/element_activation.md | TITLE: !ELEMENT_ACTIVATION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/element_activation.html -->

# !ELEMENT_ACTIVATION

 요소의 활성화/비활성화 정의

 
## 매개변수

 
```text
MODE      = 제어 모드 (FIXED, AMPLITUDE, DAMAGE) [필수]
STATE     = 요소 상태 (ON: 활성, OFF: 비활성) (MODE=FIXED일 때 필수)
AMP       = 진폭명 (MODE=AMPLITUDE일 때 필수)
MEASURE   = 판정량 (STRESS, STRAIN) (MODE=DAMAGE일 때 필수)
EPSILON   = 더미 강성의 계수(기본값: 1.0E-6)
GRPID     = 제어 그룹 ID
```

 ** 두 번째 행부터 **

 
### `MODE = FIXED, AMPLITUDE`인 경우

 
```text
ELEM_GRP_NAME
```

 
### `MODE = DAMAGE`인 경우

 
```text
ELEM_GRP_NAME, LOWER, UPPER
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| ELEM_GRP_NAME | C | 요소 그룹명 |
| LOWER | R | 응력/변형률의 하한값 |
| UPPER | R | 응력/변형률의 상한값 |

 
## 사용 예

 
```text
!ELEMENT_ACTIVATION, MODE=FIXED, STATE=OFF, EPSILON=1.0E-3, GRPID=1
PART_1
!ELEMENT_ACTIVATION, MODE=AMPLITUDE, AMP=AMP1, EPSILON=1.0E-3, GRPID=2
PART_2
!ELEMENT_ACTIVATION, MODE=DAMAGE, MEASURE=STRESS, EPSILON=1.0E-6, GRPID=3
PART_3, 100.0, 200.0
!STEP, SUBSTEPS=10
BOUNDARY,1
LOAD,1
!STEP, SUBSTEPS=10
BOUNDARY,1
LOAD,1
ELEMACT,1
!STEP, SUBSTEPS=10
BOUNDARY,1
LOAD,1
ELEMACT,1
ELEMACT,2
```

 
## 비고

 
- `MODE`로 동작을 배타적으로 지정한다(`MODE=FIXED|AMPLITUDE|DAMAGE`).
- `MODE=FIXED`인 경우 `STATE=ON/OFF`로 항상 활성/비활성을 지정한다.
- `MODE=AMPLITUDE`인 경우 진폭 값에 따라 활성/비활성을 전환한다(`amp_val > 0.5`이면 활성, 그 외에는 비활성).
- `MODE=DAMAGE`인 경우 `MEASURE=STRESS|STRAIN`으로 지정한 양이 범위를 벗어난(LOWER ≤ 값 ≤ UPPER를 만족하지 않는) 요소를 비활성화한다. 비활성화 후에는 다시 활성화하지 않는다.
- GRPID로 지정한 제어 그룹은 `!STEP` 내의 `ELEMACT, id`로 활성화한다.
- `!STEP` 내에서 `ELEMACT, id`를 지정하지 않은 경우 해당 요소 활성화 설정은 적용되지 않는다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/embed.md | TITLE: !EMBED -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/embed.html -->

# !EMBED

 임베딩의 정의

 
## 매개변수

 
```text
GRPID       = 경계조건 그룹 ID
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) PAIR_NAME
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| PAIR_NAME | C | 임베딩 페어명 (`!EMBED PAIR`에서 정의) |

 
## 사용 예

 
```text
!CONTACT_ALGO, TYPE=SLAGRANGE
!EMBED, GRPID=1
  IP1
!STEP
CONTACT,1
```

 
## 비고

 
- 스텝에서 임베딩을 활성화하려면 !STEP의 데이터 행에서 "CONTACT"를 키워드로 하여 GRPID를 지정한다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/end.md | TITLE: !END -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/end.html -->

# !END

 제어 데이터의 끝을 나타냅니다.

 
## 매개변수

 없음


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/expansion_coeff.md | TITLE: !EXPANSION_COEFF -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/expansion_coeff.html -->

# !EXPANSION_COEFF

 선팽창계수 정의

 여기서 입력하는 선팽창계수는 각 온도에서의 선팽창계수 \(\alpha\) 값 자체가 아니라, 다음 식에 나타낸 기준 온도 \(T_\mathrm{ref}\)에서 각 온도 \(T\)까지의 구간 평균값이다.

 \[\begin{equation} \hat{\alpha}_T = \frac{1}{T-T_\mathrm{ref}} \int_{T_\mathrm{ref}}^{T^\prime} \alpha(T^\prime) dT^\prime \end{equation}\] 그림에서 파란 점선의 기울기는 온도 \(T\)에서의 선팽창계수 \(\alpha(T)\)이고, 녹색 실선의 기울기는 기준 온도 \(T_\mathrm{ref}\)에서 온도 \(T\)까지의 구간 평균값 \(\hat{\alpha}_T\)이다.

 ![](/keyword_reference/image/expansion_coeff.svg)

 
## 매개변수

 
```text
TYPE = 재질 유형
       ISOTROPIC(등방성:기본값) / ORTHOTROPIC(직교이방성)
DEPENDENCIES = 0(기본값) / 1
```

 ** 두 번째 행부터 **

 
### `TYPE=ISOTROPIC`인 경우

 
```text
(두 번째 행) expansion, Temperature
```

 
### `TYPE=ORTHOTROPIC`인 경우

 
```text
(두 번째 행) α11, α22, α33, Temperature
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| expansion | R | 선팽창계수 |
| α11, α22, α33 | R | 선팽창계수 |
| Temperature | R | 온도(DEPENDENCIES=1일 때 필요) |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/film.md | TITLE: !FILM -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/film.html -->

# !FILM

 경계면에 부여하는 열전달계수 정의

 
## 매개변수

 
```text
AMP1 = 열전달계수 이력 테이블명 (!AMPLITUDE 로 지정)
AMP2 = 주위 온도 이력 테이블명 (!AMPLITUDE 로 지정)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) ELEMENT_GRP_NAME, LOAD_type, Value, Sink
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| ELEMENT_GRP_NAME | C/I | 요소 그룹명 또는 요소 번호 |
| LOAD_type | C | 하중 유형 번호 |
| Value | R | 열전달계수 |
| Sink | R | 주위 온도 |

 
## 사용 예

 
```text
!FILM
  FSURF, F1, 1.0, 800.0
!FILM, AMP1=TFILM
  FSURF, F1, 1.0, 1.0
```

 
### 하중 매개변수

 
| 하중 유형 번호 | 작용면 | 매개변수 |
| --- | --- | --- |
| F1 | 제1면 | 열전달계수와 주위 온도 |
| F2 | 제2면 | 열전달계수와 주위 온도 |
| F3 | 제3면 | 열전달계수와 주위 온도 |
| F4 | 제4면 | 열전달계수와 주위 온도 |
| F5 | 제5면 | 열전달계수와 주위 온도 |
| F6 | 제6면 | 열전달계수와 주위 온도 |
| F0 | 셸 면 | 열전달계수와 주위 온도 |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/fixtemp.md | TITLE: !FIXTEMP -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/fixtemp.html -->

# !FIXTEMP

 지정 온도를 정의합니다.

 
## 매개변수

 
```text
AMP = 플럭스 이력 테이블명(!AMPLITUDE로 지정)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NODE_GRP_NAME, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_GRP_NAME | C/I | 절점 그룹명 또는 절점 번호 |
| Value | R | 온도(기본값: 0) |

 
## 사용 예

 
```text
!FIXTEMP
  ALL, 20.0
!FIXTEMP, AMP=FTEMP
  ALL, 1.0
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/fload.md | TITLE: !FLOAD -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/fload.html -->

# !FLOAD

 주파수 응답 해석용 가진력 정의

 
## 매개변수

 
```text
LOAD CASE = (실수부 지정: 1, 허수부 지정: 2)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NODE_ID, DOF_id, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_ID | I/C | 절점 번호 또는 절점 그룹명 또는 면 그룹명 |
| DOF_id | I | 자유도 번호 |
| Value | R | 하중값. 절점 번호 또는 절점 그룹명을 지정한 경우에는 집중 하중이며, 면 그룹명을 지정한 경우에는 단위 면적당 면 하중입니다. |

 
## 사용 예

 
```text
!FLOAD, LOAD CASE=2
  _PickedSet5, 2, 1.0
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/fluid.md | TITLE: !FLUID -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/fluid.html -->

# !FLUID

 유체 정의

 
## 매개변수

 
```text
TYPE = INCOMP_NEWTONIAN (기본값)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) mu
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| mu | R | 점도 |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/heat.md | TITLE: !HEAT -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/heat.html -->

# !HEAT

 계산에 관한 제어 데이터 정의

 
## 매개변수

 
```text
TIMEPOINTS = 시각 목록명 (!TIME_POINTS, NAME으로 지정)
BETA       = 시간 적분 스킴을 결정하는 매개변수
             정상 계산의 기본값: 1.0 (후진 차분)
             비정상 계산의 기본값: 0.5 (Crank-Nicolson법)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) DT, ETIME, DTMIN, DELTMX, ITMAX, ESP
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| DT | R | 초기 시간 증분<br>≦ 0: 정상 계산<br>> 0: 비정상 계산 |
| ETIME | R | 비정상 계산 시간(비정상 계산 시 필수) |
| DTMIN | R | 최소 시간 증분<br>≦ 0: 고정 시간 증분<br>> 0: 자동 시간 증분 |
| DELTMX | R | 허용 온도 변화 |
| ITMAX | I | 비선형 계산 최대 반복 횟수(기본값: 20) |
| EPS | R | 수렴 판정값 (기본값: 1.0e-6) |

 
## 사용 예

 
```text
!HEAT
  (데이터 없음) -------------- 정상 계산
!HEAT
  0.0 ----------------------- 정상 계산
!HEAT
  10.0, 3600.0 -------------- 고정 시간 증분 비정상 계산
!HEAT
  10.0, 3600.0, 1.0 --------- 자동 시간 증분 비정상 계산
!HEAT
  10.0, 3600.0, 1.0, 20.0 --- 자동 시간 증분 비정상 계산
```

 
## 비고

 
- 자동 시간 증분 비정상 계산의 경우에만 TIMEPOINTS 매개변수를 사용하여 임의 시각의 결과·시각화 출력을 할 수 있음


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/hyperelastic.md | TITLE: !HYPERELASTIC -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/hyperelastic.html -->

# !HYPERELASTIC

 초탄성 재료를 정의합니다.

 
## 매개변수

 
```text
TYPE = NEOHOOKE (기본값)
       MOONEY-RIVLIN
       ARRUDA-BOYCE
       MOONEY-RIVLIN-ANISO
       USER
```

 ** 두 번째 행부터 **

 
### `TYPE = NEOHOOKE`인 경우

 (두 번째 행) C10, D

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| C10 | R | 재료 상수 |
| D | R | 재료 상수 |

 
### `TYPE = MOONEY-RIVLIN`인 경우

 (두 번째 행) C10, C01, D

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| C10 | R | 재료 상수 |
| C01 | R | 재료 상수 |
| D | R | 재료 상수 |

 
### `TYPE = ARRUDA-BOYCE`인 경우

 
```text
(두 번째 행) mu, lambda_m, D
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| mu | R | 재료 상수 |
| lambda_m | R | 재료 상수 |
| D | R | 재료 상수 |

 
### `TYPE = MOONEY-RIVLIN-ANISO`인 경우

 (두 번째 행) C10, C01, D, C42, C43

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| C10 | R | 재료 상수 |
| C01 | R | 재료 상수 |
| D | R | 재료 상수 |
| C42 | R | 재료 상수 |
| C43 | R | 재료 상수 |

 
### `TYPE = USER`인 경우

 
```text
(두 번째 행-10번째 행) v1, v2, v3, v4, v5, v6, v7, v8, v9, v10
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/index.md | TITLE: 해석 제어 데이터 (cnt) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/index.html -->

# 해석 제어 데이터 (cnt)

 해석 제어 데이터 파일(`.cnt`)에서 사용하는 키워드 목록입니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/initial_condition.md | TITLE: !INITIAL_CONDITION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/initial_condition.html -->

# !INITIAL_CONDITION

 초기조건 지정.

 
## 매개변수

 
```text
TYPE = TEMPERATURE/VELOCITY/ACCELERATION
```

 
### `TYPE = TEMPERATURE`인 경우

 (두 번째 행) ng1, t1

 (세 번째 행) ng2, t2

 ...

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| ng1,ng2, ... | C/I | 절점 그룹명/절점 번호 |
| t1, t2, ... | R | 온도값 |

 
### `TYPE= VELOCITY/ACCELERATION`인 경우

 (두 번째 행) ng1, dof1, v1

 (세 번째 행) ng2, dof2, v2

 ...

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| ng1,ng2, ... | C/I | 절점 그룹명/절점 번호 |
| dof1, dof2, ... | I | 자유도 번호 |
| v1, v2, ... | R | 속도/가속도값 |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/material.md | TITLE: !MATERIAL -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/material.html -->

# !MATERIAL

 
## 재료 특성 정의

 재료 특성은 `!MATERIAL`과 그 뒤의 `!ELASTIC`, `!PLASTIC` 등의 항목을 함께 사용하여 정의합니다. `!MATERIAL`은 자신보다 앞에 놓인 `!ELASTIC`, `!PLASTIC` 등의 항목을 무시합니다.

 참고: 해석 제어 데이터에 `!MATERIAL`이 정의되어 있으면 메시 데이터의 `!MATERIAL` 정의는 무시됩니다. 해석 제어 데이터에 `!MATERIAL`이 정의되어 있지 않으면 메시 데이터의 `!MATERIAL` 정의가 사용됩니다.

 
## 매개변수

 
```text
NAME = 재료 이름
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/method.md | TITLE: !METHOD -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/method.html -->

# !METHOD

 면과의 단면을 지정할 때 해당 면의 설정 방법을 지정한다.

 
```text
!surface_num =2
!surface
!surface_style=3
!method=5
!coef=0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.35
!color_comp_name = temperature
```

 그림 7.4.17 `method` 설정 예

 이에 따라 평면 z=0.35와 z=-0.35.의 단면이 시각화된다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/orientation.md | TITLE: !ORIENTATION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/orientation.html -->

# !ORIENTATION

 로컬 좌표계를 정의한다.

 
## 매개변수

 
```text
NAME = 로컬 좌표계 이름
DEFINITION = COORDINATES (기본값)/NODES/LOCAL_NODES
```

 
### `DEFINITION = COORDINATES`인 경우

 (두 번째 행) a1, a2, a3, b1, b2, b3, c1, c2, c3

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| a1, a2, a3 | R | 점 a의 전역 좌표 |
| b1, b2, b3 | R | 점 b의 전역 좌표 |
| c1, c2, c3 | R | 점 c의 전역 좌표 |

 
### `DEFINITION= NODES`인 경우

 (두 번째 행) a, b, c

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| a, b, c | I | 절점 번호 |

 
### `DEFINITION = LOCAL_NODES`인 경우

 (두 번째 행) a, b, c

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| a, b, c | I | 요소 내의 로컬 절점 번호 |

 각 요소에 대해 그 요소 자체의 절점 a, b, c의 좌표로부터 로컬 좌표계를 정의한다. a, b, c와 좌표축의 관계는 그림과 같다. 요소마다 방향이 다른 로컬 좌표계를 가질 수 있으므로, 이 지정은 이방성 재료의 재료 주축을 요소 형상에 따라 부여할 때 사용한다. 로컬 좌표계는 변형 후의 절점 좌표에서 평가되므로 요소의 회전과 변형을 따라간다.

 
- a, b, c를 모두 지정해야 한다. 하나라도 생략하면 오류가 발생하고 해석이 정지한다.
- 로컬 절점 번호는 1에서 10 사이의 값이어야 하며, 대상 요소의 절점 수를 초과해서는 안 된다.

 ![방향](/keyword_reference/image/analysis05_02.png)

 
## 사용 예

 각 요소의 로컬 절점 5, 1, 8에서 로컬 좌표계를 정의하고 SECNUM=1인 섹션에 적용하는 경우

 
```text
!ORIENTATION, NAME=ORI_LOCAL, DEFINITION=LOCAL_NODES
  5, 1, 8
!SECTION, SECNUM=1, ORIENTATION=ORI_LOCAL
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/output_res.md | TITLE: !OUTPUT_RES -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/output_res.html -->

# !OUTPUT_RES

 해석 결과 데이터 파일에 출력할 물리량을 지정한다.

 `!WRITE, RESULT` 지정이 필요

 
## 매개변수

 없음

 ** 두 번째 행부터 **

 (두 번째 행부터) 변수명, ON/OFF

 다음 변수명을 지정할 수 있다.

 
| 변수명 | 물리량 |
| --- | --- |
| DISP | 변위(기본 출력) |
| ROT | 회전(761, 781 셸 요소에서만 유효) |
| REACTION | 절점 반력 |
| NSTRAIN | 절점 변형률 |
| NSTRESS | 절점 응력(기본 출력) |
| NMISES | 절점 Mises 응력(기본 출력) |
| ESTRAIN | 요소 변형률 |
| ESTRESS | 요소 응력 |
| EMISES | 요소 Mises 응력 |
| ORIENTATION | 요소의 재료 배향(!ORIENTATION으로 정의한 로컬 좌표계) |
| ISTRAIN | 적분점 변형률 |
| ISTRESS | 적분점 응력 |
| PL_ISTRAIN | 적분점 상당 소성 변형률 |
| TH_NSTRAIN | 절점 열변형률(미구현) |
| TH_ESTRAIN | 요소 열변형률(미구현) |
| TH_ISTRAIN | 적분점 열변형률(미구현) |
| VEL | 속도 |
| ACC | 가속도 |
| TEMP | 온도 |
| PRINC_NSTRESS | 절점 주응력(스칼라 값) |
| PRINCV_NSTRESS | 절점 주응력(벡터 값) |
| PRINC_NSTRAIN | 절점 주 변형률(스칼라 값) |
| PRINCV_NSTRAIN | 절점 주 변형률(벡터 값) |
| PRINC_ESTRESS | 요소 주응력(스칼라 값) |
| PRINCV_ESTRESS | 요소 주응력(벡터 값) |
| PRINC_ESTRAIN | 요소 주 변형률(스칼라 값) |
| PRINCV_ESTRAIN | 요소 주 변형률(벡터 값) |
| SHELL_LAYER | 적층 셸 요소의 적층별 출력 |
| SHELL_SURFACE | 셸 요소의 표면 정보 출력 |
| CONTACT_NFORCE | 접촉 법선력(벡터 값) |
| CONTACT_FRICTION | 접촉 마찰력(벡터 값) |
| CONTACT_RELVEL | 접촉 상대 미끄럼 속도(벡터 값) *슬레이브 절점만 |
| CONTACT_STATE | 접촉 상태(스칼라 값) *슬레이브 절점만. -1:분리, 0:미정의, 1:접촉(고착), 2:접촉(미끄럼) |
| CONTACT_NTRACTION | 단위 면적당 접촉 법선력(벡터 값) |
| CONTACT_FTRACTION | 단위 면적당 접촉 마찰력(벡터 값) |
| BEAM_NQM | 보 요소의 축력·전단력·굽힘 모멘트 |
| ELEMACT | 요소 활성화 상태(스칼라 값): 활성 요소는 1, 비활성 요소는 0 |

 
## 사용 예

 
```text
!OUTPUT_RES
  ESTRESS, OFF
  ISTRESS, ON
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/output_sstype.md | TITLE: !OUTPUT_SSTYPE -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/output_sstype.html -->

# !OUTPUT_SSTYPE

 결과 출력에 사용하는 응력-변형률 척도를 지정합니다. `TYPE=SOLUTION`은 `!SOLUTION`으로 지정한 해석 유형에 따라 결정되며, 선형 해석에서는 선형 응력과 변형률을 사용하고 비선형 해석에서는 진응력과 로그 변형률을 사용합니다. `TYPE=MATERIAL`은 재료 유형에 따라 결정되며, 선형 해석에서는 선형 응력과 변형률을 사용하고, Updated Lagrangian법으로 해석되는 재료에서는 진응력과 전체 변형률을 출력하며, Total Lagrangian법으로 해석되는 재료에서는 제2 Piola-Kirchhoff 응력과 Green 변형률을 출력합니다.

 
## 매개변수

 
```text
TYPE = SOLUTION(Default)/MATERIAL
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/output_vis.md | TITLE: !OUTPUT_VIS -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/output_vis.html -->

# !OUTPUT_VIS

 메모리 내 시각화기가 출력하는 시각화 데이터의 물리량을 지정합니다.

 `!WRITE, VISUAL`을 지정해야 합니다.

 
## 매개변수

 없음

 ** 두 번째 행부터 **

 (두 번째 행부터) 변수 이름, ON/OFF

 다음 변수 이름을 지정할 수 있습니다.

 
| 변수 이름 | 물리량 |
| --- | --- |
| DISP | 변위(기본 출력) |
| ROT | 회전(761 및 781 셸 요소에서만 유효) |
| REACTION | 절점 반력 |
| NSTRAIN | 절점 변형률 |
| NSTRESS | 절점 응력(기본 출력) |
| NMISES | 절점 Mises 응력(기본 출력) |
| TH_NSTRAIN | 절점 열변형률(미구현) |
| VEL | 속도 |
| ACC | 가속도 |
| TEMP | 온도 |
| PRINC_NSTRESS | 절점 주응력(스칼라 값) |
| PRINCV_NSTRESS | 절점 주응력(벡터 값) |
| PRINC_NSTRAIN | 절점 주변형률(스칼라 값) |
| PRINCV_NSTRAIN | 절점 주변형률(벡터 값) |
| SHELL_LAYER | 적층 셸 요소의 각 층별 출력 |
| SHELL_SURFACE | 셸 요소 표면 정보 출력 |
| CONTACT_NFORCE | 접촉 법선력(벡터 값) |
| CONTACT_FRICTION | 접촉 마찰력(벡터 값) |
| CONTACT_RELVEL | 접촉 상대 미끄럼 속도(벡터 값) *슬레이브 절점만 |
| CONTACT_STATE | 접촉 상태(스칼라 값) *슬레이브 절점만. -1: 분리, 0: 미정의, 1: 접촉(고착), 2: 접촉(미끄럼) |
| CONTACT_NTRACTION | 단위 면적당 접촉 법선력(벡터 값) |
| CONTACT_FTRACTION | 단위 면적당 접촉 마찰력(벡터 값) |
| BEAM_NQM | 보 요소의 축력, 전단력 및 굽힘 모멘트 |
| ELEMACT | 요소 활성화 상태(스칼라 값): 활성 요소는 1, 비활성 요소는 0 |

 
## 사용 예

 
```text
!OUTPUT_VIS
  NSTRAIN, ON
  NSTRESS, OFF
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/plastic.md | TITLE: !PLASTIC -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/plastic.html -->

# !PLASTIC

 소성 재료의 정의

 `!PLASTIC`을 정의할 때는 동일한 `!MATERIAL` 안에서 `!ELASTIC`도 함께 정의해야 한다.

 
## 매개변수

 
```text
YIELD        = MISES (기본값)、Mohr-Coulomb、DRUCKER-PRAGER、USER
HARDEN       = BILINEAR (기본값)、MULTILINEAR、SWIFT、RAMBERG-OSGOOD、KINEMATIC、COMBINED
DEPENDENCIES = 0 (기본값) / 1
INFINITESIMAL    지정한 경우 미소 변형 가정이 적용된다
```

 ** 두 번째 행부터 **

 
### `YIELD = MISES`인 경우 (기본값)

 ** `HARDEN = BILINEAR` (기본값)인 경우 **

 
```text
(두 번째 행) YIELD0, H
```

 ** `HARDEN = MULTILINEAR`인 경우 **

 
```text
(두 번째 행) YIELD, PSTRAIN, Temperature
(세 번째 행) YIELD, PSTRAIN, Temperature

...계속
```

 ** `HARDEN = SWIFT`인 경우 **

 
```text
(두 번째 행) ε0, K, n
```

 ** `HARDEN = RAMBERG-OSGOOD`인 경우 **

 
```text
(두 번째 행) ε0, D, n
```

 ** `HARDEN = KINEMATIC`인 경우 **

 
```text
(두 번째 행) YIELD0, C
```

 ** `HARDEN = COMBINED`인 경우 **

 
```text
(두 번째 행) YIELD0, H, C
```

 
### `YIELD =` Mohr-Coulomb 또는 Drucker-Prager인 경우

 ** `HARDEN = BILINEAR`(기본값)인 경우 **

 
```text
(두 번째 행) c, $\phi$, H, $\psi$
```

 ** `HARDEN = MULTILINEAR`인 경우 **

 ```text
(두 번째 행) $\phi$, $\psi$
(세 번째 행) c, PSTRAIN
(4번째 행) c, PSTRAIN
...계속
```

 `HARDEN =`의 다른 값은 무시되며 기본값(`BILINEAR`)이 된다.

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| YIELD0 | R | 초기 항복응력 |
| H | R | 경화계수 |
| PSTRAIN | R | 소성 변형률 |
| YIELD | R | 항복응력 |
| \(\varepsilon0, K, n\) | R | \(\overline{\sigma} = k\left( \varepsilon_{0} + \overline{\varepsilon} \right)^{n}\) |
| \(\varepsilon0, D, n\) | R | \(\varepsilon = \frac{\sigma}{E} + \varepsilon_{0}\left( \frac{\sigma}{D} \right)^{n}\) |
| \(\phi\) | R | 내부 마찰각 |
| \(\psi\) | R | 팽창각(기본값: \(\phi\)와 같은 값) |
| c | R | 점착력 |
| C | R | 선형 이동 경화계수 |
| Temperature | R | 온도(DEPENDENCIES=1일 때 필요) |
| v1, v2...v10 | R | 재료 상수 |

 ** YIELD= USER인 경우 **

 
```text
(두 번째 행부터) v1, v2, v3, v4, v5, v6, v7, v8, v9, v10
```

 
## 사용 예

 
```text
!PLASTIC, YIELD=MISES, HARDEN=MULTILINEAR, DEPENDENCIES=1
  276.0, 0.0, 20.
  296.0, 0.0018, 20.
  299.0, 0.0053, 20.
  303.0, 0.008, 20.
  338.0, 0.0173, 20.
  372.0, 0.0271, 20.
  400.0, 0.037, 20.
  419.0, 0.0471, 20.
  437.0, 0.0571, 20.
  450.0, 0.0669, 20.
  460.0, 0.0767, 20.
  469.0, 0.0867, 20.
  477.0, 0.0967, 20.
  276.0, 0.0, 100.
  276.0, 0.0018, 100.
  282.0, 0.0053, 100.
  295.0, 0.008, 100.
  330.0, 0.0173, 100.
  370.0, 0.0271, 100.
  392.0, 0.037, 100.
  410.0, 0.0471, 100.
  425.0, 0.0571, 100.
  445.0, 0.0669, 100.
  450.0, 0.0767, 100.
  460.0, 0.0867, 100.
  471.0, 0.0967, 100.
  128.0, 0.0, 400.
  208.0, 0.0018, 400.
  243.0, 0.0053, 400.
  259.0, 0.008, 400.
  309.0, 0.0173, 400.
  340.0, 0.0271, 400.
  366.0, 0.037, 400.
  382.0, 0.0471, 400.
  396.0, 0.0571, 400.
  409.0, 0.0669, 400.
  417.0, 0.0767, 400.
  423.0, 0.0867, 400.
  429.0, 0.0967, 400.
```

 지정된 온도 또는 소성 변형률에 관한 위 입력 데이터를 보간하여 가공경화계수를 계산하게 된다. 각 온도값에 대해 동일한 `PSTRAIN` 배열을 입력해야 한다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/radiate.md | TITLE: !RADIATE -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/radiate.html -->

# !RADIATE

 경계면에 적용하는 복사 계수를 정의한다.

 
## 매개변수

 
```text
AMP1 = 복사 계수 이력 테이블명(!AMPLITUDE 로 지정)
AMP2 = 주위 온도 이력 테이블명(!AMPLITUDE 로 지정)
```

 ** 두 번째 행부터 **

 
```text
(2번째 행) ELEMENT_GRP_NAME, LOAD_type, Value, Sink
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| ELEMENT_GRP_NAME | C/I | 요소 그룹명 또는 요소 번호 |
| LOAD_type | C | 하중 유형 번호 |
| Value | R | 복사 계수 |
| Sink | R | 주위 온도 |

 
## 사용 예

 
```text
!RADIATE
  RSURF, R1, 1.0E-9, 800.0
!RADIATE, AMP2=TRAD
  RSURF, R1, 1.0E-9, 1.0
```

 
### 하중 매개변수

 
| 하중 유형 번호 | 작용면 | 매개변수 |
| --- | --- | --- |
| R1 | 제1면 | 복사 계수와 주위 온도 |
| R2 | 제2면 | 복사 계수와 주위 온도 |
| R3 | 제3면 | 복사 계수와 주위 온도 |
| R4 | 제4면 | 복사 계수와 주위 온도 |
| R5 | 제5면 | 복사 계수와 주위 온도 |
| R6 | 제6면 | 복사 계수와 주위 온도 |
| R0 | 셸 면 | 복사 계수와 주위 온도 |

 
## 비고

 
- 복사 경계 조건은 슈테판-볼츠만 법칙에 따라 온도를 절대온도로 변환하여 열유속을 계산한다. 변환에 사용하는 절대 영도는 해석 제어 파일이 아니라 **메시 파일**의 [`!ZERO`](../msh/zero.ja.md)에서 지정한다.
- 섭씨 온도로 입력하는 경우 `!ZERO`를 생략할 수 있다(기본값 `-273.15`). 켈빈 온도로 입력하는 경우 `!ZERO`에 `0.0`을 지정한다.
- `Value`(복사 계수)도 입력 온도와 동일한 단위계로 지정해야 한다.

 
## 관련 항목

 
- [`!SRADIATE`](sradiate.ja.md)
- [`!ZERO`](../msh/zero.ja.md)
- [`!FILM`](film.ja.md)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/reftemp.md | TITLE: !REFTEMP -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/reftemp.html -->

# !REFTEMP

 열응력 해석에서의 기준 온도 정의

 
## 매개변수

 없음

 ** 두 번째 행부터 **

 
```text
(두 번째 행) Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| Value | R | 기준 온도(기본값: 0) |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/restart.md | TITLE: !RESTART -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/restart.html -->

# !RESTART

 리스타트 데이터의 입출력을 지정한다.

 
## 매개변수

 
```text
FREQUENCY = n        :출력할 스텝 간격(기본값: 0)
            n > 0    :n 스텝마다 출력
            n < 0    :먼저 리스타트 데이터 파일을 읽은 후 |n| 스텝마다 출력
```

 
## 비고

 
- 정적 해석 및 동적 해석의 시간 이력 해석에서는 서브스텝 도중의 기록과 스텝 끝의 기록을 구분하여 재개 위치를 복원한다.
- 동적 해석에서는 리스타트 데이터에 현재 스텝, 서브스텝, 누적 서브스텝, 현재 시각, 시간 증분, 속도·가속도 등의 동적 상태가 저장된다.
- 여러 `!STEP`을 사용하는 리스타트 재개에서는 완료된 `!STEP`을 재개용 cnt에서 삭제하고, 미완료 또는 미실행 `!STEP`부터 기술한다.

 
## 사용 예

 
```text
!RESTART, FREQUENCY=-2
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/section.md | TITLE: !SECTION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/section.html -->

# !SECTION

 단면의 국소 좌표계와 1차 육면체 요소(요소 유형 361), 1차 사면체 요소(341) 및 보 요소(611)의 정식화를 지정합니다. 요소 정식화 선택 지침은 [기능 참조: 요소 라이브러리](../../functions_reference/02_element.html)를 참조하십시오.

 
## 매개변수

 
```text
SECNUM = 메시 데이터에서의 !SECTION 입력 순서 번호
ORIENTATION　=　국소 좌표계명
FORM361 = FBAR (비선형 해석의 기본값)/IC (선형 해석의 기본값)/BBAR/FI/UP
FORM341 = FI (일반 1차 사면체 요소의 기본값)/SELECTIVE_ESNS(평활화 요소)
FORM611 = EULER (Bernoulli-Euler 보, 기본값)/TIMOSHENKO (Timoshenko 보)
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/sfilm.md | TITLE: !SFILM -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/sfilm.html -->

# !SFILM

 면 그룹에 의한 열전달 계수 정의

 
## 매개변수

 
```text
AMP1 = 열전달 계수 이력 테이블명 (!AMPLITUDE로 지정)
AMP2 = 주위 온도 이력 테이블명 (!AMPLITUDE로 지정)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) SURFACE_GRP_NAME, Value, Sink
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| SURFACE_GRP_NAME | C | 면 그룹명 |
| Value | R | 열전달률 |
| Sink | R | 주위 온도 |

 
## 사용 예

 
```text
!SFILM
  SFSURF, 1.0, 800.0
!SFILM, AMP1=TSFILM, AMP2=TFILM
  SFSURF, 1.0, 1.0
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/sflux.md | TITLE: !SFLUX -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/sflux.html -->

# !SFLUX

 면 그룹에 의한 분포 열유속 정의

 
## 매개변수

 
```text
AMP = 열유속 이력 테이블명 (!AMPLITUDE로 지정)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) SURFACE_GRP_NAME, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| SURFACE_GRP_NAME | C | 면 그룹명 |
| Value | R | 열유속 값 |

 
## 사용 예

 
```text
!SFLUX
  SURF, 1.0
!SFLUX, AMP=FLUX3
  SURF, 1.0
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/solution.md | TITLE: !SOLUTION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/solution.html -->

# !SOLUTION

 해석 종류를 지정한다.

 필수 제어 데이터이다.

 
## 매개변수

 
```text
TYPE=
     STATIC       : 선형 정적 해석
     NLSTATIC     : 비선형 정적 해석(TYPE=STATIC, NONLINEAR와 동일)
     HEAT         : 열전도 해석
     EIGEN        : 모드 해석
     DYNAMIC      : 동적 해석
     STATICEIGEN  : 비선형 정적 해석 -> 모드 해석
     ELEMCHECK    : 요소 형상 확인
NONLINEAR         : 비선형성을 고려(TYPE=STATIC/DYNAMIC만)
```

 
## 사용 예

 선형 정적 해석

 
```text
!SOLUTION, TYPE=STATIC
```

 비선형 정적 해석

 
```text
!SOLUTION, TYPE=STATIC, NONLINEAR
```

 
## 비고

 
- `TYPE=NLSTATIC`은 `TYPE=STATIC, NONLINEAR`과 동등한 기존 표기이다. 신규 입력에서는 비선형성의 유무를 `NONLINEAR` 매개변수로 명시하는 `TYPE=STATIC, NONLINEAR` 형식을 권장한다.
- 동적 해석의 시간 설정(시간 증분·스텝 시간 폭)은 `!STEP`에 통합되어 있다. `!DYNAMIC`의 `n_step`·`t_delta`는 `!STEP`을 생략한 경우 또는 하위 호환성을 위한 기본값이며, 신규 입력에서는 `!STEP`으로 지정한다. 자세한 내용은 [!STEP](step.html)과 [스텝 제어](../functions_reference/07_step.ko_KR.md)를 참조한다.
- `TYPE=ELEMCHECK`의 검사 내용과 출력은 [해석 유형: 요소 검사](../../functions_reference/01_analysis_type.html#요소-검사)를 참조한다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/solver.md | TITLE: !SOLVER -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/solver.html -->

# !SOLVER

 솔버 제어

 필수 제어 데이터.

 
## 매개변수

 
```text
METHOD =    해법 (CG、BiCGSTAB、GMRES、GPBiCG、CR、PipeCG、GroppCG、DIRECT、DIRECTmkl、MUMPS)
            DIRECT     : 접촉 해석 이외의 직접법(순차 처리만) (현재 사용 불가)
            DIRECTmkl  : Intel MKL에 의한 직접법
            MUMPS      : MUMPS에 의한 직접법
            직접법을 선택하면 데이터 행은 무시된다.
            반복법은 OpenMP 스레드 병렬화를 사용할 수 있다

PRECOND =   반복법의 전처리 방식 (1､2､3､5､10､11､12､22)
            1, 2       : (Block) SSOR (멀티컬러 오더링 포함)
            3          : (Block) Diagonal Scaling
            5          : 멀티그리드 전처리 패키지 ML에 의한 AMG
            10         : Block ILU(0)
            11         : Block ILU(1)
            12         : Block ILU(2)
            22         : HEC-MW 내장 SA-AMG(평활화 집계 AMG)에 의한 AMG
            10､11､12는 3자유도 문제에서만 사용 가능
            22는 LAPACK을 활성화한 빌드에서만 사용 가능
            OpenMP 스레드 병렬화 시에는 SSOR, Diagonal Scaling, ML, SA-AMG,
            또는 3자유도 문제에서의 Block ILU(10, 11, 12)를 권장한다

ITERLOG =   반복법 솔버 수렴 이력 출력 여부          (YES/NO)(기본값: NO)

TIMELOG =   솔버 계산 시간 출력 여부                (YES/NO/VERBOSE)(기본값: NO)

LOGLEVEL =  전처리기 진단 정보 출력 레벨                (정수)(기본값: 출력 없음)
            ITERLOG, TIMELOG와 독립적으로 지정
            1 이상이면 AMG 계열 전처리기(PRECOND=5､22)의 계층 구성 등의 정보를 출력
            2 이상이면 더 자세한 정보(메모리 사용량 등)를 출력

USEJAD =    벡터 머신용 오더링 여부                  (YES/NO)(기본값: NO)
            반복법을 사용할 때만 유효하다

SCALING =   행렬의 대각 성분을 1로 만드는 스케일링 여부 (YES/NO)(기본값: NO)
            3자유도 문제에서 반복법 사용 시에만 유효

DUMPTYPE =  행렬 덤프 형식(NONE、MM、CSR、BSR) (주로 디버깅용)
            NONE : 덤프하지 않음(기본값)
            MM   : Matrix Market 형식
            CSR  : Compressed Sparse Row(CSR) 형식
            BSR  : Blocked CSR 형식

DUMPEXIT =  행렬 덤프 직후 프로그램 종료            (YES/NO)(기본값: NO)

MPCMETHOD = 다점구속조건 처리 방식(1､3)
            1: 페널티법(직접법 사용 시 기본값)
            3: 명시적 자유도 제거법(반복법 사용 시 기본값)
            2에 해당하는 MPC-CG법은 삭제되었다. 2를 지정하면 경고 후 기본 방식으로 처리한다.

ESTCOND =   조건수 추정 빈도 (시험적)
            지정된 반복 간격마다 및 반복 종료 시 조건수 추정 수행
            0인 경우 추정하지 않음

RECOMPUTE_RESIDUAL = 잔차 벡터 재계산 빈도
            METHOD에 CG, BiCGSTAB, GPBiCG, PipeCG를 지정한 경우에만 유효
            GroppCG의 재계산 주기는 50으로 고정되며 이 설정은 사용되지 않는다
            반복법은 잔차 벡터를 점화식으로 갱신하므로 반올림 오차가 누적됨
            지정된 반복 간격마다 잔차 벡터를 {b}-[A]{x}에서 다시 계산
            0인 경우 각 해법의 기본값(CG: 50, BiCGSTAB: 100, GPBiCG: 20, PipeCG: 50)을 사용
            음수를 지정한 경우 주기적인 재계산을 수행하지 않음
            수렴 판정이 성립했을 때 확인을 위한 재계산은 이 설정과 관계없이 항상 수행
            주기를 지나치게 짧게 하면 행렬-벡터 곱이 늘어날 뿐만 아니라
            수렴성 자체가 악화되거나 수렴하지 않을 수 있음
            기본값에서 변경한 경우 수렴 이력과 최종 상대 잔차를 확인할 것

METHOD2 =   제2 해법 (BiCGSTAB、GMRES、GPBiCG) (시험적)
            METHOD에 CG, PipeCG 또는 GroppCG를 지정한 경우에만 유효하다
            METHOD로 지정한 해법이 발산하면 자동으로 전환하여 해를 구한다
            다른 매개변수와 데이터 행의 정보는 동일한 것을 사용

CONTACT_ELIM = 접촉 해석에서 자유도 제거를 수행할지 지정 (0,1)
               0: 반복법 사용 시에만 자유도 제거 수행(기본값)
               1: 항상(직접법 사용 시에도) 자유도 제거 수행
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NITER, iterPREmax, NREST, NCOLOR_IN, RECYCLEPRE
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NITER | I | 반복 횟수(기본값: 100) |
| iterPREmax | I | Additive Schwarz에 의한 전처리 반복 횟수(기본값: 1)<br>(권장값은 1 (병렬 계산에서는 2가 유효한 경우도 있음)) |
| NREST | I | Krylov 부분공간 수(기본값: 10)<br>(해법으로 GMRES를 선택한 경우에만 유효) |
| NCOLOR_IN | I | 멀티컬러 오더링의 목표 색 수(기본값: 10)<br>(OpenMP 스레드 수가 2 이상일 때만 유효) |
| RECYCLEPRE | I | 전처리 설정 정보 재사용 횟수(기본값: 3)<br>(비선형 해석에서만 유효) |

 
```text
(세 번째 행) RESID, SIGMA_DIAG, OMEGA
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| RESID | R | 수렴 판정값(기본값: 1.0e-8) |
| SIGMA_DIAG | R | 전처리 행렬 계산 시 대각 성분에 곱하는 배율(기본값: 1.0)<br>(ILU 전처리에서 0으로 나누기 또는 발산이 발생하는 경우 1.0보다 큰 값을 설정하면 해를 구할 수 있는 경우가 있다)<br>(ILU 전처리 및 대각 스케일링 전처리에서만 유효하다. SSOR 전처리는 OMEGA를 사용한다.) |
| OMEGA | R | SSOR 전처리의 완화 매개변수(기본값: 1.0)<br>(1.0은 기존 SSOR 전처리와 동일하다)<br>(유효 범위는 0보다 크고 2보다 작다. 범위를 벗어난 값과 0 또는 2와의 차이가 머신 엡실론보다 작은 값은 1.0으로 간주한다.) |

 
### PRECOND=5인 경우(생략 가능)

 PRECOND에 5 또는 22 이외의 값을 지정한 경우 4번째 행은 무시된다.

 
```text
(4번째 행) ML_CoarseSolver, ML_Smoother, ML_MGCycle, ML_MaxLevels, ML_CoarseningScheme, ML_NumSweep
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| ML_CoarseSolver | I | ML의 조대 격자용 솔버(1: 스무더로 대용, 2: KLU(순차 직접법), 3: MUMPS(병렬 직접법))(기본값: 1)<br>(권장값은 풀기 어려운 문제에서는 3 또는 2, 일반 문제에서는 1) |
| ML_Smoother | I | ML의 스무더(1: Chebyshev, 2: SymBlockGaussSeidel, 3: Jacobi)(기본값: 1)<br>(권장값은 1) |
| ML_MGCycle | I | ML의 멀티그리드 사이클(1: V-cycle, 2: W-cycle, 3: Full-V-cycle)(기본값: 1)<br>(권장값은 풀기 어려운 문제에서는 2, 일반 문제에서는 1) |
| ML_MaxLevels | I | ML의 최대 레벨 수(기본값: 10)<br>(특히 풀기 어려운 문제에서는 조대 격자 솔버를 직접법으로 하고 2(메모리가 부족한 경우 3), 일반 문제에서는 10을 권장) |
| ML_CoarseningScheme | I | ML의 코스닝 스킴(1: UncoupledMIS, 2: METIS, 4: Zoltan, 5: DD)(기본값: 1)<br>(권장값은 1 또는 5) |
| ML_NumSweep | I | ML 스무더의 스윕 횟수(Chebyshev인 경우 다항식 차수)(기본값: 2)<br>(권장값은 Chebyshev인 경우 2, SymBlockGaussSeidel인 경우 1) |

 
### PRECOND=22인 경우(생략 가능)

 PRECOND에 22 이외의 값을 지정한 경우 5번째 행은 무시된다. 4번째 행과 5번째 행은 모두 행 전체를 생략할 수 있으며, 각 행의 뒤쪽 항목만 생략할 수도 있다. 0을 지정한 항목에는 기본값이 사용된다.

 4번째 행의 1~7번째 항목은 PRECOND=5(ML)의 옵션 행과 같은 순서로 배치되어 있으므로 ML 옵션 행을 그대로 재사용할 수 있다. 8~10번째는 SA-AMG 전용 항목이다.

 
```text
(4번째 행) SA_CoarseSolver, SA_Smoother, SA_MGCycle, SA_MaxLevels, (미사용), SA_ChebyDegree, SA_CoarseSize, SA_MaxAggSize, SA_GalerkinLowMem, (미사용)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| SA_CoarseSolver | I | 조대 격자용 솔버(0: 자동, 1: 스무더로 대용, 2: 조밀 행렬 직접법, 3: MUMPS(병렬 직접법))(기본값: 0)<br>(자동 모드에서는 MUMPS를 활성화한 빌드에서 MUMPS를 사용하고, 그 외에는 조밀 행렬 직접법을 사용) |
| SA_Smoother | I | 스무더(0 또는 1: Chebyshev)(기본값: 0)<br>(Chebyshev만 지원한다. 다른 값을 지정하면 경고를 출력하고 Chebyshev를 사용) |
| SA_MGCycle | I | 멀티그리드 사이클(0: 기본값(W-cycle), 1: V-cycle, 2: W-cycle)(기본값: 0)<br>(Full-V-cycle은 지원하지 않음) |
| SA_MaxLevels | I | 최대 레벨 수(기본값: 20) |
| (미사용) | I | ML 코스닝 스킴에 대응하는 위치. SA-AMG는 항상 비결합 집계를 사용하므로 0이 아닌 값을 지정하면 경고를 출력하고 무시 |
| SA_ChebyDegree | I | Chebyshev 스무더의 다항식 차수(기본값: 2) |
| SA_CoarseSize | I | 조대 격자 절단 크기(기본값: 조대 격자용 솔버로 MUMPS를 사용하는 경우 50000, 그 외에는 100) |
| SA_MaxAggSize | I | 집계체 최대 크기(기본값: 96) |
| SA_GalerkinLowMem | I | 조대 격자 행렬 계산 방식(0: 일반(속도 우선), 양수: 메모리 절약)(기본값: 0)<br>(설정 시 메모리가 부족한 경우 양수를 지정) |
| (미사용) | I | 향후 확장을 위해 예약 |

 5번째 행은 SA-AMG 전용이며 ML에서는 읽지 않는다.

 
```text
(5번째 행) SA_Theta, SA_ChebyAlpha, SA_Safety, SA_TaperK, SA_AggOrder, SA_MinAggSize, SA_Verify, SA_DumpVTK
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| SA_Theta | R | 강연결 판정 임계값(기본값: 0.0)<br>(0.0에서는 행렬의 비영 구조를 그대로 그래프로 사용) |
| SA_ChebyAlpha | R | Chebyshev 스무더가 대상으로 하는 고유값 구간의 하한을 정하는 비율(기본값: 20.0) |
| SA_Safety | R | 추정 최대 고유값에 곱하는 안전 계수(기본값: 1.1) |
| SA_TaperK | R | 조대화 테이퍼 계수 K(기본값: 100)<br>(레벨 2 이후에는 약 K개 이상의 집계체가 남도록 조대화를 완만하게 한다. 음수를 지정하면 테이퍼링을 비활성화) |
| SA_AggOrder | R | 집계 순회 순서(0: 기본값(BFS 순서), 1: BFS 순서, 2: 절점 번호의 해시 순서(시험적), 3: 절점 차수 오름차순(시험적), 4: 절점 차수 내림차순(시험적), 음수: 절점 번호 순서)(기본값: 0) |
| SA_MinAggSize | R | 집계체 최소 크기(기본값: 3) |
| SA_Verify | R | 설정 시 자체 검사를 수행할지 여부(0: 수행하지 않음, 양수: 수행)(기본값: 0)(주로 디버깅용) |
| SA_DumpVTK | R | 최상위 레벨의 집계를 VTK 형식으로 출력할지 여부(0: 출력하지 않음, 양수: 출력)(기본값: 0)(주로 디버깅용) |

 
## 사용 예

 SSOR 전처리된 CG법을 사용하고 최대 반복 횟수를 10000, 수렴 판정값을 1.0e-8로 설정한다

 
```text
!SOLVER, METHOD=CG, PRECOND=1, ITERLOG=YES, TIMELOG=YES
  10000, 1
  1.0e-8, 1.0, 0.0
```

 SSOR 전처리된 GMRES법을 사용하고 Krylov 부분공간 수를 40, 멀티컬러 오더링의 목표 색 수를 100으로 설정한다

 
```text
!SOLVER, METHOD=GMRES, PRECOND=1, ITERLOG=YES, TIMELOG=YES
  10000, 1, 40, 100
  1.0e-8, 1.0, 0.0
```

 ILU(0) 전처리된 CG법을 사용하고 전처리 행렬 계산 시 대각 성분에 곱하는 배율을 1.1로 설정한다

 
```text
!SOLVER, METHOD=CG, PRECOND=10, ITERLOG=YES, TIMELOG=YES
  10000, 1
  1.0e-8, 1.1, 0.0
```

 CG법의 전처리를 ML에 의한 AMG법으로 한다

 
```text
!SOLVER, METHOD=CG, PRECOND=5, ITERLOG=YES, TIMELOG=YES
  10000, 1
  1.0e-8, 1.0, 0.0
```

 CG법의 전처리를 ML에 의한 AMG법으로 하고 조대 격자용 솔버를 MUMPS로 한다(풀기 어려운 문제용)

 
```text
!SOLVER, METHOD=CG, PRECOND=5, ITERLOG=YES, TIMELOG=YES
  10000, 1
  1.0e-8, 1.0, 0.0
  3
```

 CG법의 전처리를 ML에 의한 AMG법으로 하고 멀티그리드 사이클을 W-cycle로 한다(풀기 어려운 문제용)

 
```text
!SOLVER, METHOD=CG, PRECOND=5, ITERLOG=YES, TIMELOG=YES
  10000, 1
  1.0e-8, 1.0, 0.0
  1, 1, 2
```

 CG법의 전처리를 ML에 의한 AMG법으로 하고 조대 격자용 솔버를 MUMPS, 최대 레벨 수를 2로 한다(매우 풀기 어려운 문제용)

 
```text
!SOLVER, METHOD=CG, PRECOND=5, ITERLOG=YES, TIMELOG=YES
  10000, 1
  1.0e-8, 1.0, 0.0
  3, 1, 1, 2
```

 CG법의 전처리를 HEC-MW 내장 SA-AMG로 한다

 
```text
!SOLVER, METHOD=CG, PRECOND=22, ITERLOG=YES, TIMELOG=YES
  10000, 1
  1.0e-8, 1.0, 0.0
```

 CG법의 전처리를 SA-AMG로 하고 조대 격자용 솔버를 MUMPS로 한다

 
```text
!SOLVER, METHOD=CG, PRECOND=22, ITERLOG=YES, TIMELOG=YES
  10000, 1
  1.0e-8, 1.0, 0.0
  3
```

 마찰 접촉 해석에서 BiCGSTAB법의 전처리를 SA-AMG로 하고 AMG 전처리기의 계층 구성을 출력한다

 
```text
!SOLVER, METHOD=BiCGSTAB, PRECOND=22, ITERLOG=YES, TIMELOG=YES, LOGLEVEL=1
  10000, 1
  1.0e-8, 1.0, 0.0
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/spring.md | TITLE: !SPRING -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/spring.html -->

# !SPRING

 스프링 경계조건의 정의

 지정한 절점의 지정한 자유도를 지면에 대해 스프링으로 탄성 지지하는 경계조건이다. 메시 데이터 측에 요소를 정의할 필요는 없다. 두 절점을 스프링으로 연결하는 경우에는 커넥터 요소의 스프링 [`!SPRING_A`](spring_a.ja.md) / [`!SPRING_D`](spring_d.ja.md)를 사용한다(용도가 다름).

 
## 매개변수

 
```text
GRPID = 그룹 ID
AMP = 진폭 함수명 (!AMPLITUDE로 지정)
INCREMENTAL = NO(누적 변위에 저항·기본값) / YES(증분 변위에만 저항)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NODE_ID, DOF_id, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_ID | I/C | 절점 번호 또는 절점 그룹명 |
| DOF_id | I | 구속 자유도 |
| Value | R | 스프링 상수 |

 
## 사용 예

 
```text
!SPRING, GRPID=1
  1, 1, 0.5
```

 절점 1에 `INCREMENTAL=YES`, 절점 2에 `AMP=AMP1`을 지정하는 경우

 
```text
!SPRING, GRPID=1, INCREMENTAL=YES
  1, 1, 0.5
!SPRING, GRPID=1, AMP=AMP1
  2, 1, 0.5
```

 
## 비고

 
- `!SPRING`은 정적 해석 및 암시적 해법 동적 해석에서 사용할 수 있다.
- `GRPID`는 `!STEP`의 `LOAD` 행에서 참조하여 스프링 지지를 스텝별로 활성화하는 데 사용한다.
- `INCREMENTAL`과 `AMP`에 따른 스프링 지지의 동작은 [기능 레퍼런스: 경계 조건과 하중](../../functions_reference/04_boundary_load.html)을 참조한다.

 
## 관련 항목

 
- [`!SPRING_A`](spring_a.ja.md)
- [`!SPRING_D`](spring_d.ja.md)
- [`!BOUNDARY`](boundary.ja.md)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/spring_a.md | TITLE: !SPRING_A -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/spring_a.html -->

# !SPRING_A

 축방향 스프링의 정의.

 2절점 커넥터 요소(요소 유형 511)가 연결하는 두 절점 사이에, 두 절점을 잇는 축방향 스프링을 정의한다. 스프링의 방향은 모델의 변형을 따른다. 특정 자유도끼리 연결하는 스프링을 정의하려면 [`!SPRING_D`](spring_d.ja.md)를 사용한다.

 
## 메시 데이터 측 정의

 `!SPRING_A`는 재료로서 스프링 상수를 지정하는 키워드이며, 이것만으로는 스프링이 존재하지 않는다. 메시 데이터(msh) 측에 다음 두 가지를 정의해야 한다.

 
- 스프링이 연결하는 두 절점을 연결 정보로 갖는 요소 유형 511의 요소(→ [`!ELEMENT`](../msh/element.ja.md))
- 해당 요소 그룹에 대한 `TYPE=INTERFACE` 섹션(→ [`!SECTION`](../msh/section.ja.md)). `MATERIAL`에는 이 키워드로 스프링 상수를 정의한 재료명을 지정한다

 
## 매개변수

 없음

 ** 두 번째 행 **

 축방향 스프링 계수를 지정한다.

 
```text
(두 번째 행) SPRING_COEFF
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| SPRING_COEFF | R | 스프링 상수 |

 
## 사용 예

 절점 1과 절점 2를 연결하는 축방향 스프링(스프링 상수 1000)을 정의하는 경우

 msh

 
```text
!ELEMENT, TYPE=511, EGRP=ESPRING
1, 1, 2
!SECTION, TYPE=INTERFACE, EGRP=ESPRING, MATERIAL=SPRING1
1.0
```

 cnt

 
```text
!MATERIAL, NAME=SPRING1
!SPRING_A
1000
```

 
## 비고

 
- 축방향 스프링은 축방향의 한 성분에만 강성을 갖는다. 스프링 축에 직교하는 방향의 자유도는 구속하거나 다른 요소로 지지해야 한다.
- 두 절점의 좌표가 일치하는 경우(스프링 길이가 0인 경우)에는 축방향을 정할 수 없으므로, 세 방향에 동일하게 스프링 상수가 적용된다.
- 스프링 상수를 정의한 재료는 커넥터 요소 전용 재료로 취급된다. 동일한 재료명에 탄성 물성을 부여하여 솔리드 요소와 함께 사용할 수 없다.
- `!SPRING_A`는 두 절점을 연결하는 커넥터 요소의 스프링이며, 절점을 지면에 대해 탄성 지지하는 경계 조건 [`!SPRING`](spring.ja.md)과는 용도가 다르다.

 
## 관련 항목

 
- [`!SPRING_D`](spring_d.ja.md)
- [`!DASHPOT_A`](dashpot_a.ja.md)
- [`!SPRING`](spring.ja.md)
- [`!ELEMENT`](../msh/element.ja.md)
- [`!SECTION`](../msh/section.ja.md)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/spring_d.md | TITLE: !SPRING_D -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/spring_d.html -->

# !SPRING_D

 자유도 지정 스프링의 정의.

 2절점 커넥터 요소(요소 타입 511)가 연결하는 두 절점에 대해, 절점 1의 자유도 `DOF1`과 절점 2의 자유도 `DOF2`를 연결하는 스프링을 정의한다. 스프링의 방향은 자유도 번호로 정해지며 모델의 변형을 따라가지 않는다. 두 절점을 잇는 축 방향의 스프링을 정의하려면 [`!SPRING_A`](spring_a.ja.md)를 사용한다.

 여기서 절점 1과 절점 2는 `!ELEMENT`의 연결 정보에 지정한 순서에서 각각 첫 번째와 두 번째 절점이다.

 
## 메시 데이터 측 정의

 `!SPRING_D`는 스프링 상수를 재료로 지정하는 키워드이며, 이것만으로는 스프링이 존재하지 않는다. 메시 데이터(msh) 측에 다음 두 가지를 정의해야 한다.

 
- 스프링이 연결하는 두 절점을 연결 정보로 하는 요소 타입 511의 요소(→ [`!ELEMENT`](../msh/element.ja.md))
- 해당 요소 그룹에 대한 `TYPE=INTERFACE` 섹션(→ [`!SECTION`](../msh/section.ja.md)). `MATERIAL`에는 이 키워드로 스프링 상수를 정의한 재료명을 지정한다

 
## 매개변수

 없음

 ** 두 번째 행부터 **

 자유도 지정 스프링의 계수를 지정한다. 자유도 조합마다 한 줄을 작성하며, 여러 줄을 나열하여 여러 자유도 조합에 스프링을 정의할 수 있다.

 
```text
(두 번째 행) DOF1, DOF2, SPRING_COEFF
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| DOF1 | I | 절점 1의 자유도 번호(1,2,3) |
| DOF2 | I | 절점 2의 자유도 번호(1,2,3) |
| SPRING_COEFF | R | 스프링 상수 |

 
## 사용 예

 절점 1과 절점 2를 연결하고 x, y, z 각 방향에 서로 다른 스프링 상수를 설정하는 경우

 msh

 
```text
!ELEMENT, TYPE=511, EGRP=ESPRING
1, 1, 2
!SECTION, TYPE=INTERFACE, EGRP=ESPRING, MATERIAL=SPRING1
1.0
```

 cnt

 
```text
!MATERIAL, NAME=SPRING1
!SPRING_D
1, 1, 100.0
2, 2, 300.0
3, 3, 400.0
```

 
## 비고

 
- `DOF1`과 `DOF2`에 서로 다른 자유도 번호를 지정하면 절점 1의 한 방향 변위와 절점 2의 다른 방향 변위를 연결하는 스프링이 된다. 같은 방향의 스프링으로 하려면 `DOF1`과 `DOF2`에 같은 번호를 지정한다.
- 스프링을 정의하지 않은 자유도에는 이 스프링의 강성이 추가되지 않는다. 필요한 자유도 조합을 모두 기술하거나 경계 조건으로 구속한다.
- 스프링 상수를 정의한 재료는 커넥터 요소 전용 재료로 취급된다. 같은 재료명에 탄성 물성을 부여하여 솔리드 요소와 함께 사용할 수 없다.
- `!SPRING_D`는 두 절점을 연결하는 커넥터 요소의 스프링이며, 절점을 지면에 대해 탄성 지지하는 경계 조건 [`!SPRING`](spring.ja.md)과는 용도가 다르다.

 
## 관련 항목

 
- [`!SPRING_A`](spring_a.ja.md)
- [`!DASHPOT_D`](dashpot_d.ja.md)
- [`!SPRING`](spring.ja.md)
- [`!ELEMENT`](../msh/element.ja.md)
- [`!SECTION`](../msh/section.ja.md)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/sradiate.md | TITLE: !SRADIATE -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/sradiate.html -->

# !SRADIATE

 표면 그룹별 복사 계수 정의

 
## 매개변수

 
```text
AMP1 = 복사 계수 이력 테이블 이름(!AMPLITUDE 에서 지정)
AMP2 = 주위 온도 이력 테이블 이름(!AMPLITUDE 에서 지정)
```

 ** 두 번째 행부터 **

 
```text
(2번째 행) SURFACE_GRP_NAME, Value, Sink
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| SURFACE_GRP_NAME | C | 표면 그룹 이름 |
| Value | R | 복사 계수 |
| Sink | R | 주위 온도 |

 
## 사용 예

 
```text
!SRADIATE
  RSURF, 1.0E-9, 800.0
!SRADIATE, AMP2=TSRAD
  RSURF, 1.0E-9, 1.0
```

 
## 비고

 
- 복사 경계 조건은 슈테판-볼츠만 법칙에 따라 온도를 절대온도로 변환하여 열유속을 계산한다. 변환에 사용하는 절대 영도는 해석 제어 파일이 아니라 **메시 파일**의 [`!ZERO`](../msh/zero.ja.md)에서 지정한다.
- 섭씨 온도로 입력하는 경우 `!ZERO`를 생략할 수 있다(기본값 `-273.15`). 켈빈 온도로 입력하는 경우 `!ZERO`에 `0.0`을 지정한다.
- `Value`(복사 계수)도 입력 온도와 동일한 단위계로 지정해야 한다.

 
## 관련 항목

 
- [`!RADIATE`](radiate.ja.md)
- [`!ZERO`](../msh/zero.ja.md)
- [`!SFILM`](sfilm.ja.md)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/static.md | TITLE: !STATIC -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/static.html -->

# !STATIC

 정적 해석을 수행한다.(기본값, 생략 가능)

 
## 매개변수

 없음


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/step.md | TITLE: !STEP -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/step.html -->

# !STEP

 해석 스텝 설정

 비선형 정적 해석, 비선형 동적 해석에서는 필수

 위 이외의 해석에서 이 정의를 생략하면 모든 경계조건이 활성화되어 1스텝으로 계산

 재료 특성이 점탄성 및 크리프인 경우 `TYPE=VISCO`를 지정하고 계산 시간 조건을 설정

 참고로 `TYPE`(`STATIC`/`VISCO`)은 재료에 시간 증분을 전달할지 여부를 정하는 스텝 종류이며, `!SOLUTION`의 해석 종류(STATIC/DYNAMIC 등)와는 별개의 지정이다(아래 비고 참조)

 
## 매개변수

 
```text
TYPE     = STATIC(기본값) / VISCO (준정적 해석)
SUBSTEPS = 경계조건의 분할 스텝 수(기본값: 1)
CONVERG  = 수렴 판정 임계값(기본값: 1.0e-6)
MAXITER  = 비선형 해석에서의 최대 반복 계산 횟수(기본값: 50)
AMP      = 시간 함수명(!AMPLITUDE로 지정)
AMPLITUDE = RAMP / STEP (강제 변위를 서브스텝에 배분하는 방법)
            기본값: TYPE=STATIC에서는 RAMP, TYPE=VISCO에서는 STEP
INC_TYPE = FIXED(고정 증분·기본값) / AUTO(자동 증분)
MAXRES   = 최대 허용 잔차 설정(기본값: 1.0e+10)
TIMEPOINTS = 시각 목록명(!TIME_POINTS, NAME으로 지정)
AUTOINCPARAM = 자동 증분 매개변수 세트명(!AUTOINC_PARAM, NAME으로 지정)
MAXCONTITER = 접촉 해석에서의 최대 접촉 반복 횟수(기본값: 10)
```

 ** 두 번째 행 **

 INC_TYPE=FIXED인 경우(TYPE=STATIC인 경우 생략 가능)

 
```text
(두 번째 행) DTIME, ETIME
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| DTIME | R | 시간 증분(기본값: 1/SUBSTEPS) |
| ETIME | R | 스텝 시간 폭(기본값: 1) |

 INC_TYPE=AUTO인 경우(TYPE에 관계없이 지정)

 
```text
(두 번째 행) DTIME_INIT, ETIME, MINDT, MAXDT
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| DTIME_INIT | R | 초기 시간 증분 |
| ETIME | R | 스텝 시간 폭 |
| MINDT | R | 시간 증분 하한 |
| MAXDT | R | 시간 증분 상한 |

 ** 세 번째 행부터 **

 
```text
BOUNDARY, id                                  id=!BOUNDARY, !VELOCITY, !ACCELERATION에서 정의한 GRPID
  LOAD, id                                      id=!CLOAD, !DLOAD, !SPRING, !TEMPERATURE에서 정의한 GRPID
  CONTACT, id                                   id=!CONTACT에서 정의한 GRPID
```

 
## 사용 예

 고정 시간 증분 사용 예

 
```text
!STEP, CONVERG=1.E-8
  0.1, 1.0
  BOUNDARY, 1
  LOAD, 1
  CONTACT, 1
```

 자동 증분 조정을 활성화하고 초기 시간 증분 0.01, 스텝 시간 폭 2.5, 시간 증분 하한 1E-5, 시간 증분 상한 0.3, 최대 서브스텝 수를 200으로 설정한다

 
```text
!STEP, INC_TYPE=AUTO, SUBSTEPS=200
    0.01, 2.5, 1E-5, 0.3
```

 자동 증분 조정을 활성화하고 시각 목록 TP1을 계산·결과 출력 시각으로 지정한다

 
```text
!STEP, INC_TYPE=AUTO, TIMEPOINTS=TP1
    0.1, 2.0, 1E-3, 0.2
```

 
## 비고

 
- `TYPE`(`STATIC`/`VISCO`)은 해당 스텝에서 재료 구성 법칙에 시간 증분을 전달할지 여부를 결정하는 스위치이며, `!SOLUTION`에서 지정하는 해석 종류와는 독립된 별개의 축이다. `STATIC`에서는 재료가 시간 증분을 참조하지 않으며(레이트 비의존), `VISCO`에서는 점탄성·크리프가 실제 시간 증분을 참조한다. 이 구분은 정적 해석에서만 유효하며, 동적 해석(`!SOLUTION, TYPE=DYNAMIC`)에서는 `TYPE`이 참조되지 않고 시간 증분은 항상 실제 시간으로 취급된다
- 자동 증분 조정인 경우 SUBSTEPS는 최대 서브스텝 수로 취급된다
- `AMPLITUDE`에 의한 강제 변위 배분과 `RAMP`·`STEP`의 사용 구분은 [기능 참조: 스텝 제어](../../functions_reference/07_step.html#강제-변위의-서브스텝-배분)를 참조한다.
- 시각 목록명 TIMEPOINTS 및 자동 접촉 매개변수 세트명 AUTOINCPARAM의 지정은 INC_TYPE=AUTO일 때만 유효하다
- TIMEPOINTS를 지정하는 경우 지정 대상 !TIME_POINTS는 !STEP 카드보다 앞에 정의되어 있어야 한다.
- AUTOINCPARAM을 지정하는 경우 지정 대상 !AUTOINC_PARAM은 !STEP 카드보다 앞에 정의되어 있어야 한다. 또한 이 매개변수를 생략하면 기본 자동 증분 매개변수 세트가 사용된다
- 동적 해석에서 !STEP을 생략하면 !DYNAMIC의 `n_step`과 `t_delta`에 기반한 1스텝 해석으로 취급되며, 정의된 경계조건·하중이 활성화된다. 이는 하위 호환을 위한 처리이며, 신규 입력에서는 !STEP을 정의할 것을 권장한다
- 동적 해석의 고정 증분에서는 !STEP의 두 번째 행에 `DTIME, ETIME`을 지정하여 시간 증분과 스텝 시간 폭을 제어할 것을 권장한다. !DYNAMIC의 `n_step`과 `t_delta`는 !STEP 생략 시 또는 하위 호환을 위한 기본값으로 취급된다
- 동적 해석에서 INC_TYPE=AUTO를 지정한 경우 두 번째 행의 DTIME_INIT, ETIME, MINDT, MAXDT를 사용하여 시간 증분을 제어한다. 자동 증분·cutback은 주로 비선형 암시적 동적 해석에서 이용한다
- 동적 해석에서는 BOUNDARY 행에서 !BOUNDARY, !VELOCITY, !ACCELERATION을, LOAD 행에서 !CLOAD 및 암시적 방법의 !SPRING을 스텝별로 활성화할 수 있다


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/temperature.md | TITLE: !TEMPERATURE -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/temperature.html -->

# !TEMPERATURE

 열응력 해석에 사용하는 절점 온도 지정

 
## 매개변수

 
```text
READRESULT = 열전도 해석 결과의 스텝 수.
             지정하면 열전도 해석 결과 파일에서 온도를 순차적으로 읽고, 두 번째 행부터는 무시합니다.
SSTEP      = 열전도 해석 결과를 읽기 시작하는 첫 스텝 번호(기본값: 1)
INTERVAL   = 열전도 해석 결과를 읽는 스텝 간격(기본값: 1)
READTYPE   = STEP(기본값) / TIME
             TIME을 지정하면 열전도 해석과 열응력 해석의 해석 시간을 동기화합니다(INTERVAL 값은 무시되며,
             현재 해석 시간의 직전과 직후 열전도 해석 결과 사이를 선형 보간하여 온도를 결정합니다).
```

 열전도 해석이 자동 시간 증분을 사용하는 비정상 해석이고 !TIME_POINTS를 사용하여 지정 시간에 결과를 출력하는 경우, 출력 스텝 간격이 일정하지 않으므로 READTYPE=TIME을 지정해야 합니다.

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NODE_ID, Temp_Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_ID | I/C | 절점 ID 또는 절점 그룹 이름 |
| Temp_Value | R | 온도(기본값: 0) |

 
## 사용 예

 
```text
!TEMPERATURE
  1, 10.0
  2, 120.0
  3, 330.0
!TEMPERATURE
  ALL, 20.0
!TEMPERATURE, READRESULT=1, SSTEP=1
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/time_points.md | TITLE: !TIME_POINTS -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/time_points.html -->

# !TIME_POINTS

 
## 매개변수

 
| 매개변수명 | 유형 | 설명 |
| --- | --- | --- |
| NAME | C | 시각 목록명(필수) |
| TIME | C | STEP(스텝 시작 시각을 기준으로 한 시각 입력·기본값)/TOTAL(초기부터의 누적 시각으로 입력) |
| GENERATE | - | 시작 시각, 종료 시각, 시간 간격으로 시각점을 자동 생성 |

 ** 두 번째 행부터 **

 GENERATE를 사용하지 않는 경우

 
```text
(두 번째 행부터) TIME
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| TIME | R | 시각 |

 GENERATE를 사용하는 경우

 
```text
(두 번째 행) STIME, ETIME, INTERVAL
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| STIME | R | 시작 시각 |
| ETIME | R | 종료 시각 |
| INTERVAL | R | 시각점 간격 |

 
## 사용 예

 시각 1.5, 2.7, 3.9를 누적 시각으로서 GENERATE를 사용하지 않고 정의한다.

 
```text
!TIME_POINTS, TIME=TOTAL, TIME=,NAME=TP1
1.5
2.7
3.9
```

 시각 1.5, 2.7, 3.9를 스텝 시각으로서 GENERATE를 사용하여 정의한다.

 
```text
!TIME_POINTS, TIME=STEP, GENERATE, NAME=TP1
1.5, 3.9, 1.2
```

 
## 비고

 
- 시각점은 오름차순으로 입력해야 한다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/trs.md | TITLE: !TRS -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/trs.html -->

# !TRS

 열유변학적 단순성(Thermorheological Simplicity)에 의한 점탄성 재료의 온도 의존성 정의

 이 정의는 `!VISCOELASTIC` 뒤에 두어야 한다. 앞에 있으면 이 정의는 무시된다.

 
## 매개변수

 
```text
DEFINITION = WLF(기본값)／ARRHENIUS
```

 ** 두 번째 행부터 **

 (두 번째 행) θ0, C1, C2

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| θ0 | R | 기준 온도 |
| C1, C2 | R | 재료 상수(정식화는 [재료 데이터](../../functions_reference/03_material.html) 참조) |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/uload.md | TITLE: !ULOAD -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/uload.html -->

# !ULOAD

 사용자 정의 하중 입력

 
## 매개변수

 
```text
FILE = 파일명(필수)
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/user_material.md | TITLE: !USER_MATERIAL -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/user_material.html -->

# !USER_MATERIAL

 사용자 정의 재료 입력

 
## 매개변수

 
```text
NSTATUS = 재료 상태 변수의 개수를 지정합니다(기본값: 1)
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행-10번째 행) v1, v2, v3, v4, v5, v6, v7, v8, v9, v10
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/velocity.md | TITLE: !VELOCITY -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/velocity.html -->

# !VELOCITY

 속도 경계 조건의 정의

 
## 매개변수

 
```text
TYPE = INITIAL (초기 속도 경계 조건)
     = TRANSIT (!AMPLITUDE로 지정한 시간 이력 속도 경계 조건; 기본값)
GRPID = 그룹 ID
AMP  = 시간 함수명 (!AMPLITUDE로 지정)
       !AMPLITUDE로 시간 t와 계수 f(t)의 관계를 정의한다.
       아래 Value에 계수 f(t)를 곱한 값이 해당 시각의 구속값이 된다
       (지정하지 않은 경우: 시간과 계수의 관계는 f(t) = 1.0이 된다).
ROT_CENTER = 회전 속도 구속의 중심 절점 번호 또는 절점 그룹명.
             지정한 경우 해당 !VELOCITY는 회전 속도 구속으로 인식된다.
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) NODE_ID, DOF_idS, DOF_idE, Value
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_ID | I/C | 절점 번호 또는 절점 그룹명 |
| DOF_idS | I | 구속 자유도의 시작 번호 |
| DOF_idE | I | 구속 자유도의 종료 번호 |
| Value | R | 구속값(기본값: 0) |

 
## 사용 예

 
```text
!VELOCITY, TYPE=TRANSIT, AMP=AMP1
  1, 1, 1, 0.0
  ALL, 3, 3
  ※ 구속값은 0.0
!VELOCITY, TYPE=INITIAL
  1, 3, 3, 1.0
  2, 3, 3, 1.0
  3, 3, 3, 1.0
```

 
- 속도 경계 조건은 변위 경계 조건과 달리 여러 자유도를 한 번에 정의할 수 없으므로 DOF_idS와 DOF_idE는 같은 번호여야 한다.
- `TYPE`이 `INITIAL`인 경우 `AMP`는 무효가 된다.
- `GRPID`는 `!STEP`의 `BOUNDARY` 행에서 참조되며, 단계별로 속도 경계 조건을 활성화하는 데 사용된다.

 
## 회전 속도 구속 (ROT_CENTER)

 `ROT_CENTER`를 지정하면 `!BOUNDARY`의 `ROT_CENTER`(회전 변위 구속)와 마찬가지로 회전 중심 주위의 각속도를 속도 경계 조건으로 지정할 수 있다. 동적 해석에서만 유효하다.

 각 데이터 행의 `Value`는 각속도 벡터 \(\boldsymbol{\omega}=(\omega_x,\omega_y,\omega_z)\) [rad/s]의 성분(자유도 1~3)으로 지정한다. 각 절점에는 접선 속도 \(\boldsymbol{v}=\boldsymbol{\omega}\times(\boldsymbol{x}-\boldsymbol{x}_c)\)(\(\boldsymbol{x}_c\)는 회전 중심)가 적용된다. `AMP`를 함께 사용하면 \(\boldsymbol{\omega}(t)=\boldsymbol{\omega}_0 f(t)\)가 된다.

 
```text
!VELOCITY, TYPE=TRANSIT, AMP=AMP1, ROT_CENTER=CENTER
  RIM, 1, 1, 0.0        ! ωx
  RIM, 2, 2, 0.0        ! ωy
  RIM, 3, 3, 6.283185   ! ωz [rad/s]
```

 `ROT_CENTER`를 지정한 경우

 
- `ROT_CENTER`에 지정하는 절점 그룹은 정확히 하나의 절점으로 구성되어야 한다.
- 회전 변위 구속(`!BOUNDARY, ROT_CENTER`)이 Rodrigues 공식을 이용한 유한 회전인 반면, 회전 속도 구속은 각속도 벡터의 외적으로 구한 접선 속도이다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/version.md | TITLE: !VERSION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/version.html -->

# !VERSION

 솔버 버전 번호를 지정한다. 현재 버전 번호는 5

 
## 사용 예

 
```text
!VERSION
  5
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/viscoelastic.md | TITLE: !VISCOELASTIC -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/viscoelastic.html -->

# !VISCOELASTIC

 점탄성 재료를 정의합니다.

 `!VISCOELASTIC`을 정의할 때 동일한 `!MATERIAL` 안에 `!ELASTIC` 정의도 포함되어 있어야 합니다.

 
## 매개변수

 
```text
DEPENDENCIES = 종속 변수 수(미구현)
INFINITESIMAL    지정하면 미소변형 가정을 적용
```

 ** 두 번째 행부터 **

 
```text
(두 번째 행) mu_m, lambda_m
... 계속
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| mu_m | R | 전단 완화 계수 |
| lambda_m | R | 완화 시간 |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/ambient_coef.md | TITLE: !ambient_coef -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/ambient_coef.html -->

# !ambient_coef

 조명 모델의 계수 설정

 `ambient_coef`를 증가시키면 3차원 깊이 방향의 정보가 손실된다.

 ![조명 모델 매개변수 설정 예](/keyword_reference/image/analysis05_14.png)

 그림 7.4.13 조명 모델 매개변수 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/color_comp_name.md | TITLE: !color_comp_name -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/color_comp_name.html -->

# !color_comp_name

 물리량에서 컬러맵으로의 대응을 지정한다. 필요한 물리량과 그 자유도 번호에 이름을 붙인다. 이에 따라 결과 데이터 구조체 node_label(:) 및 nn_dof(:)에 이름이 들어간다.

 그런 다음 색상으로 매핑할 항목을 다음과 같이 정의할 수 있다.

 
## `!color_comp_name` (문자열, 기본값: 첫 번째 변수)

 
```text
예:

!color_comp_name = pressure
     정적 해석에서는  = DISPLACEMENT   : 결과 변위 데이터 지정
               = STRAIN         : 변형률 데이터 지정
               = STRESS         : 응력 데이터 지정
    열전달 해석에서는 = TEMPERATURE    : 결과 온도 데이터 지정
```

 
## `!color_comp` (정수, 기본값: 0)

 물리량의 식별 번호 (0 이상의 정수)

 
```text
예:

!color_comp = 2
```

 결과 데이터 종류의 식별 번호 지정과 성분명이지만, 미구현이다.

 
## `!color_subcomp` (정수, 기본값: 1)

 물리량이 벡터량처럼 자유도 수가 1 이상일 때 그 자유도 번호

 
```text
예:
!color_subcomp = 0

    !color_comp_name=DISPLACEMENT 지정의 경우
        1: X 성분,  2: Y 성분, 3: Z 성분

    !color_comp_name=STRAIN 지정의 경우
        1: εx,  2: εy,  3: εz
        4: εxy, 5: εyz, 6: εzx

    !color_comp_name=STRESS 지정의 경우
        1: σx, 2: σy, 3: σz
        4 : τxy 5 : τyz 6 : τzx

    !color/comp_name=TEMPERATURE 지정의 경우
        1: 온도
```

 구조 해석에서는 예를 들어

 
| 물리량 | 변위 | 변형률 | 응력 |
| --- | --- | --- | --- |
| 자유도 수 | 3 | 6 | 7 |

 ![color_comp, color_subcomp 및 color_comp_name 설정 예](/keyword_reference/image/analysis05_07.png)

 그림 7.4.5 `color_comp`, `color_subcomp` 및 `color_comp_name` 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/color_mapping_bar_on.md | TITLE: !color_mapping_bar_on -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/color_mapping_bar_on.html -->

# !color_mapping_bar_on

 
```text
!color_mapping_bar_on : 컬러 매핑 바를 표시할지 여부를 지정합니다.
                         0: 끔 1: 켬 (기본값: 0)

!scale_marking_on     : 컬러 매핑 바에 눈금을 표시할지 여부를 지정합니다.
                         0: 끔 1: 켬 (기본값: 0)

!num_of_scales        : 눈금 수를 지정합니다.
                         (기본값: 3)
```

 ![color_mapping_bar 표시 예](/keyword_reference/image/analysis05_16.png)

 그림 7.4.14 `color_mapping_bar` 표시 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/data_comp_name.md | TITLE: !data_comp_name -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/data_comp_name.html -->

# !data_comp_name

 `surface_style=2`일 때 시각화할 등가면의 물리량을 지정한다.

 ![data_comp, data_subcomp 및 data_comp_name 설정 예](/keyword_reference/image/analysis05_18.png)

 그림 7.4.16 `data_comp`, `data_subcomp` 및 `data_comp_name` 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/deform_scale.md | TITLE: !deform_scale -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/deform_scale.html -->

# !deform_scale

 변형을 표시할 때의 변위 스케일을 지정한다.

 기본값: 자동

 `standard_scale` = 0.1 * sqrt(`x_range`2 + `y_range`2 + `z_range`2) / `max_deform`

 ![display_styles 설정 예](/keyword_reference/image/analysis05_09.png)

 그림 7.4.7 `display_styles` 설정 예

 ![deform_scale 설정 예](/keyword_reference/image/analysis05_10.png)

 그림 7.4.8 `deform_scale` 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/display_method.md | TITLE: !display_method -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/display_method.html -->

# !display_method

 표시 방법 (기본값: 1)

 
1. 색상 코드 표시
2. 경계선 표시
3. 색상 코드 및 경계선 표시
4. 지정한 단일 색상으로 표시
5. 색 구분에 의한 등치선 표시

 ![display_method 설정 예](/keyword_reference/image/analysis05_06.png)

 그림 7.4.4 display_method 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/font_size.md | TITLE: !font_size -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/font_size.html -->

# !font_size

 배경색과 문자 폰트를 지정한다.

 ![background 및 font 설정 예](/keyword_reference/image/analysis05_17.png)

 그림 7.4.15 `background` 및 `font` 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/index.md | TITLE: 시각화용 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/index.html -->

# 시각화용

 시각화 제어(비주얼라이저)에서 사용하는 키워드 목록입니다. 이 키워드는 솔버의 setup에서는 읽히지 않으며, 비주얼라이저 (`hecmw_vis_read_control`)에서 처리됩니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/initial_style.md | TITLE: !initial_style -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/initial_style.html -->

# !initial_style

 초기 형상과 변형 후 형상의 표시 스타일을 지정한다.

 
1. 없음
2. 실선 메시(지정이 없으면 파란색으로 표시)
3. 회색 채우기
4. 셰이딩(물리 속성을 색상에 대응시킴)
5. 점선 메시(지정이 없으면 파란색으로 표시)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/isoline_number.md | TITLE: !isoline_number -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/isoline_number.html -->

# !isoline_number

 `display_method=2`,`3` 또는 `5`일 때

 ![isoline_number와 isoline_color의 설정 예](/keyword_reference/image/analysis05_08.png)

 그림 7.4.6 `isoline_number`와 `isoline_color`의 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/output_type.md | TITLE: !output_type -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/output_type.html -->

# !output_type

 출력 파일의 형식을 지정한다. (기본값: AVS)

 
```text
AVS                   : AVS용 UCD 데이터(물체 표면에서만)
BMP                   : 이미지 데이터(BMP 형식)
VTK                   : ParaView용 VTK 데이터
BIN_TVK               : ParaView용 VTK 데이터(바이너리 형식)
COMPLETE_AVS          : AVS용 UCD 데이터
COMPLETE_REORDER_AVS  : AVS용 UCD 데이터에서 절점·요소 번호를 재정렬한다
SEPARATE_COMPLETE_AVS : 분할 영역별 AVS용 UCD 데이터
COMPLETE_MICROAVS     : AVS용 UCD 데이터에서 물리량을 스칼라로 출력한다
BIN_COMPLETE_AVS      : COMPLETE_AVS를 바이너리 형식으로 출력한다
FSTR_FEMAP_NEUTRAL    : FEMAP용 뉴트럴 파일
```

 ![output_type의 예](/keyword_reference/image/analysis05_11.png)

 그림 7.4.9 output_type의 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/surface_num.md | TITLE: !surface_num -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/surface_num.html -->

# !surface_num

 
## `!surface_num` (P1-1)

 하나의 서피스 렌더링 내 서피스 수

 예: 그림 7.4.1에는 4개의 서피스가 있으며, 2개는 pressure=1000.0 및 pressure=-1000.0의 등가면이고, 2개는 z= -1.0 및 z= 1.0의 평면 절단면이다.

 ![surface_num 설정 예](/keyword_reference/image/analysis05_03.png) 그림 7.4.1 `surface_num` 설정 예

 
## `!surface` (P1-2)

 서피스의 내용을 설정한다.

 예: 그림 7.4.2에는 4개의 서피스가 있으며 내용은 다음과 같다.

 ![surface 설정 예](/keyword_reference/image/analysis05_04.png)

 그림 7.4.2 `surface` 설정 예

 
```text
!surface_num = 2
!SURFACE
!surface_style = 2
!data_comp_name = press
!iso_value = 1000.0
!display_method = 4
!specified_color = 0.45
!output_type = BMP
!SURFACE
!surface_style = 2
!data_comp_name = press
!iso_value = -1000.0
!display_method = 4
!specified_color = 0.67
```

 
## `!surface_style` (P1-3)

 서피스 스타일을 지정한다.

 
1. 경계면
2. 등가면
3. 임의의 2차 곡면
  coef[1]x2 + coef[2]y2 + coef[3]z2 + coef[4]xy + coef[5]xz
  + coef[6]yz + coef[7]x + coef[8]y + coef[9]z + coef[10]=0

 ![surface/style 설정 예](/keyword_reference/image/analysis05_05.png)

 그림 7.4.3 surface/style 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/viewpoint.md | TITLE: !viewpoint -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/viewpoint.html -->

# !viewpoint

 
## `viewpoint`: 시점의 위치를 좌표로 지정한다.

 
```text
기본값
x = (xmin + xmax)/2.0,
y = ymin + 1.5 *(ymax – ymin),
z = zmin + 1.5 *(zmax – zmin)
```

 
## `look_at_point`: 시선의 위치를 지정한다.

 
```text
(기본값: 데이터의 중심)

up_direction: Viewpoint, look_at_point 및 up_direction으로 뷰 프레임을 지정한다.
              default : 0.0 0.0 1.0

View coordinate frame:
              원점    : look_at_point
              z축     : viewpoint - look_at_point
              x축     : up × z axis
              y축     : z axis × x axis
```

 ![뷰 프레임 결정 방법](/keyword_reference/image/analysis05_15.png)

 그림 7.4.11 뷰 프레임 결정 방법

 ![!viewpoint, !look_at_point 및 up_direction 설정 예](/keyword_reference/image/analysis05_13.png)

 그림 7.4.12 `!viewpoint`, `!look_at_point` 및 `up_direction` 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/visual.md | TITLE: !VISUAL -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/visual.html -->

# !VISUAL

 시각화 방법과 시각화 데이터의 출력 조건을 지정합니다. `!WRITE,VISUAL`로 시각화 출력을 활성화한 다음 이 카드로 형식, 표면 및 기타 설정을 지정합니다.

 
## 매개변수

 
```text
METHOD = PSR             : 표면 렌더링
  visual_start_step      : 시각화 처리를 시작할 시간 스텝 번호 지정  (기본값: 1)
  visual_end_step        : 시각화 처리를 종료할 시간 스텝 번호 지정 (기본값: 전체)
  visual_interval_step   : 시각화 처리의 시간 스텝 간격 지정    (기본값: 1)
```

 
## 표면 지정(필수)

 `METHOD=PSR`에서는 시각화할 표면 수 `surface_num`과 각 표면의 정의 `!surface N`을 지정해야 합니다. 최소 구성에서는 전체 외부 표면을 나타내는 표면 하나만 지정하면 충분합니다.

 출력 형식은 `!output_type`으로 선택합니다. `VTK`를 지정하면 ParaView 등의 소프트웨어에서 읽을 수 있는 VTK 형식으로 데이터가 기록됩니다.

 
## 최소 데이터셋

 다음은 전체 외부 표면을 VTK 형식으로 출력하는 최소 예입니다. `!WRITE,VISUAL`과 함께 사용하십시오.

 
```text
!WRITE,VISUAL
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
```

 `surface_num` 및 `!surface` 지정은 생략할 수 없습니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/visualization/x_resolution.md | TITLE: !x_resolution -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/x_resolution.html -->

# !x_resolution

 `output_type=BMP`일 때 해상도를 지정합니다.

 ![x_resolution 및 y_resolution 설정 예](/keyword_reference/image/analysis05_12.png)

 그림 7.4.10 `x_resolution` 및 `y_resolution` 설정 예


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/weld_line.md | TITLE: !WELD_LINE -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/weld_line.html -->

# !WELD_LINE

 용접선(직선) 정의

 
## 매개변수

 없음

 ** 두 번째 행 **

 
```text
(두 번째 행) I, U, Coef, V
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| I | R | 전류 |
| U | R | 전압 |
| Coef | R | 입열 효율 |
| V | R | 용접 토치의 이동 속도 |

 ** 세 번째 행 **

 
```text
(세 번째 행) EGROUP, XYZ, C1, C2, H, tstart
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| EGROUP | C | 입열하는 요소 그룹명 |
| XYZ | I | 용접 토치의 이동 방향(자유도 번호) |
| C1 | R | 용접 토치의 시작점 좌표 |
| C2 | R | 용접 토치의 끝점 좌표 |
| H | R | 용접 열원의 폭, 이 폭 범위 내의 요소는 입열을 받음 |
| tstart | R | 용접 시작 시각 |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/write.md | TITLE: !WRITE -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/write.html -->

# !WRITE

 다양한 파일(결과 파일, 시각화 파일, 로그 및 FEMAP 중립 파일)로의 출력을 활성화하고 출력 빈도를 지정합니다. 구현상 이는 하나의 카드이며, `RESULT` / `VISUAL` / `LOG` / `FEMAP`(구 `UTABLE`) 매개변수로 출력 대상을 선택합니다(`fistr1/src/common/fstr_ctrl_common.f90`의 `fstr_ctrl_get_WRITE`, `fistr1/src/lib/m_out.f90`의 `fstr_ctrl_get_output`).

 시각화 형식 및 표면 지정과 같은 시각화 전용 제어는 별도의 [`!VISUAL`](visualization/visual.ja.md) 카드에서 수행합니다.

 
## 매개변수

 
| 매개변수 | 역할 |
| --- | --- |
| `RESULT` | 해석 결과 데이터 파일 출력을 활성화합니다. |
| `VISUAL` | 메모리 내 시각화기를 통한 시각화 데이터 출력을 활성화합니다. |
| `LOG` | 로그 파일 출력을 활성화합니다. |
| `FEMAP` | FEMAP 중립 형식 출력을 활성화합니다. |
| `FREQUENCY = n` | 출력 스텝 간격을 지정합니다. 기본값은 1입니다. |

 `RESULT` / `VISUAL` / `LOG` / `FEMAP`은 동일한 카드에 함께 지정할 수 있으며 각각 독립적으로 활성화됩니다.

 
## 사용 예

 
### !WRITE,RESULT — 결과 파일

 해석 결과 데이터 파일 출력을 활성화합니다. 출력할 물리량은 `!OUTPUT_RES`로 선택합니다.

 
```text
!WRITE, RESULT, FREQUENCY=2
```

 
### !WRITE,VISUAL — 시각화 파일

 시각화 데이터 출력을 활성화합니다. 최소 데이터셋 예를 아래에 나타냅니다. `!VISUAL` 카드에서 `surface_num` 및 `!surface` 지정은 생략할 수 없습니다.

 
```text
!WRITE, VISUAL, FREQUENCY=2
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
```

 
### !WRITE,LOG — 로그

 로그 파일 출력의 스텝 간격을 지정합니다.

 
```text
!WRITE, LOG, FREQUENCY=2
```

 
### 조합 예

 하나의 `!WRITE` 카드로 여러 출력 대상을 동시에 활성화할 수 있습니다.

 
```text
!WRITE, RESULT, VISUAL, FREQUENCY=1
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/write_result.md | TITLE: !WRITE,RESULT -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/write_result.html -->

# !WRITE,RESULT

 해석 결과 데이터 파일 출력을 활성화합니다. 구현상 하나의 `!WRITE` 카드에 대한 `RESULT` 매개변수이며, `!WRITE,VISUAL` / `!WRITE,FEMAP`과 나란히 출력 대상 지정으로 동작합니다(`fistr1/src/common/fstr_ctrl_common.f90`의 `fstr_ctrl_get_WRITE`, `fistr1/src/lib/m_out.f90`의 `fstr_ctrl_get_output`).

 
## 매개변수

 
```text
FREQUENCY = 출력 스텝 간격(기본값: 1)
```

 
## 사용 예

 
```text
!WRITE, RESULT, FREQUENCY=2
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/cnt/write_visual.md | TITLE: !WRITE,VISUAL -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/cnt/write_visual.html -->

# !WRITE,VISUAL

 메모리 전달 비주얼라이저에 의한 시각화 데이터 출력을 활성화한다. 구현상 단일 `!WRITE` 카드에 대한 `VISUAL` 매개변수이며, `!WRITE,RESULT` / `!WRITE,FEMAP`과 나란히 출력 대상 지정으로 동작한다(`fistr1/src/common/fstr_ctrl_common.f90`의 `fstr_ctrl_get_WRITE`, `fistr1/src/lib/m_out.f90`의 `fstr_ctrl_get_output`).

 시각화 형식 및 서피스 지정은 별도 카드 [`!VISUAL`](visualization/visual.ja.md)에서 지정한다.

 
## 매개변수

 
```text
FREQUENCY = 출력할 스텝 간격 (기본값: 1)
```

 
## 사용 예

 시각화를 활성화하기 위한 최소 데이터 세트 예를 아래에 나타낸다. `!VISUAL` 카드의 `surface_num`과 `!surface` 지정은 생략할 수 없다.

 
```text
!WRITE, VISUAL, FREQUENCY=2
!VISUAL,method=PSR
!surface_num=1
!surface 1
!output_type=VTK
```


---

