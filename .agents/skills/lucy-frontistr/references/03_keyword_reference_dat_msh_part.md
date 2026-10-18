# FrontISTR 키워드 레퍼런스 — 입력규칙, 전체제어(.dat), 메시(.msh), 영역분할(.part)

> 크롤링 출처: `https://docs.frontistr.com/ko/` (수록 페이지 수: 24개)

## 수록 문서 목차

- `keyword_reference/dat/control.md` — !CONTROL
- `keyword_reference/dat/index.md` — 전체 제어 데이터 (dat)
- `keyword_reference/dat/mesh.md` — !MESH
- `keyword_reference/dat/restart.md` — !RESTART
- `keyword_reference/dat/result.md` — !RESULT
- `keyword_reference/dat/subdir.md` — !SUBDIR
- `keyword_reference/input_rules.md` — 입력 규칙
- `keyword_reference/msh/amplitude.md` — !AMPLITUDE
- `keyword_reference/msh/contactpair.md` — !CONTACTPAIR
- `keyword_reference/msh/egroup.md` — !EGROUP
- `keyword_reference/msh/element.md` — !ELEMENT
- `keyword_reference/msh/embedpair.md` — !EMBEDPAIR
- `keyword_reference/msh/end.md` — !END
- `keyword_reference/msh/equation.md` — !EQUATION
- `keyword_reference/msh/header.md` — !HEADER
- `keyword_reference/msh/index.md` — 메시 데이터 (msh)
- `keyword_reference/msh/material.md` — !MATERIAL
- `keyword_reference/msh/ngroup.md` — !NGROUP
- `keyword_reference/msh/node.md` — !NODE
- `keyword_reference/msh/section.md` — !SECTION
- `keyword_reference/msh/sgroup.md` — !SGROUP
- `keyword_reference/msh/zero.md` — !ZERO
- `keyword_reference/part/index.md` — 영역 분할 (part)
- `keyword_reference/part/partition.md` — !PARTITION

---


<!-- ========================================== -->
<!-- FILE: keyword_reference/dat/control.md | TITLE: !CONTROL -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/dat/control.html -->

# !CONTROL

 해석 제어 데이터 파일을 지정한다.

 
```text
!CONTROL, NAME=<name>
file

#사용 예
!CONTROL, NAME=fstrCNT
myctrl.cnt
```

 
| 매개변수 |  |
| --- | --- |
| `NAME` | 식별자(필수) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NAME` | `fstrCNT` | 해석 제어 데이터 |

 
| 변수명 | 내용 |
| --- | --- |
| file | 해석 제어 데이터 파일명. 상대 경로와 절대 경로를 모두 지정할 수 있다. 상대 경로인 경우 현재 디렉터리부터의 경로가 된다. |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/dat/index.md | TITLE: 전체 제어 데이터 (dat) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/dat/index.html -->

# 전체 제어 데이터 (dat)

 전체 제어 데이터 파일(`.dat`)에서 사용하는 키워드 목록입니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/dat/mesh.md | TITLE: !MESH -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/dat/mesh.html -->

# !MESH

 메시 데이터 파일을 지정한다.

 
```text
!MESH, NAME=<name>, TYPE=<type> [,선택적 매개변수]
fileheader

#사용 예
!MESH, NAME=fstrMSH, TYPE=HECMW-DIST, REFINE=1
Mesh.in
```

 
| 매개변수 |  |
| --- | --- |
| `NAME` | 식별자(필수) |
| `TYPE` | 메시 유형(필수) |
| `IO` | 입출력 지정(생략 가능) |
| `REFINE` | 메시 세분화 지정(선택 사항) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NAME` | `fstrMSH` | 솔버 입력 데이터 |
|  | `part_in` | 분할기 입력 데이터 |
|  | `part_out` | 분할기 출력 데이터 |
| `TYPE` | `HECMW-DIST` | HEC-MW 분산 메시 데이터 |
|  | `HECMW-ENTIRE` | HEC-MW 단일 영역 메시 데이터 |
|  | `INP` | inp 형식 메시 데이터(`.inp`) |
|  | `NASTRAN` | Nastran 형식 메시 데이터 |
| `IO` | `IN` | 입력용(기본값) |
|  | `OUT` | 출력용 |
| `REFINE` | <integer> | 메시 세분화 횟수 |

 
| 변수명 | 내용 |
| --- | --- |
| fileheader | 메시 데이터 파일명의 헤더. 상대 경로와 절대 경로 모두 지정 가능. 상대 경로인 경우 현재 디렉터리로부터의 경로가 된다 |

 주의

 `IO` 매개변수의 유무 및 매개변수 값은 동작에 아무런 영향을 주지 않는다.

 `TYPE`이 `HECMW-DIST`인 경우 데이터 행에 지정하는 fileheader는 파일명 끝의 ".<rank>"를 제외한 것이다.

 `TYPE`이 `INP`인 경우 inp 형식의 입력 파일(`.inp`)을 직접 읽는다. 지원되는 키워드는 `*NODE`, `*ELEMENT`, `*NSET`, `*ELSET`, `*MATERIAL`, `*ELASTIC`, `*DENSITY`, `*SOLID SECTION` 등이다. 해석 제어 데이터(`.cnt`)에서 `!MATERIAL`을 정의한 경우 메시 내부의 재료 정의는 덮어쓴다.

 `TYPE`이 `NASTRAN`인 경우 Nastran 형식의 bulk 데이터 파일을 직접 읽는다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/dat/restart.md | TITLE: !RESTART -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/dat/restart.html -->

# !RESTART

 재시작 데이터 파일을 지정합니다.

 
```text
!RESTART, NAME=<name>, IO=<io>
fileheader

#Example
!RESTART, NAME=restart-in, IO=IN
restart.in
```

 
| 매개변수 | 설명 |
| --- | --- |
| `NAME` | 식별자(필수) |
| `IO` | 입출력 지정(필수) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NAME` | <name> | 사용자 식별자 |
| `IO` | IN | 입력 |
|  | OUT | 출력 |
|  | INOUT | 입출력 |

 
| 변수 이름 | 설명 |
| --- | --- |
| fileheader | 재시작 데이터 파일 이름의 헤더<br>(상대 경로와 절대 경로 모두 지정 가능; 상대 경로는 현재 디렉터리를 기준으로 해석됨) |

 ** 주의 ** 이 정의로 생성되는 파일 이름은 "fileheader+.<rank>"입니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/dat/result.md | TITLE: !RESULT -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/dat/result.html -->

# !RESULT

 해석 결과 데이터 파일을 지정합니다.

 
```text
!RESULT, NAME=<name> [,선택적 매개변수]
fileheader

#사용 예
!RESULT, NAME=fstrRES, IO=OUT, TYPE=BINARY
result.out
```

 
| 매개변수 | 내용 |
| --- | --- |
| `NAME` | 식별자(필수) |
| `IO` | 입출력 지정(생략 가능) |
| `TYPE` | 출력 형식(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NAME` | `fstrRES` | 솔버 출력 데이터, 시각화 도구 입력 데이터 |
|  | `fstrTEMP` | 온도 입력 데이터(열전도 해석 결과) |
|  | `vis_out` | 시각화 도구 출력 데이터 |
| `IO` | `IN` | 입력용 |
|  | `OUT` | 출력용 |
| `TYPE` | `TEXT` | 텍스트 형식(기본값) |
|  | `BINARY` | 바이너리 형식 |

 
| 변수명 | 내용 |
| --- | --- |
| fileheader | 해석 결과 데이터 파일명의 헤더(상대 경로와 절대 경로 모두 지정 가능. 상대 경로인 경우 현재 디렉터리에서의 경로가 됨) |

 ** 주의 ** 이 정의에 의해 생성되는 파일명은 "fileheader+.<rank>"가 됩니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/dat/subdir.md | TITLE: !SUBDIR -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/dat/subdir.html -->

# !SUBDIR

 파일 저장을 위한 서브디렉터리 분할을 지정한다.

 
```text
!SUBDIR, ON [,선택적 매개변수]

#사용 예
!SUBDIR, ON, LIMIT=8000
```

 
| 매개변수 | 내용 |
| --- | --- |
| `ON` | 유효(필수) |
| `LIMIT` | 파일 수(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `ON` | 없음 |  |
| `LIMIT` | <integer> | 디렉터리당 최대 파일 수(기본값: 5000) |

 ** 주의 ** 이 정의에 따라 여러 파일로 구성된 입출력 데이터는 자동으로 서브디렉터리 아래에 배치된다. 병렬 수가 LIMIT를 초과하는 경우, 추가로 `TRUNK0`, `TRUNK1`, ... 서브디렉터리 아래에 배치된다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/input_rules.md | TITLE: 입력 규칙 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/input_rules.html -->

# 입력 규칙

 FrontISTR의 입력 파일(전체 제어 데이터 `.dat`, 해석 제어 데이터 `.cnt`, 메시 데이터 `.msh`)은 공통 기술 규칙을 따른다.

 
## 파일의 특징

 
- 자유 형식 기반의 ASCII 형식 파일이다.
- `!`로 시작하는 헤더와 그 뒤에 이어지는 데이터로 구성된다.
- 헤더를 기술하는 순서는 기본적으로 자유롭다.
- 데이터 구분 기호에는 `,`를 사용한다.

 
> 참고: 해석 제어 데이터 고유의 특징
>
>  해석 제어 데이터(`.cnt`)에는 다음과 같은 추가 특징이 있다.
>
>  
> - 파일 내부는 크게 계산 제어 데이터, 솔버 제어 데이터, 후처리(시각화) 제어 데이터의 세 영역으로 나뉜다.
> - 파일 마지막에 `!END`를 입력하여 종료한다.

 
## 헤더

 데이터의 의미와 데이터 블록을 식별한다. 행의 첫 문자가 `!`로 시작하면 헤더로 간주한다.

 
## 헤더 행

 헤더와 그에 수반되는 매개변수를 기술한다.

 
- 헤더 행은 헤더로 시작해야 한다.
- 매개변수가 필요한 경우 `,`를 사용하여 뒤에 이어서 기술해야 한다.
- 매개변수가 값을 갖는 경우 매개변수 뒤에 `=`를 쓰고 그 다음에 값을 기술한다.
- 헤더 행은 여러 행에 걸쳐 기술할 수 없다.

 
## 데이터 행

 헤더 행 다음 행부터 시작하며 필요한 데이터를 기술한다.

 
- 데이터 행은 여러 행에 걸칠 수 있으며, 이는 각 헤더에서 정의하는 데이터 기술 규칙에 따라 결정된다.
- 데이터 행이 필요하지 않은 경우도 있다.

 
## 구분 문자

 데이터의 구분 문자에는 쉼표 `,`를 사용한다.

 
## 공백 처리

 공백은 무시된다.

 
## 이름

 이름에 사용할 수 있는 문자는 밑줄 `_`, 하이픈 `-`, 영숫자 `a-z A-Z 0-9`이며, 첫 문자는 `_` 또는 영문자 `a-z A-Z`로 시작해야 한다.

 
- 대소문자를 구분하지 않으며 내부적으로 모두 대문자로 처리한다.
- 이름의 최대 길이는 63자이다.

 
## 파일명

 파일명에 사용할 수 있는 문자는 밑줄 `_`, 하이픈 `-`, 마침표 `.`, 슬래시 `/`, 영숫자 `a-z A-Z 0-9`이다.

 
- 별도의 설명이 없는 한 파일명에는 경로를 포함해도 된다. 상대 경로와 절대 경로 모두 지정할 수 있다.
- 파일명의 최대 길이는 1023자이다.

 
## 부동소수점 데이터

 지수는 있어도 되고 없어도 된다. 지수 앞에는 `E` 또는 `e` 기호를 붙여야 한다. `E`와 `e` 중 어느 것을 사용해도 된다.

 
> 참고:
>
>  `D` 또는 `d`는 사용할 수 없다.

 
## 주석 행

 행의 시작이 `!!` 또는 `#`인 행은 주석 행으로 간주되어 무시된다.

 
- 주석 행은 파일 내 임의의 위치에 삽입할 수 있으며 개수에 제한이 없다.

 
## !END

 데이터의 끝을 나타낸다. 이 헤더가 나타나면 데이터 읽기를 종료한다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/amplitude.md | TITLE: !AMPLITUDE -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/amplitude.html -->

# !AMPLITUDE

 하중 조건을 정의하는 변수의 시간 변화를 지정한다. 시간 기준은 `TIME`으로 지정한다.

 
```text
!AMPLITUDE, NAME=<name> [, 선택적 매개변수]
VAL1, T1, VAL2, T2, VAL3, T3

# (한 줄에 최대 4개 항목)
```

 
| 매개변수 |  |
| --- | --- |
| `NAME` | 이름(필수) |
| `DEFINITION` | 유형(선택) |
| `TIME` | 시간 유형(선택) |
| `VALUE` | 값 유형(선택) |
| `INPUT` | 외부 파일명(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NAME` | <name> | `AMPLITUDE` 이름 |
| `DEFINITION` | TABULAR | 기본값(현재 버전에서 유일한 선택지) |
| `TIME` | STEP TIME | 스텝 시작부터의 시각(기본값) |
|  | TOTAL TIME | 해석 시작부터의 시각 |
| `VALUE` | RELATIVE | 상대값(기본값) |
|  | ABSOLUTE | 절대값 |
| `INPUT` | <filename> | 외부 파일명(생략 가능), 두 번째 행부터의 입력과 함께 사용할 수도 있음 |

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| `VAL1` | R | 시간 T1에서의 값 |
| `T1` | R | 시간 T1 |
| `VAL2` | R | 시간 T2에서의 값 |
| `T2` | R | 시간 T2 |
| `VAL3` | R | 시간 T3에서의 값 |
| `T3` | R | 시간 T3 |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/contactpair.md | TITLE: !CONTACTPAIR -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/contactpair.html -->

# !CONTACTPAIR

 접촉 해석에 사용하는 접촉면 쌍을 정의합니다.

 첫 번째 행

 
```text
!CONTACT PAIR, NAME=<name> [,선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| `NAME` | 접촉 쌍 이름(필수) |
| `TYPE` | 유형(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NAME` | <name> | 접촉 쌍 이름 |
| `TYPE` | `NODE-SURF` | 슬레이브 면은 절점 그룹<br>마스터 면은 표면 그룹(기본값) |
|  | `SURF-SURF` | 슬레이브 면과 마스터 면 모두 표면 그룹 |

 두 번째 행부터

 
```text
(두 번째 행부터) SLAVE_GRP, MASTER_GRP
(이하 동일)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| `SLAVE_GRP` | C | 슬레이브 면의 절점/표면 그룹명 |
| `MASTER_GRP` | C | 마스터 면의 표면 그룹명 |

 주의

 
- 해석 중 메시를 세분화(REFINE)할 때는 반드시 `TYPE=SURF-SURF`를 사용합니다.
- `TYPE=SURF-SURF`를 사용해도 FrontISTR 내부의 접촉 알고리즘은 점-대-면(point-to-surface) 알고리즘입니다.

 참고

 
- `TYPE=SURF-SURF`를 사용하면 FrontISTR는 내부적으로 슬레이브 표면 그룹에서 절점 그룹을 자동 생성하고, 해당 쌍을 절점 그룹과 표면 그룹으로 이루어진 접촉면 쌍으로 변환합니다.
- 자동 생성되는 절점 그룹의 이름은 원래 슬레이브 표면 그룹명 앞에 `FSTR_S2N_`를 붙여 구성합니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/egroup.md | TITLE: !EGROUP -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/egroup.html -->

# !EGROUP

 요소 그룹 정의

 첫 번째 행

 
```text
!EGROUP, EGRP=<egrp> [, 선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| `EGRP` | 요소 그룹 이름 (필수) |
| `GENERATE` | 요소 그룹에 속하는 절점 자동 생성 (생략 가능) |
| `INPUT` | 외부 파일명(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `EGRP` | <egrp> | 요소 그룹 이름 |
| `GENERATE` | 없음 | 요소 그룹에 속하는 절점 자동 생성 |
| `INPUT` | <filename> | 외부 파일명(생략 가능), 두 번째 행부터의 입력과 함께 사용할 수도 있음 |

 두 번째 행부터(GENERATE를 사용하지 않는 경우)

 
```text
(두 번째 행) elem1, elem2, elem3 ...
(이하 동일)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| elemX | I | 요소 그룹에 속하는 요소 번호 |

 두 번째 행부터(GENERATE를 사용하는 경우)

 
```text
(두 번째 행) elem1, elem2, elem3
(이하 동일)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| elem1 | I | 요소 그룹 내 첫 번째 요소 번호 |
| elem2 | I | 요소 그룹 내 마지막 요소 번호 |
| elem3 | I | 요소 번호 증분(생략 가능, 생략 시 elem3=1) |

 주의

 
- 한 행에 임의 개수의 요소를 넣을 수 있다. 또한 다음 옵션이 시작될 때까지 임의 개수의 행을 삽입할 수 있다.
- 지정하는 요소는 `!EGROUP`보다 먼저 정의되어 있어야 한다.
- `!ELEMENT` 옵션에서 정의되지 않은 요소는 제외되고 경고 메시지가 표시된다.
- 지정된 요소가 이미 같은 그룹에 존재하는 경우 무시되고 경고 메시지가 표시된다.
- 모든 요소는 `ALL`이라는 이름의 요소 그룹에 속한다(자동 생성).
- 하나의 그룹을 여러 번에 나누어 정의할 수 있다.

 사용 예

 
```text
!EGROUP, EGRP=EA01
1, 2, 3, 4, 5, 6
101, 102
205

!EGROUP, EGRP=EA02
101, 102

!EGROUP, EGRP=EA01                 그룹 "EA01"에 "501, 505"가 추가된다.
501, 505

!EGROUP, EGRP=EA04,　GENERATE      그룹 "NA04"에
301, 309, 2                       "301,303,305,307,309,311,312,313"이 추가된다.

311, 313
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/element.md | TITLE: !ELEMENT -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/element.html -->

# !ELEMENT

 요소 정의

 첫 번째 행

 
```text
!ELEMENT, TYPE=<type> [, 선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| `TYPE` | 요소 유형(필수) |
| `EGRP` | 요소 그룹명(생략 가능) |
| `MATITEM` | 재료 물성을 요소별로 정의하는 경우의 물성 항목 수 |
|  | (섹션별로 물성을 정의하는 경우에는 사용하지 않음) |
| `INPUT` | 외부 파일명(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| TYPE | 111 | 로드, 링크 요소(1차) |
|  | 231 | 삼각형 요소(1차) |
|  | 232 | 삼각형 요소(2차) |
|  | 241 | 사각형 요소(1차) |
|  | 242 | 사각형 요소(2차) |
|  | 301 | 트러스 요소(1차) |
|  | 341 | 사면체 요소(1차) |
|  | 342 | 사면체 요소(2차) |
|  | 351 | 삼각기둥 요소(1차) |
|  | 352 | 삼각기둥 요소(2차) |
|  | 361 | 육면체 요소(1차) |
|  | 362 | 육면체 요소(2차) |
|  | 511 | 커넥터 요소(스프링·댐퍼용, 2절점, 1차) |
|  | 541 | 인터페이스 요소(사각형 단면, 1차) |
|  | 611 | 보 요소(1차) |
|  | 641 | 보 요소(1차, 3자유도 절점 4개 사용) |
|  | 731 | 삼각형 셸 요소(1차) |
|  | 741 | 사각형 셸 요소(1차) |
|  | 743 | 사각형 셸 요소(2차) |
|  | 761 | 삼각형 셸 요소(1차, 3자유도 절점 6개 사용) |
|  | 781 | 사각형 셸 요소(1차, 3자유도 절점 8개 사용) |
| `EGRP` | <egrp> | 요소 그룹명(생략 가능) |
| `INPUT` | <filename> | 외부 파일명(생략 가능), 두 번째 행부터의 입력과 함께 사용할 수도 있음 |

 두 번째 행부터

 
```text
(두 번째 행) ELEM_ID, nod1, nod2, nod3, ..., MAT1, MAT2, ...

(이하 동일)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| ELEM_ID | I | 요소 번호 |
| nodX | I | 연결 정보 |
| MATy | R | 요소별 물성값 |

 주의

 
- 요소 유형, 연결 정보의 자세한 내용은 "4장 요소 라이브러리"를 참조할 것.
- 연결 정보에서 지정하는 절점은 "`!ELEMENT`"보다 앞에 정의되어 있어야 한다.
- 요소 번호는 연속일 필요가 없다.
- "`!ELEMENT`" 옵션은 여러 번 정의할 수 있다.
- 요소 번호는 자연수여야 한다. 생략할 수 없다.
- 같은 요소 번호를 중복하여 사용하는 경우 마지막에 입력한 값이 사용된다. 이 경우 경고 메시지가 출력된다.
- 정의되지 않은 절점을 연결 정보에 사용할 수 없다.
- 하나의 요소 정의를 여러 줄에 걸쳐 기술해도 된다.
- 요소 유형 511은 스프링(`!SPRING_A` / `!SPRING_D`) 또는 댐퍼(`!DASHPOT_A` / `!DASHPOT_D`)를 나타내는 2절점 커넥터 요소이다. 연결 정보에는 결합할 두 절점을 지정하고, 대응하는 [`!SECTION`](section.ja.md)에는 `TYPE=INTERFACE`를 지정한다. 스프링 상수와 감쇠 계수는 해석 제어 데이터(cnt)의 `!MATERIAL` 아래에서 지정한다.

 사용 예

 
```text
!ELEMENT, TYPE=231
1, 1, 2, 3
2, 4, 8, 5
4, 6, 7, 8
!ELEMENT, TYPE=361, EGRP=A
101, 101, 102, 122, 121, 201, 202, 222, 221
102, 102, 103, 123, 122, 202, 203, 223, 222
103, 103, 104, 124, 123, 203, 204, 224, 223
!ELEMENT, TYPE=511, EGRP=ESPRING
201, 1, 101
202, 2, 102
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/embedpair.md | TITLE: !EMBEDPAIR -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/embedpair.html -->

# !EMBEDPAIR

 임베딩 페어 정의

 첫 번째 행

 
```text
!EMBED PAIR, NAME=<name>
```

 
| 매개변수 |  |
| --- | --- |
| `NAME` | 임베딩 페어명(필수) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NAME` | <name> | 임베딩 페어명 |

 두 번째 행부터

 
```text
(두 번째 행부터) SLAVE_GRP, MASTER_GRP
(이하 동일)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| `SLAVE_GRP` | C | 임베딩 대상 절점 그룹명 |
| `MASTER_GRP` | C | 임베딩 대상이 들어가는 요소 그룹명 |


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/end.md | TITLE: !END -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/end.html -->

# !END

 메시 데이터의 끝

 이 헤더가 나타나면 메시 데이터 읽기가 종료됩니다.

 첫 번째 행

 
```text
!END
```

 매개변수

 없음

 두 번째 행부터

 없음


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/equation.md | TITLE: !EQUATION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/equation.html -->

# !EQUATION

 구속 절점 그룹 정의

 첫 번째 행

 
```text
!EQUATION [, 선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| INPUT | 외부 파일명(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| INPUT | <filename> | 외부 파일명(생략 가능), 두 번째 행부터의 입력과 함께 사용할 수도 있음 |

 두 번째 행부터

 
```text
(두 번째 행) NEQ, CONST
```

 
```text
(세 번째 행부터) nod1, DOF1, A1, nod2, DOF2, A2 ... (한 줄에 최대 7개 항목)
이후 반복
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| `NEQ` | I | 방정식의 항 수 |
| `CONST` | R | 방정식의 상수항(우변 값) |
| nod1 | I/C | 첫 번째 절점 또는 절점 그룹 |
| DOF1 | I | 첫 번째 절점 또는 절점 그룹의 구속 자유도 |
| A1 | R | 첫 번째 절점 또는 절점 그룹의 계수 |
| nod2 | I/C | 두 번째 절점 또는 절점 그룹 |
| DOF2 | I | 두 번째 절점 또는 절점 그룹의 구속 자유도 |
| A2 | R | 두 번째 절점 또는 절점 그룹의 계수 |

 주의

 
- `!NODE`로 정의되지 않은 절점 또는 절점 그룹을 지정하면 무시되고 경고 메시지가 표시됩니다.
- nod1=nod2이면 무시되고 경고 메시지가 표시됩니다.
- 절점 그룹을 지정한 경우 절점 수가 일치하지 않으면 오류가 발생합니다.
- 자유도 번호는 해석 유형과 요소에 따라 달라집니다. 일치하지 않는 자유도는 무시되고 경고 메시지가 표시됩니다.

 예

 
```text
!EQUATION
3
101, 1, 1.0, 102, 1, -1.0, 103, 1, -1.0
2
NG1, 2, 1.0, NG5, 2, -1.0
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/header.md | TITLE: !HEADER -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/header.html -->

# !HEADER

 메시 데이터의 제목

 첫 번째 행

 
```text
!HEADER
```

 매개변수

 없음

 두 번째 행부터

 
```text
(두 번째 행) TITLE
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| TITLE | C | 헤더 제목 |

 사용 예

 
```text
!HEADER
Mesh for CFD Analysis
```

 주의

 
- 생략 가능.
- 헤더는 여러 행에 걸쳐도 되지만, 헤더로 인식되는 것은 첫 번째 행의 127번째 열까지이다.
- 「`!HEADER`」를 여러 번 정의하면 내용이 갱신되고 경고 메시지가 표시된다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/index.md | TITLE: 메시 데이터 (msh) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/index.html -->

# 메시 데이터 (msh)

 메시 데이터 파일 (`.msh`)에서 사용하는 키워드 목록입니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/material.md | TITLE: !MATERIAL -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/material.html -->

# !MATERIAL

 재료 물성의 정의

 물성이 온도 의존적인 경우에는 대응하는 각 온도에 대해 테이블로 입력할 수 있다.

 첫 번째 행

 
```text
!MATERIAL, NAME=<name> [, 선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| `NAME` | 재료명(필수) |
| `ITEM` | 물성 항목 수(생략 가능, 생략한 경우 "1"이 됨) |
| `INPUT` | 외부 파일명(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NAME` | <name> | 재료명 |
| `ITEM` | <ITEMnum> | 사용자 정의 물성 항목 수 |
| `INPUT` | <filename> | 외부 파일명(생략 가능), 두 번째 행부터의 입력과 함께 사용할 수도 있음 |

 두 번째 행부터

 
```text
(두 번째 행) !ITEM=1, SUBITEM=<k>
(세 번째 행) VAL1-1-1, VAL1-1-2, … VAL1-1-k, TEMP1-1
(4번째 행) VAL1-2-1, VAL1-2-2, … VAL1-2-k, TEMP1-2
・・

(L+2번째 행)VAL1-L-1, VAL1-L-2, … VAL1-L-k, TEMP1-L

이하 "!ITEM=<ITEMnum>"까지 반복하여 정의한다
```

 
| 하위 매개변수(`!ITEM`에 대한 것) |  |
| --- | --- |
| `SUBITEM` | 각 물성 항목에서 정의되는 하위 물성 항목 수 |
|  | (생략 가능, 생략한 경우 "1"이 됨.) |

 
| 하위 매개변수명 | 매개변수 값 | 내용 |
| --- | --- | --- |
| `SUBITEM` | <subITEMnum> | 사용자 정의 하위 물성 항목 수 |

 【m번째 물성이 온도 의존적인 경우】

 온도 의존 테이블의 항목 수가 N인 경우, 다음과 같이 입력한다:

 
```text
!ITEM=m, SUBITEM=k
VALm1-1, ..., VALm1-k, TEMPm1
VALm2-1, ..., VALm2-k, TEMPm2
...
VALmN-1, ..., VALmN-k, TEMPm-N
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| VALmn-k | R | 물성값(온도 의존) |
| TEMPmn | R | 대응하는 온도 |

 TEMPm1 < TEMPm2 < ... < TEMPmN 이어야 한다.

 온도가 TEMPm1 이하인 경우에는 VALm1, TEMPmN 이상인 경우에는 VALmN이 사용된다.

 【m번째 물성이 온도 의존적이지 않은 경우】

 
```text
!ITEM=m, SUBITEM=k
VALm1-1, ..., VALm1-k
VALm2-1, ..., VALm2-k
...
VALmN-1, ..., VALmN-k
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| VALmn-k | R | 물성값(온도 의존 없음) |

 
> 참고:
>
>  
> - 재료명이 중복된 경우 오류가 발생한다.
> - `!SECTION` 옵션에서 참조하는 `MATERIAL`이 정의되어 있지 않은 경우 오류가 발생한다.
> - `!ELEMENT` 옵션에서 매개변수 `MATITEM`을 사용하여 요소별로 물성을 입력한 경우의 값이 우선 사용된다. 이 경우 `!MATERIAL` 옵션을 사용하여 입력한 물성값은 사용되지 않는다.
> - `!ITEM=m` 하위 옵션의 수와 매개변수 `ITEM`의 수가 일치하지 않거나 정의되지 않은 하위 옵션이 있는 경우 오류가 발생한다.
> - `!ITEM=m` 하위 옵션은 m이 작은 순서대로 배열하지 않아도 된다.
> - `SUBITEM=k` 하위 옵션과 온도 의존성을 사용하는 경우, 생략한 값은 "0.0"이 된다.
> - 온도 의존성을 사용하는 경우 온도가 낮은 순서대로 정의해야 한다.
> - 온도 의존성을 사용하는 경우 같은 온도를 2회 이상 사용하면 오류가 발생한다.

 사용 예

 
```text
!MATERIAL, NAME= STEEL, ITEM= 2
!ITEM=1 온도 의존성 없음
35.0
!ITEM=2
40.0, 0.0
45.0, 100.0
50.0, 200.0

!MATERIAL, NAME= COPPER         항목 수＝1(기본값)
!ITEM=1 온도 의존성 없음
80.0
```

 잘못된 사용 예

 예 1【매개변수 `ITEM`과 `!ITEM=m` 하위 옵션의 수가 일치하지 않음-1】

 
```text
!MATERIAL, NAME= STEEL, ITEM= 2
!ITEM=3
20.0
!ITEM=1
35.0
!ITEM= 2
40.0
```

 예 2【매개변수 「ITEM」과 「!ITEM=m」 하위 옵션의 수가 일치하지 않음-2】

 
```text
!MATERIAL, NAME= STEEL, ITEM= 3
!ITEM=3
20.0
!ITEM= 2
40.0
!MATERIAL, NAME= COPPER
...
```

 
## 탄성 정적 해석 및 모드 해석

 
```text
!MATERIAL, NAME=<name>, ITEM=<ITEMnum>
!ITEM=1, SUBITEM=2
<Young_modulus>, <Poisson_ratio>
!ITEM=2
<Density>
!ITEM=3
<Expansion_coeff>
```

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| NAME | <name> | 재료명, !SECTION의 MATERIAL에 대응 |
| ITEM | <ITEMnum> | 사용자 정의 물성 항목 수(1 이상)<br><Young_modulus>-----영률(필수)<br><Poisson_ratio>-------포아송비(필수)<br><Density>--------------질량 밀도(ITEMnum=2일 때 필수)<br>< Expansion_coeff >--선팽창 계수(ITEMnum=3일 때) |

 (예)

 
```text
!MATERIAL, NAME=M1,
ITEM=3                   --- 재료명 M1의 재료에서는 3종의 물성값을 정의한다는 의미
!ITEM=1, SUBITEM=2       --- !ITEM=1에서는 영률과 포아송비를 정의(필수)
4000.0, 0.3
!ITEM=2                  --- !ITEM=2에서 질량 밀도를 정의해야 함(ITEM=2일 때 필수)
8.0102E-0
!ITEM=3                  --- !ITEM=3에서 선팽창 계수를 정의
1.0E-5
```

 
> 참고:
>
>  
> - **모드 해석에서는 온도 의존성이 있는 재료를 사용할 수 없다.**

 
## 열전도 해석

 링크, 평면, 솔리드, 3차원 판 요소의 경우

 
```text
!MATERIAL, NAME=<name>, ITEM=3
!ITEM=1, SUBITEM=2
<Density>, <Temperature>

!ITEM=2, SUBITEM=2
<Specific_heat>, <Temperature>

!ITEM=3, SUBITEM=2
<Conductivity>, <Temperature>
```

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NAME` | <name> | 재료명, !SECTION의 MATERIAL에 대응 |
| `ITEM` | <ITEMnum> | 사용자 정의 물성 항목 수(**항상 3**)<br><Density>----------밀도<br><Specific_heat>----비열<br><Conductivity>-----열전도율<br><Temperature>------온도 |

 (예)

 
```text
!MATERIAL, NAME=M1,
ITEM=3                   ---　재료명 M1의 재료에서는 3종의 물성값을 정의한다는 의미

!ITEM=1, SUBITEM=1       ---　!ITEM=1에서는 밀도와 온도를 정의(필수)
7850., 300.
7790., 500.
7700., 800.

!ITEM=2, SUBITEM=1       ---　!ITEM=2에서는 비열과 온도를 정의(필수)
0.465, 300.
0.528, 500.
0.622, 800.

!ITEM=3                  ---　!ITEM=3에서는 열전도율과 온도를 정의(필수)
43., 300.
38.6, 500.
27.7, 800.
```

 인터페이스 요소의 경우

 `!SECTION` 헤더에서 정의한다.(재료 데이터는 불필요)

 (예)

 
```text
!SECTION, TYPE=INTERFACE, EGRP=GAP   ---　섹션 정의
1.0, 20.15, 8.99835E-9, 8.99835E-9
```

 위의 `!SECTION`에서는 인터페이스 요소 중 그룹명=GAP에 속하는 요소의 갭 매개변수를 정의한다.

 
- 제1 매개변수: 갭 폭
- 제2 매개변수: 갭 열전달 계수
- 제3 매개변수: 갭 복사 계수 1
- 제4 매개변수: 갭 복사 계수 2

 
## 참고

 
```text
program TEST
use hecmw
implicit REAL*8 (A-H,O-Z)
type (hecmwT_local_mesh) :: hecMESH

!C
!C    !MATERIAL, NAME=SUS304, ITEM=3
!C    !ITEM=1, SUBITEM= 3
!C      100.0, 200.0, 300.0, 0.00
!C      101.0, 210.0, 301.0, 1.00
!C      102.0, 220.0, 302.0, 2.00
!C      103.0, 230.0, 303.0, 3.00
!C    !ITEM=3, SUBITEM= 2
!C      1000.0, , 0.00
!C      1001.0, 1., 1.00
!C      1002.0, 2., 2.00
!C      1003.0, 3., 3.00
!C    !ITEM=2
!C      5000.0
!C
!C    !MATERIAL, NAME=FEC, ITEM=2
!C    !ITEM=1, SUBITEM= 3
!C      2100.0, 2200.0, 2300.0, 0.00
!C      2101.0, 2210.0, 2301.0, 1.00
!C      2102.0, 2220.0, 2302.0, 2.00
!C      2103.0, 2230.0, 2303.0, 3.00
!C      3103.0, 3230.0, 2304.0, 4.00
!C    !ITEM=2
!C      6000.0, 10.0
!C      6500.0, 30.0
!C

hecMESH%material%n_mat = 2

nn= hecMESH%material%n_mat
allocate (hecMESH%material%mat_name(nn))

hecMESH%material%mat_name(1)= 'SUS304'
hecMESH%material%mat_name(2)= 'FEC'

nn= hecMESH%material%n_mat
allocate (hecMESH%material%mat_ITEM_index(0:nn))
hecMESH%material%mat_ITEM_index(0)= 0
hecMESH%material%mat_ITEM_index(1)= 3
hecMESH%material%mat_ITEM_index(2)= hecMESH%material%mat_ITEM_index(1) + 2

hecMESH%material%n_mat_ITEM= hecMESH%material%mat_ITEM_index(hecMESH%material%n_mat)

nn= hecMESH%material%n_mat_ITEM
allocate (hecMESH%material%mat_subITEM_index(0:nn))

hecMESH%material%mat_subITEM_index(0)= 0
hecMESH%material%mat_subITEM_index(1)= 3
hecMESH%material%mat_subITEM_index(2)= hecMESH%material%mat_subITEM_index(1) + 1
hecMESH%material%mat_subITEM_index(3)= hecMESH%material%mat_subITEM_index(2) + 2
hecMESH%material%mat_subITEM_index(4)= hecMESH%material%mat_subITEM_index(3) + 3
hecMESH%material%mat_subITEM_index(5)= hecMESH%material%mat_subITEM_index(4) + 1

hecMESH%material%n_mat_subITEM=
&        hecMESH%material%mat_subITEM_index(hecMESH%material%n_mat_ITEM)

nn= hecMESH%material%n_mat_subITEM
allocate (hecMESH%material%mat_TABLE_index(0:nn))
hecMESH%material%mat_TABLE_index( 0)= 0
hecMESH%material%mat_TABLE_index( 1)= 4
hecMESH%material%mat_TABLE_index( 2)= hecMESH%material%mat_TABLE_index( 1) + 4
hecMESH%material%mat_TABLE_index( 3)= hecMESH%material%mat_TABLE_index( 2) + 4
hecMESH%material%mat_TABLE_index( 4)= hecMESH%material%mat_TABLE_index( 3) + 1
hecMESH%material%mat_TABLE_index( 5)= hecMESH%material%mat_TABLE_index( 4) + 4
hecMESH%material%mat_TABLE_index( 6)= hecMESH%material%mat_TABLE_index( 5) + 4
hecMESH%material%mat_TABLE_index( 7)= hecMESH%material%mat_TABLE_index( 6) + 5
hecMESH%material%mat_TABLE_index( 8)= hecMESH%material%mat_TABLE_index( 7) + 5
hecMESH%material%mat_TABLE_index( 9)= hecMESH%material%mat_TABLE_index( 8) + 5
hecMESH%material%mat_TABLE_index(10)= hecMESH%material%mat_TABLE_index( 9) + 2

hecMESH%material%n_mat_TABLE=
&        hecMESH%material%mat_TABLE_index(hecMESH%material%n_mat_subITEM)

nn= hecMESH%material%n_mat_TABLE
allocate (hecMESH%material%mat_VAL (nn))
allocate (hecMESH%material%mat_TEMP(nn))

hecMESH%material%mat_VAL = 0.d0
hecMESH%material%mat_TEMP= 0.d0

hecMESH%material%mat_VAL ( 1)= 100.0d0
hecMESH%material%mat_TEMP( 1)=   0.0d0
hecMESH%material%mat_VAL ( 2)= 101.0d0
hecMESH%material%mat_TEMP( 2)=   1.0d0
hecMESH%material%mat_VAL ( 3)= 102.0d0
hecMESH%material%mat_TEMP( 3)=   2.0d0
hecMESH%material%mat_VAL ( 4)= 103.0d0
hecMESH%material%mat_TEMP( 4)=   3.0d0

hecMESH%material%mat_VAL ( 5)= 200.0d0
hecMESH%material%mat_TEMP( 5)=   0.0d0

hecMESH%material%mat_VAL (13)= 5000.0d0

hecMESH%material%mat_VAL (14)= 1000.0d0
hecMESH%material%mat_TEMP (14)=   0.0d0
hecMESH%material%mat_VAL (15)= 1001.0d0
hecMESH%material%mat_TEMP (15)=   1.0d0
hecMESH%material%mat_VAL (16)= 1002.0d0
hecMESH%material%mat_TEMP (16)=   2.0d0
hecMESH%material%mat_VAL (17)= 1003.0d0
hecMESH%material%mat_TEMP (17)=   3.0d0

hecMESH%material%mat_VAL (18)=  0.0d0
hecMESH%material%mat_TEMP (18)= 0.0d0
hecMESH%material%mat_VAL (19)=  1.0d0
hecMESH%material%mat_TEMP (19)= 1.0d0
hecMESH%material%mat_VAL (20)=  2.0d0
hecMESH%material%mat_TEMP (20)= 2.0d0
hecMESH%material%mat_VAL (21)=  3.0d0
hecMESH%material%mat_TEMP (21)= 3.0d0

hecMESH%material%mat_VAL (22)= 2100.0d0
hecMESH%material%mat_TEMP (22)=   0.0d0
hecMESH%material%mat_VAL (23)= 2101.0d0
hecMESH%material%mat_TEMP (23)=   1.0d0
hecMESH%material%mat_VAL (24)= 2102.0d0
hecMESH%material%mat_TEMP (24)=   2.0d0
hecMESH%material%mat_VAL (25)= 2103.0d0
hecMESH%material%mat_TEMP (25)=   3.0d0
hecMESH%material%mat_VAL (26)= 3103.0d0
hecMESH%material%mat_TEMP (26)=   4.0d0

write (*,'(a,i10)') '%n_mat_ITEM ', hecMESH%material%n_mat_ITEM
write (*,'(a,i10)') '%n_mat_subITEM', hecMESH%material%n_mat_subITEM
write (*,'(a,i10)') '%n_mat_TABLE ', hecMESH%material%n_mat_TABLE

end program TEST
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/ngroup.md | TITLE: !NGROUP -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/ngroup.html -->

# !NGROUP

 절점 그룹 정의

 첫 번째 행

 
```text
!NGROUP, NGRP=<ngrp> [, 선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| `NGRP` | 절점 그룹 이름(필수) |
| `GENERATE` | 절점 그룹에 속하는 절점 자동 생성(생략 가능) |
| `INPUT` | 외부 파일명(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `NGRP` | <ngrp> | 절점 그룹 이름 |
| `GENERATE` | 없음 | 절점 그룹에 속하는 절점 자동 생성 |
| `INPUT` | <filename> | 외부 파일명(생략 가능), 두 번째 행부터의 입력과 함께 사용할 수도 있음 |

 두 번째 행부터(GENERATE를 사용하지 않는 경우)

 
```text
(두 번째 행) nod1, nod2, nod3
(이하 동일)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| nodX | I | 절점 그룹에 속하는 절점 번호 |

 두 번째 행부터(GENERATE를 사용하는 경우)

 
```text
(두 번째 행) nod1, nod2, nod3
(이하 동일)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| nod1 | I | 절점 그룹 내 첫 번째 절점 번호 |
| nod2 | I | 절점 그룹 내 마지막 절점 번호 |
| nod3 | I | 절점 번호 증분(생략 가능, 생략 시 nod3=1) |

 주의

 
- 한 행에 임의 개수의 절점을 넣을 수 있다. 또한 다음 옵션이 시작될 때까지 임의 개수의 행을 삽입할 수 있다.
- 지정하는 절점은 `!NGROUP`보다 먼저 정의되어 있어야 한다.
- `!NODE` 옵션에서 정의되지 않은 절점은 제외되고 경고 메시지가 표시된다.
- 지정된 절점이 이미 같은 그룹에 존재하는 경우 무시되고 경고 메시지가 표시된다.
- 모든 절점은 `ALL`이라는 이름의 절점 그룹에 속한다(자동 생성).
- 하나의 그룹을 여러 번에 나누어 정의할 수 있다.

 사용 예

 
```text
!NGROUP, NGRP= NA01
1, 2, 3, 4, 5, 6
101, 102
!NGROUP, NGRP= NA02
101, 102
!NGROUP, NGRP= NA01              --- 그룹 "NA01"에 "501, 505"가 추가된다.
501, 505
!NGROUP, NGRP= NA02              --- 그룹 "NA02"에 "501, 505"가 추가된다.
501, 505
!NGROUP, NGRP= NA04,GENERATE     --- 그룹 "NA04"에
301, 309, 2                      --- "301,303,305,307,309,311,312,313"이 추가된다.
311, 313
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/node.md | TITLE: !NODE -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/node.html -->

# !NODE

 절점 좌표의 정의

 첫 번째 행

 
```text
!NODE [, 선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| `SYSTEM` | 좌표계(생략 가능) |
| `NGRP` | 절점 그룹명(생략 가능) |
| `INPUT` | 외부 파일명(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `SYSTEM` | R | 데카르트 좌표계(기본값) |
|  | C | 원통 좌표계 |
| `NGRP` | <ngrp> | 절점 그룹명(생략 가능) |
| `INPUT` | <filename> | 외부 파일명(생략 가능), 두 번째 행부터의 입력과 함께 사용할 수도 있음 |

 두 번째 행부터

 
```text
(두 번째 행) NODE_ID, Xcoord, Ycoord, Zcoord
(이하 동일)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| NODE_ID | I | 절점 번호 |
| Xcoord | R | X 좌표 |
| Ycoord | R | Y 좌표 |
| Zcoord | R | Z 좌표 |

 주의

 
- 구분 기호를 포함하여 절점 좌표를 생략하면 값은 「0.0」이 된다.
- 이미 정의된 절점을 다시 정의하면 내용이 갱신되고 경고 메시지가 표시된다.
- 「`!ELEMENT`」에서 참조되지 않는 절점은 제외된다.
- 「`!ELEMENT`」에서 정의되는 절점은 「`!ELEMENT`」보다 먼저 정의되어 있어야 한다.

 사용 예

 
```text
!NODE, NGRP=TEST
1, 0.0, 0.0, 0.5
2, 0.0, 0.0, 1.0
3, 0.0,,1.5                   --- Y 좌표는 「0.0」
4,                            --- X,Y,Z 좌표는 「0.0」
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/section.md | TITLE: !SECTION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/section.html -->

# !SECTION

 단면 정의

 첫 번째 행

 
```text
!SECTION, TYPE=<type>, EGRP=<egrp> [, 선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| `TYPE` | 단면 유형(필수) |
| `EGRP` | 요소 그룹 이름(필수) |
| `MATERIAL` | 사용자 정의 재료 이름(필수) |
| `SECOPT` | 요소 유형용 보조 매개변수(선택; 생략 시 기본값 0) |
| `INPUT` | 외부 파일명(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `TYPE` | `SOLID` | 봉, 삼각형, 사각형, 사면체, 프리즘 및 육면체 요소 |
|  | `SHELL` | 셸 요소 |
|  | `BEAM` | 보 요소 |
|  | `INTERFACE` | 인터페이스 요소, 커넥터 요소 |
| `EGRP` | <egrp> | 요소 그룹 이름 |
| `MATERIAL` | <material > | 사용자 정의 재료 이름 |
| `SECOPT` | <secopt> | = 0: 미지정, 평면 응력<br>= 1: 평면 변형률<br>= 2: 축대칭<br>=10: 0 + 저차 적분<br>=11: 1 + 저차 적분<br>=12: 2 + 저차 적분 |
| `INPUT` | <filename> | 외부 파일명(생략 가능), 두 번째 행부터의 입력과 함께 사용할 수도 있음 |

 섹션 타입은 `EGRP`로 지정한 요소 그룹에 포함된 요소 타입과 일치해야 한다. 대응 관계는 다음과 같다(요소 타입의 자세한 내용은 [`!ELEMENT`](element.ja.md) 참조).

 
| 섹션 타입 | 대응하는 요소 타입 |
| --- | --- |
| `SOLID` | 111, 231, 232, 241, 242, 301, 341, 342, 351, 352, 361, 362 |
| `SHELL` | 731, 741, 743, 761, 781 |
| `BEAM` | 611, 641 |
| `INTERFACE` | 511(커넥터 요소), 541(인터페이스 요소) |

 두 번째 행부터

 【TYPE=SOLID】의 경우

 
```text
(두 번째 행) `THICKNESS`
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| `THICKNESS` | R | 트러스 요소의 단면적(필수) |

 `TYPE=SOLID`에서는 `THICKNESS`를 생략할 수 있다.

 【`TYPE=SHELL`】의 경우

 
```text
(두 번째 행) THICKNESS, INTEGPOINTS
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| `THICKNESS` | R | 쉘 단면 두께 |
| `INTEGPOINTS` | I | 쉘 단면 두께 방향의 적분점 수 |

 【`TYPE=BEAM`】의 경우

 
```text
(두 번째 행) vx,vy,vz,area,Iyy,Iz,Jx
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| vx,vy,vz | R | 기준축 방향 |
| area | R | 단면적 |
| Iyy, Izz | R | 단면 2차 모멘트 |
| Jx | R | 비틀림 상수 |

 【`TYPE=INTERFACE`】의 경우

 
```text
(두 번째 행) THICKNESS, GAPCON, GAPRAD1, GAPRAD2
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| `THICKNESS` | R | 단면 두께 |
| `GAPCON` | R | 간극 열전달 계수(생략 시 0) |
| `GAPRAD1` | R | 간극 복사 열전달 계수 1(생략 시 0) |
| `GAPRAD2` | R | 간극 복사 열전달 계수 2(생략 시 0) |

 `TYPE=INTERFACE`는 인터페이스 요소(541)뿐 아니라 스프링·댐퍼를 나타내는 커넥터 요소(511)에도 사용한다. 커넥터 요소의 경우 `THICKNESS`는 단면이 없으므로 강성·감쇠 계산에는 사용되지 않지만, 두 번째 행 자체는 생략할 수 없으며 양의 값을 지정해야 한다(`1.0`이면 된다). `GAPCON`, `GAPRAD1`, `GAPRAD2`는 열전달 계산에 사용하는 매개변수이며 커넥터 요소에서는 사용하지 않는다.

 커넥터 요소의 `MATERIAL`에는 스프링 상수 또는 감쇠 계수를 정의한 재료명(cnt 측의 `!MATERIAL` 아래에서 `!SPRING_A` / `!SPRING_D` / `!DASHPOT_A` / `!DASHPOT_D`를 지정한 재료)을 지정한다.

 주의

 
- `TYPE` 매개변수와 요소 유형이 일치하지 않으면 오류가 발생한다.
- `SECTION` 정보가 없는 요소가 하나라도 있으면 오류가 발생한다.
- 단면 이름이 중복되면 오류가 발생한다.

 예

 
```text
!SECTION, EGRP=SOLID1, TYPE=SOLID, MATERIAL=STEEL
!SECTION, EGRP=SHELL2, TYPE=SHELL, MATERIAL=STEEL
1.0, 5
```

 커넥터 요소(요소 타입 511)에 섹션을 지정하는 예. 솔리드 요소와 혼재시키는 경우에는 요소 그룹을 나누어 섹션을 정의한다.

 
```text
!ELEMENT, TYPE=341, EGRP=ESLD
...
!ELEMENT, TYPE=511, EGRP=ESPRING
...
!SECTION, TYPE=SOLID,     EGRP=ESLD,    MATERIAL=MATERIAL1
1.0
!SECTION, TYPE=INTERFACE, EGRP=ESPRING, MATERIAL=MATERIAL2
1.0
```

 
## 관련 항목

 
- [`!ELEMENT`](element.ja.md)
- [`!SPRING_A`](../cnt/spring_a.ja.md)
- [`!SPRING_D`](../cnt/spring_d.ja.md)
- [`!DASHPOT_A`](../cnt/dashpot_a.ja.md)
- [`!DASHPOT_D`](../cnt/dashpot_d.ja.md)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/sgroup.md | TITLE: !SGROUP -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/sgroup.html -->

# !SGROUP

 표면 그룹 정의

 첫 번째 행

 
```text
!SGROUP, SGRP=<sgrp> [, 선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| `SGRP` | 표면 그룹 이름(필수) |
| `INPUT` | 외부 파일명(생략 가능) |

 
| 매개변수명 | 매개변수 값 | 설명 |
| --- | --- | --- |
| `SGRP` | <sgrp> | 표면 그룹 이름 |
| `INPUT` | <filename> | 외부 파일명(생략 가능), 두 번째 행부터의 입력과 함께 사용할 수도 있음 |

 두 번째 행부터

 
```text
(두 번째 행) elem1, lsuf1, elem2, lsuf2, elem3, lsuf3, ...
(이후 동일)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| elemX | I | 표면 그룹에 속하는 요소 번호 |
| lsufX | I | 표면 그룹에 속하는 요소의 국소 표면 번호 |

 주의

 
- 요소 유형과 표면 번호는 "4장 요소 라이브러리"를 참조하십시오.
- 표면은 요소와 국소 표면 번호의 조합으로 정의됩니다. 한 줄에 임의 개수의 표면을 입력할 수 있습니다. 다음 옵션이 시작될 때까지 임의 개수의 줄을 추가할 수도 있습니다. 각 (요소, 국소 표면 번호) 쌍은 동일한 줄에 있어야 합니다.
- 지정할 요소는 "`!SGROUP`"보다 먼저 정의되어 있어야 합니다.
- 요소가 "`!ELEMENT`" 옵션으로 정의되어 있지 않으면 무시되고 경고 메시지가 표시됩니다.
- "`!ELEMENT`" 옵션으로 정의되지 않은 요소를 포함하는 표면은 제외되고 경고 메시지가 표시됩니다.
- 요소 유형과 표면 번호가 일치하지 않는 표면은 제외되고 경고 메시지가 표시됩니다.
- 하나의 그룹을 여러 개의 분리된 정의로 정의할 수 있습니다.

 예

 
```text
!SGROUP, SGRP= SUF01
101, 1, 102, 1, 103, 2, 104, 2
201, 1, 202, 1
501, 1
!SGROUP, SGRP= SUF02
101, 2, 102, 2
!SGROUP, SGRP= EA01           --- "(601,1), (602,2)"가 그룹 "SUF01"에 추가됩니다.
601, 1
602, 2
```

 잘못된 예

 예 1 [(요소, 국소 표면 번호) 쌍이 여러 줄에 걸쳐 있는 경우]

 
```text
!SGROUP, SGRP= SUF01
101, 1, 102, 1, 103
1, 104, 1
```

 예 2 [국소 표면 번호가 요소 유형과 일치하지 않는 경우]

 
```text
!ELEMENT, TYPE= 211, SECTION= A
101, 1, 2, 3
102, 2, 3, 4
...
!SGROUP, SGRP= SUF01
101, 1
101, 2
101, 4              --- 삼각형 요소에는 네 번째 표면이 없으므로 이 조합은 무시됩니다.
```


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/msh/zero.md | TITLE: !ZERO -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/msh/zero.html -->

# !ZERO

 절대 영도

 첫 번째 행

 
```text
!ZERO
```

 매개변수

 없음

 두 번째 행부터

 
```text
(2번째 행) ZERO
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| `ZERO` | R | 절대 영도 |

 절대 영도는 복사 경계 조건(제어 파일의 [`!RADIATE`](../cnt/radiate.ja.md) / [`!SRADIATE`](../cnt/sradiate.ja.md))에서 입력 온도를 절대온도로 변환하는 데 사용한다. 입력 온도의 단위에 따라 다음 값을 지정한다.

 
| 입력 온도 단위 | `ZERO` |
| --- | --- |
| 섭씨(°C) | `-273.15`(기본값) |
| 켈빈(K) | `0.0` |
| 화씨(°F) | `-459.67` |

 복사 계수 등 다른 입력값도 입력 온도와 동일한 단위계로 지정해야 한다.

 주의

 
- 생략할 수 있다. 생략하면 “절대 영도 = -273.15”(섭씨 온도 입력)가 된다.
- 켈빈 온도로 입력하는 경우 “`!ZERO`”에 `0.0`을 명시해야 한다.
- “`!ZERO`”가 정의되지 않은 경우 기본값을 사용했다는 경고 메시지가 표시된다.
- “`!ZERO`”를 여러 번 정의하면 내용이 갱신되고 경고 메시지가 표시된다.

 사용 예

 
```text
!ZERO
-273.16
```

 
## 관련 항목

 
- [`!RADIATE`](../cnt/radiate.ja.md)
- [`!SRADIATE`](../cnt/sradiate.ja.md)
- [`!HEAT`](../cnt/heat.ja.md)


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/part/index.md | TITLE: 영역 분할 (part) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/part/index.html -->

# 영역 분할 (part)

 영역 분할에 사용하는 키워드 목록입니다.


---


<!-- ========================================== -->
<!-- FILE: keyword_reference/part/partition.md | TITLE: !PARTITION -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/keyword_reference/part/partition.html -->

# !PARTITION

 
## 첫 번째 행

 
```text
!PARTITION, TYPE=<type>, METHOD=<method>, DOMAIN=<domain> [, 선택적 매개변수]
```

 
| 매개변수 |  |
| --- | --- |
| TYPE | 영역 분할 유형(필수) |
| METHOD | 영역 분할 방법(필수) |
| DOMAIN | 영역 분할 수(필수) |
| DEPTH | 오버랩 깊이(생략 가능) |
| UCD | 영역 분할 이미지용 UCD 파일 출력(생략 가능) |
| CONTACT | 영역 분할에서 접촉점 처리(생략 가능) |
| CONTACT_OWNER | 병렬 접촉의 소유 방식(생략 가능) |

 
| 매개변수 | 매개변수 값 | 내용 |
| --- | --- | --- |
| TYPE | NODE-BASED | 절점 기반 분할 |
|  | ELEMENT-BASED | 요소 기반 분할 |
| METHOD | RCB | RCB법, 두 번째 행에 분할 기준축 지정이 필수 |
|  | KMETIS | kMETIS |
|  | PMETIS | pMETIS |
| DOMAIN |  | 영역 분할 수 |
| DEPTH |  | 부분영역의 오버랩 깊이(생략 시 DEPTH=1)TYPE=ELEMENT-BASED일 때는 지정 불가 |
| UCD |  | UCD 파일 이름(생략 가능) |
| CONTACT | DEFAULT | 접촉점을 고려하지 않고 일반 영역 분할을 수행(SIMPLE과 동일하게 처리) |
|  | AGGREGATE | 마스터 면과 슬레이브 면을 같은 부분영역에 할당 |
|  | DISTRIBUTE | 마스터 면의 절점을 각 부분영역에 균등하게 분배 |
|  | SIMPLE | 접촉점을 고려하지 않고 일반 영역 분할을 수행(DEFAULT와 동일하게 처리) |
| CONTACT_OWNER | MASTER | 마스터 소유 방식(기본값). 마스터 면을 요소 소유 영역별로 분할하고 슬레이브 절점을 각 마스터 소유 영역에 복제한다 |
|  | SLAVE | 슬레이브 소유 방식. 각 슬레이브 절점은 해당 절점을 소유한 영역에만 유지하고, 그 영역에 마스터 면 전체를 배치한다TYPE=NODE-BASED일 때만 지정 가능 |

 `CONTACT`와 `CONTACT_OWNER`는 서로 독립적인 매개변수이며 임의로 조합할 수 있다.

 
### CONTACT_OWNER

 병렬 접촉 해석에서 어떤 영역이 접촉 쌍의 어느 부분을 담당하는지 지정한다.

 `MASTER`(기본값)는 마스터 면을 요소 소유 영역별로 분할하고 슬레이브 절점을 마스터 면을 가진 모든 영역에 복제한다. 중복된 접촉 후보는 영역 간 집계로 조정된다.

 `SLAVE`는 이 구성을 반대로 한다. 각 슬레이브 절점은 해당 절점을 소유한 영역에만 유지되며, 그 영역은 접촉 쌍의 마스터 면 전체를 받는다. 유한 미끄럼(`!CONTACT`의 `INTERACTION=FSLID`)에서는 슬레이브 절점이 미끄러져 마스터 면의 영역 분할 경계를 넘을 수 있다. `MASTER`에서는 마스터 면의 인접 관계가 각 영역 내부에서만 구축되므로 경계에서 인접 탐색이 끊길 수 있고, 그 결과 슬레이브 절점이 일시적으로 비접촉으로 판정되어 Lagrange 승수와 마찰 이력이 손실된다. `SLAVE`에서는 탐색에 필요한 마스터 면이 항상 같은 영역에 있으므로 이 문제가 발생하지 않으며 해가 영역 분할 수에 의존하지 않는다.

 `SLAVE`를 지정한 영역은 마스터 면 전체를 유지하므로 해당 영역의 메모리 사용량과 인접 영역 수가 증가한다. `TYPE=ELEMENT-BASED`와 함께 사용할 수 없다.

 
## 두 번째 행(METHOD=RCB인 경우에만 필수)

 
```text
(두 번째 행) DIR1, DIR2, DIR3, ... (분할 횟수만큼 지정)
```

 
| 변수명 | 유형 | 설명 |
| --- | --- | --- |
| DIRX | C | 분할 기준축(소문자 x, y, z로 지정) |

 
## 제어 파일 예

 예 1

 
```text
!PARTITION, TYPE=NODE-BASED, METHOD=RCB, DOMAIN=8, DEPTH=1, UCD=mesh.inp
x, y, z
```

 예 2

 
```text
!PARTITION, TYPE=ELEMENT-BASED, METHOD=PMETIS, DOMAIN=32
```

 예 3 (유한 미끄럼 접촉 해석)

 
```text
!PARTITION, TYPE=NODE-BASED, METHOD=PMETIS, DOMAIN=8, CONTACT_OWNER=SLAVE
```


---

