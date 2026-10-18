# FrontISTR 코딩 가이드 (요소 분류, 사용자 서브루틴 UMAT/UYIELD/ULOAD/UELASTIC) 및 릴리즈 노트

> 크롤링 출처: `https://docs.frontistr.com/ko/` (수록 페이지 수: 6개)

## 수록 문서 목차

- `coding_guide/01_element_taxonomy.md` — 요소 번호 체계와 형상 함수 라이브러리
- `coding_guide/usersubroutine/01_overview.md` — 사용자 서브루틴
- `coding_guide/usersubroutine/02_umat.md` — 사용자 정의 재료용 서브루틴 (umat.f)
- `coding_guide/usersubroutine/03_uyield.md` — 탄소성 변형 관련 서브루틴 (uyield.f90)
- `coding_guide/usersubroutine/04_uload.md` — 사용자 정의 외력 처리를 위한 서브루틴 (uload.f)
- `coding_guide/usersubroutine/05_uelastic.md` — 탄성 변형용 서브루틴 (uelastic.f90)

---


<!-- ========================================== -->
<!-- FILE: coding_guide/01_element_taxonomy.md | TITLE: 요소 번호 체계와 형상 함수 라이브러리 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/coding_guide/01_element_taxonomy.html -->

# 요소 번호 체계와 형상 함수 라이브러리

 [형상 함수와 유한요소 근사](../theory/discretization/01_shape_functions.html)에서 도입한 형상 함수 \(N_\alpha^e(\boldsymbol{r})\)는 요소 유형마다 구체적인 형태가 다릅니다. 이 장에서는 FrontISTR가 제공하는 요소군을 세 자리 요소 유형 번호로 분류하고, 각 요소를 동일한 방식으로 다루기 위한 형상 함수 라이브러리의 인터페이스를 설명합니다. 3차원 솔리드 요소의 구체적인 자연좌표계와 형상 함수 및 2차 요소의 중간 절점 순서 변환은 [3차원 솔리드 요소의 형상 함수](../theory/discretization/08_element_solid.html)에서 다루고, 요소 유형 선택 지침은 [요소 라이브러리(기능)](../functions_reference/02_element.html)에서 제공합니다.

 
## 요소 분류와 명명 규칙

 FrontISTR의 요소 유형은 세 자리 숫자로 식별됩니다. 각 자리의 의미는 다음과 같습니다.

 
- **첫째 자리(요소군)**: 1 = 선 요소/트러스, 2 = 평면 요소, 3 = 솔리드 요소, 5 = 인터페이스 요소, 6 = 보 요소, 7 = 셸 요소.
- **둘째 자리(기본 형상)**: 요소군 내의 기하학적 형상(삼각형, 사각형, 사면체, 프리즘, 육면체)을 구분합니다.
- **셋째 자리(보간 차수)**: 1 = 1차(꼭짓점 절점만), 2 = 2차(중간 절점 포함). 예외로 2절점 트러스 요소는 301번입니다.

 제공되는 요소를 요소군별로 정리하면 다음 표와 같습니다.

 
| 요소군 | 요소 유형 | 절점 수 | 설명 |
| --- | --- | --- | --- |
| 선 요소 | 111 | 2 | 2절점 링크 요소(응력 해석에는 사용 불가) |
|  | 112 | 3 | 3절점 링크 요소(응력 해석에는 사용 불가) |
| 평면 요소 | 231 | 3 | 3절점 1차 삼각형 요소 |
|  | 232 | 6 | 6절점 2차 삼각형 요소 |
|  | 241 | 4 | 4절점 1차 사각형 요소 |
|  | 242 | 8 | 8절점 2차 사각형 요소(Serendipity 계열) |
| 솔리드 요소 | 301 | 2 | 2절점 트러스 요소 |
|  | 341 | 4 | 4절점 1차 사면체 요소 |
|  | 342 | 10 | 10절점 2차 사면체 요소 |
|  | 351 | 6 | 6절점 1차 프리즘 요소 |
|  | 352 | 15 | 15절점 2차 프리즘 요소 |
|  | 361 | 8 | 8절점 1차 육면체 요소 |
|  | 362 | 20 | 20절점 2차 육면체 요소(Serendipity 계열) |
| 인터페이스 요소 | 541 | 4×2 | 4절점 1차 사각형 면 요소(응력 해석에는 사용 불가) |
|  | 542 | 8×2 | 8절점 2차 사각형 면 요소(응력 해석에는 사용 불가) |
| 보 요소 | 611 | 2 | 2절점 보 요소(절점당 6자유도, Bernoulli-Euler 보 또는 Timoshenko 보) |
|  | 641 | 2×2 | 2절점 Bernoulli-Euler 보 요소(절점당 3자유도, 혼합 자유도용) |
| 셸 요소 | 731 | 3 | MITC3 삼각형 셸 요소(절점당 6자유도) |
|  | 741 | 4 | MITC4 사각형 셸 요소(절점당 6자유도) |
|  | 743 | 9 | MITC9 사각형 셸 요소(절점당 6자유도) |
|  | 761 | 3×2 | MITC3 삼각형 셸 요소(절점당 3자유도, 혼합 자유도용) |
|  | 781 | 4×2 | MITC4 사각형 셸 요소(절점당 3자유도, 혼합 자유도용) |

 1차 육면체 요소 361에는 완전 적분, 비적합 모드, B-bar, F-bar 등 여러 정식화가 있으며 해석 제어에서 선택합니다. 자세한 내용은 [고급 요소 정식화](../theory/discretization/09_element_advanced.html)를 참조하십시오.

 보 요소 641과 셸 요소 761/781은 절점당 6자유도를 갖는 구조 요소를 동일한 절점 위치에서 절점당 3자유도를 갖는 솔리드 요소와 연결하기 위한 정식화입니다. 하나의 구조 절점은 두 개의 3자유도 절점으로 표현됩니다. 정식화 자체는 대응하는 6자유도 버전(611, 731, 741)과 동일합니다. 다만 641은 Bernoulli-Euler 보만 지원하며, 611에서 선택할 수 있는 Timoshenko 보는 지원하지 않습니다.

 
## 형상 함수 라이브러리

 요소 계산 루틴(강성 행렬 및 내력 벡터 구성)은 요소 유형마다 별도의 루틴을 두지 않습니다. 대신 요소 유형 번호 `fetype`을 인수로 받는 통일 인터페이스를 통해 형상 함수 라이브러리를 호출합니다. `elementInfo` 모듈(`fistr1/src/lib/element/element.f90`)은 다음 함수를 제공합니다.

 
- `NumOfQuadPoints(fetype)`: 해당 요소 유형의 적분점 수 \(n_q\)를 반환합니다.
- `getQuadPoint(fetype, np, pos)`: 적분점 \(i\)의 자연좌표 \(\boldsymbol{r}_i\)를 반환합니다.
- `getWeight(fetype, np)`: 적분점 \(i\)의 가중치 \(w_i\)를 반환합니다.
- `getShapeFunc(fetype, r, N)`: 자연좌표 \(\boldsymbol{r}\)에서의 형상 함수 값 \(N_\alpha^e(\boldsymbol{r})\)를 반환합니다.
- `getShapeDeriv(fetype, r, dN)`: 형상 함수의 자연좌표 미분 \(\partial N_\alpha^e/\partial \boldsymbol{r}\)를 반환합니다.
- `getGlobalDeriv(fetype, n_e, r, X^e, J, dN_x)`: 요소 절점 좌표 \(\boldsymbol{X}^e\)가 주어지면 물리좌표 미분 \(\partial N_\alpha^e/\partial \boldsymbol{x}\)와 야코비안 행렬식 \(J\)를 반환합니다.
- `getNumberOfSubface(fetype)`, `getSubFace(fetype, k, ...)`: 요소 경계면 수와 각 면을 구성하는 절점 번호를 반환합니다. 표면 하중과 접촉면 구성에 사용됩니다.

 각 함수는 내부적으로 요소 유형 번호에 따라 분기하고, 각 요소 유형에 제공되는 하위 모듈(`tet4n.f90`, `tet10n.f90`, `prism6n.f90`, `prism15n.f90`, `hex8n.f90`, `hex20n.f90`, `tri3n.f90`, `tri6n.f90`, `quad4n.f90`, `quad8n.f90`, `quad9n.f90`, `line2n.f90`, `line3n.f90`)의 형상 함수 및 형상 함수 미분 서브루틴을 호출합니다. 적분점 좌표와 가중치의 수치 표는 `quadrature.f90`에 통합되어 있습니다. 새 요소 유형을 추가하려면 대응하는 하위 모듈을 구현하고 `elementInfo`의 각 함수에 해당 분기를 추가합니다.

 
## 2차 삼각형 요소의 중간 절점 순서 변환

 2차 삼각형 요소 232에서는 HEC-MW 메시 입력 형식(요소 둘레의 꼭짓점 절점을 먼저 배치한 다음 변 1-2, 2-3, …을 따라 중간 절점을 배치)과 요소 라이브러리의 형상 함수가 전제로 하는 FrontISTR 내부 형식 사이에 중간 절점 순서 차이가 있습니다. 두 형식의 차이는 메시를 읽은 직후 `hecmw2fstr_mesh_conv`(`fistr1/src/common/hecmw2fstr_mesh_conv.f90` 및 `hecmw2fstr_connect_conv.c`)가 변환 테이블 `Table232`에 따라 6개 절점을 \(\{1,2,3,6,4,5\}\)로 재정렬하여 해소합니다. 변환은 요소 연결 정보를 저장하는 배열에 직접 적용되므로 이후의 모든 요소 라이브러리 호출에서는 FrontISTR 내부 절점 순서가 사용됩니다. 결과 출력 시에는 역변환 `fstr2hecmw_mesh_conv`가 쓰기 전에 HEC-MW 형식으로 되돌리므로, 사용자는 입력과 출력 모두 HEC-MW 형식만 인지하면 됩니다.

 2차 3차원 솔리드 요소 342(사면체)와 352(프리즘)에도 유사한 자동 변환이 있으며, 362(2차 육면체)는 변환하지 않습니다. 자세한 내용은 [3차원 솔리드 요소의 형상 함수](../theory/discretization/08_element_solid.html)를 참조하십시오. 2차 평면 사각형 요소 242와 인터페이스 요소 542는 HEC-MW 입력 형식과 내부 형식의 중간 절점 순서가 같으므로 변환하지 않습니다.

 
## 관련 항목

 
- [형상 함수와 유한요소 근사](../theory/discretization/01_shape_functions.html) — 형상 함수의 수학적 정의
- [수치 적분](../theory/discretization/05_numerical_integration.html) — 적분점, 가중치 및 요소 적분의 근사 형태
- [3차원 솔리드 요소의 형상 함수](../theory/discretization/08_element_solid.html) — 연속체 솔리드 요소의 구체적인 자연좌표와 형상 함수
- [고급 요소 정식화](../theory/discretization/09_element_advanced.html) — B-bar / F-bar / 비적합 모드 / 셸 / 보
- [요소 라이브러리(기능)](../functions_reference/02_element.html) — 요소 유형 옵션과 선택 지침


---


<!-- ========================================== -->
<!-- FILE: coding_guide/usersubroutine/01_overview.md | TITLE: 사용자 서브루틴 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/coding_guide/usersubroutine/01_overview.html -->

# 사용자 서브루틴

 사용자가 프로그래밍으로 FrontISTR의 기능을 확장할 수 있도록 인터페이스를 제공한다. 이러한 인터페이스는 기본적으로 서브루틴 헤더를 포함하는 FORTRAN 서브루틴으로, 입출력 변수의 설명과 이 변수들에 대한 선언문으로 구성된다. 루틴의 주요 부분은 사용자가 작성해야 한다.

 FrontISTR는 다음 사용자 서브루틴 인터페이스를 제공한다.

 
## 사용자 정의 재료 입력

 사용자 정의 재료를 사용하는 경우 최대 100개의 사용자 정의 재료 상수를 사용할 수 있다. 재료 상수는 다음과 같이 제어 데이터 파일에 한 행당 10개 수치, 최대 10행까지 입력할 수 있다.

 *두 번째 행~최대 10번째 행*

 v1, v2, v3, v4, v5, v6, v7, v8, v9, v10


---


<!-- ========================================== -->
<!-- FILE: coding_guide/usersubroutine/02_umat.md | TITLE: 사용자 정의 재료용 서브루틴 (umat.f) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/coding_guide/usersubroutine/02_umat.html -->

# 사용자 정의 재료용 서브루틴 (`umat.f`)

 탄성, 초탄성 또는 탄소성 여부와 관계없이 일반 재료의 변형 해석을 위한 인터페이스를 제공합니다. 사용자 정의 재료를 사용하려면 먼저 입력 파일에 `!USER_MATERIAL`을 설정하고 필요한 재료 상수를 입력한 다음 `uMatlMatrix` 및 `uUpdate` 서브루틴을 작성합니다.

 
## (1) 강성 행렬 계산 서브루틴

 
```text
subroutine uMatlMatrix( mname, matl, strain, stress, fstat, D, dtime, ttime, temperature )
        character(len=*), intent(in)  :: mname
        real(kind=kreal), intent(in)  :: matl(:)
        real(kind=kreal), intent(in)  :: strain(6)
        real(kind=kreal), intent(in)  :: stress(6)
        real(kind=kreal), intent(in)  :: fstat(:)
        real(kind=kreal), intent(out) :: D(:,:)
        real(kind=kreal), intent(in)  :: dtime
        real(kind=kreal), intent(in)  :: ttime
        real(kind=kreal), optional    :: temperature
```

 
- `mname`: 재료 이름
- `matl`: 재료 상수를 저장하는 배열(최대 100개)
- `strain`: Green-Lagrange 변형률
- `stress`: 제2 Piola-Kirchhoff 응력
- `fstat`: 상태 변수
- `D`: 구성 관계
- `dtime`: 시간 증분
- `ttime`: 현재 시간 증분 시작 시점의 총 시간
- `temperature`: 온도

 
## (2) 변형률 및 응력 갱신 서브루틴

 
```text
subroutine uUpdate(  mname, matl, strain, stress, fstat, dtime, ttime, temperature )
        character(len=\*), intent(in)    :: mname
        real(kind=kreal), intent(in)    :: matl(:)
        real(kind=kreal), intent(in)    :: strain(6)
        real(kind=kreal), intent(inout) :: stress(6)
        real(kind=kreal), intent(inout) :: fstat(:)
        real(kind=kreal), intent(in)    :: dtime
        real(kind=kreal), intent(in)    :: ttime
        real(kind=kreal), optional      :: temperature
```

 
- `mname`: 재료 이름
- `matl`: 재료 상수를 저장하는 배열(최대 100개)
- `strain`: 변형률
- `stress`: 제2 Piola-Kirchhoff 응력
- `fstat`: 상태 변수
- `dtime`: 시간 증분
- `ttime`: 현재 시간 증분 시작 시점의 총 시간
- `temperature`: 온도


---


<!-- ========================================== -->
<!-- FILE: coding_guide/usersubroutine/03_uyield.md | TITLE: 탄소성 변형 관련 서브루틴 (uyield.f90) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/coding_guide/usersubroutine/03_uyield.html -->

# 탄소성 변형 관련 서브루틴 (`uyield.f90`)

 탄소성 강성 행렬 및 응력의 리턴 매핑을 계산하기 위한 서브루틴을 제공한다. 사용자 정의 항복 함수를 사용하는 경우 먼저 입력 파일에 `!PLASTIC, YIELD=USER`를 설정하고 필요한 재료 상수를 입력한 뒤, 다음으로 함수 `uElastoPlasticNumStatus`와 서브루틴 `uElastoPlasticMatrix` 및 `uBackwardEuler` 을 작성해야 한다.

 
## (1) 실수형 상태 변수의 개수를 반환하는 함수

 
```text
integer function uElastoPlasticNumStatus( matl )
    real(kind=kreal),   intent(in)  :: matl(:)
```

 
- `matl`: 재료 상수를 저장하는 배열(1-100: 시스템 정의 재료 상수, 101-200: 사용자 정의 재료 상수)

 
## (2) 탄소성 강성 행렬 계산 서브루틴

 
```text
subroutine uElastoPlasticMatrix( matl, stress, istat, fstat, plstrain, D, temp, hdflag )
    real(kind=kreal),   intent(in)  :: matl(:)
    real(kind=kreal),   intent(in)  :: stress(6)
    integer(kind=kint), intent(in)  :: istat
    real(kind=kreal),   intent(in)  :: fstat(:)
    real(kind=kreal),   intent(in)  :: plstrain
    real(kind=kreal),   intent(out) :: D(:,:)
    real(kind=kreal),   intent(in)  :: temp
    integer(kind=kint), intent(in)  :: hdflag
```

 
- `matl`: 재료 상수를 저장하는 배열(1-100: 시스템 정의 재료 상수, 101-200: 사용자 정의 재료 상수)
- `stress`: 제2 Piola-Kirchhoff 응력
- `istat`: 정수형 상태 변수
- `fstat`: 실수형 상태 변수 배열
- `plstrain`: 현재 서브스텝 시작 시점의 소성 변형률
- `D`: 탄소성 행렬
- `temp`: 온도
- `hdflag`: 전체 성분(0), 편차 성분만(1), 체적 성분만(2) 계산

 
## (3) 응력 리턴 매핑 계산 서브루틴

 
```text
subroutine uBackwardEuler( matl, stress, plstrain, istat, fstat, temp, hdflag )
    real(kind=kreal),   intent(in)    :: matl(:)
    real(kind=kreal),   intent(inout) :: stress(6)
    real(kind=kreal),   intent(in)    :: plstrain
    integer(kind=kint), intent(inout) :: istat
    real(kind=kreal),   intent(inout) :: fstat(:)
    real(kind=kreal),   intent(in)    :: temp
    integer(kind=kint), intent(in)    :: hdflag
```

 
- `matl`: 재료 상수를 저장하는 배열(1-100: 시스템 정의 재료 상수, 101-200: 사용자 정의 재료 상수)
- `stress`: 시험 응력의 탄성 변형을 가정하여 얻은 제2 Piola-Kirchhoff 응력
- `plstrain`: 현재 서브스텝 시작 시점의 소성 변형률
- `istat`: 정수형 상태 변수
- `fstat`: 실수형 상태 변수 배열
- `temp`: 온도
- `hdflag`: 전체 성분(0), 편차 성분만(1), 체적 성분만(2) 계산


---


<!-- ========================================== -->
<!-- FILE: coding_guide/usersubroutine/04_uload.md | TITLE: 사용자 정의 외력 처리를 위한 서브루틴 (uload.f) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/coding_guide/usersubroutine/04_uload.html -->

# 사용자 정의 외력 처리를 위한 서브루틴 (`uload.f`)

 사용자 정의 외력을 처리하기 위한 인터페이스를 제공합니다.

 사용자 정의 외력을 사용하려면 먼저 외력을 정의하기 위한 수치 구조체 `tULoad`를 정의하고, 입력 파일에서 `!ULOAD, FILE=ファイル名`을 사용하여 파일명으로 지정된 파일에서 정의를 읽습니다. 그런 다음 아래 인터페이스를 사용하여 외력을 반영합니다.

 
## (1) 외력 읽기 서브루틴

 
```text
integer function ureadload( fname )
    character(len=\*), intent(in) :: fname
```

 
- `fname`: 외부 파일명. 이 파일에서 사용자 정의 외력을 읽습니다.

 
## (2) 외력을 전체 하중 벡터에 반영하는 서브루틴

 
```text
subroutine uloading( cstep, factor, exForce )
    integer, INTENT(IN) :: cstep
    REAL(KIND=kreal), INTENT(IN) :: factor
    REAL(KIND=kreal), INTENT(INOUT) :: exForce(:)
```

 
- `cstep`: 현재 해석 스텝 번호
- `factor`: 현재 스텝의 하중 계수
- `exForce`: 전체 하중 벡터

 
## (3) 잔류 응력 계산 서브루틴

 
```text
subroutine uResidual( cstep, factor, residual )
    integer, INTENT(IN) :: cstep
    REAL(KIND=kreal), INTENT(IN) :: factor
    REAL(KIND=kreal), INTENT(INOUT) :: residual(:)
```

 
- `cstep`: 현재 해석 스텝 번호
- `factor`: 현재 스텝의 하중 계수
- `residual`: 전체 잔차력 벡터


---


<!-- ========================================== -->
<!-- FILE: coding_guide/usersubroutine/05_uelastic.md | TITLE: 탄성 변형용 서브루틴 (uelastic.f90) -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/coding_guide/usersubroutine/05_uelastic.html -->

# 탄성 변형용 서브루틴 (`uelastic.f90`)

 탄성 및 초탄성 문제에서 탄성 강성 행렬을 계산하고 응력을 갱신하기 위한 서브루틴이 제공됩니다. 사용자 정의 탄성 또는 초탄성 구성 관계를 사용하려면 먼저 입력 파일에 `!ELASTIC, TYPE=USER` 또는 `!HYPERELASTIC, TYPE=USER`를 설정하고 필요한 재료 상수를 입력한 다음 `uElasticMatrix` 및 `uElasticUpdate` 서브루틴을 구현합니다.

 
## (1) 탄성 강성 행렬 계산 서브루틴

 
```text
subroutine uElasticMatrix( matl, strain, D )
    REAL(KIND=kreal), INTENT(IN) :: matl(:)
    REAL(KIND=kreal), INTENT(IN) :: strain(6)
    REAL(KIND=kreal), INTENT(OUT) :: D(6,6)
```

 
- `matl`: 재료 상수를 저장하는 배열(최대 100개)
- `strain`: Green-Lagrange 변형률
- `D`: 탄성 행렬

 
## (2) 응력 계산 서브루틴

 
```text
subroutine uElasticUpdate ( matl, strain, stress )
    REAL(KIND=kreal), INTENT(IN) :: matl(:)
    REAL(KIND=kreal), INTENT(IN) :: strain(6)
    REAL(KIND=kreal), INTENT(OUT) :: stress(6)
```

 
- `matl`: 재료 상수를 저장하는 배열(최대 100개)
- `strain`: Green-Lagrange 변형률
- `stress`: 응력


---

