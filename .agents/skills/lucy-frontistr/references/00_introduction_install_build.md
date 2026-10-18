# FrontISTR 소개, 설치 및 빌드 가이드

> 크롤링 출처: `https://docs.frontistr.com/ko/` (수록 페이지 수: 13개)

## 수록 문서 목차

- `build_guide/cmake_build.md` — CMake를 이용한 빌드 절차
- `build_guide/dependencies.md` — 필수 및 선택 의존 라이브러리
- `build_guide/gpu_build.md` — GPU 지원 빌드
- `build_guide/index.md` — 빌드
- `index.md` — FrontISTR 문서
- `install_guide/binary_install.md` — 바이너리 패키지를 사용한 설치
- `install_guide/docker.md` — Docker 이미지 사용
- `install_guide/index.md` — 설치
- `introduction/about.md` — FrontISTR 소개
- `introduction/ai_collaboration.md` — AI 연계
- `introduction/environment.md` — 동작 환경
- `introduction/features.md` — 주요 기능과 지원 해석
- `introduction/workflow_guide.md` — 전처리/후처리 및 실행 환경

---


<!-- ========================================== -->
<!-- FILE: build_guide/cmake_build.md | TITLE: CMake를 이용한 빌드 절차 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/build_guide/cmake_build.html -->

# CMake를 이용한 빌드 절차

 이 페이지에서는 FrontISTR 소스 코드의 입수부터 빌드·설치까지의 절차를 설명한다. 의존 라이브러리의 자세한 내용은 [필수·선택 의존 라이브러리](dependencies.html)를 참조한다.

 
## 소스 코드 입수

 FrontISTR 소스 코드는 GitLab 또는 GitHub에서 입수할 수 있다.

 
```sh
# GitLab
git clone https://gitlab.com/FrontISTR-Commons/FrontISTR.git

# GitHub(미러)
git clone https://github.com/FrontISTR/FrontISTR.git
```

 
## 빌드 절차

 FrontISTR는 아웃오브소스 빌드를 전제로 한다(소스 트리 바로 아래에서의 빌드는 금지되어 있다). `build` 디렉터리를 만들고 그 안에서 cmake를 실행한다.

 
```sh
cd FrontISTR
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_MPI=ON -DWITH_OPENMP=ON ..
make -j$(nproc)
```

 빌드 후에는 `/path/to/FrontISTR/build/fistr1/fistr1`로 직접 실행할 수 있다. 시스템 경로(`CMAKE_INSTALL_PREFIX`)에 설치하는 경우에는 추가로 다음을 실행한다.

 
```sh
cmake --build . --target install
```

 
## CMake 옵션 목록

 「검출 시 ON」으로 기재된 옵션은 cmake 실행 시 시스템에서 해당 라이브러리가 발견되면 자동으로 활성화된다. 명시적으로 `-DWITH_XXX=OFF`를 지정하면 비활성화할 수 있다.

 
| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `WITH_MPI` | `ON` | MPI 병렬을 활성화한다 |
| `WITH_OPENMP` | 검출 시 ON | OpenMP 스레드 병렬을 활성화한다 |
| `WITH_METIS` | 검출 시 ON | METIS에 의한 영역 분할을 활성화한다 |
| `WITH_MUMPS` | 검출 시 ON | MUMPS 병렬 직접법 솔버를 활성화한다 |
| `WITH_MKL` | 검출 시 ON | Intel MKL(PARDISO 포함)을 활성화한다 |
| `WITH_LAPACK` | 검출 시 ON | LAPACK에 의한 조건수 추정을 활성화한다 |
| `WITH_ML` | 검출 시 ON | Trilinos ML에 의한 AMG 전처리를 활성화한다 |
| `WITH_REFINER` | 검출 시 ON | REVOCAP_Refiner에 의한 메시 세분화를 활성화한다 |
| `WITH_REVOCAP` | 검출 시 ON | REVOCAP_Coupler에 의한 연성 해석을 활성화한다 |
| `WITH_TOOLS` | `ON` | 부속 도구 모음을 빌드한다 |
| `WITH_DOC` | `OFF` | Doxygen으로 API 문서를 생성한다 |
| `WITH_NETCDF` | 검출 시 ON | Exodus II 형식 출력(NetCDF)을 활성화한다 |
| `WINDOWS` | `OFF` | Windows용 빌드를 활성화한다 |
| `CMAKE_BUILD_TYPE` | `RELEASE` | 빌드 유형(`RELEASE` / `DEBUG`) |
| `CMAKE_INSTALL_PREFIX` | `/usr/local` | 설치 대상 디렉터리 |
| `DEBUG_EXTRA` | `OFF` | 메모리 누수 검사를 활성화한다(GCC용, `-fsanitize=address`) |
| `NVHPC_NO_ACC` | `OFF` | NVIDIA HPC SDK 컴파일러를 사용할 때 OpenACC를 비활성화하고 CPU 버전을 빌드한다 |

 
## DEBUG 빌드

 `CMAKE_BUILD_TYPE=DEBUG`를 지정하면 컴파일러별로 다음 디버그 플래그가 추가된다.

 
| 컴파일러 | 추가 플래그 |
| --- | --- |
| GCC | `-fbounds-check -fbacktrace` |
| Intel | `-check uninit,bound -traceback` |

 
```sh
cmake -DCMAKE_BUILD_TYPE=DEBUG ..
```

 
## GPU를 사용한 빌드

 NVIDIA HPC SDK(`nvc`/`nvc++`/`nvfortran`)를 컴파일러로 지정한 경우 CMake는 자동으로 다음 플래그를 추가한다.

 
| 플래그 | 설명 |
| --- | --- |
| `-acc` | OpenACC를 활성화한다 |
| `-Minfo=accel` | GPU 오프로딩 정보를 컴파일 시 표시한다 |
| `-gpu=ccnative,mem:managed` | 빌드 환경의 GPU를 자동 검출하여 타깃을 결정하고 Unified Memory를 사용한다 |

 `-DNVHPC_NO_ACC=ON`을 지정하면 이러한 플래그는 추가되지 않는다.

 OpenACC를 이용한 GPU 빌드의 자세한 절차는 [GPU를 사용한 빌드](gpu_build.html)를 참조한다.


---


<!-- ========================================== -->
<!-- FILE: build_guide/dependencies.md | TITLE: 필수 및 선택 의존 라이브러리 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/build_guide/dependencies.html -->

# 필수 및 선택 의존 라이브러리

 이 페이지에서는 FrontISTR 빌드에 필요한 라이브러리와 기능을 확장하는 선택 라이브러리를 설명합니다. 각 라이브러리의 설치 절차는 해당 라이브러리에서 제공하는 문서를 참조하십시오.

 
## 필수 의존 라이브러리

 다음 도구와 라이브러리는 FrontISTR 빌드에 필수입니다.

 
| 라이브러리 | 버전 요구 사항 | 용도 |
| --- | --- | --- |
| [CMake](https://cmake.org/) | ≥ 3.13.4 | 빌드 시스템 |
| C/C++/Fortran 컴파일러 | — | 소스 코드 컴파일 |
| [MPI](https://www.mpi-forum.org/) | — | 병렬 통신 라이브러리(OpenMPI, IntelMPI 등) |

 
## 선택 의존 라이브러리

 다음 라이브러리는 선택 사항이며 필요한 기능에 따라 빌드 시 활성화할 수 있습니다. 해당 CMake 옵션은 [CMake를 사용한 빌드 절차](cmake_build.html)를 참조하십시오.

 
| 라이브러리 | CMake 옵션 | 용도 | 입수처 |
| --- | --- | --- | --- |
| [METIS](https://github.com/KarypisLab/METIS) (v5) | `WITH_METIS` | 그래프 분할을 이용한 영역 분할 | Karypis Lab |
| [MUMPS](https://mumps-solver.org/) + [ScaLAPACK](https://www.netlib.org/scalapack/) | `WITH_MUMPS` | 병렬 직접법 솔버 | MUMPS-Solver.org / Netlib |
| [Intel oneAPI MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | `WITH_MKL` | 직접법 솔버(PARDISO), BLAS/LAPACK | Intel |
| [OpenBLAS](https://www.openblas.net/) / [LAPACK](https://www.netlib.org/lapack/) | `WITH_LAPACK` | 수치 선형대수(조건수 추정) | OpenBLAS.net / Netlib |
| [Trilinos ML](https://trilinos.github.io/) | `WITH_ML` | AMG 전처리 | Trilinos Project |
| REVOCAP_Refiner | `WITH_REFINER` | 메시 세분화 | [FrontISTR 다운로드 페이지](https://www.frontistr.com/download/) |
| REVOCAP_Coupler | `WITH_REVOCAP` | 연성 해석 | [FrontISTR 다운로드 페이지](https://www.frontistr.com/download/) |
| [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) | `WITH_NETCDF` | Exodus II 형식 출력 | Unidata |
| [Doxygen](https://www.doxygen.nl/) | `WITH_DOC` | API 문서 생성 | Doxygen.nl |

 
## GPU 빌드에 필요한 추가 도구

 FrontISTR의 GPU 가속은 OpenACC를 사용하여 구현되어 있으며, 현재는 NVIDIA HPC SDK를 사용한 구성에서 동작이 확인되어 있습니다. GPU 빌드에는 위의 필수 라이브러리 외에 다음 도구가 필요합니다.

 
| 도구 | 용도 | 입수처 |
| --- | --- | --- |
| [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) (`nvc`, `nvc++`, `nvfortran`) | OpenACC GPU 오프로딩 컴파일 | NVIDIA |

 지원되지 않는 라이브러리 처리 방법을 포함한 GPU 빌드 절차의 자세한 내용은 [GPU를 사용한 빌드](gpu_build.html)를 참조하십시오.


---


<!-- ========================================== -->
<!-- FILE: build_guide/gpu_build.md | TITLE: GPU 지원 빌드 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/build_guide/gpu_build.html -->

# GPU 지원 빌드

 FrontISTR는 NVIDIA HPC SDK의 OpenACC 기능을 사용하여 GPU에서 실행할 수 있습니다. 이 페이지에서는 GPU 지원을 활성화하여 FrontISTR를 빌드하고 실행하는 방법을 설명합니다.

 필수 라이브러리는 [필수 및 선택 의존 라이브러리](dependencies.html)를 참조하십시오.

 
## 빌드 절차

 NVIDIA HPC SDK 컴파일러(`nvc`, `nvc++`, `nvfortran`)를 사용합니다. HPC 클러스터 환경에서는 환경 매뉴얼에 따라 빌드 전에 NVIDIA HPC SDK를 로드해야 합니다(예: `module load nvhpc`).

 METIS, MUMPS, MKL과 같은 선택 라이브러리는 GPU를 지원하지 않으므로 아래 명령 예에서는 비활성화합니다(예: `WITH_METIS=OFF`).

 [CMake를 사용한 빌드](cmake_build.html#gpu를-사용한-빌드)에 설명된 것처럼 NVIDIA HPC SDK(`nvc`, `nvc++`, `nvfortran`)를 컴파일러로 지정하면 CMakeLists.txt에서 `-gpu=ccnative,mem:managed`가 설정됩니다. `ccnative`는 빌드 환경의 GPU를 자동 감지해 최적의 타깃을 선택하는 옵션이므로 환경별로 수동 변경할 필요가 없습니다. `mem:`은 GPU 메모리 처리 방식을 지정하는 옵션이며, FrontISTR의 GPU 구현이 Unified Memory를 전제로 하기 때문에 필요합니다. `mem:managed`는 CPU와 GPU의 메모리가 분리된 일반적인 환경의 기본 설정입니다.

 
> **참고**: Miyabi(GH200)처럼 CPU와 GPU가 NVLink로 통합된 아키텍처에서는 `mem:managed` 대신 `mem:unified`를 사용합니다. 사용 환경에 맞게 설정을 변경하십시오(`-gpu=ccnative,mem:unified`).

 
### OpenACC + OpenMP (MPI 없음)

 
```sh
mkdir build
cd build
CC=nvc CXX=nvc++ FC=nvfortran cmake \
  -DWITH_MPI=OFF -DWITH_OPENMP=ON \
  -DWITH_METIS=OFF -DWITH_MUMPS=OFF \
  -DWITH_MKL=OFF -DWITH_LAPACK=OFF \
  -DWITH_ML=OFF -DWITH_REFINER=OFF -DWITH_REVOCAP=OFF \
  ..
make -j
```

 
### OpenACC + OpenMP + MPI

 MPI를 사용할 경우 MPI 래퍼 컴파일러(예: `mpicc`)가 내부적으로 NVIDIA HPC SDK 컴파일러(예: `nvc`)를 호출하도록 구성해야 합니다. 많은 HPC 클러스터 환경에서는 `module load` 등의 명령으로 NVIDIA HPC SDK와 MPI 라이브러리를 로드하면 이 구성이 자동으로 제공됩니다. 다만 권장 버전과 로드 절차는 환경마다 다르므로 자세한 내용은 각 환경의 사용자 매뉴얼 등의 문서를 참조하십시오.

 
```sh
mkdir build
cd build
CC=mpicc CXX=mpic++ FC=mpif90 cmake \
  -DWITH_MPI=ON -DWITH_OPENMP=ON \
  -DWITH_METIS=OFF -DWITH_MUMPS=OFF \
  -DWITH_MKL=OFF -DWITH_LAPACK=OFF \
  -DWITH_ML=OFF -DWITH_REFINER=OFF -DWITH_REVOCAP=OFF \
  ..
make -j
```

 
### OpenACC 미사용(CPU 버전)

 NVIDIA HPC SDK 컴파일러로 OpenACC를 사용하지 않는 CPU 버전을 빌드하려면 `-DNVHPC_NO_ACC=ON`을 지정합니다. 이때 `-acc`, `-Minfo=accel`, `-gpu=ccnative,mem:managed`는 설정되지 않습니다. 동일한 컴파일러로 빌드한 GPU 버전과 계산 결과를 비교할 때 사용합니다.

 
```sh
mkdir build
cd build
CC=nvc CXX=nvc++ FC=nvfortran cmake \
  -DNVHPC_NO_ACC=ON \
  -DWITH_MPI=OFF -DWITH_OPENMP=ON \
  ..
make -j
```

 
## 실행

 GPU 빌드의 실행 절차는 기본적으로 CPU 빌드와 동일합니다. 일반적인 실행 절차는 [실행 가이드](../execution_guide/overview/01_flow.html)를 참조하십시오.

 
## 프로파일링

 NVIDIA Nsight Systems(`nsys`)를 사용하여 프로파일링 데이터를 수집할 수 있습니다.

 
### MPI 없음

 
```sh
nsys profile -t cuda -s none -o profile_result /path/to/FrontISTR/build/fistr1/fistr1
```

 
### MPI 사용

 
```sh
nsys profile -t cuda,mpi -s none -o profile_result \
  mpiexec -n <프로세스수> /path/to/FrontISTR/build/fistr1/fistr1 -t 8
```


---


<!-- ========================================== -->
<!-- FILE: build_guide/index.md | TITLE: 빌드 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/build_guide/index.html -->

# 빌드

 이 절에서는 FrontISTR 를 소스 코드에서 빌드하는 절차를 설명한다.

 
| 페이지 | 내용 |
| --- | --- |
| [필수·선택 의존 라이브러리](dependencies.html) | 빌드에 필요한 라이브러리 목록과 입수처 |
| [CMake 를 이용한 빌드 절차](cmake_build.html) | 소스 코드 입수부터 CMake 설정, 컴파일까지의 절차 |
| [GPU 를 이용한 빌드](gpu_build.html) | NVIDIA HPC SDK 와 OpenACC 를 이용한 GPU 지원 빌드 |


---


<!-- ========================================== -->
<!-- FILE: index.md | TITLE: FrontISTR 문서 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/index.html -->

# FrontISTR 문서

 FrontISTR는 대규모 병렬 유한요소법 프로그램입니다. 이 문서에서는 설치부터 이론적 배경까지 FrontISTR 사용에 필요한 정보를 포괄적으로 제공합니다.

 
---

 **[소개](introduction/about.html)** — FrontISTR 프로젝트의 개요와 기본 정보를 소개합니다.

 **[설치](install_guide/binary_install.html)** — FrontISTR의 설치 절차를 설명합니다.

 **[빌드](build_guide/dependencies.html)** — 소스 코드에서 빌드하는 방법을 설명합니다.

 **[실행](execution_guide/overview/01_flow.html)** — FrontISTR의 실행 방법과 튜토리얼을 제공합니다.

 **[기능 레퍼런스](functions_reference/02_element.html)** — 요소, 재료 모델, 분할, 해석 스텝 등 FrontISTR의 각 기능을 설명합니다.

 **[키워드 레퍼런스](keyword_reference/input_rules.html)** — 해석 제어 데이터(cnt), 메시 데이터(msh), 전체 제어 데이터(dat), 분할(part)의 입력 키워드를 나열합니다.

 **[이론](theory/appendix_static_linear.html)** — 정적 선형·비선형 해석, 동적 해석, 모드 해석, 주파수 응답 해석, 열 해석의 이론적 배경을 설명합니다.

 **[예제집](examples/index.html)** — 탄성 정적 해석, 초탄성, 탄소성, 점탄성, 크리프, 접촉 해석, 동적 해석, 모드 해석, 열전도 해석, 주파수 응답 해석 등 17개의 예제를 수록하고 있습니다.

 **[전처리·후처리 및 실행 환경](introduction/workflow_guide.html)** — 전처리·후처리와 실행 환경의 조합(경로)을 선택하기 위한 가이드입니다.

 **[사례집](showcase/index.html)** — FrontISTR의 응용 사례를 소개합니다.

 **[V&V](verification_and_validation/01_simple.html)** — 검증 및 타당성 확인(Verification & Validation) 결과를 게재합니다.

 **[코딩 가이드](coding_guide/usersubroutine/01_overview.html)** — FrontISTR 개발에 참여할 때의 코딩 규약을 설명합니다.


---


<!-- ========================================== -->
<!-- FILE: install_guide/binary_install.md | TITLE: 바이너리 패키지를 사용한 설치 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/install_guide/binary_install.html -->

# 바이너리 패키지를 사용한 설치

 미리 빌드된 FrontISTR 바이너리는 [다운로드 페이지](https://www.frontistr.com/download/)에서 받을 수 있습니다. 배포물에는 솔버 본체 `fistr1` 외에도 메시 분할(`hecmw_part1`), 시각화(`hecmw_vis1`), 메시 변환(`neu2fstr`) 등의 부속 도구가 포함되어 있습니다.

 
## Windows

 Windows 버전은 병렬화 방식과 사용하는 수치 계산 라이브러리가 서로 다른 ZIP 파일로 배포됩니다. 모든 버전에는 실행에 필요한 MPI 런타임과 수치 계산 라이브러리(DLL)가 포함되어 있으므로 별도로 설치하지 않고 압축을 푼 뒤 사용할 수 있습니다.

 
| 버전 | 병렬화 방식 | MPI | 수치 계산 라이브러리 |
| --- | --- | --- | --- |
| thread_openblas | OpenMP 스레드 병렬 | (없음) | OpenBLAS |
| hybrid_msmpi_openblas | MPI + OpenMP 하이브리드 병렬 | MS-MPI(포함) | OpenBLAS |
| hybrid_impi_mkl_intelthread | MPI + OpenMP 하이브리드 병렬 | Intel MPI(포함) | Intel MKL |

 
> 참고: 하이브리드 버전에는 `mpiexec`와 MPI 런타임(`msmpi.dll` 또는 `impi.dll` 등)이 포함되어 있으므로 MPI를 별도로 설치할 필요가 없습니다.

 
### 설치 절차

 
1. 다운로드 페이지에서 사용할 버전의 ZIP 파일을 내려받아 압축을 풉니다.
2. 압축을 푼 디렉터리를 실행 파일 검색 경로에 포함된 위치에 두거나 해당 디렉터리에서 직접 해석을 실행합니다.

 
### 설치 확인

 명령 프롬프트에서 다음 명령을 실행합니다. 버전 정보가 표시되면 설치가 정상적으로 완료된 것입니다.

 
```sh
fistr1.exe -v
```

 MPI 병렬 실행 시에는 포함된 `mpiexec`를 사용합니다. 실행 절차에 대한 자세한 내용은 [해석 흐름](../execution_guide/overview/01_flow.html)을 참조하십시오.

 
## Linux

 Ubuntu용 deb 패키지가 배포됩니다. deb 패키지는 의존 라이브러리를 선언하므로 `apt-get`을 통해 설치하면 필요한 라이브러리가 자동으로 해결됩니다.

 
### deb 패키지 다운로드

 [다운로드 페이지](https://www.frontistr.com/download/)에서 Ubuntu용 deb 패키지를 직접 다운로드합니다. 명령줄에서 가져오려면 배포 버전에 맞는 파일 이름으로 바꾸어 다음과 같은 명령을 사용합니다.

 
```sh
curl -L -O https://frontistr-commons.gitlab.io/FrontISTR/release/deb/FrontISTR_master-0+ubuntu2204_amd64.deb
```

 
### deb 패키지 설치

 다운로드한 deb 패키지를 다음과 같이 설치합니다. 파일 이름은 배포 버전에 맞는 것으로 바꾸십시오.

 
```sh
sudo apt-get install -y ./FrontISTR_master-0+ubuntu2204_amd64.deb
```

 
> 참고: 패키지를 설치하려면 관리자 권한이 필요합니다. root 사용자로 작업 중인 경우 `sudo`는 필요하지 않습니다. 일반 사용자로 실행하는 경우 위와 같이 `sudo`를 붙이십시오.

 
### 설치 확인

 다음 명령을 실행합니다. 버전 정보가 표시되면 설치가 정상적으로 완료된 것입니다.

 
```sh
fistr1 -v
```

 
## 관련 항목

 
- [Docker 이미지 사용](docker.html)
- [빌드](../build_guide/index.html)
- [동작 환경](../introduction/environment.html)
- [해석 흐름](../execution_guide/overview/01_flow.html)


---


<!-- ========================================== -->
<!-- FILE: install_guide/docker.md | TITLE: Docker 이미지 사용 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/install_guide/docker.html -->

# Docker 이미지 사용

 공식 런타임 Docker 이미지를 사용하면 의존 라이브러리를 설치하지 않고 FrontISTR를 실행할 수 있습니다. Docker 자체의 설치 방법은 사용 중인 환경의 문서를 참조하십시오.

 
## 이미지 가져오기

 
```sh
docker pull registry.gitlab.com/frontistr-commons/frontistr/fistr1:latest
```

 이미지에는 실행에 필요한 다음 라이브러리가 포함되어 있습니다.

 
- MPI: Open MPI
- METIS
- MUMPS
- Trilinos ML
- Intel MKL (x86_64 환경)

 
## 해석 실행

 호스트의 작업 디렉터리(입력 파일이 있는 위치)를 컨테이너와 공유하여 해석을 실행합니다. `-v`로 디렉터리를 공유하고 `-w`로 작업 디렉터리를 지정합니다.

 
```sh
docker run --rm -v $PWD:/work -w /work \
  registry.gitlab.com/frontistr-commons/frontistr/fistr1:latest fistr1
```

 이미지의 기본 명령은 `fistr1`입니다(출처: `docker/container/Dockerfile`, `CMD ["fistr1"]`). 버전 확인 등의 옵션을 지정하려면 이미지 이름 뒤에 명령을 지정합니다.

 
```sh
docker run --rm registry.gitlab.com/frontistr-commons/frontistr/fistr1:latest fistr1 -v
```

 입력 파일 준비 및 제어 파일 작성에 대해서는 [해석 흐름](../execution_guide/overview/01_flow.html)을 참조하십시오.

 
## MPI 병렬 실행

 MPI 병렬 실행은 컨테이너 내부의 `mpirun`을 통해 수행합니다. 구체적인 실행 명령과 영역 분할 절차는 [병렬 해석](../execution_guide/procedure/02_parallel.html)을 참조하십시오.

 
## 관련 항목

 
- [바이너리 패키지를 사용한 설치](binary_install.html)
- [빌드](../build_guide/index.html)
- [동작 환경](../introduction/environment.html)
- [해석 흐름](../execution_guide/overview/01_flow.html)


---


<!-- ========================================== -->
<!-- FILE: install_guide/index.md | TITLE: 설치 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/install_guide/index.html -->

# 설치

 이 섹션에서는 배포되는 FrontISTR 바이너리와 Docker 이미지를 입수하여 사용하는 절차를 설명한다. 소스 코드에서 빌드하는 경우에는 [빌드](../build_guide/index.html)를 참조하기 바란다.

 
| 페이지 | 내용 |
| --- | --- |
| [바이너리 패키지를 이용한 설치](binary_install.html) | Windows·Linux용 배포 바이너리의 입수·배치·동작 확인 |
| [Docker 이미지 사용](docker.html) | 공식 Docker 이미지의 입수·실행·병렬 실행 |

 어떤 방법이 자신의 환경에 적합한지는 [동작 환경](../introduction/environment.html)을 참조하기 바란다.


---


<!-- ========================================== -->
<!-- FILE: introduction/about.md | TITLE: FrontISTR 소개 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/introduction/about.html -->

# FrontISTR 소개

 FrontISTR는 오픈 소스 **대규모 병렬 유한요소법(FEM) 구조 해석 소프트웨어**입니다. 재료 비선형, 대변형, 접촉을 포함한 비선형 변형 및 응력 해석뿐만 아니라 모드 해석, 주파수 응답 해석, 열전도 해석 등 산업 및 학술 실무에 필요한 해석 기능을 제공합니다. FrontISTR는 **MIT License**로 배포됩니다.

 FrontISTR는 노트북에서의 시험 해석부터 PC 클러스터 및 슈퍼컴퓨터에서의 대규모 병렬 계산까지 폭넓은 계산 환경에서 사용할 수 있습니다. 지원되는 해석 기능 목록은 [주요 기능 및 지원 해석](features.html)을 참조하십시오.

 
## 개발 및 운영

 FrontISTR는 대학, 연구기관, 기업으로 구성된 개발 커뮤니티인 일반사단법인 **FrontISTR Commons**가 지속적으로 개발하고 운영합니다. 소스 코드 공개, 개선 및 배포는 이 커뮤니티를 통해 이루어집니다. 커뮤니티 활동과 참여 방법은 [공식 웹사이트](https://www.frontistr.com)를 참조하십시오.

 
## 공식 정보

 
| 종류 | URL |
| --- | --- |
| 공식 웹사이트 | [https://www.frontistr.com](https://www.frontistr.com) |
| 공식 문서 | [https://docs.frontistr.com/ko/](https://docs.frontistr.com/ko/) (한국어) / [https://docs.frontistr.com/en/](https://docs.frontistr.com/en/) (English) |
| 개발 저장소 (GitLab) | [https://gitlab.com/FrontISTR-Commons/FrontISTR](https://gitlab.com/FrontISTR-Commons/FrontISTR) |
| 미러 (GitHub) | [https://github.com/FrontISTR/FrontISTR](https://github.com/FrontISTR/FrontISTR) |
| 다운로드 | [https://www.frontistr.com/ko/download/](https://www.frontistr.com/ko/download/) |

 GitLab이 주요 개발 플랫폼입니다. 버그 보고, 개선 제안, 코드 기여를 포함한 이슈 및 병합 요청도 이곳에서 접수합니다.

 
## 슈퍼컴퓨터에서의 이용

 FrontISTR는 Fugaku를 포함한 HPCI 공용 계산 자원에 사전 설치되어 있어 대규모 병렬 계산 환경을 제공합니다.

 
## 관련 항목

 
- [주요 기능 및 지원 해석](features.html)
- [동작 환경](environment.html)
- [전처리/후처리 및 실행 환경 가이드](workflow_guide.html)
- [설치](../install_guide/index.html)


---


<!-- ========================================== -->
<!-- FILE: introduction/ai_collaboration.md | TITLE: AI 연계 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/introduction/ai_collaboration.html -->

# AI 연계

 FrontISTR는 환경 구축, 해석 실행, 코드 개선 등 다양한 소프트웨어 관련 작업에 AI 에이전트를 결합하는 활용을 추진하고 있습니다. 모델 작성 지원 및 외부 도구와의 연계 등 AI를 활용할 수 있는 범위는 앞으로 더욱 확대될 것으로 예상됩니다. 이 페이지에서는 이러한 AI 연계 활동을 소개합니다.

 
## Fabrica — AI 에이전트 활용을 위한 워크숍

 **[FrontISTR-Fabrica](https://gitlab.com/FrontISTR-Commons/frontistr-fabrica)**는 FrontISTR 사용자와 개발자 모두를 대상으로 AI 에이전트를 활용하기 위한 워크숍(Fabrica)입니다. 작업별로 준비된 프롬프트 템플릿에 요구사항을 기입하여 AI 에이전트에 전달하면, AI가 작업 계획을 작성하고 필요에 따라 확인을 요청하면서 작업을 수행한 뒤 결과를 보고서로 정리합니다. 이를 통해 사용자는 결과 검토와 판단에 집중할 수 있습니다.

 템플릿을 사용하면 다음과 같은 작업을 대화형으로 수행할 수 있습니다.

 
- **환경 구축** — 빌드 설정 조사부터 컴파일 및 동작 확인까지
- **해석 실행** — 메시 확인, 입력 파일 작성, 솔버 실행 및 결과 정리
- **기능 추가 및 버그 수정** — 코드 조사부터 구현, 테스트 및 커밋까지
- **문서 작성** — 매뉴얼 작성 및 코드와의 일관성 확인
- **기여** — Issue 및 Merge Request 초안 작성과 fork로의 push 안내

 Fabrica를 사용하려면 FrontISTR를 빌드하고 실행할 수 있는 환경과 GitHub Copilot, Claude Code, Codex 등의 AI 에이전트가 필요합니다. 설치 절차, 사전 요구사항 및 각 템플릿의 사용 방법은 저장소의 [README](https://gitlab.com/FrontISTR-Commons/frontistr-fabrica/-/blob/main/README.md)를 참조하십시오.

 
> AI가 제안한 명령이나 코드를 채택할지에 대한 최종 판단은 사용자의 책임입니다. 템플릿은 해당 검토를 쉽게 하기 위한 것이며 AI 생성 출력의 정확성을 보장하지 않습니다.

 
## 관련 항목

 
- [FrontISTR 소개](about.html)
- [전처리/후처리 및 실행 환경](workflow_guide.html)
- [설치](../install_guide/index.html)


---


<!-- ========================================== -->
<!-- FILE: introduction/environment.md | TITLE: 동작 환경 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/introduction/environment.html -->

# 동작 환경

 여기에서는 FrontISTR를 사용하기 위한 지원 OS와 필수 요건을 개요 수준에서 설명합니다. 구체적인 도입 절차는 [설치](../install_guide/index.html)(배포 바이너리·Docker) 또는 [빌드](../build_guide/index.html)(소스에서 빌드)를 참조하십시오.

 
## 지원 OS

 
| OS | 이용 방법 |
| --- | --- |
| Linux | 배포 바이너리, Docker, 소스에서 빌드 |
| Windows | 배포 바이너리(MinGW판), Docker |
| macOS | Docker 사용을 권장 |

 
> 참고: macOS에서는 Docker 컨테이너 사용을 권장합니다.

 
## 필수 요건

 병렬 계산을 수행할 경우 다음 라이브러리가 필요합니다.

 
- **MPI 구현**: Open MPI / MPICH / Intel MPI / MS-MPI 중 하나
- **METIS**: 메시 분할에 사용

 배포 바이너리나 Docker 이미지를 사용하는 경우 이들은 함께 제공됩니다. 순차 계산(1개 영역)만 사용하는 경우 MPI는 필수가 아닙니다.

 
## 소스에서 빌드하는 경우

 소스에서 빌드하는 경우 C / C++ / Fortran90을 지원하는 컴파일러(GCC, Intel oneAPI, Clang 등)가 필요합니다. 필요한 라이브러리와 절차의 자세한 내용은 [빌드](../build_guide/cmake_build.html)를 참조하십시오.

 
## GPU를 사용하는 경우

 NVIDIA GPU에서 OpenACC를 이용한 병렬 계산을 실행할 수 있습니다. GPU용으로 빌드하려면 NVIDIA HPC SDK가 필요합니다. 절차의 자세한 내용은 [GPU를 이용한 빌드](../build_guide/gpu_build.html)를 참조하십시오.

 
## 관련 항목

 
- [설치](../install_guide/index.html)
- [빌드](../build_guide/index.html)
- [GPU를 이용한 빌드](../build_guide/gpu_build.html)
- [전처리·후처리 및 실행 환경 선택](workflow_guide.html)


---


<!-- ========================================== -->
<!-- FILE: introduction/features.md | TITLE: 주요 기능과 지원 해석 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/introduction/features.html -->

# 주요 기능과 지원 해석

 여기에서는 FrontISTR가 지원하는 해석 종류와 주요 기능을 개요 수준에서 정리합니다. 각 기능의 정식화, 입력 방법, 선택지의 자세한 내용은 기능 레퍼런스([해석 종류](../functions_reference/01_analysis_type.html) 등)를 참조하십시오.

 
## 지원하는 해석 종류

 
| 해석 종류 | 개요 |
| --- | --- |
| 정적 해석 | 선형·비선형(대변형·접촉·비선형 재료) 변형·응력 해석 |
| 동적 해석 | 암시적 해법·명시적 해법에 의한 동적 응답 해석 |
| 모드 해석 | 고유진동수·모드 형상의 계산 |
| 주파수 응답 해석 | 가진에 대한 주파수 응답 계산 |
| 열전도 해석 | 정상·비정상 열전도 해석 |

 비선형 재료로 탄소성, 초탄성, 크리프, 점탄성 등을 지원합니다. 자세한 내용은 [해석 종류](../functions_reference/01_analysis_type.html)를 참조하십시오.

 
## 병렬 계산

 MPI(분산 메모리 병렬)와 OpenMP(공유 메모리 병렬)를 지원하며, 두 방식을 결합한 하이브리드 병렬이 가능합니다. 이를 통해 단일 컴퓨터에서 대규모 슈퍼컴퓨터까지 계산 규모에 맞는 병렬 계산을 수행할 수 있습니다.

 
## 솔버·전처리

 
| 구분 | 지원 방식 |
| --- | --- |
| 반복법 솔버 | CG / BiCGSTAB / GMRES / GPBiCG |
| 직접법 솔버 | MUMPS / MKL PARDISO |
| 전처리 | SSOR / ILU(0) / AMG(Trilinos-ML) 등 |

 선택 방법과 설정의 자세한 내용은 [선형 솔버](../functions_reference/06_solver.html)를 참조하십시오.

 
## 요소 라이브러리

 1차·2차 솔리드 요소(사면체 TET, 오면체 PRISM, 육면체 HEX), 평면 요소(삼각형·사각형), 보 요소, 셸 요소, 트러스 요소 등을 지원합니다. 자세한 내용은 [요소](../functions_reference/02_element.html)를 참조하십시오.

 
## 관련 항목

 
- [해석 종류](../functions_reference/01_analysis_type.html)
- [요소](../functions_reference/02_element.html)
- [선형 솔버](../functions_reference/06_solver.html)
- [동작 환경](environment.html)


---


<!-- ========================================== -->
<!-- FILE: introduction/workflow_guide.md | TITLE: 전처리/후처리 및 실행 환경 -->
<!-- ========================================== -->

<!-- Source: https://docs.frontistr.com/ko/introduction/workflow_guide.html -->

# 전처리/후처리 및 실행 환경

 FrontISTR는 해석 솔버 자체입니다. 실제로 해석을 수행하려면 목적에 따라 전처리(메시 생성 및 경계 조건 설정)와 후처리(결과 시각화) 도구를 조합하여 사용합니다. 이 페이지에서는 대표적인 조합(루트)을 정리합니다. 자신에게 맞는 루트를 선택한 뒤 해당 도구의 다운로드 정보 또는 설치/실행 장으로 진행하십시오.

 
## 전체 해석 워크플로

 
```text
전처리 (메시 및 경계 조건 작성)
   ↓
솔버 실행 (FrontISTR)
   ↓
후처리 (결과 시각화)
```

 각 단계의 자세한 내용은 [해석 워크플로](../execution_guide/overview/01_flow.html)를 참조하십시오.

 
## 루트 선택

 FrontISTR와 함께 사용할 수 있는 주변 도구에는 프로젝트가 공식 배포하는 도구, FrontISTR 커뮤니티에서 지속적으로 소개하는 도구, 제3자가 독자적으로 연계한 도구가 있습니다. 아래 루트는 전처리/후처리 조합, 공식/서드파티 또는 상용 등의 위치, 지원 운영체제를 기준으로 정리했습니다.

 
| 루트 | 전처리 | 후처리(시각화) | 지원 OS | 특징 | 다운로드 / 상세 |
| --- | --- | --- | --- | --- | --- |
| REVOCAP_PrePost | REVOCAP_PrePost | REVOCAP_PrePost / ParaView | Windows | 전통적으로 공식 배포되는 GUI. 전처리부터 시각화까지의 워크플로를 지원 | [다운로드](https://www.frontistr.com/download/) / [비디오 튜토리얼](https://www.youtube.com/watch?v=sOBoHlOu6tA) |
| FreeCAD + FEM_FrontISTR | FreeCAD (FEM_FrontISTR 애드온) | FreeCAD / ParaView | Windows / Linux | FreeCAD와 통합된 공식 계열 GUI. CAD에서 일관된 워크플로를 제공하며 기능이 계속 확장 중 | [FEM_FrontISTR (GitHub)](https://github.com/FrontISTR/FEM_FrontISTR) |
| EasyISTR | EasyISTR (UNV 메시 가져오기) | ParaView | Windows / Linux | 서드파티 통합 지원 GUI. 일본어 문서가 풍부함 | [OpenCAE-DALAB Wiki](https://opencae-dalab.org/pukiwiki/?SoftEasyIstr) |
| 입력 파일 직접 편집 | 텍스트 편집기(CAD→Gmsh→필요에 따라 형식 변환, Femap neutral→neu2fstr) | ParaView | 멀티플랫폼 | 표준 루트. 모든 기능을 활용할 수 있으며 자동화와 재현성에 적합 | [해석 워크플로](../execution_guide/overview/01_flow.html) / [메시 및 데이터 변환](../functions_reference/12_data_conversion.html) |
| Keiei-Tensha | Keiei-Tensha (inp 가져오기 / FrontISTR 입력의 3D 확인 및 편집) | ParaView | Windows / Linux / macOS | 서드파티 VS Code 확장. 텍스트와 3D 뷰를 연계하고 선택 위치를 !SGROUP/!NGROUP으로 다시 기록하며 AI 에이전트 연계를 지원. 직접 편집 루트를 보조 | [Marketplace](https://marketplace.visualstudio.com/items?itemName=structia.keiei) / [structia.jp](https://structia.jp) |
| Jupiter (상용) | Jupiter | Jupiter | Windows | 상용, 일본 개발. 대규모 모델 지원 | [Jupiter-FrontISTR Interface](https://www.e-technostar.com/our-products/15508/) |
| WELSIM (상용) | WELSIM | WELSIM / ParaView | Windows / Linux | 상용, 일본 외 지역에서 개발. 여러 OSS 솔버 지원 | [welsim.com](https://welsim.com/) |

 
> 공개 배포되는 REVOCAP_PrePost에는 구세대 FrontISTR 솔버가 포함되어 있습니다. 최신 솔버로 해석할 때는 솔버를 별도로 입수하여 번들된 솔버를 교체하십시오.
>
>  또한 FrontISTR Commons 정회원에게 "Advance REVOCAP_PrePost for FrontISTR"가 제공됩니다(지원 없음). AdvanceSoft의 상용 전처리/후처리기 Advance/REVOCAP도 이용할 수 있습니다.

 
## 처음 사용하는 사용자

 FrontISTR를 처음 사용하는 경우 **FreeCAD + FEM_FrontISTR** 또는 **EasyISTR**처럼 전처리부터 시각화까지 모두 GUI에서 수행할 수 있는 루트로 시작하면 첫 해석을 완료하기 쉽습니다. 두 루트 모두 계속 활발히 업데이트되고 있으며 일본어 정보도 쉽게 구할 수 있습니다. 워크플로에 익숙해지면 FrontISTR 기능에 가장 확실하게 접근할 수 있는 "입력 파일 직접 편집" 루트로 이동하는 것이 좋습니다. 대규모 모델이나 상용 지원이 중요하다면 Jupiter와 WELSIM 같은 상용 도구도 선택지가 됩니다.

 
## 후처리(시각화)

 FrontISTR는 `!WRITE, VISUAL`로 시각화 출력을 활성화하고 `!VISUAL`(예: `output_type=VTK`)로 시각화 형식을 지정하여 ParaView 등의 도구에서 읽을 수 있는 비정형 격자 데이터(VTK / AVS 형식)를 출력합니다. 이 출력을 시각화 도구로 불러와 변위, 응력, 변형률 등의 컬러맵을 표시할 수 있습니다. 출력 형식과 변환 도구의 자세한 내용은 [메시 및 데이터 변환](../functions_reference/12_data_conversion.html)을 참조하십시오.

 
## 관련 항목

 
- [해석 워크플로](../execution_guide/overview/01_flow.html)
- [설치](../install_guide/index.html)
- [빌드](../build_guide/index.html)
- [메시 및 데이터 변환](../functions_reference/12_data_conversion.html)


---

