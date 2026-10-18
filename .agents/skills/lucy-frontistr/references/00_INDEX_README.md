# FrontISTR 한국어 공식 문서 (전체 크롤링 아카이브)

- **원본 URL**: https://docs.frontistr.com/ko/
- **총 수집 페이지 수**: 205개
- **개별 문서 경로**: `D:\PythonCodeStudy\FrontISTR\docs_ko\<카테고리>\<문서명>.md`
- **Fabrica 연동 경로**: `D:\PythonCodeStudy\FrontISTR\frontistr-fabrica-main\frontistr-fabrica-main\frontistr_documents\docs\<카테고리>\<문서명>.ko.md`
- **통합 문서 경로**: `D:\PythonCodeStudy\FrontISTR\docs_ko\consolidated\`

## 카테고리별 통합 문서 (`consolidated/`)

| 파일명 | 설명 | 페이지 수 |
|:---|:---|---:|
| [`consolidated/00_introduction_install_build.md`](consolidated/00_introduction_install_build.md) | FrontISTR 소개, 설치 및 빌드 가이드 | 13 |
| [`consolidated/01_execution_guide.md`](consolidated/01_execution_guide.md) | FrontISTR 실행 가이드 및 입출력 튜토리얼 | 8 |
| [`consolidated/02_functions_reference.md`](consolidated/02_functions_reference.md) | FrontISTR 기능 레퍼런스 (해석종류, 요소, 재료, 경계/하중, 접촉, 솔버, 출력, 에러메시지) | 15 |
| [`consolidated/03_keyword_reference_dat_msh_part.md`](consolidated/03_keyword_reference_dat_msh_part.md) | FrontISTR 키워드 레퍼런스 — 입력규칙, 전체제어(.dat), 메시(.msh), 영역분할(.part) | 24 |
| [`consolidated/04_keyword_reference_cnt.md`](consolidated/04_keyword_reference_cnt.md) | FrontISTR 키워드 레퍼런스 — 해석제어(.cnt) 전체 키워드 사전 | 81 |
| [`consolidated/05_theory_reference.md`](consolidated/05_theory_reference.md) | FrontISTR 이론 매뉴얼 (지배방정식, 이산화, 구성칙, 접촉역학, 비선형 해법, 선형 솔버, 병렬화) | 33 |
| [`consolidated/06_examples_showcase_vv.md`](consolidated/06_examples_showcase_vv.md) | FrontISTR 예제집(1~17), 응용 사례집(Showcase), 검증 및 타당성 확인(V&V) | 25 |
| [`consolidated/07_coding_guide_and_release_notes.md`](consolidated/07_coding_guide_and_release_notes.md) | FrontISTR 코딩 가이드 (요소 분류, 사용자 서브루틴 UMAT/UYIELD/ULOAD/UELASTIC) 및 릴리즈 노트 | 6 |

## 전체 개별 문서 목록

| 상대 경로 (`.md`) | 문서 제목 | 원본 URL |
|:---|:---|:---|
| [`build_guide/cmake_build.md`](build_guide/cmake_build.md) | CMake를 이용한 빌드 절차 | https://docs.frontistr.com/ko/build_guide/cmake_build.html |
| [`build_guide/dependencies.md`](build_guide/dependencies.md) | 필수 및 선택 의존 라이브러리 | https://docs.frontistr.com/ko/build_guide/dependencies.html |
| [`build_guide/gpu_build.md`](build_guide/gpu_build.md) | GPU 지원 빌드 | https://docs.frontistr.com/ko/build_guide/gpu_build.html |
| [`build_guide/index.md`](build_guide/index.md) | 빌드 | https://docs.frontistr.com/ko/build_guide/index.html |
| [`coding_guide/01_element_taxonomy.md`](coding_guide/01_element_taxonomy.md) | 요소 번호 체계와 형상 함수 라이브러리 | https://docs.frontistr.com/ko/coding_guide/01_element_taxonomy.html |
| [`coding_guide/usersubroutine/01_overview.md`](coding_guide/usersubroutine/01_overview.md) | 사용자 서브루틴 | https://docs.frontistr.com/ko/coding_guide/usersubroutine/01_overview.html |
| [`coding_guide/usersubroutine/02_umat.md`](coding_guide/usersubroutine/02_umat.md) | 사용자 정의 재료용 서브루틴 (umat.f) | https://docs.frontistr.com/ko/coding_guide/usersubroutine/02_umat.html |
| [`coding_guide/usersubroutine/03_uyield.md`](coding_guide/usersubroutine/03_uyield.md) | 탄소성 변형 관련 서브루틴 (uyield.f90) | https://docs.frontistr.com/ko/coding_guide/usersubroutine/03_uyield.html |
| [`coding_guide/usersubroutine/04_uload.md`](coding_guide/usersubroutine/04_uload.md) | 사용자 정의 외력 처리를 위한 서브루틴 (uload.f) | https://docs.frontistr.com/ko/coding_guide/usersubroutine/04_uload.html |
| [`coding_guide/usersubroutine/05_uelastic.md`](coding_guide/usersubroutine/05_uelastic.md) | 탄성 변형용 서브루틴 (uelastic.f90) | https://docs.frontistr.com/ko/coding_guide/usersubroutine/05_uelastic.html |
| [`examples/index.md`](examples/index.md) | 예제 모음 | https://docs.frontistr.com/ko/examples/index.html |
| [`examples/tutorial_01.md`](examples/tutorial_01.md) | 선형 정적 해석 (탄성) | https://docs.frontistr.com/ko/examples/tutorial_01.html |
| [`examples/tutorial_02.md`](examples/tutorial_02.md) | 선형 정적 해석 (탄성, 병렬) | https://docs.frontistr.com/ko/examples/tutorial_02.html |
| [`examples/tutorial_03.md`](examples/tutorial_03.md) | 비선형 정적 해석 (초탄성 1) | https://docs.frontistr.com/ko/examples/tutorial_03.html |
| [`examples/tutorial_04.md`](examples/tutorial_04.md) | 비선형 정적 해석 (초탄성, 2부) | https://docs.frontistr.com/ko/examples/tutorial_04.html |
| [`examples/tutorial_05.md`](examples/tutorial_05.md) | 비선형 정적 해석 (탄소성 1) | https://docs.frontistr.com/ko/examples/tutorial_05.html |
| [`examples/tutorial_06.md`](examples/tutorial_06.md) | 비선형 정적 해석(탄소성, 2부) | https://docs.frontistr.com/ko/examples/tutorial_06.html |
| [`examples/tutorial_07.md`](examples/tutorial_07.md) | 비선형 정적 해석 (점탄성) | https://docs.frontistr.com/ko/examples/tutorial_07.html |
| [`examples/tutorial_08.md`](examples/tutorial_08.md) | 정적 해석(크리프) | https://docs.frontistr.com/ko/examples/tutorial_08.html |
| [`examples/tutorial_09.md`](examples/tutorial_09.md) | 접촉 해석 (1부) | https://docs.frontistr.com/ko/examples/tutorial_09.html |
| [`examples/tutorial_10.md`](examples/tutorial_10.md) | 접촉 해석(2부) | https://docs.frontistr.com/ko/examples/tutorial_10.html |
| [`examples/tutorial_11.md`](examples/tutorial_11.md) | 접촉 해석 (3부) | https://docs.frontistr.com/ko/examples/tutorial_11.html |
| [`examples/tutorial_12.md`](examples/tutorial_12.md) | 선형 동적 해석 | https://docs.frontistr.com/ko/examples/tutorial_12.html |
| [`examples/tutorial_13.md`](examples/tutorial_13.md) | 비선형 동적 해석 | https://docs.frontistr.com/ko/examples/tutorial_13.html |
| [`examples/tutorial_14.md`](examples/tutorial_14.md) | 비선형 접촉 동적 해석 | https://docs.frontistr.com/ko/examples/tutorial_14.html |
| [`examples/tutorial_15.md`](examples/tutorial_15.md) | 모드 해석 | https://docs.frontistr.com/ko/examples/tutorial_15.html |
| [`examples/tutorial_16.md`](examples/tutorial_16.md) | 열전도 해석 | https://docs.frontistr.com/ko/examples/tutorial_16.html |
| [`examples/tutorial_17.md`](examples/tutorial_17.md) | 주파수 응답 해석 | https://docs.frontistr.com/ko/examples/tutorial_17.html |
| [`execution_guide/data_conversion.md`](execution_guide/data_conversion.md) | 데이터 변환 도구 | https://docs.frontistr.com/ko/execution_guide/data_conversion.html |
| [`execution_guide/overview/01_flow.md`](execution_guide/overview/01_flow.md) | 해석 흐름 | https://docs.frontistr.com/ko/execution_guide/overview/01_flow.html |
| [`execution_guide/procedure/01_serial.md`](execution_guide/procedure/01_serial.md) | 직렬 해석 | https://docs.frontistr.com/ko/execution_guide/procedure/01_serial.html |
| [`execution_guide/procedure/02_parallel.md`](execution_guide/procedure/02_parallel.md) | 병렬 해석 | https://docs.frontistr.com/ko/execution_guide/procedure/02_parallel.html |
| [`execution_guide/tutorial_01/01_overview.md`](execution_guide/tutorial_01/01_overview.md) | 선형 정적 해석 (탄성) | https://docs.frontistr.com/ko/execution_guide/tutorial_01/01_overview.html |
| [`execution_guide/tutorial_01/02_dat.md`](execution_guide/tutorial_01/02_dat.md) | 전체 제어 데이터hecmw_ctrl.dat | https://docs.frontistr.com/ko/execution_guide/tutorial_01/02_dat.html |
| [`execution_guide/tutorial_01/03_msh.md`](execution_guide/tutorial_01/03_msh.md) | 메시 데이터 | https://docs.frontistr.com/ko/execution_guide/tutorial_01/03_msh.html |
| [`execution_guide/tutorial_01/04_cnt.md`](execution_guide/tutorial_01/04_cnt.md) | 해석 제어 데이터hinge.cnt | https://docs.frontistr.com/ko/execution_guide/tutorial_01/04_cnt.html |
| [`functions_reference/01_analysis_type.md`](functions_reference/01_analysis_type.md) | 해석 종류 | https://docs.frontistr.com/ko/functions_reference/01_analysis_type.html |
| [`functions_reference/02_element.md`](functions_reference/02_element.md) | 요소 라이브러리 | https://docs.frontistr.com/ko/functions_reference/02_element.html |
| [`functions_reference/03_material.md`](functions_reference/03_material.md) | 재료 데이터 | https://docs.frontistr.com/ko/functions_reference/03_material.html |
| [`functions_reference/04_boundary_load.md`](functions_reference/04_boundary_load.md) | 경계 조건과 하중 | https://docs.frontistr.com/ko/functions_reference/04_boundary_load.html |
| [`functions_reference/04b_mpc.md`](functions_reference/04b_mpc.md) | 다점 구속 | https://docs.frontistr.com/ko/functions_reference/04b_mpc.html |
| [`functions_reference/05_contact.md`](functions_reference/05_contact.md) | 접촉 및 임베딩 | https://docs.frontistr.com/ko/functions_reference/05_contact.html |
| [`functions_reference/06_solver.md`](functions_reference/06_solver.md) | 솔버·전처리 | https://docs.frontistr.com/ko/functions_reference/06_solver.html |
| [`functions_reference/06b_solution_algorithm.md`](functions_reference/06b_solution_algorithm.md) | 비선형 반복·시간 적분 | https://docs.frontistr.com/ko/functions_reference/06b_solution_algorithm.html |
| [`functions_reference/07_step.md`](functions_reference/07_step.md) | 스텝 제어 | https://docs.frontistr.com/ko/functions_reference/07_step.html |
| [`functions_reference/08_element_activation.md`](functions_reference/08_element_activation.md) | 요소 활성화 및 비활성화 | https://docs.frontistr.com/ko/functions_reference/08_element_activation.html |
| [`functions_reference/09_output_restart.md`](functions_reference/09_output_restart.md) | 출력·리스타트 | https://docs.frontistr.com/ko/functions_reference/09_output_restart.html |
| [`functions_reference/10_partition.md`](functions_reference/10_partition.md) | 영역 분할 | https://docs.frontistr.com/ko/functions_reference/10_partition.html |
| [`functions_reference/11_error_messages.md`](functions_reference/11_error_messages.md) | 오류 메시지 목록 | https://docs.frontistr.com/ko/functions_reference/11_error_messages.html |
| [`functions_reference/12_data_conversion.md`](functions_reference/12_data_conversion.md) | 메시 및 데이터 변환 | https://docs.frontistr.com/ko/functions_reference/12_data_conversion.html |
| [`functions_reference/13_user_extension.md`](functions_reference/13_user_extension.md) | 사용자 서브루틴 확장 | https://docs.frontistr.com/ko/functions_reference/13_user_extension.html |
| [`index.md`](index.md) | FrontISTR 문서 | https://docs.frontistr.com/ko/index.html |
| [`install_guide/binary_install.md`](install_guide/binary_install.md) | 바이너리 패키지를 사용한 설치 | https://docs.frontistr.com/ko/install_guide/binary_install.html |
| [`install_guide/docker.md`](install_guide/docker.md) | Docker 이미지 사용 | https://docs.frontistr.com/ko/install_guide/docker.html |
| [`install_guide/index.md`](install_guide/index.md) | 설치 | https://docs.frontistr.com/ko/install_guide/index.html |
| [`introduction/about.md`](introduction/about.md) | FrontISTR 소개 | https://docs.frontistr.com/ko/introduction/about.html |
| [`introduction/ai_collaboration.md`](introduction/ai_collaboration.md) | AI 연계 | https://docs.frontistr.com/ko/introduction/ai_collaboration.html |
| [`introduction/environment.md`](introduction/environment.md) | 동작 환경 | https://docs.frontistr.com/ko/introduction/environment.html |
| [`introduction/features.md`](introduction/features.md) | 주요 기능과 지원 해석 | https://docs.frontistr.com/ko/introduction/features.html |
| [`introduction/workflow_guide.md`](introduction/workflow_guide.md) | 전처리/후처리 및 실행 환경 | https://docs.frontistr.com/ko/introduction/workflow_guide.html |
| [`keyword_reference/cnt/acceleration.md`](keyword_reference/cnt/acceleration.md) | !ACCELERATION | https://docs.frontistr.com/ko/keyword_reference/cnt/acceleration.html |
| [`keyword_reference/cnt/autoinc_param.md`](keyword_reference/cnt/autoinc_param.md) | !AUTOINC_PARAM | https://docs.frontistr.com/ko/keyword_reference/cnt/autoinc_param.html |
| [`keyword_reference/cnt/boundary.md`](keyword_reference/cnt/boundary.md) | !BOUNDARY | https://docs.frontistr.com/ko/keyword_reference/cnt/boundary.html |
| [`keyword_reference/cnt/cflux.md`](keyword_reference/cnt/cflux.md) | !CFLUX | https://docs.frontistr.com/ko/keyword_reference/cnt/cflux.html |
| [`keyword_reference/cnt/cload.md`](keyword_reference/cnt/cload.md) | !CLOAD | https://docs.frontistr.com/ko/keyword_reference/cnt/cload.html |
| [`keyword_reference/cnt/contact.md`](keyword_reference/cnt/contact.md) | !CONTACT | https://docs.frontistr.com/ko/keyword_reference/cnt/contact.html |
| [`keyword_reference/cnt/contact_algo.md`](keyword_reference/cnt/contact_algo.md) | !CONTACT_ALGO | https://docs.frontistr.com/ko/keyword_reference/cnt/contact_algo.html |
| [`keyword_reference/cnt/contact_param.md`](keyword_reference/cnt/contact_param.md) | !CONTACT_PARAM | https://docs.frontistr.com/ko/keyword_reference/cnt/contact_param.html |
| [`keyword_reference/cnt/couple.md`](keyword_reference/cnt/couple.md) | !COUPLE | https://docs.frontistr.com/ko/keyword_reference/cnt/couple.html |
| [`keyword_reference/cnt/creep.md`](keyword_reference/cnt/creep.md) | !CREEP | https://docs.frontistr.com/ko/keyword_reference/cnt/creep.html |
| [`keyword_reference/cnt/damping.md`](keyword_reference/cnt/damping.md) | !DAMPING | https://docs.frontistr.com/ko/keyword_reference/cnt/damping.html |
| [`keyword_reference/cnt/dashpot_a.md`](keyword_reference/cnt/dashpot_a.md) | !DASHPOT_A | https://docs.frontistr.com/ko/keyword_reference/cnt/dashpot_a.html |
| [`keyword_reference/cnt/dashpot_d.md`](keyword_reference/cnt/dashpot_d.md) | !DASHPOT_D | https://docs.frontistr.com/ko/keyword_reference/cnt/dashpot_d.html |
| [`keyword_reference/cnt/density.md`](keyword_reference/cnt/density.md) | !DENSITY | https://docs.frontistr.com/ko/keyword_reference/cnt/density.html |
| [`keyword_reference/cnt/dflux.md`](keyword_reference/cnt/dflux.md) | !DFLUX | https://docs.frontistr.com/ko/keyword_reference/cnt/dflux.html |
| [`keyword_reference/cnt/dload.md`](keyword_reference/cnt/dload.md) | !DLOAD | https://docs.frontistr.com/ko/keyword_reference/cnt/dload.html |
| [`keyword_reference/cnt/dynamic.md`](keyword_reference/cnt/dynamic.md) | !DYNAMIC | https://docs.frontistr.com/ko/keyword_reference/cnt/dynamic.html |
| [`keyword_reference/cnt/echo.md`](keyword_reference/cnt/echo.md) | !ECHO | https://docs.frontistr.com/ko/keyword_reference/cnt/echo.html |
| [`keyword_reference/cnt/eigen.md`](keyword_reference/cnt/eigen.md) | !EIGEN | https://docs.frontistr.com/ko/keyword_reference/cnt/eigen.html |
| [`keyword_reference/cnt/eigenread.md`](keyword_reference/cnt/eigenread.md) | !EIGENREAD | https://docs.frontistr.com/ko/keyword_reference/cnt/eigenread.html |
| [`keyword_reference/cnt/elastic.md`](keyword_reference/cnt/elastic.md) | !ELASTIC | https://docs.frontistr.com/ko/keyword_reference/cnt/elastic.html |
| [`keyword_reference/cnt/element_activation.md`](keyword_reference/cnt/element_activation.md) | !ELEMENT_ACTIVATION | https://docs.frontistr.com/ko/keyword_reference/cnt/element_activation.html |
| [`keyword_reference/cnt/embed.md`](keyword_reference/cnt/embed.md) | !EMBED | https://docs.frontistr.com/ko/keyword_reference/cnt/embed.html |
| [`keyword_reference/cnt/end.md`](keyword_reference/cnt/end.md) | !END | https://docs.frontistr.com/ko/keyword_reference/cnt/end.html |
| [`keyword_reference/cnt/expansion_coeff.md`](keyword_reference/cnt/expansion_coeff.md) | !EXPANSION_COEFF | https://docs.frontistr.com/ko/keyword_reference/cnt/expansion_coeff.html |
| [`keyword_reference/cnt/film.md`](keyword_reference/cnt/film.md) | !FILM | https://docs.frontistr.com/ko/keyword_reference/cnt/film.html |
| [`keyword_reference/cnt/fixtemp.md`](keyword_reference/cnt/fixtemp.md) | !FIXTEMP | https://docs.frontistr.com/ko/keyword_reference/cnt/fixtemp.html |
| [`keyword_reference/cnt/fload.md`](keyword_reference/cnt/fload.md) | !FLOAD | https://docs.frontistr.com/ko/keyword_reference/cnt/fload.html |
| [`keyword_reference/cnt/fluid.md`](keyword_reference/cnt/fluid.md) | !FLUID | https://docs.frontistr.com/ko/keyword_reference/cnt/fluid.html |
| [`keyword_reference/cnt/heat.md`](keyword_reference/cnt/heat.md) | !HEAT | https://docs.frontistr.com/ko/keyword_reference/cnt/heat.html |
| [`keyword_reference/cnt/hyperelastic.md`](keyword_reference/cnt/hyperelastic.md) | !HYPERELASTIC | https://docs.frontistr.com/ko/keyword_reference/cnt/hyperelastic.html |
| [`keyword_reference/cnt/index.md`](keyword_reference/cnt/index.md) | 해석 제어 데이터 (cnt) | https://docs.frontistr.com/ko/keyword_reference/cnt/index.html |
| [`keyword_reference/cnt/initial_condition.md`](keyword_reference/cnt/initial_condition.md) | !INITIAL_CONDITION | https://docs.frontistr.com/ko/keyword_reference/cnt/initial_condition.html |
| [`keyword_reference/cnt/material.md`](keyword_reference/cnt/material.md) | !MATERIAL | https://docs.frontistr.com/ko/keyword_reference/cnt/material.html |
| [`keyword_reference/cnt/method.md`](keyword_reference/cnt/method.md) | !METHOD | https://docs.frontistr.com/ko/keyword_reference/cnt/method.html |
| [`keyword_reference/cnt/orientation.md`](keyword_reference/cnt/orientation.md) | !ORIENTATION | https://docs.frontistr.com/ko/keyword_reference/cnt/orientation.html |
| [`keyword_reference/cnt/output_res.md`](keyword_reference/cnt/output_res.md) | !OUTPUT_RES | https://docs.frontistr.com/ko/keyword_reference/cnt/output_res.html |
| [`keyword_reference/cnt/output_sstype.md`](keyword_reference/cnt/output_sstype.md) | !OUTPUT_SSTYPE | https://docs.frontistr.com/ko/keyword_reference/cnt/output_sstype.html |
| [`keyword_reference/cnt/output_vis.md`](keyword_reference/cnt/output_vis.md) | !OUTPUT_VIS | https://docs.frontistr.com/ko/keyword_reference/cnt/output_vis.html |
| [`keyword_reference/cnt/plastic.md`](keyword_reference/cnt/plastic.md) | !PLASTIC | https://docs.frontistr.com/ko/keyword_reference/cnt/plastic.html |
| [`keyword_reference/cnt/radiate.md`](keyword_reference/cnt/radiate.md) | !RADIATE | https://docs.frontistr.com/ko/keyword_reference/cnt/radiate.html |
| [`keyword_reference/cnt/reftemp.md`](keyword_reference/cnt/reftemp.md) | !REFTEMP | https://docs.frontistr.com/ko/keyword_reference/cnt/reftemp.html |
| [`keyword_reference/cnt/restart.md`](keyword_reference/cnt/restart.md) | !RESTART | https://docs.frontistr.com/ko/keyword_reference/cnt/restart.html |
| [`keyword_reference/cnt/section.md`](keyword_reference/cnt/section.md) | !SECTION | https://docs.frontistr.com/ko/keyword_reference/cnt/section.html |
| [`keyword_reference/cnt/sfilm.md`](keyword_reference/cnt/sfilm.md) | !SFILM | https://docs.frontistr.com/ko/keyword_reference/cnt/sfilm.html |
| [`keyword_reference/cnt/sflux.md`](keyword_reference/cnt/sflux.md) | !SFLUX | https://docs.frontistr.com/ko/keyword_reference/cnt/sflux.html |
| [`keyword_reference/cnt/solution.md`](keyword_reference/cnt/solution.md) | !SOLUTION | https://docs.frontistr.com/ko/keyword_reference/cnt/solution.html |
| [`keyword_reference/cnt/solver.md`](keyword_reference/cnt/solver.md) | !SOLVER | https://docs.frontistr.com/ko/keyword_reference/cnt/solver.html |
| [`keyword_reference/cnt/spring.md`](keyword_reference/cnt/spring.md) | !SPRING | https://docs.frontistr.com/ko/keyword_reference/cnt/spring.html |
| [`keyword_reference/cnt/spring_a.md`](keyword_reference/cnt/spring_a.md) | !SPRING_A | https://docs.frontistr.com/ko/keyword_reference/cnt/spring_a.html |
| [`keyword_reference/cnt/spring_d.md`](keyword_reference/cnt/spring_d.md) | !SPRING_D | https://docs.frontistr.com/ko/keyword_reference/cnt/spring_d.html |
| [`keyword_reference/cnt/sradiate.md`](keyword_reference/cnt/sradiate.md) | !SRADIATE | https://docs.frontistr.com/ko/keyword_reference/cnt/sradiate.html |
| [`keyword_reference/cnt/static.md`](keyword_reference/cnt/static.md) | !STATIC | https://docs.frontistr.com/ko/keyword_reference/cnt/static.html |
| [`keyword_reference/cnt/step.md`](keyword_reference/cnt/step.md) | !STEP | https://docs.frontistr.com/ko/keyword_reference/cnt/step.html |
| [`keyword_reference/cnt/temperature.md`](keyword_reference/cnt/temperature.md) | !TEMPERATURE | https://docs.frontistr.com/ko/keyword_reference/cnt/temperature.html |
| [`keyword_reference/cnt/time_points.md`](keyword_reference/cnt/time_points.md) | !TIME_POINTS | https://docs.frontistr.com/ko/keyword_reference/cnt/time_points.html |
| [`keyword_reference/cnt/trs.md`](keyword_reference/cnt/trs.md) | !TRS | https://docs.frontistr.com/ko/keyword_reference/cnt/trs.html |
| [`keyword_reference/cnt/uload.md`](keyword_reference/cnt/uload.md) | !ULOAD | https://docs.frontistr.com/ko/keyword_reference/cnt/uload.html |
| [`keyword_reference/cnt/user_material.md`](keyword_reference/cnt/user_material.md) | !USER_MATERIAL | https://docs.frontistr.com/ko/keyword_reference/cnt/user_material.html |
| [`keyword_reference/cnt/velocity.md`](keyword_reference/cnt/velocity.md) | !VELOCITY | https://docs.frontistr.com/ko/keyword_reference/cnt/velocity.html |
| [`keyword_reference/cnt/version.md`](keyword_reference/cnt/version.md) | !VERSION | https://docs.frontistr.com/ko/keyword_reference/cnt/version.html |
| [`keyword_reference/cnt/viscoelastic.md`](keyword_reference/cnt/viscoelastic.md) | !VISCOELASTIC | https://docs.frontistr.com/ko/keyword_reference/cnt/viscoelastic.html |
| [`keyword_reference/cnt/visualization/ambient_coef.md`](keyword_reference/cnt/visualization/ambient_coef.md) | !ambient_coef | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/ambient_coef.html |
| [`keyword_reference/cnt/visualization/color_comp_name.md`](keyword_reference/cnt/visualization/color_comp_name.md) | !color_comp_name | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/color_comp_name.html |
| [`keyword_reference/cnt/visualization/color_mapping_bar_on.md`](keyword_reference/cnt/visualization/color_mapping_bar_on.md) | !color_mapping_bar_on | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/color_mapping_bar_on.html |
| [`keyword_reference/cnt/visualization/data_comp_name.md`](keyword_reference/cnt/visualization/data_comp_name.md) | !data_comp_name | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/data_comp_name.html |
| [`keyword_reference/cnt/visualization/deform_scale.md`](keyword_reference/cnt/visualization/deform_scale.md) | !deform_scale | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/deform_scale.html |
| [`keyword_reference/cnt/visualization/display_method.md`](keyword_reference/cnt/visualization/display_method.md) | !display_method | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/display_method.html |
| [`keyword_reference/cnt/visualization/font_size.md`](keyword_reference/cnt/visualization/font_size.md) | !font_size | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/font_size.html |
| [`keyword_reference/cnt/visualization/index.md`](keyword_reference/cnt/visualization/index.md) | 시각화용 | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/index.html |
| [`keyword_reference/cnt/visualization/initial_style.md`](keyword_reference/cnt/visualization/initial_style.md) | !initial_style | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/initial_style.html |
| [`keyword_reference/cnt/visualization/isoline_number.md`](keyword_reference/cnt/visualization/isoline_number.md) | !isoline_number | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/isoline_number.html |
| [`keyword_reference/cnt/visualization/output_type.md`](keyword_reference/cnt/visualization/output_type.md) | !output_type | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/output_type.html |
| [`keyword_reference/cnt/visualization/surface_num.md`](keyword_reference/cnt/visualization/surface_num.md) | !surface_num | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/surface_num.html |
| [`keyword_reference/cnt/visualization/viewpoint.md`](keyword_reference/cnt/visualization/viewpoint.md) | !viewpoint | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/viewpoint.html |
| [`keyword_reference/cnt/visualization/visual.md`](keyword_reference/cnt/visualization/visual.md) | !VISUAL | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/visual.html |
| [`keyword_reference/cnt/visualization/x_resolution.md`](keyword_reference/cnt/visualization/x_resolution.md) | !x_resolution | https://docs.frontistr.com/ko/keyword_reference/cnt/visualization/x_resolution.html |
| [`keyword_reference/cnt/weld_line.md`](keyword_reference/cnt/weld_line.md) | !WELD_LINE | https://docs.frontistr.com/ko/keyword_reference/cnt/weld_line.html |
| [`keyword_reference/cnt/write.md`](keyword_reference/cnt/write.md) | !WRITE | https://docs.frontistr.com/ko/keyword_reference/cnt/write.html |
| [`keyword_reference/cnt/write_result.md`](keyword_reference/cnt/write_result.md) | !WRITE,RESULT | https://docs.frontistr.com/ko/keyword_reference/cnt/write_result.html |
| [`keyword_reference/cnt/write_visual.md`](keyword_reference/cnt/write_visual.md) | !WRITE,VISUAL | https://docs.frontistr.com/ko/keyword_reference/cnt/write_visual.html |
| [`keyword_reference/dat/control.md`](keyword_reference/dat/control.md) | !CONTROL | https://docs.frontistr.com/ko/keyword_reference/dat/control.html |
| [`keyword_reference/dat/index.md`](keyword_reference/dat/index.md) | 전체 제어 데이터 (dat) | https://docs.frontistr.com/ko/keyword_reference/dat/index.html |
| [`keyword_reference/dat/mesh.md`](keyword_reference/dat/mesh.md) | !MESH | https://docs.frontistr.com/ko/keyword_reference/dat/mesh.html |
| [`keyword_reference/dat/restart.md`](keyword_reference/dat/restart.md) | !RESTART | https://docs.frontistr.com/ko/keyword_reference/dat/restart.html |
| [`keyword_reference/dat/result.md`](keyword_reference/dat/result.md) | !RESULT | https://docs.frontistr.com/ko/keyword_reference/dat/result.html |
| [`keyword_reference/dat/subdir.md`](keyword_reference/dat/subdir.md) | !SUBDIR | https://docs.frontistr.com/ko/keyword_reference/dat/subdir.html |
| [`keyword_reference/input_rules.md`](keyword_reference/input_rules.md) | 입력 규칙 | https://docs.frontistr.com/ko/keyword_reference/input_rules.html |
| [`keyword_reference/msh/amplitude.md`](keyword_reference/msh/amplitude.md) | !AMPLITUDE | https://docs.frontistr.com/ko/keyword_reference/msh/amplitude.html |
| [`keyword_reference/msh/contactpair.md`](keyword_reference/msh/contactpair.md) | !CONTACTPAIR | https://docs.frontistr.com/ko/keyword_reference/msh/contactpair.html |
| [`keyword_reference/msh/egroup.md`](keyword_reference/msh/egroup.md) | !EGROUP | https://docs.frontistr.com/ko/keyword_reference/msh/egroup.html |
| [`keyword_reference/msh/element.md`](keyword_reference/msh/element.md) | !ELEMENT | https://docs.frontistr.com/ko/keyword_reference/msh/element.html |
| [`keyword_reference/msh/embedpair.md`](keyword_reference/msh/embedpair.md) | !EMBEDPAIR | https://docs.frontistr.com/ko/keyword_reference/msh/embedpair.html |
| [`keyword_reference/msh/end.md`](keyword_reference/msh/end.md) | !END | https://docs.frontistr.com/ko/keyword_reference/msh/end.html |
| [`keyword_reference/msh/equation.md`](keyword_reference/msh/equation.md) | !EQUATION | https://docs.frontistr.com/ko/keyword_reference/msh/equation.html |
| [`keyword_reference/msh/header.md`](keyword_reference/msh/header.md) | !HEADER | https://docs.frontistr.com/ko/keyword_reference/msh/header.html |
| [`keyword_reference/msh/index.md`](keyword_reference/msh/index.md) | 메시 데이터 (msh) | https://docs.frontistr.com/ko/keyword_reference/msh/index.html |
| [`keyword_reference/msh/material.md`](keyword_reference/msh/material.md) | !MATERIAL | https://docs.frontistr.com/ko/keyword_reference/msh/material.html |
| [`keyword_reference/msh/ngroup.md`](keyword_reference/msh/ngroup.md) | !NGROUP | https://docs.frontistr.com/ko/keyword_reference/msh/ngroup.html |
| [`keyword_reference/msh/node.md`](keyword_reference/msh/node.md) | !NODE | https://docs.frontistr.com/ko/keyword_reference/msh/node.html |
| [`keyword_reference/msh/section.md`](keyword_reference/msh/section.md) | !SECTION | https://docs.frontistr.com/ko/keyword_reference/msh/section.html |
| [`keyword_reference/msh/sgroup.md`](keyword_reference/msh/sgroup.md) | !SGROUP | https://docs.frontistr.com/ko/keyword_reference/msh/sgroup.html |
| [`keyword_reference/msh/zero.md`](keyword_reference/msh/zero.md) | !ZERO | https://docs.frontistr.com/ko/keyword_reference/msh/zero.html |
| [`keyword_reference/part/index.md`](keyword_reference/part/index.md) | 영역 분할 (part) | https://docs.frontistr.com/ko/keyword_reference/part/index.html |
| [`keyword_reference/part/partition.md`](keyword_reference/part/partition.md) | !PARTITION | https://docs.frontistr.com/ko/keyword_reference/part/partition.html |
| [`showcase/index.md`](showcase/index.md) | 사례집 | https://docs.frontistr.com/ko/showcase/index.html |
| [`showcase/mold.md`](showcase/mold.md) | 대규모 금형 모델의 정적 해석 | https://docs.frontistr.com/ko/showcase/mold.html |
| [`theory/99_reference.md`](theory/99_reference.md) | 참고문헌 | https://docs.frontistr.com/ko/theory/99_reference.html |
| [`theory/appendix_static_linear.md`](theory/appendix_static_linear.md) | 선형 탄성 정적 해석 (입문) | https://docs.frontistr.com/ko/theory/appendix_static_linear.html |
| [`theory/constitutive/01_linear_elastic.md`](theory/constitutive/01_linear_elastic.md) | 선형 탄성 | https://docs.frontistr.com/ko/theory/constitutive/01_linear_elastic.html |
| [`theory/constitutive/02_hyperelastic.md`](theory/constitutive/02_hyperelastic.md) | 초탄성 | https://docs.frontistr.com/ko/theory/constitutive/02_hyperelastic.html |
| [`theory/constitutive/03_elastoplastic.md`](theory/constitutive/03_elastoplastic.md) | 탄소성 | https://docs.frontistr.com/ko/theory/constitutive/03_elastoplastic.html |
| [`theory/constitutive/04_viscoelastic.md`](theory/constitutive/04_viscoelastic.md) | 점탄성 | https://docs.frontistr.com/ko/theory/constitutive/04_viscoelastic.html |
| [`theory/constitutive/05_creep.md`](theory/constitutive/05_creep.md) | 크리프 | https://docs.frontistr.com/ko/theory/constitutive/05_creep.html |
| [`theory/constitutive/06_thermal_properties.md`](theory/constitutive/06_thermal_properties.md) | 열물성 | https://docs.frontistr.com/ko/theory/constitutive/06_thermal_properties.html |
| [`theory/contact/01_contact.md`](theory/contact/01_contact.md) | 접촉 해석 | https://docs.frontistr.com/ko/theory/contact/01_contact.html |
| [`theory/discretization/01_shape_functions.md`](theory/discretization/01_shape_functions.md) | 형상함수와 유한요소 근사 | https://docs.frontistr.com/ko/theory/discretization/01_shape_functions.html |
| [`theory/discretization/02_global_derivative.md`](theory/discretization/02_global_derivative.md) | 형상함수의 공간 미분 | https://docs.frontistr.com/ko/theory/discretization/02_global_derivative.html |
| [`theory/discretization/03_internal_force.md`](theory/discretization/03_internal_force.md) | 내력 가상일의 이산화 | https://docs.frontistr.com/ko/theory/discretization/03_internal_force.html |
| [`theory/discretization/04_external_force_and_assembly.md`](theory/discretization/04_external_force_and_assembly.md) | 외력의 가상일과 전체 방정식의 조립 | https://docs.frontistr.com/ko/theory/discretization/04_external_force_and_assembly.html |
| [`theory/discretization/05_numerical_integration.md`](theory/discretization/05_numerical_integration.md) | 수치 적분 | https://docs.frontistr.com/ko/theory/discretization/05_numerical_integration.html |
| [`theory/discretization/08_element_solid.md`](theory/discretization/08_element_solid.md) | 3차원 솔리드 요소의 형상 함수 | https://docs.frontistr.com/ko/theory/discretization/08_element_solid.html |
| [`theory/discretization/09_element_advanced.md`](theory/discretization/09_element_advanced.md) | 고성능 요소 정식화 | https://docs.frontistr.com/ko/theory/discretization/09_element_advanced.html |
| [`theory/governing/01a_kinematics.md`](theory/governing/01a_kinematics.md) | 운동·변형·변형률 | https://docs.frontistr.com/ko/theory/governing/01a_kinematics.html |
| [`theory/governing/01b_stress_balance.md`](theory/governing/01b_stress_balance.md) | 응력과 보존 법칙 | https://docs.frontistr.com/ko/theory/governing/01b_stress_balance.html |
| [`theory/governing/01c_virtual_work.md`](theory/governing/01c_virtual_work.md) | 가상일의 원리 | https://docs.frontistr.com/ko/theory/governing/01c_virtual_work.html |
| [`theory/governing/01d_incremental_analysis.md`](theory/governing/01d_incremental_analysis.md) | 증분해석의 틀 | https://docs.frontistr.com/ko/theory/governing/01d_incremental_analysis.html |
| [`theory/governing/02_heat_equation.md`](theory/governing/02_heat_equation.md) | 열전도 방정식 | https://docs.frontistr.com/ko/theory/governing/02_heat_equation.html |
| [`theory/index.md`](theory/index.md) | 이론 매뉴얼 | https://docs.frontistr.com/ko/theory/index.html |
| [`theory/linear_solver/01_solver.md`](theory/linear_solver/01_solver.md) | 선형 솔버 | https://docs.frontistr.com/ko/theory/linear_solver/01_solver.html |
| [`theory/notation/01_tensor_notation.md`](theory/notation/01_tensor_notation.md) | 텐서 표기법과 수학적 기초 | https://docs.frontistr.com/ko/theory/notation/01_tensor_notation.html |
| [`theory/notation/02_symbol_list.md`](theory/notation/02_symbol_list.md) | 물리량 기호 목록 | https://docs.frontistr.com/ko/theory/notation/02_symbol_list.html |
| [`theory/parallel/01_parallel.md`](theory/parallel/01_parallel.md) | 병렬 유한요소법 | https://docs.frontistr.com/ko/theory/parallel/01_parallel.html |
| [`theory/solution_methods/01_newton_raphson.md`](theory/solution_methods/01_newton_raphson.md) | Newton-Raphson 법 | https://docs.frontistr.com/ko/theory/solution_methods/01_newton_raphson.html |
| [`theory/solution_methods/01b_convergence.md`](theory/solution_methods/01b_convergence.md) | 수렴 판정 | https://docs.frontistr.com/ko/theory/solution_methods/01b_convergence.html |
| [`theory/solution_methods/02_dynamic.md`](theory/solution_methods/02_dynamic.md) | 동적 해석 기법 | https://docs.frontistr.com/ko/theory/solution_methods/02_dynamic.html |
| [`theory/solution_methods/03_eigen.md`](theory/solution_methods/03_eigen.md) | 모드 해석 | https://docs.frontistr.com/ko/theory/solution_methods/03_eigen.html |
| [`theory/solution_methods/04_frequency_response.md`](theory/solution_methods/04_frequency_response.md) | 주파수 응답 해석 | https://docs.frontistr.com/ko/theory/solution_methods/04_frequency_response.html |
| [`theory/solution_methods/05_transient_heat.md`](theory/solution_methods/05_transient_heat.md) | 비정상 열전도 해석 | https://docs.frontistr.com/ko/theory/solution_methods/05_transient_heat.html |
| [`theory/solution_methods/06_geometric_bc.md`](theory/solution_methods/06_geometric_bc.md) | 기하학적 경계 조건의 처리 | https://docs.frontistr.com/ko/theory/solution_methods/06_geometric_bc.html |
| [`verification_and_validation/01_simple.md`](verification_and_validation/01_simple.md) | 단순 형상 모델에 의한 검증 | https://docs.frontistr.com/ko/verification_and_validation/01_simple.html |
| [`verification_and_validation/02_static.md`](verification_and_validation/02_static.md) | 탄성 정적 해석용 실제 모델 예제 | https://docs.frontistr.com/ko/verification_and_validation/02_static.html |
| [`verification_and_validation/03_dynamic.md`](verification_and_validation/03_dynamic.md) | 선형 동적 해석용 실제 모델 예제 | https://docs.frontistr.com/ko/verification_and_validation/03_dynamic.html |
| [`verification_and_validation/04_eigen.md`](verification_and_validation/04_eigen.md) | 모드 해석용 실제 모델 예제 | https://docs.frontistr.com/ko/verification_and_validation/04_eigen.html |
| [`verification_and_validation/05_heat.md`](verification_and_validation/05_heat.md) | 열전도 해석의 실모델 예 | https://docs.frontistr.com/ko/verification_and_validation/05_heat.html |
