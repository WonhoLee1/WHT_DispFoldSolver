# VTKHDF Exporter 구현 요약

기존의 `.vtu` 출력 대신 ParaView 5.10 이상에서 네이티브로 지원하는 고성능 HDF5 기반의 `.vtkhdf`(UnstructuredGrid) 포맷으로 출력 기능을 전환했습니다.

## 주요 변경 사항

*   **VTKHDF 익스포터 신규 작성 (`vtkhdf_exporter.py`)**:
    *   `h5py` 라이브러리를 활용하여 HDF5 포맷 파일을 생성합니다.
    *   비정형 격자(UnstructuredGrid) 구조를 지원하도록 작성되었으며, `Q4` 요소(VTK_QUAD) 및 `T3` 요소(VTK_TRIANGLE)가 혼합된 하이브리드 메시도 유연하게 처리할 수 있도록 동적으로 `Connectivity`, `Offsets`, `Types`를 구성합니다.
    *   절점 변위 데이터를 `PointData`에, 상태 변수를 `CellData` 및 `PointData` 그룹에 각각 체계적으로 분리하여 저장합니다.
*   **패키지 설정 변경**:
    *   `dispsolver/export/__init__.py`가 이제 `export_vtkhdf` 함수를 노출합니다.
    *   `pyproject.toml`의 의존성(dependencies)에 `h5py>=3.0`을 추가했습니다.
*   **통합 예제 수정**:
    *   `examples/ex03_display_fold.py` 내부의 결과를 `.vtkhdf` 확장자로 출력하도록 적용했습니다.

## 검증 결과

### 1. 단위 테스트 (`pytest`)
*   기존과 동일하게 84개의 유닛 테스트를 실행했으며, 이전에 보고된 1건의 `test_static.py` 실패 외에 **익스포터 변경으로 인한 추가 회귀(Regression) 에러는 발생하지 않았습니다**.

### 2. 시뮬레이션 및 익스포트 검증
*   `ex03_display_fold.py` 실행 시, 동적 시뮬레이션 과정에서 타임스텝(`dt`) 적응형 제어로 인한 물리적 수렴 한계로 시뮬레이션이 조기 중단(Abort)되었으나, 종료 직전에 **정상적으로 `output/ex03_fold.vtkhdf` 파일을 익스포트**하는 데 성공했습니다.
*   생성된 파일은 내부에 `VTKHDF` 구조 (버전 2.2) 및 `Points`, `NumberOfCells`, `PointData`, `CellData` 등의 올바른 HDF5 데이터셋 계층을 유지하고 있습니다.

> [!TIP]
> 이제 생성된 `output/ex03_fold.vtkhdf` 파일을 최신 ParaView에서 열어 변위와 요소/절점 데이터를 바로 확인할 수 있습니다. 시뮬레이션의 조기 수렴 실패 문제는 익스포트 포맷과 무관하게 이전 세션부터 이어진 대변형 솔버의 강성 조립 또는 적응형 스텝 로직과 관련된 것으로, 향후 별도 디버깅이 권장됩니다.
