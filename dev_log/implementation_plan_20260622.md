# VTKHDF Exporter Implementation Plan

vtu 형식 대신 고성능의 HDF5 기반 **VTKHDF** 형식을 활용해 격자 및 결과(변위, 상태 변수 등)를 내보내도록 구현하는 계획입니다.

## User Review Required

> [!NOTE]
> `h5py` 라이브러리가 이미 시스템에 설치되어 동작하는 것을 확인했습니다. 프로젝트 의존성에 이를 공식적으로 등록하기 위해 `pyproject.toml`에 `h5py`를 추가합니다.

> [!WARNING]
> 이 변경을 적용하면 기존의 `.vtu` 형식 대신 `.vtkhdf` 형식의 바이너리 파일이 생성됩니다. ParaView 5.10 이상 버전에서는 `.vtkhdf` 파일을 즉시 불러와 3D/2D 시각화가 가능합니다.

## Proposed Changes

### 1. [dispsolver/export](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/export)

#### [NEW] [vtkhdf_exporter.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/export/vtkhdf_exporter.py)
* HDF5 기반의 `export_vtkhdf` 함수를 구현합니다.
* Q4 및 T3 요소를 포함한 비정형 격자(UnstructuredGrid)를 모두 정상 수용할 수 있도록 `Connectivity`, `Offsets`, `Types`를 유동적으로 매핑합니다.
* 절점 변위(`Displacement`) 및 기타 절점/요소 상태 변수를 VTKHDF 표준 포맷에 맞추어 기록합니다.

#### [MODIFY] [__init__.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/export/__init__.py)
* `export_vtu` 대신 `export_vtkhdf`를 기본적으로 내보내도록 변경합니다.

### 2. [examples](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/examples)

#### [MODIFY] [ex03_display_fold.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/examples/ex03_display_fold.py)
* `export_vtu` 관련 임포트와 호출을 `export_vtkhdf`로 전환합니다.
* 출력 경로의 확장자를 `.vtu`에서 `.vtkhdf`로 변경합니다.

### 3. [pyproject.toml](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/pyproject.toml)

#### [MODIFY] [pyproject.toml](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/pyproject.toml)
* `dependencies` 리스트에 `h5py`를 추가합니다.

---

## Verification Plan

### Automated Tests
* 모든 유닛 테스트가 새로운 패키지 및 구조 하에서 성공적으로 임포트되고 실행되는지 확인합니다:
  ```powershell
  pytest
  ```

### Manual Verification
* `examples/ex03_display_fold.py`를 실행하여 `output/ex03_fold.vtkhdf` 파일이 문제없이 생성되는지 확인합니다.
* Python 스크립트를 작성하여 생성된 `.vtkhdf` 파일 내부 구조가 VTKHDF 규격에 적합한지 데이터를 검증합니다:
  ```powershell
  python -c "import h5py; f = h5py.File('output/ex03_fold.vtkhdf', 'r'); print(list(f['VTKHDF'].keys()))"
  ```
