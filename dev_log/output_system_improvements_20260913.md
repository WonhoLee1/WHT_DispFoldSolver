# Output System Improvements — 20260913 (세션 후반)

## 작업 일시
- 2026-09-13 (세션 재개 후 연속 작업)

## 완료된 개선사항 (3가지)

### 1. FieldVar / HistoryVar 상수 클래스 도입 (`variables.py`)

**배경**: 기존 `FIELD_VARIABLE_COMPONENTS` Dict, `HISTORY_ENERGY_VARIABLES` Set은  
IntelliSense 자동완성이 지원되지 않았음.

**변경**: `class FieldVar` / `class HistoryVar` 상수 클래스 추가
- 모든 출력 변수를 클래스 속성(`str` 타입 힌트)으로 정의
- 각 속성에 상세 docstring → IDE IntelliSense에서 즉시 설명 표시
- 기존 Dict/Set는 하위 호환성 유지 (FieldVar/HistoryVar 상수 참조로 변경)
- `__init__.py`에 `FieldVar`, `HistoryVar` 추가 노출

**사용 예시**:
```python
from dispsolver.output import FieldVar, HistoryVar

req = step.FieldOutputRequest("F-1", variables=[FieldVar.U, FieldVar.S, FieldVar.PEEQ])
hist = step.HistoryOutputRequest("H-1", variables=[HistoryVar.ALLIE, HistoryVar.ALLPD])
```

---

### 2. 실시간 스트리밍 VTKHDF 저장 (`vtkhdf_writer.py` 전면 재설계)

**배경**: 기존 `TransientVTKHDFWriter`는 모든 스텝을 메모리에 버퍼링 후  
`close()` 시 한 번에 기록 → 해석 중 결과 확인 불가.

**변경**: OpenRadioss-to-vtkhdf 프로젝트의 `add_or_create_dataset` 패턴 채용
- `__init__()`: h5.File 열기 + 정적 토폴로지(Connectivity, Offsets, Types) 즉시 기록  
  + resizable dataset(`maxshape=(None, ...)`) 미리 생성
- `add_step()`: 매 스텝마다 `dataset.resize()` + 슬라이스 할당으로 즉시 append  
  + `h5.File.flush()` 호출 → ParaView가 실시간으로 파일 읽기 가능
- `close()`: h5.File 핸들만 닫음 (데이터는 이미 기록 완료)
- 컨텍스트 매니저 `with ... as writer:` 지원 (`__enter__`/`__exit__`)

**내부 헬퍼**:
- `_append_scalar()`: 1-D resizable dataset에 스칼라 1개 append
- `_append_dataset()`: N-D resizable dataset에 배열 axis-0 append
- `_append_field()`: 첫 등장 시 dataset 생성, 이후 append (PointData/CellData)

---

### 3. OutputManager에 vtkhdf_path 연동 (`manager.py`)

**배경**: OutputManager가 ODB에만 기록하고 VTKHDF는 `odb.export_vtkhdf()`를  
수동으로 호출해야 했음.

**변경**:
- `OutputManager.__init__(vtkhdf_path=None)` 파라미터 추가
- Lazy 초기화: 첫 번째 `record_step()` 호출 시 `solver.mesh`로 writer 생성
- `_record_field_frame()`: OdbFrame 생성과 동시에 `writer.add_step()` 호출
  - `field_dict`의 FieldOutput을 point_data/cell_data로 분류하여 전달
- `finalize()`: VTKHDF writer close (solver2d/3d의 `try/finally`에서 자동 호출)

**solver2d/3d `solve_with_odb()` 변경**:
- `vtkhdf_path: Optional[str] = None` 파라미터 추가
- `try/finally`로 `output_mgr.finalize()` 보장 호출

**사용 예시**:
```python
odb = solver.solve_with_odb(
    step=step,
    sys=sys,
    vtkhdf_path="result.vtkhdf",  # 해석 중 ParaView에서 실시간 확인 가능
)
```

---

## 테스트 결과

```
10 passed, 1 warning in 30.87s
```

- `TestOutputRequestsAndTiming`: 4개 PASSED
- `TestFieldDataVectorization`: 2개 PASSED
- `TestHistoryOutputAndPandas`: 1개 PASSED
- `TestVTKHDFExportAndH5ODB`: 2개 PASSED (스트리밍 방식 포함)
- `TestModelAndSolverIntegration::test_2d_solve_with_odb`: PASSED

---

## 참고: OpenRadioss-to-vtkhdf 프로젝트 패턴

경로: `D:\PythonCodeStudy\openradioss-to-vtkhdf-main\src\vtkhdfwriter.py`

핵심 채용 패턴:
- step 0: `group.create_dataset(name, data=data, maxshape=(None,...))` → resizable
- step n: `dataset.resize(old_len + len(data), axis=0); dataset[old_len:] = data`
- 파일 핸들을 __init__에서 열고 close()에서만 닫음 (append 모드 유지)
