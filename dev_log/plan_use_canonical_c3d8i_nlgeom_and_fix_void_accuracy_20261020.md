# Implementation Plan: 표준 C3D8I/CPE4I 요소 경로(nlgeom=True) 사용 및 14-Layer 보이드 고립 절점 소거 (2026-10-20)

## Goal Description
사용자의 지시사항에 따라:
1. **`_CR` 명시적 접미사 제거 및 Abaqus 표준 요소 경로(`C3D8I`, `CPE4I`, `C3D8H`, `CPE4H`, `C3D8`, `CPE4`) 채택**:
   - `DynamicSolver3D` 및 `DynamicSolver2D`는 `nlgeom=True`가 주어지면 `ElementCatalog`를 통해 표준 요소명(`C3D8I` 등)을 자동으로 최적의 대변형/회전 커널로 프로모션하여 실행하도록 아키텍처가 구축되어 있습니다.
   - 따라서 수동으로 `_CR`을 지정하던 경로를 폐기하고, 표준 `C3D8I` 및 `CPE4I` 경로(`elem_case = "3D-Opt1"`, `"2D-Opt1"`)를 단일 진실 공급원(SSOT)으로 사용합니다.
2. **테이프 보이드(`PSA_0_Voided`) 내 고립 절점(Isolated Nodes) 소거 및 14-Layer 정확도 복원**:
   - 힌지 영역의 비연결 고립 절점 29개(3D는 수백 개)를 메쉬에서 완전 제거하여 강성 행렬의 0-대각 특이성을 소거하고, 형상 루프 폭($\Delta X \approx 4.5\,\text{mm}$) 및 구동 모멘트 전달을 정상화합니다.

---

## User Review Required
> [!IMPORTANT]
> **표준 요소명 일원화 및 자동 프로모션 (`nlgeom=True`)**:
> - **3D 솔리드 (PET, Glass)**: `C3D8I` (솔버가 `nlgeom=True`일 때 탄성이면 커널 11, 비탄성이면 커널 0으로 자동 매핑)
> - **3D 점착제 (PSA)**: `C3D8H` (솔버가 `nlgeom=True`일 때 커널 2 `C3D8H co-rotational u-p hybrid`로 자동 매핑)
> - **3D 서포트 플레이트**: `C3D8` (기존 `C3D8_CR` 하드코딩 제거 $\to$ `C3D8` 지정 시 솔버가 커널 1 `C3D8 B-bar`로 자동 매핑)
> - **2D 모델**: `CPE4I` (솔리드) + `CPE4H` (PSA) + `CPE4` (플레이트)

---

## Proposed Changes

### 1. Gmsh 메쉬 생성기: 표준 요소명 적용 및 고립 절점 자동 소거
#### [MODIFY] `example_disp_fold/gmsh_mesh_generator.py`
- 플레이트 요소 타입: `C3D8_CR` $\to$ 표준 `C3D8`로 변경.
- 메쉬 빌드 완료 시 고립 절점 필터링 로직 추가:
  ```python
  def _clean_isolated_nodes(self, mesh: Union[Mesh2D, Mesh3D]) -> None:
      connected = set()
      for elem in mesh.elements.values():
          for nid in elem.node_ids:
              connected.add(nid)
      isolated = set(mesh.nodes.keys()) - connected
      for nid in isolated:
          del mesh.nodes[nid]
  ```
- 노드 삭제에 따른 절점 인덱싱 연속성 보존 및 `meta` 구동 절점 목록 정제.

### 2. 벤치마크 러너: 표준 `elem_case` 지정 및 후처리 정밀화
#### [MODIFY] `example_disp_fold/benchmark_mesh_refinement_fistr_wht.py`
- `run_wht_case`의 `elem_case` 기본값을 `2D-Opt1` (`CPE4I`+`CPE4H`), `3D-Opt1` (`C3D8I`+`C3D8H`)로 설정.
- `loop_width_x` 계산 시 실제 연결된 절점만 마스킹하여 폭 왜곡($14.0\,\text{mm}$)을 원천 차단.

---

## Verification Plan

### Automated Execution
1. **메쉬 생성 검증**:
   - `python -c "from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator; ..."` 실행 후 `isolated_nodes == 0`, 플레이트 요소 타입 `C3D8` 확인.
2. **WHT 2D 14-Layer 벤치마크 재실행**:
   ```bash
   python example_disp_fold/benchmark_mesh_refinement_fistr_wht.py --preset teardrop --part wht --substeps 30 --threads 6
   ```
   - 루프 폭 $\Delta X$가 $14.0\,\text{mm} \to 4.4 \sim 4.5\,\text{mm}$로 정상 산출되는지 검증.
   - 3D WHT $nz=1$ 케이스가 특이 행렬 없이 수렴 개선되는지 검증.
3. **종합 보고서 및 몽타주 갱신**:
   - `dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md` 업데이트.
