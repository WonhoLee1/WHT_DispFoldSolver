# Implementation Plan: 6-Core CPU Parallel Acceleration for FrontISTR & WHT Solver Benchmarks (2026-10-20)

## Goal Description
사용자의 요청에 따라 FrontISTR 및 WHT_DispFoldSolver의 벤치마크 연산에 **6개 CPU 코어(`threads=6`)**를 적용하여 대규모 3D 14-Layer Full-Stack 해석 속도를 극대화합니다.

---

## User Review Required
> [!IMPORTANT]
> **현재 실행 중인 4-Core 3D 작업 중단 및 6-Core 재실행 여부**:
> - 기존에 4코어로 실행 중이던 백그라운드 태스크(`task-11860`)를 즉시 중단(`manage_task kill`)하고, `--threads 6`으로 3D 벤치마크를 재실행합니다.
> - **2D 5단계 벤치마크**: 방금 4코어로 완료된 2D 5단계 결과(총 약 12분 소요)를 6코어로 다시 측정할지, 아니면 3D 벤치마크부터 6코어로 집중 진행할지 선택하실 수 있습니다. (기본 권장: 3D 및 WHT를 6코어로 우선 완주한 뒤 2D 재측정).

---

## Proposed Changes

### 1. 스레드 환경 변수 및 인자 설정 (`threads = 6`)
- **FrontISTR**:
  - `OMP_NUM_THREADS = "6"`
  - 실행 인수: `fistr1.exe -t 6`
  - MUMPS 멀티스레드 병렬 희소 행렬 분해에 6개 코어 풀가동
- **WHT Solver**:
  - `MKL_NUM_THREADS = "6"`
  - `OMP_NUM_THREADS = "6"`
  - Intel oneMKL PARDISO 대칭 희소 행렬 분해($LL^T$)에 6개 코어 동시 연산

### 2. 코드 기본값 및 스크립트 업데이트
#### [MODIFY] `example_disp_fold/benchmark_mesh_refinement_fistr_wht.py`
- `threads` 기본 인자를 `6`으로 변경 (`parser.add_argument("--threads", type=int, default=6)`).
- `WHT Solver` 러너 내부에 `os.environ["MKL_NUM_THREADS"] = str(threads)` 및 `os.environ["OMP_NUM_THREADS"] = str(threads)` 명시 설정.

---

## Verification Plan

### Automated Execution
1. **현재 4-Core 태스크 종료**:
   - `manage_task kill` on `task-11860`.
2. **3D 14-Layer 6-Core 벤치마크 실행**:
   ```bash
   python example_disp_fold/benchmark_mesh_refinement_fistr_wht.py --preset teardrop --part 3d --substeps 30 --threads 6
   ```
3. **WHT Solver 6-Core 비교 실행**:
   ```bash
   python example_disp_fold/benchmark_mesh_refinement_fistr_wht.py --preset teardrop --part wht --substeps 30 --threads 6
   ```
4. **PyVista 3D Depth Isometric 캡처**:
   ```bash
   python example_disp_fold/capture_3d_depth_iso_pyvista.py
   python example_disp_fold/capture_all_benchmark_cases.py
   ```
5. **보고서 종합 정리**:
   - `dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md`에 6코어 속도 측정 결과 반영.
