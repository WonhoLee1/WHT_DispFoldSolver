# Implementation Plan — MPI 도메인 분해 병렬 해석 종합 테스트 계획

**문서 번호**: `PLAN-20260913-MPI-COMPREHENSIVE-TEST`  
**작성 일자**: 2026-09-13  
**대상 스크립트**: [`examples/ex16_mpi_domain_decomposition_2d.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/examples/ex16_mpi_domain_decomposition_2d.py), [`tests/test_mesh_partitioner.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/tests/test_mesh_partitioner.py)  

---

## 1. 개요 (Goal Description)

사용자의 요청에 따라, `mpiexec -n <N>`으로 프로세스 수(`size`)를 변경했을 때 각 프로세스가 자신의 `rank`와 전체 `size`를 인지하여 메쉬를 $N$개로 분할하고, 완전 독립적인 로컬 어셈블리 후 전역 결합까지 수행하는 전 과정을 **1개, 2개, 4개, 8개 MPI 프로세스 스윕(Sweep) 테스트**를 통해 검증합니다.

---

## 2. 검증 시나리오 및 테스트 매트릭스 (Test Matrix)

| 테스트 단계 | 실행 명령어 | 검증 목표 | 예상 분할 결과 |
|:---:|:---|:---|:---|
| **Test 1** | `python examples/ex16_mpi_domain_decomposition_2d.py` | MPI 비활성 / 단일 프로세스 기준선 (Baseline) | 1개 서브도메인 (160개 요소 전수 소유) |
| **Test 2** | `mpiexec -n 2 python -u examples/ex16_mpi_domain_decomposition_2d.py` | 2-Rank 이중 분할 검증 | Rank 0: 80개 / Rank 1: 80개 |
| **Test 3** | `mpiexec -n 4 python -u examples/ex16_mpi_domain_decomposition_2d.py` | 4-Rank 4등분 분할 검증 | Rank 0, 1, 2, 3: 각 40개 요소 |
| **Test 4** | `mpiexec -n 8 python -u examples/ex16_mpi_domain_decomposition_2d.py` | 8-Rank 고도 병렬 분할 검증 (16 코어 활용) | Rank 0~7: 각 20개 요소 |

---

## 3. 합격 판정 기준 (Verification Criteria)

1. **프로세스 인식**: 모든 프로세스가 `mpiexec -n <N>`에 지정된 $N$을 `comm.Get_size()`로 정확히 인식할 것.
2. **요소 보존**: 각 서브도메인의 요소 수 합계 $\sum_{p=0}^{N-1} |E_p| = 160$ (누락/중복 0개).
3. **기계 정밀도 일치**:
   - 내력 벡터 상대 오차: $\|f_{\text{mpi}} - f_{\text{mono}}\| / \|f\| < 10^{-10}$
   - 접선 강성 상대 오차: $\|K_{\text{mpi}} - K_{\text{mono}}\| / \|K\| < 10^{-10}$
   - 변위 해석 상대 오차: $\|\Delta u_{\text{mpi}} - \Delta u_{\text{mono}}\| / \|\Delta u\| < 10^{-10}$
4. **확장성 및 통신 시간**: 프로세스 수 증가에 따른 통신 오버헤드 및 분할 시간 측정.

---

## 4. 실행 계획

1. `ex16_mpi_domain_decomposition_2d.py`를 순차적으로 `-n 1`, `-n 2`, `-n 4`, `-n 8`로 실행.
2. 각 실행의 Rank별 할당 내역과 오차 수치를 수집.
3. 종합 비교 요약표를 작성하여 사용자에게 보고.
