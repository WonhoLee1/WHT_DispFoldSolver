# SOTA(최신 연구 동향) 반영 상용화 수준 3D 비선형 해석 아키텍처 및 구현 계획서
**작성일**: 2026-09-10
**목표**: WHT_DispFoldSolver를 DOLFINx(FEniCS), MFEM 등 최신 고성능 연구(2015-2026)에서 입증된 아키텍처를 도입하여 상용 Abaqus 수준의 범용 비선형 유한요소해석 프레임워크로 승격.

---

## 1. SOTA 기반 코어 아키텍처 설계 원칙 (SOTA Architectural Principles)

### 1.1 Data-Oriented Design (DOD) 및 메모리 병렬화 최적화
* **참고 문헌**: *Bordeu et al. (2023) "High-performance implementation of FEA in Python using Numba"*, *Scroggs et al. (2022) "DOLFINx..."*
* **적용 설계**: 
  * 파이썬 객체를 배제하고 절점, 연결성, 물성을 **SoA(Struct of Arrays)** 구조의 1D/2D 연속 메모리 블록으로 배치합니다.
  * 동적 행렬 크기 변경 병목을 피하기 위해 해석 전 **CSR/CSC 스파시티 토폴로지를 1회 사전 할당(Pre-allocate)** 합니다.
  * 다중 코어에서 Race Condition 없이 동시 조립하기 위해 **Element Coloring 알고리즘**을 도입해 Numba `prange` 병렬처리를 극대화합니다.

### 1.2 Stateless / Batched Material Constitutive Interface
* **참고 문헌**: 최신 멀티스케일 및 JIT 기반 연속체 역학 솔버 설계론.
* **적용 설계**:
  * 객체지향적 `material.compute_stress()`를 폐기하고, Numba가 컴파일 가능한 **상태 독립적 함수형 커널(Stateless Functional Kernel)**로 전환합니다.
  * 단일 요소가 아닌 **배치(Batched) 형태의 변형기울기 $\mathbf{F}$ (형태: `[N_elements, N_quads, 3, 3]`)**를 받아 병렬로 응력 및 접선 모듈러스를 도출합니다.
  * 내부 상태 변수(ISV; Internal State Variables)는 글로벌 텐서에 저장하여 C-레벨 포인터로 갱신합니다.

### 1.3 강건한 비선형 솔버 제어 (Robust Non-linear Solver Control)
* **참고 문헌**: *Perez et al. (2019) "Adaptive time-stepping and cutback algorithms..."*, *Wriggers et al. (2021)*
* **적용 설계**:
  * **스마트 컷백(Smart Cutback)**: 반복(Iteration) 횟수뿐만 아니라 요소 역전(Jacobian < 0)이나 소성 변형률 증분이 물리적 한계($\Delta\epsilon > \Delta\epsilon_{max}$)를 초과할 경우 즉각 컷백을 수행하는 휴리스틱스를 도입합니다.
  * **고급 라인 서치(Advanced Line Search)**: Armijo-Goldstein 룰을 적용해 에너지 직교성을 검사하고, 오버슈팅으로 인한 요소 찌그러짐을 방지합니다.
  * **Arc-Length Method (Riks)**: 향후 좌굴(Buckling) 후 거동 해석을 위한 모듈을 수용할 수 있도록 하중 제어기를 확장 설계합니다.

---

## 2. 3단계 구현 마일스톤 (Implementation Roadmap)

### Phase 1: DOD Assembly & Batched Material Interface (실시 중)
* **목표**: Numba JIT 및 Element Coloring 기법을 이용해 파이썬 오버헤드가 제로인 초고속 메쉬 조립기 및 물성 디스패처 구현.
* **작업 내역**:
  1. `dispsolver/material3d/numba_materials.py`에 Stateless UMAT 인터페이스 도입 (수치 오류/Inversion 검출 시 Flag 반환).
  2. `assembly_utils.py`를 확장하여 CSR 인덱스 선할당 및 Numba 커널 내 직접 업데이트 구현.

### Phase 2: Smart Cutback & Stabilization Engine 탑재
* **목표**: 가혹한 폴딩(Folding) 및 국부적 파단 조건에서도 발산하지 않는 끈질긴 비선형 추적기 탑재.
* **작업 내역**:
  1. `detF < 0` 등 물리적 파탄 발생 시 뉴턴 스텝을 기각하고 `dt`를 0.25배로 축소하는 스마트 컷백 컨트롤러 분리 구현.
  2. 수치적 안정을 위한 인공 점성(Artificial Damping) 및 고급 Backtracking 라인 서치 알고리즘 정밀화.

### Phase 3: Extension Framework (Contact & Arc-length)
* **목표**: 상용 솔버의 고급 기능 확장을 위한 뼈대 마련.
* **작업 내역**:
  1. 동적 스파시티 결합을 위한 BVH 기반 접촉 탐색(Broad-phase) 프로토타입.
  2. 호장 길이(Arc-length) 추적을 위한 잉여 자유도(Load multiplier) 컨트롤 모듈 예비 설계.
