# Contact 속도: `apply_boundary_conditions` LIL→COO (2026-09-20)

## 증상

- `tests/test_hertz_contact_benchmark.py` 전체 행 (>600s 무출력).
- faulthandler: 90s째 단일 `scipy lil __setitem__` 내부
  (`dynamic3d.py:875 apply_boundary_conditions` ← `solve_step` ← `_solve_with_cutback`).

## 원인

- 구코드: `K.tolil()` 후 고정DOF마다 `K_bc[dof,:]=0` + `K_bc[:,dof]=0`.
  LIL 열 슬라이싱은 전 행 리스트 순회 → 반복당 O(고정DOF×nnz).
  Hertz 구성(약 1k 고정DOF, nnz 수십만)에서 반복 1회에 수 분.
- 매 Newton 반복마다 호출이라 전체 행으로 증폭.

## 수정 (수치 bit-identical)

- `dispsolver/solver3d/dynamic3d.py::apply_boundary_conditions`:
  COO 변환 1회 → 고정행/열 접촉 엔트리 마스크 제거 → penalty 대각 추가. 반복당 O(nnz).
- 동등성: 난수행렬 대조 `max|dK|=0.0, max|dr|=0.0`.

## 효과

- 동일 Hertz uniform 구성: 행 → **CONVERGED 40스텝 완주**.
- `test_hertz_contact_benchmark.py`: **4 passed in 73.35s**.
- `run_all`: 11/14. 3 FAIL(`bending_2pt`, elastica 2종)은 2D 벤치로
  수정 모듈(`solver3d`)을 import하지 않음 → 본 수정과 무관.
  의심처는 미커밋 `solver/dynamic.py`(40행)·`solver2d/dynamic2d.py`(349행) agy diff.

## 추가 속도 여지 (미착수)

1. `skip_fully_prescribed` (§4.13, 구현済·미사용): Hertz 하중판/대칭면 완전고정 요소 조립 스킵.
2. 접촉 탐색: Phase1 경로 BVH 사용 여부 확인. brute-force 잔재 시 교체.
3. PARDISO 싱글톤은 이미 적용됨 (§4.11). 재발 방지: 반복 내 `PyPardisoSolver()`·`.tolil()` 금지.
