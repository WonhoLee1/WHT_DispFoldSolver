# 2D Reference Node (RP) UR3 Rotation BC & RigidBody Solver Verification Report

> **작성일**: 2026-10-18  
> **목적**: Package P4 선행 필수 요건 확인 (`dev_log/agent_work_packages_20261018.md` §P4 선행 확인)  
> **진단 대상**: `DynamicSolver2D` 및 `create_solver2d` 파이프라인에서 `*RIGID BODY` (REF NODE=RP)와 `*BOUNDARY` (UR3 회전)의 실제 동작 여부 실측  

---

## 1. 진단 배경 및 P4 요구사항

`dev_log/agent_work_packages_20261018.md` P4 명세서 (line 127):
> **선행 확인 (가장 먼저)**: 2D 솔버가 RP의 UR3 처리를 실제로 하는가. 작은 덱(RP 하나에 강체 요소 하나, UR3 처방)으로 확인한다. 동작하지 않으면 고치지 말고 재현 덱과 증상을 hand-off의 통합자 요청에 적고, 가능한 범위(예: 3번 항목의 exporter 형식 정리, 다단계 구조)를 계속 진행한다.

---

## 2. 최소 재현 데크 (Minimal Reproducer Deck)

```inp
*HEADING
2D RP UR3 Test
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
100, 0.0, 0.0
*ELEMENT, TYPE=CPE4, ELSET=PLATE
1, 1, 2, 3, 4
*NSET, NSET=SLAVE_NODES
1, 2, 3, 4
*NSET, NSET=RP_NODE
100
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=PLATE, MATERIAL=STEEL
*RIGID BODY, REF NODE=100, TIE NSET=SLAVE_NODES
*STEP, NAME=STEP1, NLGEOM=YES
*STATIC
0.1, 1.0
*BOUNDARY
RP_NODE, 1, 2, 0.0
RP_NODE, 6, 6, 0.1
*END STEP
```

---

## 3. 실측 진단 결과 (Actual Diagnostic Execution)

`python scratch/test_rp_ur3_2d.py` 실행 결과:

```
Reading deck from C:\Users\GOODMAN\AppData\Local\Temp\tmp8xmy4s9v.inp...
read_inp successful!
RigidBody constraints registered: ['RB_1']
BCs in STEP1:
  BC BC_RP_NODE_DOF1_2: region=RP_NODE, u1=0.0, u2=0.0, ur3=None
  BC BC_RP_NODE_DOF6_6: region=RP_NODE, u1=None, u2=None, ur3=0.1

Attempting create_solver2d('STEP1')...
create_solver2d succeeded! num_dofs=10
Solver constraints count: 1
  Constraint 0: _UnimplementedConstraint
Solver fixed_dofs: {8: 0.0, 9: 0.0}

Attempting solver.solve_step(0.1)...

[DIAGNOSTIC RESULT] Failed with exception:
Traceback (most recent call last):
  File "D:\PythonCodeStudy\WHT_DispFoldSolver\scratch\test_rp_ur3_2d.py", line 62, in test_rp_ur3
    conv, iters = solver.solve_step(0.1)
  File "D:\PythonCodeStudy\WHT_DispFoldSolver\dispsolver\solver2d\dynamic2d.py", line 724, in solve_step
    return run_increment(self, 2, self._solve_step_newton, dt, f_ext, f_ext_n, ...
  File "D:\PythonCodeStudy\WHT_DispFoldSolver\dispsolver\solver2d\dynamic2d.py", line 761, in _solve_step_newton
    K_g, f_int = self.assemble_system(u_k, dt=dt)
  File "D:\PythonCodeStudy\WHT_DispFoldSolver\dispsolver\solver2d\dynamic2d.py", line 606, in _assemble_physical
    res = c.assemble(u_vec)
  File "D:\PythonCodeStudy\WHT_DispFoldSolver\dispsolver\model\model.py", line 69, in assemble
    raise NotImplementedError(
NotImplementedError: constraint 'RB_1' of type RigidBody has no solver implementation (it cannot be enforced, so the analysis would solve a different model)
```

---

## 4. 원인 분석 (Root Cause Analysis)

1. **파서 및 모델 등록 (`read_inp` / `model.RigidBody`)**:
   - `read_inp`는 `*RIGID BODY` 키워드를 정상적으로 파싱하여 `model.constraints["RB_1"] = RigidBody(...)`로 등록함.
   - `*BOUNDARY`의 DOF 6 (UR3) 역시 `DisplacementBC(ur3=0.1)`로 정상 등록됨.
2. **솔버 런타임 매핑 부재 (`_solver_constraint`)**:
   - `dispsolver/model/model.py`의 `_solver_constraint()`에서 `RigidBody`는 현재 `_UnimplementedConstraint`로 매핑되어 있음.
   - 런타임 `c.assemble(u_vec)` 호출 시 즉시 `NotImplementedError`를 발생시켜 실행을 중단함.
3. **`DynamicSolver2D`의 자유도 체계**:
   - `DynamicSolver2D`는 기본적으로 노드당 2개 자유도($U_x, U_y$)의 연속체 솔리드 메쉬 체계임.
   - 레거시 솔버(`dispsolver/solver/dynamic.py`)에서는 `RBE2HingeConstraint` 또는 `RBE2CondensationManager`를 통해 `u_extra`에 $\theta$ 자유도를 추가하거나 KKT/축약 기법을 적용했으나, 프로덕션 솔버 `DynamicSolver2D`에는 아직 `KinematicRBE2Constraint` 어셈블러가 직접 포팅되지 않음.

---

## 5. 결론 및 Package P4 권장 구현 방향

- **현 상태 결론**:
  - `dev_log/agent_work_packages_20261018.md` §P4의 선행 확인 결과, **2D 솔버(`DynamicSolver2D`)는 현재 `*RIGID BODY` 및 RP의 UR3 회전 처리를 지원하지 않음(`NotImplementedError`)**.
- **P4 진행 전략**:
  1. `example_disp_fold/abaqus_api_builder.py` 및 exporter는 완전한 Abaqus 규격 덱(`*RIGID BODY`, `*BOUNDARY` UR3)을 내보내도록 형식 정리를 진행함.
  2. 2D 폴딩 해석 시에는 `ex1_dispfold_teardrop.py`와 같이 회전 기구학적 예측기(`x_new = R(theta) @ (x - x_c) + x_c`)를 직접 적용하거나, `KinematicRBE2Constraint`를 `DynamicSolver2D`에 정식 포팅하는 후속 워크패키지를 통합 단계에서 진행할 것을 권장함.
