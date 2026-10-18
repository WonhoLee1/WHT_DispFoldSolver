# example_disp_fold 대비 Abaqus 미구현 기능 목록 (구현 용이도 순)

**작성일**: 2026-10-18
**범위**: `example_disp_fold/` 접힘 프레임워크가 필요로 하는 Abaqus 기능 중 솔버(`dispsolver/`)에 없는 것.
**근거**: `example_disp_fold/*.py`, `dispsolver/model/`, `dispsolver/io/keywords/`, `dispsolver/output/`, `dispsolver/element/` 소스 직접 확인. 리더 등록 키워드 76종.
**미확인 표시**: "(미확인)"은 코드로 확인하지 못한 항목. 구현 전에 먼저 확인할 것.

## 0. 현재 상태 (확인됨)

- `example_disp_fold`는 `Model`/`Step`/`FieldOutputRequest` API를 쓰지 않는다. `solver.fix_dof`로 회전 Dirichlet BC를 걸고 수동 Newton 루프를 돈다.
- 내보내는 `model.inp`는 실제 해석과 다르다. `*RIGID BODY`, `*TIE`, 회전 BC(UR3), `*CONTACT PAIR`, `*OUTPUT`이 없고 핀 BC와 주석만 있다 (`abaqus_api_builder.py:177-208`).

## 1. 쉬움: 백엔드는 있고 리더·연결만 필요

1. `*CONTROLS` 리더 키워드. `Model.SolutionControl`은 이미 있음.
2. `*CONTACT CONTROLS`, `*CONTACT STABILIZATION` 리더 키워드. `Model.ContactControl`, `ContactStabilization`은 이미 있음.
3. 출력 변수 추가: `ALLAE`, `ALLSD`, `ALLFD`, `CSHEAR`, `COPEN`, `CSTATUS`가 `output/variables.py`에 없음. `MISES`, `MAXP`/`MINP`도 이름으로는 없음 (다른 곳에서 파생될 수 있음, 미확인). 기존 S와 접촉 데이터에서 계산 가능.
   - 접힘 의미: UTG 유리 최대 주응력(파단 판정), 안정화 에너지 비율(해 품질 게이트), 접촉압.
4. exporter 패리티: `*RIGID BODY`, UR3 회전 BC + `*AMPLITUDE`, `*TIE`, `*SURFACE INTERACTION`, `*CONTACT PAIR`, `*OUTPUT`을 내보내고 `execute_inp`로 손루프 결과와 대조.
   - 선행 확인: 2D 솔버가 RP의 UR3 BC를 실행하는지 (미확인). 키워드 맵에는 UR3가 있음.
5. 다단계 접힘: 접기 → 유지(PSA 이완) → 펴기 → 반복. `MultiStepExecutor`는 있으나 프레임워크가 단일 스텝만 사용. 영구변형(springback) 평가에 필요.
6. `*SPRING`, `*DASHPOT` 리더 키워드. 요소 `SPRING1`/`SPRING2`는 카탈로그에 있음. 우선순위 낮음.

## 2. 중간

7. `*PLASTIC, HARDENING=KINEMATIC/COMBINED`, `*CYCLIC HARDENING`. 커널 `material/plastic_hardening.py`는 있으나 어떤 솔버도 참조하지 않음. 리더는 ISOTROPIC만 허용 (`keywords/material.py:278`). 반복 접힘 PET의 Bauschinger 효과에 필수. Chaboche 지원 범위는 미확인.
8. `*STATIC, STABILIZE`. 리더 `Static`은 `direct`만 받음. 안정화는 레거시 `solver/stabilization.py`에만 있고 solver2d/3d에는 없음 (접촉 채터링 안정화는 별개). 포팅 필요.
9. `*VISCO` 스텝 (시간영역 점탄성, CETOL). 스텝 종류는 STATIC/DYNAMIC뿐. 프레임워크는 PSA Prony(τ 0.05, 0.5 s)를 total_time=1.0과 함께 쓰므로 접힘 속도에 민감. STATIC dt가 물리 시간으로 쓰이는지 먼저 확인 (미확인).
10. `*INITIAL CONDITIONS, TYPE=STRESS`. **2026-10-18 부분 해결**: 3D `C3D8`/`C3D8H`(NLGEOM 켬·끔 모두)는 동작, 나머지 커널과 2D는 명시적 거부 (`dev_log/prd_initial_stress_20261018.md`). 종전에는 `create_solver*`에서 `AttributeError`. 열팽창(`*EXPANSION`)은 있음. 합지 잔류응력·curl 재현용.
11. `*RESTART`. `result_io`는 결과 저장만 있고 재시작은 없음.
    - **2026-10-18 조사**: `MultiStepExecutor`에 메모리 내 `StateCheckpoint`(u, v, a, `elem_sdvs`, 누적 시간)와 스텝 분기용 복원이 있고 `tests/test_multi_step_*.py`가 있음. 디스크 재시작은 없음(분기 결과 pkl만 저장). 리더에 `*RESTART` 미등록.
    - 체크포인트에 빠진 상태: 접촉 마찰 상태(`commit_friction_state` 대상), 갱신 라그랑지안 기준 `_ul_F_n`, 구속 객체 내부 상태, dt 제어기. 이대로 재개하면 모델이 달라질 수 있음.
    - 판단: 설계(PRD)가 필요한 별도 작업. 이번 라운드에서는 구현하지 않음.
12. `*ORIENTATION`은 PARTIAL. 재료 커널에 미연결. 이방성 필름용.
    - **2026-10-18 조사**: 리더는 파싱·저장하고, 이방성 재료(`ENGINEERING CONSTANTS`)에 방향이 걸린 섹션은 `orientation-not-connected`로 **정의를 거부**(조용히 무시하지 않음, `keywords/material.py` `finalize_materials`). 등방성 재료는 정보 로그만 남기고 영향 없음.
    - 폴더블 프레임워크의 재료(PET, PSA, GLASS, PLATE)는 모두 등방성이라 현재 영향 없음. 이방성 필름을 쓰게 될 때까지 보류.

## 3. 어려움

13. 점착(cohesive) 요소 `COH2D4`/`COH3D8` + `*COHESIVE BEHAVIOR`, traction-separation. 요소 카탈로그에 전혀 없음. PSA 박리에 중요. 새 요소군이라 가장 큼.
14. `*SUBMODEL`. 유리 국부 응력용.
15. 커넥터·빔 요소 (`CONN2D2`, `B21`). 힌지 기구 모델링용.
16. `*SHELL TO SOLID COUPLING`. 전체 shell + 국부 solid.
17. `*STATIC, RIKS`. 변위 제어 접힘이라 필요성 낮음.

## 4. 권장 순서

3 → 1·2 → 4 → 5 → 7 → 9.
- 3, 1, 2는 반나절급. 4·5가 끝나면 프레임워크를 Abaqus 덱 기준으로 검증 가능.
- 7은 반복 접힘 정확도의 핵심.
- 4와 9는 착수 전 위의 "미확인" 선행 확인을 먼저 할 것.

## 5. 기존 문서와의 관계

- 중복 방지: `release_notes_keyword.md` §8(명시 중단 키워드 로드맵), `dev_log/area_analysis_*_20260919.md`의 Open 항목, `dev_log/implicit_solver_parity_report_20260913.md`.
- 본 문서는 접힘 프레임워크 관점의 미구현 목록이며, 위 문서들의 Open 항목은 다시 적지 않았다.
