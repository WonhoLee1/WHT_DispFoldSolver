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
11. `*RESTART` — **2026-10-18 구현** (`dev_log/prd_restart_20261018.md`): `*RESTART, WRITE/READ`, `solve(restart=)`, `execute_inp(job, restart_dir, oldjob)`. 증분 중간·스텝 끝에서 이어 푼 결과가 끊김 없는 풀이와 1e-12 일치 (3D·2D, NLGEOM 켬·끔). 남은 것: `capture_state`가 없는 접촉 구속의 이력 복원 (경고로 보고됨), 동역학 스텝 재개 검증, 출력 이어쓰기.
12. `*ORIENTATION` — **2026-10-18 구현(3D)** (`dev_log/prd_orientation_20261018.md`): 직교이방성 `ENGINEERING CONSTANTS`가 3D 모든 솔리드 요소에서 동작하고 RECTANGULAR 방향과 추가 회전각이 강성에 반영됨. 남은 것: 요소마다 달라지는 프레임(원통·구), 2D 직교이방성 강성 자체, C3D20 + 방향.

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

## 6. 2026-10-18 후속 발견 (재시작·방향 작업 중 측정)

N1. **C3D8 코로테이셔널 커널(k=1, 16)이 소성 상태를 저장하지 않았음 — 수정함.** `c3d8_corotational_numba.py`가 `material_dispatch_3d`의 `sdv_new`를 계산만 하고 `sdvs`에 되쓰지 않아, 탄소성 C3D8이 이력 없는 비선형 탄성처럼 거동했다 (단조 비례 하중에서는 같은 답이라 가려졌고, 제하 후 잔류응력이 정확히 0이었음). 다른 3D 커널(C3D6, C3D10M, C3D8R, EAS, 하이브리드)은 되쓰고 있었다. `tests/test_c3d8_plastic_history.py`.
N2. **2D `elem_sdvs`가 `(n, 9, 7)` 고정이라 표형 J2(SDV 8개)가 소성 이력을 읽지 못했음 — 수정함.** 재료별 필요한 최대 SDV 수로 할당. CPE4(NLGEOM 끔)와 CPE4I가 이제 이력을 가짐. `tests/test_plastic_history_2d.py`.
N3. **2D 일부 커널이 재료 모델을 호출하지 않고 E, ν로 선형 탄성만 계산하던 문제 — 수정함** (`dev_log/prd_2d_material_wiring_20261018.md`). CPE4H, CPE4R, CPE4(NLGEOM)를 재료 디스패치에 연결하고, 그 뒤 CPE3, 6, 6M, 8, 8R도 연결해 2D 평면변형률 요소 9종 전부가 재료를 따른다 (정량 검증: 일축 변형 J2 닫힌 형식과 1e-6). 연결하지 않은 커널(CPE4 F-bar, 탄성 전용 co-rotational CPE4, 축대칭, 트러스)은 `elastic_only`로 선언해 비선형 재료를 거부한다.
N4. **프레임워크가 재료를 E, ν로만 넘기던 문제 — 수정함.** `AbaqusApiBuilder.solver_materials()`가 내보낸 덱의 재료를 솔버에 전달. PSA 체적 상수(D 고정 0.001 → K 2000 MPa, 스펙 8.33 MPa)도 함께 수정.
N6. **[열림] 초탄성의 유한변형 정확도**: 디스패치가 소변형 `eps`와 `F=I`를 받아 20% 신장에서 닫힌 형식 대비 −22% (CPE4I). PSA의 큰 전단 평가에는 부적합. 재료-요소 인터페이스 재설계(별도 PRD) 필요.
N5. **초기 응력 Voigt 순서 결함 — 수정함.** `dev_log/prd_initial_stress_20261018.md` §8.
N7. **`CPE4I`(EAS)의 소성 접선 불일치와 소성 굽힘 수렴 실패 — 수정함** (`prd_2d_material_wiring_20261018.md` §9). 보강 변위 α를 평형까지 반복. 소성 굽힘이 수렴하고 오라클과 2% 이내.
N8. **삼각형 co-rotational 프레임이 극분해 회전과 O(변형률) 달랐던 문제 — 수정함** (같은 문서 §9). 3D의 같은 성질(`cr_generic_3d` Kabsch)은 미수정.
N9. **[열림] 굽힘에서 `CPE4`(완전적분)와 `CPE4H`(평균 팽창)는 전단 잠김** (탄성 구간에서도 +9.6%, +8.2%). 접힘 모델에서 굽힘 지배층에 이 요소를 쓰면 안 된다. `ELEM_CASES_2D`의 `plate: CPE4`는 강체에 가까운 판이라 무방하나 `2D-Base`(solid: CPE4)는 부적합.
