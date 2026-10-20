# PRD: `*RESTART` (재시작 쓰기/읽기)

**작성일**: 2026-10-18
**목록 번호**: `abaqus_feature_gap_dispfold_20261018.md` 11번
**브랜치**: `feat/p7-restart-orientation` (P3과 같은 파일을 만지므로 별도 워크트리)
**상태**: 구현 완료 (2026-10-18). `tests/test_restart.py` 16개 통과.

## 1. 배경

접힘 해석은 수 분~수십 분이 걸리고 반복 접힘(다단계)은 더 길다. 중단·실패 지점에서 처음부터 다시 풀지 않고, 또는 접은 상태에서 다른 스텝(유지, 펴기)을 새로 붙여 풀 수 있어야 한다. 현재는 `MultiStepExecutor`의 메모리 내 체크포인트(u, v, a, `elem_sdvs`)만 있어 프로세스가 끝나면 사라지고, 마찰·접촉 구속 상태와 시간 증분 제어기 상태는 담지 않는다.

## 2. 현황 (코드 확인)

- `solve()`(3D `dynamic3d.py`, 2D `dynamic2d.py`)는 증분 루프의 상태를 지역 변수 `t`, `step_inc`, `inc_no`, `total_iters`, `dt_ctrl`에 모아 둔다.
- 구속 객체에는 컷백 롤백용 `capture_state()`/`restore_state()` 프로토콜이 있다. 구현체는 `DeformableSurfaceContactConstraint3D` 하나뿐이고, 다른 구속은 이 메서드가 없다.
- `AdaptiveDtController`의 상태는 `dt, dt_min, dt_max, target_iters` (속성 전부).
- 리더에 `*RESTART`는 등록돼 있지 않다.

## 3. 요구사항

R1. 상태 캡처: u, v, a, `elem_sdvs`, 구속 상태(`capture_state` 보유분), `nlgeom`/`_nlgeom_engaged`, `_u_step_start`, 온도 필드, dt 제어기, 스텝 이름·스텝 시간·증분 번호·누적 반복 수.
R2. 파일: 순수 dict(numpy 배열과 내장형만, 클래스 인스턴스 없음)를 pickle로 저장. `result_io`와 같은 원칙으로 리팩터링해도 옛 파일이 깨지지 않게 한다. 버전 필드를 둔다.
R3. 모델 지문: 자유도 수, 요소 수, 요소 종류·재료 배열 해시. 읽을 때 지문이 다르면 오류 (다른 모델에 상태를 얹지 않는다).
R4. 증분 단위 재개: 재개한 풀이의 최종 결과가 끊김 없이 푼 결과와 일치한다 (고정 증분, 소성 SDV 포함, 1e-12).
R5. 스텝 경계 재개: 완료된 스텝 이후부터 새 스텝을 실행한다.
R6. 리더 `*RESTART, WRITE, FREQUENCY=n[, OVERLAY]` (모델 수준 또는 스텝 수준)와 `*RESTART, READ, STEP=s[, INC=i | END STEP]`. 이전 작업은 Abaqus처럼 덱 밖에서 `execute_inp(..., oldjob=)`로 지정한다 (`OLDJOB`은 `*RESTART`의 파라미터가 아니라 명령행 옵션). 지원하지 않는 파라미터는 경고 원장에 보고한다.
R7. 상태를 담지 못하는 구속(`capture_state` 없음)은 침묵하지 않는다: 메타데이터에 이름을 기록하고 읽을 때 경고 목록에 올린다.
R8. 2D와 3D 둘 다.

## 4. 비범위

- 구속 객체 각각에 `capture_state` 신규 구현 (접촉 마찰 이력 등은 기존 보유분만 저장). 필요한 구속은 후속 과제로 열거한다.
- `*RESTART, WRITE`의 `NUMBER INTERVAL`, `TIME MARKS`, `ENDINCREMENT`(제외 시 거부).
- Abaqus 바이너리 `.res/.prt/.mdl` 호환.
- 출력(ODB) 이어쓰기.

## 5. 시임과 경계 분류

| 시임 | 경계 | 기대값의 독립 출처 |
|---|---|---|
| S1. `restart.capture/apply/save/load` 라운드트립 | 프로세스 내 (실제 솔버) | 저장 전 배열과 비트 동일 |
| S2. 증분 중간 저장 → 새 솔버로 재개 → 완료 | 프로세스 내 | 끊김 없는 같은 풀이의 최종 u, SDV (불변식, 코드 출력이 아님) |
| S3. `.inp`의 `*RESTART` WRITE → 파일 생성, READ → 이어서 풀이 | 프로세스 내 + 파일시스템(tmp_path) | S2와 같은 끊김 없는 풀이 |
| S4. 지문 불일치, 알 수 없는 파라미터, 상태 미보유 구속 | 프로세스 내 | 요구사항 R3, R6, R7 |

대체물 없음. 실제 리더, 실제 솔버, 실제 파일.

## 6. 완료 기준

- S1~S4 모두 먼저 실패하고 최소 수정으로 통과. 2D와 3D 각각.
- 시작 시 전체 `pytest tests/` 실패 목록 기록 후 A/B 비교.
- `python -m verification.run_all --changed` 실행.
- 에이전트용 hand-off (AGENTS.md §3.2 형식).

## 7. 결과 (2026-10-18)

- 신규 `dispsolver/solver/restart.py`(캡처·적용·저장·읽기·지문·기록 찾기), `dispsolver/io/keywords/restart_kw.py`(리더), `solve(..., restart=)`(3D·2D), `step.restart_write`, `MultiStepExecutor.execute_step(restart=)`, `execute_inp(job, restart_dir, oldjob)`.
- 불변식 검증: 증분 5에서 쓴 기록을 새 솔버에 읽혀 이어 푼 결과가 끊김 없는 풀이와 일치 (u, SDV 모두 1e-12; 3D C3D8, 2D CPE4I; NLGEOM 켬·끔). 기록을 쓰는 것이 답을 바꾸지 않음(비트 동일). 이어 푼 쪽 증분 수는 정확히 나머지 5.
- 스텝 경계 재개(`END STEP`)는 나머지 스텝만 실행하고 `steps_executed`에 반영.
- 안전장치: 모델 지문 불일치(`different model`), 파일 버전 불일치, 기록 없음(`FileNotFoundError`에 기대 파일명), 상태 미보유 구속은 이름을 기록하고 읽을 때 경고. 변이 검사로 구속 복원 테스트가 실제로 잡는 것을 확인.
- 실제 Abaqus 덱 형태 대응: `FREQ=0`은 기록 안 함, 축약 파라미터(`FREQ`, `NUM`, `TIMEMARKS`), `READ`와 `WRITE` 동시. (`benchmark_inp`에 `*RESTART` 줄이 수백 개 있음. 코퍼스 스윕이 처음 구현의 과도한 엄격함을 잡았음.)
- **한계(열린 항목)**: (a) `capture_state`가 있는 구속은 `DeformableSurfaceContactConstraint3D` 하나뿐이라 다른 접촉 구속의 이력(마찰 슬립, 활성 접촉 집합)은 새로 시작. 경고로 보고됨. 이 구속들에 상태 캡처를 구현해야 접촉 모델의 재시작이 완전해짐. (b) 동역학(`*DYNAMIC`) 스텝의 재개는 `a`, `v`를 복원하지만 별도 검증 안 함. (c) `NUMBER INTERVAL`, `TIME MARKS`는 기록 시점 선택이라 경고만. (d) 출력(ODB/VTKHDF) 이어쓰기 없음. (e) `fields`/온도 필드는 `current_temperature`만 담음.
- **재시작 테스트 중 발견·수정한 별개 결함**: `dev_log/abaqus_feature_gap_dispfold_20261018.md` §6 N1~N4.
