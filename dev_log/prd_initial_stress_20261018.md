# PRD: `*INITIAL CONDITIONS, TYPE=STRESS` 실행 연결

**작성일**: 2026-10-18
**목록 번호**: `abaqus_feature_gap_dispfold_20261018.md` 10번
**상태**: 3D 구현 및 2D 명시적 거부 완료 (2026-10-18). 테스트 `tests/test_initial_stress.py` 8개 통과. 변경 파일: `dispsolver/model/model.py`(`create_solver3d`, `create_solver2d`의 초기 응력 블록). 2D 계산과 나머지 3D 커널은 미구현.

## 1. 배경과 사용자

폴더블 디스플레이는 합지·경화 공정에서 층마다 잔류응력을 가진 채로 접힌다. 접힘 해석의 초기 상태를 Abaqus와 같은 방식(`*INITIAL CONDITIONS, TYPE=STRESS`)으로 지정할 수 있어야 한다.

## 2. 실측 현황 (2026-10-18, 직접 재현)

- 리더: `PredefinedField(field_type="STRESS")`로 저장됨. 정상.
- `Model.create_solver3d()`: `solver.element_states.get(gid)`에서 `AttributeError: 'DynamicSolver3D' object has no attribute 'element_states'`. `element_states`는 어느 솔버에도 없는 레거시 이름. 3D 솔버는 이미 `elem_stress_init (n_elems, 8, 6)`을 갖고 커널에 넘김.
- `Model.create_solver2d()`: 같은 `AttributeError`. 게다가 `DynamicSolver2D`에는 `elem_stress_init`이 없고, 2D 커널(`cpe4_numba.py` 등)은 시그니처로만 받고 본문에서 쓰지 않음. 즉 2D는 배선뿐 아니라 계산 자체가 없음.
- 어느 쪽도 테스트가 없음 (`tests/` grep 결과 없음).

## 3. 요구사항

R1. 3D: `*INITIAL CONDITIONS, TYPE=STRESS`로 지정한 요소집합의 초기 응력이 `solver.elem_stress_init`에 들어가고, u=0에서 내부력이 Bᵀσ₀가 된다.
R2. 성분 순서는 Abaqus 3D 순서(S11, S22, S33, S12, S13, S23). 데이터 라인에서 생략된 성분은 0 (Keywords Guide).
R3. 6개를 초과하는 값은 조용히 자르지 않고 오류.
R4. 3D에서 초기 응력을 소비하는 요소(`C3D8`, `C3D8I`, `C3D8R`, `C3D8H`)별로 R1을 확인. 소비하지 않는 요소가 있으면 조용히 무시하지 말고 명시적으로 거부.
R5. 2D: 이번 범위에서는 계산을 구현하지 않는다. 대신 `create_solver2d`가 `AttributeError`로 죽거나 조용히 무시하지 않고 "2D 미지원"이라는 명확한 오류를 낸다. 2D 계산 구현은 별도 PRD.

## 4. 비범위

- 2D 커널의 초기 응력 계산 (CPE4/4I/4R/4H, CR 래퍼 전부 수정 필요).
- 초기 응력의 자기평형화(Abaqus는 `*STEP` 첫 증분에서 불평형 힘을 그대로 해에 반영하는 것이 기본).
- 함수형 값(callable)의 가우스점별 평가. 현재 요소 중심값 근사를 유지하고 그렇게 문서화.
- `*INITIAL CONDITIONS, TYPE=STRESS, GEOSTATIC` 및 사용자 서브루틴.

## 5. 시임과 경계 분류

| 시임 | 경계 | 기대값의 독립 출처 |
|---|---|---|
| S1. 덱 → `Model` (`read_inp`) | 프로세스 내 | Keywords Guide의 데이터 라인 정의 |
| S2. `Model` → `create_solver3d` → `elem_stress_init` | 프로세스 내 | 입력 σ₀ 값 자체 (성분 순서와 0 채움은 Keywords Guide) |
| S3. `solver3d` 조립 f_int(u=0) | 프로세스 내 | 균일 σ₀=10, 단위 정육면체에서 한 면의 절점력 합 = σ₀·A = 10 N (연속체역학 정의, 구현과 무관) |
| S4. `create_solver2d` 거동 | 프로세스 내 | 요구사항 R5 (명시적 오류) |

대체물 없음. 실제 리더, 실제 솔버, 실제 Numba 커널을 쓴다.

## 6. 완료 기준

- S1~S4 테스트가 먼저 실패하고(이유 기록), 최소 수정으로 통과.
- `tests/inp_corpus`, 리더 테스트, 3D 요소 테스트 무회귀. 시작 시 실패 목록을 기록해 A/B 비교.
- `python -m verification.run_all --changed` 실행 (이번 변경은 `model/`만 건드리므로 해당 없음일 가능성이 높고, 그 결과를 그대로 기록).
- 사전 손질 패치 `.omc/handoffs/initial_stress_3d_fix_early.patch`는 테스트 없이 먼저 작성된 것이므로 참고만 하고, 테스트가 요구하는 만큼만 다시 적용.

## 7. 결과 (2026-10-18)

- 소비 확인된 커널: `C3D8` (NLGEOM 아래 k=1, C3D8_CR), `C3D8H` (k=2). u=0 내부력이 σ·A와 일치(1e-9).
- 소비하지 않아 `NotImplementedError`로 거부: `C3D8I`, `C3D8R` (측정: NLGEOM 아래 S11=10에서 f_int=0). 이전에는 어떤 요소든 `create_solver3d`에서 `AttributeError`로 죽어 이 문제에 도달하지도 못했음.
- 2D: `NotImplementedError("2D solver does not support initial stress ...")`.
- 미검증이라 거부 목록에 남은 커널: EAS TL(k=0), F-bar(k=3), C3D10M(k=4), C3D6(k=7), C3D4_ANP(k=9), C3D8I_CR(k=11), NLGEOM 끔 선형 C3D8(k=10) 등. 소스상 소비하는 것으로 보이는 것(k=0, 3, 7)은 테스트를 추가해 `stress_kernels`에 넣으면 지원됨.
- 후속: (a) 위 커널별 테스트 추가, (b) C3D8I/C3D8R 커널 구현, (c) 2D 커널 전체 구현(별도 PRD).
- 교훈 초안(통합자가 `LESSONS_LEARNED.md`에 반영): 파싱·저장되는 키워드가 있어도 솔버 생성 경로에서 레거시 속성(`solver.element_states`)을 참조해 도달 즉시 `AttributeError`로 죽고 테스트가 없어 방치됨. 키워드 "PARTIAL" 표기는 실행 경로 테스트로만 신뢰할 것. 또 소비하지 않는 커널이 있어도 조용히 무시되므로 요소별 u=0 내부력 검증이 필요.
