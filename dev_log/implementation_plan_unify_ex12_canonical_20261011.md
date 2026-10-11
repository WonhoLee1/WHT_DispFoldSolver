# [Implementation Plan] ex12 정본 승격 및 단일화 통합 (Canonical ex12 Unification Plan)

## 1. Goal Description
다른 에이전트(Claude Opus 5.5)가 규명한 **수치역학적 정밀도(원인 규명, 절점 회전 예측자 `carry_contacting_nodes`, 코너 필렛 접촉 모델, 엄밀한 잔차/변위 수렴 판정)**와 본 에이전트(Antigravity)가 구현한 **Abaqus Python API 객체 계층(`Model`, `Step`, `SolutionControl`, `ContactControl`), Numba 초고속 JIT 병렬 커널(3배 가속, 88초대), 및 Universal Distortion Control**을 단일 정본 `examples/ex12_abaqus_inp_plate_fold.py`로 완전 통합합니다.

---

## 2. User Review Required

> [!IMPORTANT]
> **단일 정본화 및 파일 역할 정리:**
> - 통합 후 공인 표준 스크립트는 `examples/ex12_abaqus_inp_plate_fold.py` 및 `examples/gen_ex12_inp.py`가 됩니다.
> - 기존에 작성된 `examples/ex12a_*` 파일들은 통합 정본의 기능이 완벽히 검증된 후, 레퍼런스 백업 또는 정리 대상으로 취급됩니다.
> - 디폴트 JIT 백엔드는 3배 빠른 **`numba`**로 설정하되, R&D 및 연구용 **`jax`** 백엔드도 플래그(`--elem_jit jax`)로 100% 동일하게 동작하도록 지원합니다.

---

## 3. Proposed Changes

### Component 1: Input Deck Generator (`examples/gen_ex12_inp.py`)
- `gen_ex12_inp.py`를 최신 `make_teardrop_config` 및 표준 Abaqus 요소 디스패치 체계와 동기화합니다.
- `PET`/`GLASS` 레이어는 `CPE4R`(ENHANCED hourglass), `PSA` 레이어는 `CPE4H`(하이브리드 $u-p$), 강체 플레이트는 `CPE4`로 지정.
- `OUTPUT = examples/ex12_rigid_plate_display_fold.inp`를 생성하며, 섹션 컨트롤 키워드(`*SECTION CONTROLS, DISTORTION CONTROL=YES, HOURGLASS=ENHANCED`) 완벽 지원.

### Component 2: Unified Benchmark Script (`examples/ex12_abaqus_inp_plate_fold.py`)
- **Abaqus API 컨테이너 도입:**
  ```python
  def build_abaqus_model_hierarchy(config: FoldModelConfig, result, preset: str = "ADAPTIVE") -> Model:
      model = Model(name="Ex12_FoldingModel", dim=2)
      step = model.Step(name="Step-1_Folding", procedure="STATIC", ...)
      sol_ctrl = model.SolutionControl(name="AdaptiveSolutionControl", preset=preset)
      step.solution_control = sol_ctrl
      ...
  ```
- **고급 접촉 기구학 (`RotatingRigidSegmentContact2D`) 통합:**
  - 코너 필렛 반경(`fillet_radius`), 접촉 연화(`softening_gap`), 유효 갭 공차(`stiffness_gap_tol`)
  - 강체 회전 선행 예측자(`carry_contacting_nodes`)를 통해 접촉면 절점을 회전 변위만큼 사전 이동시켜 뉴턴 반복 초기 추정치 제공.
  - 컷백 시 committed 상태로 복원 및 재예측 로직 완비.
- **초고속 Numba 디스패치 기본 활성화:**
  - `elem_jit="numba"` 기본값 설정.
  - 최근 구현 완료된 Numba Puso ENHANCED 아워글래스 `CPE4R` 및 Universal Distortion Control 강성 행렬 연동.

---

## 4. Verification Plan

### Automated Tests
1. **단위 및 접촉 역학 테스트 검증:**
   ```powershell
   pytest tests/test_rigid_segment_contact2d.py tests/test_cpe4r_element.py tests/test_distortion_control_barrier.py -v
   ```
2. **정본 `ex12` 스모크 테스트 (5 스텝):**
   ```powershell
   python examples/gen_ex12_inp.py
   python examples/ex12_abaqus_inp_plate_fold.py --max_steps 5
   ```
3. **정본 `ex12` 풀 90° 벤치마크 (101 스텝, 180° 합산 접힘):**
   ```powershell
   python examples/ex12_abaqus_inp_plate_fold.py
   ```
   **정량적 성공 기준:**
   - 101/101 스텝 완주 (t=1.000s, Target ±90.00°)
   - 0 cutbacks
   - Wall-clock time < 100초 (Numba 가속 검증)
   - 인터레이어 전단 슬립 계측치 정상 (PSA가 97% 이상 흡수)
   - 형상 시각화 이미지 `examples/ex12_final_folding_shape.png` 확인
