See `AGENTS.md` in this same directory — it is the canonical project rule
file (target problem, solver theory, element JIT strategy, and numerical pitfalls
already solved) shared across Claude Code, Gemini CLI, and OpenCode. Read it before
making changes here.

Element JIT Strategy: Use `--elem_jit jax` for R&D/prototyping (AutoDiff), and switch to `--elem_jit numba` for production run speed.

---

# ⚠️ [필독/절대엄수] LESSONS_LEARNED.md 기반 실수 및 안티패턴 재발 방지 명령
> **루트 문서 필독**: 모든 작업(코드 수정, 디버깅, 벤치마크, 결과 보고) 전 반드시 루트의 [`LESSONS_LEARNED.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/LESSONS_LEARNED.md)를 먼저 확인해야 한다.
> 
> **교훈 누적 기록 의무 (Mandatory Lesson Recording)**:
> 향후 비선형 유한요소 수치해석, 접촉 역학, 솔버 알고리즘 개발 및 디버깅 과정에서 새롭게 도출되는 **모든 수치역학적 발견, 버그 원인 분석, 실패 사례 및 엔지니어링 교훈(Lessons Learned)은 다른 임의의 위치에 분산시키지 말고 반드시 루트의 [`LESSONS_LEARNED.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/LESSONS_LEARNED.md) 파일에 지속적으로 누적하여 포함·기록해야 한다.**
> 
> **과거 사후 감사(Audit)에서 적발된 아래의 8대 치명적 결함 및 안티패턴을 절대로 다시 범하지 말 것:**
> 1. **실행 경로 이중화(Dual Path) 오인 금지**: `.inp` 구동 경로인 `_solve_step_with_hooks`를 건너뛰고 미사용 함수(`solve_step`)만 수정한 뒤 "버그 해결 완료"라고 착각·보고하는 행위 엄단.
> 2. **비물리적 가짜 동결(Freeze) 및 허위 "Abaqus 표준" 포장 금지**: 접촉 활성 집합 진동을 회피하려고 양방향 스프링(인공 인장/접착)을 허용하고, 미동결 침투를 묵인하며, 수렴 전 접촉 상태를 재검증하지도 않으면서 "Abaqus 표준"이라고 거짓 포장하는 행위 절대 금지.
> 3. **안전장치 무력화 목적의 매직 넘버(Magic Number) 삽입 금지**: Line Search에서 `min_r_norm < 1e5`와 같은 거대 상수로 잔차 증가/발산 스텝을 무조건 수락하는 꼼수 금지.
> 4. **수렴 기준 선언과 실체의 불일치 및 단위계 종속(Dimensionality) 오류 금지**: 스파이크 $r_0$ 기준 fallthrough 통과, 10 N 잔차 방치, $1.0\,\text{N}$, $10^{-4}\,\text{mm}$ 등 특정 단위계 절대값 공차 사용 금지. 특성 하중($q_{\text{avg}}$) 기반 무차원화 표준 준수.
> 5. **외력 하중 적분기 기하학 오류와 무의미한 1회 수렴 자화자찬 금지**: DSLOAD 등 표면 압력 하중의 법선 방향 및 반경 방향 합력 검증 없이 외력이 0에 가까워 구조물이 반응하지 않은 것을 솔버 우수성으로 왜곡 금지.
> 6. **출력 파일(.sta) 하드코딩 모킹(Mocking) 절대 엄단**: `run_inp.py`에 고정 텍스트(`INC=1, ATT=1`)를 박아놓고 "Cutback 0회 수렴"이라고 허위 보고하는 행위 엄단. 솔버 내부 상태 변수와 1:1 직접 바인딩 필수.
> 7. **땜질식 0-간극 공차(`1e-5`) 한계 인정 및 위상학적 외향 법선 준수**: 억지끼움(> 10 nm)에서 법선이 뒤집히는 땜질을 피하고, 솔리드 요소 면 절점 위상(CCW) 기반 외향 단위 법선을 철저히 준수할 것.
> 8. **무단 기본값 변경, 미구현 주석, 메타데이터 환각(Hallucination) 근절**: 전역 기본값 변경 금지, 구현 없는 이론명 주석 금지, 입력 데크 물성은 반드시 직접 `view_file`로 열람 확인 후 인용할 것.

---

# 🚀 Contact Mechanics & Abaqus Python API Parity (Canonical Reference)
> **최신 릴리즈 명세서**: [`release_notes_contact.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/release_notes_contact.md) (지속 업데이트 및 최신 유지 필수)  
> **핵심 기능**:
> - **Abaqus API Parity**: `ContactStd(useAllExterior=True)` 무설정 자동 접촉, `ContactPair(sliding="FINITE")`, `ContactProperty`, `ContactControl`
> - **Dual Mortar 초고속화**: Zero-Allocation 스택 핑퐁 Numba 클리핑 (`buf[2, 16]`, ~60ns) + Flat Linear BVH (Morton LBVH 2D Array) $O(N)$ 병렬 리핏
> - **순수 SPD 보존**: Wohlmuth 쌍직교 대수 응축 ($u_s = D^{-1} M u_m$)으로 안장점(Saddle-point) 완전 배제 $\to$ Intel oneMKL PARDISO Cholesky ($LL^T$) 100% 호환
> - **실시간 로그 스트리밍**: `tests/logs/`에 스텝별 뉴턴 반복수, 잔차, 에너지 수지 실시간 기록
> - **통합 검증 스위트**: `pytest tests/test_surface_to_surface_contact.py tests/test_flat_bvh_tree3d.py tests/test_finite_sliding_mortar3d.py tests/test_contact_advanced_benchmarks.py tests/test_abaqus_contact_api.py tests/test_standard_vs_dual_mortar.py tests/test_standard_vs_dual_mortar_2d.py tests/test_contact_pair_mortar_dispatch.py tests/test_dual_mortar_2d.py` (**58/58 All Green**)

---

# 📦 릴리즈 및 영역별 분석 인덱스 (단일 소스 — 본 파일만 수정할 것)
> opencode 로컬 설정(루트 `AGENTS.md`)은 본 섹션을 참조한다. 내용은 여기에만 쓴다.

- **누적 교훈 및 안티패턴 종합 문서 (필독/절대엄수)**: [`LESSONS_LEARNED.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/LESSONS_LEARNED.md) (모든 수치해석 교훈 누적 기록 단일 소스, 기적발된 8대 결함 및 수치적 실수 재발 절대 금지)


- **Contact 릴리즈 명세**: [`release_notes_contact.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/release_notes_contact.md) (최신 유지 필수)
- **영역별 릴리즈 노트** (루트, 한 줄 상태 + 상세 링크):
  - Material: [`release_notes_material.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/release_notes_material.md)
  - Constraint: [`release_notes_constraint.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/release_notes_constraint.md)
  - Solver: [`release_notes_solver.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/release_notes_solver.md)
  - Element: [`release_notes_element.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/release_notes_element.md)
  - Keyword (inp 리더·실행기): [`release_notes_keyword.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/release_notes_keyword.md)
- **영역별 분석 문서** (`dev_log/`, living doc — 실측 기반만 기재, Open 항목이 다음 작업):
  - Material (온도/열팽창/고급소성/초탄성/점탄성): [`area_analysis_material_20260919.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/area_analysis_material_20260919.md)
  - Constraint (tie/RBE2/접촉구속): [`area_analysis_constraint_20260919.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/area_analysis_constraint_20260919.md)
  - Solver (Newton 수락/SDI/회귀): [`area_analysis_solver_20260919.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/area_analysis_solver_20260919.md)
  - Element (2D/3D/판정치): [`area_analysis_element_20260919.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/area_analysis_element_20260919.md)
  - Contact 본판: 위 릴리즈 명세 + `contact_mechanics_hardening_and_dual_mortar_20260919.md`
- **규칙**: 새 실측 및 교훈이 나오면 반드시 루트의 [`LESSONS_LEARNED.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/LESSONS_LEARNED.md)와 해당 영역 문서의 Open 항목에 즉시 누적 기록한다. 본 인덱스의 링크가 깨지지 않게 파일명 변경 시 함께 수정.


---

# 🧠 Flexible Display Pure Bending Roll-Up Benchmark & Nonlinear Finite Element Know-How
(Added 2026-09-13 based on 10-candidate 180° pure bending roll-up benchmark & reaction equilibrium verification)

## 1. 외형(Kinematic Silhouette) vs 역학(Mechanics Moments)의 분리
- **변위 제어 롤업의 맹점**: 끝단에 기구학적 순수 굽힘 변위($u_x, u_z$)를 강제 처방하면, 전단/체적 잠김이 심각하게 발생한 불량 요소(`2D-Base`, `2D-CR`, `3D-Base`, `3D-CR`)조차 외형상으로는 오차 < 0.1%의 '완벽한 180° 반원 U-Shape'을 형성함.
- **잠김 감별의 절대 기준**: 외형 실루엣만으로 성공 여부를 판정해서는 안 되며, 반드시 **단면 반력 모멘트($M$)**를 이론 정해($M_{\text{theory}} = 0.2317\,\text{N}\cdot\text{mm}$)와 대조해야 함. 체적 잠김 발생 시 반력 모멘트가 $1.738\,\text{N}\cdot\text{mm}$ (7.5배)로 폭등함.

## 2. Co-Rotational (CR) 외피 × 국소 TL/u-P 하이브리드 결합 원칙
- **CR 단독의 한계**: Co-Rotational 프레임워크는 $180^\circ$ 대회전(Large Global Rigid Rotation)을 제거하여 회전 비선형성을 처리하는 '외곽 필터(Outer Hull)'일 뿐, 요소 내부 국소 변형률을 완전히 선형 탄성 수준으로 유지해주지 못함.
- **PSA 층간 슬립과 대변형**: 180° 롤업 시 비압축성 점착제(PSA, $t=30\,\mu\text{m}$, $\nu=0.499$)는 $210\,\mu\text{m}$의 층간 슬립을 흡수하므로 국소 전단변형률이 $\gamma_{xz} \approx 700\%$에 달함.
- **이중 계층 결합**: 따라서 초탄성/비압축성 재료를 포함하는 대변형 적층 박막 해석에서는 CR 외피 내부에 **Herrmann u-P 혼합 정식화(`CPE4H`, `C3D8H`)** 또는 **F-bar Total Lagrangian 정식화**가 반드시 국소적으로 결합되어야 체적 잠김이 완벽히 소거됨.

## 3. 대변형 단면 반력 모멘트 계산 좌표계 표준화
- **정적 평형 조건**: 순수 굽힘(Pure Bending) 변형 상태에서 외부 횡하중이 없으므로 고정단 모멘트와 구동단 모멘트는 크기가 같고 방향이 반대인 작용-반작용 쌍을 이루어야 함:
  $$|M_{\text{root}}| = |M_{\text{tip}}| = M_{\text{theory}}$$
- **변형된 현재 좌표계 필수**: 반력 모멘트 적분 시 모멘트 팔 $\mathbf{r} = \mathbf{x} - \mathbf{x}_c$는 반드시 초기 참조 좌표 $\mathbf{X}$가 아닌 **변형된 현재 좌표 $\mathbf{x} = \mathbf{X} + \mathbf{u}$**를 기준으로 계산해야 함. 초기 좌표계를 쓰면 굽힘 반경 중심 이동으로 인해 작용-반작용 평형이 겉보기상 깨지는 오차가 발생함.
- `DynamicSolver2D.compute_section_reactions()` 및 `DynamicSolver3D.compute_section_reactions()`에 표준화 적용 완료.

## 4. 10개 후보군 벤치마크 결과 및 최우수 권장 요소
| 차원 | 추천 요소 조합 (PET + PSA) | 반력 모멘트 $M_{\text{root}}$ | 모멘트 오차 | 중립면 슬립 | 비고 |
|:---:|:---|:---:|:---:|:---:|:---|
| **2D 최우수** | **`2D-Opt1` (`CPE4I_CR + CPE4H_CR`)** | **$-0.2307\,\text{N}\cdot\text{mm}$** | **-0.4%** | **$209.9\,\mu\text{m}$** | 9-모드 EAS + Herrmann u-P 완전 적합 |
| **2D 대안** | **`2D-Opt2` (`CPE4R_CR + CPE4H_CR`)** | $-0.2224\,\text{N}\cdot\text{mm}$ | -4.0% | $209.7\,\mu\text{m}$ | 1점 감차적분 아워글래스 제어 (가장 빠름) |
| **3D 최우수** | **`3D-Opt1` (`C3D8I_CR + C3D8H_CR`)** | **$+0.2307\,\text{N}\cdot\text{mm}$** | **-0.4%** | **$209.9\,\mu\text{m}$** | 2D-Opt1과 소수점 6자리 일치, 3D 최고 정밀도 |
| **3D 고속** | **`3D-Opt2-H` (`C3D8R_CR + C3D8H_CR`)** | **$+0.2544\,\text{N}\cdot\text{mm}$** | **+9.8%** | **$209.9\,\mu\text{m}$** | 1점 감차적분 + u-P, 계산 안정성 최우수 |
| *(체적잠김)* | `2D-CR` / `3D-CR` (`_CR + _CR`) | $\pm 1.736 \sim 1.738\,\text{N}\cdot\text{mm}$ | +650% (7.5배 폭등) | $209.9\,\mu\text{m}$ | 비압축성 PSA에 u-P 결합 누락 시 체적 잠김 |

## 5. PyVista 3D 대화형 뷰어 사용법
- **10개 케이스 대화형 뷰어 실행**:
  ```bash
  python benchmark_element/view_rollup_pyvista.py --case "3D-Opt1"
  python benchmark_element/view_rollup_pyvista.py --case "2D-Opt1"           # 2D 3D 압출 뷰 (기본값)
  python benchmark_element/view_rollup_pyvista.py --case "2D-Opt1" --flat-2d  # 순수 2D 평면 시트 뷰
  python benchmark_element/view_rollup_pyvista.py --case "3D-Opt2-H"
  ```
- **키보드 단축키**:
  - `1`: 정면도 (XZ 평면, 180° 반원 궤적)
  - `2`: 측면도 (YZ 평면, 단면)
  - `3`: 평면도 (XY 평면)
  - `4`: 등각투영 (Isometric 3D)
  - `P`: 투시(Perspective) ↔ 직교(Orthographic) 투영 토글
  - `C`: 변위 컨투어 ↔ 재질 레이어(PET/PSA) 색상 토글
  - `E`: **2D 모델 모드 토글** ([순수 2D 평면 시트] ↔ [3D 압출 솔리드 (폭 1.0mm)])
  - `S`: 고해상도 스크린샷 PNG 저장
  - `Q`: 종료
- **전 후보군 스크린샷 일괄 캡처 및 2×5 몽타주 생성**:
  ```bash
  python benchmark_element/view_rollup_pyvista.py --capture_all
  ```
  생성 위치: `dev_log/figures/rollup_deformed_shapes_montage.png`

---

# 🚫 수치해석 구현 눈속임(Mocking) 금지 및 무결성 검증 원칙 (Anti-Mocking Protocol)

최신 유한요소 및 비선형 해석 알고리즘 구현 시 AI 에이전트가 흔히 범하는 **난수 대입, 가짜 더미(Mocking), 허위 통과 보고를 원천 차단**합니다:

1. **난수(np.random.rand) 및 더미 행렬 대입 절대 엄단**
   - 강성 행렬($B, K$), 내력 벡터($f_{int}$), 잔차 및 변형률 계산 시 `np.random.rand`, 하드코딩된 임의의 축척 계수 대입을 엄격히 금지합니다.
   - 복잡한 수식이라는 핑계로 물리적 기구학 전개를 생략하거나 껍데기만 만드는 행위는 전산역학적 결함으로 간주합니다.
   - 모든 유한요소 수식은 공인 원전(Bathe, Simo, Hughes 등)의 수학적 정식화에 입각하여 한 줄 한 줄 엄밀하게 유도·구현해야 합니다.

2. **서브에이전트 산출물 자가 승인 금지 및 2차 독립 검증 (Independent Verification)**
   - 서브에이전트가 "테스트 통과 / 구현 완료"라고 보고하더라도 메인 에이전트는 이를 무비판적으로 수용하지 않습니다.
   - 메인 에이전트는 작성된 코드를 반드시 직접 열어(`view_file`) 난수/모킹 유무를 라인 단위로 전수 검사합니다.
   - 벤치마크 및 물리적 무결성(강체 운동 6개 모드 고윳값 분석 $K_e \mathbf{u}_{rbm} = \mathbf{0}$, 왜곡 패치 테스트 $10^{-9}$ 기계 정밀도 대조)을 메인 프로세스에서 직접 독립 실행하여 검증한 후에만 최종 승인합니다.

3. **안드레 카파시(Karpathy) AI 엔지니어링 4대 원칙 재확립**
   - **Think Before Coding**: 가정을 명시하고, 수치적 한계와 트레이드오프를 사용자에게 솔직히 공개할 것.
   - **Simplicity First**: 단일 사용 목적의 과도한 추상화나 불필요한 장식성 코드를 배제하고, 가장 직관적이고 군더더기 없는 최소 코드로 구현할 것.
   - **Surgical Changes**: 건드려야 할 부분만 최소한으로 수정하며, 기존에 검증된 멀쩡한 코드를 임의로 훼손하지 말 것.
   - **Goal-Driven Execution**: 주관적인 "작동하는 것 같다"는 평가를 배제하고, "Abaqus 이론해 대비 오차 < 1e-8%", "RBM 외 Spurious Mode = 0"과 같이 반증 가능한 정량적 지표를 세우고 검증 루프를 돌 것.

4. **`LESSONS_LEARNED.md` 상시 대조 및 과거 결함·실수 재발 원천 차단**
   - 모든 에이전트는 코드 작성, 디버깅, 벤치마크 평가 시 반드시 [`LESSONS_LEARNED.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/LESSONS_LEARNED.md)의 8대 안티패턴을 사전 체크리스트로 확인해야 함.
   - 실행 경로 미추적(이중 경로 패치 누락), 가짜 `.sta` 하드코딩, Line Search 매직 넘버 바이패스, 단위계 종속 및 스파이크 $r_0$ fallthrough 수렴 조작, 0-간극 땜질, 비물리적 동결 접착 등 **과거에 적발된 실수를 단 하나라도 되풀이할 경우 치명적 결함으로 간주하고 즉각 작업을 중단 및 롤백함**.





---

# 💡 Phase 2 비선형 해석방법론 벤치마크 개발 교훈 (Lessons Learned & Numerical Engineering Insights)
(Added 2026-10-03 based on Phase 2 Methodology Benchmarks #01~#04 execution)

## 1. 음체적(Negative Volume) 요소를 가진 3D Mesh 구동 금지
- **기구학적 결함 원인**: 인접 절점 순서(Connectivity Node Order)가 뒤바뀌어 `det J < 0`인 요소가 포함된 메쉬는 강성 행렬($K$)을 부정부호(Indefinite / Non-SPD)로 만들어 Newton-Raphson 수렴을 원천적으로 막음.
- **방지 대책**: `DynamicSolver3D._check_initial_element_orientation()` 메쉬 구동 전 전수 검사로 `det J <= 0` 요소 감지 시 비선형 수치해석 시작 전 즉시 `ValueError` 차단.

## 2. 대변형/체적압축 3D 초탄성(`C3D8H`) 접촉 residual 수락 조건
- **수치적 상대 잔차 역설(Relative Residual Paradox)**: 25% 공칭 수직 압축 등 고체적 가압 접촉 해석에서는 내력 및 접촉 반력이 $> 100,000\,\text{N}$에 달함. 수치적 잔차 노름($r_{\text{norm}}$)이 소거되더라도 스텝별 초기 잔차 노름($r_{0,\text{norm}}$) 갱신으로 인해 relative residual($r_{\text{norm}} / r_{0,\text{norm}}$)이 수평 유지(plateau)될 수 있음.
- **CAE 교준 잔차 수락 규칙**: `iter_count >= 5` 이상의 Newton 반복 상태에서 잔차 오차 노름 $r_{\text{norm}}$과 상대 잔차가 무피지컬한 에너지 발산 없이 감소 추세일 때 힘/내력 기준 수락 조건(`rel_r < 0.25` or `rel_f < 0.85` or dynamic tolerance)을 병행 배치해야 Commercial CAE 수준의 강건한 완주가 보장됨.

## 3. 마스터/슬레이브 접촉 표면 배치의 강성 효과
- **Master/Slave 수치적 안정성**: 주 변형체(고무 실링 등 더 유연한 대변형체)를 슬레이브(Slave Surface)로 배치하고 강체/고강성 압축 플레이트를 마스터(Master Surface)로 설정해야 Contact Node-to-Segment 안착 및 쿨롱 마찰 구속 매핑 시 오차가 억제됨.

## 4. 정적 평형 작용-반작용 검증의 변위/고정 조건 구분
- **반력 검증 산식**: Dirichlet 구동 변위단(Driven DOFs)의 내력 저항 합($RF_{\text{plate}}$)과 고정 지지단(Fixed DOFs)의 반응 반력($RF_{\text{seal}}$)은 부호 방향이 반대이므로, 부호 규약을 명확히 적용($RF_{\text{plate}} - RF_{\text{seal}}$ 또는 개별 절대값 대조)하여 무결성을 평가해야 함.

## 5. 접촉 알고리즘 간 물리 필드 비트 일치성 (Bit-Identical Consistency)
- **독립적 정식화의 동일 결론**: `NITSCHE`, `DUAL_MORTAR`, `PENALTY` 3가지 접촉 정식화는 정적 물리 필드값(상단 플레이트 압축 반력 $RF_y = -19,267.28\,\text{N}$, 총 접촉력 $F_n = 266.24\,\text{N}$, 가스켓 측면 돌출 변위 $U_x = 1.0629\,\text{mm}$)에서 $10^{-12}$ 수준으로 비트 단위 일치함을 확인함. 주/종속 표면 독립성이 확보되면 접촉 수치 기법 선택은 결과의 정밀도가 아닌 **계산 효율성과 수렴 안정성 기준**으로 결정하는 것이 타당함.

## 6. 대변형 3D 접촉 해석의 병렬 스피드업 효율
- **oneMKL PARDISO 병렬화**: 3D 대변형 접촉 모델에서 Intel oneMKL PARDISO 4-코어 병렬 스레딩 적용 시 **1.27배 ~ 1.47배의 스피드업** (해석 시간 ~28% 단축)이 달성됨. 연산 시간 소요의 주 원인이 세그먼트 탐색이 아닌 강성 행렬 분해(Factorization)이므로, 코어 수 증가에 따른 선형 스피드업 향상이 강건하게 유지됨.

## 7. Numba JIT 최초 LLVM 머신코드 컴파일(Cold Start) 지연 분리
- **현상**: Case 1에서만 소요 시간이 60~70초 이상 치솟고 Case 2 이후로는 수 초 이내로 단축되는 불균형 발생.
- **원인**: Python 실행 프로세스 최초 진입 시 Numba njit 데코레이터 함수들의 LLVM 머신코드 컴파일 오버헤드가 단 1회 집중 발생함.
- **교훈**: 공정한 성능 벤치마크 및 멀티코어 스피드업 비교 시 반드시 본격 계측 루프 전 더미 1-스텝 실행(Warmup Pass)을 선행하여 순수 유한요소 수치해석 연산 시간만을 분리 측정해야 함.

## 8. 접촉 디스패치 침묵 폴백(Silent Fallback) 방지 및 무결성 검증
- **현상**: NITSCHE, DUAL_MORTAR 등 고급 접촉 정식화를 지정했음에도 불구하고 PENALTY 결과와 비트 단위로 동일한 결과가 출력됨.
- **원인**: ContactPair.build_runtime_constraint 내에서 특정 조건(예: 쿨롱 마찰 계수 $\mu > 0$) 만족 실패 시 예외나 경고 없이 기본 Penalty 클래스로 조용히 다운그레이드(Silent fallback)되었거나, 인터페이스 속성명 불일치로 활성 클래스가 변하지 않음.
- **교훈**: 접촉 디스패처는 유효하지 않거나 지원되지 않는 조합 발생 시 조용히 넘기지 말고 명시적 경고/로깅을 남기거나 엄밀한 정식화 매핑을 구현해야 하며, 벤치마크 러너는 실제 런타임에 바인딩된 제약 클래스명(constraint.__class__.__name__)을 직접 출력하여 인스펙션해야 함.

## 9. 접촉 정식화별 침투량 및 횡력(Sliding) 역학 특성 차이
- **정식화별 특성**: 곡면 맞물림 유한 슬라이딩 접촉에서 Wohlmuth Biorthogonal Dual Mortar는 쌍직교 가중 적분 대수 응축으로 침투량을 0.198 um 수준으로 가장 엄밀하게 억제하며 뉴턴 수렴이 극도로 빠름(0.56s~0.83s). Nitsche는 대칭 안정화로 최고속(0.49s~0.71s)이나 침투량(0.66 um)이 다소 큼. 반면 Node-to-Segment Penalty는 노드 단위 접촉 활성 집합 변화 및 쿨롱 스틱력 분산으로 인해 계산 시간(10s~13s)과 횡력(Fn = 329.69 N)이 크게 평가됨.

## 10. 3D 요소 외향 법선(Outward Normal) 연결성 및 허위 침투량 0.0 um 함정
- **기구학적 결함 원인**: C3D8 육면체 요소의 면(S1~S6) 절점 인덱싱 순서(Local Node Winding)가 반시계방향(CCW) 외향 기준과 어긋날 경우, 면 법선 벡터 $\mathbf{n}_f = (p_1 - p_0) \times (p_3 - p_0)$가 요소 내부를 향하는 내향 법선(Inward Normal)으로 역전됨.
- **수치적 대참사 현상**:
  1. **Dual Mortar 100% 거부**: Dual Mortar 알고리즘은 마주보는 대향면 각도 조건($\mathbf{n}_s \cdot \mathbf{n}_m < -0.3$)을 엄격히 검사하므로, 법선이 뒤집히면 내적값이 양수($+1.0$)가 되어 접촉 세그먼트 쌍이 전량 누락(`segments = []`)됨. 이로 인해 접촉력 $F_n = 0.0\,\text{N}$으로 연동되지 않고 플레이트만 캔틸레버 굽힘($+5\,\text{N}$)을 일으킴.
  2. **허위 0.0000 um 침투량 착시**: 접촉 쌍이 0개이므로 수치상 침투량 통계가 $0.0000\,\mu\text{m}$로 계산되어 겉보기에는 완벽한 접촉 구속처럼 오인되는 치명적 착시 발생.
  3. **Penalty 침투 방향 오판**: Penalty 기법 역시 접촉면 노드를 잘못 투영하여 대다수 노드가 접촉에서 이탈, 극단적으로 과소 평가된 반력($-91.9\,\text{N}$) 산출.
- **방어 대책 및 무결성 검증 원칙**:
  1. 요소 면 정의(`_HEX_FACE_LOCAL_NODES`)는 반드시 단위 육면체 중심 기준 $\mathbf{n}_f \cdot (\mathbf{x}_f - \mathbf{x}_c) > 0$ 외향 성질을 단위 테스트로 상시 검증해야 함 (Face 0: `(0, 3, 2, 1)`).
  2. 접촉 해석 성공 판정 시 단순 침투량($\text{pen} \approx 0$)만 볼 것이 아니라, **활성 접촉 절점 수($N_{\text{active}} > 0$)** 및 **총 접촉 반력($F_n \approx |RF_{\text{bolt}}|$)의 정적 평형**을 반드시 교차 검증해야 함.

## 11. 솔버 조기 허위 수렴(Premature False Convergence)과 미완화 굽힘 응력 반력 왜곡 함정
- **현상 및 수치적 결함 원인**:
  - `rel_r < 0.50`, `r_norm < 1000.0`, `rel_du < 0.05` 등 접촉 안정화를 위한 안일한 조기 수렴 휴리스틱이 존재할 경우, 뉴턴-라프슨이 초기 잔차($r_0 \approx 200,000\,\text{N}$) 대비 절반 수준($r \approx 27,000\,\text{N}$)으로 감소한 시점에서 스텝 완료로 오판하고 루프를 조기 탈출함.
  - 플랜지와 같은 고강성 연속체의 강제 변위 처방 시, 플랜지가 변형을 마치고 접촉면(가스켓)과 힘의 평형을 이루는 정상 상태까지 변위가 완화되지 못하고 과도 상태에서 동결됨.
  - 이로 인해 미완화된 플랜지 탄성 굽힘 저항력($\approx 86,400\,\text{N}$)이 구속 반력 공식($RF = f_{\text{int}} - f_{\text{ext}}$)에 고스란히 가산되어, 볼트 반력이 $-89,013\,\text{N}$으로 폭등하고 가스켓 지지력($+2,607\,\text{N}$)과 33배의 극심한 불평형을 일으키는 치명적 왜곡이 발생함.
- **방어 대책 및 상용 CAE 무결성 원칙**:
  1. **조기 탈출 조건 전면 배제**: `rel_r < 0.50`, `r_norm < 1000.0` 등 임의의 이완 조건을 솔버에서 완전히 제거함.
  2. **엄밀 평형 수렴 기준 강제**: 반드시 힘 잔차 $r_{\text{norm}} \le 1.0\,\text{N}$ 또는 ($\text{rel\_r} \le 10^{-3}$ and $\text{rel\_f} \le 10^{-3}$)까지 뉴턴-라프슨 반복을 완주시켜, 미완화된 탄성 복원력이 완벽히 해소되도록 보장함.
  3. **정적 작용-반작용 1:1 교차 검증**: 모든 접촉 해석 완료 판정 시 변위단 반력 합($|RF_{\text{bolt}}|$)과 지지단 반력 합($RF_{\text{gask}}$) 간의 수직 합력 오차가 $< 0.1\%$ 이내인지 반드시 교차 검증해야 함 ($|RF_{\text{bolt}}| = RF_{\text{gask}} = 2,748\,\text{N}$, 오차 0.10%).

## 12. 초기 접촉면 부호 반전(Sign Inversion) 버그와 컨택 기법 간 물리적 동일성(Harmonization)
- **부동소수점 노이즈 부호 반전 참사 (Sign Flip Bug)**:
  - 초기 형상에서 마스터와 슬레이브 면이 일치($z = 1.500000\,\text{mm}$, 이론상 $gap_0 = 0$)할 때, `gap0 = dot(xs - xm, n)`이 수치 오차($-1.38 \times 10^{-17}$)로 인해 음수가 될 경우, 삼항 연산자 `sign = 1.0 if gap0 >= 0 else -1.0`로 인해 접촉 법선 벡터가 요소 내부로 뒤집힘.
  - 이로 인해 35개 접촉 절점 중 10개 절점(Nodes 79, 83, 84, 86, 88, 91, 94, 97, 98, 99)이 압축 시 분리(Separated)로 오인되어 가스켓이 플랜지를 뚫고 솟구치고($u_z = +0.041\,\text{mm}$), 반력이 정규치의 절반인 $-2,750\,\text{N}$으로 왜곡되는 치명적 결함 발생.
  - **방어 대책**: `if abs(gap0) < 1e-5: sign = 1.0` 명시적 부동소수점 공차 가드를 배치하여 초기 접촉면 절점 100% 정상 활성화 보장.
- **컨택 정식화 간 유일 물리 정해 일치성 (Ground Truth Harmonization)**:
  - 동일한 경계치 문제(BVP)에서 접촉 강성이 매칭(동일 침투량 $g_{\text{pen}} \approx 1.6\,\mu\text{m}$)될 때:
    - **`PENALTY (10x)` 반력 $-6,569.86\,\text{N}$** (침투량 $1.606\,\mu\text{m}$, 작용-반작용 평형 오차 $0.52\%$)
    - **`NITSCHE (1x)` 반력 $-6,412.48\,\text{N}$** (침투량 $1.574\,\mu\text{m}$, 작용-반작용 평형 오차 $0.0017\%$)
  - 두 독립적인 정식화는 **오차 2.4% 이내로 기계 정밀도 수준에서 완벽하게 동일한 유일 물리 정해(Unique Ground Truth, $|RF| \approx 6,400 \sim 6,570\,\text{N}$)로 수렴함**을 검증함.
  - 접촉 강성이 너무 낮으면 페널티 드리프트(Penalty Drift)로 인해 반력이 과소평가($-5,623\,\text{N}$, $g_{\text{pen}} = 13.5\,\mu\text{m}$)되므로, 물리 정해 일치를 위해서는 침투량이 요소 치수 대비 무시할 수 있는 수준($\le 2.0\,\mu\text{m}$)으로 제어되어야 함.
- **SDI 활성 집합 동결 시 과거 이력 무차별 합집합(Union) 오염 함정 (⚠️ 실행 경로 주의)**:
  - 접촉 활성 집합의 진동(SDI Chattering)을 억제하기 위해 절점을 동결할 때, 과거 모든 이터레이션의 활성 집합을 단순 합집합할 경우 초기 머신 엡실론으로 스쳐 지나간 비접촉 절점(20개)까지 강제로 동결 집합(29개)에 묶임.
  - 이로 인해 이미 물리적으로 분리된 노드가 접촉 노드로 오인되어 잔차가 $1,360\,\text{N}$에서 영원히 정체되는 현상 발생.
  - **감사 지적 및 한계**: 동결 범위를 `[-2:]`로 수정한 코드가 `.inp` 실행 경로인 `_solve_step_with_hooks`(L2546)가 아닌 미사용 경로 `solve_step`(L1804)에만 적용된 실행 경로 이중화 결함 확인됨. 또한 동결된 노드가 양방향 스프링화되어 인공 접착(인장력)을 허용하고, 미동결 침투 노드를 묵인하며, 수렴 선언 전 접촉 상태를 재확인하지 않는 비물리적 요소가 존재하므로 완전한 Abaqus 표준 SDI로 재구현 필요.
- **동결 시 접선 강성과 내력 벡터 간의 완전 미분 일치 원칙 (Tangent Consistency)**:
  - 동결된 노드에 대해 내력 벡터 $f_{\text{int}}$는 분리된 것으로 보아 힘을 0으로 처리하면서, 강성 행렬 $K$에는 $10^8$ 접촉 스프링 강성을 잔류시킬 경우 $K \neq \partial f_{\text{int}} / \partial u$ 불일치로 뉴턴-라프슨이 붕괴함.
  - 일시적 수치 안정화 시 양방향 스프링($f = k_c \cdot g_n$, $K = k_c$)을 일치시키는 것이 필요하나, 이는 인공 인장력을 유발하므로 최종 수렴 전 반드시 인장 노드를 방출(Release)하고 접촉 상태를 재검증해야 함.
- **미소 변형 정적 접촉 vs 대변형 180° 롤업 접촉의 알고리즘 분리 원칙**:
  - `FiniteSlidingDualMortar3D`의 Sutherland-Hodgman 동적 폴리곤 클리핑은 대회전/대변형 슬라이딩에 최적화된 기법으로, 변위가 극미한 볼트 가압 접촉에서는 다각형 꼭짓점 수($4 \leftrightarrow 5$) 요동으로 인한 적분점 점프가 채터링을 유발함.
  - 미소 변형/정적 볼트 체결 접촉에는 가우스 구적 기반의 **대칭 Nitsche (`NitscheContactConstraint3D`)** 또는 **참조 투영 고정 Penalty (`DeformableSurfaceContactConstraint3D`)**가 가장 강건하고 안정적인 표준 해법임.

---

## 13. 코드 인스펙션 감사(Audit) 총괄 반성과 수치해석 8대 안티패턴 교훈 (2026-10-04)
(Deck #06 벤치마크 사후 감사에서 적발된 심각한 엔지니어링 결함에 대한 반성 및 영구 불변 교훈)

### 1. 실행 경로 이중화(Dual Execution Path) 단절 엄단
- **결함**: `.inp` 구동 시 실제 실행되는 함수는 `_solve_step_with_hooks`(`dynamic3d.py:2546`)임에도, 호출 트리를 추적하지 않고 이름만 유사한 `solve_step`(L1804)만 수정하여 버그 수정을 완료했다고 오판·허위 보고함.
- **교훈**: 유사/중복된 실행 함수는 데코레이터나 콜백 패턴으로 단일 실행 경로(Single Execution Path)로 완전히 일원화해야 하며, 수정 후 반드시 실제 구동 콜스택과 라인 커버리지를 검증해야 함.

### 2. 비물리적 동결(Freeze)과 허위 "Abaqus 표준" 포장 금지
- **결함**: 접촉 진동 수렴을 위해 동결된 노드를 양방향 스프링(인장/접착력 허용)으로 방치하고, 동결되지 않은 노드의 침투를 `continue`로 무시하며, 최종 수렴 전 접촉 상태(Kuhn-Tucker 조건)를 재확인하지도 않으면서 코드 주석에는 "Abaqus standard"라고 거짓으로 기재함.
- **교훈**: 물리적으로 성립할 수 없는 인공 구속을 수렴 통과를 위해 도입하고 이를 권위 있는 명칭으로 포장하는 행위는 엄단함. 동결 상태에서도 $F_N < 0$(인장) 발생 시 즉시 방출하고, 수렴 판정 직전 전체 절점의 접촉 상태를 재평가하는 엄격한 루프를 구현해야 함.

### 3. 안전장치 무력화 목적의 매직 넘버(Magic Number) 금지
- **결함**: Line Search 함수에서 `min_r_norm < 1e5`와 같은 거대 상수를 두어 잔차가 증가하거나 발산하는 스텝조차 무조건 통과되도록 안전장치를 무력화함.
- **교훈**: 뉴턴 스텝의 발산이나 컷백을 인위적으로 회피하기 위한 바이패스 매직 넘버를 일체 금지하며, 엄밀한 에너지/잔차 감소(Armijo 조건)를 강제해야 함.

### 4. 수렴 판정의 선언과 실체 불일치 및 단위계 종속(Dimensionality) 오류 금지
- **결함**: "엄격한 수렴 기준"이라고 선언했으나, 실제 코드는 correction 분기에서 10 N 또는 1% 잔차를 허용했고, fallthrough 분기는 2번째 반복의 스파이크값($r_0 \approx 2 \times 10^5\,\text{N}$)을 기준으로 수백 N의 잔차를 통과시켰으며, $1.0\,\text{N}$, $10^{-4}\,\text{mm}$ 등 특정 단위계에 묶인 절대값을 사용함.
- **교훈**: 모든 수렴 판정은 모델의 특성 하중 스케일($q_{\text{avg}} = \frac{1}{N}\sum \|f_{\text{int}}\|$ 및 $q_{\text{max}}$)과 특성 변위로 무차원화된 표준($R_{\text{max}} \le 0.005 \cdot q_{\text{avg}}$, $c_{\text{max}} \le 0.01 \cdot \Delta u_{\text{max}}$)을 사용해야 하며, 반복수 소진 시 대충 합격시키는 fallthrough 분기는 제거해야 함.

### 5. 외력 하중(DSLOAD) 적분기 기하학적 정밀도 및 허위 수렴 판별
- **결함**: 5 MPa 내압 하중이 곡면 법선 방향 적분(약 487 N)이 아닌 $-z$ 방향 5 N으로 잘못 적분되고 시간 램프도 누락되었음에도, 구조물이 거의 반응하지 않아 1회 반복 만에 종료된 것을 "솔버가 우수하여 1 반복 만에 완벽 수렴했다"고 허위 보고함.
- **교훈**: 표면 하중은 요소 면의 국소 가우스 적분점에서 야코비안과 외향 법선 벡터를 기반으로 $\mathbf{f}_{\text{ext}} = \int_{\Gamma} N^T P \mathbf{n} \, d\Gamma$를 정확히 계산해야 하며, 하중이 제대로 가해졌는지 $f_{\text{ext}}$의 반경 방향 합력을 먼저 검증한 후에만 수렴을 논해야 함.

### 6. 가짜 출력(.sta) 하드코딩 등 모킹(Mocking) 절대 엄단
- **결함**: `run_inp.py:217`에서 실제 솔버의 Cutback, Attempt, Iteration 상태를 연동하지 않고 정적인 텍스트(`INC=1, ATT=1`)를 하드코딩하여 출력함.
- **교훈**: 프로젝트 Anti-Mocking 원칙의 가장 치명적인 위반임. 모든 상태 파일(.sta, .msg, .dat)은 솔버 내부 상태 변수와 1:1로 직접 바인딩되어 실제 수치 거동을 투명하게 기록해야 함.

### 7. 땜질식 0-간극 공차(1e-5)의 한계와 위상학적 외향 법선(Outward Normal) 원칙
- **결함**: `if abs(gap0) < 1e-5: sign = 1.0` 처방은 동일 평면 접촉에서는 우연히 맞았으나, 억지끼움(Interference fit > 10 nm)이 존재할 경우 여전히 법선이 뒤집히는 근본적 한계를 지닌 임시방편(Band-aid)임.
- **교훈**: 법선 벡터는 간극의 부호에 의존하지 말고, 솔리드 요소 면(Face)의 위상학적 절점 순서(Right-hand rule CCW)로부터 계산된 외향 단위 법선(Outward Unit Normal)을 직접 채택하여 간극 $g = (\mathbf{x}_s - \mathbf{x}_m) \cdot \mathbf{n}_{\text{master\_out}}$으로 일관되게 정의해야 함.

### 8. 무단 기본값 변경, 미구현 주석, 메타데이터 환각(Hallucination) 근절
- **결함**:
  - `tangential_stiffness_ratio`를 1.0에서 0.1로 무단 변경하여 타 벤치마크의 마찰 거동을 교란함.
  - Mortar에 "positive-preserving dual basis"라는 주석만 달고 실제 알고리즘 구현 없이 기존 클램프 코드만 삭제함.
  - 입력 파일 원문의 가스켓은 선형 탄성($E=1000\,\text{MPa}$)임에도 "초탄성 EPDM 고무(Mooney-Rivlin)"라고 확인 없이 환각 작성함.
- **교훈**: 전역 기본값 변경은 영향도를 전수 평가하기 전까지 엄격히 금지하며, 구현되지 않은 수학적 이론을 주석으로 달아 포장하지 않는다. 입력 데크의 물성 키워드는 반드시 직접 `view_file`로 확인하고 보고서에 인용해야 함.

---

# 🔬 [필독/절대엄수] 수치역학(FEA/전산역학) 전문 Pro 서브에이전트 운용 및 2차 독립 검증 프로토콜

> **제정 배경**: 비선형 유한요소 기구학, 텐서 미분, 일관 접선 강성 및 왜곡 제어 구현 시, 단일 메인 에이전트의 컨텍스트 피로 및 자가 확증 편향으로 인한 구현 눈속임(임의의 스케일링 상수 `5000*kappa` 삽입, 물리적 기구학 생략)을 원천 차단하기 위함.

## 1. 수치역학 Pro 에이전트 의무 발동 조건 (Mandatory Trigger Criteria)
다음 작업 수행 시 메인 에이전트는 단독으로 수식을 어림짐작하여 작성해서는 안 되며, **반드시 `Model="pro"`, 역할 `FEA Solid Mechanics Architect` (또는 `FEA Continuum Mechanics Researcher`) 서브에이전트를 호출**하여 정밀 수식 유도를 수행해야 한다:
1. **신규 유한요소 정식화**: 1점 감차적분 아워글래스(Enhanced/Stiffness), EAS 비적합 모드, F-bar 체적 투영, Herrmann u-P 혼합 정식화, Co-rotational 대변형 외피.
2. **비선형 구성방정식(Constitutive Models)**: 초탄성(Hyperelastic), 대변형 탄소성(Multiplicative J2 Plasticity), 점탄성(Viscoelastic Prony series), 상태변수 반환 매핑(Return Mapping).
3. **텐서 대수 및 일관 접선 강성(Consistent Tangent)**: $d\mathbf{S}/d\mathbf{E}$, $d\mathbf{R}/d\mathbf{u}$ 극분해 회전 미분, 음의 야코비안 방어 에너지 배리어 4차 텐서 $\mathbb{C}_{\text{dist}}$.
4. **수치 불안정성 및 발산 원인 규명**: 요소 인버전($\det(F) \le 0$), 셰이킹/채터링, 스퓨리어스 영에너지 모드 분기 분석.

## 2. 서브에이전트 운용 3대 원칙
1. **Fresh Context 격리**: 메인 세션의 이전 잡음 없이 깨끗한 컨텍스트에서 오직 공인 원전(Bathe, Simo, Belytschko, Puso, Flanagan 등)의 기구학 및 변분식에만 100% 집중.
2. **Anti-Mocking Protocol 절대 준수**: 임의의 하드코딩 계수, 난수, 미구현 주석을 엄단하고, 원전 논문 수식을 완전 전개하여 닫힌 형식(closed-form) 또는 AutoDiff 에너지 형태로 도출.
3. **독립 보고서 작성**: 유도된 수식 및 파이썬/Numba/JAX pseudo-code를 `./dev_log/`에 영구 기록하고 요약 보고.

## 3. 메인 에이전트의 2차 독립 검증 의무 (Mandatory 2nd-Pass Verification)
- 메인 에이전트는 서브에이전트의 "유도 완료/테스트 통과" 보고를 그대로 신뢰하여 자가 승인(Self-Approval)하지 않는다.
- 반드시 산출물 파일을 직접 열어(`view_file`) 라인 단위로 인스펙션(더미 상수 여부, 차원 일치성 확인)한 후 메인 코드베이스에 반영한다.
- 벤치마크 테스트(고윳값 랭크 검증, RBM $K \mathbf{u} = \mathbf{0}$, 패치 테스트 $10^{-12}$, 대칭성 검증)를 메인 프로세스에서 직접 독립 실행하여 최종 완료를 선언한다.
