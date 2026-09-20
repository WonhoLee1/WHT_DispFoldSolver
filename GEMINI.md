See `AGENTS.md` in this same directory — it is the canonical project rule
file (target problem, solver theory, element JIT strategy, and numerical pitfalls
already solved) shared across Claude Code, Gemini CLI, and OpenCode. Read it before
making changes here.

Element JIT Strategy: Use `--elem_jit jax` for R&D/prototyping (AutoDiff), and switch to `--elem_jit numba` for production run speed.

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
- **규칙**: 새 실측이 나오면 해당 영역 문서의 Open 항목부터 갱신. 본 인덱스의 링크가 깨지지 않게 파일명 변경 시 함께 수정.

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


