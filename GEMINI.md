See `AGENTS.md` in this same directory — it is the canonical project rule
file (target problem, solver theory, element JIT strategy, and numerical pitfalls
already solved) shared across Claude Code, Gemini CLI, and OpenCode. Read it before
making changes here.

Element JIT Strategy: Use `--elem_jit jax` for R&D/prototyping (AutoDiff), and switch to `--elem_jit numba` for production run speed.

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

