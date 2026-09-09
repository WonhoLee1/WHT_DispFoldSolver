# [구현 계획서] 물방울형(Teardrop-shaped) 디스플레이 폴딩 기구학 구현 및 검증

---

## 1. 개요 및 배경 (Overview & Problem Statement)

사용자가 제시한 폴딩 기구학 다이어그램([`uploaded_media_1788579328267.png`](file:///C:/Users/GOODMAN/.gemini/antigravity-cli/brain/de24dc3b-89b5-45d6-9b20-29037c2bd557/.user_uploaded/uploaded_media_1788579328267.png))은 폴더블 디스플레이의 두 가지 핵심 폴딩 구성을 명확히 정의하고 있습니다:

```
[1] U-shaped folding configuration:
    - 닫힌 상태에서 플레이트(Fixture) 간격 D = 루프 직경 2R
    - 디스플레이가 플레이트 간격과 동일한 폭으로 하강하여 단순 반원형 U-루프 형성

[2] Teardrop shaped folding configuration (물방울형):
    - 닫힌 상태에서 플레이트(Fixture)가 디스플레이를 사이에 두고 완벽히/밀착하여 닫힘 (Neck Gap d ≈ 1.2 ~ 2.0 mm)
    - 힌지부의 미지지 디스플레이(Unsupported Length)가 외측으로 팽창/확장 (W_max > d)
    - 디스플레이 파손을 방지하는 안전 굽힘 반경(R_min)을 유지하며 부드러운 물방울(Teardrop) 루프 완성
```

![U-Shape vs Teardrop Comparison](file:///C:/Users/GOODMAN/.gemini/antigravity-cli/brain/de24dc3b-89b5-45d6-9b20-29037c2bd557/ex13_teardrop_vs_u_shape.png)

### 기존 3개 예제 형상 분석 결과
1. **`ex13_roundtrip_final_folding_shape.png` (90도 미도달):**
   - 14계층 PET-PSA 다층 적층 모델에서 PSA 초탄성 연화로 인한 선형 탐색 실패로 step 8 ($\theta = 15.76^\circ$)에서 조기 중단되어 기울어진 상태로 저장됨.
2. **`ex13_improved_folding_shape.png` (우수한 U-Shape):**
   - $x_{\text{pivot}} = 2.5\text{ mm}$ (플레이트 갭 $5.0\text{ mm}$), $x_{\text{gap}} = 7.0\text{ mm}$ ($L_{\text{free}} = 14\text{ mm}$) 조건에서 Corning 2점 굽힘 문헌과 0.088% 오차로 일치하는 이상적인 U자형 루프 형성.
3. **`ex13_build_final_folding_shape.png` (원형/풍선형):**
   - 플레이트 갭이 이미 $5.0\text{ mm}$로 넓은 상태에서 미지지 길이만 $20.0\text{ mm}$로 과도하게 길어, 목 부위 좁힘 없이 원형 풍선 모양으로 팽창함.

---

## 2. 기구학적 공식 유도 (Kinematic Derivation)

평판 상태($t=0$)에서 $Y=0$에 놓인 디스플레이와 $Y \in [-t_{\text{plate}}, 0]$에 위치한 플레이트가 각각 $\pm 90^\circ$ 회전할 때:

1. **닫힘 시 플레이트 면 간격 ($d_{\text{neck}}$):**
   $$x_{\text{plate, left}} = -x_{\text{pivot}} - y_{\text{pivot}}$$
   $$x_{\text{plate, right}} = +x_{\text{pivot}} + y_{\text{pivot}}$$
   $$y_{\text{pivot}} = 0 \implies d_{\text{neck}} = 2 \cdot x_{\text{pivot}}$$
   - **U-Shape:** $x_{\text{pivot}} = 2.5\text{ mm} \implies d_{\text{neck}} = 5.0\text{ mm}$ (넓은 갭)
   - **Teardrop:** $x_{\text{pivot}} = 0.8\text{ mm} \implies d_{\text{neck}} = 1.6\text{ mm}$ (밀착 갭)

2. **미지지 힌지 디스플레이 길이 ($L_{\text{free}}$):**
   $$L_{\text{free}} = 2 \cdot x_{\text{gap}}$$
   - 플레이트 안쪽 모서리 $y$ 좌표: $y_{\text{inner}} = x_{\text{gap}} - x_{\text{pivot}}$

3. **물방울 형상 형성 조건 (Elastica Teardrop Condition):**
   - 목(Neck) 부위 간격 $d_{\text{neck}} = 2 x_{\text{pivot}} = 1.6\text{ mm}$에서 접선각 $\theta = 0$ (수직)으로 진입.
   - 호 길이 $L_{\text{free}} = 2 x_{\text{gap}} = 15.0\text{ mm} \gg \pi \cdot x_{\text{pivot}} \approx 2.51\text{ mm}$.
   - 굽힘 강성 $EI$에 의해 압축된 보가 양측으로 볼록하게 좌굴/팽창하여 최대 폭 $W_{\text{max}} \approx 3.7\text{ mm}$ 도달.
   - 곡률이 끝단에서 부드럽게 돌아 $R_{\text{eff}} \approx 1.85\text{ mm}$의 안전 반경을 확보하며 완전한 **물방울형(Teardrop)**을 형성함.

---

## 3. 코드 구현 및 변경 계획 (Implementation Details)

```mermaid
flowchart TD
    A[fold_model_config.py] -->|1. make_teardrop_config 추가| B[GeometryConfig / Presets]
    B -->|2. --fold_shape {teardrop, u_shape}| C[ex13_unified_model_io.py]
    B -->|3. Inp generation 지원| D[gen_ex12_inp.py]
    C -->|4. PARDISO SPD 고속 해석| E[DynamicSolver & Result]
    E -->|5. 비교 플롯 및 검증| F[ex13_teardrop_vs_u_shape.png]
    E -->|6. 단위 테스트| G[tests/test_teardrop_folding.py]
```

### 1) `dispsolver/fold_model_config.py`
- `make_teardrop_config(thickness_mm=0.1, n_rows=2, substrate="glass", neck_gap_mm=1.6, free_span_mm=15.0)` 함수 추가.
- `GeometryConfig`에 `fold_shape: str = "teardrop"` 및 적응형 메쉬 분할 기본값 연동.
- `MeshGradingConfig`가 `hinge_half_gap`에 맞춰 클러스터 경계를 자동으로 검증하도록 invariant 보호.

### 2) `examples/ex13_unified_model_io.py`
- CLI 인자 `--fold_shape {teardrop, u_shape}` 추가 (기본값: `teardrop`).
- `--fold_shape teardrop` 선택 시:
  - `hinge_pivot_x = 0.8 mm` (목 갭 1.6 mm)
  - `hinge_half_gap = 7.5 mm` (자유 구간 15.0 mm)
  - `hinge_span_half_width = 6.0 mm`, `hinge_span_dx = 0.2 mm`
- `--fold_shape u_shape` 선택 시:
  - `hinge_pivot_x = 2.5 mm` (목 갭 5.0 mm)
  - `hinge_half_gap = 7.0 mm` (자유 구간 14.0 mm)
- 결과 시각화 타이틀 및 통계 출력에 물방울형 기구학 파라미터(목 갭, 최대 루프 폭, 형상비) 표시.

### 3) `examples/verify_two_point_bending_vs_ex13.py`
- `plot_teardrop_vs_u_shape()` 함수를 정식 내장하여 두 형상의 차이를 정량적/시각적으로 언제든 재현할 수 있도록 통합.

### 4) `tests/test_teardrop_folding.py` (신규 테스트)
- 물방울형 폴딩 기구학 검증:
  1. $d_{\text{neck}} \le 2.0\text{ mm}$ (밀착 닫힘 검증)
  2. $W_{\text{max}} \ge 1.8 \times d_{\text{neck}}$ (외측 루프 팽창 및 물방울 형상비 검증)
  3. 요소 왜곡/반전 없음 (`n_inverted == 0`)
  4. 90도 완전 폴딩 수렴 확인 (`reached_target == True`)

---

## 4. 검증 기준 및 절차 (Verification Criteria)

| 항목 | 성공 기준 (Acceptance Criteria) | 검증 방법 |
|---|---|---|
| **플레이트 밀착도** | 닫힌 상태 플레이트 간격 $d_{\text{neck}} \le 2.0\text{ mm}$ | `test_teardrop_folding.py` |
| **물방울 형상비** | 루프 최대 폭 $W_{\text{max}} \ge 3.2\text{ mm}$ ($> 1.8 \times d_{\text{neck}}$) | 루프 절점 좌표 분석 |
| **메쉬 품질** | 반전 요소 0개 (`n_inverted == 0`), 뒤틀림 없음 | `solver._check_mesh_quality()` |
| **수렴성** | Cutback 0회, 증분당 Newton 반복수 $\le 3$, 전 구간 수렴 | `ex13` 16 steps 100% 완료 |
| **회귀 방지** | 기존 22개 단위 테스트 및 11개 코어 벤치마크 100% 통과 | `pytest` & `verification.run_all` |
| **작업 로그 보관** | `./dev_log/plan_teardrop_folding_configuration_20260905.md` 저장 | UTF-8 no BOM 파일 보관 |
