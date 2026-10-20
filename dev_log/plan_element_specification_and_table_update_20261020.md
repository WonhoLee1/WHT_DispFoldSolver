# Implementation Plan: 요소 정식화(Element Formulations) 표 표기 및 고속 대안 요소(Opt2-H) 비교 방안 (2026-10-20)

## Goal Description
사용자 요청에 따라:
1. **보고서 표에 사용한 요소 정식화(Element Formulation) 명시**:
   - 모든 2D 및 3D 벤치마크 결과 표에 각 재료층(PET, Glass, PSA)에 적용된 구체적 요소 정식화 명칭을 별도 컬럼으로 표기합니다.
2. **요소 정식화별 연산 비용 특성 명시 및 대안 제안**:
   - 현재 WHT 솔버에 적용된 `Opt1-IH` 계열(`CPE4I_CR` / `C3D8I_CR`)은 Wilson 비적합 변위 모드(Incompatible Modes)의 요소 레벨 국소 대수 응축(Static Condensation)으로 인해 전단 잠김(Shear Locking) 억제력은 우수하지만 연산 부하가 큽니다.
   - 이에 따라 1점 감차적분 + 아워글래스 제어 기반의 **고속 대안 요소 `Opt2-H` (`CPE4R_CR` / `C3D8R_CR` + `u-P Herrmann` PSA)**를 보고서에 함께 제안하고, 필요시 비교 벤치마크를 수행할 수 있도록 구조를 마련합니다.

---

## User Review Required
> [!NOTE]
> **보고서 표에 추가할 요소 정식화 표기 표준**:
> - **FrontISTR (2D/3D)**:
>   - 솔리드층(PET, Glass, Plate): `Hex 361 FI` (8-절점 완전적분 B-bar 호환)
>   - 점착제층(PSA): `Hex 361 FI (Hyperelastic Neo-Hookean Total Lagrangian)`
> - **WHT Solver (2D)**:
>   - `2D-Opt1-IH`: 솔리드 `CPE4I_CR` (4절점 EAS/비적합 9모드) + PSA `CPE4H_CR` (Herrmann u-P 혼합요소)
>   - `2D-Opt2-H` *(제안)*: 솔리드 `CPE4R_CR` (1점 감차적분 + Flanagan-Belytschko 아워글래스) + PSA `CPE4H_CR` (Herrmann u-P)
> - **WHT Solver (3D)**:
>   - `3D-Opt1-IH`: 솔리드 `C3D8I_CR` (8절점 비적합 9모드) + PSA `C3D8H_CR` (Herrmann u-P)
>   - `3D-Opt2-H` *(제안)*: 솔리드 `C3D8R_CR` (1점 감차적분 + 아워글래스) + PSA `C3D8H_CR` (Herrmann u-P)

---

## Proposed Changes

### 1. `benchmark_mesh_refinement_fistr_wht.py` 보고서 생성기 업데이트
#### [MODIFY] `example_disp_fold/benchmark_mesh_refinement_fistr_wht.py`
- Table 1 (2D), Table 2 (3D In-plane), Table 3 (3D Depth), Table 4 (Parity)의 마크다운 표 헤더에 **`적용 요소 (PET / PSA)`** 컬럼 추가:
  - FISTR: `Solid-FI / PSA-FI`
  - WHT: `CPE4I_CR / CPE4H_CR` 또는 `C3D8I_CR / C3D8H_CR`
- 종합 보고서 `dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md`에 동일 반영.

### 2. 고속 감차적분 요소 `Opt2-H` (`CPE4R_CR`, `C3D8R_CR`) 실행 지원 준비
- WHT Solver 러너에 `--elem_case` 옵션을 추가하여 사용자가 필요 시 `Opt1-IH` 외에 초고속 `Opt2-H`도 바로 비교 측정할 수 있도록 구성.

---

## Verification Plan
1. WHT 솔버 백그라운드 태스크 완료 후, 결과 JSON에 기록된 요소 명칭을 확인.
2. `benchmark_mesh_refinement_fistr_wht.py`의 `generate_benchmark_report_and_plots()`를 실행하여 `dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md`의 표에 요소 컬럼이 온전히 생성되는지 검증.
