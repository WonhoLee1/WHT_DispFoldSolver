# Implementation Plan: WHT 솔버 14-Layer 풀스택 정확도 저하 원인 규명 및 해결 방안 (2026-10-20)

## Goal Description
14-Layer Full-Stack (`teardrop`, 총 두께 $630\,\mu\text{m}$) 해석에서 WHT 솔버의 구동 모멘트($1.99\,\text{N}\cdot\text{mm}$ vs FISTR $35.15\,\text{N}\cdot\text{mm}$)와 루프 폭($14.0\,\text{mm}$ vs FISTR $4.53\,\text{mm}$)의 편차가 크게 발생한 근본 원인을 수치역학적·코드 레벨로 전수 분석하고, 이를 완전히 해결하기 위한 수정 계획을 수립합니다.

---

## User Review Required
> [!IMPORTANT]
> **핵심 진단 결과: 3대 근본 원인 규명**
> 1. **[치명적 버그] 테이프 보이드(`PSA_0_Voided`) 내 고립 절점(Isolated Nodes) 잔류 및 루프 폭 $14.0\,\text{mm}$ 고정 왜곡**:
>    - `GmshMeshGenerator`에서 Layer 0 (`PSA_0_Voided`, $30\,\mu\text{m}$)의 힌지 보이드 구간($X \in [-7.5, +7.5]\,\text{mm}$) 생성 시 요소를 생성하지 않으면서도 절점(`mesh.add_node`)은 생성함.
>    - 이로 인해 **어떤 요소에도 연결되지 않은 29개(3D는 수백 개)의 고립 절점**이 $y=0, X \in [-7.0, +7.0]\,\text{mm}$에 잔류함.
>    - 고립 절점은 하중과 강성이 없어 변위 $u \equiv 0$으로 남게 되며, 결과 후처리 시 `loop_width_x = max(X_def) - min(X_def) = +7.0 - (-7.0) = 14.0000 mm`로 **변형되지 않은 고립 절점의 초기 폭이 그대로 반환**되어 형상 오차가 $9.47\,\text{mm}$로 왜곡되었습니다.
>    - 또한 강성 행렬에 대각선 0을 유발하여 **3D WHT 솔버의 조기 수렴 중단($60^\circ$ 한계)의 직접적 원인**이 되었습니다.
> 
> 2. **[역학적 원인] 초유연 점착제(`PSA_0`, $E=0.05\,\text{MPa}$)를 통한 플레이트 간접 전단 구동(Shear-Lag Decoupling)**:
>    - 4층 모델(`compact_4`, 0.09% 오차로 FISTR와 완벽 일치했던 케이스)은 바닥면이 고강성 `PET_Bottom` ($E = 4000\,\text{MPa}$)이어서 플레이트 회전이 디스플레이 전체로 100% 직결 전달되었습니다.
>    - 반면 14층 풀스택(`teardrop`)은 바닥면이 $30\,\mu\text{m}$ 초유연 `PSA_0` ($E = 0.05\,\text{MPa}$, PET 대비 80,000배 유연)이어서, 플레이트가 회전할 때 PSA 층의 극심한 전단 슬립(Shear Lag)으로 인해 상부 13개 층(PET 1~6, Glass)으로 회전 모멘트가 온전히 전달되지 않고 감쇠되었습니다.
> 
> 3. **[요소 정식화 원인] 3D `C3D8I_CR` 비적합 모드의 14-Layer 대변형 굽힘 응축 한계**:
>    - 사용자께서 지적하신 대로 9-모드 비적합 변위 모드의 정적 응축(Static Condensation) 비용이 크고, 고립 절점의 특이성과 결합되어 3D 대변형에서 수렴 한계를 유발했습니다.

---

## Proposed Changes

### 1. 메쉬 생성기 고립 절점 자동 소거 (`clean_isolated_nodes`)
#### [MODIFY] `example_disp_fold/gmsh_mesh_generator.py`
- 2D 및 3D 메쉬 생성 완료 후, 요소 연결성(Element Connectivity)을 검사하여 어떤 요소에도 속하지 않는 고립 절점을 자동으로 필터링 및 소거하고 연속된 절점 번호로 리매핑합니다.

### 2. 14-Layer 풀스택 구동 절점 정의 표준화
#### [MODIFY] `example_disp_fold/gmsh_mesh_generator.py` & `benchmark_mesh_refinement_fistr_wht.py`
- 테이프 보이드가 있는 14-Layer 풀스택의 경우, 구동 경계조건을 `PSA_0` 단독이 아닌 `PSA_0` 외곽 및 상부 `PET_1` 바닥 계면(실제 디스플레이 라미네이션 부착면)과 일치시켜 전단 슬립으로 인한 모멘트 유실을 방지합니다.

### 3. 후처리 루프 폭(`loop_width_x`) 계산 유효 노드 마스킹
#### [MODIFY] `example_disp_fold/benchmark_mesh_refinement_fistr_wht.py`
- `loop_width_x` 적분 시 디스플레이 최상면(Top Surface, UTG Glass) 또는 연결된 실제 힌지 절점만을 대상으로 마스킹하여 기구학적 U-루프 폭을 정확하게 추출합니다.

---

## Verification Plan

### Automated Tests
1. **고립 절점 소거 검증 단위 테스트**:
   - `isolated_nodes count == 0` 확인.
2. **WHT 2D 14-Layer 100% 재해석**:
   - `python example_disp_fold/benchmark_mesh_refinement_fistr_wht.py --preset teardrop --part wht --substeps 30 --threads 6`
   - 루프 폭 $\Delta X$가 $14.0\,\text{mm}$에서 FrontISTR 수준($\approx 4.5\,\text{mm}$)으로 정상화되는지 확인.
   - 구동 모멘트 $\|M_z\|$가 정상적인 14-Layer 대변형 굽힘 값으로 일치하는지 검증.
3. **종합 보고서 및 몽타주 재생성**:
   - `dev_log/benchmark_mesh_refinement_fistr_wht_20261020.md` 업데이트.
