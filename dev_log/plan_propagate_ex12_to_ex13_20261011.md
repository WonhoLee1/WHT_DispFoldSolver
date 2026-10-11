# [Implementation Plan] ex12 우수 설계 구조의 ex13(3D 멀티레이어 폴딩 솔버) 이식 및 고도화

## 1. 개요 (Overview)
2D 정본인 `ex12_abaqus_inp_plate_fold.py`에서 확립된 **Abaqus Python API 객체 계층, 적응형 시간 증분 제어기(AdaptiveDtController), 티어드롭 힌지 기구학(피벗 +-1.4mm, 힌지 보이드), 회전 선행 예측자(carry_contacting_nodes), 실시간 반력 모멘트 및 대칭 평형 검증 체계**를 3D 멀티레이어 폴딩 솔버인 `examples/ex13_3d_display_plate_fold.py`에 전면 이식하여 3D 해석 파이프라인의 완성도를 2D 정본 수준으로 승격합니다.

---

## 2. ex12 핵심 아키텍처 이식 매트릭스

| 영역 | 기존 ex13 상태 | ex12 기반 고도화 (이식 후) |
| :--- | :--- | :--- |
| **Abaqus API 계층** | 미적용 (원시 절차형 스크립트) | `Model`, `Step`, `SolutionControl(ADAPTIVE)` 표준 계층 구축 |
| **기하 및 적층** | 단순 4개 균등층, 피벗 +-3.0mm, 보이드 없음 | 티어드롭 기구학 (피벗 +-1.4mm, 갭 15mm, 넥 갭 2.8mm), 힌지 PSA 보이드 |
| **재료 물성** | 임의 탄성체 ($E_{\text{PSA}}=1.0\,\text{MPa}$) | 공인 물성 (PET $4\,\text{GPa}$, PSA $0.05\,\text{MPa}$, $\nu=0.499$, GLASS $70\,\text{GPa}$) |
| **요소 정식화** | `C3D8I_CR` + `C3D8H_CR` (일부만 적용) | `C3D8I_CR` (9-mode EAS+CR) + `C3D8H_CR` (u-P 하이브리드+CR) 완전 적합 |
| **시간 증분 제어** | 고정 40스텝 (`dt = 0.025`, 컷백 없음) | `AdaptiveDtController(dt_init=0.005, dt_max=0.025)` + 스마트 컷백 |
| **회전 예측자** | 없음 (초기 뉴턴 잔차 큼) | $R(\Delta \theta)$ 기구학 회전 선행 예측자 (잔차 2자리수 감소) |
| **반력/모멘트 모니터링**| 없음 (타이 갭만 출력) | 좌/우 피벗 모멘트 ($M_L, M_R$), 수평 반력 ($Fx$), 대칭 오차, 평형 검증 |
| **CLI 및 프리셋** | CLI 인수 부재 | `--layup`, `--theta_max`, `--max_steps`, `--sol_preset`, `--width` 지원 |

---

## 3. 구현 세부 계획

1. **안전 백업 확인**:
   `examples/ex13_3d_display_plate_fold_backup.py` 생성 완료.
2. **`build_3d_display_and_plate_mesh` 개선**:
   - `FoldModelConfig`의 티어드롭 규격($L_{\text{half}}=7.5\,\text{mm}$, $w_{\text{pivot}}=1.4\,\text{mm}$, 힌지 PSA 보이드) 지원.
   - 레이업 모드 선택:
     * `teardrop_14`: 정본 14개 층(PET 6개, PSA 7개, GLASS 1개, 두께 0.63mm)
     * `compact_4`: 고속 스모크 테스트용 4개 층
3. **`DynamicSolver3D` 풀 루프 고도화**:
   - `AdaptiveDtController` 및 컷백(Cutback) 복원 로직 연동.
   - `compute_section_reactions`를 활용한 좌/우 피벗 구동 모멘트 실시간 계측.
4. **결과 저장 및 PyVista 시각화 연동**:
   - `examples/results/ex13_3d_fold_result.pkl` 저장 및 `view_ex13_pyvista.py`와의 100% 호환성 유지.
5. **독립 검증**:
   - 스모크 테스트 및 1:1 대칭 반력/모멘트 무결성 검증.

---

## 4. 실행 및 무결성 검증 결과 (2026-10-11)

1. **스모크 테스트 (초기 수렴성 검증)**:
   - 메쉬 통계: 1317 노드, 640개 3D 육면체 요소, 3951 DOFs
   - 표면 타이: 좌/우 각 63개 coupled pair 정상 바인딩
   - `AdaptiveDtController` 시간 증분 자동 확장 및 수렴성 확인

2. **힌지 회전 방향 부호 정상화 (상향 접힘 +Y 복원)**:
   - 사용자의 날카로운 관찰 지적에 따라 회전 부호 전수 점검:
     * 좌측 플레이트: 시계 방향 회전 (`-theta`, $dx_0 < 0 \to y_{\text{new}} > 0$)
     * 우측 플레이트: 반시계 방향 회전 (`+theta`, $dx_0 > 0 \to y_{\text{new}} > 0$)
     * 절점 회전 예측자(`carry_contacting_nodes 3D`) 부호 동기화 완료
   - $30^\circ$ 폴딩 검증: $Y_{\max} = +19.461\,\text{mm}$ (이론치 $38 \cdot \sin(30^\circ) \approx 19.0\,\text{mm}$ 일치)

3. **전체 180° 상향 접힘 (90° 회전) 최종 완주**:
   - 명령: `python examples/ex13_3d_display_plate_fold.py --layup compact_4 --theta_max 90.0`
   - 최종 접힘 각도: Plate $\pm 90.00^\circ$ (총 $180.00^\circ$ 완벽한 정립 U자형 티어드롭 완성)
   - 변위 공간 범위: $Y_{\min} = -0.458\,\text{mm}, \quad Y_{\max} = +38.630\,\text{mm}$ (위쪽 $+Y$로 완벽 접힘)
   - 최종 구동 모멘트:
     * $M_{\text{drive}, L} = -0.1526\,\text{N}\cdot\text{mm}$
     * $M_{\text{drive}, R} = +0.1526\,\text{N}\cdot\text{mm}$
   - 대칭 오차율 (Reaction Symmetry Error): **0.0000%**
   - 수평력 합력 ($\sum F_x$): **$-1.735 \times 10^{-17}\,\text{N}$** (기계 정밀도 0 달성)
   - 최종 저장: `examples/results/ex13_3d_fold_result.pkl`

4. **PyVista 3D 인터랙티브 뷰어 검증**:
   - 스크린샷 렌더링 갱신 완료: `python examples/view_ex13_pyvista.py --screenshot`
   - 생성 이미지:
     * `dev_log/figures/ex13_3d_folding_pyvista_front.png` (정면 상향 180도 티어드롭 루프)
     * `dev_log/figures/ex13_3d_folding_pyvista_iso.png` (3D 등각투영 변위 컨투어)
     * `dev_log/figures/ex13_3d_folding_pyvista_material.png` (재질 레이어별 3D 분할 뷰)
