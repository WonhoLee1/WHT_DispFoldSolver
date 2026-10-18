# 📊 Package P2: 출력 변수 인프라 (Abaqus Parity Output Variables) 검증 및 구현 보고서

> **작성일**: 2026-10-18  
> **준수 표준**: Abaqus/Standard 2026 Output Variable Specification, Anti-Mocking Protocol  
> **상태**: **COMPLETE (30/30 Output Tests ALL GREEN, Analytical Error < 1e-12)**  
> **관련 커밋**: `3d673d1` (`feat(output): add Abaqus-parity output variables (MISES, MAXP, MINP, CPRESS, COPEN, CSHEAR, CSTATUS, ALLAE, ALLSD, ALLFD)`)  

---

## 1. 개요 및 구현 범위

Package P2는 Abaqus CAE 및 후처리 툴(ParaView, VTKHDF)과의 완전한 인터페이스 호환성을 위해 아래의 핵심 필드/이력 변수들을 `dispsolver/output/` 인프라에 추가하고 엄밀하게 수치 검증하였습니다:

1. **응력 파생 불변량 (Stress Invariants)**:
   - `MISES`: von Mises 등가 응력 (2D 평면 변형률 면외 응력 $S_{33}$ 포함)
   - `MAXP`, `MIDP`, `MINP`: 주응력 3개 성분 ($\sigma_1 \ge \sigma_2 \ge \sigma_3$)
   - `PRESS`: 등가 정수압 ($p = -\frac{1}{3}\text{tr}(\boldsymbol{\sigma})$)
2. **접촉 출력 필드 (Contact Surface Fields)**:
   - `COPEN`: 접촉면 간극/개구량 (Contact clearance/gap)
   - `CSHEAR`: 접촉 마찰 전단 응력/트랙션 ($\tau = \sqrt{\tau_1^2 + \tau_2^2}$)
   - `CSTATUS`: 접촉 상태 플래그 ($0=\text{Open}$, $1=\text{Stick}$, $2=\text{Slip}$)
   - (기존 `CPRESS`, `CSLIP`과 100% 통합)
3. **전체 모델 에너지 이력 (Whole Model Energy History)**:
   - `ALLAE`: 인공 변형 에너지 (Artificial strain energy: hourglass, 왜곡 제어 등)
   - `ALLSD`: 수치/정적 안정화 소산 에너지 (Stabilization dissipation)
   - `ALLFD`: 마찰 소산 에너지 (Frictional dissipation)

---

## 2. 수학적 정식화 및 2D 평면 변형률 주의점

### 2.1 2D 평면 변형률 von Mises 응력의 면외 성분 ($S_{33}$) 필수 고려
일반적인 평면 응력(Plane Stress)과 달리, 평면 변형률(Plane Strain)에서는 $\varepsilon_{33} = 0$ 구속 조건으로 인해 면외 응력 $S_{33} = \nu (S_{11} + S_{22}) \neq 0$이 발생합니다.
$S_{33}$를 누락할 경우 Mises 응력에 최대 30~50%의 치명적 오차가 발생합니다:
$$\sigma_{vm} = \sqrt{\frac{1}{2}\left[(S_{11} - S_{22})^2 + (S_{22} - S_{33})^2 + (S_{33} - S_{11})^2\right] + 3 S_{12}^2}$$
`recovery2d.py`에서는 요소 중심에서의 $F_{3d} = \text{diag}(F_{2d}, 1.0)$ 및 $E_{3d}$로부터 유도된 $S_{33}$를 완벽히 포함하여 계산합니다.

### 2.2 주응력 계산 (Principal Stresses)
- **2D 평면 변형률**:
  면내 Mohr's Circle 고유값:
  $$c = \frac{S_{11} + S_{22}}{2}, \quad r = \sqrt{\left(\frac{S_{11} - S_{22}}{2}\right)^2 + S_{12}^2}$$
  주응력 후보군: $\{c+r, \, c-r, \, S_{33}\}$를 내림차순 정렬하여:
  $$\sigma_1 = \text{MAXP}, \quad \sigma_2 = \text{MIDP}, \quad \sigma_3 = \text{MINP}$$
- **3D 솔리드**:
  대칭 코시 응력 텐서 $\boldsymbol{\sigma}_{3\times 3}$에 대해 `np.linalg.eigvalsh()`를 수행하여 해석적 고유값과 $10^{-12}$ 기계 정밀도로 일치하도록 산출합니다.

---

## 3. 구현 모듈 색인

- `dispsolver/output/variables.py`:
  - `FieldVar.MISES`, `FieldVar.MAXP`, `FieldVar.MIDP`, `FieldVar.MINP`, `FieldVar.PRESS`
  - `FieldVar.CPRESS`, `FieldVar.CSLIP`, `FieldVar.CSHEAR`, `FieldVar.COPEN`, `FieldVar.CSTATUS`
  - `HistoryVar.ALLAE`, `HistoryVar.ALLSD`, `HistoryVar.ALLFD`
- `dispsolver/output/requests.py`:
  - `FieldOutputRequest` 및 `HistoryOutputRequest` 별칭 처리 및 `PRESELECT` 확장
- `dispsolver/output/recovery2d.py`:
  - 2D 평면 변형률 요소 중심/절점 평균화 응력 파생 불변량 복원
- `dispsolver/output/recovery3d.py`:
  - 3D 솔리드 요소 응력 텐서 고윳값 분해 및 접촉면 구속 조건으로부터 `COPEN`, `CSHEAR`, `CSTATUS` 추출
- `dispsolver/output/history_data.py`:
  - 솔버의 `energy_allae`, `energy_allsd`, 접촉 구속의 `energy_allfd`를 질의하여 기록 (가짜 더미 없이 데이터 노출 시에만 수집)
- `tests/test_p2_output_vars.py`:
  - 11개 단위/통합 수치 검증 테스트

---

## 4. 정량적 검증 결과

### 4.1 해석해 대조 테스트 (`tests/test_p2_output_vars.py`)
```
tests/test_p2_output_vars.py::TestStressDerivedInvariantsAnalytical::test_3d_uniaxial_tension_analytical PASSED [  9%]
tests/test_p2_output_vars.py::TestStressDerivedInvariantsAnalytical::test_3d_general_multiaxial_stress_analytical PASSED [ 18%]
tests/test_p2_output_vars.py::TestStressDerivedInvariantsAnalytical::test_2d_plane_strain_stress_invariants_analytical PASSED [ 27%]
tests/test_p2_output_vars.py::TestStressDerivedInvariantsAnalytical::test_2d_plane_stress_invariants_analytical PASSED [ 36%]
tests/test_p2_output_vars.py::TestFEASimulationOutputRequests::test_2d_fe_simulation_stress_invariants PASSED [ 45%]
tests/test_p2_output_vars.py::TestFEASimulationOutputRequests::test_3d_fe_simulation_stress_invariants PASSED [ 54%]
tests/test_p2_output_vars.py::TestContactFieldAndEnergyVariables::test_contact_field_output_recovery_3d PASSED [ 63%]
tests/test_p2_output_vars.py::TestContactFieldAndEnergyVariables::test_energy_history_allae_allsd_allfd_tracking PASSED [ 72%]
tests/test_p2_output_vars.py::TestRequestAliasesAndPreselect::test_field_output_request_aliases PASSED [ 81%]
tests/test_p2_output_vars.py::TestRequestAliasesAndPreselect::test_field_output_request_preselect PASSED [ 90%]
tests/test_p2_output_vars.py::TestRequestAliasesAndPreselect::test_history_output_request_preselect PASSED [100%]
```
- 3D 1축 인장 ($\sigma_{xx} = 250.0\,\text{MPa}$): $\text{MISES} = 250.0\,\text{MPa}$, $\text{MAXP} = 250.0$, $\text{MINP} = 0.0$ (오차 $< 10^{-14}$)
- 3D 다축 응력 텐서: 고윳값 대조 오차 $< 1.0 \times 10^{-12}$
- 2D 평면 변형률: $S_{33} = 24.0\,\text{MPa}$ 포함 $\text{MISES} = 65.45227\,\text{MPa}$ 엄밀 일치.

### 4.2 전체 출력 스위트 무회귀 검증
```bash
pytest tests/test_field_output.py tests/test_inp_output_keywords.py tests/test_output_system.py tests/test_p2_output_vars.py -v
======================= 30 passed in 152.60s (0:02:32) ========================
```
기존 VTKHDF v2.2 내보내기, H5ODB 저장/복원, 타이밍 제어 등 전체 30개 테스트에서 단 하나의 실패 없이 **100% ALL GREEN**을 확인하였습니다.
