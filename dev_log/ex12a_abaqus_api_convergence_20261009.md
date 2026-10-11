# ex12a Abaqus Python API Architecture & Full Folding Convergence Plan (2026-10-09)

## 1. Objectives & User Requirements
1. **ex12 예제를 복제하여 ex12a 생성**:
   - `examples/ex12a_abaqus_inp_plate_fold.py`
   - `examples/gen_ex12a_inp.py`
   - `examples/ex12a_rigid_plate_display_fold.inp`
2. **최신 Abaqus Python API 아키텍처 적용**:
   - `dispsolver.model` 기반 (`Model`, `Part`, `Step`, `SolutionControl`, `ContactControl`, `Tie`, `RigidBody`) 표준 객체 지향 구조 적용.
   - 솔루션 컨트롤 (`SolutionControl(preset="ADAPTIVE")`), 접촉 제어 (`ContactControl`, `ContactProperty`, `RotatingRigidSegmentContact2D`), 스텝 설정 등 Abaqus 표준 인터페이스 반영.
3. **요소명 표준화 (Abaqus Element Catalog Parity)**:
   - 비표준 레거시 이름 (`Q4_COROTATIONAL_SRI` 등) 제거.
   - Abaqus 표준 2D solid 요소명으로 전면 개편:
     - PET: `CPE4_CR` (Frame-invariant co-rotational, 45도 방향 인공 강성 폭증 제거)
     - GLASS: `CPE4_CR` (Frame-invariant co-rotational)
     - PSA: `CPE4H` (u-P hybrid, 비압축성 체적 잠김 및 EAS 국부 수렴 실패 원천 방지) / `CPE4I`
     - RIGID PLATE: `CPE4` / `CPE4_CR`
4. **90°/side (180° combined) 완전 폴딩 수렴 확보**:
   - 49.5° 벽의 원인이었던 rigid plate corner pinching 해소:
     - `tie_attach_start = 8.5` (또는 코너 버퍼 스트립) + 활성 접촉 (`plate_contact=True`) 적용을 통한 코너 리프트오프(lift-off) 허용.
     - `CPE4H` u-P hybrid 배치를 통한 PSA 체적 잠김 방지.
     - 잔차 기반 수렴 무결성 검증 (엄격한 Anti-Mocking 원칙 준수).

---

## 2. Implementation Architecture

### 2.1 Model Generation (`gen_ex12a_inp.py`)
- `build_display_grid(config)` 및 `create_folding_plate_parts`를 활용하여 고정밀 메쉬 구성.
- `*ELEMENT, TYPE=CPE4_CR`, `*ELEMENT, TYPE=CPE4H` 등 Abaqus 표준 키워드 출력.
- 버퍼 스트립과 접촉 표면 정의.

### 2.2 Solve Pipeline (`ex12a_abaqus_inp_plate_fold.py`)
- `Model` 컨테이너 및 `SolutionControl(preset="ADAPTIVE")` 활용.
- `AdaptiveToleranceEngine` 및 Abaqus Alternative Convergence check ($C_a \cdot \Delta u_{\max}$, $R_p \cdot q_{\text{avg}}$) 연동.
- 90도 회전 완료 시까지 적응형 타임 스텝 및 잔차 무결성 모니터링.

---

## 3. Success Verification Criteria
1. $t = 1.0$ (목표 회전각 $\pm 90^\circ$, 총 $180^\circ$ 완전 접힘) 도달.
2. 메쉬 반전 없음 ($n_{\text{inverted}} = 0$).
3. 힌지 영역 곡률이 꺾임 없이 매끄러운 U자(teardrop) 형태 유지.
4. 양 플레이트가 평행하게 마주보며 닫힘 (Neck gap $2 \times x_{\text{pivot}}$).
