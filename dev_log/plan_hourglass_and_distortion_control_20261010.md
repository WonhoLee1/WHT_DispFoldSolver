# 2D/3D 유한요소 아워글래스 제어·왜곡 제어 마스터 구현 및 수치역학 Pro 에이전트 규칙화 계획서

> **문서 버전**: v3.0 (수치역학 Pro 서브에이전트 규칙 제정 및 최종 수식 검증 반영)  
> **일자**: 2026-10-10  
> **작성자**: Antigravity (Advanced Agentic Pair-Programming w/ Pro Subagent)  
> **상태**: 사용자 승인 대기 중 (`RequestFeedback=True`)

---

## 1. [Goal Description]

본 작업은 다음 네 가지 핵심 과제를 체계적으로 완수합니다:

1. **`GEMINI.md` 수치역학 전문 Pro 서브에이전트 규칙 제정 (Protocol Rule Creation)**:
   - 전산역학/비선형 유한요소 정식화, 구성방정식, 텐서 미분 및 수치 발산 분석 시 **`Pro` 모델 전문 서브에이전트(`FEA Solid Mechanics Architect`)**를 의무적으로 호출하여 신선한 컨텍스트(Fresh Context)에서 엄밀한 수식 유도를 수행하고, 메인 에이전트가 2차 독립 검증(2nd-Pass Independent Verification)하는 안티-모킹 규칙을 `GEMINI.md`에 영구 명문화.
2. **타 에이전트 브랜치 통합 (`wip/cpe4r-enhanced`, 커밋 `47fb5f1`)**:
   - JAX 기반 `CPE4R` (Q1E4 기반 ENHANCED 및 Flanagan-Belytschko STIFFNESS) 정합성 통합.
   - 로컬 작업 트리에서 달성한 **`ex12a` 90°/side 완전 폴딩 성공 솔버 상태**(Direct Equilibrium Check, Small Correction Floor, Local Peak DOF 수치 진동 감쇠, Rigid Segment 접촉 노드 전진 추적)를 100% 보존.
3. **2D/3D 감차적분 솔리드 요소 아워글래스 체계 완성 (Full Hourglass Suite)**:
   - 1점 감차적분(1-point reduced integration)으로 인해 스퓨리어스 영에너지 모드(hourglass mode)가 존재하는 모든 요소에 대해 **`ENHANCED`** 및 **`STIFFNESS`** 정식화 완비.
   - **2D**: `CPE4R` (평면 변형률 Numba 및 JAX), `CPS4R` (평면 응력), `CAX4R` (축대칭).
   - **3D**: `C3D8R` (8절점 육면체) — **Puso (2000) & Belytschko-Bindeman (1993)** 9모드 EAS 정적 응축 기반 **3D `ENHANCED` 물리적 안정화** 및 중심 극분해(Polar Decomposition $\mathbf{F}_0 = \mathbf{R} \mathbf{U}$) 대회전 프레임 불변화 신규 구현.
4. **전천후 왜곡 제어 (Universal Distortion Control Engine)**:
   - Abaqus `*SECTION CONTROLS, DISTORTION CONTROL=YES, LENGTH RATIO=r` 표준 구현.
   - 국소 체적비 $J = \det(\mathbf{F}) < r_c$ (기본값 $0.10$) 영역에서 $C^1/C^2$ 연속 에너지 배리어 $\Psi_{\text{dist}}(J)$, 복원 응력 $\mathbf{S}_{\text{dist}}$, 4차 일관 접선 강성 텐서 $\mathbb{C}_{\text{dist}}$ 주입.
   - 2D 및 3D 지원 가능한 모든 요소군(`CPE4/R/I/H/FBAR`, `CPS4/R/I`, `CAX4/R`, `C3D8/R/I/H/FBAR/CR`, `C3D4/ANP`, `C3D6`, `C3D10`) 적용.

---

## 2. User Review Required

> [!IMPORTANT]
> **1. 수치역학 Pro 에이전트 2차 독립 검증 의무화**:
> - 앞으로 비선형 유한요소 기구학, 텐서 미분, 접선 강성 행렬을 유도할 때, 메인 에이전트 단독의 "어림짐작이나 임의 상수(`5000*kappa` 등) 삽입"을 금지합니다.
> - Pro 서브에이전트가 유도한 수식을 메인 에이전트가 라인 단위로 인스펙션한 후에만 코드베이스에 반영합니다.

> [!IMPORTANT]
> **2. 요소 적용 범위의 물리적 한계 (Anti-Overstiffening Protocol)**:
> - Hourglass 제어는 1점 감차적분 요소(`CPE4R`, `CPS4R`, `CAX4R`, `C3D8R`)에만 한정됩니다. 완전 적분, EAS, F-bar, 하이브리드 등 이미 풀 랭크(Full Rank)를 갖는 요소에는 적용하지 않습니다.
> - 왜곡 제어(Distortion Control)는 모든 연속체 요소의 대변형 인버전($J \le 0$) 방지를 위해 전 요소에 일관 적용됩니다.

---

## 3. Open Questions

1. **규칙 반영 즉시 실행 여부**:
   - 계획서 승인 즉시 Phase 1(GEMINI.md 규칙 제정 및 Git 브랜치 통합)을 실행하여 환경을 확정한 후, Phase 2(2D/3D 요소 및 왜곡제어 개발)로 진입하고자 합니다. 동의하십니까?
2. **왜곡 제어 기본 임계값(`length_ratio`)**:
   - Abaqus 기본값 $r_c = 0.10$을 디폴트로 설정하고 덱(`*SECTION CONTROLS`)에서 오버라이드 가능하게 구성합니다.

---

## 4. Proposed Changes

- **Component 0**: 프로젝트 룰 ([`GEMINI.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/GEMINI.md)) 수치역학 전문 Pro 에이전트 프로토콜 제정
- **Component 1**: Git 브랜치 통합 (`wip/cpe4r-enhanced`) 및 솔버 무결성 보존
- **Component 2**: 2D 솔리드 요소군 Hourglass (`CPE4R` Numba, `CPS4R`, `CAX4R`) 확장
- **Component 3**: 3D 솔리드 요소군 Enhanced Hourglass (Puso 2000 & Polar Decomposition) 구현
- **Component 4**: Universal Distortion Control Engine (2D & 3D 에너지 배리어 및 Line-search Inversion Safeguard)

---

## 5. Verification Plan

- `python -m pytest tests/test_cpe4r_element.py -v` (CPE4R 8개 테스트 100% PASS 확인)
- `python -m pytest tests/test_hourglass_3d_puso.py -v` (3D Puso Enhanced 굽힘 락킹 방지)
- `python -m pytest tests/test_distortion_control_barrier.py -v` (극심 압축 인버전 방어 A/B 테스트)
- `python examples/ex12a_abaqus_inp_plate_fold.py` (90도 101 스텝 무결점 폴딩 최종 확인)
