# WHT_DispFoldSolver: 3D MITC Shell Element (MITC3+ / MITC4+) Design Roadmap

## 개요
본 설계서는 K.J. Bathe 및 P.S. Lee 교수가 정립한 최신 MITC (Mixed Interpolation of Tensorial Components) 쉘 정식화(MITC3+, MITC4+)를 바탕으로, WHT_DispFoldSolver에 3차원 공간 곡면 쉘 요소를 탑재하기 위한 아키텍처 및 구현 로드맵을 정의합니다. 본 요소들은 상용 유한요소 프로그램(Abaqus 등)의 성능을 능가하는 정확도와 수치적 안정성을 목표로 합니다.

## 1. 3D 공간 상의 쉘 요소 기구학 및 자유도 (Kinematics & DOFs)

### 1.1 절점 자유도 설계
*   **6 DOFs per Node**: 3차원 공간상에서 작동하는 범용 쉘 요소로써 각 절점당 6개의 자유도($u_x, u_y, u_z, \theta_x, \theta_y, \theta_z$)를 부여합니다.
*   **Director Vector 정식화**: 쉘의 두께 방향 법선 벡터(Director)의 거동을 추적하여 대변형 회전(Co-rotational / Director formulation)을 효과적으로 모사합니다.

### 1.2 드릴링 자유도 (Drilling DOF) 안정화
*   평면 내 회전(In-plane rotation)에 해당하는 법선 회전(Drilling DOF)은 기본 기구학에서 강성을 갖지 않으므로 수치적 특이점(Singularity)을 유발할 수 있습니다.
*   이를 해결하기 위해 **Hughes-Brezzi 드릴링 변분(Variational formulation)** 원리를 도입하여 가상 강성을 부여합니다. 이를 통해 3D 복합 조립체(보, 쉘, 솔리드 결합)와의 6-DOF 결합성을 완벽하게 보장합니다.

### 1.3 복합재 및 두께 방향 적분
*   단일 등방성 물질뿐만 아니라 **복합재 적층(Composite/Multilayer)** 구조를 지원합니다.
*   두께 방향 수치 적분은 **Gauss-Lobatto 5~7점** 또는 **Simpson's rule**을 적용하여, 표면 응력의 정확한 도출과 소성 항복(Plastic yielding) 시작점의 정밀한 포착이 가능하도록 아키텍처를 구성합니다.

---

## 2. MITC 최신 후속 이론 분석 및 적용

### 2.1 MITC3+ (3-node triangular shell)
*   **이론적 배경**: Lee, Y., Lee, P.S., & Bathe, K.J. (2014, 2017)
*   **주요 특징**:
    *   전통적인 3절점 쉘 요소가 겪는 **막 잠김(Membrane locking) 및 전단 잠김(Shear locking)** 현상을 혼합 텐서 보간법(Mixed Tensorial Interpolation)을 통해 완전히 해결하였습니다.
    *   **가짜 영에너지 모드(Spurious zero-energy mode)**가 존재하지 않아 안정적인 해를 제공합니다.
*   **경쟁력**: Abaqus S3/S3R 모델이 종종 과도한 강성(Stiff behavior)을 보이는 반면, MITC3+는 비정형 삼각 메쉬에서도 뛰어난 유연성과 빠른 수렴 속도를 입증합니다.

### 2.2 MITC4+ (4-node quadrilateral shell)
*   **이론적 배경**: Ko, Y., Lee, P.S., & Bathe, K.J. (2017)
*   **주요 특징**:
    *   비평면 워핑(Out-of-plane warping distortion) 및 심각한 사다리꼴 메쉬 왜곡 환경에서도 잠김 현상을 극복한 궁극의 4절점 쉘 요소입니다.
*   **경쟁력**: Abaqus S4R의 경우 1점 감차적분으로 인한 아워글래스(Hourglass) 튜닝이 필수적이며 S4는 왜곡에 취약한 반면, MITC4+는 인공적인 튜닝 파라미터 없이도 대변형 비선형 굽힘에서 압도적인 수렴성과 정확도를 보여줍니다.

---

## 3. 구현 아키텍처 및 로드맵

### 3.1 모듈 구조 설계 (`dispsolver/element_shell/`)
새로운 쉘 요소들은 Numba JIT 컴파일러를 활용해 최상의 연산 성능을 발휘하도록 아래와 같은 구조로 구현됩니다.
*   `shell_kinematics.py`: 6-DOF 기구학, Director 벡터 업데이트, Hughes-Brezzi 드릴링 강성, 두께 방향 적분점(Gauss-Lobatto) 생성 유틸리티.
*   `mitc3_plus_numba.py`: 3절점 MITC3+ 정식화 (변형률 텐서 혼합 보간법, 강성 행렬 및 내력 벡터 계산).
*   `mitc4_plus_numba.py`: 4절점 MITC4+ 정식화 (Warping 보정, 타이 변형률(Tying strain) 보간, 자코비안 매핑).

### 3.2 Co-rotational 3D 쉘 프레임워크 결합
*   기하학적 비선형성을 위해 Co-rotational 프레임워크 내에 MITC 요소를 임베딩합니다.
*   강체 회전(Rigid body rotation)을 국부 좌표계에서 분리하고, 순수 변형(Pure deformation)에 대해서만 MITC+, MITC4+ 정식화를 적용한 후 글로벌 좌표계로 변환합니다.

### 3.3 검증 벤치마크 테스트 (Validation Plan)
정확성과 신뢰성을 증명하기 위해 다음의 공인 쉘 벤치마크를 수행합니다.
1.  **Scordelis-Lo 지붕 (Scordelis-Lo Roof)**: 복합 굽힘 및 멤브레인 거동 검증 (메쉬 수렴성).
2.  **침곡관 핀칭 (Pinched Cylinder)**: 극심한 전단 잠김 극복 여부 및 쉘 유연성 검증.
3.  **비틀린 보 / 왜곡 판재 굽힘 (Twisted Beam / MacNeal's Distorted Plate)**: 워핑 및 메쉬 찌그러짐(Distortion) 하에서의 강건성(Robustness) 검증.
