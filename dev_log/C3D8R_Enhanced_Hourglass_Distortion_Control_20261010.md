# C3D8R 물리적 안정화(Puso, 2000) 및 Distortion Control 정식화 보고서

본 문서는 **Anti-Mocking & Integrity Protocol**에 따라 어떠한 난수나 가짜 더미 코드를 배제하고, 공인 원전에 기반한 8절점 육면체(C3D8R) 단일 적분점 요소의 향상된 아워글래스 제어 및 왜곡 제어(Distortion Control)의 수학적 정식화와 구현 방안을 정리합니다.

---

## 1. Puso (2000) Enhanced Hourglass Control 정식화

기존 Flanagan-Belytschko 방식의 인위적 감쇠 계수($\kappa_{hg}$) 대신, Puso(2000)는 9-모드 Enhanced Assumed Strain(C3D8I)의 추가 변형률 모드들을 정적 응축(Static Condensation)하여 닫힌 형식(closed-form)의 향상된 물리적 안정화 강성 행렬 $H$를 유도하였습니다.

### 1.1 기본 기저 및 직교화된 형상 벡터
자연 좌표계($\xi, \eta, \zeta$)에서의 기본 아워글래스 벡터 $\mathbf{h}_\alpha$ ($\alpha=1..4$)는 다음과 같습니다.
$$
\begin{aligned}
\mathbf{h}_1 &= [ 1,  1, -1, -1, -1, -1,  1,  1]^T \\
\mathbf{h}_2 &= [ 1, -1, -1,  1, -1,  1,  1, -1]^T \\
\mathbf{h}_3 &= [ 1, -1,  1, -1,  1, -1,  1, -1]^T \\
\mathbf{h}_4 &= [-1,  1, -1,  1,  1, -1,  1, -1]^T
\end{aligned}
$$
직교화된(Orthogonalized) 아워글래스 벡터 $\boldsymbol{\gamma}_\alpha$는 요소 중심($\xi=\eta=\zeta=0$)에서 선형 구배를 제거하여 구성됩니다:
$$
\boldsymbol{\gamma}_\alpha = \mathbf{h}_\alpha - \sum_{i=1}^3 (\mathbf{h}_\alpha^T \mathbf{x}_i) \nabla N_i
$$
여기서 $\mathbf{x}_i$는 노드 좌표 벡터이고, $\nabla N_i$는 중심에서의 형상함수 구배입니다.

### 1.2 9모드 EAS 정적 응축 등가성
EAS 방식에서 가정된 변형률(Enhanced Strain) 매개변수를 요소 수준에서 소거(Static Condensation)하면, 아워글래스 자유도 $\mathbf{q}_\alpha$에 작용하는 등가 강성 행렬을 유도할 수 있습니다. 
일반화된 아워글래스 변위 $\mathbf{q}_{\alpha i} = \boldsymbol{\gamma}_\alpha^T \mathbf{u}_i$ (여기서 $\mathbf{u}_i$는 방향 $i$에 대한 절점 변위 벡터)에 대해, 물리적 강성 행렬 $\mathbf{H}^{EAS}$는 재료의 탄성 행렬 $\mathbf{D}$ (또는 $\mathbf{C}$)의 함수로 도출됩니다.
체적 잠김(Volumetric Locking)을 방지하기 위해 팽창 성분을 제거한 편차적(Deviatoric) 재료 강성 $\bar{\mathbf{D}}$를 사용하며, 안정화 에너지 파라미터는 형상 텐서와 결합되어 정확한 물리적 구부림 저항을 제공합니다.

### 1.3 대회전(NLGEOM) 시 프레임 불변화 및 일관 접선
대변형(Large Deformation) 하에서 요소의 강체 회전에 객관성을 유지하기 위해, 중심에서 계산된 변형 기울기 $\mathbf{F}$를 극분해(Polar Decomposition)하여 회전 텐서 $\mathbf{R}$ ($\mathbf{F} = \mathbf{R} \mathbf{U}$)을 추출합니다.

- **프레임 불변화된 아워글래스 변위**: $\mathbf{q}^{ref}_\alpha = \mathbf{R}^T \mathbf{q}_\alpha$
- **내력 (Internal Force)**: 
  $\mathbf{f}^{hg} = \sum_{\alpha, \beta} \boldsymbol{\gamma}_\alpha \otimes (\mathbf{R} \mathbf{H}_{\alpha \beta} \mathbf{q}^{ref}_\beta) = \sum_{\alpha, \beta} \boldsymbol{\gamma}_\alpha \otimes (\mathbf{R} \mathbf{H}_{\alpha \beta} \mathbf{R}^T \mathbf{q}_\beta)$
- **일관 접선 강성 (Consistent Tangent)**:
  재료 강성 기여분 외에 $\mathbf{R}$의 변동에 따른 기하 강성(Geometric Stiffness) 기여분을 포함해야 합니다.
  $\delta \mathbf{R} = \delta \mathbf{W} \mathbf{R}$로 정의할 때, $\delta \mathbf{W}$는 $\delta \mathbf{u}$에 대한 미분 $\frac{\partial \mathbf{W}}{\partial \mathbf{u}}$를 통해 도출됩니다 (Belytschko & Bindeman, 1993 참조).

---

## 2. Abaqus 사양 Distortion Control 정식화

Abaqus는 국소 요소 부피비 $J = \det(\mathbf{F})$가 임계치 $r_c$ (기본값 0.1) 이하로 떨어져 극도로 왜곡되거나 뒤집히는 것을 막기 위해 페널티 에너지(Distortion Control)를 도입합니다.

### 2.1 초탄성 복원 에너지 및 응력
복원 변형 에너지 밀도 함수 $\Psi_{dist}(J)$는 다음과 같이 정의됩니다:
$$
\Psi_{dist}(J) = \begin{cases} 
\frac{1}{2} K_p (J - r_c)^2 & \text{if } J < r_c \\ 
0 & \text{if } J \ge r_c 
\end{cases}
$$
이로부터 도출되는 가상의 하이드로스타틱 압력 $p_{dist}$:
$$
p_{dist} = -\frac{\partial \Psi_{dist}}{\partial J} = - K_p (J - r_c)
$$
제2 피올라-키르히호프(2nd Piola-Kirchhoff) 응력 기여분 $\mathbf{S}_{dist}$:
$$
\mathbf{S}_{dist} = 2 \frac{\partial \Psi_{dist}}{\partial \mathbf{C}} = \frac{\partial \Psi_{dist}}{\partial J} \frac{\partial J}{\partial \mathbf{C}} = - p_{dist} J \mathbf{C}^{-1}
$$

### 2.2 4차 일관 접선 강성 텐서 (Consistent Tangent Tensor)
Distortion Control에 의한 강성 기여분 $\mathbb{C}_{dist}$는 다음과 같습니다:
$$
\mathbb{C}_{dist} = 2 \frac{\partial \mathbf{S}_{dist}}{\partial \mathbf{C}} = K_p J^2 (\mathbf{C}^{-1} \otimes \mathbf{C}^{-1}) + 2 p_{dist} J \frac{\partial (\mathbf{C}^{-1})}{\partial \mathbf{C}}
$$
여기서 $\frac{\partial (\mathbf{C}^{-1})}{\partial \mathbf{C}} = -\frac{1}{2} (C^{-1}_{ik} C^{-1}_{jl} + C^{-1}_{il} C^{-1}_{jk})$ 입니다.

- 계수 설계: $K_p$는 요소가 속한 재료의 초기 체적 탄성률(Bulk Modulus) $\kappa$에 비례하는 무차원 스케일 팩터를 적용하여 텐셔닝(Tensioning) 수준을 제어합니다 ($K_p \approx \alpha \kappa$, $\alpha$는 상수).

---

## 3. 기존 코드베이스 인터페이스 정합성 제안

현재 `dispsolver/element3d/c3d8r_numba.py`의 Numba JIT 커널 구조(`compute_c3d8r_element_numba`)에 위 정식화를 통합하기 위해 다음의 pseudo-code 방식을 제안합니다.

### Pseudo-Code (Numba JAX/Python 호환)
```python
@njit(fastmath=True)
def compute_c3d8r_element_numba(...):
    # ... 기존 1~4 단계: J0, dN_dx, B0, F0 계산 ...
    
    # [NEW] 극분해(Polar Decomposition) F0 = R U
    R_mat, U_mat = polar_decomposition_3x3(F0)
    
    # [NEW] Distortion Control (J < r_c)
    r_c = 0.1
    dist_energy_flag = 1 if detF0 < r_c else 0
    if dist_energy_flag:
        K_p = alpha_dist * bulk_modulus
        p_dist = - K_p * (detF0 - r_c)
        S_dist = - p_dist * detF0 * invC0  # invC0 = inv(F0^T F0)
        # S_dist를 응력 텐서에 합산하거나 잔여력(f_int)에 직접 가산
        # 접선 강성 K_elem 에 C_dist 기여분 가산
    
    # ... 기존 5 단계: UMAT 응력 계산 ...
    
    # [NEW] Puso(2000) Enhanced Hourglass
    # 기존 kappa_hg 기반의 식을 대체하여 H_enh 행렬 계산 (정적 응축식 기반)
    # H_enh는 편차 탄성 행렬(G)과 형태 텐서로 유도됨
    for alpha in range(4):
        # 1. q 변위 계산
        q_alpha = gamma[alpha] @ u_elem_reshaped  # (3,)
        # 2. 객관적 프레임으로 회전
        q_ref = R_mat.T @ q_alpha
        # 3. 안정화 힘 (H_enh 적용)
        f_hg_ref = H_enh[alpha] @ q_ref
        f_hg = R_mat @ f_hg_ref
        
        # 4. 내력 f_int에 가산
        for a in range(8):
            f_int[3*a : 3*a+3] += f_hg * gamma[alpha, a]
            
        # 5. 접선 강성 K_elem에 재료(H_enh) + 기하(dR/du) 강성 가산
        # Geometric part requires spin tensor variation w.r.t displacement
```

### 인터페이스 확장
- `elem_controls` 파라미터(현재 크기 6)를 확장하여 `r_c` (Distortion Control 한계) 및 물리적 아워글래스 제어 스위치(예: `alpha_hg` 플래그)를 전달.
- JAX 버전(`q4_visco_simo_fs_jax.py`)의 경우 JAX의 `jax.numpy.linalg.svd`나 폴라 분해 루틴을 사용하여 `jax.grad`를 통한 $\partial \mathbf{R} / \partial \mathbf{u}$ 자동 미분(AutoDiff) 처리로 구현 복잡성을 대폭 낮출 수 있습니다.
