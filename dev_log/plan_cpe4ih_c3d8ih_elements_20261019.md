# Implementation Plan: `CPE4IH` (2D) & `C3D8IH` (3D) Incompatible Modes + Hybrid ($u\text{-}\boldsymbol{\alpha}\text{-}\mathbf{p}$) Elements

## 1. 배경 및 문제 정의 (Background & Motivation)

### 1.1 물리적·수치적 배경
적층 폴더블 디스플레이에서 점착층(PSA)이나 탄성/초탄성 코팅층이 굽힘 변형을 겪을 때, 기존 1차 요소들은 다음 두 가지 잠김(Locking) 중 하나에 노출됩니다:
1. **`CPE4H` / `C3D8H` (표준 Hybrid $u\text{-}p$ 요소)**:
   - 요소 평균 체적 변형률 $\bar{\varepsilon}_v$만 상수 압력 $p_0$로 치환하므로 $\nu \to 0.5$ 체적 잠김(Volumetric Locking)은 해소됩니다.
   - 그러나 1차 완전적분(Full Integration) 기구학을 그대로 사용하므로 순수 굽힘 시 가우스 적분점에서 기생 전단 변형률($\gamma_{xy} \sim \xi$ 또는 $\eta$)이 발생하여 **단일 층 탄성 굽힘에서 $+8.2\%$ 전단 잠김(Shear Locking)**이 남습니다.
2. **`CPE4I` / `C3D8I` (비적합 모드 / EAS 요소)**:
   - Wilson–Taylor–Simo–Rifai 비적합 모드($\boldsymbol{\alpha}$)를 통해 기생 전단 변형률과 푸아송 두께 수축 변형률을 흡수하여 굽힘 오차를 $0\%$로 만듭니다.
   - 그러나 체적 구속이 모든 가우스 적분점에서 페널티 계수 $K = \frac{E}{3(1-2\nu)}$로 직접 부과되므로, $\nu \to 0.4999\dots$ (근접 비압축성) 극한에서는 정적 응축 행렬 $\mathbf{K}_{\alpha\alpha}$ 및 전체 강성 행렬의 조건수 악화·체적 잠김이 발생할 수 있습니다.

### 1.2 현재 코드베이스의 `CPE4IH` / `C3D8IH` 상태
1. **Numba 솔버 카탈로그 (`dispsolver/model/element_catalog.py`)**:
   - `CPE4IH` (Line 220): `NLGEOM=YES`일 때 커널 12 (`CPE4H_CR`, 비적합 모드 없음)로 폴백되고, `NLGEOM=NO`일 때 커널 1 (`CPE4I`, 하이브리드 압력 없음)로 폴백됩니다.
   - `C3D8IH` (Line 181): `NLGEOM=YES/NO` 모두 `C3D8I` (커널 11/18, 하이브리드 압력 없음)로만 구동됩니다.
2. **JAX 프로토타입 (`dispsolver/element/q4_visco_eas_jax.py`)의 기존 결함 C6, C7, C8 (`tests/element_contract/test_contract.py`)**:
   - `_residuals_h` (Line 400)에서 `_MASK = [0, 1, 1, 0]`으로 자연좌표 수직 모드($\alpha_0 \xi, \alpha_3 \eta$)를 인위적으로 마스킹하고 상수 압력 $p_0$만 사용했습니다.
   - 이로 인해 회전된 요소에서 텐서 등방성이 깨져 **C7(45° 회전 이방성 5.68%)**, **C6(0° 굽힘 시 과도한 연화 0.467배 vs 90° 굽힘 시 잠김 20.0배)** 결함이 발생했습니다.

---

## 2. 수학적 정식화 (Mathematical Formulation: Abaqus `CPE4IH` & `C3D8IH` Theory)

Abaqus Theory Manual 3.2.3 & Elements Guide에서 `CPE4IH`와 `C3D8IH`의 공식 명칭은 다음과 같습니다:
- **`CPE4IH`**: *4-node bilinear, incompatible modes, hybrid with **linear pressure***
- **`C3D8IH`**: *8-node linear brick, incompatible modes, hybrid with **linear pressure***

### 2.1 왜 선형 하이브리드 압력(Linear Hybrid Pressure)인가?
4절점 평면변형률 요소가 $x$축 방향으로 순수 굽힘($\kappa$)을 받을 때:
- 적합 변위장으로부터 나오는 축방향 변형률은 $\varepsilon_{xx}^h = -\kappa y \propto \eta$입니다.
- 비압축성($\nu \to 0.5$) 평면변형률 굽힘의 이론 정해는 모든 점에서 체적 변형률이 0($\varepsilon_{xx} + \varepsilon_{yy} = 0$)이어야 하므로, 두께 방향 수직 변형률이 **$\varepsilon_{yy} = +\kappa y \propto \eta$**가 되어야 하며 이때 굽힘 강성은 $\frac{E}{1-\nu^2} I = \frac{4}{3} E I$입니다.
- 만약 EAS 수직 모드($\tilde{\varepsilon}_{yy} = \alpha_1 \eta$)를 켜되 체적 구속을 **상수 압력 $p_0$ (1개 자유도)**로만 걸면, $\int_\Omega p_0 \eta \, d\Omega = 0$이므로 선형 팽창 성분($\xi, \eta$)에 대한 체적 강성이 $0$이 되어 $\sigma_{yy} \to 0$으로 풀려버립니다(평면변형률이 평면응력이나 비정상 연화 모드로 붕괴).
- 반대로 EAS 수직 모드를 끄면(`[0, 1, 1, 0]`), $\varepsilon_{yy} = 0$이 되어 가우스점에서 $\operatorname{tr}(\boldsymbol{\varepsilon}) = -\kappa y \neq 0$이 남고 좌표 회전에 대해 불변성이 깨집니다.
- 따라서 **전체 EAS 모드(2D 4-모드 / 3D 9-모드)**와 **선형 압력장(2D 3-자유도 $p_0 + p_\xi \xi + p_\eta \eta$ / 3D 4-자유도 $p_0 + p_\xi \xi + p_\eta \eta + p_\zeta \zeta$)**을 결합하면:
  1. 선형 압력장 $\mathbf{p}$가 1차 다항식 공간 $\{1, \xi, \eta(, \zeta)\}$에서 점별 비압축성($\varepsilon_{xx}^h + \tilde{\varepsilon}_{yy} = \frac{p}{K} \to 0$)을 라그랑주 승수로 부과하여 $\tilde{\varepsilon}_{yy} = -\varepsilon_{xx}^h = +\kappa y$를 정확히 강제합니다.
  2. 전단 EAS 모드가 기생 전단 변형률을 $0$으로 소거합니다.
  3. 결과적으로 **임의의 요소 회전각($0^\circ \dots 360^\circ$)과 $\nu \to 0.499999$ 비압축성 극한에서도 전단 잠김 $0\%$, 체적 잠김 $0\%$, 이론 굽힘 강성비 $1.0000$**을 달성합니다.

### 2.2 3-장 혼합 변분 정식화 및 요소 수준 이중 정적 응축 (Element-Level Static Condensation)
요소 내 총 변형률 $\boldsymbol{\varepsilon}$을 적합 변형률 $\mathbf{B}\mathbf{u}$와 Simo–Rifai 비적합 변형률 $\tilde{\mathbf{B}}\boldsymbol{\alpha}$의 합으로 정의합니다:
$$\boldsymbol{\varepsilon}(\mathbf{u}, \boldsymbol{\alpha}) = \mathbf{B}(\boldsymbol{\xi})\mathbf{u}_e + \tilde{\mathbf{B}}(\boldsymbol{\xi})\boldsymbol{\alpha}, \qquad \tilde{\mathbf{B}}(\boldsymbol{\xi}) = \frac{j_0}{j(\boldsymbol{\xi})} \mathbf{T}_0^{-\mathsf{T}} \mathbf{M}(\boldsymbol{\xi})$$
응력-변형률 구성 방정식을 편차(Deviatoric) 성분과 선형 하이브리드 압력(Hydrostatic Pressure) $\tilde{p}(\boldsymbol{\xi}) = \mathbf{N}_p(\boldsymbol{\xi}) \mathbf{p}_e$로 분리합니다:
$$\boldsymbol{\sigma}(\mathbf{u}, \boldsymbol{\alpha}, \mathbf{p}_e) = \boldsymbol{\sigma}_{\text{dev}}\!\left(\boldsymbol{\varepsilon}(\mathbf{u}, \boldsymbol{\alpha})\right) + \tilde{p}(\boldsymbol{\xi})\,\mathbf{m}$$

요소 수준 내부 자유도 $\mathbf{q}_e = [\boldsymbol{\alpha}^\mathsf{T}, \mathbf{p}_e^\mathsf{T}]^\mathsf{T}$를 정적 응축하여 외부에는 순수 변위 자유도($8\times 8$ 또는 $24\times 24$)의 **대칭 양정치(SPD) 강성 행렬 $\mathbf{K}_e$**와 내력 벡터 $\mathbf{f}_{\text{int},e}$만 노출합니다:
$$\mathbf{K}_e = \mathbf{K}_{uu} - \mathbf{K}_{uq} \mathbf{K}_{qq}^{-1} \mathbf{K}_{qu}$$

---

## 3. 구현 단계 (Implementation Steps)

1. **Phase 1**: `dispsolver/element2d/cpe4ih_numba.py` 작성 (`CPE4IH` small-strain 및 `CPE4IH_CR` co-rotational 커널, 병렬 어셈블러)
2. **Phase 2**: `dispsolver/element3d/c3d8ih_numba.py` 작성 (`C3D8IH` small-strain 및 `C3D8IH_CR` co-rotational 커널, 병렬 어셈블러)
3. **Phase 3**: `dispsolver/element/q4_visco_eas_jax.py` 내 `_residuals_h` 교정 (4-모드 EAS + 선형 압력장으로 C6, C7, C8 결함 해소)
4. **Phase 4**: `element_catalog.py`, `dynamic2d.py`, `dynamic3d.py`, `example_disp_fold/model_spec.py` 연결
5. **Phase 5**: `tests/test_cpe4ih_c3d8ih.py` 및 `tests/element_contract/test_contract.py` 전수 검증 및 문서(`LESSONS_LEARNED.md`, `release_notes_element.md`) 업데이트
