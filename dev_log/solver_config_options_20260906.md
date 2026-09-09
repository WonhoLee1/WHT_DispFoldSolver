# WHT_DispFoldSolver — FEA 솔버 전체 설정 옵션 레퍼런스 Guide

**작성일자:** 2026년 9월 6일  
**문서 위치:** `dev_log/solver_config_options_20260906.md`  
**관련 모듈:** `dispsolver/fold_model_config.py`, `dispsolver/solver/dynamic.py`, `dispsolver/solver/dt_controller.py`

---

## 1. 개요 (Overview)

본 문서는 `WHT_DispFoldSolver` 폴딩 해석 솔버에서 사용하는 **모든 비선형 FEA 솔버 설정 옵션 및 매개변수**를 카테고리별로 정량화하여 정리한 표준 레퍼런스 가이드입니다.

디스플레이 패널 폴딩 시 초박형 유리($60\,\mu\text{m}$)와 점탄성 PSA 접착층의 고비선형 대변형 결합 문제를 안정적이고 고속으로 해석하기 위해 각 옵션의 서술, 기본값, 설정가능 범위 및 물리적/수치적 의미를 명시합니다.

---

## 2. 솔버 설정 옵션 전체 분류표 (Solver Configuration Options Table)

### 2.1 시간 적분 및 동역학 설정 (Time Integration & Dynamic Modes)

| 파라미터명 | 타입 | 기본값 | 설정 가능 옵션/범위 | 물리적 / 수치적 의미 및 추천 설정 |
| :--- | :--- | :--- | :--- | :--- |
| `integration_mode` | `str` | `"moderate-4"` | `"quasistatic"`, `"transient"`, `"moderate-1"`~`"moderate-5"`, `"generalized-0.9"`~`"generalized-0.1"` | 시간 적분 스킴 지정. Abaqus 표준 `moderate-3` ($\alpha=-0.15$), 활성 설정 `moderate-4` ($\alpha=-0.22$), OptiStruct 0.1-step 레퍼런스는 [optistruct_galpha_table_20260906.md](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/optistruct_galpha_table_20260906.md) 참조 |
| `alpha` ($\alpha$) | `float` | `-0.15` | `-0.33` ~ `0.00` | HHT-$\alpha$ 시간 적분 감쇄 파라미터. 음수가 클수록 고주파 수치 노이즈 강력 감쇄 |
| `beta` ($\beta$) | `float` | `0.3306` | $(1-\alpha)^2 / 4$ | Newmark 가속도 증분 파라미터. HHT-$\alpha$ 호환을 위해 자동 계산됨 |
| `gamma` ($\gamma$) | `float` | `0.6500` | $0.5 - \alpha$ | Newmark 속도 증분 파라미터. HHT-$\alpha$ 호환을 위해 자동 계산됨 |
| `static_mode` | `bool` | `False` | `True` / `False` | 관성력 항($\mathbf{M}\mathbf{a}$) 활성화 여부. `quasistatic` 선택 시 `True`로 자동 변경 |

---

### 2.2 적응형 시간 증분 제어 (Adaptive Time-Stepping & Drive)

| 파라미터명 | 타입 | 기본값 | 설정 가능 옵션/범위 | 물리적 / 수치적 의미 및 추천 설정 |
| :--- | :--- | :--- | :--- | :--- |
| `t_total` | `float` | `1.0` | $> 0.0$ | 전체 해석 규격화 시간 ($t \in [0, t_{\text{total}}]$) |
| `dt_init` | `float` | `0.0025` | `dt_min` ~ `dt_max` | 해석 시작 시 초기 시간 증분 (초) |
| `dt_max` | `float` | `0.1` | $> \text{dt\_init}$ | 최대 허용 시간 증분 (전체 해석 시간의 10%까지 자동 성장을 허용하여 해석 대폭 가속) |
| `dt_min` | `float` | `1e-5` | $> 0.0$ | 최소 시간 증분. 이 값 미만으로 축소 시 해석 Cutback 실패 처리 |
| `target_iters` | `int` | `6` (또는 `8`) | `3` ~ `15` | 목표 뉴튼 반복 횟수. Iteration $< \text{target\_iters}$ 시 `dt` 증가, 초과 시 `dt` 감소 |
| `theta_max_deg` | `float` | `90.0` | `0.0` ~ `90.0` | 사이드 플레이트당 목표 회전각 ($90^\circ \to$ 합계 $180^\circ$ 완접) |

---

### 2.3 비선형 수렴 판정 및 뉴튼-랩슨 제어 (Nonlinear Newton-Raphson Controls)

| 파라미터명 | 타입 | 기본값 | 설정 가능 옵션/범위 | 물리적 / 수치적 의미 및 추천 설정 |
| :--- | :--- | :--- | :--- | :--- |
| `tol` | `float` | `1e-2` | `1e-4` ~ `1e-1` | 변위 수정량 상대 비율 허용오차 ($\|\Delta \mathbf{u}\| / (\|\mathbf{u}\| + \epsilon)$) |
| `rtol` | `float` | `1e-4` | `1e-6` ~ `1e-2` | 잔차력 상대 비율 수렴 기준 ($\|\mathbf{R}\| / \|\mathbf{R}_0\|$) |
| `atol` | `float` | `1e-6` | `1e-8` ~ `1e-4` | 절대 변위 수렴 허용오차 (mm) |
| `max_iter` | `int` | `50` | `10` ~ `100` | 한 시간 증분당 허용되는 최대 뉴튼 반복 횟수 |
| `max_displacement_corr` | `float` | `0.03` | `0.01` ~ `0.10` (또는 `None`) | 뉴튼 1회 시도 시 최대 노드 이동량 제한값 (mm). 초박형 PSA 요소 반전 방지 |

---

### 2.4 재료별 FE 요소 정식화 및 왜곡 제어 (Element Formulations & Distortion Control)

| 파라미터명 | 타입 | 기본값 | 설정 가능 옵션/범위 | 물리적 / 수치적 의미 및 추천 설정 |
| :--- | :--- | :--- | :--- | :--- |
| `pet_element_type` | `str` | `"Q4_COROTATIONAL_EAS"` | `"Q4"`, `"Q4_EAS"`, `"Q4_COROTATIONAL_EAS"`, `"Q4_COROTATIONAL_SRI"` | PET 층 요소 유형. 고가로비($AR > 15$) 굽힘 시 전단 잠금(Shear Locking) 방지 |
| `psa_element_type` | `str` | `"Q4_VISCO_SIMO"` | `"Q4_VISCO_SIMO"`, `"Q4_HYBRID_EAS"`, `"Q4_SRI"` | PSA 점탄성 층 전용 요소. Simo-Hughes 대변형 미분 체적 분할 적용 |
| `glass_element_type` | `str` | `"Q4_COROTATIONAL_EAS"` | `"Q4_COROTATIONAL_EAS"`, `"Q4_HYBRID_EAS"` | $60\,\mu\text{m}$ Ultra-Thin Glass 요소. EAS 4-mode 결합으로 대변형 굽힘 정밀 표현 |
| `distortion_control` | `bool` | `True` | `True` / `False` | Abaqus 사양 요소 왜곡 제어 (*SECTION CONTROLS, DISTORTION CONTROL=YES) |
| `distortion_j_crit` | `float` | `0.20` | `0.05` ~ `0.50` | 임계 체적 변화비 threshold ($J = \det(\mathbf{F}) \ge J_{\text{crit}}$ 보장) |
| `ul_mode` | `bool` | `False` | `True` / `False` | Total Lagrangian (`False`) vs Updated Lagrangian (`True`) 프레임 전환 |

---

### 2.5 선형 방정식 솔버 설정 (PARDISO Direct Linear Solver Options)

| 파라미터명 | 타입 | 기본값 | 설정 가능 옵션/범위 | 물리적 / 수치적 의미 및 추천 설정 |
| :--- | :--- | :--- | :--- | :--- |
| `pardiso_mtype` | `str` | `"auto"` | `"auto"`, `"spd"`, `"indefinite"`, `"nonsymmetric"` | PARDISO 행렬 유형 지정. `"auto"` 설정 시 구속조건 유무에 따라 최적 선택 |
| `pardiso_phase_reuse` | `bool` | `True` | `True` / `False` | Phase 11 (Symbolic Factorization) 뉴튼 반복 간 재사용으로 연산 속도 2~3배 향상 |

---

### 2.6 구속 조건 및 경계 구동 설정 (Constraints & RBE2 Prescribed Rotation)

| 파라미터명 | 타입 | 기본값 | 설정 가능 옵션/범위 | 물리적 / 수치적 의미 및 추천 설정 |
| :--- | :--- | :--- | :--- | :--- |
| `theta_penalty_k` | `float` | `1e8` | `1e6` ~ `1e10` | RBE2 페널티 구동 강성 계수 |
| `use_kinematic_condensation` | `bool` | `True` | `True` / `False` | RBE2 Master-Slave 자유도 대수 축약 (KKT 시스템 조건수 $10^{16} \to 10^3$ 대폭 개선) |

---

### 2.7 JIT 컴파일 실행 백엔드 (JIT Acceleration Engine)

| 파라미터명 | 타입 | 기본값 | 설정 가능 옵션/범위 | 물리적 / 수치적 의미 및 추천 설정 |
| :--- | :--- | :--- | :--- | :--- |
| `elem_jit` | `str` | `"jax"` | `"jax"`, `"numba"` | 요소 어셈블리 JIT 백엔드. R&D/개발 시 `"jax"`, 고속 대규모 해석 시 `"numba"` (3.15배 속도 우수) |

---

## 3. 대표 해석 시나리오별 추천 설정 세트 (Recommended Preset Profiles)

### 시나리오 A: 초박형 14-Layer 디스플레이 풀 폴딩 ($90^\circ$/side 완접) - **[현재 표준]**
```python
solver_config = SolverTuningConfig(
    alpha=-0.15,                        # moderate-3 (Abaqus standard dissipation)
    pet_element_type="Q4_COROTATIONAL_EAS",
    glass_element_type="Q4_COROTATIONAL_EAS",
    psa_element_type="Q4_VISCO_SIMO",
    max_displacement_corr=0.03,         # Capping displacement step to prevent det(F)<=0
    distortion_control=True,
    distortion_j_crit=0.20,
    pardiso_mtype="auto",
    pardiso_phase_reuse=True,
)
```

### 시나리오 B: 관성 진동 없는 순수 준정적 이완 테스트 (Quasi-Static Relaxation)
```python
solver_config = SolverTuningConfig(
    alpha=0.0,
    # dynamic.py 실행 시 integration_mode="quasistatic" 지정
)
```

---
*본 문서는 솔버 핵심 파라미터 변경 시 지속적으로 업데이트됩니다.*
