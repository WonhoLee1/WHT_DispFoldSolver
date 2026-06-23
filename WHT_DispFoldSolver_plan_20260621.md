# WHT_DispFoldSolver — Implementation Plan (Rev 2)
**Status**: pending approval  
**Date**: 2026-06-21  
**Session**: DispFoldSolver

---

## 1. 요구사항 요약

| 항목 | 선택 |
|------|------|
| 형상 | 2D SOLID, Plane Strain |
| 변형 | Large Deformation (기하 비선형) |
| 재료 | Hyperelastic (Neo-Hookean / YEOH / Arruda-Boyce) + Viscoelastic (Prony + WLF TTS) + Elasto-plastic (table) |
| 시간 적분 | Implicit Dynamic (Newmark-β) |
| 요소 | Q4 (SRI) + T3 (F-bar) |
| 구속 | RBE2 힌지 (Lagrange Multiplier) + Tie constraint |
| 온도 | 상태 변수 (열전달 없음), WLF time shift 입력용 |
| 백엔드 | JAX (jit + grad) |
| 위치 | `d:\PythonCodeStudy\WHT_DispFoldSolver` (독립 패키지) |

---

## 2. 물리 시스템 개요

```
[Display Panel]
      ↕  Tie constraint (완전 결합)
[Support Plate (Left)]──[Hinge Axis]──[Support Plate (Right)]
                              ↑
                         RBE2 (회전만 허용, Lagrange Multiplier)
```

- **Display Panel ↔ Plate 상면**: Tie (모든 DOF 결합)
- **Plate ↔ Plate**: Hinge (RBE2, 힌지축 회전 DOF만 자유)
- **온도 T**: 각 요소/노드의 상태 변수 → Prony τ_i 보정에 사용 (WLF)

---

## 3. 핵심 기술 결정 (ADR)

### 3-A. 운동학: Total Lagrangian

F(변형기울기) → C = Fᵀ·F → W(C) → S = 2·∂W/∂C (JAX autograd)  
K_mat = 4·∂²W/∂C² (JAX Hessian)

### 3-B. Hyperelastic 재료 모델 (3종)

**MaterialModel ABC** 기반, `strain_energy_density(F, params)` 인터페이스 통일:

| 모델 | 에너지 함수 W | 파라미터 |
|------|-------------|---------|
| **Neo-Hookean** | μ/2·(I₁−3) − μ·ln J + λ/2·(ln J)² | μ, λ |
| **YEOH** | C₁₀(I₁−3) + C₂₀(I₁−3)² + C₃₀(I₁−3)³ | C10, C20, C30 |
| **Arruda-Boyce** | μ·Σₖ αₖ/λₘ²⁽ᵏ⁻¹⁾·(I₁ᵏ−3ᵏ) + bulk(J) | μ, λ_m (5항) |

모두 JAX-differentiable W(F) → S, K 자동 유도.

### 3-C. Viscoelastic + WLF Time-Temperature Superposition

**Prony series** (편차 응력):
```
S_eff(t) = g_∞·S_el + Σᵢ hᵢ(t)
```

**WLF 방정식** (온도 shift):
```
log₁₀(aT) = −C1·(T − T_ref) / (C2 + T − T_ref)
```
→ 유효 완화 시간: `τᵢ_eff = τᵢ · aT(T)`  
→ Δt_reduced = Δt / aT(T) 를 Prony 재귀에 사용

**온도 입력**: 노드별 또는 요소별 스칼라 상태 변수 T(x,t)  
(사용자 지정 테이블 또는 상수로 입력, 열전달 해석 없음)

**내부 변수 업데이트** (Simo 1987 재귀):
```python
beta_i = exp(−Δt_red / τᵢ)
hᵢ(t+Δt) = beta_i·hᵢ(t) + (1−beta_i)·ΔS_dev
```

### 3-D. Elasto-plastic (Table-based, J2 소성)

**항복 조건**: f = σ_vm − σ_y(ε_p) = 0  
**경화**: 사용자 입력 테이블 `[(ε_p, σ_y), ...]` → JAX piecewise linear interpolation  
**복귀 사상**: Radial Return Mapping (von Mises)

```python
@dataclass
class PlasticState:
    eps_p: float        # 등가 소성 변형률
    back_stress: ...    # (kinematic hardening 시)

def return_mapping(S_trial, eps_p_prev, hardening_table) -> (S, eps_p):
    """JAX-compatible radial return"""
```

**J2 + Large Deformation**: 중간 배치(intermediate configuration) F = Fᵉ·Fᵖ 곱 분해  
→ 초기 구현은 소변형 J2 + 대변형 운동학 분리 (실용적 근사)

### 3-E. RBE2 힌지 (Lagrange Multiplier)

**개념**: 마스터 노드 m에 슬레이브 노드 집합 {s}를 강체 결합.  
힌지축 (예: z축) 회전만 허용 → 나머지 DOF 구속.

**Plane Strain 2D에서 RBE2 힌지 (z축 회전)**:
- 슬레이브 노드 uˢ = uᵐ + ω × (xˢ − xᵐ)
- 2D: uˢₓ = uᵐₓ − ω·(yˢ − yᵐ), uˢᵧ = uᵐᵧ + ω·(xˢ − xᵐ)
- 구속 행렬 C: shape (2·n_slave, 3) per slave pair

**Lagrange Multiplier 조립**:
```
K_aug = [K_tan   Cᵀ]   f_aug = [f_int]
        [C       0 ]            [g    ]

C·u = g  (g=0 for homogeneous constraint)
```

**전역 DOF 확장**: ndof + n_lambda (λ: Lagrange multiplier 수)

### 3-F. Tie Constraint

**구현**: 면-면 Tie = RBE2의 특수 케이스 (모든 DOF 구속, ω=0)  
또는 Penalty 방식 (Tie는 수치 오차 허용 가능하므로 Penalty도 OK)

**전략**:
- RBE2와 동일한 Lagrange Multiplier 프레임워크 재사용
- `RBE2HingeConstraint`, `TieConstraint` 모두 `BaseConstraint` ABC 상속

---

## 4. 프로젝트 구조 (Rev 2)

```
d:\PythonCodeStudy\WHT_DispFoldSolver\
├── dispsolver/
│   ├── __init__.py
│   ├── mesh/
│   │   ├── mesh.py          # Mesh, Node, Element, NodeSet
│   │   └── mesh_io.py       # meshio 입출력
│   ├── element/
│   │   ├── q4.py            # Q4 Plane Strain (SRI)
│   │   └── t3.py            # T3 Plane Strain (F-bar)
│   ├── material/
│   │   ├── base.py          # MaterialModel ABC
│   │   ├── neohookean.py
│   │   ├── yeoh.py
│   │   ├── arruda_boyce.py
│   │   ├── viscoelastic.py  # Prony + WLF TTS wrapper
│   │   └── elastoplastic.py # Table-based J2, return mapping
│   ├── constraint/
│   │   ├── base.py          # BaseConstraint ABC → C matrix, g vector
│   │   ├── rbe2.py          # RBE2HingeConstraint (Lagrange Multiplier)
│   │   └── tie.py           # TieConstraint (Lagrange Multiplier)
│   ├── state/
│   │   └── field.py         # TemperatureField, InternalVarStore (pytree)
│   ├── solver/
│   │   ├── assembler.py     # K_tan + f_int + C 증강 조립
│   │   ├── newmark.py       # Newmark-β integrator
│   │   └── newton.py        # Newton-Raphson + line search
│   └── export/
│       └── vtu_exporter.py  # VTU/PVD (변위/응력/온도/소성변형률)
├── tests/
│   ├── test_kinematics.py
│   ├── test_hyperelastic.py  # NeoHookean, YEOH, ArrudaBoyce 각각
│   ├── test_viscoelastic.py  # Prony + WLF shift 검증
│   ├── test_plasticity.py    # Return mapping, hardening 곡선
│   ├── test_rbe2.py          # 힌지 구속 강체 회전 검증
│   ├── test_tie.py           # Tie 면-면 결합 검증
│   ├── test_static.py        # Cook's membrane
│   └── test_dynamic.py       # Newmark + 복합 재료
├── examples/
│   ├── ex01_static_patch.py
│   ├── ex02_cook_membrane.py
│   ├── ex03_display_fold.py  # 힌지 + Tie + Viscoelastic + 온도
│   └── ex04_wlf_relaxation.py
├── pyproject.toml
└── README.md
```

---

## 5. 구현 단계 (Phase)

### Phase 0: 스캐폴딩 (½일)
- 폴더/패키지 생성, pyproject.toml, git init
- **완료**: `python -c "import dispsolver"` 성공

### Phase 1: 메시 + 요소 운동학 (1일)
- `mesh.py`, `q4.py`, `t3.py`
- F, B 행렬, Jacobian, SRI quadrature 분리
- **완료**: Patch test (균일 응력 오차 < 1e-10)

### Phase 2: Hyperelastic 3종 (1일)
- `neohookean.py`, `yeoh.py`, `arruda_boyce.py`
- 공통 인터페이스: `W(F, params)` → JAX grad/hessian
- **완료**: 소변형 선형 탄성 극한 일치 (상대 오차 < 1e-4), 전 모델

### Phase 3: Viscoelastic + WLF (1일)
- `viscoelastic.py`: Prony Simo 재귀 + WLF aT(T) 적용
- `state/field.py`: TemperatureField (노드/요소 할당)
- **완료**: 응력 완화 곡선 L2 < 1e-6, WLF shift 검증 (2개 온도)

### Phase 4: Elasto-plastic (1일)
- `elastoplastic.py`: J2 radial return + 테이블 경화
- JAX `jnp.interp` 기반 piecewise linear 항복 응력
- **완료**: 단축 인장 응력-변형률 테이블 재현, 언로딩 탄성 복원

### Phase 5: 구속 (RBE2 + Tie) (1½일)
- `constraint/rbe2.py`: C 행렬 생성, K_aug 증강
- `constraint/tie.py`: 면-면 노드 쌍 탐색 + C 조립
- **완료**: 힌지 강체 회전 → 슬레이브 노드 오차 < 1e-10, Tie gap < 1e-10

### Phase 6: 전역 조립 + 정적 해석 (1일)
- `assembler.py`: K_tan + C_constraint → K_aug (JAX sparse BCOO)
- Newton-Raphson + line search
- **완료**: Cook's membrane ±1% 문헌값

### Phase 7: Newmark Implicit Dynamic (1½일)
- `newmark.py`: Newmark-β + 복합 재료 (내부 변수 시간 적분 포함)
- K_eff = K_tan + γ/(β·Δt)·C_damp + 1/(β·Δt²)·M
- **완료**: SDOF 주파수 오차 < 0.1%, 점탄성 에너지 단조 감소

### Phase 8: 통합 예제 + Export (½일)
- `ex03_display_fold.py`: Plate + Hinge + Display + WLF 온도 변화
- VTU export: 변위, Cauchy 응력, Von Mises, 소성 변형률, 온도, h_i norm
- **완료**: ParaView에서 시각화 성공

---

## 6. 위험 및 완화

| 위험 | 영향 | 완화 |
|------|------|------|
| Q4 체적 잠금 | 비압축 재료에서 완전 오류 | SRI 필수, patch test 자동화 |
| JAX sparse BCOO 조립 | 대형 모델에서 느림 | 초기 dense fallback, vmap 배치 최적화 |
| Prony + WLF + JAX JIT | 내부 변수 고정 크기 필요 | Prony 항수 M을 컴파일 시 고정 |
| Arruda-Boyce J→1 수치 불안정 | nan 발산 | bulk penalty 항으로 압축성 처리 |
| Lagrange K_aug 양정치성 소실 | 직접 솔버 필요 | `scipy.sparse.linalg.spsolve` (UMFPACK) fallback |
| J2 + Large Deformation | 중간 배치 분해 복잡 | 초기엔 소변형 J2 + 대변형 운동학 분리 구현 |
| WLF 극단 온도 | aT 오버플로 | log-domain 연산, clamp |

---

## 7. 수용 기준 (Acceptance Criteria)

| # | 기준 | 검증 |
|---|------|------|
| AC-1 | `import dispsolver` 성공 | `python -c "import dispsolver"` |
| AC-2 | Patch test 통과 (Q4, T3) | `test_kinematics.py`, 오차 < 1e-10 |
| AC-3 | Hyperelastic 3종 소변형 극한 일치 | `test_hyperelastic.py`, 상대 오차 < 1e-4 |
| AC-4 | Prony 응력 완화 해석해 일치 | `test_viscoelastic.py`, L2 < 1e-6 |
| AC-5 | WLF shift: 2개 온도에서 유효 완화 시간 검증 | `test_viscoelastic.py::test_wlf_shift` |
| AC-6 | 테이블 경화 재현 + 언로딩 탄성 | `test_plasticity.py` |
| AC-7 | RBE2 힌지 강체 회전 (슬레이브 오차 < 1e-10) | `test_rbe2.py` |
| AC-8 | Tie gap < 1e-10 | `test_tie.py` |
| AC-9 | Cook's membrane ±1% | `test_static.py` |
| AC-10 | SDOF 주파수 오차 < 0.1% | `test_dynamic.py` |
| AC-11 | 통합 예제 (Fold + WLF) VTU 출력 | `ex03_display_fold.py` |

---

## 8. 의존성

```toml
[project]
dependencies = [
  "jax[cpu]>=0.4.25",
  "numpy>=1.26",
  "scipy>=1.12",      # spsolve (K_aug 직접 풀이)
  "meshio>=5.3",
]
[project.optional-dependencies]
test = ["pytest>=8.0"]
vis  = ["pyvista>=0.43"]
```

---

## 9. 구현 가능성 평가

**결론: 구현 가능. 예상 소요 기간 8~10일 (집중 작업 시)**

| 항목 | 난이도 | 비고 |
|------|--------|------|
| Q4/T3 운동학 + SRI | ★★☆ | 교과서 공식 |
| Hyperelastic 3종 + JAX grad | ★★☆ | W→S→K 자동 유도 |
| Prony Viscoelastic | ★★★ | 내부 변수 pytree 관리 |
| WLF TTS | ★★☆ | aT(T) log-domain 계산 |
| Table-based J2 소성 | ★★★ | Return mapping + JAX interp |
| RBE2 Lagrange Multiplier | ★★★ | K_aug 비대칭 → spsolve 필요 |
| Tie Constraint | ★★☆ | RBE2 프레임 재사용 |
| Newmark + 복합 재료 | ★★★ | 내부 변수 시간 적분 연동 |

*Plan saved: `.omc/plans/WHT_DispFoldSolver_plan.md` (Rev 2)*  
*Status: **pending approval***
