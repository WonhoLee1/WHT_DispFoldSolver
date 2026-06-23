# Abaqus 호환 솔버 성장 로드맵 (2026-06-24 기준)

## TL;DR

> **현재**: FEM 코어 완성 (105 tests PASS, ex03 180° 폴딩 완주)
> **목표**: `read_abaqus_input("model.inp")` 한 줄로 Abaqus 해석을 그대로 실행
> **전략**: 파서(A) → P0 솔버(B) → 접촉(C) → P1 솔버(D) → 고급(E)의 5단계
> **예상**: 전체 완료까지 4~6개월 (단계별 병행 가능)

---

## 1. 현재 진단: 어디까지 왔나

### ✅ 이미 구현된 것 (이미지 → Abaqus 키워드 대응)

| Abaqus 키워드 | dispsolver 구현 | 파일 |
|:---|---|:---|
| `*NODE`, `*ELEMENT` | `Mesh.add_node()`, QUAD4/TRIA3 | `mesh.py` |
| `*ELASTIC` | `NeoHookean(E, nu)` | `neohookean.py` |
| `*HYPERELASTIC, NEO HOOKE` | `NeoHookean()` | `neohookean.py` |
| `*HYPERELASTIC, YEOH` | `Yeoh(C10, C20, C30)` | `yeoh.py` |
| `*HYPERELASTIC, ARRUDA-BOYCE` | `ArrudaBoyce(mu, lambda_m, K)` | `arruda_boyce.py` |
| `*VISCOELASTIC, TIME=PRONY` | `Viscoelastic(base, g_i, tau_i)` + WLF | `viscoelastic.py` |
| `*DENSITY` | `DynamicSolver(rho=...)` | `dynamic.py` |
| `*BOUNDARY` | `set_prescribed_dofs()` | `dynamic.py` |
| `*CLOAD` | `apply_load()` | `dynamic.py` |
| `*AMPLITUDE` | `Amplitude` (linear/smooth/step) | `load/amplitude.py` |
| `*DYNAMIC` (adaptive dt) | `solve()` + `solve_step()` + cutback | `dynamic.py` |
| `*TIE` | `TieConstraint` | `constraint/tie.py` |
| `*MPC` (RBE2) | `RBE2HingeConstraint` | `constraint/rbe2.py` |
| `*CONTACT` (self-contact) | `PenaltyContactConstraint` (spatial hash) | `constraint/contact_jax.py` |
| `*SOLID SECTION` (두께) | pid별 `section_thickness` | `dynamic.py` |
| — (VTK export) | `VTKHDFExporter` / `VTUExporter` | `export/` |
| — (강건화) | 퇴화요소 정규화, NaN guard, 배치검증 | `material/*.py` |
| — (멀티코어) | PARDISO 4-core + 반복정련 | `dynamic.py` |

### ❌ 아직 없는 것 (구현 필요)

| 구분 | 기능 | 비고 |
|:---:|:---|---|
| **🔧 파서** | `dispsolver/io/` 전량 | 가장 큰 단일 gap |
| **🔧 파서** | Part/Assembly flattening | instance 변환 |
| **🔧 파서** | `*SURFACE` 정의 | auto exterior detection |
| **🔧 하중** | 분포하중 (DLOAD/DSLOAD) | surface traction |
| **🔧 요소** | Q1P0 하이브리드 배치화 | 현재 순차 loop |
| **🔧 솔버** | Static mode | 관성항 제거 |
| **🔧 솔버** | HHT-α 적분기 | Newmark 외 option |
| **🔧 재료** | J2 소성 비선형 경화 테이블 | 현재 선형 H만 |
| **🔧 접촉** | `*CONTACT PAIR` (NTS) | 현재 self-contact만 |
| **🔧 접촉** | 마찰 (Coulomb) | 현재 frictionless |
| **🔧 접촉** | 접촉면 자동탐지 | *SURFACE TYPE=ELEMENT |
| **📋 고급** | `*INCLUDE` | 다중 파일 |
| **📋 고급** | 다중 `*STEP` | 순차 해석 |
| **📋 고급** | 온도耦合 | 초기조건 + 열팽창 |
| **📋 고급** | 고유치/좌굴 | `*FREQUENCY`, `*BUCKLE` stubs |

---

## 2. 전체 로드맵 (5개 Phase)

```
지금: FEM 코어 완성 (105 tests)
  │
  ├─ Phase A: Parser (8주) ← 가장 큼
  │   ├─ A1: Lexer + AbaqusModel dataclasses [2주]
  │   ├─ A2: Parser 20+ keyword handlers [3주]
  │   ├─ A3: ModelBuilder + read_abaqus_input() [2주]
  │   └─ A4: Part/Assembly flattening [1주]
  │
  ├─ Phase B: P0 Solver 기능 (4주, A와 병행 가능)
  │   ├─ B1: Surface traction (DLOAD) [2주]
  │   ├─ B2: J2 소성 비선형 경화 테이블 [1주]
  │   └─ B3: Parser 연동 테스트 + 픽스처 [1주]
  │
  ├─ Phase C: Contact 체계화 (6주, A 완료 후)
  │   ├─ C1: *SURFACE 정의 + auto exterior [2주]
  │   ├─ C2: *CONTACT PAIR parser + ContactSurface [1주]
  │   ├─ C3: Node-to-Surface penalty contact [2주]
  │   └─ C4: Coulomb friction [1주]
  │
  ├─ Phase D: P1 Solver 기능 (4주, C와 병행 가능)
  │   ├─ D1: Static mode [1주]
  │   ├─ D2: HHT-α integrator [1주]
  │   └─ D3: Q1P0 하이브리드 배치화 [2주]
  │
  └─ Phase E: 고급 기능 (8주, 선택)
      ├─ E1: *INCLUDE + 다중 STEP [2주]
      ├─ E2: 온도 초기조건 + 열팽창 [2주]
      ├─ E3: *FREQUENCY / *BUCKLE stubs [2주]
      └─ E4: 공학 지표 추출 + 결과 가시화 [2주]
```

---

## 3. Phase A: Parser (Priority: P0)

가장 크고 가장 중요한 gap. Solver 기능은 대부분 갖춰져 있고, **입구(parser)가 없어서 Abaqus 입력을 못 읽는** 상황.

### A1: Lexer + AbaqusModel (2주)

**파일**: `dispsolver/io/abaqus_lexer.py`, `dispsolver/io/abaqus_model.py`

**Lexer**:
- `tokenize(filepath) → List[AbaqusKeywordBlock]`
- `*` 키워드 식별, `**` 주석 제거, `,` 파라미터 파싱
- utf-8/cp949 fallback

**AbaqusModel dataclasses**:
```python
AbaqusNode, AbaqusElement, AbaqusMaterial
AbaqusBoundary, AbaqusLoad, AbaqusStep
AbaqusAmplitude, AbaqusDload
AbaqusSurface, AbaqusContactPair
AbaqusMpc, AbaqusTie
AbaqusPart, AbaqusInstance, AbaqusAssembly
AbaqusModel  # container for all above
```

### A2: Parser 20+ keyword handlers (3주)

**파일**: `dispsolver/io/abaqus_parser.py`

**핸들러 목록** (23개):
- 노드/요소: `*NODE`, `*ELEMENT` (CPE4/CPE3/CPE4R, C3D8→에러)
- 세트: `*NSET`, `*NSET, GENERATE`, `*ELSET`
- 재료: `*MATERIAL`, `*ELASTIC`, `*PLASTIC`, `*HYPERELASTIC` (3종), `*VISCOELASTIC`, `*TRS`, `*DENSITY`
- 단면: `*SOLID SECTION`
- 하중/경계: `*BOUNDARY`, `*CLOAD`, `*DLOAD`, `*AMPLITUDE`
- 접촉: `*SURFACE`, `*CONTACT PAIR`, `*SURFACE INTERACTION`
- 구속: `*TIE`, `*MPC`
- 스텝: `*STEP`, `*DYNAMIC`, `*STATIC`, `*END STEP`
- 초기조건: `*INITIAL CONDITIONS` (TEMPERATURE)
- Part/Assembly: `*PART`, `*END PART`, `*ASSEMBLY`, `*INSTANCE`, `*TRANSLATION`, `*ROTATION`
- 미지원: 경고 후 skip (3D/SHELL/BEAM/FREQUENCY/BUCKLE)

### A3: ModelBuilder + read_abaqus_input() (2주)

**파일**: `dispsolver/io/model_builder.py`, `dispsolver/io/__init__.py`

**변환 매핑**:
| AbaqusModel 필드 | dispsolver 객체 |
|---|---|
| `nodes` | `Mesh.add_node()` |
| `elements` | `Mesh.add_element(QUAD4/TRIA3)` |
| `nsets`/`elsets` | `Mesh.add_nodeset()`/`add_elementset()` |
| `materials` | `NeoHookean()` / `Yeoh()` / `J2Plasticity()` / `Viscoelastic()` |
| `boundaries` | `solver.set_prescribed_dofs()` |
| `loads` | `solver.apply_load()` |
| `amplitudes` | `Amplitude()` → solver 등록 |
| `dloads` | `SurfaceTraction()` → solver 등록 |
| `surfaces`/`contact_pairs` | `ContactSurface` → `ContactPair` → solver 등록 |
| `mpcs`/`ties` | `RBE2HingeConstraint` / `TieConstraint` |
| `step.dynamic_params` | `solver.solve(dt_init, t_total, dt_min, dt_max)` |

**공개 API**:
```python
def read_abaqus_input(path: str) -> Tuple[Mesh, Materials, Constraints, Config, Amplitudes, Contacts]:
    ...
```

### A4: Part/Assembly flattening (1주)

**파일**: `dispsolver/io/model_builder.py` (내부 메서드)

**변환 규칙**:
```
*INSTANCE, NAME=I1, PART=P1
  *TRANSLATION, dx, dy
→ I1_n1 = P1_n1 + (dx, dy)
→ node ID: "I1_1", element ID: "I1_e1"
→ set name: "I1_NSET1"

*INSTANCE, NAME=I2, PART=P1
  *ROTATION, angle, 0, 0, 1
→ I2_n1 = R(θ) · P1_n1
```

**Edge case**: instance가 같은 part를 공유 → 각각 독립 복사

---

## 4. Phase B: P0 Solver 기능 확장 (Priority: P0)

### B1: Surface traction / 분포하중 (2주)

**파일**: `dispsolver/load/traction.py` (신규)

**필요 기능**:
- Q4/T3 요소 면 번호 체계 (Abaqus convention)
- 일정 압력 p → consistent nodal force
- `f = ∫ N^T · p · n · dΓ` (초기 형상 기준, small-displacement)
- Solver 연동: `_compute_external_force()`에서 traction forces 통합

**검증**: 단일 Q4 요소, face 3 압력 → sum(nodal forces) = p × face_length

### B2: J2 소성 비선형 경화 테이블 (1주)

**파일**: `dispsolver/material/plastic.py` (기존 수정)

**변경사항**:
- `J2Plasticity.__init__(..., hardening_table=None)`
- `hardening_table = np.array([(eqps, sigma_y), ...])`
- `_yield_stress(eqps)`: table → `np.interp`, None → 기존 선형
- `*PLASTIC` 다중점 매핑: `sigma_y0 = table[0,1]`, `H` 생략

### B3: Parser 연동 테스트 + 픽스처 (1주)

**파일**: `tests/test_abaqus_parser.py`, `tests/fixtures/*.inp`

**픽스처** (5개):
1. `simple_quad.inp` — 단일 CPE4 + 탄성 + BOUNDARY + CLOAD
2. `hyperelastic_patch.inp` — CPE4 4개 + HYPERELASTIC NEO HOOKE
3. `viscoelastic_relax.inp` — CPE4 + ELASTIC + VISCOELASTIC + TRS
4. `two_part_assembly.inp` — Part/Assembly + INSTANCE 2개 (translation + rotation)
5. `contact_pair.inp` — CPE4 2블록 + *CONTACT PAIR

---

## 5. Phase C: 접촉 체계화 (Priority: P0)

### C1: *SURFACE 정의 + auto exterior detection (2주)

**파일**: `dispsolver/contact/contact_surface.py`

**auto exterior detection**:
```python
def auto_detect_exterior(mesh, instance_or_set_name) -> ContactSurface:
    elements = mesh.get_elements(instance_or_set_name)
    edge_counts = {}
    for elem in elements:
        for edge in elem.edges:  # Q4: 4, T3: 3
            key = tuple(sorted(edge.node_ids))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    exterior = [(n1,n2) for (n1,n2),c in edge_counts.items() if c == 1]
    # 각 exterior edge의 법선 결정
    normals = [compute_outward_normal(elem_centroid, edge) for edge in exterior]
    return ContactSurface(name, exterior, normals)
```

### C2: *CONTACT PAIR parser + ContactSurface (1주)

**파일**: `dispsolver/io/abaqus_parser.py` (파서), `dispsolver/contact/contact_pair.py` (데이터)

**파싱 대상**:
```
*SURFACE, NAME=SLAVE_SURF, TYPE=ELEMENT
PART-1-1, SPOS

*SURFACE INTERACTION, NAME=SOFT
1e6, 0.3

*CONTACT PAIR, INTERACTION=SOFT
SLAVE_SURF, MASTER_SURF
```

### C3: Node-to-Surface penalty contact (2주)

**파일**: `dispsolver/contact/nts_contact.py` (신규)

**현재 상태**: `contact_jax.py`에 node-to-node self-contact만 있음 (spatial hash + JAX penalty)

**필요한 확장**:
1. **Master segment 정의**: exterior edges as linear segments
2. **Closest-point projection**: slave node → master segment
3. **Gap function**: signed distance, penetration detection
4. **Soft penalty**: `tanh()` regularization (C1 continuous)
5. **기존 self-contact와 통합**: PenaltyContactConstraint 확장 or 새 클래스

**Implementation plan**:
- 기존 `contact_jax.py`의 spatial hash 재사용 (이미 검증됨)
- NTS contact logic은 JAX 없이 numpy로 (closest-point projection은 간단한 2x2)
- 기존 self-contact와 병렬共存

### C4: Coulomb friction (1주)

**파일**: `dispsolver/contact/nts_contact.py` (확장)

- Stick/slip state machine
- Elastic stick regularization (penalty)
- Friction force = μ × |f_N| (Coulomb)
- Tangent stiffness coupling

---

## 6. Phase D: P1 Solver 기능 (Priority: P1)

### D1: Static mode (1주)

**파일**: `dispsolver/solver/dynamic.py`

**변경사항**:
```python
class DynamicSolver:
    def __init__(self, ..., static_mode=False):
        self.static_mode = static_mode
    
    def solve_step(self, dt):
        if self.static_mode:
            K_eff = K_T
            R_u = f_ext - f_int - f_contact
        else:
            K_eff = K_T + M * inv_beta_dt2
            R_u = f_ext - M @ a_k - f_int - f_contact
```

### D2: HHT-α integrator (1주)

**파일**: `dispsolver/solver/dynamic.py`

**변경사항**:
```python
class DynamicSolver:
    def __init__(self, ..., alpha=0.0):
        self.alpha = alpha  # 0=Newmark, -0.05=HHT-α default
    
    def _compute_effective_residual(self):
        if self.alpha != 0:
            R_u = f_ext - M@a_k - (1+α)*f_int_k + α*f_int_n
            K_eff = (1+α)*K_T + M*inv_beta_dt2
```

### D3: Q1P0 하이브리드 배치화 (2주)

**파일**: `dispsolver/element/q4_visco_hybrid.py` (기존)

**현재**: 스텝당 수초 순차 루프

**목표**: 모든 요소를 배치(array) 연산으로 변환
- 응력/접선을 (n_elem, ...) shape으로 벡터화
- `visco_state` 업데이트도 배치

---

## 7. Phase E: 고급 기능 (Priority: P2, 선택)

| 항목 | 파일 | 내용 | 예상 |
|:---|:---|---:|
| **E1** `*INCLUDE` | `abaqus_lexer.py` | 재귀 파일 포함, 중복 방지 | 1주 |
| **E1** 다중 `*STEP` | `model_builder.py` | Step별 solver config, 순차 실행 | 1주 |
| **E2** 온도 초기조건 | `parser` + `state/field.py` | `*INITIAL CONDITIONS, TYPE=TEMPERATURE` → `TemperatureField` | 2주 |
| **E2** 열팽창 | `material/base.py` | 열변형률 `ε_th = α·ΔT` | 2주 |
| **E3** `*FREQUENCY` | `solver/eigen.py` (신규) | Subspace iteration, Lanczos | 2주 |
| **E3** `*BUCKLE` | `solver/buckle.py` (신규) | 선형 좌굴: (K + λ·K_G)φ = 0 | 2주 |
| **E4** 공학 지표 | `post/` (신규) | 최대응력/소성변형/반력모멘트 추출 | 1주 |
| **E4** 결과 가시화 | `export/vtkhdf_exporter.py` | 스텝별 프레임 + 필드 출력 | 1주 |

---

## 8. 병행 실행 전략

```
Week:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
      ┌─────────────────────────────────────────────────────────────────────────
A1    │████████  (Lexer + dataclasses)
A2    │          ████████████  (Parser)          ← Critical Path
A3    │                    ████████  (Builder)   
A4    │                          ████  (Part/Assembly)
      │
B1    │████████████████████████  (Surface traction ← A3 필요)
B2    │████████████  (비선형 경화 ← 독립)
B3    │                    ██████  (Parser tests ← A3 필요)
      │
C1    │████████████████████████████████  (auto exterior ← A4 필요)
C2    │████████████████████████████  (Contact parser ← A1 필요)
C3    │██████████████████████████████████████  (NTS ← C1+C2 필요)
C4    │████████████████████  (마찰 ← C3 필요)
      │
D1    │████████  (Static mode ← 독립)
D2    │████████  (HHT-α ← 독립)
D3    │████████████████  (Hybrid 배치화 ← 독립)
      │
E1    │                                        ████████  (INCLUDE)
E2    │                                        ████████████  (온도)
E3    │                                        ████████  (고유치)
E4    │                                        ████████  (가시화)
```

### 병행 가능한 작업들
1. **A1→A4 (Parser)**: Critical path — 순차 의존. 다른 모든 phase가 A2/Parser 완료를 기다림
2. **B2 (비선형 경화)**: 완전 독립
3. **D1 (Static mode)**: 완전 독립
4. **D2 (HHT-α)**: 완전 독립
5. **D3 (Hybrid 배치화)**: 완전 독립

### A2 (Parser) 완료 후 병행 가능
6. **B1 (Surface traction)**: A3의 Builder 필요
7. **B3 (Parser tests)**: A3 필요
8. **C1 (auto exterior)**: A4 (Part/Assembly) 필요
9. **C2 (Contact parser)**: A1 (dataclasses) 필요

---

## 9. Abaqus 키워드 전체 대응 현황 (최종 목표)

| 키워드 | Phase | 파서 | 솔버 | 테스트 | 상태 |
|:---|---|:---:|:---:|:---:|:---:|
| `*NODE` | A | ✅ | ✅ | ✅ | **DONE** |
| `*ELEMENT, TYPE=CPE4/CPE3` | A | ✅ | ✅ | ✅ | **DONE** |
| `*NSET` / `*ELSET` | A | ✅ | ✅ | ✅ | **DONE** |
| `*SOLID SECTION` | A | 🔧 | ✅ | ✅ | 파서만 |
| `*ELASTIC` | A | ✅ | ✅ | ✅ | **DONE** |
| `*HYPERELASTIC` (3종) | A | ✅ | ✅ | ✅ | **DONE** |
| `*VISCOELASTIC, TIME=PRONY` | A | ✅ | ✅ | ✅ | **DONE** |
| `*TRS, DEFINITION=WLF` | A | ✅ | ✅ | ✅ | **DONE** |
| `*DENSITY` | A | ✅ | ✅ | ✅ | **DONE** |
| `*BOUNDARY` | A | ✅ | ✅ | ✅ | **DONE** |
| `*CLOAD` | A | ✅ | ✅ | ✅ | **DONE** |
| `*AMPLITUDE` | A | ✅ | ✅ | ✅ | **DONE** |
| `*DYNAMIC` (adaptive dt) | A | ✅ | ✅ | ✅ | **DONE** |
| `*TIE` | A | ✅ | ✅ | ✅ | **DONE** |
| `*MPC` (RBE2) | A | ✅ | ✅ | ✅ | **DONE** |
| `*SURFACE, TYPE=ELEMENT` | C | 🔧 | 🔧 | 🔧 | Phase C |
| `*CONTACT PAIR` | C | 🔧 | 🔧 | 🔧 | Phase C |
| `*SURFACE INTERACTION` | C | 🔧 | 🔧 | 🔧 | Phase C |
| `*PART` / `*INSTANCE` | A | 🔧 | — | 🔧 | Phase A |
| `*DLOAD` / `*DSLOAD` | B | 🔧 | 🔧 | 🔧 | Phase B |
| `*PLASTIC` (비선형) | B | ✅ | 🔧 | 🔧 | **DONE** |
| `*STATIC` | D | 🔧 | 🔧 | 🔧 | Phase D |
| `*INITIAL CONDITIONS, TYPE=TEMPERATURE` | E | 🔧 | 🔧 | 🔧 | Phase E |
| `*INCLUDE` | E | 🔧 | — | 🔧 | Phase E |
| `*FREQUENCY` | E | 🔧 | 🔧 | 🔧 | Phase E |
| `*BUCKLE` | E | 🔧 | 🔧 | 🔧 | Phase E |
| C3D8 등 3D 요소 | — | ❌ | ❌ | ❌ | NotImplementedError |

---

## 10. 예상 일정 요약

| Phase | 내용 | 작업량 | 병행 가능? | 낙관 | 현실 |
|:---|:---|---|:---:|:---:|:---:|
| **A** | Parser (Lexer→Parser→Builder) | 8주 | 부분 | 6주 | 8주 |
| **B** | P0 (traction + 경화 + test) | 4주 | A와 병행 | 3주 | 4주 |
| **C** | Contact (surface + pair + NTS + friction) | 6주 | A 완료 후 | 4주 | 6주 |
| **D** | P1 (static + HHT-α + hybrid 배치) | 4주 | 병행 가능 | 3주 | 4주 |
| **E** | 고급 (INCLUDE + 온도 + 고유치) | 8주 | D 완료 후 | 6주 | 8주 |
| **합계** | | **30주** | | **18주** | **24주 (6개월)** |

---

## 11. 핵심 아키텍처 원칙

### 1. Parser ↔ Solver 분리
```
.inp → [Lexer] → KeywordBlocks → [Parser] → AbaqusModel → [Builder] → dispsolver objects
```
- **Lexer/Parser**: Abaqus 형식에만 의존 (솔버 독립)
- **Builder**: `dispsolver` API에만 의존
- **중간 데이터모델 (AbaqusModel)**: 향후 다른 solver로의 매핑에도 재사용 가능

### 2. 기존 기능 우선 활용
- Amplitude, adaptive dt, contact spatial hash 등 **이미 있는 것은 최대한 재사용**
- Parser는 기존 API를 호출만 하고, 새 로직은 최소화
- `read_abaqus_input("file.inp")` → 기존 solver 객체 반환

### 3. 점진적 확장
- 미지원 키워드 → `warnings.warn()` 후 스킵 (절대 크래시 ❌)
- 3D 요소 → 명시적 `NotImplementedError`
- 새 기능 추가 시 기존 105개 회귀 테스트 유지

### 4. 품질 기준
- 모든 태스크: 파서 + 솔버 + 테스트의 3박자
- 모든 PR: 기존 회귀 테스트 전부 통과
- 새 기능: 최소 2개 이상의 단위 테스트
