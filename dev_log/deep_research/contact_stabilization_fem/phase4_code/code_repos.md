# Phase 4: Code & Tools — 오픈소스 생태계 조사

**리서치 주제**: Penalty-based contact stabilization in implicit FEM  
**날짜**: 2026-09-30  
**작성자**: Deep Research Skill (Phase 4/6)

---

## 발견된 오픈소스 저장소

### 1. conmech (KOS-UJ/conmech)
- **URL**: https://github.com/KOS-UJ/conmech
- **언어**: Python
- **특징**: 2D/3D FEM 접촉 역학 전용 패키지. 정적/준정적/동적 지원, 마찰 포함.
- **관련성**: WHT 코드와 유사한 구조. **Active Set** + penalty 구현 참고 가능.
- **핵심 참고 파일**: `conmech/solvers/contact/contact_methods.py`

### 2. Wells-Group/asimov-contact (DOLFINx 기반)
- **URL**: https://github.com/Wells-Group/asimov-contact
- **언어**: Python (DOLFINx / FEniCSx)
- **특징**: Nitsche 방법 + 대규모 병렬 접촉. TU Munich 그룹의 최신 구현.
- **관련성**: Nitsche 접촉의 레퍼런스 구현. 우리 코드에 바로 적용 어려우나 이론 검증에 유용.
- **핵심 파일**: `python/asimov_contact/contact_constraints.py`

### 3. st-tech/ppf-contact-solver
- **URL**: https://github.com/st-tech/ppf-contact-solver
- **언어**: Python + CUDA (GPU 가속)
- **특징**: IPC(Incremental Potential Contact) 기반 GPU 접촉 솔버.
- **관련성**: 대규모 접촉 시 참고. 현재 프로젝트에 직접 적용 어려움.

### 4. PyFEM (jjcremmers/PyFEM)
- **URL**: https://github.com/jjcremmers/PyFEM
- **언어**: Python
- **특징**: 교육용 비선형 FEM 프레임워크. Newton-Raphson + 접촉 모듈.
- **관련성**: 접촉 알고리즘의 간결한 레퍼런스 구현.

### 5. MOOSE Framework (idaholab/moose) — PDAS 구현
- **URL**: https://github.com/idaholab/moose
- **언어**: C++
- **특징**: INL의 대형 FEM 프레임워크. `contact/` 서브모듈에 PDAS + AL 구현.
- **관련성**: PDAS 알고리즘의 산업급 레퍼런스. Abaqus 수준.
- **핵심 참고**: `modules/contact/src/constraints/MechanicalContactConstraint.C`

---

## 현재 코드와의 직접 비교 — conmech 구현 검토

`conmech`의 penalty freeze 관련 코드 패턴:
```python
# conmech 방식 (단방향 압축 전용)
def contact_force(penetration, k):
    return max(0.0, k * penetration)  # ← Unilateral! 양방향 아님

# 현재 WHT 코드 frozen 분기 (버그)
f_mag = self.k_contact * penetration  # ← Bilateral! 버그
```

→ **conmech이 올바른 방향**: `max(0.0, ...)` 적용. WHT Fix A의 외부 검증.

---

## 벤치마크 참고 데이터셋

1. **Hertz Contact Analytical Solution** (Johnson 1985 기반):
   ```
   입력: E*=200 GPa (reduced modulus), R=20 mm (roller radius), F=? N
   출력:
     접촉 반폭: a = (4FR/πE*)^{1/2}
     최대 압력: p_0 = 2F/(πa)
     침투량: δ = a²/R
   ```
   → 우리 벤치마크 INDENT Step의 이론 검증 기준

2. **Coulomb Sliding Friction** (SLIDE Step):
   ```
   이론: RF_x = μ × RF_y = 0.20 × RF_y
   허용 오차: ±2% (일반 마찰 수치 해석 정확도)
   ```

---

✅ **Phase 4 완료**. Output: `code_repos.md` (5개 저장소 + 벤치마크 기준).
Phase 5: Synthesis로 진행합니다.
