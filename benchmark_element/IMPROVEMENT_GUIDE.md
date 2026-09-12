# 3D Solid Elements Diagnostic & Renovation Guide for Agents

> **대상 독자**: WHT_DispFoldSolver의 3D 유한요소(`dispsolver/element3d/`)를 수정, 디버깅, 최적화하려는 **AI 서브에이전트 및 FEA 코어 개발자**  
> **기준 일자**: 2026-09-13  
> **위치**: `benchmark_element/IMPROVEMENT_GUIDE.md`

---

## 1. 개요 및 목적 (Mission & Architecture)

이 문서는 `benchmark_element/` 폴더 내의 벤치마크 테스트 결과(Rank, Tangent, Patch, Bending, Volumetric)를 바탕으로, **어떤 요소에 어떤 결함이 존재하며, 어느 소스 코드의 어떤 수식/알고리즘을 수정해야 하는지**를 명확하고 구체적인 처방전(Actionable Recipes) 형태로 제공하는 기술 가이드북입니다.

본 프로젝트의 불변 원칙:
> **"증명된 숫자가 없는 주장은 거짓이다 (AGENTS.md §4.14-§4.18)."**  
> 어떤 요소를 '개선했다'거나 '수정했다'고 선언하기 전에, 반드시 본 폴더의 벤치마크 스크립트와 회귀 테스트(`pytest tests/test_3d_mechanics_benchmarks.py`)를 실행하여 변경 전/후의 실측 수치로 입증해야 합니다.

---

## 2. 벤치마크 체계 및 판정 지표 해석법

`benchmark_element/`에는 두 개의 상호 보완적인 벤치마크 스크립트가 존재합니다:

| 스크립트 | 측정 대상 | 소요 시간 | 핵심 합격 기준 |
|:---|:---|:---:|:---|
| **`benchmark_3d_elements.py`** | **수학적 건전성 & 연산 성능** | 수 초 | - 6개 강체 모드 분리 및 유효 랭크 보존 (Rank 18, 12, 6, 24)<br>- FD 대비 접선 오차 < 1e-4<br>- Numba 병렬 조립 속도 |
| **`benchmark_3d_mechanics.py`** | **역학적 정확도 & 잠김 현상** | 1~2 분 | - Irons 패치 테스트 오차 < 1e-6 (기계 정밀도 권장)<br>- 공식 Abaqus VM 패치 테스트 오차 < 1e-6<br>- 외팔보 굽힘 처짐비 > 0.85 (Locking-Free)<br>- 비압축성($\nu=0.49999$) 체적 잠김 없음 |

### 결과표 판정 기준 요약
1. **패치 테스트 (Tension / Compression / Shear XY / Shear YZ / Abaqus VM)**:
   - **무조건 전원 ✅ (오차 < 1e-6) 필수**. 패치 테스트 실패는 선형 변위장조차 만족하지 못한다는 의미이므로, 굽힘이나 체적 잠김 수치는 신뢰할 수 없습니다.
2. **굽힘 처짐비 (Bending Ratio, $L/h=10$)**:
   - $\eta = w_{tip}^{computed} / w_{tip}^{exact}$
   - **LOCKING-FREE ($\eta > 0.85$)**: `C3D8I`, `C3D8R`, `C3D10`, `C3D10M`이 도달해야 하는 목표.
   - **LOCKED ($\eta < 0.50$)**: 완전 가우스 적분 선형 요소(`C3D8`, `C3D4`, `C3D6`)의 전형적인 기생 전단 잠김(정상 고전 거동).
3. **비압축성 극한 (Volumetric Verdict, $\nu=0.49999$)**:
   - **INCOMPRESSIBLE-OK**: `C3D8_FBAR`, `C3D8H`, `C3D4_ANP`, `C3D10M`이 도달해야 하는 목표.
   - **VOL-LOCKED**: 비압축성 체적 잠김으로 변위가 0으로 동결되는 현상.
   - **ERROR / DIVERGED**: 뉴턴 반복법 발산 또는 접선 불량.

---

## 3. 현 시점 11종 요소별 진단 및 개선 처방전 (Actionable Recipes)

### 3.1 `C3D8_FBAR` (Multiplicative F-bar Hex)
- **현재 상태**: 모든 패치 테스트 100% 통과 (오차 ~1e-11). 실용 대역($\nu \le 0.499$)에서는 3~28회 뉴턴 반복으로 정상 수렴. 그러나 $\nu \in [0.4999, 0.49992]$ 및 $\nu=0.49999$에서 수치 불안정/발산 발생.
- **이론적 근본 원인**:
  - `B_L` 가상변형도 연산자는 실제 $F$로 정상 수정되었으나, 알고리즘 접선 $K$에 체적 투영 스케일러 $(J_0 / J)^{1/3}$의 변분 항이 누락되어 있음.
  - de Souza Neto et al. (1996) §4.2 식 (45)에 따르면, $d\bar{F}$는 $dF$뿐만 아니라 중심점 체적비 $J_0$의 변분 $dJ_0$에 의한 교차 접선 항을 포함해야 함.
- **수정 대상 파일**: `dispsolver/element3d/c3d8_fbar_tl_numba.py`
- **개선 방법**:
  1. 중심 적분점(GP 0)에서의 변형도 구배 $F_0$에 대한 잔차 미분 항 $\Delta K_{vol}$을 계산하여 각 가우스점의 강성 행렬에 조립.
  2. $\nu \to 0.5$ 극한에서 페널티 강성이 과도하게 발산하지 않도록 중간 정규화 적용.
- **검증 명령**:
  ```bash
  python -u benchmark_element/benchmark_3d_mechanics.py
  ```
  `C3D8_FBAR`의 비압축성 판정이 `ERROR`에서 `INCOMPRESSIBLE-OK`로 전환되는지 확인.

---

### 3.2 `C3D4_ANP` (Average Nodal Pressure Tet)
- **현재 상태**: 패치 테스트 통과. 단, 매끄러운 외팔보 굽힘 벤치마크에서 일반 `C3D4`와 동일한 처짐비(0.21) 및 `VOL-LOCKED` 판정.
- **이론적 근본 원인**:
  - Bonet & Burton (1998) 2-Pass 체적 평균화($J_a = v_a / V_a$, $\bar{F}_e = (\bar{J}_e / J_e)^{1/3} F$) 자체는 정상 구현됨 (`dev_log/solve_step_false_convergence_20260913.md` 참조).
  - 그러나 현재의 벤치마크는 규칙적인 외팔보 메쉬에 부드러운 하중이 걸려 각 요소의 $J_e$와 인접 절점 평균 $\bar{J}_e$의 차이가 $10^{-7}$ 미만임. 즉, 평균화할 체적 압력 진동(Checkerboard pressure)이 애초에 존재하지 않아 일반 C3D4와 차이가 드러나지 않음.
- **수정 대상 파일**: `benchmark_element/mechanics_patches.py` 및 `dispsolver/element3d/c3d4_anp_numba.py`
- **개선 방법**:
  1. ANP의 진정한 장점은 **불균일 국소 하중(Indentation/Punch test)** 또는 **심한 메쉬 왜곡** 하에서 요소 간 체적 압력 분산을 제거하는 것임.
  2. 벤치마크에 국소 구속 블록 시험을 추가하여 ANP의 잠김 완화 효과를 명시적으로 평가.
- **검증 명령**:
  ```bash
  pytest tests/test_3d_c3d4_anp.py -v
  ```

---

### 3.3 `C3D8H` (Mixed u-p Hybrid Hex) vs `C3D8_CR` (Corotational B-bar Hex)
- **현재 상태**: 패치 테스트 통과. 섹션 3 디스패치 이상 감지에서 `C3D8_CR`과 `C3D8H`의 굽힘 및 체적 변위가 $10^{-12}$ 수준으로 일치하는 이상 현상 감지.
- **이론적 근본 원인**:
  - `C3D8H`는 Hellinger-Reissner 변분 원리에 기반한 체적 전용 하이브리드 요소이며, 전단 잠김 완화 기구는 없음 (Abaqus Theory Guide §3.2.3). 따라서 굽힘 처짐비 0.50(LOCKED)은 이론적 정상 결과임.
  - 그러나 `C3D8_CR`과 소수점 12자리까지 일치하는 것은 `dispsolver/solver3d/dynamic3d.py`의 디스패치 테이블(Group $k=1$ vs $k=2$)에서 두 요소가 동일한 커널이나 프로퍼티를 참조하고 있을 가능성을 시사함 (AGENTS.md §4.2/§4.8/§4.16 전례).
- **수정 대상 파일**: `dispsolver/solver3d/dynamic3d.py` (Line 190~225)
- **개선 방법**:
  1. `dynamic3d.py` 내 `elem_kernel_indices` 분류부에서 `C3D8_CR`과 `C3D8H`가 각각 `c3d8_corotational_numba`와 `c3d8_hybrid_numba`로 엄격히 분기되는지 확인.
  2. 디스패치 분기 누락이 없는지 단위 테스트로 격리 검증.

---

### 3.4 `C3D8I` (9-Mode EAS Incompatible Hex)
- **현재 상태**: 패치 테스트 통과 (1.7e-11). 굽힘 처짐비 0.96으로 전단 잠김 완전 극복 (`LOCKING-FREE`).
- **알려진 추가 확장 영역 (Next Step)**:
  - 현재 구현은 9개의 비적합 변위 구배 모드($\alpha_1 \dots \alpha_9$)만 정적 축약되어 있음.
  - Abaqus 정식 C3D8I 및 Simo & Armero (1992) 정식화는 비압축성 체적 잠김 완화를 위해 4개의 추가 체적 모드($\beta_1 \dots \beta_4$, 총 13모드)를 포함함.
  - 비압축성 극한($\nu \to 0.5$)에서 추가적인 유연성이 요구될 경우 4개 체적 모드 확장 구현 권장.
- **수정 대상 파일**: `dispsolver/element3d/c3d8_eas_tl_numba.py`

---

### 3.5 표준 1차 요소군 (`C3D8`, `C3D4`, `C3D6`)
- **현재 상태**: 패치 테스트 전원 100% 통과. 굽힘 처짐비 0.21~0.48 (`LOCKED`).
- **가이드**: 완전 적분 1차 요소가 굽힘에서 전단 잠김을 일으키는 것은 **FEM 수치역학의 지극히 정상적인 이론적 거동**입니다. 이들 기본 요소의 강성을 인위적으로 낮추는 수정을 가해서는 안 되며, 굽힘 지배 문제에서는 고차 또는 감차적분 요소(`C3D8I`, `C3D8R`, `C3D10M`)를 사용하는 것이 표준 설계입니다.

---

## 4. 에이전트 작업 수칙 (Multi-Agent Protocol)

1. **자가 승인 금지 (No Self-Approval)**:
   - "코드를 수정했으므로 정상 작동할 것입니다"라는 설명만으로 완료를 선언하지 마십시오.
   - 반드시 `python -u benchmark_element/benchmark_3d_mechanics.py`를 실행하여 갱신된 수치를 마크다운 리포트에 반영하십시오.
2. **외과적 변경 (Surgical Changes)**:
   - 수정 대상이 아닌 인접 요소 커널이나 솔버 공통 루프를 임의로 리팩토링하지 마십시오.
3. **회귀 방지 게이트 통과 (Regression Gate)**:
   - 작업을 마치기 전 반드시 다음 명령을 실행하여 회귀가 없음을 입증하십시오:
     ```bash
     pytest tests/test_3d_mechanics_benchmarks.py -v
     pytest tests/test_3d_c3d8r.py tests/test_3d_c3d4_anp.py tests/test_3d_c3d10m.py tests/test_3d_c3d6.py -q
     ```
