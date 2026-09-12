# Implementation Plan - 3D Element Benchmark & Multi-Agent Renovation Framework

3D 솔리드 요소 11종의 벤치마크 결과를 바탕으로, **다른 에이전트 및 개발자가 벤치마크 코드와 마크다운 리포트를 보고 즉각적으로 원인을 분석하고 개선/업데이트를 진행할 수 있도록 결과를 코드화하여 출력**하고, `benchmark_element/` 폴더 내에 **종합 진단 및 개선 가이드라인(IMPROVEMENT_GUIDE.md)**을 체계화하는 구현 계획입니다.

---

## 1. 목표 및 배경 (Goal & Background)

현재 `benchmark_element/` 폴더에는 11종 3D 요소에 대한 수학적 랭크/접선 벤치마크(`benchmark_3d_elements.py`)와 역학 정확도/잠김 벤치마크(`benchmark_3d_mechanics.py`), 그리고 회귀 게이트(`tests/test_3d_mechanics_benchmarks.py`)가 구축되어 있습니다.

하지만 현재 출력되는 결과표는 수치와 합격/불합격 판정 위주로 되어 있어, **새로 투입된 서브에이전트나 후속 개발자가 특정 요소의 결함(예: C3D8_FBAR의 좁은 대역 불안정성, C3D4_ANP의 비압축성 거동 해석, C3D8_CR과 C3D8H의 수치 일치 이상 현상)을 보았을 때 무엇이 문제이고 어떤 소스 파일의 어느 함수를 어떻게 수정해야 하는지 직관적으로 파악하기 어렵습니다.**

따라서:
1. **벤치마크 실행 엔진(`benchmark_3d_mechanics.py`) 자체를 개선**하여, 벤치마크 실행 즉시 터미널 콘솔 및 생성되는 마크다운 리포트에 **"요소별 원인 분석 및 실천적 개선 처방전(Actionable Renovation Recipes)"**을 데이터 기반으로 자동 출력하도록 코드화합니다.
2. `benchmark_element/` 폴더 내에 **다른 에이전트를 위한 영구 가이드북(`IMPROVEMENT_GUIDE.md`)**을 구축하여, 누구나 3D 요소의 수학적/역학적 배경과 구체적인 코드 개선 절차를 이해하고 안전하게 업데이트할 수 있도록 지원합니다.

---

## 2. User Review Required

> [!IMPORTANT]
> **에이전트 맞춤형 개선 처방전(Actionable Recipes)의 자동 출력**
> - 벤치마크 스크립트 실행 시, 단순히 PASS/FAIL 표만 출력하는 것이 아니라:
>   - **결함/경고 요소 식별**
>   - **이론적 원인 및 물리적 한계 구분** (예: 선형 요소의 전단 잠김은 이론적 정상 거동 vs 커널/디스패치 버그 구분)
>   - **수정해야 할 소스 파일 및 함수 위치** (예: `dispsolver/element3d/c3d8_fbar_tl_numba.py`)
>   - **구체적인 수정 방향 및 재검증 명령**
>   을 콘솔과 리포트 Section 5에 실시간 생성합니다.

> [!TIP]
> **검증 데이터 보존 및 회귀 방지**
> - 기존 `tests/test_3d_mechanics_benchmarks.py`의 엄격한 pytest 게이트를 그대로 유지하여, 새로운 개선 시도가 기존에 합격한 11개 요소의 패치 테스트(100% 통과)를 퇴행(regression)시키지 않도록 안전장치를 유지합니다.

---

## 3. 제안 변경 사항 (Proposed Changes)

### 1단계: 벤치마크 실행 엔진에 액션 중심 진단 출력 코드화
#### [MODIFY] `benchmark_element/benchmark_3d_mechanics.py`
- `generate_actionable_improvement_guide()` 함수 구현:
  - 벤치마크 결과(`patch_res`, `abq_patch_res`, `bending_res`, `vol_res`)와 이상 감지(`anomalies`) 데이터를 분석하여 요소별 실질적 수정 처방전 생성.
  - **콘솔 터미널 출력**: 실행 종료 직후 화면에 핵심 조치 사항 요약 출력.
  - **마크다운 리포트 Section 5 추가**: `dev_log/benchmark_3d_mechanics_YYYYMMDD.md`에 다음 형식으로 자동 기록:
    ```markdown
    ## 5. 다른 에이전트를 위한 요소별 개선 및 업데이트 가이드 (Actionable Improvement Guide)
    ### [C3D8_FBAR]
    - 현상: nu=0.49999에서 수치 불안정 (DIVERGED/WRONG answer)
    - 원인: 체적 일관성 접선(Volumetric-consistency tangent) 항 누락 및 극한 비압축성 조달 불량
    - 수정 대상: `dispsolver/element3d/c3d8_fbar_tl_numba.py` 내 `compute_c3d8_fbar_element_numba`
    - 개선 방법: de Souza Neto (1996) 식 (45)의 체적 투영 보정 접선 항 추가
    - 검증 명령: `python -u benchmark_element/benchmark_3d_mechanics.py`
    ```

### 2단계: 에이전트용 종합 개선 가이드북 영구 구축
#### [NEW] `benchmark_element/IMPROVEMENT_GUIDE.md`
- 다른 AI 에이전트와 엔지니어가 참조할 수 있는 체계적인 기술 문서 작성:
  1. **11종 3D 요소의 수치역학 정식화 및 책임 영역 요약**
  2. **4대 역학 벤치마크(Irons 패치, Abaqus VM 패치, 외팔보 굽힘, 비압축성 체적 스윕)의 물리적 의미와 합격 판정 기준**
  3. **현 시점 확인된 주요 이슈 및 요소별 구체적 개선 로드맵**:
     - `C3D8_FBAR`: 좁은 불안정성 대역($\nu \in [0.4999, 0.49992]$) 원인과 체적 접선 보정 가이드.
     - `C3D4_ANP`: 매끄러운 굽힘에서의 절점 압력 차이 미미 현상 및 불균일 집중 하중 벤치마크 적용 가이드.
     - `C3D8H`: 체적 하이브리드 요소의 전단 잠김(정상 거동) vs 체적 완화 메커니즘 설명.
     - `C3D8_CR` vs `C3D8H`: 디스패치 및 커널 공유 여부 점검 프로토콜.
     - `C3D8I`: 9개 알파 모드 외에 4개 체적 베타 모드 추가 가이드.
  4. **자가 승인 금지 및 회귀 방지 검증 워크플로우** (`pytest tests/test_3d_mechanics_benchmarks.py -v`)

### 3단계: 디렉터리 안내 및 상위 문서 연동
#### [MODIFY] `benchmark_element/README.md`
- `IMPROVEMENT_GUIDE.md` 링크 및 "에이전트가 개선을 시작할 때 반드시 읽어야 할 지침" 섹션 추가.
#### [MODIFY] `AGENTS.md`
- §2.2 및 §5에 `benchmark_element/IMPROVEMENT_GUIDE.md`를 공식 가이드라인으로 명시.

---

## 4. 검증 계획 (Verification Plan)

### Automated Tests
1. **개선된 벤치마크 실행 및 액션 출력 검증**:
   ```bash
   python -u benchmark_element/benchmark_3d_mechanics.py
   ```
   - 콘솔 및 `dev_log/benchmark_3d_mechanics_YYYYMMDD.md`에 Section 5 (Actionable Improvement Guide)가 정상 생성되는지 확인.
2. **기존 회귀 게이트 통과 검증**:
   ```bash
   pytest tests/test_3d_mechanics_benchmarks.py -v
   ```
   - 11개 요소의 패치 테스트 100% 통과 및 기존 xfail 항목의 안정성 재확인.
3. **3D 요소 단위 테스트 스위트 확인**:
   ```bash
   pytest tests/test_3d_c3d8r.py tests/test_3d_c3d4_anp.py tests/test_3d_c3d10m.py tests/test_3d_c3d6.py -q
   ```

### Manual Verification
- `benchmark_element/IMPROVEMENT_GUIDE.md`를 열람하여, 다른 에이전트가 별도의 추가 배경지식 없이도 즉시 특정 요소의 코드 수정에 착수할 수 있을 만큼 구체적인지 확인.
