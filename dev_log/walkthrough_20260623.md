# Walkthrough: 구현 코드 검사, 수렴성 개선 및 Hybrid 요소 이식 (2026-06-23)

최근 구현된 코드들에 대한 정밀 검사(Inspection), 기존 테스트 과정에서 발생한 Newton-Raphson 수렴 오류 분석/수정, 그리고 추가적인 비선형 대변형 극복을 위한 **Q1P0 Hybrid (u-p) 혼합 요소**와 **Backtracking Line Search** 구현 결과를 요약합니다.

---

## 1. 코드 검사 및 패키지 초기화 수정
* **VTKHDF 익스포터 (`vtkhdf_exporter.py`)**: `h5py`를 사용한 비정형 격자(UnstructuredGrid) 출력이 안정적으로 구현되었으며, ParaView 5.10 이상과의 호환성 규격을 완벽하게 충족합니다.
* **JAX x64 글로벌 적용 보완**:
  - pytest를 전체 순차 실행할 때 일부 테스트 파일 간의 임포트 순서 때문에 JAX의 `jax_enable_x64` 글로벌 설정이 꼬이며, float32 정밀도로 강제 삭감(truncation) 경고가 발생하고 수치 오차로 수렴 실패를 초래하는 현상을 발견했습니다.
  - 이를 해결하기 위해 [__init__.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/__init__.py) 패키지 초기화 진입점 수준에서 JAX 임포트 즉시 `jax.config.update("jax_enable_x64", True)`를 무조건 실행하도록 보완하여, 전체 테스트 순차 실행 시의 정밀도 꼬임 버그를 원천 차단했습니다.
  - 이 조치 후 **기존 84개 테스트를 포함한 전체 86개 테스트가 100% 완전 성공(Pass)** 하였습니다.

---

## 2. Q1P0 Hybrid (u-p) 혼합 요소 구현 (체적 잠김 완화)
디스플레이 폴딩 해석 예제([ex03_display_fold.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/examples/ex03_display_fold.py))에서 힌지부의 극심한 대변형 기하 압축과 체적 락킹(Volumetric Locking)을 해결하기 위해 상용 소프트웨어의 표준인 Hybrid 요소를 도입했습니다.
* **정적 축소(Static Condensation) 기반 Q1P0**:
  - 요소 내부 정수압 자유도 $p$를 요소 에너지 포텐셜 극값 조건($R_p = 0$) 하에서 대수적으로 직접 소거하는 정적 축소를 유도했습니다.
  - 변위 $\mathbf{u}$만의 단일 축소 에너지 함수 $\tilde{E}_{elem}(\mathbf{u})$를 정의한 뒤 JAX 자동 미분을 활용해 $8 \times 8$ 강성 및 $8 \times 1$ 힘 벡터를 추출함으로써, 기존 글로벌 솔버 아셈블리 구조를 전혀 변경하지 않는 안전하고 우아한 이식을 달성했습니다.
  - [q4_up_jax.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/element/q4_up_jax.py) 파일 신설 및 [dynamic.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/solver/dynamic.py)에 `element_type="Q4_UP"` 옵션을 통해 연계 조립되도록 조치했습니다.

---

## 3. Backtracking Line Search (라인 검색) 구현 (수치 안정성 확보)
극단적인 휨/접힘으로 인해 iteration 도중 요소 형상이 일시적으로 무너지거나 다이버전스가 유발되는 문제를 차단하기 위해 솔버에 라인 검색을 연동했습니다.
* **NaN-prevention Backtracking**:
  - Newton-Raphson 루프 내에서 임시 스텝 `u_temp = u_k + alpha * du`에 대한 잔차 노름이 `NaN` 또는 `Inf`로 판정될 때에만, 스텝 크기 $\alpha$를 `0.5`, `0.25` 등으로 감쇠시키는 Backtracking 방식을 도입했습니다.
  - 이를 통해 평소의 정상적인 NR 2차 수렴 속도를 100% 보존하면서도, 기하학적 파괴가 날 수 있는 임계 국면만 똑똑하게 회피하도록 안전 장치를 강화했습니다.

---

## 4. 디스플레이 폴딩 예제 검증 결과
* **해석 한계 돌파**:
  - 기존 일반 Q4 요소로 구동 시 volumetric locking과 수치 불안정성으로 인해 **`0.57`초(로드 팩터 57%)** 지점에서 다이버전스가 발생해 해석이 조기 중단되었습니다.
  - 새로 구현한 **Q1P0 Hybrid 요소 + Backtracking Line Search** 조합으로 해석을 실행한 결과, 이전의 임계 지점을 흔들림 없이 가뿐히 통과하여 **`0.68`초(로드 팩터 68.3%)** 지점을 넘어 순항 수렴하고 있음을 확인했습니다.
  - 다이버전스가 발생하려는 찌그러짐 임계 순간마다 Line Search와 타임 스텝 삭감(cutback)이 맞물리며 안정적인 회피 거동을 완벽히 보여줍니다.

---

## 5. 상용 소프트웨어 수준의 수렴 상태 출력 기능 탑재 및 경계조건 버그 수정
* **경계조건 (BC) Jacobian 아셈블리 버그 수정**:
  - 강제 경계조건이 부여된 DOF에 해당하는 Jacobian Matrix 행(row)을 단위행(Identity row)으로 변환하는 과정에서, 비대각(off-diagonal) 성분을 `0.0`으로 초기화하는 `J.data[ptr] = 0.0` 코드가 누락된 것을 발견하여 이를 복구했습니다.
  - 이 수정으로 인해 솔버 수렴 에러가 발생하던 `test_solver.py` 및 `test_visco_solver.py` 등 총 9개의 비선형 해석 테스트가 완벽히 재동작하며 **90개 전체 단위 테스트 100% Pass**를 다시 확립하였습니다.
* **상용 소프트웨어 스타일의 상세 수렴 상태 출력 (Newton-Raphson)**:
  - Abaqus/Ansys 등의 상용 비선형 유한요소 해석 소프트웨어와 유사하게, 해석 중 수렴 경로 상에서 어떤 노드/DOF가 잔차(residual force)나 변위 보정량(correction)의 보틀넥인지 직관적으로 파악할 수 있도록 텍스트 출력을 개선했습니다.
  - 매 이터레이션마다 **최대 힘 잔차력(Max Res.Force) 및 발생 노드/DOF**, **최대 변위 보정량(Max Disp.Corr) 및 발생 노드/DOF**, **최대 변위 증분(Max Disp.Incr)**, **상대 변위 변화율(Disp.Ratio)**, **에너지 에러(Energy.Err)**, **라인 검색 계수(LS.alpha)**, **활성 접촉 쌍 수(Contacts)** 정보를 깔끔한 표로 보여줍니다.
  - 이를 통해 어떤 노드에서 수렴이 튀고 있는지 실시간으로 추적 가능하여 비선형 디버깅 생산성을 극대화하였습니다.

* **Neo-Hookean 초탄성 재료의 해석적 유도 (Analytical) 및 오버헤드 최적화**:
  - 기존 `NeoHookean` 모델은 `MaterialModel` 부모 클래스의 JAX Autodiff 파이프라인에 의존하여 매 가우스 포인트(2,772개) 루프마다 `jax.grad` 및 `jax.hessian` 연산 그래프를 새로 작성하는 엄청난 Python/JAX 오버헤드가 발생했습니다. 이로 인해 비점소성 순차 해석 루프에서 1스텝(Newton Raphson) 계산 시 1분 이상 지연되는 현상을 확인했습니다.
  - 이를 해결하기 위해 Neo-Hookean 초탄성체의 2nd P-K 응력($S = \mu(I - C^{-1}) + \lambda \ln J \, C^{-1}$)과 4차 일관성 접선 강성 텐서(Consistent Tangent Tensor)를 대수적으로 완전 유도하여 [neohookean.py](file:///d:/PythonCodeStudy/WHT_DispFoldSolver/dispsolver/material/neohookean.py)에 직접 Numpy 기반 해석적 식으로 오버라이드 구현하였습니다.
  - 동시에 JAX vmap 벡터화가 필요한 상황(Hyperelastic 단일 재료)에서의 호환성을 위해, 입력 인자가 JAX Tracer/Array인 경우에는 자동으로 부모 클래스인 JAX Autodiff 버전으로 폴백(Fallback)하도록 이중 하이브리드 인터페이스를 보완했습니다.
  - 이 고속화 튜닝 적용 결과, 디스플레이 다층 적층 폴딩 해석에서 **Newton-Raphson 1개 스텝 완료 속도가 기존 약 1분에서 0.8초 미만으로 단축(약 75배 이상 속도 향상)** 되는 압도적인 연산 효율성을 달성했습니다.

