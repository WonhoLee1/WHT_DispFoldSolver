# Abaqus Verification Reference Manual 26개 문서 분석 및 WHT_DispFoldSolver 재료 물성 고도화 의뢰서

## 1. 분석 대상 Abaqus Verification Reference Manual 문서군 (26개)
1. `simaver-c-elastic.htm` (선형 등방/이방성 탄성, Orthotropic/Anisotropic Elasticity)
2. `simaver-c-viscoelastic.htm` (소변형/대변형 시간영역 점탄성 완화 및 크립)
3. `simaver-c-mullinseffect.htm` (Mullins 효과: 충전 고무의 주기적 하중에 따른 손상 및 응력 연화)
4. `simaver-c-hysteretic.htm` (초탄성/고무 재료의 비탄성 이력 현상 및 에너지 소산)
5. `simaver-c-tempelastic.htm` (온도 의존 탄성: **반드시 포함 대상**)
6. `simaver-c-fieldelastic.htm` (사용자 정의 필드 변수 Field Variable 의존 탄성)
7. `simaver-c-viscohyper.htm` (초탄성 기반 대변형 점탄성 결합)
8. `simaver-c-nonlinviscohyper.htm` (비선형 초-점탄성 및 변형률 의존 완화)
9. `simaver-c-rateindepplast.htm` (속도 무관 Mises 소성 및 등방/이동 경화)
10. `simaver-c-viscocylinder.htm` (내압을 받는 두꺼운 점탄성 실린더 해석해 대조)
11. `simaver-c-ratedepplastexp.htm` (속도 의존 소성: Cowper-Symonds 및 Johnson-Cook 동적 항복)
12. `simaver-c-annealtemp.htm` (소성 변형 이력 소거 및 어닐링 온도 제어)
13. `simaver-c-tempplastic.htm` (온도 의존 소성 거동: **반드시 포함 대상**)
14. `simaver-c-fieldplastic.htm` (필드 변수 의존 소성 거동)
15. `simaver-c-johnsoncook.htm` (Johnson-Cook 소성 및 열-기계 연성 파손)
16. `simaver-c-damage.htm` (연성 파손 및 전단 파손 모델: Ductile & Shear Damage initiation/evolution)
17. `simaver-c-creep.htm` (고온 크립 변형: Power-law, Norton, Time-hardening 크립)
18. `simaver-c-viscous.htm` (재료 내부 점성 감쇠 및 안정화 점성)
19. `simaver-c-concrete.htm` (콘크리트 소성: **제외/후순위**)
20. `simaver-c-concretedamaged.htm` (콘크리트 손상 소성: **제외/후순위**)
21. `simaver-c-damping.htm` (Rayleigh 감쇠 및 구조 감쇠)
22. `simaver-c-simpleexpansion.htm` (열팽창 계수 $\alpha \Delta T$ 열-변형률: **반드시 포함 대상**)
23. `simaver-c-plasticity-corrections.htm` (소성 알고리즘 수치 보정 및 Return Mapping)
24. `simaver-c-fieldvardependentconductivity.htm` (필드 변수 의존 열전도도: **제외**)
25. `simaver-c-conductivityspecificheat.htm` (열전도도 및 비열: **제외**)
26. `simaver-c-gapconductance.htm` (접촉 열전도: **제외**)

---

## 2. 사용자 핵심 지침 및 분석 조건
1. **열 전달(Heat Transfer / Gap Conductance / Conductivity / Specific Heat)은 우선 순위에서 제외** (추후 조절).
2. **온도 의존 필드(Temperature-dependent field) 정보 및 직접적인 온도 처방(Thermal Expansion, Temperature-dependent Elasticity & Plasticity)은 반드시 포함**.
3. 현재 우리 솔버(`WHT_DispFoldSolver`)에 이미 완성된 재료 기능:
   - Abaqus 2026 초탄성 전제품군 (Arruda-Boyce, Mooney-Rivlin, Neo-Hookean, Ogden N=1..6, Polynomial, Reduced Polynomial/Yeoh, Marlow) Numba 고속화.
   - 최대 6개 항(M=6, 42-Float SDV) Prony 대변형 점탄성.
   - Radioss LAW36 / Abaqus 테이블 탄-소성 스무딩 및 $H_{\text{tail}} = 10^{-3} E$ 안정화 연장 테일 완비.
4. **목표 과제**:
   - 위 26개 문서 중 제외 항목을 배제하고, 솔버에 추가로 도입해야 할 핵심 물성 모델(온도 의존 탄소성, 열팽창, Mullins 효과, Johnson-Cook / 속도 의존 소성, 연성 파손 등)의 정식화 및 아키텍처 수립.
   - 각 문서에 연동된 표준 `.inp` 벤치마크 파일 사양 및 검증 시나리오 도출.
   - 벤치마크 수행 시점은 기능 구현과 점검이 완료된 이후로 설정.

---

## 3. 6단계 전방위 분석 요구사항 (모든 패널 공통)
1. **System Design (설계)**: 온도 필드 변수 및 신규 물성 확장을 위한 `elem_props` 및 SDV 버퍼 아키텍처.
2. **Technical Research (조사)**: Abaqus 이론 매뉴얼 및 Verification 매뉴얼의 관련 정식화 조사.
3. **Theoretical Evaluation (이론평가)**: 수치 안정성, 일관 접선 강성(Consistent Tangent), 체적 비압축성과의 결합 평가.
4. **Strategic Planning (계획)**: 우선순위별 구현 및 벤치마크 로드맵.
5. **Core Problem Analysis (문제분석)**: 폴딩 디스플레이 및 다층 박막에서의 온도 변화/열팽창 미스매치 및 속도 의존 소성의 영향 분석.
6. **Concrete Resolution Plan (해결방안)**: 구체적인 Numba 커널 시그니처 및 솔버 연동 수식 제시.
