# WHT_DispFoldSolver — 3대 극복 기법 (대안 A·B·C) 통합 구현 및 검증 계획서

**작성일**: 2026-07-25  
**문서 상태**: 3대 기법 통합 아키텍처 확정 및 파이프라인 수립  
**목표**: Q4 연속체 요소의 38° 변형 한계 완전 극복 및 Abaqus/Standard 수준의 90° 폴딩 수렴성·정확도 100% 달성  

---

## 1. 3대 기법 통합 전략 (Integrated Multi-Strategy Pipeline)

단일 수치 기법의 한계를 완벽히 보완하기 위해 **대안 A(Co-rotational), 대안 B(Solid-Shell), 대안 C(Adaptive Remeshing)**를 모두 구현하고 모듈화하여 솔버에 통합합니다.

```
========================================================================================
                      3대 기법 통합 시너지 파이프라인 (A + B + C)
========================================================================================
 [대안 A: Co-rotational Q4]   -->  [대안 B: Solid-Shell/Beam]  -->  [대안 C: Adaptive Remesh]
  기존 2D 다층 메쉬 유효성 유지    박막 굽힘/전단 락킹 완전 제어     힌지 극곡률 구역 평활화 & 재분할
  38°->70°+ 대회전 수렴성 확보     초박막(PSA/PET) 전용 쉘 옵션      최종 90° 무결점 완주 보장
========================================================================================
```

---

## 2. 세부 구현 로드맵 (Phased Execution Plan)

### Phase A: Co-rotational Q4 Formulation (우선 구현)
- **파일**: `dispsolver/element/q4_corotational_jax.py`
- **핵심 내용**:
  - 각 Q4 요소의 대변형 회전각 $\alpha = \text{atan2}(e_1, E_1)$ 기반 국소 동회전 좌표계 매트릭스 $R$ 계산
  - 변위 $u \to \bar{u} = R^T u$ 변환 및 pure strain 상태에서 B-bar / EAS 수치 업데이트
  - 요소 강성 변환: $K_e = R K_{local} R^T + K_{geo}$
- **검증 기준**: 기존 38° 정체 구역을 돌파하고 수렴 회전각 60°~90° 확장 확인

### Phase B: 2D Solid-Shell / Mindlin-Reissner Shell Formulation
- **파일**: `dispsolver/element/shell_2d_jax.py`
- **핵심 내용**:
  - 0.0167mm 초박막 레이어(PET/PSA) 전용 2D 쉘/빔 요소 구축 (Abaqus S4R 패러다임)
  - ANS (Assumed Natural Strains) 및 EAS 기법 결합으로 두께 및 전단 락킹 100% 제거
  - 7층 다층 복합체 층간 접합 인터페이스(Tie)와 호환되는 Solid-Shell formulation
- **검증 기준**: 굽힘 락킹 없는 순수 굽힘/대회전 변형 정확도 확보

### Phase C: Adaptive Local Remeshing & Inversion-Free Mesh Smoothing
- **파일**: `dispsolver/mesh/adaptive_remesh.py`
- **핵심 내용**:
  - 힌지부 근방 요소의 변형 왜곡 지표($det(J)/J_0 < 0.1$ 또는 Aspect Ratio > 50) 감지 시 자동 트리거
  - Laplacian & Metric Mesh Smoothing으로 요소 찌그러짐 해소
  - 힌지 곡률 집중 구역 국소 절점 재분할(Subdivision) 및 수치 이력 변수($S, \alpha, h_i$) 보존 보간(Conservative State Mapping)
- **검증 기준**: 극심한 대변형 구간에서도 요소 뒤집힘(Inversion/NaN) 0회 달성

---

## 3. 단계별 검증 절차 (Verification Plan)

```
1. Phase A 구현 -> verify: ex03_display_fold.py 실행 시 38° 돌파 확인
2. Phase B 구현 -> verify: pytest tests/test_solid_shell.py 통과 확인
3. Phase C 구현 -> verify: adaptive remesh 모듈 작동 및 state transfer 오차 < 0.01% 확인
4. 종합 통합 검증 -> verify: 90° 완전 폴딩 100% 완주 및 Abaqus/Standard 오차 < 0.1% 확인
```

---

*Updated & Persisted in `./dev_log/integrated_folding_strategy_plan_20260725.md`*
