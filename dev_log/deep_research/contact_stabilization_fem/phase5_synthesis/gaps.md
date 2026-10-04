# Phase 5: Gaps Analysis

## 핵심 격차 요약

1. **Bilateral Freeze 버그 [P1]**: Wriggers §9.4 = "compressive only" → 현재 코드 bilateral
2. **SDI 스텝 완화 미구현 [P2]**: Hüeber 2008 β=0.5 → 현재 β=1.0 (full step) → Iter 2 폭발
3. **r_tol 불일치 [P3]**: Abaqus §2.2.4 = 8%~10% → 현재 5% → 경계선 실패
4. **k_contact 과대 [P4]**: Rebel 2002 = α×E/h = 2e6 → 현재 1e8 (50×) → 조건수 악화
5. **Hysteresis Gate 미활성 [P5]**: Vu-Quoc 2007 → 현재 코드에 구현은 있으나 비활성화

## 참고 없이 구현된 기능 (이미 올바름)

- AL + PDAS: Hintermüller 2002와 일치
- Coulomb 마찰: Alart-Curnier 1991과 일치  
- Dual Mortar (2D/3D): Popp 2012와 일치
