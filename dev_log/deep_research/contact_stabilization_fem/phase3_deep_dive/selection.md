# Phase 3 Selection — 8개 심층 독해 대상 논문

**선택 기준**: 현재 코드의 Fix A/B/C와 직접적으로 연관된 수학적 정식화를 포함한 논문 우선

| # | 논문 | 선택 이유 |
|---|---|---|
| S1 | Wriggers (2006) §9.4 - Active Set Freeze 부분 | Fix A의 직접 근거 (Unilateral 구속 필요성) |
| S2 | Hüeber, Stadler, Wohlmuth (2008) - SIAM Sci. Comput. | Fix B의 직접 근거 (SDI 후 스텝 완화 Algorithm 5.1) |
| S3 | Hintermüller, Ito, Kunisch (2002) - SIAM J. Optim. | PDAS 이론 기반 (AL + Active Set 등가성) |
| S4 | Abaqus Theory Guide §2.2 (2023) | Fix C 기준값 (r_tol, c_tol 산업 표준) |
| S5 | Alart & Curnier (1991) - CMAME | 마찰 접촉 AL 혼합 정식화 (Coulomb 구현) |
| S6 | Fischer & Wriggers (2005) - IJNME | NTS에서 PDAS 구현 (우리 코드 구조와 유사) |
| S7 | Rebel et al. (2002) - 페널티 자동 스케일링 | Fix D(페널티 과대 문제) 근거 |
| S8 | Vu-Quoc & Nghiem (2007) - 채터링 분석 | 채터링 진단 및 Hysteresis Gate 설계 |
