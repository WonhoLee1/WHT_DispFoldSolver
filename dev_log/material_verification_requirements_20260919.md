# 재료 검증 요구사항 정리 (Abaqus Verification Manual 기준, 2026-09-19)

소스: `C:\SIMULIA\Documentation\2024LE\English\SIMACAEVERRefMap\simaver-c-*.htm` (27종) +
inp 예제 `...\SIMAINPRefResources\` (25,056개 중 필요분 실존 확인).
벤치마크 실행은 구현·점검 완료 후 (이번 문서는 요구 파악까지).

## 1. 범위 필터 (사용자 규칙 적용)

- **제외 (열전달)**: fieldvardependentconductivity, conductivityspecificheat, gapconductance
  → 열요소(C3D8T 등)·과도해석 영역. 추후.
- **포함 (온도 직접부여)**: tempelastic, fieldelastic, tempplastic, fieldplastic,
  annealtemp, simpleexpansion (predefined T/f 필드 기반 — 열해석 없이 T를 외생 입력).

## 2. 기능 매트릭스 (파일 → Abaqus 기능 → 현 상태)

| # | 검증 파일 | 핵심 기능 | 현 구현 | 갭 |
|:---|:---|:---|:---:|:---|
| 1 | elastic | 등방+직교+이방 탄성, 다공탄성, hypo, 초탄성(poly/reduced/NH/Mooney/Yeoh/Ogden/AB/VdW/Marlow/Fung) | △ (NH·AB·Yeoh만) | Ogden·Mooney·poly·이방성·다공 |
| 2 | viscoelastic | 대변형 시간영역+hyperfoam, 소변형+이방, 주파수영역(저장/손실 직접) | △ (Prony+WLF, 등방) | 주파수영역, hyperfoam 결합, 이방 |
| 3 | mullinseffect | Mullins+영구변형+폼 에너지소산 (Marlow/Ogden) | ✗ | PSA 관련 최우선 후보 |
| 4 | hysteretic | Bergström-Boyce 시간의존/무관 (CAX4·CPE4 단축압축) | ✗ | — |
| 5 | tempelastic | T의존 등방/직교/이방/lamina + predefined T (T2D2~M3D4R) | ✗ | T-테이블 보간 공통인프라 |
| 6 | fieldelastic | 필드의존 탄성 (同上 구조) | ✗ | 同上 (FIELD 변수) |
| 7 | viscohyper | 대변형 점탄성+hyper (Explicit, CAX4R·CPE4R~M3D4R) | △ (Standard형만) | Explicit 검증 케이스 |
| 8 | nonlinviscohyper | 비선형 점탄성망 (Standard+Explicit, C3D8(H)(R)·CPE4(H)(R)) | △ | 네트워크형 (Bergström-Boyce망) |
| 9 | rateindepplast | Mises 등방/선형·다중선형·복합경화, 단열, Hill, 변형소성, D-P, Cap, Clay, Foam, 다공금속, M-C | △ (Mises 등방만) | 경화 3종이 1순위, 나머지는 비금속용 |
| 10 | viscocylinder | Prony 다항·자동증분·동적/정적 (CAX4I·CAX8R·CPE4I·CPE8R) | △ | 자동증분은 solver 영역 |
| 11 | ratedepplastexp | ratedep Mises + JC (Explicit 1요소 인장) | ✗ | JC는 별도 파일과 묶음 |
| 12 | annealtemp | 어닐링T 이상에서 경화·소성변형 리셋 (단일요소 가감) | ✗ | T-인프라 위 소규모 |
| 13 | tempplastic | T의존 Mises/D-P/Hill/Foam (+predefined T) | ✗ | T-인프라 + 소성 결합 |
| 14 | fieldplastic | 필드의존 소성 (同上) | ✗ | FIELD-인프라 + 소성 결합 |
| 15 | johnsoncook | JC (단축인장/압축/전단, T2D2~M3D4R) | ✗ | 변형률속도·온도항 |
| 16 | damage | 연성/JC/Hosford-Coulomb/전단·FLD/FLSD/M-K/MSFLD 개시+진전+요소삭제 | ✗ | 별도 대공사 — 후순위 |
| 17 | creep | Mises/Hill/D-P/Cap 크리프+소성연성 (거듭제곱·쌍곡사인·Anand) | ✗ | 시간적분 인프라와 묶음 |
| 18 | viscous | 2층 점소성 (Mises 완전+선형 이동경화, Hill) | ✗ | 소규모 (병렬 rheology) |
| 19 | concrete | smeared cracking (인장균열+압축경화) | ✗ | 비주력 — 최후순위 |
| 20 | concretedamaged | CDP (팽창각·편심·Kb·Kc·점성) | ✗ | 同上 |
| 21 | damping | 강성비례·대역제한 감쇠 (파동 전파 감쇠) | ✗ | 동적 영역 — solver와 묶음 |
| 22 | simpleexpansion | predefined T 기반 열팽창 (등방~초탄성 Ogden/AB/VdW) | ✗ | T-인프라 위 소규모, 범위 넓음 |
| 23 | plasticity-corrections | Neuber/Glinka 등가응력 보정 (후처리) | ✗ | 솔버 불필요 — 독립 소규모 |
| 24–26 | (열전달 3종) | 전도율·비열·간극전도 | — | 제외 |

## 3. 공통 인프라 (여러 파일을 관통 — 먼저 깔아야 할 것)

1. **T/FIELD 의존 물성 테이블**: `*ELASTIC, DEPENDENCIES=n` + `*TEMPERATURE`/`*FIELD` predefined 필드.
   파일 5·6·12·13·14·22가 전부 이 위에 있음. CPE4R·C3D8R 1요소 패턴 공유.
2. **어닐링 리셋** (12): 등가소성변형·경화 리셋. T-인프라 위 한 함수.
3. **열팽창 변형률** (22): `*EXPANSION` + predefined T. 응력적분과 직교 — 독립 추가 가능.
4. **속도·온도 의존 항** (11·15): JC형 `σ(εp, ε̇, T)`. 소성 귀환사상 확장.
5. **시간적분** (10·17): 크리프/점탄성용 강건 증분 (solver 영역과 협의).

## 4. 우선순위 제안 (디스플레이 관련도 순)

- P0: T/FIELD 테이블 인프라 (6개 파일 잠금해제) + simpleexpansion + annealtemp
- P1: rateindepplast 경화 3종 (선형이동·다중선형·복합) + JC + ratedep
- P2: Mullins + Bergström-Boyce (PSA 점탄성 고도화) + Ogden/Mooney 초탄성
- P3: creep + viscous 2층 + damping (시간·동적) + plasticity-corrections (후처리, 언제든)
- P4: damage + concrete 2종 (별도 대공사)

## 5. 벤치마크 준비 상태

- inp 실존: 필요 19종 중 18종 확인 (`SIMAINPRefResources\`).
  유일 결번 `hut.inp` — 변형 `mcoo* hut*.inp` 다수 실존으로 대체 가능.
- 패턴: 단위요소 (CPE4R/C3D8R 1개) + 단순하중 (단축·전단·정적분) → 러너 1종으로 다수 파일 커버 가능.
- temp_elastic.inp 구조 확인: 소자집합별 ELSET + `*INCLUDE temp_elastic_ef1.inp`
  (재료정의 분리). 우리 러너도 재료블록 분리 구조 권장.
- 실행은 보류 (사용자 지시): 구현+점검 완료 후 일괄 수행.
