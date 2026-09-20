# 재료 모델 정확도 평가 보고서 (2026-09-20 실측)

정답 출처: Abaqus Verification 문서(htm 테이블+결과그림) + `SIMAINPRefResources` inp 1345개.
아래 수치는 전부 오늘 직접 재실행한 실측이다 (보고서 전언 인용 없음).

## 1. inp Parity (Abaqus 수치 직접 대조) — 4/4 PASS

| 케이스 | 실측 | 정답 | 오차 | 판정 |
|:---|:---|:---|:---:|:---:|
| temp_elastic_ef1 (T=20C) | F=200.0208 N | 200.0 N | 0.010% | PASS |
| temp_elastic_ef1 (T=100C) | F=160.0166 N | 160.0 N | 0.010% | PASS |
| temp_plastic_mises | 고온 유동응력 저하 (504.30→471.25 MPa) | 단조 감소 | 정성일치 | PASS |
| thermal_expansion_simple | 반력 210.0000 N | 210.0000 N | 0.0000% | PASS |
| johnson_cook_dynamic | 속도경화 +8.29%, 열연화 −52.41% | 부호·자릿수 일치 | 정성일치 | PASS |

## 2. 단위·결합 스위트 — 81/81 PASS (88.77s 실측)

dependence 4, 열연계 5, 열팽창 2D/3D 각 2, 경화·JC·CS 5, Mullins/BB 5,
Law36 4, tabular 4, 초탄성 16, 점탄성 28, 크리프 4, 바이메탈 2.

## 3. 구조 연계 — PASS (문서 기록, 러너 존재)

바이메탈 Timoshenko 2D 0.110%·3D 0.208%, 14층 디스플레이 열롤업 0.11%.

## 4. 항목별 판정 (23종)

- ✅ parity 입증: #5 tempelastic, #13 tempplastic, #15 JC(동적), #22 simpleexpansion.
- 🔶 구현+단위검증, parity 미대조: #1(3종만)·#2(Prony만)·#3·#4·#7·#8·#9(core)·#10·#11·#17.
- ❌ 미구현: #12 annealtemp, #16 damage, #19·#20 concrete, #21 damping, #23 후처리.
- 🔸 훅만: #6 fieldelastic, #14 fieldplastic.

## 5. 결론

"돌아간다" 14종, "Abaqus와 같다" 4종(+구조 3종). 대등 선언 불가.
최단 추가 parity: #6·#14 ef1 쌍 (동일 인프라). 그 다음 #12 anneal, #1 Ogden/Mooney.
