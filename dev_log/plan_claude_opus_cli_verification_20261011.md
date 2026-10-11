# [Implementation Plan] Claude Code CLI (Opus 모델) 독립 수치역학 교차 검증 계획서

## 1. 개요 및 목적 (Goal Description)
본 계획은 사용자의 지시(**"claude code cli로 이 분석을 요청하라. opus 모델 사용 요청으로."**)에 따라, Anthropic의 최상위 추론 모델인 **Claude Opus (Claude 3 / 3.5 Opus)**를 Claude Code CLI를 통해 호출하여, 다층 디스플레이 180도 폴딩(`ex12`)의 하중 및 힌지 모멘트 이론 정합성에 대한 **제3자 독립 교차 검증(Third-Party Independent Verification)**을 수행하기 위한 실행 계획입니다.

```
 [Antigravity & FEA 서브에이전트 1차 분석]
  - D_slip = 1.7988 N·mm², D_mono = 154.76 N·mm²
  - Elastica BVP: M_pivot = 2.4503 N·mm, Fx = 0.2274 N
  - FEM 실측: M_pivot = 2.5265 N·mm, Fx = 0.2881 N (오차 +3.11%)
       │
       ▼
 [Claude Code CLI (Opus 모델) 파견]
  - 명령어: Get-Content prompt.txt | claude -p --model opus
  - 독립된 프롬프트로 고체역학 닫힌 해 및 비선형 BVP 풀이 의뢰
       │
       ▼
 [독립 교차 검증 (Cross-Verification)]
  - Opus 모델의 독자 계산 결과와 Antigravity/FEA 에이전트 결과 1:1 대조
  - 불일치 여부 확인 및 최종 권위 있는 수치역학 판정문 확보
```

---

## 2. 사용자 검토 요구사항 (User Review Required)

> [!IMPORTANT]
> **Claude Code CLI 환경 확인 완료:**
> - `claude.exe` 실행 경로: `C:\Users\GOODMAN\.local\bin\claude.exe` (Version 2.1.296.0)
> - `--model opus` 옵션 정상 응답 확인 (`Pong. Ready.` 수신 완료)
> - 프롬프트 전달 방식: 복잡한 수식과 기하 파라미터의 PowerShell 이스케이프 왜곡을 방지하기 위해 `dev_log/prompt_claude_opus_verification_20261011.txt` 파일로 작성 후 파이프라인 전달(`Get-Content ... | claude -p --model opus`)을 적용합니다.

---

## 3. Claude Opus에 전달할 의뢰 프롬프트 설계 (Prompt Specification)

Claude Opus에 전달할 전문 질의문([`dev_log/prompt_claude_opus_verification_20261011.txt`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/prompt_claude_opus_verification_20261011.txt)):

1. **모델 파라미터 (단위: mm, N, MPa, 단위 폭 b = 1.0 mm):**
   - 14개 적층판 (평면 변형률):
     * 7개 PSA ($t = 0.03\,\text{mm}, E = 0.05\,\text{MPa}, \nu = 0.499, G_0 = 16.7\,\text{kPa}$)
     * 6개 PET ($t = 0.06\,\text{mm}, E = 4000\,\text{MPa}, \nu = 0.30, \sigma_{y0} = 80\,\text{MPa}, H = 400\,\text{MPa}$)
     * 1개 GLASS ($t = 0.06\,\text{mm}, E = 70000\,\text{MPa}, \nu = 0.22$)
   - 기하 조건:
     * 자유 힌지 스팬: $L = 15.0\,\text{mm}$ ($L_{\text{half}} = 7.5\,\text{mm}$)
     * 힌지 구간 하부 1층 PSA는 테이프 보이드로 제외됨 (활성 13개 층)
     * 회전 피벗: $x = \pm 1.4\,\text{mm}, y = 0.0\,\text{mm}$ (넥 갭 $2.8\,\text{mm}$)
     * 회전각: 양측 각 $\pm 90^\circ$ (총 $180^\circ$ 굽힘)
2. **Opus 모델에 요구하는 5대 계산 과제:**
   - **과제 1:** 평면 변형률 굽힘 강성 $D_{\text{slip}}$ 및 일체형 강성 $D_{\text{mono}}$, 중립축 $y_{\text{NA}}$, 강성비 $D_{\text{mono}}/D_{\text{slip}}$ 독립 유도.
   - **과제 2:** 대변형 티어드롭 엘라스티카 지배방정식($D \theta''(s) + F_x \sin\theta = 0$)의 2점 경계치 문제(BVP) 정해 유도 ($F_x, M_{\text{apex}}, R_{\text{apex}}, \Delta Y, M_{\text{pivot}}$).
   - **과제 3:** FEM 실측값($M_{\text{pivot}} = 2.5265\,\text{N}\cdot\text{mm}$, $F_x = 0.2881\,\text{N}$, $R_{\text{apex}} = 1.9059\,\text{mm}$, $\Delta Y = 6.3154\,\text{mm}$)과의 1:1 오차율 및 물리적 타당성 평가.
   - **과제 4:** PSA의 유한 전단($\gamma \approx 500\sim 1000\%$)에 의한 수평 장력 증가 현상($+0.06\,\text{N}$) 해석.
   - **과제 5:** PET 층의 최대 표면 굽힘 응력 및 소성 항복($80\,\text{MPa}$) 도달 여부 판정.

---

## 4. 변경 계획 (Proposed Changes)

### 4.1 프롬프트 파일 생성
#### [NEW] `dev_log/prompt_claude_opus_verification_20261011.txt`
- Opus에 전달할 정밀 수치역학 질의서 전문 저장.

### 4.2 실행 및 결과 수집
#### [NEW] `dev_log/claude_opus_analysis_report_20261011.md`
- Claude Opus의 원문 분석 응답 저장 및 Antigravity 결과와의 1:1 비교 섹션 작성.

### 4.3 영구 교훈 보강
#### [MODIFY] `LESSONS_LEARNED.md`
- **Section 12 보강**: "Claude Opus 모델과의 3자 독립 교차 검증 일치성 결과" 추가 등재.

---

## 5. 실행 및 검증 절차 (Verification Plan)

### 터미널 실행 명령
```powershell
Get-Content dev_log/prompt_claude_opus_verification_20261011.txt | claude -p --model opus > dev_log/claude_opus_analysis_report_20261011.md
```
- 성공 기준:
  1. Claude Opus의 계산 결과가 정상 반환되고 종료 코드 0 달성.
  2. Opus가 도출한 $D_{\text{slip}}$, $M_{\text{pivot}}$, $F_x$가 기존 수치해석 결과(2.45 N·mm)와 비트 일치/오차 1% 이내로 상호 검증됨.
  3. Opus의 최종 판정문이 "체적 잠김 없음 및 참 물리량(True Mechanics) 부합"으로 수렴하는지 확인.
