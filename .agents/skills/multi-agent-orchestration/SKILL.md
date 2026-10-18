---
name: multi-agent-orchestration
description: 3-Tier Multi-Agent Orchestration Framework for WHT_DispFoldSolver (Antigravity Lead Orchestrator + Claude CLI Opus + OpenCode CLI Space Bunny + Anti-Mocking Verification). Use whenever the user mentions multi-agent, 멀티에이전트, 서브에이전트 파견, or collaborative agent workflows.
---

# 🤖 3-Tier Multi-Agent Orchestration Framework (`multi-agent-orchestration`)

본 스킬은 **WHT_DispFoldSolver** 프로젝트 전반에서 복잡한 유한요소 수치해석, 비선형 연속체역학, 대규모 코드 리팩토링, 자동화된 테스트 검증을 수행할 때 **3단계(Tiered) 멀티 에이전트 분업 및 오케스트레이션 전략**을 표준화하여 일관되게 적용하기 위한 실전 지침서입니다.

---

## 👥 1. 3단계 에이전트 역할 및 책임 분담 (R&R)

```
┌────────────────────────────────────────────────────────────────────────┐
│ 🧠 [Tier 1: 심층 이론/수치역학 최고 지능 자문단]                       │
│  - 엔진: Claude CLI (Opus)                                            │
│  - 커맨드: claude -p --model opus                                      │
│  - 담당 업무:                                                          │
│    * 비선형 대변형 티어드롭 엘라스티카 BVP 수식 유도 & 닫힌해 해석      │
│    * 복합재 다층 적층판 계면 전단-슬립(Shear-lag) 역학 이론 평가       │
│    * 비선형 솔버 수렴 교착(Stalling) 및 미완화 잔차의 루트 원인 규명  │
│    * 접촉 역학(Dual Mortar 쌍직교 응축, Nitsche 안정화) 대수 정식화   │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ (심층 자문 & 수식 증명)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 🏛️ [Tier 2: 총괄 기획 & TDD 통합 오케스트레이터]                       │
│  - 엔진: Antigravity Lead Agent (Gemini 3.1 Pro / Flash)              │
│  - 담당 업무:                                                          │
│    * PRD 요구사항 정의 및 TDD 경계 인터페이스(Seams) 설계              │
│    * 서브에이전트 태스크 생성, 조율, 메시징, 라이프사이클 관리        │
│    * 핵심 소스 코드 구현 및 2차 독립 무결성 전수 검증 (Anti-Mocking)  │
│    * Abaqus API, VTKHDF v2.2 DB, PySide6 GUI 통합 및 Git 형상 관리    │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ (기계적 반복 작업 파견)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ ⚡ [Tier 3: 경량/고속 무료 보조 에이전트]                               │
│  - 엔진: OpenCode CLI (opencode/space-bunny-free) / Antigravity Flash  │
│  - 커맨드: opencode run --model opencode/space-bunny-free --auto      │
│  - 담당 업무:                                                          │
│    * pytest 단위/통합 테스트 반복 실행 및 요약 보고 (노이즈 격리)      │
│    * 해석 완료 후 results/D.../run.log 파싱 및 수렴성 지표 요약        │
│    * 정적 린트(Lint), 미사용 import, docstring 누락 검사               │
│    * 토큰 소모 제로화(Free Tier)를 통한 비용 최적화                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 2. 에이전트 호출 및 연동 프로토콜 (Execution Recipes)

### ① Tier 1: Claude Opus 심층 자문 호출
Claude CLI는 `stdin` 파이프를 통해 프롬프트를 전달하고 비대화형(`-p`)으로 실행합니다:

```python
import subprocess

def query_claude_opus(prompt: str) -> str:
    """Invokes Claude Opus for deep theoretical and mathematical consulting."""
    cmd = ["claude", "-p", "--model", "opus"]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        shell=True,
    )
    stdout, stderr = process.communicate(input=prompt)
    if process.returncode != 0 and stderr:
        return f"[Claude Opus Error]: {stderr.strip()}"
    return stdout.strip()
```

### ② Tier 3: OpenCode Space Bunny 무료 보조 에이전트 호출
OpenCode CLI는 `--auto` 플래그로 비대화형 자율 실행을 허용하며 `stdin` 파이프로 작업을 지시합니다:

```python
import subprocess

def dispatch_space_bunny_task(task_instruction: str) -> str:
    """Dispatches lightweight tasks (testing, log parsing) to space-bunny-free."""
    cmd = ["opencode", "run", "--model", "opencode/space-bunny-free", "--auto"]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        shell=True,
    )
    stdout, stderr = process.communicate(input=task_instruction)
    return stdout.strip()
```

*(참고: OpenCode API 키가 만료된 환경에서는 Antigravity 내장 `invoke_subagent`의 `Model="flash_lite"` 또는 `Model="flash"` 서브에이전트가 완벽한 대안으로 자동 동작합니다.)*

---

## 🚫 3. 수치해석 무결성 및 안티 모킹 원칙 (Anti-Mocking Protocol)

멀티 에이전트 협업 시 하위 에이전트가 임의로 작성한 코드에 대해 **반드시 다음 3대 가드레일을 강제**합니다:

1. **난수 및 가짜 더미(Dummy) 코드 원천 엄단**:
   - `np.random`, 임의의 스케일링 상수, 하드코딩된 더미 반환값 절대 금지.
   - 모든 유한요소 수식은 공인 원전(Bathe, Belytschko, Simo 등)의 수학적 정식화에 입각하여 한 줄 한 줄 엄밀하게 유도·구현.
2. **서브에이전트 산출물 자가 승인 금지 및 2차 독립 검증 (Independent Verification)**:
   - 서브에이전트가 "테스트 통과 / 구현 완료"라고 보고하더라도 메인 에이전트는 작성된 소스 코드를 반드시 직접 열어(`view_file`) 난수/더미 유무를 라인 단위로 전수 검사.
   - 벤치마크 테스트 및 물리적 수렴성을 메인 프로세스에서 직접 독립 실행하여 확인한 후에만 최종 완료 선언.
3. **안드레 카파시(Karpathy) 4대 원칙 준수**:
   - **Think Before Coding**: 가정을 명시하고 수학적/수치적 한계를 투명하게 공개.
   - **Simplicity First**: 단일 사용 목적의 과도한 추상화를 배제하고 직관적인 최소 코드 작성.
   - **Surgical Changes**: 건드려야 할 부분만 최소한으로 수정.
   - **Goal-Driven Execution**: "작동하는 것 같다"는 주관적 평가 배제, 정량적 반증 가능 지표(오차 < 1e-5, 잔차 < tol)로 검증 루프 수행.

---

## 🧹 4. 프로세스 수명주기 및 자원 정리 수칙 (Task Kill Protocol)

- 작업이나 테스트 완료 시, 백그라운드에 남아 있는 고아 프로세스를 방지하기 위해 반드시 자원을 정리합니다:
  - Antigravity 서브에이전트: `manage_subagents`의 `kill` 또는 `kill_all` 호출.
  - 외부 CLI 프로세스: 필요 시 `taskkill /F /IM opencode.exe` 또는 `taskkill /F /IM claude.exe` 수행.

---

## 📌 5. 트리거 및 참조 규칙

대화 중 사용자가 아래의 키워드를 언급하거나 관련 요청을 할 때 본 스킬의 3-Tier 아키텍처를 즉시 참조하여 역할을 분담합니다:
- `"멀티에이전트"`, `"multi-agent"`, `"서브에이전트 파견"`
- `"Opus와 상의"`, `"Opus 자문"`, `"Claude CLI"`
- `"Space Bunny"`, `"OpenCode"`, `"무료 모델로 테스트"`
- `"TDD 기반 개발"`, `"PRD 기반 개발"`
