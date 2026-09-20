---
name: release-docs
description: >
  Update release notes and per-area analysis docs on request.
  Use when user says "릴리즈 문서 업데이트", "release update",
  "영역별 분석 업데이트", or asks to refresh release_notes_contact.md
  / dev_log/area_analysis_*.md living docs with latest measurements.
---

# 📦 Release Docs Updater (project-local)

> NOTE (mirror): this file is mirrored to the Gemini global skill
> `~/.gemini/skills/release-docs/SKILL.md` (copy — symlink needs privilege).
> After editing here, re-copy there to keep them in sync.

Single source rule: content lives ONLY in `GEMINI.md`
("📦 릴리즈 및 영역별 분석 인덱스") + root
`release_notes_contact/constraint/solver/element.md` +
`dev_log/area_analysis_*.md`. Never fork into root `AGENTS.md`
(it holds only a pointer, §6).

## Workflow (on "릴리즈 문서 업데이트" request)

1. **Read index**: `GEMINI.md` index section → list linked docs. Check links alive.
2. **Collect changes** since last doc touch (`git log --oneline -- dispsolver/ tests/ benchmark_element/`):
   - code changes per area (constraint / solver / element / contact)
   - new test/benchmark numbers from JSON results + last runs
3. **Verify before writing** (no agent-word trust):
   - cheap checks: re-run the specific pytest/benchmark behind a claimed number
   - expensive checks (>5 min): mark claim `unverified` instead of running
   - stale "verified" comments falsified by code moves → rewrite or delete
4. **Update each area doc**:
   - move resolved items out of Open, add new Open items with owner evidence
   - numbers only with date + command that produced them
   - keep verified-vs-open split strict
5. **Update release notes** (per-area, append-only, keep history):
   `release_notes_contact/constraint/solver/element.md`
   (root one-line status). Keep old entries; update Status line + date.
   Every area file MUST match the detail level of `release_notes_contact.md`
   (§Template below) — one-line status alone is not acceptable.

## Release-note template (all areas)

Mirror `release_notes_contact.md` structure, adapted per area:

1. Header: 최초 작성일 / 담당 / 준수 표준 / 상태 (테스트 n/m + 날짜)
2. 목차 (6 sections)
3. 개요 + ASCII 아키텍처 다이어그램 (계층·데이터 흐름)
4. 공개 API + 사용 예제 (복붙 가능 코드, import부터 실행까지)
5. 이론적 차이점 및 우월성 (상용 CAE 대비 표 — 근거 수치 포함)
6. 속도 병목 분석 + 달성 기법 (프로파일 수치, before→after)
7. 공인 벤치마크 표 (ID·규격·허용치·실측·판정) + 로그 경로
8. 변경 이력 + 주요 구현 파일 색인 (파일 링크)
9. 독립 재측정 기록 (날짜·명령·결과 — trial-run 산출물)

Area adaptations:
- constraint: §4 → tie/RBE2 강성·간극 수치, §5 → tie gap·Hertz·마찰 스위트
- solver: §4 → Newton 분기 분포·worst_r 스케일, §5 → rollup 게이트·verification 표
- element: §4 → 조립 속도·JIT 컴파일, §5 → 계약 스위트·Scordelis-Lo·롤업 모멘트 표
6. **Report**: file-by-file diff summary. No commit unless explicitly asked.

## Forbidden

- Editing root `AGENTS.md` content beyond its §6 pointer.
- Copying numbers from subagent reports without re-running the check.
- `np.random`/dummy values anywhere in verification code (Anti-Mocking).
- Marking a row green that was measured under another red (masked-contract rule).
