# 🧱 WHT_DispFoldSolver: Element Release Notes (2D / 3D / Shell)

> **최초 작성일**: 2026-09-19
> **준수 표준**: Abaqus Element Library (§3.2), Bathe/Dvorkin, Simo/Rifai, Anti-Mocking Protocol
> **상태**: **CONDITIONAL READY (2D/3D 솔리드 Green, MITC4+ Green, MITC3+·CPE6M행 제외)**
> **상세**: `dev_log/area_analysis_element_20260919.md`

---

## 📌 목차
1. [개요 및 핵심 아키텍처](#1-개요-및-핵심-아키텍처)
2. [공개 API 및 사용 예제](#2-공개-api-및-사용-예제)
3. [상용 CAE와의 이론적 차이점 및 우월성](#3-상용-cae와의-이론적-차이점-및-우월성)
4. [조립 속도·JIT 분석](#4-조립-속도jit-분석)
5. [공인 벤치마크 검증 지표](#5-공인-벤치마크-검증-지표)
6. [변경 이력 및 주요 구현 파일 색인](#6-변경-이력-및-주요-구현-파일-색인)
7. [독립 재측정 기록](#7-독립-재측정-기록)

---

## 1. 개요 및 핵심 아키텍처

```
[JAX R&D] --검증--> [Numba 생산]     (3.15× 조립, 미구현분 JAX 폴백)
[2D 12원소: CPE4(I/IH/R/H/FBAR/CR) + CPE3/6/6M/8 + COH2D4] → solver2d (CR 자동승격 / 계면 점착 박리)
[3D 12원소: C3D8(I/IH/FBAR/CR/H/R) + C3D4(ANP) + C3D10(M) + C3D6] → solver3d
[Shell: SOLID_SHELL_CR (24DOF) + MITC4+ (수정済) + MITC3+ (보류)] → solver3d/solver_shell
게이트: tests/element_contract C1–C11 (회전-참조 스위프 없는 검증 인용 금지, §4.14)
```

생산 조합: `2D-Opt1` (`CPE4I + CPE4H`), `2D-Opt1-IH` (`CPE4I + CPE4IH`), `3D-Opt1` (`C3D8I + C3D8H`), `3D-Opt1-IH` (`C3D8I + C3D8IH`). 롤업 180° 모멘트 −0.4%.

## 2. 공개 API 및 사용 예제

```python
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
# dict형 element_type만이 커널 디스패치에 도달 (문자열 단독은 silent fallthrough, §4.2)
solver = DynamicSolver2D(mesh, materials={1: {...}, 2: {...}}, nlgeom=True)
# mesh 원소의 elem_type이 "CPE4I"/"CPE4H"/"CPE4IH"이면 CR 자동승격 (k=11/12/21)

# 계약 게이트 (전 원소):
python -m tests.element_contract.test_contract
```

## 3. 상용 CAE와의 이론적 차이점 및 우월성

| 항목 | Abaqus | 본 코드 | 비고 |
|:---|:---|:---|:---|
| CPE4I/C3D8I | EAS incompatible | 동일 + CR 외피 | 롤업 동점 |
| CPE4H/C3D8H | u-p hybrid | 동일 + CR 외피 | PSA 필수 |
| CPE4IH/C3D8IH | EAS + linear hybrid p | 동일(2단계 Schur 응축) + CR 외피 | 전단+체적 잠김 동시 소거 |
| F-bar | C3D8 volumetrics | C3D8_FBAR + CPE4_FBAR | nu 0.499 한계 공유 |
| MITC4 | S4 | MITC4+ (자작) | Scordelis 0.28% |
| CPE6M | 접촉용 삼각형 | CR 래핑済 | 굽힘행은 오수락 — 제외 |

## 4. 조립 속도·JIT 분석

- Numba 생산 조립이 JAX 대비 3.15× (마이크로벤치). 미구현 원소는 JAX 폴백.
- NeoHookean `pk2_tangent_voigt_batch` 추가로 lil 행렬 스캐터 제거 — 301s→204s (32%).
- `PyPardisoSolver` 싱글톤: 테스트 스위트 410s→152s (63%, §4.11). 호출당 재생성 금지.
- SMP NT1/2/4 평탄 (1500 DOF, PARDISO 직렬 지배). 병렬은 케이스 단위 프로세스가 유효.

## 5. 공인 벤치마크 검증 지표

| ID | 항목 | 허용치 | 실측 | 판정 |
|:---|:---|:---:|:---:|:---:|
| ELM-2D-01 | rank/탄젠트/패치 | 3/1e-14/1e-19 | 달성 | PASS |
| ELM-3D-01 | in-house+Abaqus 패치双重 | 전원 | 전원 | PASS |
| ELM-IH-01 | CPE4IH/C3D8IH 패치·회전불변·비압축굽힘 | <1e-12 / <1e-9 / <1e-8 | **달성 (10/10 Green)** | **PASS** |
| ELM-ROLL-01 | 롤업 M (Opt1) | ±1% | −0.4% | PASS |
| ELM-SHELL-01 | Scordelis-Lo MITC4+ 16x16 | <2% | **0.28%** | PASS |
| ELM-COH2D4-01| COH2D4 박리 및 에너지 보존 | <1e-4 | **< 1.2e-5** | **PASS** |
| ELM-CPE6M-01 | CPE6M 롤업행 | 정확도 수렴 | +34%/U−81% | **FAIL → 제외** |
| ELM-MITC3-01 | Scordelis-Lo MITC3+ | <2% | 84% | **FAIL → 보류** |

## 6. 변경 이력 및 주요 구현 파일 색인

- `dispsolver/element2d/cpe4ih_numba.py` — 2D 4-모드 EAS + 3-DOF 선형 혼합 압력(`CPE4IH`) 2단계 Schur 응축 커널 (2026-10-19)
- `dispsolver/element3d/c3d8ih_numba.py` — 3D 9-모드 EAS + 4-DOF 선형 혼합 압력(`C3D8IH`) 2단계 Schur 응축 커널 (2026-10-19)
- `tests/test_cpe4ih_c3d8ih.py` — `CPE4IH`/`C3D8IH` 10종 무결성 및 잠김 소거 검증 스위트 (10/10 All Green)
- `dispsolver/element2d/coh2d4_numba.py` — 2D 4절점 코히시브 요소 및 바이리니어 트랙션-분리 점착 박리 커널 (2026-10-18, P6 완료)
- `tests/test_coh2d4.py` — COH2D4 11종 무결성 검증 테스트 스위트 (11/11 All Green)
- `dispsolver/element_shell/mitc4_plus_numba.py` — 전단 관례 혼합 수정 (2026-09-19, untracked)
- `dispsolver/solver_shell/dynamic_shell.py` — 6-DOF 쉘 솔버 (untracked)
- `dispsolver/element2d/cr_wrapper_2d.py` — CR 래퍼 (`CPE4I_CR`, `CPE4H_CR`, `CPE4IH_CR`, `CPE4R_CR`, `CPE6M_CR`)
- `dispsolver/element3d/cr_wrapper_3d.py` — 3D CR 래퍼 (`C3D8I_CR`, `C3D8IH_CR`, `C3D8R_CR`, `SOLID_SHELL_CR`)
- `tests/element_contract/` — C1–C11 게이트 + `_XFAIL` 특성화 베이스라인 (`CPE4IH` C6/C7/C8 xfail 해소 완료)
- `benchmark_element/benchmark_shell_mitc.py` — Scordelis-Lo 러너 (untracked)

## 7. 독립 재측정 기록 (2026-09-19 ~ 2026-10-19)
- `CPE4IH` / `C3D8IH`: RBM 영공간(2D 3개, 3D 6개) `< 1e-12`, 왜곡 Irons 패치($\nu=0.4999$) `< 1e-12`, 회전 기준틀($0^\circ\sim 90^\circ$, AR=7) 굽힘 에너지 불변성 `< 1e-10`, 단일 층 비압축성($\nu=0.4999$) 세장 보 순수 굽힘 이론해 일치 오차 `1.36e-9`, 사다리꼴 왜곡 메쉬에서 `CPE4I`/`C3D8I` 체적 잠김(80% 강성 폭등) 및 `CPE4H`/`C3D8H` 전단 잠김 동시 해결 실증 (`tests/test_cpe4ih_c3d8ih.py` 10/10 Green, `tests/element_contract/test_contract.py -k CPE4IH` 9/9 Green).
- COH2D4: 강체 모드 불변성 1.45e-11, 방향성 유한차분 탄젠트 오차 < 1.2e-5, 순수 수직/전단 파괴 에너지 보존 오차 < 1.2e-5, DynamicSolver2D 박리 3-step 수렴 실증 (11/11 Green).
- MITC4+ 수정 전후: 회전-전단 0.25→1.00, 평판 수렴 0.2504→0.9997,
  Scordelis 62.72%→**0.28%**, 6RBM 유지. 프로브: `scratch/probe_shell_*.py`.
- CPE6M: 요소레벨 무죄 4건 vs Newton 경로 유죄 (§16). 인용 금지 항목: probe2 CPE4I 6.8×.
