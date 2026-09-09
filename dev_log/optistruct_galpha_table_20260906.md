# Altair OptiStruct Generalized-α (GALPHA) 0.1-Step Parameter Reference Table

**작성일자:** 2026년 9월 6일  
**문서 위치:** `dev_log/optistruct_galpha_table_20260906.md`  
**관련 모듈:** `dispsolver/solver/dynamic.py` (`INTEGRATION_MODES`)

---

## 1. 개요 (Overview)

본 문서는 Altair OptiStruct 비선형 임플리시트 동역학(Implicit Dynamic)에서 사용하는 **Generalized-$\alpha$ 시간 적분법 (`PARAM, GALPHA`)**의 고주파 스펙트럼 반지름 $\rho_\infty$ (0.1 단계, 1.0 ~ 0.0)에 따른 유효 뉴마크 및 적분 파라미터 매핑 레퍼런스입니다.

---

## 2. 이론적 공식 (Chung & Hulbert 1993)

OptiStruct의 `PARAM, GALPHA` 값 $\rho_\infty \in [0, 1]$로부터 유도되는 뉴마크 및 힐버-휴즈 적분 파라미터 공식:

$$\alpha_m = \frac{2\rho_\infty - 1}{\rho_\infty + 1}, \quad \alpha_f = \frac{\rho_\infty}{\rho_\infty + 1}$$

$$\beta = \frac{1}{(1 + \rho_\infty)^2}, \quad \gamma = \frac{3 - \rho_\infty}{2(1 + \rho_\infty)}$$

---

## 3. OptiStruct GALPHA 0.1-Step 전체 매칭표 (`1.0` ~ `0.0`)

| OptiStruct (GALPHA) | alpha_m | alpha_f | beta | gamma | 우리 솔버 모드 (`INTEGRATION_MODES`) | 수치 감쇄 및 물리적 특성 설명 |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| GALPHA = 1.0 | 0.5000 | 0.5000 | 0.2500 | 0.5000 | `transient` | 수치 감쇄 없음 (Pure Newmark, 에너지 완전 보존) |
| GALPHA = 0.9 | 0.4211 | 0.4737 | 0.2770 | 0.5526 | `generalized-0.9` | 초경량 고주파 감쇄 |
| **GALPHA = 0.8 (기본값)** | **0.3333** | **0.4444** | **0.3086** | **0.6111** | **`generalized-0.8`** | **OptiStruct 표준 기본값 (저주파 오차 최소화)** |
| GALPHA = 0.7 | 0.2353 | 0.4118 | 0.3460 | 0.6765 | `generalized-0.7` | 경량-중등도 고주파 감쇄 |
| GALPHA = 0.6 | 0.1250 | 0.3750 | 0.3906 | 0.7500 | `generalized-0.6` | 중등도 수치 감쇄 |
| GALPHA = 0.5 | 0.0000 | 0.3333 | 0.4444 | 0.8333 | `generalized-0.5` | 중등도-강한 고주파 진동 억제 |
| GALPHA = 0.4 | -0.1429 | 0.2857 | 0.5102 | 0.9286 | `generalized-0.4` | 강한 수치 감쇄 |
| GALPHA = 0.3 | -0.3077 | 0.2308 | 0.5917 | 1.0385 | `generalized-0.3` | 고비선형 좌굴/파동 강제 억제 |
| GALPHA = 0.2 | -0.5000 | 0.1667 | 0.6944 | 1.1667 | `generalized-0.2` | 초강력 수치 감쇄 |
| GALPHA = 0.1 | -0.7273 | 0.0909 | 0.8264 | 1.3182 | `generalized-0.1` | 극심한 수치 노이즈 억제 |
| GALPHA = 0.0 | -1.0000 | 0.0000 | 1.0000 | 1.5000 | 이론적 극한 | 최대 수치 감쇄 한계 (Asymptotic Annihilation) |

---

## 4. OptiStruct Quasi-Static vs GALPHA = 0.0 비교 분석

### ❓ 질문: OptiStruct에서 Quasi-Static 설정이면 GALPHA = 0 으로 보면 되는가?
### 💡 답: **아닙니다! 두 개념은 물리적으로 완전히 다릅니다.**

| 구분 | OptiStruct Quasi-Static (`quasistatic`) | GALPHA = 0.0 (`generalized-0.0`) |
| :--- | :--- | :--- |
| **질량 행렬 ($\mathbf{M}$)** | **완전 제거 ($\mathbf{M} = \mathbf{0}$)** | **유지 및 활성화 ($\mathbf{M} \ne \mathbf{0}$)** |
| **관성력 ($\mathbf{M}\mathbf{a}$)** | 완전히 비활성화됨 | 방정식에 동적 포함됨 |
| **운동 에너지** | $E_{\text{kin}} = 0$ (순수 정적 평형) | 관성 운동 작용 (고주파 모드만 1 step 내 소멸) |
| **시간 증분 ($\Delta t$)** | 하중 램프 및 점탄성 이완 지표로만 작용 | 유효 질량 강도 $\frac{1}{\beta \Delta t^2}\mathbf{M}$으로 결합됨 |

---

## 5. 솔버 확장 구현 사항 (`dispsolver/solver/dynamic.py`)

`INTEGRATION_MODES`에 `generalized-0.9`부터 `generalized-0.1`까지의 0.1-step 파라미터를 완전하게 명시적 등록하여, 사용자가 원하는 스펙트럼 반지름을 바로 선택할 수 있도록 지원합니다.

```python
INTEGRATION_MODES = {
    "generalized-0.9": _generalized_alpha(0.9),
    "generalized-0.8": _generalized_alpha(0.8),
    "generalized-0.7": _generalized_alpha(0.7),
    "generalized-0.6": _generalized_alpha(0.6),
    "generalized-0.5": _generalized_alpha(0.5),
    "generalized-0.4": _generalized_alpha(0.4),
    "generalized-0.3": _generalized_alpha(0.3),
    "generalized-0.2": _generalized_alpha(0.2),
    "generalized-0.1": _generalized_alpha(0.1),
}
```
