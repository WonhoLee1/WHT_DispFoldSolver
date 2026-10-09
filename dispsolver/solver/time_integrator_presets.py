"""
time_integrator_presets.py
==========================
2D 및 3D 비선형 유한요소 해석용 통합 시간적분기(Time Integrator) 프리셋 및 파라미터 제어 모듈.

이 모듈은 2D 및 3D 동역학/정역학 솔버(`DynamicSolver2D`, `DynamicSolver3D`, `DynamicSolver`)에서
공통으로 사용하는 시간적분 알고리즘 옵션 및 프리셋을 단일화하고, 개별 수치 상수를 매뉴얼로 제어할 수 있는
유연한 인터페이스를 제공합니다.

========================================================================================================
 1. 지원하는 시간적분기(time_integrator) 종류 및 물리적 특징 요약
========================================================================================================

1) "quasistatic" / "static" (준정적 해석 모드 - 기본값)
   - 개념: 관성력(Mass * Acceleration) 및 감쇠력을 배제하고 완벽한 정적 평형 방정식 K(u) * u = f_ext 을 해석함.
   - 활용: 고속 충격이나 진동이 없고 시간 의존성 재질(점탄성 PSA 등)의 크리프/응력완화와 회전 구동 굽힘만 있는 디스플레이 폴딩.
   - 수치 상수: alpha_hht = 0.0, beta = 0.25, gamma = 0.5, static_mode = True.

2) "emcs" (Energy-Momentum Conserving Scheme - 에너지-운동량 보존 동역학)
   - 개념: 대변형/대회전 하이퍼엘라스틱 문제에서 수천 시간 스텝 구동 시에도 전산 에너지 폭발이나 인공 감쇠 없이
           이산 미분(Discrete Derivative, Stiemer 2006)을 통해 총 에너지(E_kin + E_strain)와 선형/각운동량을 기계 정밀도로 보존함.
   - 활용: 무감쇠 자유 진동, 보존계 회전 운동, 수치적 안정성이 극도로 중요한 대규모 동역학 해석.
   - 수치 상수: 중간점(Mid-point t_{n+1/2}) 보간 기반.

3) "transient" / "newmark" (Newmark-β 사다리꼴 적분기 - 무감쇠 동역학)
   - 개념: classical Newmark-beta 무조건 안정 trapezoidal rule (beta = 0.25, gamma = 0.5, alpha = 0.0).
   - 특징: 2차 정밀도(2nd-order accuracy), 에너지 보존적이나 고주파 링잉(high-frequency ringing)을 인공 감쇠시키지 않음.
   - 활용: 과도 응답 해석, 고주파 파동 전파.

4) "moderate-1" ~ "moderate-5" / "mod1" ~ "mod5" (HHT-α 수치 감쇠 적분기)
   - 개념: Hilber-Hughes-Taylor (1977) alpha-방법. 고주파 수치 링잉(spurious high-frequency noise)을 제어하기 위해
           내력을 중간 지점 (1+alpha)*f_int_{n+1} - alpha*f_int_n 으로 가중 처리함.
   - 관계식: beta = (1 - alpha)^2 / 4,  gamma = 0.5 - alpha  (alpha in [-1/3, 0])
   - 프리셋 세부 분류:
     * "moderate-1" (mod1): alpha_hht = -0.05 (초미세 HF 감쇠, beta=0.2756, gamma=0.55)
     * "moderate-2" (mod2): alpha_hht = -0.10 (미세 HF 감쇠, beta=0.3025, gamma=0.60)
     * "moderate-3" (mod3): alpha_hht = -0.15 (표준 HF 감쇠 - Abaqus/Standard 기본 추천값, beta=0.3306, gamma=0.65)
     * "moderate-4" (mod4): alpha_hht = -0.22 (강한 HF 감쇠, 접촉 찰나 링잉 제거, beta=0.3721, gamma=0.72)
     * "moderate-5" (mod5): alpha_hht = -0.30 (최대 HF 감쇠 - 이론적 한계 -1/3 근접, beta=0.4225, gamma=0.80)

5) "generalized-0.1" ~ "generalized-0.9" / "optistruct_galpha" (Generalized-α 일반화 적분기)
   - 개념: Chung & Hulbert (1993) 일반화 alpha 알고리즘. 무한대 주파수 분산 비율 rho_inf in [0, 1] 로 조절.
   - 관계식:
     alpha_m = (2*rho_inf - 1) / (rho_inf + 1)
     alpha_f = rho_inf / (rho_inf + 1)
     beta    = 0.25 * (1 - alpha_m + alpha_f)^2
     gamma   = 0.5 - alpha_m + alpha_f
   - 프리셋 세부 분류:
     * "generalized-0.8" / "optistruct_galpha": OptiStruct 및 대형 CAE 기본값 (rho_inf = 0.8)

========================================================================================================
 2. 개별 수치 상수 매뉴얼 제어 (Manual Parameter Overrides)
========================================================================================================
프리셋을 지정한 후에도 사용자가 직접 다음 개별 상수를 매뉴얼로 전달하면 해당 값이 수술적으로 오버라이드됩니다:
   - alpha_hht (또는 alpha): HHT-alpha 수치 감쇠 계수. (beta, gamma가 미지정이면 자동으로 수식에 맞춰 재계산됨)
   - beta: Newmark beta 계수 (예: 0.25)
   - gamma: Newmark gamma 계수 (예: 0.50)
   - alpha_m: Generalized-alpha 질량 가중치
   - alpha_f: Generalized-alpha 외력/내력 가중치
   - rho_inf: Generalized-alpha 무한대 스펙트럼 반경
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union


@dataclass
class TimeIntegratorConfig:
    """통합 시간적분기 파라미터 구조체."""
    name: str
    is_quasistatic: bool
    is_emcs: bool
    alpha_hht: float
    beta: float
    gamma: float
    alpha_m: float = 0.0
    alpha_f: float = 0.0
    rho_inf: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """설정 상태를 딕셔너리로 반환."""
        return {
            "name": self.name,
            "is_quasistatic": self.is_quasistatic,
            "is_emcs": self.is_emcs,
            "alpha_hht": self.alpha_hht,
            "beta": self.beta,
            "gamma": self.gamma,
            "alpha_m": self.alpha_m,
            "alpha_f": self.alpha_f,
            "rho_inf": self.rho_inf,
        }


def _calc_generalized_alpha(rho_inf: float) -> Dict[str, float]:
    """Generalized-alpha (Chung & Hulbert 1993) 파라미터 계산."""
    rho = float(rho_inf)
    alpha_m = (2.0 * rho - 1.0) / (rho + 1.0)
    alpha_f = rho / (rho + 1.0)
    beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2
    gamma = 0.5 - alpha_m + alpha_f
    alpha_hht = alpha_f
    return {
        "alpha_m": alpha_m,
        "alpha_f": alpha_f,
        "beta": beta,
        "gamma": gamma,
        "alpha_hht": alpha_hht,
        "rho_inf": rho,
    }


PRESETS: Dict[str, Dict[str, Any]] = {
    # 1. 준정적 평형 해석 (Quasi-Static)
    "quasistatic": {"is_quasistatic": True, "is_emcs": False, "alpha_hht": 0.0, "beta": 0.25, "gamma": 0.5},
    "static": {"is_quasistatic": True, "is_emcs": False, "alpha_hht": 0.0, "beta": 0.25, "gamma": 0.5},
    
    # 2. 에너지-운동량 보존 해석 (EMCS)
    "emcs": {"is_quasistatic": False, "is_emcs": True, "alpha_hht": 0.0, "beta": 0.25, "gamma": 0.5},
    
    # 3. 무감쇠 과도 동역학 (Newmark Trapezoidal)
    "transient": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": 0.0, "beta": 0.25, "gamma": 0.5},
    "newmark": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": 0.0, "beta": 0.25, "gamma": 0.5},
    
    # 4. HHT-alpha 수치 감쇠 적분기 (Hilber-Hughes-Taylor)
    "moderate-1": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.05, "beta": (1.0 - (-0.05))**2 / 4.0, "gamma": 0.5 - (-0.05)},
    "mod1": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.05, "beta": (1.0 - (-0.05))**2 / 4.0, "gamma": 0.5 - (-0.05)},
    "moderate-2": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.10, "beta": (1.0 - (-0.10))**2 / 4.0, "gamma": 0.5 - (-0.10)},
    "mod2": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.10, "beta": (1.0 - (-0.10))**2 / 4.0, "gamma": 0.5 - (-0.10)},
    "moderate-3": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.15, "beta": (1.0 - (-0.15))**2 / 4.0, "gamma": 0.5 - (-0.15)},
    "mod3": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.15, "beta": (1.0 - (-0.15))**2 / 4.0, "gamma": 0.5 - (-0.15)},
    "moderate-4": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.22, "beta": (1.0 - (-0.22))**2 / 4.0, "gamma": 0.5 - (-0.22)},
    "mod4": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.22, "beta": (1.0 - (-0.22))**2 / 4.0, "gamma": 0.5 - (-0.22)},
    "moderate-5": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.30, "beta": (1.0 - (-0.30))**2 / 4.0, "gamma": 0.5 - (-0.30)},
    "mod5": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.30, "beta": (1.0 - (-0.30))**2 / 4.0, "gamma": 0.5 - (-0.30)},
    
    # 5. Generalized-alpha 적분기
    "optistruct_galpha": {**{"is_quasistatic": False, "is_emcs": False}, **_calc_generalized_alpha(0.8)},

    # 6. Abaqus/Standard *DYNAMIC operators (Analysis Guide "Implicit dynamic analysis", Table 1)
    "hht-tf": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.05, "beta": 0.275625, "gamma": 0.55},
    "hht-md": {"is_quasistatic": False, "is_emcs": False, "alpha_hht": -0.41421,
               "beta": (1.0 + 0.41421) ** 2 / 4.0, "gamma": 0.5 + 0.41421},
}


def resolve_time_integrator(
    time_integrator: Optional[str] = "quasistatic",
    alpha_hht: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    alpha_m: Optional[float] = None,
    alpha_f: Optional[float] = None,
    rho_inf: Optional[float] = None,
    static_mode: Optional[bool] = None,
) -> TimeIntegratorConfig:
    """프리셋 명칭 검색 및 개별 오버라이드 상수를 결합하여 최종 시간적분기 파라미터를 해소(Resolve)함.

    Parameters
    ----------
    time_integrator : str, optional
        시간적분기 프리셋 이름 ("quasistatic", "emcs", "transient", "moderate-1".."moderate-5", "generalized-X", "optistruct_galpha").
    alpha_hht, alpha : float, optional
        개별 매뉴얼 HHT-alpha 계수 (예: -0.05, -0.15). beta/gamma 미지정 시 유도 수식에 의해 자동 계산.
    beta : float, optional
        개별 매뉴얼 Newmark beta 계수 (예: 0.25). 지정 시 프리셋 값을 오버라이드.
    gamma : float, optional
        개별 매뉴얼 Newmark gamma 계수 (예: 0.5). 지정 시 프리셋 값을 오버라이드.
    alpha_m : float, optional
        개별 매뉴얼 Generalized-alpha 질량 파라미터.
    alpha_f : float, optional
        개별 매뉴얼 Generalized-alpha 외력 파라미터.
    rho_inf : float, optional
        개별 매뉴얼 Generalized-alpha 무한대 스펙트럼 반경 (0.0~1.0).
    static_mode : bool, optional
        개별 매뉴얼 준정적 모드 강제 지정 여부.

    Returns
    -------
    TimeIntegratorConfig
        검증 및 해소 완료된 시간적분기 파라미터 구조체.
    """
    raw_name = str(time_integrator or "quasistatic").lower().strip()

    is_qs = False
    is_emcs = False
    cur_alpha_hht = 0.0
    cur_beta = 0.25
    cur_gamma = 0.5
    cur_alpha_m = 0.0
    cur_alpha_f = 0.0
    cur_rho_inf = rho_inf

    # 1. 프리셋 및 동적 문자열 패턴 분석
    if raw_name in PRESETS:
        p = PRESETS[raw_name]
        is_qs = p.get("is_quasistatic", False)
        is_emcs = p.get("is_emcs", False)
        cur_alpha_hht = float(p.get("alpha_hht", 0.0))
        cur_beta = float(p.get("beta", 0.25))
        cur_gamma = float(p.get("gamma", 0.5))
        cur_alpha_m = float(p.get("alpha_m", 0.0))
        cur_alpha_f = float(p.get("alpha_f", 0.0))
        cur_rho_inf = p.get("rho_inf", None)
    elif raw_name.startswith("generalized-") or raw_name.startswith("galpha-"):
        parts = raw_name.split("-", 1)
        try:
            val = float(parts[1])
            g = _calc_generalized_alpha(val)
            cur_alpha_hht = g["alpha_hht"]
            cur_beta = g["beta"]
            cur_gamma = g["gamma"]
            cur_alpha_m = g["alpha_m"]
            cur_alpha_f = g["alpha_f"]
            cur_rho_inf = val
        except ValueError:
            raise ValueError(f"time_integrator {time_integrator!r}: {parts[1]!r} is not a rho_inf value") from None
    elif raw_name.startswith("moderate-") or raw_name.startswith("mod-"):
        parts = raw_name.split("-", 1)
        try:
            val = float(parts[1])
        except ValueError:
            raise ValueError(f"time_integrator {time_integrator!r}: {parts[1]!r} is not an alpha value") from None
        val = -abs(val) if val != 0.0 else 0.0
        cur_alpha_hht = val
        cur_beta = (1.0 - val) ** 2 / 4.0
        cur_gamma = 0.5 - val
    elif raw_name == "hht":
        cur_alpha_hht = -0.05
        cur_beta = (1.0 - cur_alpha_hht) ** 2 / 4.0
        cur_gamma = 0.5 - cur_alpha_hht
    else:
        # An unknown name used to fall through to the Newmark defaults (beta 1/4,
        # gamma 1/2) -- a different integrator than the one asked for, silently.
        raise ValueError(f"unknown time_integrator {time_integrator!r}; known: {', '.join(sorted(PRESETS))}, "
                         f"hht, moderate-<alpha>, mod-<alpha>, generalized-<rho_inf>, galpha-<rho_inf>")

    # 2. 개별 수치 상수 매뉴얼 오버라이드
    if static_mode is not None:
        is_qs = bool(static_mode)

    if rho_inf is not None:
        g = _calc_generalized_alpha(rho_inf)
        cur_alpha_hht = g["alpha_hht"]
        cur_beta = g["beta"]
        cur_gamma = g["gamma"]
        cur_alpha_m = g["alpha_m"]
        cur_alpha_f = g["alpha_f"]
        cur_rho_inf = float(rho_inf)

    eff_alpha = alpha_hht if alpha_hht is not None else alpha
    if eff_alpha is not None:
        cur_alpha_hht = float(eff_alpha)
        if beta is None:
            cur_beta = (1.0 - cur_alpha_hht) ** 2 / 4.0
        if gamma is None:
            cur_gamma = 0.5 - cur_alpha_hht

    if alpha_m is not None:
        cur_alpha_m = float(alpha_m)
    if alpha_f is not None:
        cur_alpha_f = float(alpha_f)

    if beta is not None:
        cur_beta = float(beta)
    if gamma is not None:
        cur_gamma = float(gamma)

    return TimeIntegratorConfig(
        name=raw_name,
        is_quasistatic=is_qs,
        is_emcs=is_emcs,
        alpha_hht=cur_alpha_hht,
        beta=cur_beta,
        gamma=cur_gamma,
        alpha_m=cur_alpha_m,
        alpha_f=cur_alpha_f,
        rho_inf=cur_rho_inf,
    )
