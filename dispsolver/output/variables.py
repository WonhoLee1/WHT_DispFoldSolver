"""variables.py
=============
출력 변수 풀 상수 클래스 및 열거형 정의.

IntelliSense 지원을 위해 각 출력 변수를 클래스 속성(타입 힌트 + 인라인 주석)으로 정의.
코드 상에서 ``FieldVar.U``, ``HistoryVar.ALLIE`` 형태로 안전하게 참조 가능.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, Tuple, Set


# ---------------------------------------------------------------------------
# 위치 열거형 / 타입 열거형
# ---------------------------------------------------------------------------

class FieldPosition(str, Enum):
    """Field Output이 평가되는 위치."""
    INTEGRATION_POINTS = "INTEGRATION_POINTS"  # 가우스 적분점
    CENTROID = "CENTROID"                       # 요소 중심
    NODES = "NODES"                             # 절점 (평균화)
    ELEMENT_NODES = "ELEMENT_NODES"             # 요소별 절점 (비평균화)


class FieldType(str, Enum):
    """Field Output의 수학적 타입."""
    SCALAR = "SCALAR"
    VECTOR = "VECTOR"
    TENSOR_2D_PLANAR = "TENSOR_2D_PLANAR"  # [S11, S22, S33, S12]
    TENSOR_3D_FULL = "TENSOR_3D_FULL"      # [S11, S22, S33, S12, S13, S23]


# ---------------------------------------------------------------------------
# Field Output 변수 풀 상수 클래스
# ---------------------------------------------------------------------------

class FieldVar:
    """Field Output 요청 시 사용하는 변수 이름 상수 클래스.

    Examples::

        req = step.FieldOutputRequest("F-Output-1", variables=[FieldVar.U, FieldVar.S])

    모든 속성은 Abaqus Field Output 변수명과 1:1 대응합니다.
    """

    # ---------- 절점 변수 (Nodal Variables) ----------

    U: str = "U"
    """절점 변위 벡터 (u_x, u_y, u_z). 성분: U1, U2, U3."""

    RF: str = "RF"
    """반력 벡터 (Reaction Force). 성분: RF1, RF2, RF3."""

    V: str = "V"
    """절점 속도 벡터 (Velocity). 동적 해석 전용. 성분: V1, V2, V3."""

    A: str = "A"
    """절점 가속도 벡터 (Acceleration). 동적 해석 전용. 성분: A1, A2, A3."""

    COORD: str = "COORD"
    """변형된 현재 좌표 (Current Coordinates). 성분: COORD1, COORD2, COORD3."""

    # ---------- 요소 응력 텐서 (Stress Tensors) ----------

    S: str = "S"
    """Cauchy 응력 텐서. 성분: S11, S22, S33, S12, S13, S23.
    
    현재 변형된 구성에서 단위 면적당 힘 (MPa). 비선형 해석에서 기본 응력 출력.
    """

    # ---------- 요소 변형률 텐서 (Strain Tensors) ----------

    LE: str = "LE"
    """대수 변형률 텐서, Logarithmic (Hencky) Strain. 성분: LE11, LE22, LE33, LE12, LE13, LE23.

    대변형 해석에서 물리적으로 일관된 변형률 측도. 소변형에서 공학 변형률과 동일.
    """

    E: str = "E"
    """총 변형률 (Total Strain). 성분: E11, E22, E33, E12, E13, E23."""

    PE: str = "PE"
    """소성 변형률 텐서 (Plastic Strain). 성분: PE11, PE22, PE33, PE12, PE13, PE23."""

    EE: str = "EE"
    """탄성 변형률 텐서 (Elastic Strain). 성분: EE11, EE22, EE33, EE12, EE13, EE23."""

    # ---------- 스칼라 요소 변수 ----------

    PEEQ: str = "PEEQ"
    """등가 소성 변형률 (Equivalent Plastic Strain). 스칼라.

    J2 소성 법칙의 누적 소성 변형 크기. 소성 흐름 여부 판별 기준.
    """

    SENER: str = "SENER"
    """탄성 변형 에너지 밀도 (Strain Energy Density). 스칼라 (mJ/mm³)."""

    PENER: str = "PENER"
    """소성 소산 에너지 밀도 (Plastic Dissipation Energy Density). 스칼라 (mJ/mm³)."""

    # ---------- 응력 파생 불변량 변수 (Stress Invariants) ----------

    MISES: str = "MISES"
    """Von Mises 등가 응력 (Von Mises Equivalent Stress). 스칼라 (MPa)."""

    MAXP: str = "MAXP"
    """최대 주응력 (Maximum Principal Stress, sigma_1). 스칼라 (MPa)."""

    MIDP: str = "MIDP"
    """중간 주응력 (Intermediate Principal Stress, sigma_2). 스칼라 (MPa)."""

    MINP: str = "MINP"
    """최소 주응력 (Minimum Principal Stress, sigma_3). 스칼라 (MPa)."""

    PRESS: str = "PRESS"
    """등가 정수압 (Equivalent Hydrostatic Pressure, -1/3 tr(sigma)). 스칼라 (MPa)."""

    # ---------- 접촉 변수 (Contact Variables) ----------

    CPRESS: str = "CPRESS"
    """접촉면 압력 (Contact Pressure). 스칼라 (MPa)."""

    CSLIP: str = "CSLIP"
    """접촉면 미끄럼 변위 (Contact Slip). 성분: CSLIP1, CSLIP2."""

    CSHEAR: str = "CSHEAR"
    """접촉면 마찰 전단 응력/트랙션 (Contact Frictional Shear Stress). 성분: CSHEAR1, CSHEAR2."""

    COPEN: str = "COPEN"
    """접촉면 간극/개구량 (Contact Opening / Clearance). 스칼라 (mm)."""

    CSTATUS: str = "CSTATUS"
    """접촉 상태 플래그 (Contact Status). 스칼라: 0=Open, 1=Stick, 2=Slip."""


# ---------------------------------------------------------------------------
# History Output 변수 풀 상수 클래스
# ---------------------------------------------------------------------------

class HistoryVar:
    """History Output 요청 시 사용하는 변수 이름 상수 클래스.

    Examples::

        req = step.HistoryOutputRequest("H-Output-1", variables=[HistoryVar.ALLIE, HistoryVar.ALLPD])

    에너지 변수는 전체 모델(WholeModel) 스칼라 이력으로 기록됩니다.
    """

    # ---------- 전체 모델 에너지 변수 (Whole Model Energy) ----------

    ALLIE: str = "ALLIE"
    """전체 내부 에너지 (Total Internal Energy). 탄성 + 소성 에너지 합계.

    에너지 보존 검증 기준. ALLIE + ALLKE == ALLWK 이 만족해야 함.
    """

    ALLSE: str = "ALLSE"
    """회복 가능 탄성 변형 에너지 (Recoverable Strain Energy). ∫σ:dε_e dV."""

    ALLPD: str = "ALLPD"
    """소성 소산 에너지 (Plastic Dissipation). ∫σ:dε_p dV. 비가역 에너지."""

    ALLVD: str = "ALLVD"
    """점성 감쇠 소산 에너지 (Viscous Damping Energy). 자동 안정화 포함."""

    ALLKE: str = "ALLKE"
    """운동 에너지 (Kinetic Energy). 준정적 해석에서 ≈ 0 이어야 함."""

    ALLWK: str = "ALLWK"
    """외력이 하는 일 (External Work). ∫f_ext · du."""

    ALLAE: str = "ALLAE"
    """인공 변형 에너지 (Artificial Strain Energy). 아워글래스 / 수치 감쇠 / 왜곡 제어 에너지."""

    ALLSD: str = "ALLSD"
    """정적/접촉 수치 안정화 소산 에너지 (Stabilization Dissipation)."""

    ALLFD: str = "ALLFD"
    """마찰 소산 에너지 (Frictional Dissipation Energy)."""

    ETOTAL: str = "ETOTAL"
    """에너지 평형 잔차 (ALLIE + ALLKE - ALLWK). 0에 가까울수록 에너지 보존."""

    # ---------- 절점 이력 변수 ----------

    U1: str = "U1"
    """절점 변위 x-성분."""

    U2: str = "U2"
    """절점 변위 y-성분."""

    U3: str = "U3"
    """절점 변위 z-성분."""

    RF1: str = "RF1"
    """절점 반력 x-성분."""

    RF2: str = "RF2"
    """절점 반력 y-성분."""

    RF3: str = "RF3"
    """절점 반력 z-성분."""

    # ---------- 단면 이력 변수 ----------

    SOF1: str = "SOF1"
    """단면 합력 x-성분 (Section Output Force)."""

    SOF2: str = "SOF2"
    """단면 합력 y-성분."""

    SOF3: str = "SOF3"
    """단면 합력 z-성분."""

    SOM1: str = "SOM1"
    """단면 합력 모멘트 x-성분 (Section Output Moment)."""

    SOM2: str = "SOM2"
    """단면 합력 모멘트 y-성분."""

    SOM3: str = "SOM3"
    """단면 합력 모멘트 z-성분."""


# ---------------------------------------------------------------------------
# 하위 호환성 유지용 Dict / Set (기존 코드 호환)
# ---------------------------------------------------------------------------

FIELD_VARIABLE_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    FieldVar.U:      ("U1", "U2", "U3"),
    FieldVar.RF:     ("RF1", "RF2", "RF3"),
    FieldVar.V:      ("V1", "V2", "V3"),
    FieldVar.A:      ("A1", "A2", "A3"),
    FieldVar.COORD:  ("COORD1", "COORD2", "COORD3"),
    FieldVar.S:      ("S11", "S22", "S33", "S12", "S13", "S23"),
    FieldVar.LE:     ("LE11", "LE22", "LE33", "LE12", "LE13", "LE23"),
    FieldVar.E:      ("E11", "E22", "E33", "E12", "E13", "E23"),
    FieldVar.PE:     ("PE11", "PE22", "PE33", "PE12", "PE13", "PE23"),
    FieldVar.EE:     ("EE11", "EE22", "EE33", "EE12", "EE13", "EE23"),
    FieldVar.PEEQ:   ("PEEQ",),
    FieldVar.SENER:  ("SENER",),
    FieldVar.PENER:  ("PENER",),
    FieldVar.MISES:  ("MISES",),
    FieldVar.MAXP:   ("MAXP",),
    FieldVar.MIDP:   ("MIDP",),
    FieldVar.MINP:   ("MINP",),
    FieldVar.PRESS:  ("PRESS",),
    FieldVar.CPRESS: ("CPRESS",),
    FieldVar.CSLIP:  ("CSLIP1", "CSLIP2"),
    FieldVar.CSHEAR: ("CSHEAR1", "CSHEAR2"),
    FieldVar.COPEN:  ("COPEN",),
    FieldVar.CSTATUS:("CSTATUS",),
}

HISTORY_ENERGY_VARIABLES: Set[str] = {
    HistoryVar.ALLIE,
    HistoryVar.ALLSE,
    HistoryVar.ALLPD,
    HistoryVar.ALLVD,
    HistoryVar.ALLKE,
    HistoryVar.ALLWK,
    HistoryVar.ALLAE,
    HistoryVar.ALLSD,
    HistoryVar.ALLFD,
    HistoryVar.ETOTAL,
}

HISTORY_NODAL_VARIABLES: Set[str] = {
    HistoryVar.U1, HistoryVar.U2, HistoryVar.U3,
    "UR1", "UR2", "UR3",
    HistoryVar.RF1, HistoryVar.RF2, HistoryVar.RF3,
    "RM1", "RM2", "RM3",
    "V1", "V2", "V3",
    "A1", "A2", "A3",
}

HISTORY_SECTION_VARIABLES: Set[str] = {
    HistoryVar.SOF1, HistoryVar.SOF2, HistoryVar.SOF3,
    HistoryVar.SOM1, HistoryVar.SOM2, HistoryVar.SOM3,
}
