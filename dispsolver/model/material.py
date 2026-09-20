"""Material definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class Material:
    """Constitutive material definition with mechanical properties."""
    name: str
    mat_type: str = "ELASTIC"  # "ELASTIC", "J2_PLASTICITY", "NEO_HOOKEAN", "VISCOELASTIC"
    density: float = 0.0
    elastic: Optional[Tuple[float, float]] = None  # (E, nu)
    plastic: Optional[List[Tuple[float, float]]] = None  # [(sigma_y, eps_p), ...]
    hyperelastic: Optional[Dict[str, Any]] = None  # {'model': 'ARRUDA_BOYCE', 'c10': ..., 'd1': ...}
    viscoelastic_prony: Optional[List[Tuple[float, float, float]]] = None  # [(g, k, tau), ...]
    expansion: Optional[Dict[str, Any]] = None  # {'type': 'ISO', 'table': ((alpha, temp), ...), 'zero': 0.0}
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def Expansion(self, type: str = "ISO", table: Any = None, zero: float = 0.0) -> None:
        """Abaqus-style Thermal Expansion property definition.
        
        Parameters
        ----------
        type : str
            Expansion type ("ISO" for isotropic, "ORTHO" for orthotropic, "ANISO" for anisotropic)
        table : tuple or list
            Table of expansion coefficients. For ISO: ((alpha, temp), ...) or ((alpha,),)
        zero : float
            Reference temperature T0 where thermal strain is zero (default: 0.0)
        """
        parsed_table: List[Tuple[float, ...]] = []
        if table is not None:
            for row in table:
                if isinstance(row, (list, tuple)):
                    parsed_table.append(tuple(float(v) for v in row))
                else:
                    parsed_table.append((float(row),))
        self.expansion = {
            "type": type.upper(),
            "table": parsed_table,
            "zero": float(zero),
        }

    def Elastic(self, table: Any = None, E: Optional[float] = None, nu: Optional[float] = None) -> None:
        """Abaqus-style Elastic property definition (supports table tuple or E, nu keywords).
        
        If table rows have 3 columns: ((E, nu, temp), ...), temperature dependence is activated.
        """
        if E is not None and nu is not None:
            self.elastic = (float(E), float(nu))
        elif isinstance(table, (list, tuple)) and len(table) > 0:
            if isinstance(table[0], (list, tuple)):
                self.elastic = (float(table[0][0]), float(table[0][1]))
                if len(table[0]) >= 3:
                    self.extra_params["temp_elastic_table"] = [
                        (float(row[0]), float(row[1]), float(row[2])) for row in table
                    ]
            else:
                self.elastic = (float(table[0]), float(table[1]))

    def Plastic(self, table: Any, hardening: str = "ISOTROPIC") -> None:
        """Abaqus-style Plastic property definition.
        
        If table rows have 3 columns: ((yield, strain, temp), ...), temperature dependence is activated.
        """
        if len(table) > 0 and isinstance(table[0], (list, tuple)) and len(table[0]) >= 3:
            self.plastic = [(float(row[0]), float(row[1])) for row in table]
            self.extra_params["temp_plastic_table"] = [
                (float(row[0]), float(row[1]), float(row[2])) for row in table
            ]
            self.mat_type = "TABULAR_PLASTIC"
            self.extra_params["tabular_plastic"] = self.plastic
        else:
            self.plastic = [(float(row[0]), float(row[1])) for row in table]
            if hardening.upper() == "TABULAR" or len(self.plastic) > 2:
                self.mat_type = "TABULAR_PLASTIC"
                self.extra_params["tabular_plastic"] = self.plastic
            elif hardening.upper() in ["KINEMATIC", "COMBINED"]:
                self.extra_params["hardening_type"] = hardening.upper()

    def RateDependent(self, type: str = "POWER_LAW", table: Any = None, **kwargs) -> None:
        """Abaqus-standard *RATE DEPENDENT property definition.
        Supports:
          - "COWPER_SYMONDS": table=((D, p),) or D=..., p=...
          - "JOHNSON_COOK": table=((C, eps_dot_0),) or C=..., eps_dot_0=...
          - "POWER_LAW": table=((D, n),)
        """
        rd_dict = {"type": type.upper(), **kwargs}
        if table is not None and len(table) > 0:
            row = table[0] if isinstance(table[0], (list, tuple)) else table
            if type.upper() in ["COWPER_SYMONDS", "COWPER-SYMONDS"]:
                rd_dict["D"] = float(row[0])
                rd_dict["p"] = float(row[1])
            elif type.upper() in ["JOHNSON_COOK", "JOHNSON-COOK"]:
                rd_dict["C"] = float(row[0])
                rd_dict["eps_dot_0"] = float(row[1])
        self.extra_params["rate_dependent"] = rd_dict

    def JohnsonCook(
        self,
        A: float,
        B: float,
        n: float,
        C: float = 0.0,
        m: float = 0.0,
        T_room: float = 293.15,
        T_melt: float = 1793.15,
        eps_dot_0: float = 1.0,
    ) -> None:
        """Abaqus standard Johnson-Cook plasticity model helper."""
        self.extra_params["johnson_cook"] = {
            "A": float(A),
            "B": float(B),
            "n": float(n),
            "C": float(C),
            "m": float(m),
            "T_room": float(T_room),
            "T_melt": float(T_melt),
            "eps_dot_0": float(eps_dot_0),
        }
        self.mat_type = "JOHNSON_COOK"

    def Hyperelastic(self, model: str = "NEO_HOOKEAN", n: int = 1, table: Any = None, **kwargs) -> None:
        """Abaqus 2026 standard Hyperelastic property definition.
        
        Supports:
          - "NEO_HOOKEAN": table=((C10, D1),) or C10=..., D1=...
          - "MOONEY_RIVLIN": table=((C10, C01, D1),) or C10=..., C01=..., D1=...
          - "OGDEN": n=1..6, table=((mu_1, alpha_1, D_1), (mu_2, alpha_2, D_2), ...)
                     or single row flat table=((mu1, a1, mu2, a2, ..., D1, D2, ...),)
          - "POLYNOMIAL": n=1 or 2, table=((C10, C01, C20, C11, C02, D1, D2),)
          - "YEOH" / "REDUCED_POLYNOMIAL": n=1..3, table=((C10, C20, C30, D1, D2, D3),)
          - "ARRUDA_BOYCE": table=((mu, lambda_m, D),)
          - "MARLOW": table=((...),)
        """
        model_upper = model.upper()
        h_dict: Dict[str, Any] = {"model": model_upper, "n": n, **kwargs}

        if table is not None and len(table) > 0:
            h_dict["table"] = table
            # Parse table into explicit parameters based on model type
            if model_upper in ["MOONEY_RIVLIN", "MOONEY-RIVLIN"]:
                row = table[0] if isinstance(table[0], (list, tuple)) else table
                if len(row) >= 3:
                    h_dict.setdefault("c10", float(row[0]))
                    h_dict.setdefault("c01", float(row[1]))
                    h_dict.setdefault("d1", float(row[2]))
            elif model_upper in ["OGDEN"]:
                mu_list: List[float] = []
                alpha_list: List[float] = []
                d_list: List[float] = []
                if isinstance(table[0], (list, tuple)) and len(table[0]) == 3 and len(table) > 1:
                    # Multi-row format: ((mu1, a1, D1), (mu2, a2, D2), ...)
                    for row in table:
                        mu_list.append(float(row[0]))
                        alpha_list.append(float(row[1]))
                        d_list.append(float(row[2]) if len(row) > 2 else 0.0)
                else:
                    # Single flat row: (mu1, a1, ..., muN, aN, D1, ..., DN)
                    flat = table[0] if isinstance(table[0], (list, tuple)) else table
                    n_terms = min(n, len(flat) // 2)
                    for i in range(n_terms):
                        mu_list.append(float(flat[2 * i]))
                        alpha_list.append(float(flat[2 * i + 1]))
                    d_start = 2 * n_terms
                    for i in range(n_terms):
                        d_val = float(flat[d_start + i]) if (d_start + i) < len(flat) else 0.0
                        d_list.append(d_val)
                h_dict["mu_list"] = mu_list
                h_dict["alpha_list"] = alpha_list
                h_dict["d_list"] = d_list
                h_dict["n"] = len(mu_list)
            elif model_upper in ["POLYNOMIAL"]:
                row = table[0] if isinstance(table[0], (list, tuple)) else table
                keys = ["c10", "c01", "c20", "c11", "c02", "d1", "d2"]
                for k, v in zip(keys, row):
                    h_dict.setdefault(k, float(v))
            elif model_upper in ["YEOH", "REDUCED_POLYNOMIAL"]:
                row = table[0] if isinstance(table[0], (list, tuple)) else table
                if len(row) >= 2:
                    h_dict.setdefault("c10", float(row[0]))
                if len(row) >= 4:
                    h_dict.setdefault("c20", float(row[1]))
                    h_dict.setdefault("d1", float(row[2]))
                elif len(row) >= 3:
                    h_dict.setdefault("c20", float(row[1]))
                    h_dict.setdefault("c30", float(row[2]))
                if len(row) >= 6:
                    h_dict.setdefault("c30", float(row[2]))
                    h_dict.setdefault("d1", float(row[3]))
                    h_dict.setdefault("d2", float(row[4]))
                    h_dict.setdefault("d3", float(row[5]))
            elif model_upper in ["ARRUDA_BOYCE", "ARRUDA-BOYCE"]:
                row = table[0] if isinstance(table[0], (list, tuple)) else table
                if len(row) >= 2:
                    h_dict.setdefault("mu", float(row[0]))
                    h_dict.setdefault("lambda_m", float(row[1]))
                if len(row) >= 3:
                    d_val = float(row[2])
                    h_dict.setdefault("d", d_val)
                    h_dict.setdefault("K", 2.0 / d_val if d_val > 1e-12 else 1e6)
            elif model_upper in ["NEO_HOOKEAN", "NEOHOOKEAN"]:
                row = table[0] if isinstance(table[0], (list, tuple)) else table
                if len(row) >= 2:
                    h_dict.setdefault("c10", float(row[0]))
                    h_dict.setdefault("d1", float(row[1]))

        self.hyperelastic = h_dict
        self.mat_type = "HYPERELASTIC"

    def Viscoelastic(self, domain: str = "TIME", time: str = "PRONY", table: Any = None, **kwargs) -> None:
        """Abaqus standard Viscoelastic property definition (Prony series up to M=6).
        
        table format: ((g_1, k_1, tau_1), (g_2, k_2, tau_2), ...)
        """
        prony_list: List[Tuple[float, float, float]] = []
        if table is not None:
            for row in table:
                if len(row) == 3:
                    prony_list.append((float(row[0]), float(row[1]), float(row[2])))
                elif len(row) == 2:
                    prony_list.append((float(row[0]), 0.0, float(row[1])))
        if not prony_list and "prony" in kwargs:
            prony_list = kwargs["prony"]

        self.viscoelastic_prony = prony_list
        self.extra_params["domain"] = domain.upper()
        self.extra_params["time"] = time.upper()
        self.mat_type = "HYPER_VISCOELASTIC" if self.hyperelastic else "VISCOELASTIC"

    def MullinsEffect(self, r: float = 2.0, m: float = 0.5, beta: float = 0.1, table: Any = None) -> None:
        """Abaqus standard *MULLINS EFFECT property definition.
        
        table format: ((r, m, beta),)
        """
        if table is not None and len(table) > 0:
            row = table[0] if isinstance(table[0], (list, tuple)) else table
            r_val = float(row[0])
            m_val = float(row[1]) if len(row) > 1 else 0.5
            beta_val = float(row[2]) if len(row) > 2 else 0.1
        else:
            r_val = float(r)
            m_val = float(m)
            beta_val = float(beta)
            
        self.extra_params["mullins"] = {
            "r": r_val,
            "m": m_val,
            "beta": beta_val,
        }

    def BergstromBoyce(
        self,
        mu_A: float = 0.2,
        lambda_m: float = 3.0,
        K: float = 10.0,
        s: float = 1.5,
        gamma_dot_0: float = 0.1,
        C: float = 0.0,
        m: float = 2.0,
        tau_hat: float = 0.5,
    ) -> None:
        """Bergstrom-Boyce 2-Network Visco-Hyperplastic Model."""
        self.extra_params["bergstrom_boyce"] = {
            "mu_A": float(mu_A),
            "lambda_m": float(lambda_m),
            "K": float(K),
            "s": float(s),
            "gamma_dot_0": float(gamma_dot_0),
            "C": float(C),
            "m": float(m),
            "tau_hat": float(tau_hat),
        }
        self.mat_type = "BERGSTROM_BOYCE"

    def Creep(self, law: str = "STRAIN", table: Any = None, **kwargs) -> None:
        """Abaqus standard *CREEP property definition (Norton / Bailey-Norton).
        
        table format: ((A, n, m),)
        A: Creep multiplier (>= 0)
        n: Stress exponent (> 0)
        m: Time exponent (-1.0 < m <= 0.0, default 0.0)
        """
        law_upper = law.upper()
        A_val, n_val, m_val = 1e-12, 1.0, 0.0
        if table is not None and len(table) > 0:
            row = table[0] if isinstance(table[0], (list, tuple)) else table
            A_val = float(row[0])
            n_val = float(row[1]) if len(row) > 1 else 1.0
            m_val = float(row[2]) if len(row) > 2 else 0.0
        else:
            A_val = float(kwargs.get("A", 1e-12))
            n_val = float(kwargs.get("n", 1.0))
            m_val = float(kwargs.get("m", 0.0))

        self.extra_params["creep"] = {
            "law": law_upper,
            "A": A_val,
            "n": n_val,
            "m": m_val,
        }
        self.mat_type = "CREEP"

    def RadiossLaw36(
        self,
        E: float,
        nu: float,
        sy0: float,
        sigma_uts: float,
        eps_fail: float,
        density: Optional[float] = None,
        smooth: bool = True,
        n_points: int = 8,
        use_damage: bool = False,
        delta_eps_damage: Optional[float] = None,
        extrapolate_strain: float = 5.0,
        tail_modulus_ratio: float = 1e-3,
    ) -> None:
        """Radioss LAW36 / Abaqus Tabular Plasticity Advanced Setup Helper.
        
        Features:
          1. Smooth C1 transition: Generates smooth hardening curve between sy0 and sigma_uts
             to avoid discontinuous kinks and cutbacks in Newton-Raphson solvers.
          2. Progressive Damage (use_damage=True): Progressively softens stress after eps_fail
             to avoid sudden shock/instability.
          3. Extrapolated Regularized Tail (use_damage=False, default):
             Extrapolates hardening curve with a small positive modulus (H_tail = tail_modulus_ratio * E > 0)
             up to large strain (default eps=5.0) to eliminate zero-stiffness singularity
             and ensure robust numerical convergence even under extreme distortion.
        """
        import math
        self.elastic = (float(E), float(nu))
        if density is not None:
            self.density = float(density)

        table: List[Tuple[float, float]] = []
        E_f = float(E)
        sy0_f = float(sy0)
        s_uts_f = float(sigma_uts)
        ef_f = float(eps_fail)

        if smooth and n_points >= 4:
            # Generate smooth transition points up to eps_fail
            # Reserve 1 or 2 slots for post-failure / tail point
            n_core = max(3, n_points - (2 if use_damage else 1))
            for i in range(n_core):
                xi = i / float(n_core - 1)  # 0.0 -> 1.0
                ep_val = ef_f * (xi ** 1.2)
                # Smooth sinusoidal transition: zero curvature shock at yield, softening toward UTS
                sy_val = sy0_f + (s_uts_f - sy0_f) * math.sin(0.5 * math.pi * (xi ** 0.8))
                table.append((float(sy_val), float(ep_val)))
        else:
            table.append((sy0_f, 0.0))
            table.append((s_uts_f, ef_f))

        if use_damage:
            # Option A: Progressive damage softening post eps_fail
            d_eps = float(delta_eps_damage) if delta_eps_damage is not None else max(0.05, 0.25 * ef_f)
            e_rupture = ef_f + d_eps
            s_fail = 0.01 * s_uts_f
            # Intermediate softening point
            e_mid = ef_f + 0.4 * d_eps
            s_mid = s_fail + (s_uts_f - s_fail) * math.exp(-1.5)
            table.append((float(s_mid), float(e_mid)))
            table.append((float(s_fail), float(e_rupture)))
            self.extra_params["eps_fail"] = ef_f
            self.extra_params["sigma_fail"] = float(s_fail)
            self.extra_params["use_damage"] = True
        else:
            # Option B: Damage disabled - Regularized positive tail up to eps=5.0
            # Slope H_tail = tail_modulus_ratio * E > 0 ensures strictly positive definite tangent
            H_tail = max(1e-6 * E_f, float(tail_modulus_ratio) * E_f)
            e_tail = max(float(extrapolate_strain), ef_f * 2.0)
            s_tail = s_uts_f + H_tail * (e_tail - ef_f)
            table.append((float(s_tail), float(e_tail)))
            self.extra_params["eps_fail"] = 1e9  # No failure / softening trigger
            self.extra_params["sigma_fail"] = 0.0
            self.extra_params["use_damage"] = False
            self.extra_params["H_tail"] = float(H_tail)
            self.extra_params["extrapolate_strain"] = float(e_tail)

        self.plastic = table
        self.mat_type = "TABULAR_PLASTIC"
        self.extra_params["tabular_plastic"] = table
        self.extra_params["is_radioss_law36"] = True

    def TabularPlastic(self, table: Any, eps_fail: float = 1e9, sigma_fail: float = 0.0) -> None:
        """User-defined Tabular Plastic hardening with damage threshold."""
        self.plastic = [(float(row[0]), float(row[1])) for row in table]
        self.mat_type = "TABULAR_PLASTIC"
        self.extra_params["tabular_plastic"] = self.plastic
        self.extra_params["eps_fail"] = float(eps_fail)
        self.extra_params["sigma_fail"] = float(sigma_fail)

    def DuctileDamage(self, eps_fail: float, sigma_fail: float = 0.0) -> None:
        """Abaqus-style Ductile Damage initiation and failure stress."""
        self.extra_params["eps_fail"] = float(eps_fail)
        self.extra_params["sigma_fail"] = float(sigma_fail)

    def to_numba_props(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Numba DOD materials (Abaqus 2026 & Radioss parity)."""
        from dispsolver.material3d.numba_materials import (
            MAT_LINEAR_ELASTIC,
            MAT_J2_PLASTICITY,
            MAT_J2_PLASTICITY_NONLINEAR,
            MAT_NEO_HOOKEAN,
            MAT_ARRUDA_BOYCE,
            MAT_HYPERELASTIC_ARRUDA_BOYCE,
            MAT_YEOH,
            MAT_HYPERELASTIC_YEOH,
            MAT_HYPERELASTIC_MOONEY_RIVLIN,
            MAT_HYPERELASTIC_OGDEN,
            MAT_HYPERELASTIC_POLYNOMIAL,
            MAT_HYPERELASTIC_MARLOW,
            MAT_J2_PLASTICITY_TABULAR,
            MAT_VISCOELASTIC_PRONY,
            MAT_HYPER_VISCOELASTIC,
            MAT_CUSTOM_ELASTIC
        )
        
        props: Dict[str, Any] = {}
        upper_type = self.mat_type.upper()
        props["density"] = float(self.density)

        # Fallback attribute extraction
        E_val = getattr(self, "E", None)
        nu_val = getattr(self, "nu", None)
        if self.elastic is not None:
            E_val, nu_val = self.elastic
        elif E_val is None:
            E_val = 1.0
        if nu_val is None:
            nu_val = 0.3
        
        if upper_type in ["ELASTIC", "LINEAR_ELASTIC"]:
            props["E"] = float(E_val)
            props["nu"] = float(nu_val)
            props["mat_type"] = MAT_LINEAR_ELASTIC

        elif upper_type in ["TABULAR_PLASTIC", "RADIOSS_LAW36"]:
            props["mat_type"] = MAT_J2_PLASTICITY_TABULAR
            props["E"] = float(E_val)
            props["nu"] = float(nu_val)
            tab = self.extra_params.get("tabular_plastic", self.plastic) or [(100.0, 0.0)]
            props["tabular"] = tab
            props["eps_fail"] = float(self.extra_params.get("eps_fail", 1e9))
            props["sigma_fail"] = float(self.extra_params.get("sigma_fail", 0.0))

        elif upper_type in ["J2_PLASTICITY", "PLASTIC"]:
            sy0 = getattr(self, "sigma_y0", getattr(self, "yield_stress", None))
            H_mod = getattr(self, "H", getattr(self, "hardening_modulus", None))
            R_inf = self.extra_params.get("R_inf", None)
            b_exp = self.extra_params.get("b_exp", None)

            if "tabular_plastic" in self.extra_params:
                props["mat_type"] = MAT_J2_PLASTICITY_TABULAR
                props["E"] = float(E_val)
                props["nu"] = float(nu_val)
                props["tabular"] = self.extra_params["tabular_plastic"]
                props["eps_fail"] = float(self.extra_params.get("eps_fail", 1e9))
                props["sigma_fail"] = float(self.extra_params.get("sigma_fail", 0.0))
            elif R_inf is not None and b_exp is not None:
                props["mat_type"] = MAT_J2_PLASTICITY_NONLINEAR
                props["E"] = float(E_val)
                props["nu"] = float(nu_val)
                props["yield_stress"] = float(sy0 if sy0 is not None else 400.0)
                props["hardening_modulus"] = float(H_mod if H_mod is not None else 0.0)
                props["R_inf"] = float(R_inf)
                props["b_exp"] = float(b_exp)
            else:
                props["E"] = float(E_val)
                props["nu"] = float(nu_val)
                props["mat_type"] = MAT_J2_PLASTICITY
                if self.plastic and len(self.plastic) > 0:
                    props["yield_stress"] = float(self.plastic[0][0])
                    if len(self.plastic) > 1:
                        sy0_p, ep0 = self.plastic[0]
                        sy1_p, ep1 = self.plastic[1]
                        dep = ep1 - ep0
                        props["hardening_modulus"] = float((sy1_p - sy0_p) / dep) if dep > 1e-12 else 0.0
                    else:
                        props["hardening_modulus"] = float(H_mod) if H_mod is not None else float(self.extra_params.get("hardening_modulus", 0.0))
                else:
                    props["yield_stress"] = float(sy0) if sy0 is not None else float(self.extra_params.get("yield_stress", 1e9))
                    props["hardening_modulus"] = float(H_mod) if H_mod is not None else float(self.extra_params.get("hardening_modulus", 0.0))

        elif upper_type in ["MOONEY_RIVLIN", "MOONEY-RIVLIN"]:
            props["mat_type"] = MAT_HYPERELASTIC_MOONEY_RIVLIN
            c10 = float(self.extra_params.get("c10", 0.1))
            c01 = float(self.extra_params.get("c01", 0.05))
            d1 = float(self.extra_params.get("d1", 0.01))
            if self.hyperelastic:
                c10 = float(self.hyperelastic.get("c10", c10))
                c01 = float(self.hyperelastic.get("c01", c01))
                d1 = float(self.hyperelastic.get("d1", d1))
            props["c10"] = c10
            props["c01"] = c01
            props["d1"] = d1
            mu = 2.0 * (c10 + c01)
            K = 2.0 / d1 if d1 > 1e-12 else 1e6
            props["E"] = float((9.0 * K * mu) / (3.0 * K + mu))
            props["nu"] = float((3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu)))

        elif upper_type in ["OGDEN"]:
            props["mat_type"] = MAT_HYPERELASTIC_OGDEN
            mu_list = self.extra_params.get("mu_list", [0.2])
            alpha_list = self.extra_params.get("alpha_list", [2.0])
            d_list = self.extra_params.get("d_list", [0.01])
            if self.hyperelastic:
                mu_list = self.hyperelastic.get("mu_list", mu_list)
                alpha_list = self.hyperelastic.get("alpha_list", alpha_list)
                d_list = self.hyperelastic.get("d_list", d_list)
            props["n_terms"] = len(mu_list)
            props["mu_list"] = mu_list
            props["alpha_list"] = alpha_list
            props["d_list"] = d_list
            mu_tot = sum(mu_list)
            d1 = d_list[0] if len(d_list) > 0 else 0.01
            K = 2.0 / d1 if d1 > 1e-12 else 1e6
            props["E"] = float((9.0 * K * mu_tot) / (3.0 * K + mu_tot)) if mu_tot > 0 else 1.0
            props["nu"] = float((3.0 * K - 2.0 * mu_tot) / (2.0 * (3.0 * K + mu_tot))) if mu_tot > 0 else 0.49

        elif upper_type in ["POLYNOMIAL"]:
            props["mat_type"] = MAT_HYPERELASTIC_POLYNOMIAL
            for k in ["c10", "c01", "c20", "c11", "c02", "d1", "d2"]:
                val = self.extra_params.get(k, 0.0)
                if self.hyperelastic:
                    val = self.hyperelastic.get(k, val)
                props[k] = float(val)
            c10 = props.get("c10", 0.1)
            c01 = props.get("c01", 0.0)
            d1 = props.get("d1", 0.01)
            mu = 2.0 * (c10 + c01)
            K = 2.0 / d1 if d1 > 1e-12 else 1e6
            props["E"] = float((9.0 * K * mu) / (3.0 * K + mu)) if mu > 0 else 1.0
            props["nu"] = float((3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu))) if mu > 0 else 0.49

        elif upper_type in ["ARRUDA_BOYCE", "ARRUDA-BOYCE"]:
            props["mat_type"] = MAT_ARRUDA_BOYCE
            mu_val = getattr(self, "mu", self.extra_params.get("mu", 0.16785))
            lm_val = getattr(self, "lambda_m", self.extra_params.get("lambda_m", 3.0))
            K_val = getattr(self, "K", self.extra_params.get("K", 8.3333))
            if self.hyperelastic:
                mu_val = self.hyperelastic.get("mu", mu_val)
                lm_val = self.hyperelastic.get("lambda_m", lm_val)
                K_val = self.hyperelastic.get("K", K_val)
            props["mu"] = float(mu_val)
            props["lambda_m"] = float(lm_val)
            props["K"] = float(K_val)
            props["E"] = float((9.0 * K_val * mu_val) / (3.0 * K_val + mu_val))
            props["nu"] = float((3.0 * K_val - 2.0 * mu_val) / (2.0 * (3.0 * K_val + mu_val)))

        elif upper_type in ["YEOH"]:
            props["mat_type"] = MAT_YEOH
            c10_val = getattr(self, "c10", getattr(self, "C10", 0.1))
            c20_val = getattr(self, "c20", getattr(self, "C20", 0.0))
            c30_val = getattr(self, "c30", getattr(self, "C30", 0.0))
            d1_val = getattr(self, "d1", getattr(self, "D1", 0.01))
            d2_val = getattr(self, "d2", getattr(self, "D2", 0.0))
            d3_val = getattr(self, "d3", getattr(self, "D3", 0.0))
            if self.hyperelastic:
                c10_val = self.hyperelastic.get("c10", self.hyperelastic.get("C10", c10_val))
                c20_val = self.hyperelastic.get("c20", self.hyperelastic.get("C20", c20_val))
                c30_val = self.hyperelastic.get("c30", self.hyperelastic.get("C30", c30_val))
                d1_val = self.hyperelastic.get("d1", self.hyperelastic.get("D1", d1_val))
                d2_val = self.hyperelastic.get("d2", self.hyperelastic.get("D2", d2_val))
                d3_val = self.hyperelastic.get("d3", self.hyperelastic.get("D3", d3_val))
            props["c10"] = float(c10_val)
            props["c20"] = float(c20_val)
            props["c30"] = float(c30_val)
            props["d1"] = float(d1_val)
            props["d2"] = float(d2_val)
            props["d3"] = float(d3_val)
            mu = 2.0 * float(c10_val)
            K = 2.0 / float(d1_val) if float(d1_val) > 1e-12 else 1e6
            props["E"] = float((9.0 * K * mu) / (3.0 * K + mu))
            props["nu"] = float((3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu)))

        elif upper_type in ["NEO_HOOKEAN", "HYPERELASTIC"]:
            h_model = (self.hyperelastic.get("model", "") if self.hyperelastic else "").upper()
            if "MOONEY" in h_model:
                props["mat_type"] = MAT_HYPERELASTIC_MOONEY_RIVLIN
                c10 = float(self.hyperelastic.get("c10", 0.1))
                c01 = float(self.hyperelastic.get("c01", 0.05))
                d1 = float(self.hyperelastic.get("d1", 0.01))
                props["c10"] = c10
                props["c01"] = c01
                props["d1"] = d1
                mu = 2.0 * (c10 + c01)
                K = 2.0 / d1 if d1 > 1e-12 else 1e6
                props["E"] = float((9.0 * K * mu) / (3.0 * K + mu))
                props["nu"] = float((3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu)))
            elif "OGDEN" in h_model:
                props["mat_type"] = MAT_HYPERELASTIC_OGDEN
                mu_list = self.hyperelastic.get("mu_list", [0.2])
                alpha_list = self.hyperelastic.get("alpha_list", [2.0])
                d_list = self.hyperelastic.get("d_list", [0.01])
                props["n_terms"] = len(mu_list)
                props["mu_list"] = mu_list
                props["alpha_list"] = alpha_list
                props["d_list"] = d_list
                mu_tot = sum(mu_list)
                d1 = d_list[0] if len(d_list) > 0 else 0.01
                K = 2.0 / d1 if d1 > 1e-12 else 1e6
                props["E"] = float((9.0 * K * mu_tot) / (3.0 * K + mu_tot)) if mu_tot > 0 else 1.0
                props["nu"] = float((3.0 * K - 2.0 * mu_tot) / (2.0 * (3.0 * K + mu_tot))) if mu_tot > 0 else 0.49
            elif "POLYNOMIAL" in h_model:
                props["mat_type"] = MAT_HYPERELASTIC_POLYNOMIAL
                for k in ["c10", "c01", "c20", "c11", "c02", "d1", "d2"]:
                    props[k] = float(self.hyperelastic.get(k, 0.0))
                c10 = props.get("c10", 0.1)
                c01 = props.get("c01", 0.0)
                d1 = props.get("d1", 0.01)
                mu = 2.0 * (c10 + c01)
                K = 2.0 / d1 if d1 > 1e-12 else 1e6
                props["E"] = float((9.0 * K * mu) / (3.0 * K + mu)) if mu > 0 else 1.0
                props["nu"] = float((3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu))) if mu > 0 else 0.49
            elif "MARLOW" in h_model:
                props["mat_type"] = MAT_HYPERELASTIC_MARLOW
                props["mu"] = float(self.hyperelastic.get("mu", 0.2))
                props["K"] = float(self.hyperelastic.get("K", 10.0))
                props["E"] = float((9.0 * props["K"] * props["mu"]) / (3.0 * props["K"] + props["mu"]))
                props["nu"] = float((3.0 * props["K"] - 2.0 * props["mu"]) / (2.0 * (3.0 * props["K"] + props["mu"])))
            elif "ARRUDA" in h_model:
                props["mat_type"] = MAT_ARRUDA_BOYCE
                mu_val = self.hyperelastic.get("mu", 0.16785)
                lm_val = self.hyperelastic.get("lambda_m", 3.0)
                K_val = self.hyperelastic.get("K", 8.3333)
                props["mu"] = float(mu_val)
                props["lambda_m"] = float(lm_val)
                props["K"] = float(K_val)
                props["E"] = float((9.0 * K_val * mu_val) / (3.0 * K_val + mu_val))
                props["nu"] = float((3.0 * K_val - 2.0 * mu_val) / (2.0 * (3.0 * K_val + mu_val)))
            elif "YEOH" in h_model:
                props["mat_type"] = MAT_YEOH
                c10_val = self.hyperelastic.get("c10", self.hyperelastic.get("C10", 0.1))
                c20_val = self.hyperelastic.get("c20", self.hyperelastic.get("C20", 0.0))
                c30_val = self.hyperelastic.get("c30", self.hyperelastic.get("C30", 0.0))
                d1_val = self.hyperelastic.get("d1", self.hyperelastic.get("D1", 0.01))
                d2_val = self.hyperelastic.get("d2", self.hyperelastic.get("D2", 0.0))
                d3_val = self.hyperelastic.get("d3", self.hyperelastic.get("D3", 0.0))
                props["c10"] = float(c10_val)
                props["c20"] = float(c20_val)
                props["c30"] = float(c30_val)
                props["d1"] = float(d1_val)
                props["d2"] = float(d2_val)
                props["d3"] = float(d3_val)
                mu = 2.0 * float(c10_val)
                K = 2.0 / float(d1_val) if float(d1_val) > 1e-12 else 1e6
                props["E"] = float((9.0 * K * mu) / (3.0 * K + mu))
                props["nu"] = float((3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu)))
            else:
                props["mat_type"] = MAT_NEO_HOOKEAN
                c10_val = getattr(self, "C10", getattr(self, "c10", None))
                d1_val = getattr(self, "D1", getattr(self, "d1", None))
                if self.hyperelastic:
                    c10_val = self.hyperelastic.get("c10", self.hyperelastic.get("C10", c10_val))
                    d1_val = self.hyperelastic.get("d1", self.hyperelastic.get("D1", d1_val))
                
                if c10_val is not None and d1_val is not None:
                    props["c10"] = float(c10_val)
                    props["d1"] = float(d1_val)
                    mu = 2.0 * float(c10_val)
                    K = 2.0 / float(d1_val) if float(d1_val) > 1e-12 else 1e6
                    props["E"] = float((9.0 * K * mu) / (3.0 * K + mu))
                    props["nu"] = float((3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu)))
                elif self.elastic:
                    props["E"] = float(E_val)
                    props["nu"] = float(nu_val)
                    mu = E_val / (2.0 * (1.0 + nu_val))
                    K = E_val / (3.0 * (1.0 - 2.0 * nu_val))
                    props["c10"] = float(mu / 2.0)
                    props["d1"] = float(2.0 / K) if K > 1e-12 else 1e-6
                else:
                    props["c10"] = 0.1
                    props["d1"] = 0.01
                    props["E"] = 0.6
                    props["nu"] = 0.49

        elif upper_type in ["VISCOELASTIC", "HYPER_VISCOELASTIC"]:
            prony = self.viscoelastic_prony or self.extra_params.get("prony", None)
            base_m = (self.extra_params.get("base", "NEO_HOOKEAN")).upper()
            if self.hyperelastic:
                base_m = self.hyperelastic.get("model", base_m).upper()

            if prony and len(prony) >= 1:
                props["mat_type"] = MAT_HYPER_VISCOELASTIC
                if "ARRUDA" in base_m:
                    props["base_model"] = MAT_ARRUDA_BOYCE
                    mu_val = float(self.extra_params.get("mu", self.hyperelastic.get("mu", 0.16785) if self.hyperelastic else 0.16785))
                    lm_val = float(self.extra_params.get("lambda_m", self.hyperelastic.get("lambda_m", 3.0) if self.hyperelastic else 3.0))
                    K_val = float(self.extra_params.get("K", self.hyperelastic.get("K", 8.3333) if self.hyperelastic else 8.3333))
                    props["mu"] = mu_val
                    props["lambda_m"] = lm_val
                    props["K"] = K_val
                elif "YEOH" in base_m:
                    props["base_model"] = MAT_YEOH
                    props["c10"] = float(self.extra_params.get("c10", self.hyperelastic.get("c10", 0.1) if self.hyperelastic else 0.1))
                    props["c20"] = float(self.extra_params.get("c20", self.hyperelastic.get("c20", 0.0) if self.hyperelastic else 0.0))
                    props["c30"] = float(self.extra_params.get("c30", self.hyperelastic.get("c30", 0.0) if self.hyperelastic else 0.0))
                else:
                    props["base_model"] = MAT_NEO_HOOKEAN
                    mu_val = float(self.extra_params.get("mu", 0.2))
                    K_val = float(self.extra_params.get("K", 10.0))
                    if self.hyperelastic:
                        if "c10" in self.hyperelastic:
                            mu_val = 2.0 * float(self.hyperelastic["c10"])
                        if "d1" in self.hyperelastic and float(self.hyperelastic["d1"]) > 1e-12:
                            K_val = 2.0 / float(self.hyperelastic["d1"])
                    props["mu"] = mu_val
                    props["K"] = K_val
                props["prony"] = prony
                props["E"] = float(E_val)
                props["nu"] = float(nu_val)
            else:
                props["mat_type"] = MAT_VISCOELASTIC_PRONY
                props["E"] = float(E_val)
                props["nu"] = float(nu_val)
                if prony and len(prony) > 0:
                    props["g1"] = float(prony[0][0])
                    props["tau1"] = float(prony[0][2] if len(prony[0]) > 2 else prony[0][1])
                else:
                    props["g1"] = float(self.extra_params.get("g1", 0.2))
                    props["tau1"] = float(self.extra_params.get("tau1", 1.0))
        else:
            props["mat_type"] = MAT_CUSTOM_ELASTIC
            
        if self.expansion is not None:
            props["expansion"] = self.expansion
            exp_table = self.expansion.get("table", [])
            if len(exp_table) > 0:
                props["alpha"] = float(exp_table[0][0])
                props["T0"] = float(self.expansion.get("zero", 0.0))
                if len(exp_table[0]) >= 2:
                    props["expansion_table"] = exp_table
            else:
                props["alpha"] = 0.0
                props["T0"] = 0.0

        for k, v in self.extra_params.items():
            if k not in props:
                props[k] = v
                
        return props
