"""
contact_auto_tuner.py
======================
Commercial-Grade Autonomous In-Situ Contact Auto-Tuner & Parameter Recommendation Engine.
Detects contact non-convergence, chattering, and severe discontinuity iterations (SDI),
autonomously relaxes contact stiffness and pre-contact cushioning, locks in optimal parameters,
and outputs recommended script settings for future runs.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from dispsolver.constraint3d.pressure_overclosure import AdaptivePowerSoftLaw


class ContactAutoTuner:
    """Autonomous In-Situ Contact Auto-Tuner.

    Monitors Newton-Raphson convergence and autonomously relaxes contact parameters
    (lead cushioning, stiffness softness scale, and stabilization damping) to heal
    difficult contact increments on-the-fly without aborting or restarting the analysis.

    Parameters
    ----------
    solver : Any
        DynamicSolver3D or DynamicSolver2D instance.
    max_tune_attempts : int, optional
        Maximum tuning attempts per increment (default 4).
    verbose : bool, optional
        Whether to print autonomous tuning actions and guides to stdout (default True).
    """

    def __init__(
        self,
        solver: Any,
        max_tune_attempts: int = 4,
        verbose: bool = True,
    ):
        self.solver = solver
        self.max_tune_attempts = int(max_tune_attempts)
        self.verbose = bool(verbose)

        self.current_attempt = 0
        self.tuning_history: List[Dict[str, Any]] = []

        # Current active tuned parameters
        self.current_lead: float = 0.001
        self.current_softness: float = 0.1
        self.current_power: float = 2.0
        self.current_damping: float = 0.0
        self.current_delta: float = 0.02
        self.base_gamma: float = 1.0e7

        # Initialize from solver's existing contact constraints
        self._inspect_existing_contacts()

    def _inspect_existing_contacts(self) -> None:
        """Read initial contact configuration from solver constraints."""
        for c in getattr(self.solver, "constraints", []):
            if hasattr(c, "gamma"):
                self.base_gamma = float(c.gamma)
            if hasattr(c, "law") and isinstance(c.law, AdaptivePowerSoftLaw):
                self.current_lead = float(c.law.c0)
                self.current_delta = float(c.law.delta)
                self.current_power = float(c.law.power)
                if self.base_gamma > 0.0:
                    self.current_softness = float(c.law.k_target / self.base_gamma)
            if hasattr(c, "damping_coeff") and c.damping_coeff > 0.0:
                self.current_damping = float(c.damping_coeff)

    def tune_and_relax(self, reason: str = "chattering") -> bool:
        """Autonomously tune and relax contact stiffness and pre-contact lead."""
        if self.current_attempt >= self.max_tune_attempts:
            return False

        self.current_attempt += 1

        old_lead = self.current_lead
        old_softness = self.current_softness
        old_damping = self.current_damping

        # Heuristic tuning protocol:
        # 1. Expand pre-contact lead by 1.6x to smooth out impulse shock
        self.current_lead = max(1.0e-5, self.current_lead * 1.6)

        # 2. Relax stiffness softness scale by 0.55x to prevent penetration bounce
        self.current_softness = max(1.0e-4, self.current_softness * 0.55)

        # 3. Add or boost viscous contact stabilization damping
        if self.current_damping <= 0.0:
            self.current_damping = 1.0e-3 * self.base_gamma
        else:
            self.current_damping *= 1.5

        # Compute new target stiffness
        k_target = self.current_softness * self.base_gamma

        # Apply updated law to all general contact constraints in solver
        updated_any = False
        for c in getattr(self.solver, "constraints", []):
            if hasattr(c, "law"):
                new_law = AdaptivePowerSoftLaw(
                    k_target=k_target,
                    c0=self.current_lead,
                    delta=self.current_delta,
                    power=self.current_power,
                )
                c.law = new_law
                if hasattr(c, "damping_coeff"):
                    c.damping_coeff = self.current_damping
                if hasattr(c, "damping_clearance"):
                    c.damping_clearance = max(2.0 * self.current_lead, getattr(c, "skin", 0.05))
                updated_any = True

        tune_info = {
            "attempt": self.current_attempt,
            "reason": reason,
            "old_lead": old_lead,
            "new_lead": self.current_lead,
            "old_softness": old_softness,
            "new_softness": self.current_softness,
            "new_damping": self.current_damping,
        }
        self.tuning_history.append(tune_info)

        if self.verbose:
            print(f"\n⚡ [Contact Auto-Tuner] Non-convergence / {reason} detected! Autonomous Relaxation (Attempt {self.current_attempt}/{self.max_tune_attempts}):")
            print(f"   - Pre-contact lead (c0): {old_lead:.4g} mm -> {self.current_lead:.4g} mm (cushioning expanded)")
            print(f"   - Softness scale: {old_softness:.4g} -> {self.current_softness:.4g} (stiffness softened)")
            print(f"   - Contact damping: {old_damping:.4g} -> {self.current_damping:.4g} N*s/mm^3")

        return updated_any

    def reset_attempt_counter(self) -> None:
        """Reset increment retry attempt counter while preserving tuned parameters."""
        self.current_attempt = 0

    def get_recommendation_guide(self) -> Dict[str, Any]:
        """Generate final optimal parameter recommendation and copy-pasteable script guide."""
        code_snippet = (
            f"# ✨ Recommended Contact Setting (Optimized by Contact Auto-Tuner):\n"
            f"prop = model.ContactProperty('TunedContactProp')\n"
            f"prop.AdaptiveSoftBehavior(\n"
            f"    lead={self.current_lead:.5g},          # Pre-contact lead cushioning\n"
            f"    softnessScale={self.current_softness:.4g},   # Softness scale factor\n"
            f"    power={self.current_power:.1f},            # Power-law exponent\n"
            f"    allowablePenetration={self.current_delta:.5g},\n"
            f")\n"
        )
        if self.current_damping > 0.0:
            code_snippet += (
                f"prop.Damping(\n"
                f"    definition=DAMPING_COEFFICIENT,\n"
                f"    dampingCoefficient={self.current_damping:.4g},\n"
                f"    clearance={2.0 * self.current_lead:.5g}\n"
                f")\n"
            )

        return {
            "optimal_lead": self.current_lead,
            "optimal_softness_scale": self.current_softness,
            "optimal_power": self.current_power,
            "optimal_damping": self.current_damping,
            "optimal_delta": self.current_delta,
            "total_tunings_performed": len(self.tuning_history),
            "code_snippet": code_snippet,
        }

    def print_guide(self) -> None:
        """Print clean, user-friendly markdown guide for next run."""
        rec = self.get_recommendation_guide()
        print("\n" + "=" * 70)
        print("🎯 [Contact Auto-Tuner] Parameter Optimization Summary & Script Guide")
        print("=" * 70)
        print(f"  • Total autonomous adjustments: {rec['total_tunings_performed']}")
        print(f"  • Established Pre-contact Lead : {rec['optimal_lead']:.5g} mm")
        print(f"  • Established Softness Scale   : {rec['optimal_softness_scale']:.4g}")
        print(f"  • Established Power Exponent   : {rec['optimal_power']:.1f}")
        print("-" * 70)
        print("📋 [Copy & Paste into your Script for Next Run]:\n")
        print(rec["code_snippet"])
        print("=" * 70 + "\n")
