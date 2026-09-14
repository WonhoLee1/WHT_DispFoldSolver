"""
pressure_overclosure.py
=========================
Shared pressure-overclosure law seam for 3D contact (design doc sec B.6,
dev_log/contact_abaqus_grade_design_20260915.md). Every normal-contact
law -- today's hard penalty, the new nonlinear (4-region) penalty, and the
three soft laws (linear/exponential/tabular) -- reduces to one interface:

    evaluate(h) -> (p, dp_dh)

where `h` = overclosure = penetration = -gap (this codebase's existing
sign convention, matching SurfaceContactConstraint3D.assemble()'s own
`penetration = -gap`). `p >= 0` is the contact pressure magnitude; `dp_dh`
is its tangent, used as the diagonal stiffness contribution.

SurfaceContactConstraint3D.assemble()'s downstream code (sign convention,
3x3 stiffness scatter, stats dict) consumes only the scalar (p, dp_dh)
pair and does not care which law produced it -- confirmed by reading
assemble() end to end before this module was added (design doc sec B.6).
"""

from __future__ import annotations
from typing import Tuple, Sequence
import numpy as np


class HardLaw:
    """Today's hard-contact linear penalty: p(h) = k*(h+c0) for h>-c0, else
    0. c0=0 (default) reproduces the exact pre-existing behavior
    (`f_mag = k_contact * penetration` for gap<0) bit-for-bit."""

    def __init__(self, k_contact: float, c0: float = 0.0):
        self.k_contact = float(k_contact)
        self.c0 = float(c0)

    def evaluate(self, h: float) -> Tuple[float, float]:
        hp = h + self.c0
        if hp <= 0.0:
            return 0.0, 0.0
        return self.k_contact * hp, self.k_contact


class NonlinearPenaltyLaw:
    """Abaqus's 4-region nonlinear penalty (design doc sec A.2), C1
    (value+slope continuous) at both breakpoints `e` and `d`:

        h' = h + c0
        p(h') = 0                                           h' <= 0
        p(h') = K_i*h'                                       0 <= h' <= e
        p(h') = K_i*h' + (K_f-K_i)/(2*(d-e)) * (h'-e)^2      e <= h' <= d
        p(h') = K_i*d + (K_f-K_i)*(d-e)/2 + K_f*(h'-d)       h' > d
    """

    def __init__(self, k_i: float, k_f: float, e: float, d: float, c0: float = 0.0):
        if not (0.0 < e < d):
            raise ValueError(f"NonlinearPenaltyLaw requires 0 < e < d, got e={e}, d={d}")
        self.k_i = float(k_i)
        self.k_f = float(k_f)
        self.e = float(e)
        self.d = float(d)
        self.c0 = float(c0)

    def evaluate(self, h: float) -> Tuple[float, float]:
        hp = h + self.c0
        if hp <= 0.0:
            return 0.0, 0.0
        if hp <= self.e:
            return self.k_i * hp, self.k_i
        if hp <= self.d:
            ramp = (self.k_f - self.k_i) / (self.d - self.e)
            slope = self.k_i + ramp * (hp - self.e)
            p = self.k_i * hp + 0.5 * ramp * (hp - self.e) ** 2
            return p, slope
        p_at_d = self.k_i * self.d + 0.5 * (self.k_f - self.k_i) * (self.d - self.e)
        return p_at_d + self.k_f * (hp - self.d), self.k_f


class LinearSoftLaw:
    """Abaqus SOFT_LINEAR (design doc sec B.2): p(h) = k*(h+c0) for
    h>-c0, else 0 -- same shape as HardLaw, `k` user/auto-scale derived
    instead of the hard-contact default. (Reconciled sign convention: the
    design doc's own worked derivation of the exponential law, and its
    nonlinear-penalty section, both activate at h>=-c0; this class follows
    that same, internally-consistent convention so c0=0 reproduces hard
    contact exactly and all three laws share one activation-threshold
    meaning.)"""

    def __init__(self, k: float, c0: float = 0.0):
        self.k = float(k)
        self.c0 = float(c0)

    def evaluate(self, h: float) -> Tuple[float, float]:
        hp = h + self.c0
        if hp <= 0.0:
            return 0.0, 0.0
        return self.k * hp, self.k


class ExponentialSoftLaw:
    """Abaqus SOFT_EXPONENTIAL (design doc sec B.2/B.7.3). In clearance
    space c = -h:

        p(c) = p0 * (exp(alpha*(c0-c)) - 1) / (exp(alpha*c0) - 1)   c < c0
        p(c) = 0                                                     c >= c0

    satisfying Abaqus's own two stated boundary conditions p(c0)=0,
    p(0)=p0 for any alpha>0 -- alpha is a free shape constant Abaqus's
    published docs don't fix (design doc sec B.2's "source gap"). Sec
    B.7.3 derives `alpha = ln(10)/c0` from first principles (touching
    stiffness reaches 10x itself over a penetration depth of c0) -- this
    is a PURE FUNCTION OF c0 alone, independent of p0/softness_scale, so
    it applies identically whether (c0, p0) came from the softness_scale/
    allowable_penetration auto-derivation or were given directly as an
    explicit override (design doc sec B.7.5 step 2 says "use them
    verbatim" but doesn't separately specify alpha for that path -- this
    is the resolution: alpha is always a function of c0 alone, so
    "verbatim (c0, p0)" is already fully sufficient to construct the law).
    """

    def __init__(self, c0: float, p0: float):
        if c0 <= 0.0:
            raise ValueError(
                f"ExponentialSoftLaw requires c0 > 0 (a zero-width ramp has no "
                f"exponential shape to speak of -- use HardLaw instead), got c0={c0}"
            )
        if p0 <= 0.0:
            raise ValueError(f"ExponentialSoftLaw requires p0 > 0, got p0={p0}")
        self.c0 = float(c0)
        self.p0 = float(p0)
        self.alpha = np.log(10.0) / self.c0
        self._denom = np.exp(self.alpha * self.c0) - 1.0  # == 9.0 exactly, kept symbolic for clarity

    def evaluate(self, h: float) -> Tuple[float, float]:
        c = -h
        if c >= self.c0:
            return 0.0, 0.0
        p = self.p0 * (np.exp(self.alpha * (self.c0 - c)) - 1.0) / self._denom
        dp_dc = -self.p0 * self.alpha * np.exp(self.alpha * (self.c0 - c)) / self._denom
        dp_dh = -dp_dc  # h = -c
        return float(p), float(dp_dh)


class TabularSoftLaw:
    """Abaqus SOFT_TABULAR (design doc sec B.2): user-supplied increasing
    (h_i, p_i) pairs. p=0 for h<=h_1; linear interpolation for
    h_1<=h<=h_n; linear extrapolation using the last segment's slope for
    h>h_n (matches Abaqus's own stated extrapolation rule -- np.interp's
    default clamping does NOT do this, hence the explicit branches)."""

    def __init__(self, h_pts: Sequence[float], p_pts: Sequence[float]):
        h_arr = np.asarray(h_pts, dtype=np.float64)
        p_arr = np.asarray(p_pts, dtype=np.float64)
        if len(h_arr) < 2 or len(h_arr) != len(p_arr):
            raise ValueError("TabularSoftLaw needs >= 2 matching (h, p) points")
        if np.any(np.diff(h_arr) <= 0.0):
            raise ValueError("TabularSoftLaw's h_pts must be strictly increasing")
        self.h_pts = h_arr
        self.p_pts = p_arr

    def evaluate(self, h: float) -> Tuple[float, float]:
        if h <= self.h_pts[0]:
            return 0.0, 0.0
        if h >= self.h_pts[-1]:
            slope = (self.p_pts[-1] - self.p_pts[-2]) / (self.h_pts[-1] - self.h_pts[-2])
            return float(self.p_pts[-1] + slope * (h - self.h_pts[-1])), float(slope)
        idx = int(np.searchsorted(self.h_pts, h)) - 1
        idx = max(0, min(idx, len(self.h_pts) - 2))
        h0, h1 = self.h_pts[idx], self.h_pts[idx + 1]
        p0, p1 = self.p_pts[idx], self.p_pts[idx + 1]
        slope = (p1 - p0) / (h1 - h0)
        return float(p0 + slope * (h - h0)), float(slope)
