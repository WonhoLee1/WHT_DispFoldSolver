"""Probe the Q4_COROTATIONAL_EAS failure at AR=12.5 (free-span PET row)."""
import numpy as np
from scratch.locking_ar_test import run_case  # reuse builder

cases = [
    ("AR= 8.3 L=5.0  h=0.030", dict(L=5.0, h=0.03, nx=20, ny=1)),
    ("AR=10.0 L=5.0  h=0.025", dict(L=5.0, h=0.025, nx=20, ny=1)),
    ("AR=11.0 L=5.0  h=0.0227", dict(L=5.0, h=0.0227, nx=20, ny=1)),
    ("AR=12.5 L=5.0  h=0.020", dict(L=5.0, h=0.02, nx=20, ny=1)),
    ("AR=12.5 L=2.5  h=0.020 (half len)", dict(L=2.5, h=0.02, nx=10, ny=1)),
    ("AR=12.5 L=5.0  h=0.020 P/100", dict(L=5.0, h=0.02, nx=20, ny=1, P=1e-8)),
    ("AR=12.5 L=5.0  h=0.020 ny=2", dict(L=5.0, h=0.02, nx=20, ny=2)),
    ("AR=16.7 L=5.0  h=0.015", dict(L=5.0, h=0.015, nx=20, ny=1)),
]
for label, kw in cases:
    for et in ("Q4_COROTATIONAL_EAS", "Q4_COROTATIONAL_SRI"):
        try:
            code, tip, th, ratio = run_case(et, **kw)
            print(f"{label:36s} {et:22s} conv_code={code:4d} tip={tip:12.4e} theory={th:12.4e} ratio={ratio:8.2f}")
        except Exception as e:
            print(f"{label:36s} {et:22s} ERROR {type(e).__name__}: {str(e)[:70]}")
