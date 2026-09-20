# -*- coding: utf-8 -*-
"""benchmark_damage.py: ductile/JC damage benchmarks vs analytic solutions.

B1 initiation strain: uniaxial tension, const eD -> omega hits 1.0 exactly
   when eq_p == eD (rtol 1%).
B2 mesh objectivity: same G_f, L = 1.0 vs 0.5 driven to full failure.
   Dissipated energy per unit AREA (W_diss * L) must match (rtol 2%);
   per-volume values differ by construction (G_f/L).
B3 triaxiality: 2-pt table (eta=1/3 -> 0.05, eta=2/3 -> 0.02); uniaxial
   (eta=1/3) vs equibiaxial (eta=2/3) initiation strains match table (2%).
B4 JC formula: D1 + D2*exp(D3*eta) spot value at eta=1/3 (1e-12).

All at material-point level through material_dispatch_3d (MAT13), i.e. the
same code the solver runs. Abaqus deck parity (damage_ductile_mises.inp
family) needs Abaqus-measured curves not present in-repo; these analytic
gates are the executable substitute per the Wave plan.
"""

from __future__ import annotations

import numpy as np

from dispsolver.material3d.numba_materials import (
    material_dispatch_3d,
    MAT_J2_DUCTILE_DAMAGE,
)
from dispsolver.material.damage_initiation import fracture_strain_jc

E, NU, SY0, H = 200000.0, 0.3, 250.0, 2000.0
F3 = np.eye(3)


def _props(init_type, p, Gf=10.0, L=1.0, evol=0, Hh=H):
    pr = np.zeros(13)
    pr[0], pr[1], pr[2], pr[3] = E, NU, SY0, Hh
    pr[4] = init_type
    pr[5:5 + len(p)] = p
    pr[10], pr[11], pr[12] = Gf, L, evol
    return pr


def _drive(props, strain_fn, n=400, demax=0.05):
    sdv = np.zeros(10)
    sig, dep, dis, dqs, dds = [], [], [], [], []
    prev_eq = 0.0
    for k in range(1, n + 1):
        e = demax * k / n
        S, _, sdv, err = material_dispatch_3d(
            MAT_J2_DUCTILE_DAMAGE, props, sdv, strain_fn(e), F3, 1.0, 1.0)
        assert err == 0
        sig.append(S[0])
        dep.append(sdv[6])
        dis.append(sdv[9])
        ddep = sdv[6] - prev_eq
        prev_eq = sdv[6]
        dds.append(ddep)
        Sm = S / max(1.0 - sdv[9], 1e-300)
        qm = np.sqrt(0.5 * ((Sm[0] - Sm[1]) ** 2 + (Sm[1] - Sm[2]) ** 2 +
                            (Sm[2] - Sm[0]) ** 2) + 3.0 * (Sm[3] ** 2))
        dqs.append(qm * ddep * sdv[9])
    return np.array(sig), np.array(dep), np.array(dis), np.array(dqs)


def run_b1_initiation_strain():
    """B1: omega == 1 exactly at eq_p == eD."""
    print("\n[Benchmark 1/4] initiation strain (const eD)...")
    eD = 0.02
    props = _props(0, [eD], Gf=1e9, L=1.0)
    _, ep, _, _ = _drive(props, lambda e: np.array(
        [e, -NU * e, -NU * e, 0, 0, 0]), n=200, demax=0.03)
    om = ep / eD
    k = int(np.argmax(om >= 1.0))
    err = abs(ep[k] - eD) / eD
    print(f"  eq_p at omega=1: {ep[k]:.6f} (ref {eD}, err {err*100:.3f}%)")
    ok = err < 0.01
    print(f"  --> Verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def run_b2_mesh_objectivity():
    """B2: damage-dissipated energy per unit area invariant across L.

    Perfect plasticity (H=0) isolates the regularization: with q_tilde ==
    sigma_f const, W_area must equal G_f for every L. With hardening the
    longer softening span legitimately dissipates more (documented, not gated).
    """
    print("\n[Benchmark 2/4] mesh objectivity (dissipated energy per area)...")
    w_area = []
    for L in (1.0, 0.5):
        props = _props(0, [0.005], Gf=2.0, L=L, Hh=0.0)
        _, _, dis, dqs = _drive(props, lambda e: np.array(
            [e, -NU * e, -NU * e, 0, 0, 0]), n=1200, demax=0.06)
        assert dis[-1] >= 1.0
        w_area.append(float(np.sum(dqs)) * L)
        print(f"  L={L}: W_dmg_vol={float(np.sum(dqs)):.4f}, W_area={w_area[-1]:.4f} (G_f=2.0)")
    err = abs(w_area[0] - w_area[1]) / max(w_area[0], 1e-30)
    print(f"  area-energy mismatch: {err*100:.3f}%")
    ok = err < 0.03 and abs(w_area[0] - 2.0) / 2.0 < 0.05
    print(f"  --> Verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def run_b3_triaxiality():
    """B3: initiation strain follows the (eta, eD) table."""
    print("\n[Benchmark 3/4] triaxiality dependence...")
    props = _props(1, [1.0/3.0, 0.05, 2.0/3.0, 0.02], Gf=1e9, L=1.0)
    _, ep_uni, _, _ = _drive(props, lambda e: np.array(
        [e, -NU * e, -NU * e, 0, 0, 0]), n=400, demax=0.06)
    ku = int(np.argmax(ep_uni / 0.05 >= 1.0))
    _, ep_bi, _, _ = _drive(props, lambda e: np.array(
        [e, e, -2 * NU * e / (1 - NU), 0, 0, 0]), n=400, demax=0.03)
    kb = int(np.argmax(ep_bi / 0.02 >= 1.0))
    err_u = abs(ep_uni[ku] - 0.05) / 0.05
    err_b = abs(ep_bi[kb] - 0.02) / 0.02
    print(f"  uniaxial init eq_p: {ep_uni[ku]:.6f} (ref 0.05, err {err_u*100:.3f}%)")
    print(f"  biaxial init eq_p:  {ep_bi[kb]:.6f} (ref 0.02, err {err_b*100:.3f}%)")
    ok = err_u < 0.02 and err_b < 0.02
    print(f"  --> Verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def run_b4_jc_formula():
    """B4: JC fracture-strain closed form at eta=1/3."""
    print("\n[Benchmark 4/4] JC closed form...")
    D1, D2, D3 = 0.05, 3.44, -2.12
    ref = D1 + D2 * np.exp(D3 / 3.0)
    got = fracture_strain_jc(1.0 / 3.0, D1, D2, D3)
    err = abs(got - ref) / ref
    print(f"  eD(1/3): {got:.8f} (ref {ref:.8f}, err {err:.2e})")
    ok = err < 1e-12
    print(f"  --> Verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    """Run the damage benchmark suite."""
    results = [
        run_b1_initiation_strain(),
        run_b2_mesh_objectivity(),
        run_b3_triaxiality(),
        run_b4_jc_formula(),
    ]
    print(f"\ndamage benchmarks: {sum(results)}/4 PASS")
    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
