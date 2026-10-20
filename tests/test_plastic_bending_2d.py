"""Plastic pure bending of a plane-strain strip: finite elements vs an independent fibre oracle (the PET path).

Oracle (numpy, no solver code): a wide plate in pure bending has eps_x = kappa*y, eps_z = 0 and a free surface
sigma_y = 0 in every fibre. For each fibre the 3D small-strain J2 return map (linear isotropic hardening) is run
incrementally and eps_y is solved so that sigma_y = 0. The moment is the sum of sigma_x * y over the fibres.
"""
import numpy as np
import pytest

from dispsolver.io import execute_inp

E, NU, SY0, H = 1000.0, 0.3, 10.0, 100.0
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))


def fibre_stress_path(ex_path):
    """sigma_x of one fibre along a strain path eps_x(t); eps_z = 0, sigma_y = 0."""
    ep = np.zeros(3)          # plastic strain xx, yy, zz (no shear in this state)
    ebar = 0.0
    out = []

    def respond(ex, ey, ep, ebar):
        e = np.array([ex, ey, 0.0]) - ep
        tr = e.sum()
        sig = LAM * tr + 2 * MU * e
        s = sig - sig.sum() / 3.0
        q = np.sqrt(1.5 * s @ s)
        yld = SY0 + H * ebar
        if q <= yld:
            return sig, ep, ebar
        dg = (q - yld) / (3 * MU + H)
        n = 1.5 * s / q
        return sig - 2 * MU * dg * n, ep + dg * n, ebar + dg

    ey = 0.0
    for ex in ex_path:
        # Newton / secant on ey so that sigma_y = 0 (monotone in ey)
        for _ in range(60):
            sy = respond(ex, ey, ep, ebar)[0][1]
            if abs(sy) < 1e-12:
                break
            d = 1e-7
            slope = (respond(ex, ey + d, ep, ebar)[0][1] - sy) / d
            ey -= sy / slope
        sig, ep, ebar = respond(ex, ey, ep, ebar)
        out.append(sig[0])
    return np.array(out)


def oracle_moment(kappa, h=1.0, nfib=200, nsteps=200):
    y = (np.arange(nfib) + 0.5) / nfib * h - h / 2
    m = 0.0
    for yi in y:
        path = np.linspace(0.0, kappa * yi, nsteps + 1)[1:]
        sx = fibre_stress_path(path)[-1]
        m += sx * yi * (h / nfib)
    return m


# ---- finite element strip ---------------------------------------------------------------------------------
def build_mesh(etype, L, h, nx, ny):
    quad = etype in ("CPE8", "CPE8R", "CPE6", "CPE6M")
    sx, sy = (2 * nx, 2 * ny) if quad else (nx, ny)
    xs, ys = np.linspace(0, L, sx + 1), np.linspace(0, h, sy + 1)
    ids, nodes = {}, []
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            ids[(i, j)] = len(nodes) + 1
            nodes.append((xv, yv))
    step = 2 if quad else 1
    elems = []
    for jy in range(ny):
        for ix in range(nx):
            i0, j0 = ix * step, jy * step
            c = [(i0, j0), (i0 + step, j0), (i0 + step, j0 + step), (i0, j0 + step)]
            if etype in ("CPE4", "CPE4I", "CPE4R", "CPE4H"):
                elems.append([ids[p] for p in c])
            elif etype in ("CPE8", "CPE8R"):
                mids = [(i0 + 1, j0), (i0 + 2, j0 + 1), (i0 + 1, j0 + 2), (i0, j0 + 1)]
                elems.append([ids[p] for p in c + mids])
            elif etype == "CPE3":
                elems.append([ids[c[0]], ids[c[1]], ids[c[2]]])
                elems.append([ids[c[0]], ids[c[2]], ids[c[3]]])
            elif etype in ("CPE6", "CPE6M"):
                m01 = (i0 + 1, j0); m12 = (i0 + 2, j0 + 1); m23 = (i0 + 1, j0 + 2); m30 = (i0, j0 + 1); m02 = (i0 + 1, j0 + 1)
                elems.append([ids[c[0]], ids[c[1]], ids[c[2]], ids[m01], ids[m12], ids[m02]])
                elems.append([ids[c[0]], ids[c[2]], ids[c[3]], ids[m02], ids[m23], ids[m30]])
    return nodes, elems


def deck(etype, kappa, L=10.0, h=1.0, nx=20, ny=8, nsteps=10):
    nodes, elems = build_mesh(etype, L, h, nx, ny)
    nn = len(elems[0])
    phi = kappa * L
    yc = h / 2
    L_ = ["*HEADING", "plastic pure bending", "*NODE"] + [f"{i + 1}, {float(p[0])!r}, {float(p[1])!r}" for i, p in enumerate(nodes)]
    L_ += [f"*ELEMENT, TYPE={etype}, ELSET=EL"] + [f"{k + 1}, " + ", ".join(str(n) for n in el) for k, el in enumerate(elems)]
    L_ += ["*MATERIAL, NAME=M", "*ELASTIC", f"{E}, {NU}", "*PLASTIC", f"{SY0}, 0.0", f"{SY0 + H * 0.5}, 0.5",
           "*SOLID SECTION, ELSET=EL, MATERIAL=M"]
    left = [i for i, p in enumerate(nodes) if abs(p[0]) < 1e-12]
    right = [i for i, p in enumerate(nodes) if abs(p[0] - L) < 1e-12]
    for k in range(1, nsteps + 1):
        th = phi * k / nsteps / 2.0             # each end section turns by +-phi/2
        R = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        chord = (2.0 * (L / (phi * k / nsteps)) * np.sin(phi * k / nsteps / 2.0) - L) if phi > 0 else 0.0
        L_ += [f"*STEP, NAME=B{k}, NLGEOM=YES", "*STATIC", "1.0, 1.0, 1e-5, 1.0", "*BOUNDARY"]
        for i in left:
            X = np.array(nodes[i]); Xc = np.array([0.0, yc])
            u = R(-th) @ (X - Xc) + Xc - X
            L_ += [f"{i + 1}, 1, 1, {float(u[0])!r}", f"{i + 1}, 2, 2, {float(u[1])!r}"]
        for i in right:
            X = np.array(nodes[i]); Xc = np.array([L, yc])
            u = R(th) @ (X - Xc) + Xc - X + np.array([chord, 0.0])
            L_ += [f"{i + 1}, 1, 1, {float(u[0])!r}", f"{i + 1}, 2, 2, {float(u[1])!r}"]
        L_ += ["*END STEP"]
    return "\n".join(L_) + "\n", nodes, left


def fe_moment(etype, kappa, tmp, **kw):
    text, nodes, left = deck(etype, kappa, **kw)
    p = tmp / "b.inp"
    p.write_text(text)
    r = execute_inp(str(p), dim=2, verbose=False)
    f = np.asarray(r["solver"].f_int).reshape(-1, 2)
    y = np.array([nodes[i][1] for i in left])
    return float(abs((f[left, 0] * (y - 0.5)).sum())), r["converged"]




# ---- tests ---------------------------------------------------------------------------------------------------

def _elastic_moment(kappa, h=1.0):
    return E / (1.0 - NU ** 2) * kappa * h ** 3 / 12.0


def test_the_fibre_oracle_reproduces_plane_strain_elastic_bending():
    assert oracle_moment(0.005) == pytest.approx(_elastic_moment(0.005), rel=1e-3)


def test_plasticity_softens_the_oracle_moment():
    # fully elastic at 0.01, clearly below the elastic line once the outer fibres are at twice the yield strain
    assert oracle_moment(0.01) == pytest.approx(_elastic_moment(0.01), rel=1e-3)
    assert oracle_moment(0.04) < 0.8 * _elastic_moment(0.04)


# Elements whose bending is accurate (bilinear full integration CPE4 and the mean-dilatation CPE4H lock in
# bending, +9% and +8% elastically, and are left out on purpose). Bounds are the measured accuracy of each element.
_BENDING = [("CPE4I", 0.02), ("CPE4R", 0.03), ("CPE8R", 0.03)]


@pytest.mark.parametrize("etype, bound", _BENDING)
def test_plastic_bending_converges_and_matches_the_oracle(tmp_path, etype, bound):
    kappa = 0.04                                   # outer fibres at twice the yield strain
    moment, converged = fe_moment(etype, kappa, tmp_path)
    assert converged, f"{etype} did not converge to kappa = {kappa} in plastic bending"
    assert moment == pytest.approx(oracle_moment(kappa), rel=bound)
