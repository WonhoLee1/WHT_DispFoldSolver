"""*ORIENTATION with orthotropic ENGINEERING CONSTANTS (PRD: dev_log/prd_orientation_20261018.md).

Method: prescribe the uniform displacement field u = eps . x on every node of a unit cube. The strain is then
exactly eps in the element, and the nodal reactions are the stress times the face area, so
    sum of f on the face x = 1  ->  sigma . e_x        (and likewise for the other faces)
whatever the element integration does. The expected stress is C' eps with C' from the physical definition of
the material (compliance from the engineering constants, inverted with numpy; the stiffness rotated as a
fourth-order tensor), never from the code under test.
"""
import numpy as np
import pytest

from dispsolver.io import execute_inp

# E1 E2 E3 nu12 nu13 nu23 G12 G13 G23  (a stable orthotropic set)
EC = (20000.0, 5000.0, 4000.0, 0.30, 0.25, 0.35, 2500.0, 2000.0, 1500.0)
_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def _stiffness_voigt(ec=EC) -> np.ndarray:
    """6x6 C in the order xx yy zz xy yz xz with engineering shear strain."""
    E1, E2, E3, n12, n13, n23, G12, G13, G23 = ec
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1] = S[1, 0] = -n12 / E1
    S[0, 2] = S[2, 0] = -n13 / E1
    S[1, 2] = S[2, 1] = -n23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1 / G12, 1 / G23, 1 / G13
    return np.linalg.inv(S)


_V = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]


def _rotate(C6: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """C' = R R R R C with R = frame^T (columns are the local axes in global coordinates)."""
    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_V):
        for J, (k, l) in enumerate(_V):
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    C4[a, b, c, d] = C6[I, J]
    R = frame.T
    G4 = np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, C4)
    return np.array([[G4[i, j, k, l] for (k, l) in _V] for (i, j) in _V])


def _frame(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), np.sin(t), 0.0], [-np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def _deck(strain: np.ndarray, orientation: str = "", section_orientation: bool = False, ec=EC) -> str:
    lines = ["*HEADING", "orthotropic cube", "*NODE"]
    lines += [f"{i + 1}, {p[0]}, {p[1]}, {p[2]}" for i, p in enumerate(_CUBE)]
    lines += ["*ELEMENT, TYPE=C3D8, ELSET=EL", "1, 1,2,3,4,5,6,7,8"]
    lines += [orientation] if orientation else []
    lines += ["*MATERIAL, NAME=M", "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
              ", ".join(repr(v) for v in ec[:8]) + ",", repr(ec[8]),
              "*SOLID SECTION, ELSET=EL, MATERIAL=M" + (", ORIENTATION=ORI" if section_orientation else ""),
              "*STEP, NAME=S1, NLGEOM=NO", "*STATIC", "1.0, 1.0", "*BOUNDARY"]
    for i, p in enumerate(_CUBE):
        u = strain @ np.array(p, dtype=float)
        for d in range(3):
            lines.append(f"{i + 1}, {d + 1}, {d + 1}, {float(u[d])!r}")
    lines += ["*END STEP", ""]
    return "\n".join(lines)


def _stress_columns(tmp_path, deck: str) -> np.ndarray:
    """Rows: sigma . e_x, sigma . e_y, sigma . e_z from the reactions on the faces x=1, y=1, z=1."""
    path = tmp_path / "deck.inp"
    path.write_text(deck)
    res = execute_inp(str(path), dim=3, verbose=False)
    assert res["converged"]
    f = np.asarray(res["solver"].f_int).reshape(-1, 3)
    X = np.array(_CUBE, dtype=float)
    return np.array([f[X[:, a] > 0.5].sum(axis=0) for a in range(3)])


def _expected_stress(C6: np.ndarray, strain: np.ndarray) -> np.ndarray:
    eps_voigt = np.array([strain[0, 0], strain[1, 1], strain[2, 2],
                          2 * strain[0, 1], 2 * strain[1, 2], 2 * strain[0, 2]])
    s = C6 @ eps_voigt
    return np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])


_STRAIN = np.array([[1.0e-3, 0.4e-3, 0.0], [0.4e-3, -0.3e-3, 0.0], [0.0, 0.0, 0.2e-3]])


def test_s1_orthotropic_c3d8_without_an_orientation(tmp_path):
    got = _stress_columns(tmp_path, _deck(_STRAIN))
    np.testing.assert_allclose(got, _expected_stress(_stiffness_voigt(), _STRAIN), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("theta", [30.0, 90.0, -45.0])
def test_s2_rotated_about_the_third_axis(tmp_path, theta):
    t = np.radians(theta)
    ori = ("*ORIENTATION, NAME=ORI\n"
           f"{float(np.cos(t))!r}, {float(np.sin(t))!r}, 0.0, {float(-np.sin(t))!r}, {float(np.cos(t))!r}, 0.0\n3, 0.0")
    got = _stress_columns(tmp_path, _deck(_STRAIN, ori, section_orientation=True))
    expected = _expected_stress(_rotate(_stiffness_voigt(), _frame(theta)), _STRAIN)
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)
    # cross-check the test's own tensor rotation against the classical closed form for C'11
    C = _stiffness_voigt()
    m, n = np.cos(t), np.sin(t)
    c11 = m**4 * C[0, 0] + 2 * m**2 * n**2 * (C[0, 1] + 2 * C[3, 3]) + n**4 * C[1, 1]
    assert _rotate(C, _frame(theta))[0, 0] == pytest.approx(c11, rel=1e-10)


def test_s3_additional_rotation_on_the_second_line(tmp_path):
    t = np.radians(15.0)
    ori = ("*ORIENTATION, NAME=ORI\n"
           f"{float(np.cos(t))!r}, {float(np.sin(t))!r}, 0.0, {float(-np.sin(t))!r}, {float(np.cos(t))!r}, 0.0\n3, 15.0")
    got = _stress_columns(tmp_path, _deck(_STRAIN, ori, section_orientation=True))
    expected = _expected_stress(_rotate(_stiffness_voigt(), _frame(30.0)), _STRAIN)
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)


def test_s4_frames_that_vary_by_element_are_refused_with_the_reason(tmp_path):
    ori = "*ORIENTATION, NAME=ORI, SYSTEM=CYLINDRICAL\n0,0,0, 0,0,1\n1, 0.0"
    with pytest.raises(Exception, match="(?i)cylindrical"):
        _stress_columns(tmp_path, _deck(_STRAIN, ori, section_orientation=True))


def test_s3_additional_rotation_about_the_first_local_axis(tmp_path):
    a = np.radians(20.0)
    ori = "*ORIENTATION, NAME=ORI\n1.0, 0.0, 0.0, 0.0, 1.0, 0.0\n1, 20.0"
    got = _stress_columns(tmp_path, _deck(_STRAIN, ori, section_orientation=True))
    c, s = np.cos(a), np.sin(a)
    frame = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])     # e2' = c e2 + s e3 (right-hand about e1)
    np.testing.assert_allclose(got, _expected_stress(_rotate(_stiffness_voigt(), frame), _STRAIN), rtol=1e-8, atol=1e-8)


def test_s3_rotation_angle_without_a_direction_is_an_error(tmp_path):
    ori = "*ORIENTATION, NAME=ORI\n1.0, 0.0, 0.0, 0.0, 1.0, 0.0\n, 20.0"
    with pytest.raises(Exception, match="(?i)local direction"):
        _stress_columns(tmp_path, _deck(_STRAIN, ori, section_orientation=True))


def test_s6_an_isotropic_material_ignores_the_orientation(tmp_path):
    ori = "*ORIENTATION, NAME=ORI\n0.6, 0.8, 0.0, -0.8, 0.6, 0.0\n3, 0.0"
    deck = _deck(_STRAIN, ori, section_orientation=True)
    deck = deck.replace("*ELASTIC, TYPE=ENGINEERING CONSTANTS", "*ELASTIC").replace(
        ", ".join(repr(v) for v in EC[:8]) + ",\n" + repr(EC[8]), "1000.0, 0.3")
    got = _stress_columns(tmp_path, deck)
    lam, mu = 1000.0 * 0.3 / (1.3 * 0.4), 1000.0 / 2.6
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    np.testing.assert_allclose(got, _expected_stress(C, _STRAIN), rtol=1e-8, atol=1e-8)


def test_s5_two_d_orthotropic_elasticity_is_refused_with_the_reason(tmp_path):
    deck = (
        "*HEADING\n2D\n*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n*ELEMENT, TYPE=CPE4, ELSET=EL\n1, 1,2,3,4\n"
        "*ORIENTATION, NAME=ORI\n1,0,0, 0,1,0\n3, 0.0\n"
        "*MATERIAL, NAME=M\n*ELASTIC, TYPE=ENGINEERING CONSTANTS\n"
        + ", ".join(repr(v) for v in EC[:8]) + ",\n" + repr(EC[8]) + "\n"
        "*SOLID SECTION, ELSET=EL, MATERIAL=M, ORIENTATION=ORI\n"
        "*STEP, NAME=S1\n*STATIC\n1.0, 1.0\n*BOUNDARY\n1, 1, 2, 0.0\n*END STEP\n")
    path = tmp_path / "d2.inp"
    path.write_text(deck)
    with pytest.raises(Exception, match="(?i)2D"):
        execute_inp(str(path), dim=2, verbose=False)


def test_rotate_frame_follows_the_right_hand_rule():
    from dispsolver.material.orientation import rotate_frame
    I = np.eye(3)
    np.testing.assert_allclose(rotate_frame(I, 3, 90.0), [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(rotate_frame(I, 1, 90.0), [[1, 0, 0], [0, 0, 1], [0, -1, 0]], atol=1e-12)
    np.testing.assert_allclose(rotate_frame(I, 2, 90.0), [[0, 0, -1], [0, 1, 0], [1, 0, 0]], atol=1e-12)
    with pytest.raises(ValueError):
        rotate_frame(I, 4, 10.0)


def test_s5_c3d20_with_an_orientation_is_refused_not_ignored(tmp_path):
    # C3D20 reads the raw engineering constants through its own path and takes no section orientation yet;
    # solving it unrotated would be a different model than the deck describes.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    nodes = _CUBE + [tuple((np.array(_CUBE[a]) + np.array(_CUBE[b])) / 2.0) for a, b in edges]
    lines = ["*HEADING", "hex20", "*NODE"] + [f"{i + 1}, {p[0]}, {p[1]}, {p[2]}" for i, p in enumerate(nodes)]
    lines += ["*ELEMENT, TYPE=C3D20, ELSET=EL", "1, " + ", ".join(str(i + 1) for i in range(20)),
              "*ORIENTATION, NAME=ORI", "0.8660254, 0.5, 0.0, -0.5, 0.8660254, 0.0", "3, 0.0",
              "*MATERIAL, NAME=M", "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
              ", ".join(repr(v) for v in EC[:8]) + ",", repr(EC[8]),
              "*SOLID SECTION, ELSET=EL, MATERIAL=M, ORIENTATION=ORI",
              "*STEP, NAME=S1, NLGEOM=NO", "*STATIC", "1.0, 1.0", "*BOUNDARY", "1, 1, 3, 0.0", "*END STEP", ""]
    path = tmp_path / "h20.inp"
    path.write_text("\n".join(lines))
    with pytest.raises(NotImplementedError, match="(?i)C3D20.*orientation"):
        execute_inp(str(path), dim=3, verbose=False)
