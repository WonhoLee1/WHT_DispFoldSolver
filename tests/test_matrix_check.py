"""*MATRIX CHECK: a free body's stiffness annihilates its rigid-body modes."""
import numpy as np
import pytest
import scipy.sparse as sp

from dispsolver.solver.matrix_check import matrix_check, rigid_body_modes


def _truss_K(X, bars, EA=1.0):
    n, dim = X.shape
    K = np.zeros((n * dim, n * dim))
    for i, j in bars:
        d = X[j] - X[i]
        L = np.linalg.norm(d)
        e = d / L
        k = EA / L * np.outer(e, e)
        for a, b, s in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            K[a * dim:(a + 1) * dim, b * dim:(b + 1) * dim] += s * k
    return K


# a triangulated 2D patch is rotationally invariant to first order
X2 = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0], [1.0, 0.5]])
BARS2 = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]


def test_mode_counts():
    n2, R2 = rigid_body_modes(X2)
    assert n2 == ["UX", "UY", "ROT"] and R2.shape == (10, 3)
    X3 = np.random.default_rng(0).random((6, 3))
    n3, R3 = rigid_body_modes(X3)
    assert len(n3) == 6 and R3.shape == (18, 6)
    assert len(rigid_body_modes(X3, rotations=False)[0]) == 3


def test_free_truss_passes_dense_and_sparse():
    K = _truss_K(X2, BARS2)
    assert matrix_check(K, X2).passed
    assert matrix_check(sp.csr_matrix(K), X2).passed


def test_grounded_spring_fails_translation():
    K = _truss_K(X2, BARS2)
    K[0, 0] += 10.0                                   # a spring to ground on node 0 / x
    rep = matrix_check(K, X2)
    assert not rep.passed
    assert not {m.name: m.passed for m in rep.modes}["UX"]
    assert "NOT NULL" in str(rep)


def test_non_invariant_stiffness_fails_rotation_only():
    K = _truss_K(X2, BARS2)
    K[1::2, 1::2] *= 2.0                              # stiffen only the y-y block: not isotropic
    rep = {m.name: m.passed for m in matrix_check(K, X2).modes}
    assert rep["UX"] and rep["UY"] and not rep["ROT"]
    assert matrix_check(K, X2, rotations=False).passed


def test_dof_layout_mismatch_is_an_error():
    with pytest.raises(ValueError):
        matrix_check(np.eye(7), X2)


def test_real_c3d8_tangent_at_reference_state_passes(tmp_path):
    from dispsolver.io import execute_inp
    deck = """*HEADING
cube
*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT, TYPE=C3D8, ELSET=EL
1, 1,2,3,4,5,6,7,8
*MATERIAL, NAME=M
*ELASTIC
1000.0, 0.3
*SOLID SECTION, ELSET=EL, MATERIAL=M
*STEP, NAME=S, NLGEOM=NO
*STATIC
1.0, 1.0, 1e-4, 1.0
*BOUNDARY
1, 1, 3, 0.0
2, 2, 3, 0.0
4, 1, 1, 0.0
4, 3, 3, 0.0
5, 1, 2, 0.0
*CLOAD
8, 1, 1.0
*END STEP
"""
    path = tmp_path / "cube.inp"
    path.write_text(deck)
    res = execute_inp(str(path), dim=3, verbose=False)
    s = res["solver"]
    K, _ = s.assemble_system(np.zeros(s.num_dofs))
    rep = matrix_check(K, np.asarray(s.mesh.nodes_array())[:, :3], tol=1e-9)
    assert rep.passed, str(rep)


_CUBE = """*HEADING
cube
*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT, TYPE={et}, ELSET=EL
1, 1,2,3,4,5,6,7,8
*MATERIAL, NAME=M
*ELASTIC
1000.0, {nu}
*SOLID SECTION, ELSET=EL, MATERIAL=M
*STEP, NAME=S, NLGEOM={nl}
*STATIC
1.0, 1.0, 1e-4, 1.0
*BOUNDARY
1, 1, 3, 0.0
2, 2, 3, 0.0
4, 1, 1, 0.0
4, 3, 3, 0.0
5, 1, 2, 0.0
*CLOAD
8, 1, 1.0
*END STEP
"""


@pytest.mark.parametrize("et,nu", [("C3D8", 0.3), ("C3D8R", 0.3), ("C3D8I", 0.3),
                                   ("C3D8H", 0.499), ("C3D8IH", 0.499)])
@pytest.mark.parametrize("nlgeom", ["NO", "YES"])
def test_hex_element_tangents_null_the_rigid_body_modes(tmp_path, et, nu, nlgeom):
    """Measured 2026-10-20: every hex kernel nulls all 6 modes to <= 5e-16 at the reference state."""
    from dispsolver.io import execute_inp
    path = tmp_path / "cube.inp"
    path.write_text(_CUBE.format(et=et, nu=nu, nl=nlgeom))
    s = execute_inp(str(path), dim=3, verbose=False)["solver"]
    K, _ = s.assemble_system(np.zeros(s.num_dofs))
    rep = matrix_check(K, np.asarray(s.mesh.nodes_array())[:, :3], tol=1e-9)
    assert rep.passed, f"{et} nu={nu} NLGEOM={nlgeom}\n{rep}"
