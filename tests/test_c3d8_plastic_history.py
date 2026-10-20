"""C3D8 (co-rotational B-bar kernels) must carry the plastic state from one increment to the next.

Found while testing restart: ``c3d8_corotational_numba`` evaluated the material and dropped ``sdv_new``, so the
plastic strain never accumulated and an elastic-plastic C3D8 behaved like a nonlinear elastic one.

Independent expected value: pull a bar well past yield, then return the end displacement to zero. A material
that remembers its plastic strain is left in compression there (the residual of eps_p > 0), while one that
does not remember retraces the loading path and ends with exactly zero stress.
"""
import numpy as np
import pytest

from dispsolver.io import execute_inp

_DECK = """*HEADING
pull then return
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
*PLASTIC
10.0, 0.0
20.0, 0.1
*SOLID SECTION, ELSET=EL, MATERIAL=M
*STEP, NAME=PULL, NLGEOM={nlgeom}
*STATIC
0.1, 1.0, 1e-4, 0.1
*BOUNDARY
1, 1, 3, 0.0
4, 1, 1, 0.0
4, 3, 3, 0.0
5, 1, 2, 0.0
8, 1, 1, 0.0
2, 1, 1, 0.05
3, 1, 1, 0.05
6, 1, 1, 0.05
7, 1, 1, 0.05
*END STEP
*STEP, NAME=RETURN, NLGEOM={nlgeom}
*STATIC
0.1, 1.0, 1e-4, 0.1
*BOUNDARY
2, 1, 1, 0.0
3, 1, 1, 0.0
6, 1, 1, 0.0
7, 1, 1, 0.0
*END STEP
"""


@pytest.mark.parametrize("nlgeom", ["NO", "YES"])
def test_unloaded_to_zero_displacement_leaves_residual_compression(tmp_path, nlgeom):
    path = tmp_path / "deck.inp"
    path.write_text(_DECK.format(nlgeom=nlgeom))
    res = execute_inp(str(path), dim=3, verbose=False)
    assert res["converged"]
    solver = res["solver"]
    f = np.asarray(solver.f_int).reshape(-1, 3)
    reaction_x = f[[1, 2, 5, 6], 0].sum()               # right face, area 1
    assert np.abs(solver.u.reshape(-1, 3)[[1, 2, 5, 6], 0]).max() < 1e-9   # end displacement is back at zero
    assert reaction_x < -5.0                            # residual compression (peak stress was ~ +14)
