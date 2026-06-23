import numpy as np
import pytest
from dispsolver.mesh import Mesh
from dispsolver.constraint.rbe2 import RBE2HingeConstraint


def test_rbe2_kinematics():
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 1.0, 0.0)
    mesh.add_node(2, 0.0, 1.0)

    constraint = RBE2HingeConstraint(mesh, master_id=0, slave_ids=[1, 2], extra_primal_offset=0)

    assert constraint.n_extra_primal() == 1
    assert constraint.n_multipliers() == 4

    u = np.zeros(6)
    u[0] = 0.1
    u[1] = 0.2

    theta = 0.1
    cost = np.cos(theta)
    sint = np.sin(theta)

    dx1, dy1 = 1.0, 0.0
    u[2] = 0.1 + (cost - 1.0) * dx1 - sint * dy1
    u[3] = 0.2 + sint * dx1 + (cost - 1.0) * dy1

    dx2, dy2 = 0.0, 1.0
    u[4] = 0.1 + (cost - 1.0) * dx2 - sint * dy2
    u[5] = 0.2 + sint * dx2 + (cost - 1.0) * dy2

    u_ext = np.array([theta])
    r_u, c_u, v_u, r_ext, c_ext, v_ext, g = constraint.assemble(u, u_ext)

    assert np.allclose(g, 0.0, atol=1e-12)
    idx = np.where(r_ext == 0)[0][0]
    assert np.isclose(v_ext[idx], sint)


def test_rigid_body_parser():
    """*RIGID BODY with ELSET= parameter."""
    import os, tempfile
    from dispsolver.io import read_abaqus_input

    tmp = tempfile.gettempdir()
    inp = os.path.join(tmp, '_test_rigid_body.inp')
    with open(inp, 'w') as f:
        f.write('*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n5,2,0\n')
        f.write('*ELEMENT,TYPE=CPE4,ELSET=BLOCK_A\n1,1,2,3,4\n')
        f.write('*SOLID SECTION,ELSET=BLOCK_A,MATERIAL=EL\n')
        f.write('*ELASTIC\n1000.0,0.3\n*DENSITY\n1e-9\n')
        f.write('*RIGID BODY,ELSET=BLOCK_A,REF NODE=5\n')
        f.write('*STEP\n*STATIC\n0.1,1.0\n*BOUNDARY\n5,1,1\n*BOUNDARY\n5,2,2\n*END STEP\n')

    result = read_abaqus_input(inp)
    assert len(result.constraints) == 1
    c = result.constraints[0]
    assert isinstance(c, RBE2HingeConstraint)
    assert c.master_id == 5
    for nid in [1, 2, 3, 4]:
        assert nid in c.slave_ids
    assert c.extra_primal_offset == 0


def test_rigid_body_nset():
    """*RIGID BODY with NSET= parameter."""
    import os, tempfile
    from dispsolver.io import read_abaqus_input

    tmp = tempfile.gettempdir()
    inp = os.path.join(tmp, '_test_rb_nset.inp')
    with open(inp, 'w') as f:
        f.write('*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n5,2,0\n')
        f.write('*NSET,NSET=MY_NODES,GENERATE\n1,4,1\n')
        f.write('*ELEMENT,TYPE=CPE4,ELSET=BLOCK_A\n1,1,2,3,4\n')
        f.write('*SOLID SECTION,ELSET=BLOCK_A,MATERIAL=EL\n')
        f.write('*ELASTIC\n1000.0,0.3\n*DENSITY\n1e-9\n')
        f.write('*RIGID BODY,NSET=MY_NODES,REF NODE=5\n')
        f.write('*STEP\n*STATIC\n0.1,1.0\n*BOUNDARY\n5,1,1\n*BOUNDARY\n5,2,2\n*END STEP\n')

    result = read_abaqus_input(inp)
    assert len(result.constraints) == 1
    c = result.constraints[0]
    assert isinstance(c, RBE2HingeConstraint)
    assert c.master_id == 5
    assert sorted(c.slave_ids) == [1, 2, 3, 4]


def test_rigid_body_nset_generate():
    """*RIGID BODY with NSET,GENERATE."""
    import os, tempfile
    from dispsolver.io import read_abaqus_input

    tmp = tempfile.gettempdir()
    inp = os.path.join(tmp, '_test_rb_nset_gen.inp')
    with open(inp, 'w') as f:
        f.write('*NODE\n1,0,0\n2,0,0\n3,0,0\n4,0,0\n5,2,0\n')
        f.write('*NSET,NSET=ALL_NODES,GENERATE\n1,4,1\n')
        f.write('*ELEMENT,TYPE=CPE4,ELSET=EALL\n1,1,2,3,4\n')
        f.write('*SOLID SECTION,ELSET=EALL,MATERIAL=EL\n')
        f.write('*ELASTIC\n1000.0,0.3\n*DENSITY\n1e-9\n')
        f.write('*RIGID BODY,NSET=ALL_NODES,REF NODE=5\n')
        f.write('*STEP\n*STATIC\n0.1,1.0\n*BOUNDARY\n5,1,1\n*BOUNDARY\n5,2,2\n*END STEP\n')

    result = read_abaqus_input(inp)
    assert len(result.constraints) == 1
    c = result.constraints[0]
    assert isinstance(c, RBE2HingeConstraint)
    assert c.master_id == 5
    assert sorted(c.slave_ids) == [1, 2, 3, 4]
