import numpy as np
from dispsolver.constraint3d.rbe3_distributing3d import RBE3DistributingConstraint3D

def test_rbe3_distributing_3d_assemble():
    master_id = 100
    slave_ids = [101, 102, 103]
    nid_to_idx = {100: 0, 101: 1, 102: 2, 103: 3}
    num_dofs = 4 * 3  # 4 nodes, 3 DOFs each
    
    constraint = RBE3DistributingConstraint3D(
        master_id=master_id,
        slave_ids=slave_ids,
        nid_to_idx=nid_to_idx,
        num_dofs=num_dofs,
        penalty_stiffness=1e5
    )
    
    # Check default weights
    assert np.allclose(constraint.weights, 1.0 / 3.0)
    
    u = np.zeros(num_dofs)
    # Master node 0 moves x=1.0, slaves move x=0.0
    u[0] = 1.0
    
    f_c, (rows, cols, data), err = constraint.assemble(u)
    
    assert not err
    # Force on master should be k * (1.0 - 0)
    assert np.isclose(f_c[0], 1e5)
    
    # Force on slaves should be -k * w_i * gap
    assert np.isclose(f_c[3], -1e5 / 3.0)
    assert np.isclose(f_c[6], -1e5 / 3.0)
    assert np.isclose(f_c[9], -1e5 / 3.0)
    
    # Check stiffness matrix dimensions
    assert len(rows) == len(cols) == len(data)
    # 1 master, 3 slaves = 4 nodes per DOF. 4x4 matrix per DOF = 16 non-zeros per DOF. 3 DOFs = 48 non-zeros.
    assert len(rows) == 48

if __name__ == "__main__":
    test_rbe3_distributing_3d_assemble()
    print("Test passed!")
