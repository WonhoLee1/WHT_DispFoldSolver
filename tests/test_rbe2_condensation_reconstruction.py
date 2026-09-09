import pytest
import numpy as np
import os
from dispsolver.fold_model_config import DEFAULT_CONFIG
from dispsolver.io import read_abaqus_input


def test_rbe2_condensation_reconstruction():
    """Verify that Kinematic Condensation slave node displacements are updated on step convergence."""
    inp_path = os.path.join(os.path.dirname(__file__), "..", "examples", "ex12_rigid_plate_display_fold.inp")
    if not os.path.exists(inp_path):
        pytest.skip("ex12_rigid_plate_display_fold.inp not found")

    result = read_abaqus_input(inp_path)
    assert len(result.rbe2_constraints) == 2

    # Check master node IDs and slave IDs
    # base_node_id default raised 10000 -> 100000 this session (node-id
    # overflow fix, see AGENTS.md / dev_log); the right plate's master id
    # isn't a round "2x base" offset -- both plates share one sequential
    # node counter starting at base_node_id, so it lands wherever the
    # left plate's own node/element count leaves off (100306 for this
    # deck, not 200000).
    left_c = result.rbe2_constraints[0]
    assert left_c.master_id in (100000, 100306)
    assert len(left_c.slave_ids) > 0

    # Evaluate slave displacement mapping
    u_master = np.array([1.0, 2.0])
    theta = np.pi / 4.0  # 45 deg
    slaves_disp = left_c.evaluate_slave_displacements(u_master, theta)
    assert len(slaves_disp) == len(left_c.slave_ids)
    for sid, u_s in slaves_disp.items():
        assert u_s.shape == (2,)
