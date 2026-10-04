"""
test_cae_logger.py
==================
Unit tests for CAELogger Abaqus-parity log generator (.sta and .msg).
"""

import os
import tempfile
import pytest
from dispsolver.io.cae_logger import CAELogger


def test_cae_logger_file_creation_and_content():
    with tempfile.TemporaryDirectory() as tmp_dir:
        job_name = "test_job"
        logger = CAELogger(job_name=job_name, out_dir=tmp_dir, dim=2)

        logger.log_iteration(
            step_idx=1,
            inc_idx=1,
            iter_num=1,
            r_norm=1.234e-3,
            du_norm=5.678e-4,
            conv_rate=0.85,
            s_line_search=1.0,
            max_r_node=12,
            max_r_dof=1,
            max_du_node=12,
            max_du_dof=1,
            sdi_active_count=0,
            is_sdi=False,
        )

        logger.log_increment(
            step_idx=1,
            inc_idx=1,
            att_num=1,
            n_iters=1,
            t_total=0.1,
            t_step=0.1,
            dt=0.1,
            status_str="CONVERGED",
        )

        logger.log_cutback(step_idx=1, inc_idx=2, old_dt=0.1, new_dt=0.05, reason="Divergence")

        logger.close(status_str="THE ANALYSIS HAS COMPLETED SUCCESSFULLY")

        sta_path = os.path.join(tmp_dir, f"{job_name}.sta")
        msg_path = os.path.join(tmp_dir, f"{job_name}.msg")

        assert os.path.exists(sta_path)
        assert os.path.exists(msg_path)

        with open(sta_path, "r", encoding="utf-8") as f:
            sta_content = f.read()
            assert "STATUS FILE (.sta)" in sta_content
            assert "CONVERGED" in sta_content
            assert "CUTBACK" in sta_content
            assert "ANALYSIS SUMMARY" in sta_content
            assert "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in sta_content

        with open(msg_path, "r", encoding="utf-8") as f:
            msg_content = f.read()
            assert "MESSAGE FILE (.msg)" in msg_content
            assert "MAXIMUM RESIDUAL FORCE" in msg_content
            assert "LINE SEARCH STEP LENGTH" in msg_content
            assert "CUTBACK" in msg_content
            assert "STATUS: THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in msg_content
