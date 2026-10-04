"""
cae_logger.py
=============
Commercial CAE grade status (.sta) and message (.msg) log generation engine.

Produces production-grade Abaqus-parity diagnostic files for real-time monitoring
of Newton-Raphson iterations, convergence residuals, line search factors, contact SDI
status, and step increment progress.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class CAELogger:
    """Production CAE Logger producing Abaqus-compatible .sta and .msg files."""

    def __init__(self, job_name: str, out_dir: str, dim: int = 2):
        self.job_name = str(job_name)
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.dim = int(dim)

        self.sta_path = os.path.join(self.out_dir, f"{self.job_name}.sta")
        self.msg_path = os.path.join(self.out_dir, f"{self.job_name}.msg")

        self.sta_file = open(self.sta_path, "w", encoding="utf-8")
        self.msg_file = open(self.msg_path, "w", encoding="utf-8")

        self.total_increments = 0
        self.total_iters = 0
        self.total_cutbacks = 0
        self.total_sdis = 0

        self._write_headers()

    def _write_headers(self):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # .sta Header
        self.sta_file.write(f"{'=' * 100}\n")
        self.sta_file.write(f" WHT_DispFoldSolver - STATUS FILE (.sta)\n")
        self.sta_file.write(f" Job Name  : {self.job_name}\n")
        self.sta_file.write(f" Date/Time : {now_str}\n")
        self.sta_file.write(f" Dimension : {self.dim}D Continuum Model\n")
        self.sta_file.write(f"{'=' * 100}\n\n")
        
        self.sta_file.write(
            f"{'STEP':>5}  {'INC':>5}  {'ATT':>4}  {'SEVERE':>7}  {'EQUIL':>6}  "
            f"{'TOTAL TIME':>12}  {'STEP TIME':>12}  {'INC OF TIME':>12}  {'STATUS':>10}\n"
        )
        self.sta_file.write(f"{'-' * 100}\n")
        self.sta_file.flush()

        # .msg Header
        self.msg_file.write(f"{'=' * 100}\n")
        self.msg_file.write(f" WHT_DispFoldSolver - MESSAGE FILE (.msg)\n")
        self.msg_file.write(f" Job Name  : {self.job_name}\n")
        self.msg_file.write(f" Date/Time : {now_str}\n")
        self.msg_file.write(f"{'=' * 100}\n\n")
        self.msg_file.flush()

    def log_iteration(
        self,
        step_idx: int,
        inc_idx: int,
        iter_num: int,
        r_norm: float,
        du_norm: float,
        conv_rate: float = 1.0,
        s_line_search: float = 1.0,
        max_r_node: int = 0,
        max_r_dof: int = 0,
        max_du_node: int = 0,
        max_du_dof: int = 0,
        sdi_active_count: int = 0,
        is_sdi: bool = False,
    ):
        """Write detailed Newton-Raphson equilibrium iteration diagnostics to .msg file."""
        self.total_iters += 1
        if is_sdi:
            self.total_sdis += 1

        self.msg_file.write(f"\n  STEP {step_idx:d}  INCREMENT {inc_idx:d}  ITERATION {iter_num:d}\n")
        self.msg_file.write(f"  {'-' * 96}\n")
        self.msg_file.write(
            f"  MAXIMUM RESIDUAL FORCE          = {r_norm:12.4E} AT NODE {max_r_node:6d} (DOF {max_r_dof:1d})\n"
        )
        self.msg_file.write(
            f"  MAXIMUM DISPLACEMENT CORRECTION = {du_norm:12.4E} AT NODE {max_du_node:6d} (DOF {max_du_dof:1d})\n"
        )
        self.msg_file.write(f"  CONVERGENCE RATE                = {conv_rate:12.4f}\n")
        self.msg_file.write(f"  LINE SEARCH STEP LENGTH         = {s_line_search:12.4f}\n")

        if sdi_active_count > 0 or is_sdi:
            self.msg_file.write(f"  CONTACT SDI STATUS              = {sdi_active_count:d} PAIRS ACTIVE (SDI={is_sdi})\n")

        self.msg_file.flush()

    def log_increment(
        self,
        step_idx: int,
        inc_idx: int,
        att_num: int,
        n_iters: int,
        t_total: float,
        t_step: float,
        dt: float,
        status_str: str = "CONVERGED",
        n_sdis: int = 0,
    ):
        """Write increment summary row to .sta status file."""
        self.total_increments += 1
        self.sta_file.write(
            f"{step_idx:5d}  {inc_idx:5d}  {att_num:4d}  {n_sdis:7d}  {n_iters:6d}  "
            f"{t_total:12.4f}  {t_step:12.4f}  {dt:12.5E}  {status_str:>10}\n"
        )
        self.sta_file.flush()

    def log_cutback(self, step_idx: int, inc_idx: int, old_dt: float, new_dt: float, reason: str):
        """Log time increment cutback event to both .sta and .msg files."""
        self.total_cutbacks += 1
        msg = f"  *** CUTBACK: dt reduced from {old_dt:.4E} to {new_dt:.4E} ({reason})\n"
        self.msg_file.write(msg)
        self.msg_file.flush()
        self.sta_file.write(f"  *** CUTBACK: {reason} (dt={new_dt:.4E})\n")
        self.sta_file.flush()

    def close(self, status_str: str = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"):
        """Write summary statistics and close .sta and .msg log files."""
        self.sta_file.write(f"{'=' * 100}\n")
        self.sta_file.write(f" ANALYSIS SUMMARY:\n")
        self.sta_file.write(f"   TOTAL INCREMENTS             : {self.total_increments:d}\n")
        self.sta_file.write(f"   TOTAL NEWTON ITERATIONS      : {self.total_iters:d}\n")
        self.sta_file.write(f"   TOTAL CUTBACKS               : {self.total_cutbacks:d}\n")
        self.sta_file.write(f"   TOTAL SEVERE DISCONTINUITIES : {self.total_sdis:d}\n")
        self.sta_file.write(f" STATUS: {status_str}\n")
        self.sta_file.write(f"{'=' * 100}\n")
        self.sta_file.flush()
        self.sta_file.close()

        self.msg_file.write(f"\n{'=' * 100}\n")
        self.msg_file.write(f" STATUS: {status_str}\n")
        self.msg_file.write(f"{'=' * 100}\n")
        self.msg_file.flush()
        self.msg_file.close()
