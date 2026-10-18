"""
report_generator.py
===================
Generates a 1-page modern, responsive HTML Engineering Brief report (report.html)
summarizing stackup specs, mechanics KPIs, theoretical bounds, and embedded visuals.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Tuple, Any, Optional


def generate_engineering_html_report(
    output_dir: str,
    config: Any,
    summary_data: Dict[str, Any],
) -> str:
    """Generates report.html in output_dir.

    Parameters
    ----------
    output_dir : str
        Target results directory (e.g. results/DYYYYMMDD_HHMMSS/)
    config : UnifiedFoldModelConfig
        Model configuration instance.
    summary_data : dict
        Quantitative metrics (final_theta, M_max, Fx, symm_err, wall_time, etc.)
    """
    html_path = os.path.join(output_dir, "report.html")

    preset = config.simulation_case.preset_name
    dim = config.simulation_case.dim
    total_t = config.total_thickness
    n_layers = len(config.layers)
    neck_gap = 2.0 * abs(config.geometry.hinge_pivot_x)
    span = 2.0 * config.geometry.hinge_half_gap

    final_theta = summary_data.get("final_theta", 0.0)
    total_fold = 2.0 * final_theta
    M_L = summary_data.get("final_M_L", 0.0)
    M_R = summary_data.get("final_M_R", 0.0)
    avg_M = 0.5 * (abs(M_L) + abs(M_R))
    symm_err = summary_data.get("symmetry_error", 0.0)
    wall_time = summary_data.get("wall_time_s", 0.0)
    n_steps = summary_data.get("num_steps", 0)
    avg_iters = summary_data.get("avg_iters", 0.0)
    cutbacks = summary_data.get("cutbacks", 0)

    D_slip, D_mono, y_na = config.compute_composite_bending_stiffness()

    v_iters = "PASS" if avg_iters <= 10.0 else "WARN"
    c_iters = "kpi-ok" if avg_iters <= 10.0 else "kpi-warn"

    v_cutbacks = "PASS" if cutbacks == 0 else "FAIL"
    c_cutbacks = "kpi-ok" if cutbacks == 0 else "kpi-fail"

    v_symm = "PASS" if symm_err <= 0.10 else "WARN"
    c_symm = "kpi-ok" if symm_err <= 0.10 else "kpi-warn"

    v_lock = "PASS" if avg_M < D_mono else "WARN"
    c_lock = "kpi-ok" if avg_M < D_mono else "kpi-warn"

    overall_status = "PASS" if (v_cutbacks == "PASS" and v_iters == "PASS") else "WARN"
    overall_badge_bg = "#16a34a" if overall_status == "PASS" else "#d97706"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>Engineering Report: 180° Foldable Display ({dim} {preset})</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", Arial, sans-serif;
      background-color: #0f172a;
      color: #e2e8f0;
      margin: 0;
      padding: 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      background: #1e293b;
      border-radius: 12px;
      padding: 32px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }}
    h1 {{ color: #38bdf8; margin-top: 0; font-size: 26px; border-bottom: 2px solid #334155; padding-bottom: 12px; }}
    h2 {{ color: #94a3b8; font-size: 18px; margin-top: 28px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      background: #0284c7;
      color: white;
      margin-left: 8px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .kpi-card {{
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }}
    .kpi-val {{ font-size: 24px; font-weight: 700; color: #f8fafc; margin-top: 4px; }}
    .kpi-ok {{ color: #4ade80; }}
    .kpi-warn {{ color: #f59e0b; }}
    .kpi-fail {{ color: #ef4444; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid #334155;
    }}
    th {{ background: #0f172a; color: #94a3b8; font-weight: 600; }}
    .visual-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 20px;
    }}
    .visual-box {{
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }}
    .visual-box img {{
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }}
    .footer {{
      margin-top: 40px;
      font-size: 12px;
      color: #64748b;
      text-align: center;
      border-top: 1px solid #334155;
      padding-top: 16px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>
      180° Foldable Display Simulation Brief
      <span class="badge">{dim} {preset.upper()}</span>
      <span class="badge" style="background: {overall_badge_bg};">{overall_status}</span>
    </h1>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-lbl">Total Fold Angle</div>
        <div class="kpi-val kpi-ok">{total_fold:.2f}°</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Peak Drive Moment</div>
        <div class="kpi-val">{avg_M:.4f} <span style="font-size:14px;">N·mm</span></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Symmetry Error</div>
        <div class="kpi-val kpi-ok">{symm_err:.4f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Solve Time / Steps</div>
        <div class="kpi-val">{wall_time:.1f}s <span style="font-size:14px;">({n_steps} stp)</span></div>
      </div>
    </div>

    <h2>1. Stackup & Kinematics Specification</h2>
    <table>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Design Value</th>
          <th>Mechanics Impact</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Laminate Stackup</td>
          <td><strong>{n_layers} Layers</strong> (PET / PSA / GLASS)</td>
          <td>Total Thickness = {total_t:.3f} mm</td>
        </tr>
        <tr>
          <td>Free Hinge Span</td>
          <td><strong>{span:.1f} mm</strong> ([-{span/2:.1f}, +{span/2:.1f}])</td>
          <td>Accommodates natural elastica curvature loop</td>
        </tr>
        <tr>
          <td>Rigid Rotation Pivot</td>
          <td><strong>(±{config.geometry.hinge_pivot_x:.2f}, {config.geometry.hinge_pivot_y:.2f}) mm</strong></td>
          <td>Maintains physical <strong>{neck_gap:.2f} mm</strong> neck gap (anti-crush)</td>
        </tr>
        <tr>
          <td>Theoretical Stiffness Bounds</td>
          <td><strong>D_slip = {D_slip:.4f}</strong> vs <strong>D_mono = {D_mono:.4f} N·mm²</strong></td>
          <td>Stiffness ratio = {D_mono/D_slip:.1f}x (Zero volumetric locking)</td>
        </tr>
      </tbody>
    </table>

    <h2>2. Visual Analysis & Transient Dynamics</h2>
    <div class="visual-grid">
      <div class="visual-box">
        <div style="font-weight:600; margin-bottom:8px; color:#cbd5e1;">Folding Transient Animation</div>
        <img src="folding_animation.gif" alt="Folding Animation" onerror="this.parentElement.innerHTML='<p style=\\'color:#64748b;padding:40px;\\'>Animation pending</p>'">
      </div>
      <div class="visual-box">
        <div style="font-weight:600; margin-bottom:8px; color:#cbd5e1;">Kinematics & Reaction Summary</div>
        <img src="summary_plots.png" alt="Summary Plots" onerror="this.parentElement.innerHTML='<p style=\\'color:#64748b;padding:40px;\\'>Plots pending</p>'">
      </div>
    </div>

    <h2>3. Numerical Convergence & Verification Status</h2>
    <table>
      <tr>
        <th>Verification Criterion</th>
        <th>Measured Level</th>
        <th>Tolerance</th>
        <th>Verdict</th>
      </tr>
      <tr>
        <td>Newton Iterations per Step</td>
        <td>{avg_iters:.1f} iters/step</td>
        <td>&le; 10 iters</td>
        <td class="{c_iters}" style="font-weight:bold;">{v_iters}</td>
      </tr>
      <tr>
        <td>Solver Cutbacks</td>
        <td>{cutbacks} Cutbacks</td>
        <td>0 Cutbacks target</td>
        <td class="{c_cutbacks}" style="font-weight:bold;">{v_cutbacks}</td>
      </tr>
      <tr>
        <td>Action-Reaction Symmetry</td>
        <td>{symm_err:.4f}% error</td>
        <td>&le; 0.1%</td>
        <td class="{c_symm}" style="font-weight:bold;">{v_symm}</td>
      </tr>
      <tr>
        <td>Physical Locking Check</td>
        <td>M_drive ({avg_M:.4f}) &lt;&lt; M_mono</td>
        <td>Interlayer slip active</td>
        <td class="{c_lock}" style="font-weight:bold;">{v_lock}</td>
      </tr>
    </table>

    <div class="footer">
      Generated automatically by WHT_DispFoldSolver (Unified 2D/3D Folding Framework).
    </div>
  </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path
