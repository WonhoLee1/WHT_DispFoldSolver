"""ex17_cae_full_features_demo.py
=================================
End-to-End Demonstration of Commercial CAE Architecture:
- Smart GeneralSet & findAt geometric selection
- Model.RigidBody & Model.Coupling constraints
- Model.ConcentratedForce & Model.Pressure loads with Amplitude
- MultiStepExecutor: Step 1 (Loading) -> Branch A (Springback) vs Branch B (Overload)
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib

from dispsolver.model.model import Model
from dispsolver.model.step import Step, EntityStatus
from dispsolver.solver.step_executor import MultiStepExecutor


def build_and_run_demo():
    print("=== [1/4] Constructing Model & Mesh via Commercial CAE API ===", flush=True)
    model = Model(name="CommercialCAE_Model", dim=3)
    p = model.Part("CantileverBeam", dim=3)

    # Mesh a 4x1x1 hexahedral beam: length 4.0, height 1.0, depth 1.0
    # 4 elements in X, 1 in Y, 1 in Z
    nx, ny, nz = 4, 1, 1
    L, H, W = 4.0, 1.0, 1.0
    dx, dy, dz = L / nx, H / ny, W / nz

    nid = 1
    node_grid = np.zeros((nx + 1, ny + 1, nz + 1), dtype=int)
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                p.add_node(nid, [i * dx, j * dy, k * dz])
                node_grid[i, j, k] = nid
                nid += 1

    eid = 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n1 = node_grid[i, j, k]
                n2 = node_grid[i + 1, j, k]
                n3 = node_grid[i + 1, j + 1, k]
                n4 = node_grid[i, j + 1, k]
                n5 = node_grid[i, j, k + 1]
                n6 = node_grid[i + 1, j, k + 1]
                n7 = node_grid[i + 1, j + 1, k + 1]
                n8 = node_grid[i, j + 1, k + 1]
                p.add_element(eid, "C3D8_COROTATIONAL", [n1, n2, n3, n4, n5, n6, n7, n8])
                eid += 1

    print(f"  --> Mesh generated: {len(p.nodes)} nodes, {len(p.elements)} C3D8 elements", flush=True)

    print("\n=== [2/4] Defining Smart Sets, Sections, and Constraints ===", flush=True)
    # Smart Sets via Bounding Box
    fixed_set = p.create_set_from_box("FIXED_FACE", (-0.05, 0.05), (-0.05, 1.05), (-0.05, 1.05), entity_type="NODES")
    tip_set = p.create_set_from_box("TIP_FACE", (3.95, 4.05), (-0.05, 1.05), (-0.05, 1.05), entity_type="NODES")
    all_elems = p.create_set_from_box("ALL_ELEMS", (-0.05, 4.05), (-0.05, 1.05), (-0.05, 1.05), entity_type="ELEMENTS")

    # Reference Point (RP) near tip
    p.add_node(999, [4.5, 0.5, 0.5])
    rp_set = p.create_set_from_box("RP_SET", (4.4, 4.6), (0.4, 0.6), (0.4, 0.6), entity_type="NODES")

    # Material & Section
    mat = model.Material(name="Aluminum", mat_type="ELASTIC")
    mat.E = 70000.0  # 70 GPa
    mat.nu = 0.33
    sec = model.SolidSection(name="BeamSection", material="Aluminum")
    p.SectionAssignment(region=all_elems, sectionName="BeamSection")

    # Instance into root assembly
    model.root_assembly.Instance(name="BEAM_1", part=p)

    # Declare Constraints
    rb = model.RigidBody(name="RB_TipHandle", refPoint="BEAM_1.RP_SET", tieNset="BEAM_1.TIP_FACE")
    print(f"  --> Constraint declared: RigidBody '{rb.name}' coupling RP to {len(rb.get_slave_node_ids())} nodes", flush=True)

    print("\n=== [3/4] Configuring Multi-Step Branching Analysis Tree ===", flush=True)
    # Smooth step amplitude for realistic gradual loading
    amp = model.SmoothStepAmplitude(name="SmoothRamp", data=[(0.0, 0.0), (0.05, 1.0)])

    # Base BC
    model.DisplacementBC(name="FixedSupport", createStepName="Initial", region="BEAM_1.FIXED_FACE", u1=0.0, u2=0.0, u3=0.0)

    # Step 1: Loading
    step1 = model.Step(name="Step-1_Loading", time_period=0.05, dt_init=0.05, dt_min=1e-5, dt_max=0.05)
    model.ConcentratedForce(name="TipShear", createStepName="Step-1_Loading", region="BEAM_1.TIP_FACE", cf2=-5000.0, amplitude="SmoothRamp")

    # Branch A: Springback / Unload
    step_2a = model.Step(name="Step-2A_Springback", parent_step=step1, time_period=0.05, dt_init=0.05, dt_min=1e-5, dt_max=0.05)
    step_2a.deactivate_load("TipShear")

    # Branch B: Impact / Overload (3x force)
    step_2b = model.Step(name="Step-2B_Overload", parent_step=step1, time_period=0.05, dt_init=0.05, dt_min=1e-5, dt_max=0.05)
    step_2b.modify_load("TipShear", cf2=-15000.0)

    print("  --> Step Tree: Step-1_Loading -> [Branch A: Step-2A_Springback] vs [Branch B: Step-2B_Overload]", flush=True)

    print("\n=== [4/4] Executing Branching Engine via MultiStepExecutor ===", flush=True)
    executor = MultiStepExecutor(model=model, verbose=True)

    # 1. Solve Step 1
    ok1 = executor.execute_step(step1)
    assert ok1 is True, "Step 1 failed!"
    u_step1 = executor.results_by_step["Step-1_Loading"]["u"]
    tip_indices = executor.sys.global_nsets["BEAM_1.TIP_FACE"]
    uy_step1 = np.mean([u_step1[3 * idx + 1] for idx in tip_indices])
    print(f"  [+] Step 1 (Loading) Deflection: {uy_step1:.4f} mm", flush=True)

    # 2. Solve Branch A
    ok2a = executor.execute_step(step_2a, from_parent_checkpoint=True)
    assert ok2a is True, "Branch A failed!"
    u_step2a = executor.results_by_step["Step-2A_Springback"]["u"]
    uy_step2a = np.mean([u_step2a[3 * idx + 1] for idx in tip_indices])
    print(f"  [+] Branch A (Springback) Deflection: {uy_step2a:.4f} mm (Rebounded toward 0)", flush=True)

    # 3. Solve Branch B
    ok2b = executor.execute_step(step_2b, from_parent_checkpoint=True)
    assert ok2b is True, "Branch B failed!"
    u_step2b = executor.results_by_step["Step-2B_Overload"]["u"]
    uy_step2b = np.mean([u_step2b[3 * idx + 1] for idx in tip_indices])
    print(f"  [+] Branch B (Overload) Deflection: {uy_step2b:.4f} mm (Deeper deflection)", flush=True)

    # Plot Comparison Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    x_coords = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    # Extract bottom edge deflections along beam length
    def extract_profile(u_vec):
        disp_profile = []
        for i in range(nx + 1):
            nid_bot = node_grid[i, 0, 0]
            g_idx = executor.sys.nid_to_idx[nid_bot]
            disp_profile.append(u_vec[3 * g_idx + 1])
        return disp_profile

    ax.plot(x_coords, extract_profile(u_step1), "b-o", linewidth=2.0, label=f"Step 1: Forming Load (Uy={uy_step1:.3f} mm)")
    ax.plot(x_coords, extract_profile(u_step2a), "g--s", linewidth=2.0, label=f"Branch A: Springback (Uy={uy_step2a:.3f} mm)")
    ax.plot(x_coords, extract_profile(u_step2b), "r-^", linewidth=2.0, label=f"Branch B: Impact Overload (Uy={uy_step2b:.3f} mm)")

    ax.set_title("상용 CAE 멀티 스텝 가지치기(Branching) 해석 결과 비교", fontsize=12)
    ax.set_xlabel("빔 길이 방향 위치 X (mm)", fontsize=10)
    ax.set_ylabel("처짐 변위 Uy (mm)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    out_img = "examples/ex17_multi_step_branching_comparison.png"
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f"\n[+] Visualization saved successfully: {out_img}", flush=True)


if __name__ == "__main__":
    build_and_run_demo()
