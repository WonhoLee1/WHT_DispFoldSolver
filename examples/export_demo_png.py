"""
export_demo_png.py
==================
Quick script to run display folding simulation and export high-resolution PNG image to artifact directory.
"""

import os
import sys
import numpy as np

from dispsolver.mesh.mesh import Mesh
from dispsolver.mesh.plate_builder import create_folding_plate_parts
from dispsolver.part.rigid_body import RigidBodyPart
from dispsolver.constraint.surface_tie import SurfaceTieConstraint
from dispsolver.element.rbe2 import RBE2HingeElement
from dispsolver.material.neohookean import NeoHookean
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.export.plotter import plot_and_save_deformed_shape_png


def main():
    nx_disp = 80
    ny_disp = 4
    x_coords = np.linspace(-40.0, 40.0, nx_disp + 1)
    y_coords = np.linspace(0.0, 0.5, ny_disp + 1)

    mesh = Mesh()
    nid = 1
    grid_nids = np.zeros((ny_disp + 1, nx_disp + 1), dtype=int)
    bottom_surface_nids = []

    for j, y_val in enumerate(y_coords):
        for i, x_val in enumerate(x_coords):
            grid_nids[j, i] = nid
            mesh.add_node(nid, x_val, y_val)
            if j == 0:
                bottom_surface_nids.append(nid)
            nid += 1

    eid = 1
    for j in range(ny_disp):
        for i in range(nx_disp):
            n1 = grid_nids[j, i]
            n2 = grid_nids[j, i + 1]
            n3 = grid_nids[j + 1, i + 1]
            n4 = grid_nids[j + 1, i]
            mesh.add_element(eid, [n1, n2, n3, n4], "Q4", pid=1)
            eid += 1

    mesh.add_nodeset('BOTTOM_SURFACE', bottom_surface_nids)

    plates = create_folding_plate_parts(
        left_x_range=(-40.0, -10.0),
        right_x_range=(10.0, 40.0),
        y_range=(-0.5, 0.0),
        left_pivot=(-3.0, 0.0),
        right_pivot=(3.0, 0.0),
        nx=30,
        ny=2,
        base_node_id=10000,
        base_elem_id=10000,
    )

    left_plate = plates["left"]
    right_plate = plates["right"]

    for p_name, p_data in plates.items():
        for nid_p, coord in p_data["nodes_dict"].items():
            mesh.add_node(nid_p, coord[0], coord[1])
        for eid_p, conn in p_data["elements_dict"].items():
            mesh.add_element(eid_p, conn, "Q4", pid=2)

    mesh.add_nodeset('LEFT_PLATE_TOP', left_plate["top_surface_nids"])
    mesh.add_nodeset('RIGHT_PLATE_TOP', right_plate["top_surface_nids"])

    nid_to_idx = mesh.node_id_to_index()
    coords = mesh.nodes_array()

    left_master_idx = nid_to_idx[left_plate["master_rp_id"]]
    left_slave_indices = [nid_to_idx[sid] for sid in left_plate["slave_nids"]]
    right_master_idx = nid_to_idx[right_plate["master_rp_id"]]
    right_slave_indices = [nid_to_idx[sid] for sid in right_plate["slave_nids"]]

    rbe2_left = RBE2HingeElement(left_master_idx, left_slave_indices, coords)
    rbe2_right = RBE2HingeElement(right_master_idx, right_slave_indices, coords)
    rbe2_left.master_id = left_plate["master_rp_id"]
    rbe2_left.slave_ids = left_plate["slave_nids"]
    rbe2_right.master_id = right_plate["master_rp_id"]
    rbe2_right.slave_ids = right_plate["slave_nids"]

    rbe2_left.set_prescribed_theta(0.0)
    rbe2_right.set_prescribed_theta(0.0)

    left_display_bottom = [nid for nid in bottom_surface_nids if mesh.nodes[nid].x <= -10.0]
    right_display_bottom = [nid for nid in bottom_surface_nids if mesh.nodes[nid].x >= 10.0]

    tie_left = SurfaceTieConstraint(
        slave_node_ids=left_display_bottom,
        master_node_ids=left_plate["top_surface_nids"],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1e5,
        name="TIE_LEFT",
    )

    tie_right = SurfaceTieConstraint(
        slave_node_ids=right_display_bottom,
        master_node_ids=right_plate["top_surface_nids"],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1e5,
        name="TIE_RIGHT",
    )

    materials = {
        1: NeoHookean(),
        2: NeoHookean(),
    }
    mat_params = {
        1: {"E": 50.0, "nu": 0.45},
        2: {"E": 20000.0, "nu": 0.3},
    }

    solver = DynamicSolver(
        mesh=mesh,
        material=materials,
        material_params=mat_params,
        rho=1e-9,
        mode="quasistatic",
        rtol=1e-4,
        atol=1e-6,
        max_iter=25,
        rbe2_elements=[rbe2_left, rbe2_right],
        penalty_constraints=[tie_left, tie_right],
        ul_mode=True,
        verbose=False,
    )

    print("Running folding simulation steps with smooth theta ramp for PNG export...")
    for step in range(1, 11):
        theta = step * np.radians(2.0)
        rbe2_left.set_prescribed_theta(-theta)
        rbe2_right.set_prescribed_theta(theta)
        n_iter = solver.solve_step(0.02)
        print(f"Step {step:2d} (θ={np.degrees(theta):4.1f}°): converged in {n_iter+1} iterations.")

    artifact_dir = r"C:\Users\GOODMAN\.gemini\antigravity-cli\brain\17751c1b-5d61-4d49-980f-528b2d8cf463"
    png_path = plot_and_save_deformed_shape_png(
        mesh=mesh,
        u=solver.u,
        save_path="output/ex11_final_folding_shape.png",
        title_prefix="EX11: Rigid Plate Display Folding Final State",
        rbe2_elements=[rbe2_left, rbe2_right],
        tie_constraints=[tie_left, tie_right],
        artifact_dir=artifact_dir,
    )
    print(f"PNG generated successfully: {png_path}")


if __name__ == "__main__":
    main()
