"""
ex14_3d_display_fold.py
=======================
Full 3D Foldable Display Panel Simulation driven by 3D Rigid Support Plate Rotation.

Simulates 3D multi-layer display panel transitioning from fully flat to fully folded
(90° per plate, 180° combined face-to-face fold) using 3D Hexahedral solid elements,
3D surface-to-surface penalty ties, and implicit 3D Newton-Raphson solver.
"""

from __future__ import annotations
import sys
import time
import numpy as np

from dispsolver.mesh3d import Mesh3D, Node3D, Element3D
from dispsolver.element3d import Hexa8EASElement
from dispsolver.material3d import J2Plasticity3D
from dispsolver.constraint3d import SurfaceTieConstraint3D
from dispsolver.solver3d import DynamicSolver3D


def build_3d_block_mesh(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    nx: int,
    ny: int,
    nz: int,
    node_id_start: int = 0,
    elem_id_start: int = 0,
    part_name: str = "PART"
) -> tuple[list[Node3D], list[Element3D], dict[str, list[int]], list[tuple[int, int, int, int]]]:
    """Helper to generate a 3D hexahedral block mesh.

    Returns
    -------
    nodes : list of Node3D
    elements : list of Element3D
    node_groups : dict
    top_faces : list of 4-node tuples for Quad4 surface faces at y = y_min or y = y_max
    """
    xs = np.linspace(x_range[0], x_range[1], nx + 1)
    ys = np.linspace(y_range[0], y_range[1], ny + 1)
    zs = np.linspace(z_range[0], z_range[1], nz + 1)

    nodes = []
    node_grid = np.zeros((nx + 1, ny + 1, nz + 1), dtype=int)

    nid = node_id_start
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_grid[i, j, k] = nid
                nodes.append(Node3D(nid, xs[i], ys[j], zs[k]))
                nid += 1

    elements = []
    eid = elem_id_start
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = node_grid[i,     j,     k]
                n1 = node_grid[i + 1, j,     k]
                n2 = node_grid[i + 1, j + 1, k]
                n3 = node_grid[i,     j + 1, k]
                n4 = node_grid[i,     j,     k + 1]
                n5 = node_grid[i + 1, j,     k + 1]
                n6 = node_grid[i + 1, j + 1, k + 1]
                n7 = node_grid[i,     j + 1, k + 1]

                conn = [n0, n1, n2, n3, n4, n5, n6, n7]
                elements.append(Element3D(eid, conn, "C3D8I", pid=0))
                eid += 1

    # Extract faces at y = y_range[0] (bottom face) and y = y_range[1] (top face)
    bottom_faces = []
    top_faces = []

    for k in range(nz):
        for i in range(nx):
            # Bottom face (y = y_range[0], j = 0): nodes (n0, n1, n5, n4)
            b0 = node_grid[i,     0, k]
            b1 = node_grid[i + 1, 0, k]
            b2 = node_grid[i + 1, 0, k + 1]
            b3 = node_grid[i,     0, k + 1]
            bottom_faces.append((b0, b1, b2, b3))

            # Top face (y = y_range[1], j = ny): nodes (n3, n2, n6, n7)
            t0 = node_grid[i,     ny, k]
            t1 = node_grid[i + 1, ny, k]
            t2 = node_grid[i + 1, ny, k + 1]
            t3 = node_grid[i,     ny, k + 1]
            top_faces.append((t0, t1, t2, t3))

    return nodes, elements, bottom_faces, top_faces


def run_3d_display_fold():
    print("=" * 70)
    print(" 3D Foldable Display Panel Simulation (180° Face-to-Face Closure)")
    print("=" * 70)

    # 1. Build Mesh Components
    # Display Panel: x in [-40, 40], y in [0.0, 0.5], z in [0.0, 4.0]
    nx_disp, ny_disp, nz_disp = 40, 2, 2
    d_nodes, d_elems, d_bot_faces, d_top_faces = build_3d_block_mesh(
        (-40.0, 40.0), (0.0, 0.5), (0.0, 4.0), nx_disp, ny_disp, nz_disp,
        node_id_start=0, elem_id_start=0, part_name="DISPLAY"
    )

    # Left Rigid Plate: x in [-40, -10], y in [-1.0, 0.0], z in [0.0, 4.0]
    nx_lp, ny_lp, nz_lp = 15, 1, 2
    lp_nodes, lp_elems, lp_bot_faces, lp_top_faces = build_3d_block_mesh(
        (-40.0, -10.0), (-1.0, 0.0), (0.0, 4.0), nx_lp, ny_lp, nz_lp,
        node_id_start=len(d_nodes), elem_id_start=len(d_elems), part_name="LEFT_PLATE"
    )

    # Right Rigid Plate: x in [10, 40], y in [-1.0, 0.0], z in [0.0, 4.0]
    nx_rp, ny_rp, nz_rp = 15, 1, 2
    rp_nodes, rp_elems, rp_bot_faces, rp_top_faces = build_3d_block_mesh(
        (10.0, 40.0), (-1.0, 0.0), (0.0, 4.0), nx_rp, ny_rp, nz_rp,
        node_id_start=len(d_nodes) + len(lp_nodes), elem_id_start=len(d_elems) + len(lp_elems), part_name="RIGHT_PLATE"
    )

    all_nodes = d_nodes + lp_nodes + rp_nodes
    all_elems = d_elems + lp_elems + rp_elems

    mesh = Mesh3D()
    for n in all_nodes:
        mesh.add_node(n.id, n.x, n.y, n.z)
    for e in all_elems:
        mesh.add_element(e.id, e.node_ids, e.elem_type, pid=e.pid)

    print(f"Total 3D Mesh: {mesh.num_nodes} Nodes, {mesh.num_elements} Elements, {3 * mesh.num_nodes} DOFs.")

    # 2. Material Definition
    disp_mat = J2Plasticity3D(E=4000.0, nu=0.3, sigma_y0=80.0, H=400.0)
    materials = {0: disp_mat}

    # 3. Surface Tie Constraints (Display Bottom to Plate Top)
    # Slave nodes: Display bottom surface nodes (y = 0.0, x <= -10 or x >= 10)
    slave_l_nids = [n.id for n in d_nodes if abs(n.y - 0.0) < 1e-5 and n.x <= -10.0 + 1e-5]
    slave_r_nids = [n.id for n in d_nodes if abs(n.y - 0.0) < 1e-5 and n.x >= 10.0 - 1e-5]

    tie_left = SurfaceTieConstraint3D(
        slave_node_ids=slave_l_nids,
        master_faces=lp_top_faces,
        nid_to_idx=mesh.nid_to_idx,
        coords=mesh.coords,
        penalty_stiffness=1e5,
        name="TIE_LEFT_PLATE"
    )

    tie_right = SurfaceTieConstraint3D(
        slave_node_ids=slave_r_nids,
        master_faces=rp_top_faces,
        nid_to_idx=mesh.nid_to_idx,
        coords=mesh.coords,
        penalty_stiffness=1e5,
        name="TIE_RIGHT_PLATE"
    )

    # 4. Prescribed Rigid Rotation Boundary Conditions
    # Pivots: Left pivot (-3.0, 0.0, z), Right pivot (+3.0, 0.0, z)
    left_plate_nids = [n.id for n in lp_nodes]
    right_plate_nids = [n.id for n in rp_nodes]

    pivot_L = np.array([-3.0, 0.0, 0.0])
    pivot_R = np.array([3.0, 0.0, 0.0])

    theta_target = np.radians(90.0)  # 90 deg per plate = 180 deg combined fold

    def apply_rigid_plate_bcs(solver: DynamicSolver3D, t: float):
        # C2-smooth ramp s(t) = 10*t^3 - 15*t^4 + 6*t^5
        s = 10.0 * (t**3) - 15.0 * (t**4) + 6.0 * (t**5)
        th_L = -theta_target * s
        th_R = +theta_target * s

        R_L = np.array([
            [np.cos(th_L), -np.sin(th_L), 0.0],
            [np.sin(th_L),  np.cos(th_L), 0.0],
            [0.0,           0.0,          1.0]
        ])

        R_R = np.array([
            [np.cos(th_R), -np.sin(th_R), 0.0],
            [np.sin(th_R),  np.cos(th_R), 0.0],
            [0.0,           0.0,          1.0]
        ])

        for nid in left_plate_nids:
            idx = mesh.nid_to_idx[nid]
            X0 = mesh.coords[idx]
            X_rel = X0 - pivot_L
            X_rot = pivot_L + R_L @ X_rel
            u_prescribed = X_rot - X0

            solver.set_dirichlet_bc(3 * idx + 0, u_prescribed[0])
            solver.set_dirichlet_bc(3 * idx + 1, u_prescribed[1])
            solver.set_dirichlet_bc(3 * idx + 2, u_prescribed[2])

        for nid in right_plate_nids:
            idx = mesh.nid_to_idx[nid]
            X0 = mesh.coords[idx]
            X_rel = X0 - pivot_R
            X_rot = pivot_R + R_R @ X_rel
            u_prescribed = X_rot - X0

            solver.set_dirichlet_bc(3 * idx + 0, u_prescribed[0])
            solver.set_dirichlet_bc(3 * idx + 1, u_prescribed[1])
            solver.set_dirichlet_bc(3 * idx + 2, u_prescribed[2])

    # 5. Initialize 3D Solver
    solver = DynamicSolver3D(mesh, materials)
    solver.add_constraint(tie_left)
    solver.add_constraint(tie_right)

    # 6. Execute Time Stepping Loop
    n_steps = 10
    dt = 1.0 / n_steps
    t_curr = 0.0

    t_start_wall = time.time()

    print("\nStarting 3D Implicit Time Integration...")
    for step in range(1, n_steps + 1):
        t_next = step * dt
        apply_rigid_plate_bcs(solver, t_next)

        converged, n_iter = solver.solve_step(dt=dt)
        t_curr = t_next

        deg_L = np.degrees(theta_target * (10*(t_curr**3) - 15*(t_curr**4) + 6*(t_curr**5)))
        print(f"Step {step:2d}/{n_steps}: t={t_curr:.2f}, Drive Angle={deg_L:5.1f}°, Newton Iters={n_iter}, Converged={converged}")

        if not converged:
            print("ERROR: Solver failed to converge!")
            break

    wall_time = time.time() - t_start_wall
    print(f"\n3D Simulation Completed in {wall_time:.2f} seconds!")
    print("=" * 70)


if __name__ == "__main__":
    run_3d_display_fold()
