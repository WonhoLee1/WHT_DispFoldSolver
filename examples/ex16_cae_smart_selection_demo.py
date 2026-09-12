"""Example 16: Commercial CAE Smart Region Selection Demo (Box, Sphere, Cylinder, Normal, findAt, Set Algebra)

Demonstrates how to use the Abaqus-like CAE object hierarchy and smart selection features
on a 3D Foldable Display Assembly model:
1. Box Filtering (Bounding Box)
2. Sphere Filtering (Local Pivot Zone)
3. Cylinder Filtering (Hinge Rotational Axis)
4. Surface Normal Filtering (Top/Bottom Faces)
5. Proximity Search & findAt (n-nearest nodes/elements)
6. Set Algebra & Interface Node Overlap Extraction
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

from dispsolver.model import Model, Part, GeneralSet

# Global Matplotlib Font Size
plt.rcParams["font.size"] = 9


def build_3d_display_part() -> Part:
    """Build a 3D Hex8 block representing a foldable display panel.
    
    Dimensions:
      X in [-40.0, 40.0] mm (length)
      Y in [ 0.0,   0.5] mm (thickness)
      Z in [ 0.0,  10.0] mm (width)
    """
    part = Part(name="Display3D", dim=3)

    nx, ny, nz = 32, 2, 4
    x_coords = np.linspace(-40.0, 40.0, nx + 1)
    y_coords = np.linspace(0.0, 0.5, ny + 1)
    z_coords = np.linspace(0.0, 10.0, nz + 1)

    node_grid = {}
    nid = 1
    for k, z in enumerate(z_coords):
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                part.add_node(nid, [x, y, z])
                node_grid[(i, j, k)] = nid
                nid += 1

    eid = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                conn = [
                    node_grid[(i, j, k)],
                    node_grid[(i + 1, j, k)],
                    node_grid[(i + 1, j + 1, k)],
                    node_grid[(i, j + 1, k)],
                    node_grid[(i, j, k + 1)],
                    node_grid[(i + 1, j, k + 1)],
                    node_grid[(i + 1, j + 1, k + 1)],
                    node_grid[(i, j + 1, k + 1)],
                ]
                part.add_element(eid, "C3D8", conn)
                eid += 1

    return part


def main():
    print("======================================================================")
    print("  Ex16: Commercial CAE Smart Selection Demo (Box, Sphere, Cylinder)")
    print("======================================================================")

    model = Model(name="FoldableDisplayCAE", dim=3)
    part = build_3d_display_part()
    model.parts["Display3D"] = part

    print(f"\n[Model Info] Part '{part.name}' created:")
    print(f"  - Total Nodes:    {part.num_nodes}")
    print(f"  - Total Elements: {part.num_elements}")

    # ─── 1. Box Selection (Bounding Box) ───────────────────────────────
    print("\n--- 1. Bounding Box Selection (create_set_from_box) ---")
    left_plate_box = part.create_set_from_box(
        name="LEFT_PLATE_BOX",
        x_range=(-40.0, -5.0),  # Extends to -5.0 mm to overlap with Hinge Cylinder (R=6.0)
        y_range=(-0.1, 0.6),
        z_range=(-0.1, 10.1),
        entity_type="ALL"
    )
    right_plate_box = part.create_set_from_box(
        name="RIGHT_PLATE_BOX",
        x_range=(10.0, 40.0),
        y_range=(-0.1, 0.6),
        z_range=(-0.1, 10.1),
        entity_type="ALL"
    )
    print(f"  - LEFT_PLATE_BOX:  {len(left_plate_box.node_ids)} nodes, {len(left_plate_box.element_ids)} elements")
    print(f"  - RIGHT_PLATE_BOX: {len(right_plate_box.node_ids)} nodes, {len(right_plate_box.element_ids)} elements")

    # ─── 2. Sphere Selection (Local Pivot Zone) ───────────────────────
    print("\n--- 2. Sphere Selection (create_set_from_sphere) ---")
    left_pivot_sphere = part.create_set_from_sphere(
        name="LEFT_PIVOT_SPHERE",
        center=[-3.0, 0.25, 5.0],
        radius=5.0,
        inner_radius=0.0,
        entity_type="ALL"
    )
    right_pivot_sphere = part.create_set_from_sphere(
        name="RIGHT_PIVOT_SPHERE",
        center=[3.0, 0.25, 5.0],
        radius=5.0,
        inner_radius=0.0,
        entity_type="ALL"
    )
    print(f"  - LEFT_PIVOT_SPHERE  (Center=[-3, 0.25, 5], R=5): {len(left_pivot_sphere.node_ids)} nodes, {len(left_pivot_sphere.element_ids)} elements")
    print(f"  - RIGHT_PIVOT_SPHERE (Center=[+3, 0.25, 5], R=5): {len(right_pivot_sphere.node_ids)} nodes, {len(right_pivot_sphere.element_ids)} elements")

    # ─── 3. Cylinder Selection (Hinge Rotational Axis) ─────────────────
    print("\n--- 3. Cylinder Selection (create_set_from_cylinder) ---")
    hinge_axis_cylinder = part.create_set_from_cylinder(
        name="HINGE_AXIS_CYLINDER",
        point1=[0.0, 0.25, 0.0],
        point2=[0.0, 0.25, 10.0],  # Segment from Z=0 to Z=10
        radius=6.0,
        inner_radius=0.0,
        entity_type="ALL"
    )
    print(f"  - HINGE_AXIS_CYLINDER (Z-axis segment [0,10], R=6.0): {len(hinge_axis_cylinder.node_ids)} nodes, {len(hinge_axis_cylinder.element_ids)} elements")

    # ─── 4. Surface by Outward Normal ──────────────────────────────────
    print("\n--- 4. Surface by Normal (create_surface_from_normal) ---")
    top_surface = part.create_surface_from_normal(
        name="TOP_SURFACE",
        direction=[0.0, 1.0, 0.0],
        angle_tol_deg=15.0
    )
    bottom_surface = part.create_surface_from_normal(
        name="BOTTOM_SURFACE",
        direction=[0.0, -1.0, 0.0],
        angle_tol_deg=15.0
    )
    print(f"  - TOP_SURFACE    (Normal=[0, 1, 0]):  {len(top_surface.node_ids)} nodes, {len(top_surface.faces)} boundary faces")
    print(f"  - BOTTOM_SURFACE (Normal=[0, -1, 0]): {len(bottom_surface.node_ids)} nodes, {len(bottom_surface.faces)} boundary faces")

    # ─── 5. Proximity Search & findAt ──────────────────────────────────
    print("\n--- 5. Proximity Search & findAt ---")
    # Closest 3 nodes to Hinge Center (0, 0.25, 5.0)
    top_3_node_tuples = part.find_closest_nodes([0.0, 0.25, 5.0], n=3, return_distances=True)
    print(f"  - Top 3 closest nodes to (0.0, 0.25, 5.0):")
    for nid, d in top_3_node_tuples:
        c = part.nodes[nid]
        print(f"    * Node {nid:3d}: coords=({c[0]:6.2f}, {c[1]:4.2f}, {c[2]:4.2f}), distance={d:.4f} mm")

    # findAt GeneralSet creation
    corner_set = part.find_at(coords=[40.0, 0.5, 10.0], name="CORNER_SEED", entity_type="ALL", n=1)
    print(f"  - findAt at corner (40, 0.5, 10): Node {list(corner_set.node_ids)}, Elem {list(corner_set.element_ids)}")

    # ─── 6. Set Algebra & Interface Overlap ───────────────────────────
    print("\n--- 6. Set Algebra & Interface Node Overlap ---")
    both_plates = left_plate_box | right_plate_box
    hinge_gap_only = hinge_axis_cylinder - both_plates
    print(f"  - Union (both_plates = LEFT | RIGHT): {len(both_plates.node_ids)} nodes, {len(both_plates.element_ids)} elements")
    print(f"  - Difference (hinge_gap_only = CYLINDER - PLATES): {len(hinge_gap_only.node_ids)} nodes, {len(hinge_gap_only.element_ids)} elements")

    # Overlapping interface nodes between hinge cylinder and left plate
    shared_interface_nodes = part.get_overlapping_nodes(hinge_axis_cylinder, left_plate_box, include_elements=True)
    print(f"  - Shared Interface Nodes between HINGE_CYLINDER and LEFT_PLATE: {len(shared_interface_nodes)} nodes")
    print(f"    Node IDs: {shared_interface_nodes[:10]}...")

    # ─── 7. Visualization ──────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw all nodes in grey
    all_coords = np.array([part.nodes[nid] for nid in sorted(part.nodes.keys())])
    ax.scatter(all_coords[:, 0], all_coords[:, 2], all_coords[:, 1], c="lightgrey", s=10, alpha=0.3, label="전체 노드 (All Nodes)")

    # Highlight Box: Left & Right Plates
    left_coords = np.array([part.nodes[nid] for nid in left_plate_box.node_ids])
    right_coords = np.array([part.nodes[nid] for nid in right_plate_box.node_ids])
    ax.scatter(left_coords[:, 0], left_coords[:, 2], left_coords[:, 1], c="blue", s=25, label="Box: 좌측 플레이트 (Left Plate)")
    ax.scatter(right_coords[:, 0], right_coords[:, 2], right_coords[:, 1], c="cyan", s=25, label="Box: 우측 플레이트 (Right Plate)")

    # Highlight Sphere: Left & Right Pivots
    lpivot_coords = np.array([part.nodes[nid] for nid in left_pivot_sphere.node_ids])
    rpivot_coords = np.array([part.nodes[nid] for nid in right_pivot_sphere.node_ids])
    ax.scatter(lpivot_coords[:, 0], lpivot_coords[:, 2], lpivot_coords[:, 1], c="red", s=40, marker="^", label="Sphere: 좌측 피벗 (Left Pivot)")
    ax.scatter(rpivot_coords[:, 0], rpivot_coords[:, 2], rpivot_coords[:, 1], c="magenta", s=40, marker="^", label="Sphere: 우측 피벗 (Right Pivot)")

    # Highlight Cylinder: Hinge Axis
    cyl_coords = np.array([part.nodes[nid] for nid in hinge_axis_cylinder.node_ids])
    ax.scatter(cyl_coords[:, 0], cyl_coords[:, 2], cyl_coords[:, 1], c="green", s=30, marker="s", alpha=0.7, label="Cylinder: 힌지 축 영역 (Hinge Cylinder)")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_zlabel("Y (mm)")
    ax.set_title("Ex16: 3D 스마트 세트 선택 데모 (Box, Sphere, Cylinder)", fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "ex16_smart_selection_demo.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved Visualization Plot] {output_path}")
    print("======================================================================")


if __name__ == "__main__":
    main()
