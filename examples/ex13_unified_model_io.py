"""
ex13_unified_model_io.py
=========================
Unifies the folding-solve entry points around a single downstream solve
loop (`ex12_abaqus_inp_plate_fold.run_folding_from_result`), selectable
via `--mode`:

    read       : read the standard, on-disk .inp deck
                 (examples/ex12_rigid_plate_display_fold.inp, as produced
                 by gen_ex12_inp.py).
    roundtrip  : call gen_ex12_inp.generate() to build the .inp text
                 in-memory, write it to a temp file, then run the exact
                 same read+solve path -- proves the generator's output is
                 self-consistent without needing a separate .inp writer.
    build      : construct the identical 14-layer PET-PSA/Q4_VISCO_SIMO
                 model directly as Python objects (mesh, materials, RBE2/
                 tie constraints) -- no .inp text anywhere -- then call
                 the same `run_folding_from_result` solve loop. Proves the
                 .inp-based and pure-Python modeling paths are equivalent.
"""

import argparse
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from gen_ex12_inp import generate as generate_inp_text, _graded_display_x
from dispsolver.mesh import Mesh
from dispsolver.mesh.plate_builder import create_folding_plate_parts
from dispsolver.material.plastic import J2Plasticity
from dispsolver.material.arruda_boyce import ArrudaBoyce
from dispsolver.material.neohookean import NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.constraint.rbe2_condensed import KinematicRBE2Constraint
from dispsolver.constraint.surface_tie import SurfaceTieConstraint
from ex12_abaqus_inp_plate_fold import run_abaqus_inp_folding, run_folding_from_result


def run_read():
    """Read the standard on-disk .inp deck and solve."""
    inp_path = os.path.join(os.path.dirname(__file__), "ex12_rigid_plate_display_fold.inp")
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"{inp_path} not found. Run gen_ex12_inp.py first.")
    return run_abaqus_inp_folding(
        inp_path=inp_path,
        before_png_name="ex13_read_before_folding_shape.png",
        after_png_name="ex13_read_final_folding_shape.png",
        result_name="ex13_read_result.pkl",
    )


def run_roundtrip():
    """Build the .inp text in-memory via gen_ex12_inp.generate(), write it
    to a temp file, then read+solve it through the same path as `read`.
    """
    inp_text = generate_inp_text()
    fd, tmp_path = tempfile.mkstemp(suffix=".inp", prefix="ex13_roundtrip_")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(inp_text)
        return run_abaqus_inp_folding(
            inp_path=tmp_path,
            before_png_name="ex13_roundtrip_before_folding_shape.png",
            after_png_name="ex13_roundtrip_final_folding_shape.png",
            result_name="ex13_roundtrip_result.pkl",
        )
    finally:
        os.remove(tmp_path)


def run_build():
    """Construct the same 14-layer PET-PSA/Q4_VISCO_SIMO model directly as
    Python objects (mirroring gen_ex12_inp.py's geometry/materials, but
    with no .inp text produced or parsed anywhere), then hand it to the
    same `run_folding_from_result` solve loop as `read`/`roundtrip`.
    """
    xs_disp = _graded_display_x()
    nx_disp = len(xs_disp) - 1
    ny_disp = 14
    ys_disp = np.linspace(0.0, 0.5, ny_disp + 1)
    # pid assignment matches model_builder._build_sections(): one pid per
    # unique material name, first-seen order (PET first, then PSA, then
    # STEEL) -- NOT one pid per row/elset.
    LAYER_PID = [1 if j % 2 == 0 else 2 for j in range(ny_disp)]

    mesh = Mesh()
    grid_nids = np.zeros((ny_disp + 1, nx_disp + 1), dtype=int)
    nid = 1
    bottom_surface_nids = []
    for j, y_val in enumerate(ys_disp):
        for i, x_val in enumerate(xs_disp):
            grid_nids[j, i] = nid
            mesh.add_node(nid, x_val, y_val)
            if j == 0:
                bottom_surface_nids.append(nid)
            nid += 1

    eid = 1
    for j in range(ny_disp):
        pid = LAYER_PID[j]
        etype = "Q4_COROTATIONAL" if pid == 1 else "Q4_VISCO_SIMO"
        for i in range(nx_disp):
            n1, n2 = grid_nids[j, i], grid_nids[j, i + 1]
            n3, n4 = grid_nids[j + 1, i + 1], grid_nids[j + 1, i]
            mesh.add_element(eid, [n1, n2, n3, n4], etype, pid=pid)
            eid += 1

    plates = create_folding_plate_parts(
        left_x_range=(-40.0, -10.0), right_x_range=(10.0, 40.0),
        y_range=(-0.5, 0.0), left_pivot=(-3.0, 0.0), right_pivot=(3.0, 0.0),
        nx=30, ny=2, base_node_id=10000, base_elem_id=10000,
    )
    left_plate, right_plate = plates["left"], plates["right"]
    for p_data in plates.values():
        for nid_p, coord in p_data["nodes_dict"].items():
            mesh.add_node(nid_p, coord[0], coord[1])
        for eid_p, conn in p_data["elements_dict"].items():
            mesh.add_element(eid_p, conn, "Q4", pid=3)

    # Materials -- same params as gen_ex12_inp.py's PET/PSA/STEEL blocks.
    mat_pet = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=400.0)
    params_pet = {"E": 4000.0, "nu": 0.3, "sigma_y0": 80.0, "H": 400.0}

    base_psa = ArrudaBoyce()
    params_base_psa = {"mu": 0.16785, "lambda_m": 3.0, "K": 8.3333}
    wlf = {"T_ref": 25.0, "C1": 17.0, "C2": 51.6, "definition": "WLF"}
    prony = [(0.20, 0.0, 3.33)]
    mat_psa = ViscoelasticMaterial(base_psa, [0.20], [3.33], wlf_params=wlf)
    params_psa = {**params_base_psa, "base": base_psa, "prony": prony, "wlf": wlf}

    mat_steel = NeoHookean()
    params_steel = {"E": 20000.0, "nu": 0.3}

    materials = {1: mat_pet, 2: mat_psa, 3: mat_steel}
    material_params = {1: params_pet, 2: params_psa, 3: params_steel}

    # RBE2 exact kinematic condensation for both hinge pivots (same
    # mechanism as the .inp `*RIGID BODY` path, see AGENTS.md 4.10).
    rbe2_left = KinematicRBE2Constraint(mesh=mesh, master_id=left_plate["master_rp_id"],
                                         slave_ids=left_plate["slave_nids"])
    rbe2_right = KinematicRBE2Constraint(mesh=mesh, master_id=right_plate["master_rp_id"],
                                          slave_ids=right_plate["slave_nids"])

    nid_to_idx = mesh.node_id_to_index()
    coords = mesh.nodes_array()
    left_disp_bot = [n for n in bottom_surface_nids if mesh.nodes[n].x <= -10.0]
    right_disp_bot = [n for n in bottom_surface_nids if mesh.nodes[n].x >= 10.0]
    tie_left = SurfaceTieConstraint(
        slave_node_ids=left_disp_bot, master_node_ids=left_plate["top_surface_nids"],
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=1e4, name="TIE_LEFT",
    )
    tie_right = SurfaceTieConstraint(
        slave_node_ids=right_disp_bot, master_node_ids=right_plate["top_surface_nids"],
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=1e4, name="TIE_RIGHT",
    )

    # Boundary conditions, matching the .inp *BOUNDARY blocks exactly:
    # master RP translation fixed, rotation prescribed to +-90deg.
    boundaries = [
        SimpleNamespace(nset=str(left_plate["master_rp_id"]), dof1=1, dof2=2, value=0.0),
        SimpleNamespace(nset=str(right_plate["master_rp_id"]), dof1=1, dof2=2, value=0.0),
        SimpleNamespace(nset=str(left_plate["master_rp_id"]), dof1=6, dof2=6, value=-1.570796),
        SimpleNamespace(nset=str(right_plate["master_rp_id"]), dof1=6, dof2=6, value=1.570796),
    ]

    result = SimpleNamespace(
        mesh=mesh,
        materials=materials,
        material_params=material_params,
        # Same pid -> *MATERIAL name map the .inp path produces, so
        # postprocessing labels the layers PET/PSA identically in all modes.
        material_names={1: "PET", 2: "PSA", 3: "STEEL"},
        solver_config={"density": 1e-9, "t_total": 1.0, "dt_init": 0.005, "dt_max": 0.01, "dt_min": 1e-5},
        rbe2_constraints=[rbe2_left, rbe2_right],
        penalty_constraints=[tie_left, tie_right],
        rbe2_elements=[],
        boundaries=boundaries,
    )
    return run_folding_from_result(
        result,
        before_png_name="ex13_build_before_folding_shape.png",
        after_png_name="ex13_build_final_folding_shape.png",
        case_name="build (pure Python objects)",
        result_name="ex13_build_result.pkl",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["read", "roundtrip", "build"], default="read",
        help="Model source: 'read' (existing .inp), 'roundtrip' "
             "(generate() in-memory -> temp .inp -> read+solve), or "
             "'build' (pure Python objects, no .inp at all).",
    )
    args = parser.parse_args()

    if args.mode == "read":
        info = run_read()
    elif args.mode == "roundtrip":
        info = run_roundtrip()
    else:
        info = run_build()

    print("=" * 100)
    print(f" EX13 [{args.mode}] SUMMARY: nodes={info['n_nodes']}  elements={info['n_elements']}  "
          f"rbe2_constraints={info['n_rbe2_constraints']}  penalty_constraints={info['n_penalty_constraints']}  "
          f"reached_target={info['reached_target']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
