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

from gen_ex12_inp import generate as generate_inp_text
from dispsolver.mesh.display_builder import build_display_grid
from dispsolver.fold_model_config import FoldModelConfig, DEFAULT_CONFIG
from dispsolver.material.factory import build_material_instance
from dispsolver.material.type_tags import J2_PLASTIC
from dispsolver.mesh import Mesh
from dispsolver.mesh.plate_builder import create_folding_plate_parts
from dispsolver.constraint.rbe2_condensed import KinematicRBE2Constraint
from dispsolver.constraint.surface_tie import SurfaceTieConstraint
from ex12_abaqus_inp_plate_fold import run_abaqus_inp_folding, run_folding_from_result


def run_read(config: FoldModelConfig = DEFAULT_CONFIG, elem_jit: str = "jax"):
    """Read the standard on-disk .inp deck and solve.

    NOTE: `config` here only affects solver tuning/element type -- the
    .inp file's geometry/materials are whatever gen_ex12_inp.py last
    wrote. Re-run gen_ex12_inp.py to pick up a geometry/material config
    change (see README_folding_model.md).
    """
    inp_path = os.path.join(os.path.dirname(__file__), "ex12_rigid_plate_display_fold.inp")
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"{inp_path} not found. Run gen_ex12_inp.py first.")
    return run_abaqus_inp_folding(
        inp_path=inp_path,
        before_png_name="ex13_read_before_folding_shape.png",
        after_png_name="ex13_read_final_folding_shape.png",
        result_name="ex13_read_result.pkl",
        config=config,
        elem_jit=elem_jit,
    )


def run_roundtrip(config: FoldModelConfig = DEFAULT_CONFIG, elem_jit: str = "jax"):
    """Build the .inp text in-memory via gen_ex12_inp.generate(config),
    write it to a temp file, then read+solve it through the same path
    as `read`. Unlike `read`, this DOES pick up geometry/material
    config changes immediately (no stale .inp file involved).
    """
    inp_text = generate_inp_text(config)
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
            config=config,
            elem_jit=elem_jit,
        )
    finally:
        os.remove(tmp_path)


def run_build(config: FoldModelConfig = DEFAULT_CONFIG, elem_jit: str = "jax"):
    """Construct the same physical-layer PET-PSA/Q4_VISCO_SIMO model
    directly as Python objects (mirroring gen_ex12_inp.py's geometry/
    materials, but with no .inp text produced or parsed anywhere), then
    hand it to the same `run_folding_from_result` solve loop as
    `read`/`roundtrip`.

    Layer structure comes from `config.geometry.layer_pattern` x
    `n_layer_pairs`, the SAME config object gen_ex12_inp.py reads --
    there is no second copy of these numbers to drift out of sync.
    Each physical layer gets its own pid so it shows up as its own
    selectable Part/Layer in the Qt viewer, matching the .inp path
    (there, distinct pids come from distinct *SOLID SECTION statements,
    see dispsolver/io/model_builder.py::_build_sections; here, from a
    fresh pid per layer-loop iteration -- same pid granularity, no
    *MATERIAL-name trick needed either way since materials are now
    defined once in `config.materials.definitions` and referenced by
    name).
    """
    # Mesh topology comes from the shared builder (the .inp writer uses
    # the exact same call), so node/element numbering and void handling
    # cannot drift between the two model-construction paths.
    grid = build_display_grid(config)
    geo = config.geometry
    mats = config.materials
    st = config.solver

    # One pid per physical layer, 1-based -- matches the .inp path, where
    # model_builder.py assigns one pid per *SOLID SECTION statement.
    PID_NAME = {layer_idx + 1: mat_name
                for layer_idx, (_r0, _r1, mat_name) in enumerate(grid.layer_row_spans)}
    next_pid = len(grid.layer_row_spans) + 1

    layer_etype = {}
    for pid, mat_name in PID_NAME.items():
        mdef = mats.definitions[mat_name]
        layer_etype[pid] = (st.pet_element_type if mdef.type == J2_PLASTIC
                            else st.psa_element_type)

    mesh = Mesh()
    for nid, x_val, y_val in grid.nodes:
        mesh.add_node(nid, x_val, y_val)
    bottom_surface_nids = grid.bottom_surface_nids

    for cell in grid.cells:
        pid = cell.layer_idx + 1
        mesh.add_element(cell.eid, list(cell.conn), layer_etype[pid], pid=pid)

    plates = create_folding_plate_parts(
        left_x_range=(-geo.display_half_length, -geo.hinge_half_gap),
        right_x_range=(geo.hinge_half_gap, geo.display_half_length),
        y_range=(-geo.plate_thickness, 0.0),
        left_pivot=(-geo.hinge_pivot_x, 0.0), right_pivot=(geo.hinge_pivot_x, 0.0),
        nx=geo.plate_mesh_nx, ny=geo.plate_mesh_ny, base_node_id=10000, base_elem_id=10000,
    )
    left_plate, right_plate = plates["left"], plates["right"]
    # Each side gets its OWN pid (previously both shared one STEEL_PID) --
    # matches the .inp path, which now has separate PLATE_LEFT/PLATE_RIGHT
    # *SOLID SECTION pids (model_builder.py::_build_sections). This is also
    # what lets part_names label them "Plate Left"/"Plate Right" distinctly
    # instead of one shared "Layer N (STEEL)" entry.
    plate_pids = {}
    for side_name in plates:            # "left", "right" -- insertion order
        plate_pids[side_name] = next_pid
        next_pid += 1
    for side_name, p_data in plates.items():
        pid = plate_pids[side_name]
        for nid_p, coord in p_data["nodes_dict"].items():
            mesh.add_node(nid_p, coord[0], coord[1])
        for eid_p, conn in p_data["elements_dict"].items():
            mesh.add_element(eid_p, conn, "Q4", pid=pid)

    # Materials -- built ONCE per distinct name via material_factory
    # (the same dispatch gen_ex12_inp.py's *MATERIAL-block emission
    # uses), then reused across every pid that references it. Materials
    # are stateless (per-element internal variables live in
    # solver.state, not on the object), so sharing one instance across
    # multiple pids is safe.
    materials, material_params, material_names, material_types = {}, {}, {}, {}
    material_cache = {}  # material name -> (obj, params), built lazily
    for pid, name in PID_NAME.items():
        material_names[pid] = name
        material_types[pid] = mats.definitions[name].type
        if name not in material_cache:
            material_cache[name] = build_material_instance(mats.definitions[name])
        obj, params = material_cache[name]
        materials[pid] = obj
        material_params[pid] = params

    steel_mdef = mats.definitions["STEEL"]
    steel_obj, steel_params = build_material_instance(steel_mdef)  # built once, shared
    part_names = {}
    for side_name, pid in plate_pids.items():
        materials[pid] = steel_obj
        material_params[pid] = steel_params
        material_names[pid] = "STEEL"
        material_types[pid] = steel_mdef.type
        part_names[pid] = f"Plate {side_name.capitalize()}"

    # RBE2 exact kinematic condensation for both hinge pivots (same
    # mechanism as the .inp `*RIGID BODY` path, see AGENTS.md 4.10).
    rbe2_left = KinematicRBE2Constraint(mesh=mesh, master_id=left_plate["master_rp_id"],
                                         slave_ids=left_plate["slave_nids"])
    rbe2_right = KinematicRBE2Constraint(mesh=mesh, master_id=right_plate["master_rp_id"],
                                          slave_ids=right_plate["slave_nids"])

    nid_to_idx = mesh.node_id_to_index()
    coords = mesh.nodes_array()
    left_disp_bot = [n for n in bottom_surface_nids if mesh.nodes[n].x <= -geo.hinge_half_gap]
    right_disp_bot = [n for n in bottom_surface_nids if mesh.nodes[n].x >= geo.hinge_half_gap]
    tie_left = SurfaceTieConstraint(
        slave_node_ids=left_disp_bot, master_node_ids=left_plate["top_surface_nids"],
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=1e4, name="TIE_LEFT",
    )
    tie_right = SurfaceTieConstraint(
        slave_node_ids=right_disp_bot, master_node_ids=right_plate["top_surface_nids"],
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=1e4, name="TIE_RIGHT",
    )

    # Boundary conditions, matching the .inp *BOUNDARY blocks exactly:
    # master RP translation fixed, rotation prescribed to +-theta_max.
    theta_rad = np.radians(config.drive.theta_max_deg)
    boundaries = [
        SimpleNamespace(nset=str(left_plate["master_rp_id"]), dof1=1, dof2=2, value=0.0),
        SimpleNamespace(nset=str(right_plate["master_rp_id"]), dof1=1, dof2=2, value=0.0),
        SimpleNamespace(nset=str(left_plate["master_rp_id"]), dof1=6, dof2=6, value=-theta_rad),
        SimpleNamespace(nset=str(right_plate["master_rp_id"]), dof1=6, dof2=6, value=theta_rad),
    ]

    result = SimpleNamespace(
        mesh=mesh,
        materials=materials,
        material_params=material_params,
        # Same pid -> *MATERIAL name map the .inp path produces, so
        # postprocessing labels the layers PET/PSA identically in all modes.
        material_names=material_names,
        material_types=material_types,
        # pid -> human display-name override for the two plate parts,
        # mirroring model_builder.py's part_names so the Qt viewer/
        # model_review show "Plate Left"/"Plate Right" in build-mode too.
        part_names=part_names,
        solver_config={"density": 1e-9},
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
        config=config,
        elem_jit=elem_jit,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["read", "roundtrip", "build"], default="read",
        help="Model source: 'read' (existing .inp), 'roundtrip' "
             "(generate() in-memory -> temp .inp -> read+solve), or "
             "'build' (pure Python objects, no .inp at all).",
    )
    parser.add_argument(
        "--elem_jit", choices=["jax", "numba", "numpy"], default="jax",
        help="Element JIT backend (default: jax)",
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="Open the Qt postprocess viewer automatically once the "
             "solve finishes (loads the just-saved .pkl -- same as "
             "running with --open <result_path> afterward).",
    )
    parser.add_argument(
        "--open", metavar="PKL_PATH", default=None,
        help="Skip solving entirely and just open an existing saved "
             ".pkl result in the Qt viewer (e.g. --open "
             "examples/ex13_build_result.pkl). Ignores --mode/--viewer.",
    )
    args = parser.parse_args()

    if args.open:
        from dispsolver.postprocess.viewer import launch_from_result
        launch_from_result(args.open)
        return

    config = DEFAULT_CONFIG
    if args.mode == "read":
        info = run_read(config, elem_jit=args.elem_jit)
    elif args.mode == "roundtrip":
        info = run_roundtrip(config, elem_jit=args.elem_jit)
    else:
        info = run_build(config, elem_jit=args.elem_jit)

    print("=" * 100)
    print(f" EX13 [{args.mode}] SUMMARY: nodes={info['n_nodes']}  elements={info['n_elements']}  "
          f"rbe2_constraints={info['n_rbe2_constraints']}  penalty_constraints={info['n_penalty_constraints']}  "
          f"reached_target={info['reached_target']}")
    print("=" * 100)

    if args.viewer:
        from dispsolver.postprocess.viewer import launch_from_result
        launch_from_result(info["result_path"])


if __name__ == "__main__":
    main()
