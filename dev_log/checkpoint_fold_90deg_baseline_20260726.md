# Checkpoint: validated 90deg/side U-shape fold baseline (2026-07-26)

Tag: `fold-90deg-baseline-20260726` (commit `4273777`).

## What this locks in

`examples/ex12_abaqus_inp_plate_fold.py` + `examples/gen_ex12_inp.py`
(graded-mesh .inp generator) reach **full 90deg/side (180deg combined)
closure**, 101 steps, zero cutbacks, 1 Newton iteration/step, ~85s wall
time, `n_inverted == 0` throughout (see AGENTS.md §1.0/§4.12).

Display model at this checkpoint: **1 material (PET, elastic-plastic,
E=4000MPa/nu=0.3/sigma_y0=80MPa/H=400MPa), 4 mesh rows** through 0.5mm
thickness (`gen_ex12_inp.py:58-59`, `ny_disp=4`) — mesh-resolution rows,
not a true material stackup. No viscoelastic material wired into this
pipeline at all (`ViscoelasticMaterial` exists in
`dispsolver/material/viscoelastic.py` and is used by `ex03_*`/
`model_builder.py`, but `gen_ex12_inp.py` never referenced it).

This is the reference point for judging any regression introduced by the
next change (expanding to 14 layers / adding viscoelastic adhesive
layers).
