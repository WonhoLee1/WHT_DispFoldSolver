"""ex12 fold re-run AFTER the F6 EAS transpose fix, for before/after comparison."""
import sys, os
sys.path.insert(0, "examples")
from ex12_abaqus_inp_plate_fold import read_abaqus_input, run_folding_from_result

result = read_abaqus_input(os.path.join("examples", "ex12_rigid_plate_display_fold.inp"))
run_folding_from_result(
    result,
    before_png_name="ex12_after_f6_before.png",
    after_png_name="ex12_after_f6_final.png",
    result_name="ex12_after_f6_result.pkl",
    max_steps=30,
    elem_jit="numba",
)
