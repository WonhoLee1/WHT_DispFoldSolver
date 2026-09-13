"""Production run with all-new-Numba elements enabled (CPE4H + fixed SRI)."""
import sys
sys.path.insert(0, "examples")
from dispsolver.solver import DynamicSolver
DynamicSolver.enable_new_numba_elements = True

from ex12_abaqus_inp_plate_fold import run_abaqus_inp_folding

run_abaqus_inp_folding(elem_jit="numba", max_steps=40)
