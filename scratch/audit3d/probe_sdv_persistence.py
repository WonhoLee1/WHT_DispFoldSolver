"""
Probe: does DynamicSolver3D.assemble_system(..., update_state=True) actually
persist updated SDVs (e.g. J2 equivalent plastic strain) back into
self.elem_sdvs?

Hypothesis (from reading dynamic3d.py's fast-path loop over
self.elem_kernel_groups): `elem_indices` is built as
`np.array(indices, dtype=np.int64)` (advanced/fancy indexing). In NumPy,
`some_array[fancy_int_array]` ALWAYS returns a new array (a copy), never a
view -- confirmed separately with a plain numpy micro-test. So
`sub_sdvs = self.elem_sdvs[elem_indices]` is a copy regardless of the
update_state flag; the Numba kernel writes the new SDV values into that
copy (`sdvs_elem[gp] = sdv_gp_new`, itself a per-row view *into the copy*,
not into self.elem_sdvs), and nothing in assemble_system ever assigns the
result back into `self.elem_sdvs[elem_indices] = sub_sdvs`. If true, this
means self.elem_sdvs NEVER changes, no matter how many increments are
solved -- i.e. plastic strain (and viscoelastic history) can never
accumulate through this fast/Numba assembly path.
"""
import numpy as np
import sys
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver")
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver\scratch\audit3d")
import _bootstrap
_bootstrap.load_full_element3d_package()
# Pre-register the patched eas_tl module under its real dotted name so that
# dynamic3d.py's `from dispsolver.element3d.c3d8_eas_tl_numba import ...`
# (executed lazily inside assemble_system) hits the already-fixed copy in
# sys.modules instead of re-triggering the broken file on disk.
_bootstrap.load_element3d_module("c3d8_eas_tl_numba")

from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.material3d.numba_materials import MAT_J2_PLASTICITY

mesh = Mesh3D()
coords = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
], dtype=np.float64)
for i in range(8):
    mesh.add_node(i + 1, *coords[i])
mesh.add_element(1, [1,2,3,4,5,6,7,8], "C3D8I", pid=0)

class _MatObj:
    # NOTE: a plain {"type": ..., "props": [...]} dict crashes
    # DynamicSolver3D._setup_numba_topology() unconditionally -- see the
    # separately-reported "material dict convention" bug below. Route
    # around it here with an attribute-based mat_obj (the first branch
    # checked, hasattr(mat_obj, "mat_type") and hasattr(mat_obj, "props"))
    # so this probe tests SDV persistence in isolation.
    mat_type = MAT_J2_PLASTICITY
    props = [1000.0, 0.3, 1.0, 100.0]

materials = {0: _MatObj()}
solver = DynamicSolver3D(mesh, materials=materials)

print("elem_mat_types:", solver.elem_mat_types, " elem_sdvs.shape:", solver.elem_sdvs.shape)
print("elem_kernel_groups:", {k: v.tolist() for k, v in solver.elem_kernel_groups.items()})

# Uniaxial stretch of the top face in z, well past yield (sigma_y0=1.0 is tiny
# vs E=1000, so even a small strain triggers plastic flow).
u = np.zeros(24)
nid_map = mesh.node_id_to_index()
for nid in [5, 6, 7, 8]:  # top face nodes (z=1)
    idx = nid_map[nid]
    u[3*idx + 2] = 0.2  # uniaxial z-stretch, strain ~0.2 >> yield strain ~1e-3

sdv_before = solver.elem_sdvs.copy()
K, f = solver.assemble_system(u, dt=1.0, update_state=True)
sdv_after_call1 = solver.elem_sdvs.copy()

print()
print("=== SDV persistence probe (J2 plasticity, single C3D8I element) ===")
print("max|elem_sdvs| before any assembly       =", np.max(np.abs(sdv_before)))
print("max|elem_sdvs| after assemble(update_state=True), call #1 =", np.max(np.abs(sdv_after_call1)))
print("eq_plastic_strain (sdv index 6) after call #1 =", solver.elem_sdvs[0, :, 6] if solver.elem_sdvs.shape[2] > 6 else "N/A (sdv width too small)")
print("f_int max|f| (should be nonzero, material responded to strain) =", np.max(np.abs(f)))

# Call again with update_state=True at the SAME displacement -- if SDVs
# persisted after call #1, eq_plastic_strain is already elevated, so the
# trial stress at the SAME total strain should now be (nearly) purely
# elastic unloading from the already-updated back-stress-free J2 model
# (isotropic hardening only here) -- meaning a repeat call at IDENTICAL u
# should give IDENTICAL f_int/sdv regardless (since e_trial = E - ep_old
# with ep_old already having flowed to satisfy f_trial=0 at this E).
# The decisive test is instead: call at u=0 (unload completely) and check
# if eq_plastic_strain (a p-strain persisted from call #1) is still nonzero
# -- if elem_sdvs never actually updated, calling at u=0 will show sdv==0
# still, AND the K/f at u=0 will look like a virgin elastic state (zero
# stress), rather than reflecting any residual/back-stress from the prior
# plastic excursion.
K0, f0 = solver.assemble_system(np.zeros(24), dt=1.0, update_state=True)
sdv_after_unload_call = solver.elem_sdvs.copy()
print()
print("After a SECOND assemble_system(u=0, update_state=True) call:")
print("max|elem_sdvs| now                        =", np.max(np.abs(sdv_after_unload_call)))
print("did elem_sdvs change AT ALL from call #1's snapshot? ", not np.allclose(sdv_after_call1, sdv_after_unload_call))
print("(sdv_after_call1 itself vs the true zero-initial state -- did *that* call's SDVs ever leave the solver's persistent self.elem_sdvs array?)")
print("sdv_before was all zero:", np.allclose(sdv_before, 0.0))
print("elem_sdvs right now (post both calls) all zero too?:", np.allclose(solver.elem_sdvs, 0.0))
