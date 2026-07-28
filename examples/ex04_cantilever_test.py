"""
ex04_cantilever_test.py
=======================
Cantilever beam bending test - verify element deformation transmission
through the multi-layer PET/PSA thickness.

Left edge clamped. UY displacement applied at the bottom-right tip node.
No RBE2 constraints. Pure element behavior test.

If elements work correctly:
  - Bottom tip node moves down by prescribed UY
  - Deformation propagates upward through the thickness
  - Top-right tip node also shows significant UY displacement
  - Through-thickness displacement gradient is roughly linear

If elements have zero through-thickness stiffness:
  - Only the bottom tip node moves
  - Upper nodes show zero displacement
"""

import os, time, numpy as np

# ============================================================
# Parameters
# ============================================================
NX         = 20        # elements along length
LENGTH     = 20.0      # mm (0 to 20)
TIP_UY     = -0.5      # mm (prescribed downward at bottom-right tip)
MAX_ITER   = 50
TOL        = 1e-4
T_TOTAL    = 1.0
DT_INITIAL = 0.05
DT_MIN     = 1e-6
DT_MAX     = 0.1
MAX_CUTBACKS = 10

FAST_ASSEMBLY = True

os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["PARDISO_NUM_THREADS"] = "4"
os.environ["XLA_FLAGS"]           = "--xla_cpu_multi_thread_eigen=false"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity
from dispsolver.material.neohookean import NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.solver import DynamicSolver
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter


# ============================================================
# Smooth C2 amplitude
# ============================================================
class SmoothAmplitude:
    """C2-continuous quintic step."""
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1, self.T = t0, t1, t1 - t0
    def __call__(self, t):
        tau = np.clip((t - self.t0) / self.T, 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


# ============================================================
# Mesh - layered beam (same layout as ex03)
# ============================================================
xs = np.linspace(0.0, LENGTH, NX + 1)
nx = len(xs)

ys_list = [0.0]
row_pids = []
current_y = 0.0
layer_thickness = 0.05
for layer in range(7):
    if layer % 2 == 0:
        dy = layer_thickness / 3.0
        for _ in range(3):
            current_y += dy
            ys_list.append(current_y)
            row_pids.append(layer)
    else:
        current_y += layer_thickness
        ys_list.append(current_y)
        row_pids.append(layer)
ys = np.array(ys_list)
ny = len(ys)

total_height = ys[-1] - ys[0]
print(f"Mesh: {nx}×{ny} nodes  ({nx-1}×{ny-1} elements)")
print(f"Beam: {LENGTH}mm × {total_height:.4f}mm")
print(f"Layers: {list(row_pids)}")

mesh = Mesh()
for j, y in enumerate(ys):
    for i, x in enumerate(xs):
        mesh.add_node(j * nx + i, x, y)

elem_idx = 0
for j in range(ny - 1):
    pid = row_pids[j]
    for i in range(nx - 1):
        n1 = j * nx + i
        mesh.add_element(elem_idx, [n1, n1+1, n1+nx+1, n1+nx], "QUAD4", pid=pid)
        elem_idx += 1
print(f"Total elements: {elem_idx}")

# ============================================================
# Materials (identical to ex03)
# ============================================================
pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
psa_mat = ViscoelasticMaterial(NeoHookean(), g_i=[0.8], tau_i=[1.0])
materials = {layer: pet_mat if layer % 2 == 0 else psa_mat for layer in range(7)}
material_params = {layer: ({} if layer % 2 == 0 else {"E": 10.0, "nu": 0.49}) for layer in range(7)}

# ============================================================
# Solver (quasistatic)
# ============================================================
element_type = {layer: "Q4_EAS" if layer % 2 == 0 else "Q4_VISCO_SIMO" for layer in range(7)}

solver = DynamicSolver(
    mesh, materials, rho=1000.0, material_params=material_params,
    max_iter=MAX_ITER, tol=TOL, verbose=True,
    element_type=element_type, fast_assembly=FAST_ASSEMBLY, mode="quasistatic",
)

# ============================================================
# Boundary Conditions
# ============================================================
nid_to_idx = mesh.node_id_to_index()

bc_dofs, bc_base = [], []
# Clamp left edge
for j in range(ny):
    idx = nid_to_idx[j * nx + 0]
    bc_dofs += [idx * 2, idx * 2 + 1]
    bc_base += [0.0, 0.0]

# UY at bottom-right tip
tip_node = 0 * nx + (nx - 1)
tip_idx = nid_to_idx[tip_node]
bc_dofs.append(tip_idx * 2 + 1)
bc_base.append(TIP_UY)

# Ramp for tip only
all_amps = [None] * (len(bc_dofs) - 1) + [SmoothAmplitude(0.0, T_TOTAL)]
solver.set_prescribed_dofs(bc_dofs, bc_base, amplitudes=all_amps)

top_tip_node = (ny - 1) * nx + (nx - 1)
mid_right_node = ((ny - 1) // 2) * nx + (nx - 1)
top_tip_idx = nid_to_idx[top_tip_node]
mid_right_idx = nid_to_idx[mid_right_node]

print(f"Clamped: {ny} left-edge nodes   Tip UY: {TIP_UY}mm ramped over {T_TOTAL}s")

# ============================================================
# Run
# ============================================================
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
filepath = os.path.join(output_dir, "ex04_cantilever.vtkhdf")
exporter = TransientVTKHDFExporter(filepath, mesh)
exporter.add_step(0.0, solver.u)

dt = DT_INITIAL
step_count = cutbacks = 0

print(f"\n{'='*70}")
print(f" CANTILEVER BENDING TEST  |  {LENGTH}mm × {total_height:.4f}mm  |  Tip UY={TIP_UY}mm")
print(f"{'='*70}")

while solver.time < T_TOTAL - 1e-12:
    dt = min(dt, T_TOTAL - solver.time)
    t_end = solver.time + dt
    step_count += 1
    print(f"\n--- Step {step_count}   t={solver.time:.4f}->{t_end:.4f}   dt={dt:.3e}")

    saved = solver.save_state()
    n_iter = solver.solve_step(dt)

    if n_iter < 0:
        cutbacks += 1
        if cutbacks > MAX_CUTBACKS:
            print(f"  *** FATAL: Max cutbacks. Aborting.")
            break
        dt_new = dt * 0.4
        if dt_new < DT_MIN: break
        print(f"  *** CUTBACK {cutbacks}: {dt:.3e} -> {dt_new:.3e}")
        solver.restore_state(saved)
        dt = dt_new; step_count -= 1; continue

    cutbacks = 0
    exporter.add_step(solver.time, solver.u)

    tip_uy   = solver.u[tip_idx * 2 + 1]
    mid_uy   = solver.u[mid_right_idx * 2 + 1]
    top_uy   = solver.u[top_tip_idx * 2 + 1]

    print(f"  Bottom tip UY = {tip_uy:+.6f}   Mid-right UY = {mid_uy:+.6f}   Top tip UY = {top_uy:+.6f}")

    if abs(top_uy) > 0.01 * abs(TIP_UY):
        print(f"  *** TRANSMISSION OK  ({100*abs(top_uy/TIP_UY):.1f}% of tip)")
    elif abs(top_uy) > 1e-8:
        print(f"  ** PARTIAL  ({100*abs(top_uy/TIP_UY):.2f}%) - possible locking")
    else:
        print(f"  ** ZERO TRANSMISSION  ({top_uy:.2e}) - ELEMENT BUG!")

    if n_iter <= 3:  dt = min(dt * 2.0, DT_MAX)
    elif n_iter <= 8: dt = min(dt * 1.5, DT_MAX)
    else:             dt = max(dt * 0.7, DT_MIN)

exporter.close()
print(f"\n{'='*70}")
print(f" {step_count} steps, {cutbacks} cutbacks  |  Output: {filepath}")
print(f"{'='*70}")

if abs(top_uy) > 0.01 * abs(TIP_UY):
    print("VERDICT: Elements TRANSMIT deformation through thickness")
elif abs(top_uy) > 1e-8:
    print(f"VERDICT: Partial ({100*abs(top_uy/TIP_UY):.2f}%) - investigate locking")
else:
    print("VERDICT: ZERO through-thickness stiffness - ELEMENT BUG CONFIRMED")

