"""Quantify free-span sag vs ideal elastica arc, from a saved result."""
import sys
import numpy as np
from dispsolver.postprocess import load_result

path = sys.argv[1] if len(sys.argv) > 1 else "examples/ex12_result.pkl"
free_half = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0   # untied half-width
pivot_x = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4

r = load_result(path)
theta = np.asarray(r.scalars.get("theta_deg"))
node_ids = np.asarray(r.node_ids)
coords0 = np.asarray(r.points)
mask_disp = node_ids < 10000

# lowest surviving node per x column (free span has a voided bottom layer)
by_x = {}
for i in np.where(mask_disp)[0]:
    x0 = round(float(coords0[i, 0]), 4)
    y0 = float(coords0[i, 1])
    if x0 not in by_x or y0 < by_x[x0][1]:
        by_x[x0] = (i, y0)

pts = sorted((x0, by_x[x0][0]) for x0 in by_x if abs(x0) < free_half)
print(f"result: {path}  steps={r.n_steps}  theta_max={theta.max():.2f}deg  "
      f"free half-width={free_half}mm  n_cols={len(pts)}")

for step in sorted(set([r.n_steps // 3, (2 * r.n_steps) // 3, r.n_steps - 1])):
    u = r.displacement(step=step)
    xs = np.array([p[0] for p in pts])
    uy = np.array([u[p[1]][1] for p in pts])
    ux = np.array([u[p[1]][0] for p in pts])
    center = uy[np.argmin(np.abs(xs))]
    edge = 0.5 * (uy[0] + uy[-1])
    sag = edge - center
    th = np.deg2rad(theta[step])
    L = 2.0 * free_half
    kappa_ideal = 2.0 * th / L          # ends each rotate by th -> circular arc
    sag_ideal = kappa_ideal * L ** 2 / 8.0
    print(f"  step {step:3d}  theta={theta[step]:6.2f}deg | uy_edge={edge:8.4f} "
          f"uy_center={center:8.4f} | sag={sag:8.4f} mm  ideal_arc={sag_ideal:7.4f} mm "
          f"-> {100.0 * sag / sag_ideal if sag_ideal else 0:6.1f}% of ideal | "
          f"max|ux|={np.max(np.abs(ux)):7.4f}")
