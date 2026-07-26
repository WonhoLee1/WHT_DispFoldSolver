"""
interlayer.py
=============
Track the book-page / staircase offset between the layers of a laminated
display stack as it folds, and report it as a table.

What this measures and why
--------------------------
In the 14-layer PET-PSA stack the soft PSA rows exist so the layers can
**shear relative to each other** while folding. That interlayer shear
shows up as a staircase offset between layers, largest near the hinge
edge and still visible at the free tips. Whether it actually develops --
and whether the soft rows, not the stiff ones, are carrying it -- is the
check that tells you the layup is doing its job.

At a through-thickness column, in the *deformed* configuration:

    t_hat  : unit tangent of the display surface at that column,
             pointing outboard (away from the hinge)
    n_hat  : unit normal (t_hat rotated 90 deg)
    slip_j : (P_j - P_0) . t_hat    <- staircase offset of layer j
                                       relative to the bottom surface
    thk_j  : (P_j - P_0) . n_hat    <- remaining through-thickness

`slip_j` is identically zero for rigid-body motion, so a non-zero value
is genuine deformation.

Caveat -- shear vs. bending tilt
--------------------------------
`slip_j` cannot by itself distinguish interlayer shear from the column
merely tilting with the bend (which also produces a tangential offset).
The discriminator is **how the slip splits between materials**: PET and
PSA rows have equal thickness here, so uniform deformation would give a
~50/50 split. A strongly asymmetric split (PSA carrying ~99%) is genuine
interlayer shear; a near-50/50 split with the same per-row angle
everywhere is bending tilt. `format_table` reports the split so this can
be judged rather than assumed. Probe the tips and hinge edges, not the
hinge centre, when you want the shear signal.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

UM = 1.0e3  # mm -> um


def _display_columns(mesh, max_node_id: Optional[int]) -> Dict[float, List[int]]:
    """{x: [node ids, bottom -> top]} for the laminate block."""
    cols: Dict[float, List] = {}
    for nid, n in mesh.nodes.items():
        if max_node_id is not None and nid >= max_node_id:
            continue
        cols.setdefault(round(float(n.x), 6), []).append((float(n.y), nid))
    return {x: [nid for _, nid in sorted(v)] for x, v in sorted(cols.items())}


class LayerSlipTracker:
    """Record per-layer tip slip at every step, then print it as a table.

    Column geometry is resolved once at construction, so `record()` is a
    handful of array lookups -- cheap enough to call every increment of a
    long solve.
    """

    def __init__(
        self,
        mesh,
        probe_x: Sequence[float] = (-40.0, 40.0),
        layer_materials: Optional[Sequence[str]] = None,
        max_node_id: Optional[int] = 10000,
    ):
        """
        Parameters
        ----------
        probe_x
            x positions to probe. Defaults to the two free tips. The
            nearest existing column is used if there is no exact match.
        layer_materials
            Material name per layer, bottom -> top (len = n_nodes-1 in a
            column). Used only for the PSA/PET split diagnostic.
        max_node_id
            Nodes with id >= this are excluded (the rigid plate parts in
            the ex12/ex13 models start at 10000). `None` keeps everything.
        """
        self._mesh = mesh
        self._nid_to_idx = mesh.node_id_to_index()
        cols = _display_columns(mesh, max_node_id)
        if not cols:
            raise ValueError("no laminate columns found -- check max_node_id")
        xs = list(cols.keys())

        self.probes: List[Dict] = []
        for xt in probe_x:
            x = min(xs, key=lambda v: abs(v - xt))
            i = xs.index(x)
            # Neighbour column on the *inboard* side, so the tangent
            # (tip - neighbour) points outboard at both ends and the two
            # tips report the same sign for the same physical motion.
            j = i + 1 if x < 0 else i - 1
            j = min(max(j, 0), len(xs) - 1)
            if j == i:
                j = i + 1 if i + 1 < len(xs) else i - 1
            self.probes.append({
                "x": x,
                "col": [self._nid_to_idx[n] for n in cols[x]],
                "ref": [self._nid_to_idx[n] for n in cols[xs[j]]],
            })

        self.n_layers = len(self.probes[0]["col"]) - 1
        if layer_materials is not None and len(layer_materials) != self.n_layers:
            raise ValueError(
                f"layer_materials has {len(layer_materials)} entries but the "
                f"column has {self.n_layers} layers"
            )
        self.layer_materials = list(layer_materials) if layer_materials else None

        pts = np.asarray(mesh.nodes_array(), dtype=np.float64)
        self._ref_xy = pts

        self.theta: List[float] = []
        self.time: List[float] = []
        # slip[probe_index] -> list over steps of (n_layers+1,) arrays
        self.slip: List[List[np.ndarray]] = [[] for _ in self.probes]

    # ------------------------------------------------------------------

    def _column_slip(self, xy: np.ndarray, probe: Dict) -> np.ndarray:
        P = xy[probe["col"]]
        t_vec = P[0] - xy[probe["ref"]][0]
        nrm = np.linalg.norm(t_vec)
        if nrm < 1e-12:
            return np.zeros(len(probe["col"]))
        t_hat = t_vec / nrm
        return (P - P[0]) @ t_hat

    def record(self, u: np.ndarray, theta_deg: float, time: Optional[float] = None) -> None:
        """Capture one converged increment."""
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        n = self._ref_xy.shape[0]
        xy = self._ref_xy + u[: 2 * n].reshape(n, 2)
        self.theta.append(float(theta_deg))
        self.time.append(float(time) if time is not None else float("nan"))
        for k, probe in enumerate(self.probes):
            self.slip[k].append(self._column_slip(xy, probe))

    @property
    def n_steps(self) -> int:
        return len(self.theta)

    # ------------------------------------------------------------------

    def history(self) -> Dict[str, np.ndarray]:
        """Recorded history as arrays, for saving or further analysis."""
        out: Dict[str, np.ndarray] = {
            "theta_deg": np.asarray(self.theta, dtype=np.float64),
            "time": np.asarray(self.time, dtype=np.float64),
        }
        for k, probe in enumerate(self.probes):
            key = f"slip_x{probe['x']:+.1f}".replace(".", "p")
            out[key] = (np.vstack(self.slip[k]) if self.slip[k]
                        else np.empty((0, self.n_layers + 1)))
        return out

    def format_table(self, probe_index: Optional[int] = None,
                     max_rows: int = 25) -> str:
        """Human-readable table of per-layer tip slip vs fold angle.

        Values are cumulative slip [um] of each layer boundary relative to
        the bottom surface, so the last column is the total staircase.
        Rows are thinned to at most `max_rows` (first and last always
        kept) so a 100-step run stays readable.
        """
        if self.n_steps == 0:
            return "  (no steps recorded -- interlayer slip table empty)"

        idx = list(range(len(self.probes))) if probe_index is None else [probe_index]
        lines: List[str] = []

        for k in idx:
            probe = self.probes[k]
            S = np.vstack(self.slip[k]) * UM          # (n_steps, n_layers+1)
            theta = np.asarray(self.theta)

            rows = (np.arange(self.n_steps) if self.n_steps <= max_rows
                    else np.unique(np.linspace(0, self.n_steps - 1, max_rows).astype(int)))

            head = f" theta[deg]" + "".join(f" {f'L{j}':>8}" for j in range(1, self.n_layers + 1))
            lines.append("")
            lines.append("=" * len(head))
            lines.append(f" Interlayer slip at x = {probe['x']:+.2f} mm   "
                         f"[um, cumulative from bottom surface]")
            if self.n_steps > max_rows:
                lines.append(f" ({self.n_steps} steps recorded, showing {len(rows)})")
            lines.append("=" * len(head))
            lines.append(head)
            lines.append("-" * len(head))
            for r in rows:
                lines.append(f" {theta[r]:>10.2f}" +
                             "".join(f" {S[r, j]:>8.2f}" for j in range(1, self.n_layers + 1)))
            lines.append("-" * len(head))

            final = S[-1]
            lines.append(f" final total staircase : {final[-1]:+.2f} um "
                         f"at theta = {theta[-1]:.2f} deg")

            if self.layer_materials:
                steps_ = np.diff(final)
                by_mat: Dict[str, float] = {}
                for m, ds in zip(self.layer_materials, steps_):
                    by_mat[m] = by_mat.get(m, 0.0) + float(ds)
                den = sum(abs(v) for v in by_mat.values())
                parts = [f"{m} {v:+.2f} um ({100 * abs(v) / den if den else 0:.1f}%)"
                         for m, v in sorted(by_mat.items())]
                lines.append(" carried by            : " + ",  ".join(parts))
                lines.append(" (equal-thickness rows -> a ~50/50 split means bending "
                             "tilt, not interlayer shear)")

        return "\n".join(lines)
