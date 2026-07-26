"""
animate.py
==========
Build animations and still frames from a saved `Result` -- no solver, no
re-run.

Everything here takes a `Result` (see `result_io.py`), so a five-minute
fold can be re-plotted, re-coloured and re-animated in seconds from the
`.pkl` written at the end of the run.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .result_io import Result

# Distinct, colourblind-safe-ish hues cycled per pid. Deliberately not a
# rainbow: the point is to tell PET rows from PSA rows at a glance.
_PID_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2",
               "#CCB974", "#64B5CD", "#937860", "#DA8BC3"]


def _quad_polys(result: Result, step: int):
    """(n_cells, 4, 2) deformed quad corners at `step`."""
    xy = result.deformed_points(step)[:, :2]
    conn, offs = result.connectivity, result.offsets
    return [xy[conn[offs[i]: offs[i + 1]]] for i in range(len(offs) - 1)]


def _pid_facecolors(result: Result):
    pids = np.asarray(result.element_pid)
    uniq = sorted(set(int(p) for p in pids))
    lut = {p: _PID_COLORS[i % len(_PID_COLORS)] for i, p in enumerate(uniq)}
    return [lut[int(p)] for p in pids], lut


def plot_step(result: Result, step: int = -1, ax=None, color_by_pid: bool = True,
              title: Optional[str] = None, lw: float = 0.2):
    """Draw one deformed frame onto a matplotlib axis."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    polys = _quad_polys(result, step)
    if color_by_pid:
        fc, lut = _pid_facecolors(result)
    else:
        fc, lut = "#4C72B0", {}
    ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors="k",
                                     linewidths=lw))
    pts = result.deformed_points(step)[:, :2]
    pad = 0.05 * max(np.ptp(pts[:, 0]), 1e-9)
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    s = result.n_steps + step if step < 0 else step
    if title is None:
        title = f"t = {result.times[s]:.4f}"
        if "theta_deg" in result.scalars:
            title += f"   theta = {result.scalars['theta_deg'][s]:.2f} deg"
    ax.set_title(title)
    return ax


def save_frame(result: Result, path: str, step: int = -1, dpi: int = 200, **kw) -> str:
    """Render a single step to an image file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_step(result, step=step, ax=ax, **kw)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path)


def animate(result: Result, path: str, fps: int = 12, dpi: int = 150,
            stride: int = 1, color_by_pid: bool = True) -> str:
    """Write a GIF (or MP4, by extension) of the whole recorded history.

    Axis limits are fixed to the union over all frames so the model does
    not appear to drift or rescale as it folds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as manim
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    steps = list(range(0, result.n_steps, max(1, stride)))
    if steps[-1] != result.n_steps - 1:
        steps.append(result.n_steps - 1)

    all_pts = np.concatenate(
        [result.deformed_points(s)[:, :2] for s in steps], axis=0
    )
    pad = 0.05 * max(np.ptp(all_pts[:, 0]), 1e-9)
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    fc = _pid_facecolors(result)[0] if color_by_pid else "#4C72B0"

    fig, ax = plt.subplots(figsize=(11, 5))
    coll = PolyCollection(_quad_polys(result, steps[0]), facecolors=fc,
                          edgecolors="k", linewidths=0.2)
    ax.add_collection(coll)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    title = ax.set_title("")

    theta = result.scalars.get("theta_deg")

    def _draw(i):
        s = steps[i]
        coll.set_verts(_quad_polys(result, s))
        txt = f"step {s + 1}/{result.n_steps}   t = {result.times[s]:.4f}"
        if theta is not None:
            txt += f"   theta = {theta[s]:.2f} deg"
        title.set_text(txt)
        return coll, title

    anim = manim.FuncAnimation(fig, _draw, frames=len(steps), blit=False)
    writer = "pillow" if str(path).lower().endswith(".gif") else "ffmpeg"
    anim.save(str(path), writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return str(path)


def plot_slip_history(result: Result, path: str, probe_key: Optional[str] = None,
                      dpi: int = 200) -> str:
    """Plot per-layer interlayer slip vs fold angle from a saved run.

    Uses the `slip__*` arrays the solve loop attaches via
    `LayerSlipTracker.history()`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in result.array_names
            if k.startswith("slip__slip_x")]
    if not keys:
        raise KeyError("this result carries no slip history "
                       f"(arrays: {result.array_names})")
    if probe_key is not None:
        keys = [k for k in keys if probe_key in k] or keys

    theta = result.arrays.get("slip__theta_deg")
    layer_materials = result.meta.get("layer_materials") or []

    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 5),
                             squeeze=False)
    for ax, key in zip(axes[0], keys):
        S = np.asarray(result.arrays[key]) * 1e3      # um
        n_layers = S.shape[1] - 1
        # A stiff row barely shears, so the boundary above a PET row sits
        # almost exactly on the boundary above the PSA row beneath it --
        # the 14 curves collapse into ~7 coincident pairs. Draw the PSA
        # ones last and solid so the pairing stays legible instead of the
        # red curves hiding under the blue.
        for want_psa in (False, True):
            for j in range(1, n_layers + 1):
                mat = (layer_materials[j - 1] if j - 1 < len(layer_materials) else "")
                is_psa = mat.upper() == "PSA"
                if is_psa != want_psa:
                    continue
                ax.plot(theta, S[:, j],
                        color="#C44E52" if is_psa else "#4C72B0",
                        lw=1.6 if is_psa else 1.0,
                        ls="-" if is_psa else "--",
                        zorder=3 if is_psa else 2, alpha=0.9)
        ax.set_xlabel("fold angle [deg]")
        ax.set_ylabel("cumulative slip from bottom [um]")
        ax.set_title(key.replace("slip__slip_", "probe "))
        ax.grid(alpha=0.3)
    fig.suptitle("Interlayer slip growth vs fold angle\n"
                 "solid red = boundary above a PSA row, dashed blue = above PET "
                 "(pairs nearly coincide: the PET rows contribute almost no slip)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path)
