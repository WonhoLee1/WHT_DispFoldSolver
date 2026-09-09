"""
display_builder.py
==================
Single source of truth for the display panel's mesh topology (graded
x-coordinates, layer row spans, node numbering, element connectivity).

Both model-construction paths consume this:

- `examples/gen_ex12_inp.py`                      -> formats it as Abaqus .inp text
- `examples/ex13_unified_model_io.py::run_build()` -> `mesh.add_node`/`mesh.add_element`

Why this module exists
----------------------
The same node/element generation previously existed **twice,
independently**, and had already drifted structurally: the `.inp` path
grouped elements per physical layer (`layer_row_spans`), the build path
per row (`ROW_PID`). Void regions change node *numbering*, so
implementing them twice would have guaranteed the two paths diverge --
exactly the hazard `dispsolver/fold_model_config.py`'s docstring says the
config module was created to prevent.

Void regions
------------
A physical layer may declare x-intervals where **no elements are
created** (a "tape" layer with cutouts). Two invariants make this safe:

1. Every void edge is forced into the global x-partition
   (`_graded_display_x`), so no element column can straddle a void
   boundary -- each column is entirely inside or entirely outside each
   void, and membership is an exact midpoint test.
2. Only nodes actually referenced by a surviving element are emitted.
   Orphan nodes would survive numerically (they pick up the solver's
   diagonal regularization and resolve to `du = 0`), but the
   kinematically-condensed system is densified at `n_independent²`
   (`dispsolver/solver/dynamic.py`), so every unreferenced DOF costs
   quadratically. Dropping them also makes tie-slave filtering automatic:
   a void in the bottom layer removes those nodes from
   `bottom_surface_nids` with no special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# graded x-coordinates
# ----------------------------------------------------------------------

def _n_pts(length: float, dx: float) -> int:
    """Point count for a `np.linspace` segment of the given length/dx.

    +1 for the linspace endpoint; `max(2,...)` guards a zero-length or
    dx-larger-than-length segment from collapsing to <2 points.
    """
    return max(2, round(length / dx) + 1)


_BP_TOL = 1e-9      # breakpoints closer than this are snapped together


def graded_display_x(config, extra_breakpoints: Sequence[float] = ()) -> np.ndarray:
    """Graded (clustered) x-coordinates for the display mesh.

    Uniform spacing lets the shear/rotation gradient concentrate into a
    single element wherever the plate (exactly rigid, RBE2 condensation)
    meets a compliant region -- worst at the two free-edge tip elements
    (x=+-half_length, no outboard neighbor to redistribute strain into)
    and at the plate/hinge boundary (x=+-hinge_half_gap). Clustering
    columns there spreads that gradient across more, narrower elements,
    keeping det(F) > 0 to a much larger fold angle (AGENTS.md 4.12).

    `extra_breakpoints` forces additional x-values to become segment
    endpoints -- used for void-region edges so no element column can
    straddle a void boundary.

    Breakpoint-driven rather than a fixed segment list
    -------------------------------------------------
    Merging extra coordinates into a pre-built array (e.g. `np.union1d`)
    does not work: `np.linspace` interiors are not bit-identical to a
    requested coordinate, so a void edge lands a few ULPs off an existing
    point and produces a sliver element ~1e-15 wide. A `diff(xs) > 0`
    check *passes* on that, and the degenerate element reaches the solver
    as near-zero `det(J)`.

    Here every breakpoint is instead a `linspace` **endpoint** (exactly
    representable), and near-coincident breakpoints are snapped together
    before any segment is built, so the strictly-increasing property
    holds by construction rather than by floating-point luck.

    Grading is unchanged: `dx` for each interval is still looked up from
    the zone its midpoint falls in.

    Note: an `extra_breakpoint` placed very close to (but outside the
    snap tolerance of) a zone boundary yields one legitimately thin
    element there. That is the caller's requested geometry, not a bug --
    but check `np.min(np.diff(xs))` if a void edge is near a zone edge.
    """
    g = config.grading
    geo = config.geometry
    half_len = geo.display_half_length
    half_gap = geo.hinge_half_gap
    span = g.hinge_span_half_width
    hinge_lo = half_gap - g.hinge_edge_cluster_width  # inner edge of hinge cluster
    # Outer edge of the hinge-edge cluster, ON THE PLATE-BODY SIDE of the
    # tie boundary. Per this class's own docstring the cluster is meant to
    # be SYMMETRIC about hinge_half_gap (+-hinge_edge_cluster_width), but
    # zone_dx() previously jumped straight to the coarse plate_body_dx at
    # ax >= half_gap with no fine mesh at all on the plate-body side --
    # a one-sided cluster, contradicting the docstring and reintroducing
    # exactly the abrupt-grading-jump-at-the-boundary failure mode AGENTS.md
    # 4.12 fixed for the tip elements (here at the plate/hinge boundary
    # instead, and biting much earlier -- ~23deg -- for this teardrop
    # config's tight hinge_half_gap=7.5 geometry than the ~48deg case that
    # motivated 4.12). Fixed: extend hinge_edge_dx symmetrically to
    # hinge_hi on the plate-body side too.
    hinge_hi = half_gap + g.hinge_edge_cluster_width
    assert span <= hinge_lo, "hinge_span_half_width must be <= hinge_half_gap - hinge_edge_cluster_width"

    tip_lo = half_len - g.tip_cluster_width
    assert hinge_hi <= tip_lo, "hinge_half_gap + hinge_edge_cluster_width must be <= display_half_length - tip_cluster_width"

    def zone_dx(x_mid: float) -> float:
        """Grading dx for the zone containing x_mid, keyed on |x|."""
        if getattr(g, 'uniform', False):
            return g.plate_body_dx
        ax = abs(x_mid)
        if ax >= tip_lo:
            return g.tip_dx          # tip cluster
        if ax >= hinge_hi:
            return g.plate_body_dx   # under the rigid plate body, coarse
        if ax >= span:
            return g.hinge_edge_dx   # hinge-edge cluster (symmetric about half_gap)
        return g.hinge_span_dx       # free hinge span

    # Zone boundaries, mirrored about x=0, plus any caller-forced points.
    bps = [-half_len, -tip_lo, -hinge_hi, -half_gap, -hinge_lo, -span,
           span, hinge_lo, half_gap, hinge_hi, tip_lo, half_len]
    bps += [float(b) for b in extra_breakpoints]

    # Clip to the panel, sort, then snap near-coincident points together.
    bps = sorted(min(max(b, -half_len), half_len) for b in bps)
    uniq = [bps[0]]
    for b in bps[1:]:
        if b - uniq[-1] > _BP_TOL:
            uniq.append(b)

    segments = [np.linspace(a, b, _n_pts(b - a, zone_dx(0.5 * (a + b))))
                for a, b in zip(uniq, uniq[1:])]
    segments = [s for s in segments if s[-1] > s[0]]
    parts = [segments[0]] + [s[1:] for s in segments[1:]]
    xs = np.concatenate(parts)
    assert np.all(np.diff(xs) > _BP_TOL), \
        "graded x-coordinates must be strictly increasing (no sliver columns)"
    return xs


# ----------------------------------------------------------------------
# grid description
# ----------------------------------------------------------------------

@dataclass
class DisplayCell:
    """One surviving quad element of the display mesh."""
    eid: int
    layer_idx: int                        # 0-based physical layer
    row: int                              # mesh row j
    col: int                              # mesh column i
    conn: Tuple[int, int, int, int]       # (n1, n2, n3, n4) CCW


@dataclass
class DisplayGrid:
    """Pure topology description of the display panel mesh.

    Carries no solver/`.inp` concepts -- both writers project from this.
    """
    xs: np.ndarray                                    # graded x, strictly increasing
    ys: np.ndarray                                    # row boundaries, bottom -> top
    layer_row_spans: List[Tuple[int, int, str]]       # (r0, r1_exclusive, material_name)
    nodes: List[Tuple[int, float, float]]             # (nid, x, y) in emission order
    node_id_of: Dict[Tuple[int, int], int]            # (j, i) -> nid, referenced nodes only
    cells: List[DisplayCell]                          # in emission order (eid ascending)
    bottom_surface_nids: List[int]                    # j == 0, referenced only

    @property
    def nx(self) -> int:
        return len(self.xs) - 1

    @property
    def ny(self) -> int:
        return len(self.ys) - 1

    @property
    def n_layers(self) -> int:
        return len(self.layer_row_spans)

    def x_of_node(self, nid: int) -> float:
        """x-coordinate of an emitted node.

        Use this instead of positional arithmetic like
        `xs[(nid - 1) % (nx + 1)]` -- that is only valid for a full dense
        grid and returns the wrong x for every node past the first void.
        """
        return self._nid_to_x[nid]

    def cells_of_layer(self, layer_idx: int) -> List[DisplayCell]:
        return [c for c in self.cells if c.layer_idx == layer_idx]

    def __post_init__(self):
        self._nid_to_x = {nid: x for nid, x, _y in self.nodes}


# ----------------------------------------------------------------------
# void helpers
# ----------------------------------------------------------------------

def _normalize_voids(config) -> Dict[int, List[Tuple[float, float]]]:
    """Read + validate `geometry.layer_void_regions`.

    Keyed by **1-based** physical layer index (matching the
    `DISP_LAYER{n:02d}` elset numbering and the Qt viewer's Part/Layer
    list). Returns a 0-based dict for internal use.

    Uses getattr so this module works against a config that predates the
    field.
    """
    raw = getattr(config.geometry, "layer_void_regions", None) or {}
    if not raw:
        return {}

    geo = config.geometry
    custom_layers = getattr(geo, "custom_layers", None)
    if custom_layers:
        n_layers = len(custom_layers)
    else:
        n_layers = geo.n_layer_pairs * len(geo.layer_pattern)
    half_len = geo.display_half_length

    out: Dict[int, List[Tuple[float, float]]] = {}
    for layer_no, intervals in raw.items():
        if not (1 <= layer_no <= n_layers):
            raise ValueError(
                f"layer_void_regions: layer {layer_no} out of range 1..{n_layers}"
            )
        norm: List[Tuple[float, float]] = []
        for x0, x1 in intervals:
            x0, x1 = float(x0), float(x1)
            if x1 <= x0:
                raise ValueError(
                    f"layer_void_regions[{layer_no}]: interval ({x0}, {x1}) is empty "
                    f"or reversed; expected x0 < x1"
                )
            if x0 < -half_len - 1e-9 or x1 > half_len + 1e-9:
                raise ValueError(
                    f"layer_void_regions[{layer_no}]: interval ({x0}, {x1}) falls "
                    f"outside the display [{-half_len}, {half_len}]"
                )
            norm.append((x0, x1))
        norm.sort()
        for (a0, a1), (b0, b1) in zip(norm, norm[1:]):
            if b0 < a1 - 1e-9:
                raise ValueError(
                    f"layer_void_regions[{layer_no}]: intervals ({a0}, {a1}) and "
                    f"({b0}, {b1}) overlap"
                )
        out[layer_no - 1] = norm      # store 0-based
    return out


def _column_in_void(x_left: float, x_right: float,
                    voids: Sequence[Tuple[float, float]], tol: float = 1e-9) -> bool:
    """Is this element column inside a void?

    Exact by midpoint because every void edge is a partition line in
    `xs` (see module docstring), so a column can never straddle a void
    boundary.
    """
    xm = 0.5 * (x_left + x_right)
    return any(v0 - tol <= xm <= v1 + tol for v0, v1 in voids)


# ----------------------------------------------------------------------
# builder
# ----------------------------------------------------------------------

def build_display_grid(config) -> DisplayGrid:
    """Build the display panel's mesh topology from a `FoldModelConfig`.

    Node and element IDs are assigned exactly as the two former inline
    loops did (running counters, row-major within each layer), so a
    void-free config reproduces the previous mesh identically.
    """
    geo = config.geometry
    voids = _normalize_voids(config)

    # --- layer row spans ---
    row_heights: List[float] = []
    layer_row_spans: List[Tuple[int, int, str]] = []
    row_cursor = 0
    custom_layers = getattr(geo, "custom_layers", None)
    if custom_layers:
        layers = custom_layers
    else:
        layers = geo.layer_pattern * geo.n_layer_pairs

    for layer in layers:
        row_heights += [layer.thickness_mm / layer.n_rows] * layer.n_rows
        layer_row_spans.append((row_cursor, row_cursor + layer.n_rows, layer.material_name))
        row_cursor += layer.n_rows

    ys = np.concatenate([[0.0], np.cumsum(row_heights)])

    # --- graded x, with every void edge forced in as a partition line ---
    extra_bps: List[float] = []
    for intervals in voids.values():
        for x0, x1 in intervals:
            extra_bps.extend((x0, x1))
    xs = graded_display_x(config, extra_breakpoints=extra_bps)
    nx = len(xs) - 1

    # --- pass 1: which cells survive, and which nodes they reference ---
    layer_of_row: Dict[int, int] = {}
    for layer_idx, (r0, r1, _mat) in enumerate(layer_row_spans):
        for j in range(r0, r1):
            layer_of_row[j] = layer_idx

    surviving: set = set()          # (j, i) element cells
    referenced: set = set()         # (j, i) nodes
    for j in range(len(ys) - 1):
        layer_voids = voids.get(layer_of_row[j], ())
        for i in range(nx):
            if layer_voids and _column_in_void(xs[i], xs[i + 1], layer_voids):
                continue
            surviving.add((j, i))
            referenced.update({(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)})

    # --- pass 2: number the referenced nodes, row-major ---
    node_id_of: Dict[Tuple[int, int], int] = {}
    nodes: List[Tuple[int, float, float]] = []
    bottom_surface_nids: List[int] = []
    nid = 1
    for j, y_val in enumerate(ys):
        for i, x_val in enumerate(xs):
            if (j, i) not in referenced:
                continue
            node_id_of[(j, i)] = nid
            nodes.append((nid, float(x_val), float(y_val)))
            if j == 0:
                bottom_surface_nids.append(nid)
            nid += 1

    # --- elements, grouped per physical layer (matches the .inp ELSETs) ---
    cells: List[DisplayCell] = []
    eid = 1
    for layer_idx, (r0, r1, _mat) in enumerate(layer_row_spans):
        for j in range(r0, r1):
            for i in range(nx):
                if (j, i) not in surviving:
                    continue
                conn = (
                    node_id_of[(j, i)],
                    node_id_of[(j, i + 1)],
                    node_id_of[(j + 1, i + 1)],
                    node_id_of[(j + 1, i)],
                )
                cells.append(DisplayCell(eid=eid, layer_idx=layer_idx, row=j, col=i, conn=conn))
                eid += 1

    return DisplayGrid(
        xs=xs,
        ys=ys,
        layer_row_spans=layer_row_spans,
        nodes=nodes,
        node_id_of=node_id_of,
        cells=cells,
        bottom_surface_nids=bottom_surface_nids,
    )
