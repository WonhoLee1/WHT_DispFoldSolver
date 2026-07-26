# Interlayer shear (PSA staircase) verification — 2026-07-26

## Question

The 14-layer PET-PSA stack puts low-modulus PSA (Arruda-Boyce,
mu = 0.168 MPa) between stiff PET (E = 4000 MPa) rows **specifically so
the layers can shear relative to each other during folding**. At full
fold that should produce a book-page / staircase offset between layers.
The user could not see any staircase in `ex12_final_folding_shape.png`
and asked for this to be confirmed.

## Answer: the staircase is real and behaves as designed

New diagnostic: `examples/check_interlayer_shear.py`. It scans every
display x-column and, in the deformed configuration, decomposes each
through-thickness column into

    t_hat  : deformed tangent of the display's bottom surface at that x
    slip_j : (P_j - P_0) . t_hat     <- staircase offset of layer j
    thk_j  : (P_j - P_0) . n_hat

`slip_j` is identically zero for rigid-body motion, so non-zero values
are genuine deformation. Output PNG:
`examples/ex12_interlayer_slip_profile.png` (slip-vs-x profile + zoomed
deformed column). Displacement field is cached to
`examples/ex12_final_u.npy`, so `--cached` re-analyzes in seconds
instead of re-solving (~295s).

### Measured at full closure (90deg/side)

| location | total staircase | carried by PSA | carried by PET | max PSA shear angle |
|---|---|---|---|---|
| free tip, x = +-40      | **81 um**  | 99.5% | 0.5% | 26.7 deg |
| hinge edge, x = -10     | **203 um** | 97.3% | 2.7% | 53.6 deg |
| hinge edge, x = +10     | **190 um** | 99.4% | 0.6% | 52.1 deg |
| peak, x = +6            | **273 um** | -- | -- | -- |

**The decisive number is the PSA/PET split.** PET and PSA rows are the
same thickness (0.5/14 = 35.7 um each), 7 of each. If the stack sheared
uniformly, each material would carry ~50% of the slip. Instead PSA
carries 97-99.5% -- roughly 200x concentration into the soft rows. That
is exactly the intended design function. The zoomed column plot shows it
directly: blue (PET) segments stay nearly horizontal, red (PSA) segments
are steeply slanted.

### Why it is invisible in the full-model PNG

81 um at the tip = 0.081 mm across an 80 mm-wide model = **0.1% of the
model width** -- thinner than the plotted line weight. Not seeing it at
full-model zoom is expected and is not evidence of a modeling defect.
Use `check_interlayer_shear.py`'s zoomed view, not the overview PNG.

### Caveat on reading the metric

At the hinge centre (x = 0) the metric reports 60 um "slip" split
44.7% PSA / 55.3% PET with an identical 7.56 deg angle in *every* row.
That is not interlayer shear -- it is the column tilting with the bend,
which `slip_j` cannot distinguish from shear when the column is no
longer perpendicular to the reference tangent. **Read this metric as
interlayer shear only where the PSA/PET split is strongly asymmetric**
(the tips and hinge edges); a near-50/50 split with uniform per-row
angle means bending tilt, not shear.

### Shear-lag estimate was wrong

Pre-analysis predicted the tip staircase would be ~0.4% of the hinge
value, using single-layer shear lag
`l = sqrt(E_PET*t_PET*t_PSA/G_PSA) ~= 5.5 mm` over the 30 mm plate span.
Measured decay is 203 -> 81 um, i.e. **40% survives**, implying an
effective lag length ~33 mm. The single-layer formula does not transfer
to a 7-pair stack where all PET rows share the load. Do not use the
5.5 mm figure.

## Also done in this pass

Ran the mandatory suite from `verification/RULES.md` (required after any
change to `dispsolver/solver`, `/material`, `/constraint`, `/mesh` --
this session had modified `solver/dynamic.py`,
`material/viscoelastic.py`, `io/abaqus_parser.py`, `io/model_builder.py`
without running it). Result: **9/9 benchmarks PASS** across all
backends including `jax_q4_simo_fs` and `jax_q4_visco_fs`
(`verification/results/verification_report.md`).
