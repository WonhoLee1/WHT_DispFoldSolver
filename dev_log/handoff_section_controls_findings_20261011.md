# Hand-off: findings from the stopped section-controls agent (2026-10-11)

An agent implementing Abaqus *SECTION CONTROLS / hourglass control for the NEW 2D/3D libraries
was stopped (another session owns that area). Its uncommitted work is in worktree
`C:/Users/GOODMAN/.claude/jobs/284d3b9f/tmp/wt_hg3d` (branch `wip/section-controls-3d`, base
4f4c044) -- unreviewed, use as reference only. Findings, as reported (not yet independently
re-measured):

1. **`element2d/cpe4i_numba.py` is not frame-covariant.** K(RX) differs from T K T^T by ~2e-1
   at 30 deg on a distorted quad -- the same class as AGENTS 4.14 F6 (EAS pull-back). A reference
   EAS implementation was covariant to 2e-16 on the same quad. 3D `C3D8I` matched the reference to
   1e-16 on box and distorted hexes (rotation check not run on it). -> needs a C6/C7-style rotated
   check before CPE4I (new library) is trusted at large rotation.
2. **A 9-mode EAS projection used as ENHANCED (3D) locks near incompressibility**: on a 4:1:1
   box the Gamma_4 hourglass stiffness is 139 mu at nu=0.4999 vs 0.067 mu expected. Adding three
   bilinear volumetric modes fixes it while the bending modes still match C3D8I exactly on a
   parallelepiped.
3. **Pre-existing `C3D8R` kernel shear Voigt order is (yz, xz, xy)** while the material library
   uses (xy, yz, xz): shear stress is mismatched for hyperelastic materials. The old new-library
   `CPE4R` ignored the material type and ran linear elastic.
4. **Keywords Guide default** (confirmed): HOURGLASS=STIFFNESS in Standard, except ENHANCED for
   hyperelastic/hyperfoam materials and modified tets/triangles. STIFFNESS r_FG = 0.005 G * s_s;
   how it enters the element force is not published. (The legacy `CPE4R` in
   `dispsolver/element/cpe4r_jax.py` follows the same reading: unset -> STIFFNESS.)
5. In that worktree `test_schema_parse::test_sections` and
   `test_section_controls::test_c3d8_hybrid_distortion_control_barrier` failed in partial runs;
   not established whether caused by its changes -- check them in the main tree.

## Addendum: state of the stopped worktree (agent's final summary)

- **Not an assumption to keep**: STIFFNESS is NOT always more flexible than ENHANCED. On a 5:1:1
  C3D8R the STIFFNESS beam was STIFFER (tip / Timoshenko 0.40 vs 0.90 for ENHANCED); the two
  bending tests in its `tests/test_hourglass_controls.py` that assumed the opposite still fail.
- Files in wt_hg3d: new `dispsolver/element3d/hourglass_numba.py`, `tests/test_hourglass_controls.py`;
  rewritten `element3d/c3d8r_numba.py`, `element3d/c3d8r_jax.py`, `element2d/cpe4r_numba.py`;
  edited `model/section.py`, `model/model.py`, `model/assembly.py`, `element_catalog.py`,
  `io/keywords/sections.py`, `solver2d/dynamic2d.py`, `tests/test_3d_c3d8r.py`, two
  `benchmark_element/benchmark_nlgeo_*` scripts.
- Tests when stopped: hourglass tests 58 pass / 2 fail (on an earlier code version; the 3D
  ENHANCED projection was changed afterwards and NOT re-run); schema/keyword/defaults/section
  tests 57 pass / 2 fail (cause unknown). Not run: corpus, test_3d_mechanics_benchmarks,
  test_2d_elements, verification.run_all.
