@AGENTS.md

See `AGENTS.md` in this same directory — it is the canonical project rule
file (target problem, solver theory, and numerical pitfalls already
solved) shared across Claude Code, Gemini CLI, and OpenCode. Read it before
making changes here.

## Agent skills

### Issue tracker

GitHub Issues on WonhoLee1/WHT_DispFoldSolver via `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical labels as-is. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.

## FrontISTR Display Bending Rule (Mandatory Conforming Hybrid 362+361UP)
- **Stiff Layers (PET/GLASS)**: Always use 20-node Hex (`362`) to eliminate 93.3% shear locking.
- **Soft Adhesive Layers (PSA)**: Always use 1st-order u-P Hex (`361 UP`) with `!HYPERELASTIC, TYPE=NEOHOOKE`.
- **Conforming Subdivision**: PSA is subdivided 2x in length (X) and width (Z) so that PET mid-side nodes match PSA corner nodes 1:1. Zero hanging nodes, zero TIE penalty contacts.
- **Plate Driving**: Rotate support plates via `!BOUNDARY, ROT_CENTER=PIVOT_L/R` (never rigidly clamp the whole cross-section).
- See canonical details in `GEMINI.md` and `AGENTS.md`.
