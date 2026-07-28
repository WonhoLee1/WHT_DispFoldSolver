See `AGENTS.md` in this same directory — it is the canonical project rule
file (target problem, solver theory, element JIT strategy, and numerical pitfalls
already solved) shared across Claude Code, Gemini CLI, and OpenCode. Read it before
making changes here.

Element JIT Strategy: Use `--elem_jit jax` for R&D/prototyping (AutoDiff), and switch to `--elem_jit numba` for production run speed.
