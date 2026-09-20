# 🎯 Task Instruction: S3 P2 (Mullins Damage & Bergström-Boyce) & Abaqus Benchmark Suite

**Target Models**: `opencode-go/qwen-2.5-coder-32b`, `opencode-go/muse-spark-1.3-contributor`, `opencode-go/deepseek-r1`  
**Execution Context**: D:\PythonCodeStudy\WHT_DispFoldSolver  
**Standard**: Abaqus 2026 Material Theory Guide & Anti-Mocking Protocol  

---

## 1. Task Objective
Implement production-grade, Abaqus-compatible constitutive models for flexible display interlayer adhesive (PSA) and polymer stack:
1. **`dispsolver/material/mullins_damage.py`**:
   - Continuous damage mechanics for hyperelastic materials (Ogden & Roxburgh 1999).
   - Energy function: $W(\mathbf{F}, \eta) = \eta W_{base}(\mathbf{F}) + \phi(\eta)$.
   - Damage variable $\eta = 1 - \frac{1}{r} \text{erf}\left(\frac{U_{max} - U}{m + \beta U_{max}}\right)$.
   - SDV tracking: maximum dev strain energy $U_{max}$.
2. **`dispsolver/material/bergstrom_boyce.py`**:
   - 2-Network rate-dependent viscoplasticity for elastomers (Bergström & Boyce 1998).
   - Stress split: $\mathbf{T} = \mathbf{T}_A + \mathbf{T}_B$.
   - Network A: Arruda-Boyce 8-chain hyperelastic equilibrium stress.
   - Network B: Inelastic flow rule $\dot{\gamma}_B = \dot{\gamma}_0 (\lambda_{chain}^B - 1)^C (\tau_B / \hat{\tau})^m$.
3. **`tests/test_mullins_and_bergstrom_boyce.py`**:
   - Strict unit tests verifying virgin loading, unloading damage softening, reloading behavior, and rate dependence.
4. **`benchmark_element/benchmark_abaqus_material_inp.py`**:
   - Direct verification against Abaqus Verification Manual analytical solutions:
     - `temp_elastic_ef1` (E(T) tension/shear, error < 1.0%)
     - `temp_plastic_mises` (sigma_y(T) plasticity, error < 1.0%)
     - `thermal_expansion_simple` (expansion & reaction force, error < 1.0%)

---

## 2. Hard Constraints (Anti-Mocking)
- NO random values (`np.random.rand`), NO dummy returns, NO artificial scaling constants.
- Exact formulation lines according to official literature.
- Numba `@njit(fastmath=True, cache=True)` pure functions.
- Run tests with `pytest tests/test_mullins_and_bergstrom_boyce.py -v`.
