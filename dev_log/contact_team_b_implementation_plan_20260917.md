# Contact Team B Implementation Plan: EMCS Conserving Contact & PDASS (2026-09-17)

## 1. Objectives & Theoretical Specifications

### 1.1 EMCS Conserving Dynamic Contact Potential (`dispsolver/solver3d/emcs_contact.py`)
- **Discrete Gradient Formulation (Gonzalez 1996, Laursen & De Lorenzis 2003)**:
  $$\Psi_c(g) = \frac{1}{2} k_c \langle -g \rangle_-^2$$
  where $g$ is the normal contact penetration:
  $$g_n = -(\mathbf{x}_s - \mathbf{x}_m) \cdot \mathbf{n}$$
- **Midpoint Resisting Force**:
  $$\mathbf{f}_{c, n+1/2}^{\text{EMCS}} = \begin{cases} 
  \frac{\Psi_c(g_{n+1}) - \Psi_c(g_n)}{g_{n+1} - g_n} \mathbf{n}_{n+1/2} & \text{if } |g_{n+1} - g_n| > 10^{-12} \\
  \Psi_c'\left(\frac{g_n + g_{n+1}}{2}\right) \mathbf{n}_{n+1/2} & \text{otherwise}
  \end{cases}$$
- **Exact Energy Balance**:
  $$\Delta \mathcal{H} = \Delta E_{\text{kin}} + \Delta E_{\text{strain}} + \Delta \Psi_c \equiv 0$$
- **Symmetric Consistent Tangent**:
  $$\mathbf{K}_{\text{tan}}^{\text{EMCS}} = \frac{\partial \mathbf{f}_{c, n+1/2}}{\partial \mathbf{u}_{n+1}}$$

### 1.2 PDASS Semi-Smooth Newton & Hysteresis Chatter Band
- **Complementarity Function**:
  $$C(p_n, g_n) = p_n - \max(0, p_n - c_n g_n) = 0$$
- **Chatter-Free Hysteresis Band**:
  - Activation threshold: $g_n < -10^{-6}\text{ mm}$
  - Deactivation threshold: $g_n > +10^{-5}\text{ mm}$

### 1.3 Verification Suite (`tests/test_emcs_contact3d.py`)
1. **Gonzalez Discrete Derivative Consistency**: Verifies $\Delta \Psi_c = \mathbf{f}_{c, n+1/2} \cdot \Delta \mathbf{u}$.
2. **Tangent Matrix Numerical Derivative Consistency**: Machine precision agreement with finite differences.
3. **High-Velocity 3D Elastic Block Impact**:
   - Two C3D8 blocks colliding dynamically under EMCS midpoint integration.
   - Total energy conservation: drift $< 10^{-10}\text{ J}$ across collision cycle.
