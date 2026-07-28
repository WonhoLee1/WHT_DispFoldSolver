## Performance Report (Opt-In)

**Benchmark**: Speed Benchmark (Cantilever)

**Note**: This data is informational only — no PASS/FAIL gating.

### Mesh: (10×4) — 27 elements

| Backend | Mean (s) | Median (s) | Std (s) | Min (s) | Max (s) | N Iter |
|---------|----------|-------------|---------|---------|---------|--------|
| jax | 3.089398e-02 | 3.072130e-02 | 2.960853e-03 | 2.794900e-02 | 3.469270e-02 | 2 |
| numpy_sequential | 4.556215e-01 | 4.355640e-01 | 5.219917e-02 | 4.230939e-01 | 5.484479e-01 | 2 |

**Speedup (JAX over NumPy sequential)**: 14.75×

### Mesh: (40×4) — 117 elements

| Backend | Mean (s) | Median (s) | Std (s) | Min (s) | Max (s) | N Iter |
|---------|----------|-------------|---------|---------|---------|--------|
| jax | 5.110776e-02 | 5.184780e-02 | 2.790759e-03 | 4.812870e-02 | 5.444790e-02 | 2 |
| numpy_sequential | 1.554548e+00 | 1.563042e+00 | 3.588316e-02 | 1.518838e+00 | 1.605683e+00 | 2 |

**Speedup (JAX over NumPy sequential)**: 30.42×

### Mesh: (80×8) — 553 elements

| Backend | Mean (s) | Median (s) | Std (s) | Min (s) | Max (s) | N Iter |
|---------|----------|-------------|---------|---------|---------|--------|
| jax | 1.492515e-01 | 1.490674e-01 | 7.586901e-03 | 1.380881e-01 | 1.588736e-01 | 2 |
| numpy_sequential | 6.011903e+00 | 5.997819e+00 | 3.397804e-02 | 5.981922e+00 | 6.065421e+00 | 2 |

**Speedup (JAX over NumPy sequential)**: 40.28×
