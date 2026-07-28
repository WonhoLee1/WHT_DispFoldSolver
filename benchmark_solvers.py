"""Benchmark PARDISO vs scipy.spsolve on KKT-like system."""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from pypardiso import spsolve as pardiso_spsolve
    HAS_PARDISO = True
except ImportError:
    HAS_PARDISO = False


def make_kkt_system(n, penalty=1e5):
    """Build a KKT-like block system mimicking the solver's structure."""
    np.random.seed(42)
    A = sp.random(n, n, density=0.01, format='csr')
    A = A + A.T + n * sp.eye(n)
    A = A.tocsr()

    C = sp.random(n // 4, n, density=0.05, format='csr').tocsr()
    zero_block = sp.csr_matrix((n // 4, n // 4))
    KKT = sp.bmat([[A, C.T], [C, zero_block]], format='csr')
    b = np.random.randn(n + n // 4)
    return KKT, b


def bench(solver_fn, A, b, label, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        x = solver_fn(A, b)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    best = min(times)
    avg = sum(times) / len(times)
    res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    print(f"  {label:20s}  best={best*1000:7.1f}ms  avg={avg*1000:7.1f}ms  rel_res={res:.2e}")
    return best, res


print("=" * 70)
print("KKT-like system benchmark: PARDISO vs scipy.spsolve")
print("=" * 70)

for n in (500, 2000, 4000):
    print(f"\n--- n = {n} (KKT size {n + n//4}) ---")
    A, b = make_kkt_system(n)

    if HAS_PARDISO:
        bench(pardiso_spsolve, A, b, "PARDISO (pypardiso)")
    bench(spla.spsolve, A, b, "scipy.spsolve (SuperLU)")

print()
print("=" * 70)
print("Iterative solvers (scipy) — relevant for large systems")
print("=" * 70)


def bench_iterative(solver_fn, A, b, label, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            x, info = solver_fn(A, b, tol=1e-6, maxiter=1000)
            elapsed = time.perf_counter() - t0
            res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
            times.append((elapsed, res, info))
        except Exception as e:
            print(f"  {label:20s}  FAILED: {e}")
            return None
    best_time, best_res, best_info = min(times, key=lambda t: t[0])
    print(f"  {label:20s}  best={best_time*1000:7.1f}ms  rel_res={best_res:.2e}  iters={best_info}")
    return best_time


for n in (500, 2000, 4000):
    print(f"\n--- n = {n} ---")
    A, b = make_kkt_system(n)
    M = sp.diags(1.0 / np.abs(A.diagonal()) + 1e-12)
    bench_iterative(lambda A, b, **kw: spla.cg(A, b, M=M, **kw), A, b, "scipy.cg (jacobi)")
    bench_iterative(lambda A, b, **kw: spla.gmres(A, b, M=M, **kw), A, b, "scipy.gmres (jacobi)")
