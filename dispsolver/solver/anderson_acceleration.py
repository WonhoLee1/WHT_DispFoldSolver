"""
anderson_acceleration.py
========================
Anderson Acceleration (AA-Newton) for Non-linear Finite Element Solvers.
Implements m-depth residual history subspace optimization to eliminate
Newton chattering, cycling, and solver cutbacks in severe non-linearities.

References:
- Anderson, D. G. (1965). "Iterative Procedures for Nonlinear Integral Equations."
  Journal of the ACM, 12(4), 547-560.
- Walker, H. F., & Ni, P. (2011). "Anderson Acceleration for Fixed-Point
  Iterations." SIAM Journal on Numerical Analysis, 49(4), 1715-1735.
- Fang, H. R., & Saad, Y. (2009). "Two classes of multisecant methods for
  nonlinear acceleration." Numerical Linear Algebra with Applications, 16(3), 197-221.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


class AndersonAccelerator:
    """m-depth Anderson Acceleration engine for Newton-Raphson iterations.
    
    Accelerates the fixed-point map g(x) = x + f(x) (where f(x) = du is the
    Newton direction K^-1 * r) by finding the optimal linear combination of
    previous m iterates that minimizes the norm of the predicted residual.
    """

    def __init__(
        self,
        m: int = 4,
        beta: float = 1.0,
        reg: float = 1e-10,
        cond_max: float = 1e8,
        method: str = "lstsq"
    ):
        """
        Parameters
        ----------
        m : int
            Maximum history depth (sliding window size, typically 2 to 6).
            If m <= 0, Anderson acceleration degenerates to standard Newton.
        beta : float
            Mixing / relaxation parameter (0 < beta <= 1.0). Default 1.0.
            Values < 1.0 provide additional damping in stiff/rough non-linearities.
        reg : float
            Regularization parameter (rcond) for least-squares solve.
        cond_max : float
            Maximum allowable condition number for the difference matrix.
            If exceeded, oldest history columns are pruned to avoid ill-conditioning.
        method : str
            Least-squares method: "lstsq" (SVD-based, robust) or "qr" (QR-based).
        """
        self.m = max(0, int(m))
        self.beta = float(beta)
        self.reg = float(reg)
        self.cond_max = float(cond_max)
        self.method = method.lower()

        # History buffers: store past iterates x and residuals f = du
        self.history_x: List[np.ndarray] = []
        self.history_f: List[np.ndarray] = []

    def reset(self) -> None:
        """Clear all stored history vectors."""
        self.history_x.clear()
        self.history_f.clear()

    @property
    def current_depth(self) -> int:
        """Current number of difference pairs available (min(len(history)-1, m))."""
        if len(self.history_x) <= 1:
            return 0
        return min(len(self.history_x) - 1, self.m)

    def record(self, x: np.ndarray, f: np.ndarray) -> None:
        """Record an iterate (x, f) into history without advancing or accelerating."""
        if self.m <= 0:
            return
        self.history_x.append(np.asarray(x, dtype=np.float64).copy())
        self.history_f.append(np.asarray(f, dtype=np.float64).copy())
        # Keep at most m + 1 entries to produce at most m differences
        while len(self.history_x) > self.m + 1:
            self.history_x.pop(0)
            self.history_f.pop(0)

    def replace_last_x(self, x_new: np.ndarray) -> None:
        """Update the most recently recorded x (e.g. after line search modified it)."""
        if self.history_x:
            self.history_x[-1] = np.asarray(x_new, dtype=np.float64).copy()

    def pop_last(self) -> None:
        """Remove the most recent entry (e.g. if the trial step was completely rejected)."""
        if self.history_x:
            self.history_x.pop()
        if self.history_f:
            self.history_f.pop()

    def step(
        self,
        x: np.ndarray,
        f: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the accelerated next iterate x_{k+1}.
        
        Parameters
        ----------
        x : np.ndarray
            Current iterate (e.g. displacement u_k).
        f : np.ndarray
            Current Newton correction / residual (e.g. du_k = K^-1 * r).
        mask : Optional[np.ndarray]
            Boolean mask indicating active/free DOFs over which the least-squares
            residual norm should be minimized. If None, all DOFs are used.

        Returns
        -------
        np.ndarray
            Accelerated next iterate x_{k+1}.
        """
        x_curr = np.asarray(x, dtype=np.float64).copy()
        f_curr = np.asarray(f, dtype=np.float64).copy()

        if self.m <= 0 or len(self.history_x) == 0:
            # No acceleration: return standard relaxed step and record
            self.record(x_curr, f_curr)
            return x_curr + self.beta * f_curr

        # Form difference matrices E_k = [Delta x] and F_k = [Delta f]
        # We use the latest p points in history, where p = min(len(history), m)
        p = min(len(self.history_x), self.m)
        hist_x_sub = self.history_x[-p:]
        hist_f_sub = self.history_f[-p:]

        # Differences relative to current or consecutive:
        # Standard Walker & Ni formulation:
        # Delta x_i = x_{i+1} - x_i, Delta f_i = f_{i+1} - f_i
        # For k pairs: [x_hist[0], ..., x_hist[p-1], x_curr]
        xs = hist_x_sub + [x_curr]
        fs = hist_f_sub + [f_curr]

        p_diff = len(xs) - 1
        delta_x_list = [xs[i + 1] - xs[i] for i in range(p_diff)]
        delta_f_list = [fs[i + 1] - fs[i] for i in range(p_diff)]

        E_k = np.column_stack(delta_x_list)  # (N, p_diff)
        F_k = np.column_stack(delta_f_list)  # (N, p_diff)

        # Apply mask (e.g. free DOFs) to the least squares solve
        if mask is not None:
            A = F_k[mask, :]
            b = f_curr[mask]
        else:
            A = F_k
            b = f_curr

        # Solve least-squares problem: min_gamma || A * gamma - b ||_2
        gamma = self._solve_least_squares(A, b)

        if gamma is None or np.any(np.isnan(gamma)) or np.any(np.isinf(gamma)):
            # Least squares failure: fall back to unaccelerated step
            x_next = x_curr + self.beta * f_curr
        else:
            # Match columns in case condition monitoring pruned oldest differences
            k_cols = len(gamma)
            E_used = E_k[:, -k_cols:]
            F_used = F_k[:, -k_cols:]
            x_bar = x_curr - E_used @ gamma
            f_bar = f_curr - F_used @ gamma
            x_next = x_bar + self.beta * f_bar

        # Record current iterate into history for next step
        self.record(x_curr, f_curr)
        return x_next

    def _solve_least_squares(self, A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """Solve min_gamma || A * gamma - b ||_2 with condition monitoring and pruning."""
        n_cols = A.shape[1]
        if n_cols == 0:
            return None

        # Check conditioning and prune oldest columns if severely ill-conditioned
        while n_cols > 1:
            try:
                s = np.linalg.svd(A, compute_uv=False)
                if s[-1] > 0 and (s[0] / s[-1]) > self.cond_max:
                    # Drop oldest difference column (first column)
                    A = A[:, 1:]
                    n_cols -= 1
                    continue
            except Exception:
                pass
            break

        try:
            if self.method == "qr":
                Q, R = np.linalg.qr(A, mode="reduced")
                r_cond = np.linalg.cond(R)
                if r_cond < self.cond_max:
                    gamma = np.linalg.solve(R, Q.T @ b)
                    # If columns were pruned, pad leading zeros
                    if len(gamma) < A.shape[1]:
                        pass
                    return gamma

            # Default: robust SVD-based lstsq
            gamma, _, rank, _ = np.linalg.lstsq(A, b, rcond=self.reg)
            return gamma
        except Exception:
            return None


# ======================================================================
# AA-Newton 설정 객체 및 편의 함수
# ======================================================================

class AndersonConfig:
    """Anderson Acceleration(AA-Newton) 플래그 옵션을 하나의 객체로 관리.

    Parameters
    ----------
    use_anderson_accel : bool
        True이면 Newton-Raphson 반복에 Anderson 가속 적용. 기본값 False.
    anderson_m : int
        슬라이딩 윈도우 이력 깊이. 보통 2~6 사이. 기본값 4.
    anderson_beta : float
        믹싱/완화 파라미터 (0 < beta <= 1.0). 기본값 1.0 (완전 Anderson 단계).
        0.5~0.8로 낮추면 강한 비선형 문제에서 추가 안정성 제공.

    Notes
    -----
    AA-Newton은 다음 조건에서 효과가 뚜렷하게 나타납니다:
    - Newton 반복이 5회 이상 필요한 강한 비선형 문제
    - 소성 이력, 대변형 좌굴, 접촉 스태터링
    - 단순 선형 문제나 이미 2~3회 반복으로 수렴하는 경우: 효과 없음
    """

    def __init__(
        self,
        use_anderson_accel: bool = False,
        anderson_m: int = 4,
        anderson_beta: float = 1.0,
    ):
        self.use_anderson_accel = bool(use_anderson_accel)
        self.anderson_m = int(anderson_m)
        self.anderson_beta = float(anderson_beta)

    def to_dict(self) -> dict:
        """solve_step() 키워드 인수로 직접 전달 가능한 dict 반환."""
        return {
            "use_anderson_accel": self.use_anderson_accel,
            "anderson_m": self.anderson_m,
            "anderson_beta": self.anderson_beta,
        }

    def __repr__(self) -> str:
        return (
            f"AndersonConfig("
            f"use_anderson_accel={self.use_anderson_accel}, "
            f"anderson_m={self.anderson_m}, "
            f"anderson_beta={self.anderson_beta})"
        )


def solve_with_anderson(solver, aa_config: AndersonConfig, **kwargs):
    """AA-Newton 설정을 적용하여 solver.solve_step() 한 스텝을 실행하는 편의 함수.

    Parameters
    ----------
    solver : DynamicSolver3D | DynamicSolver2D
        WHT 비선형 솔버 인스턴스.
    aa_config : AndersonConfig
        Anderson Acceleration 설정 객체.
    **kwargs
        solve_step()에 추가로 전달할 키워드 인수
        (예: dt=1.0, f_ext=f_ext, max_iters=25).

    Returns
    -------
    (converged: bool, n_iters: int)
        solve_step()의 반환값 그대로 전달.

    Examples
    --------
    >>> from dispsolver.solver.anderson_acceleration import AndersonConfig, solve_with_anderson
    >>> aa = AndersonConfig(use_anderson_accel=True, anderson_m=5, anderson_beta=0.8)
    >>> conv, iters = solve_with_anderson(solver, aa, dt=1.0, f_ext=f_ext, max_iters=30)
    """
    kwargs.update(aa_config.to_dict())
    return solver.solve_step(**kwargs)
