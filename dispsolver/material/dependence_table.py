"""
Abaqus *DEPENDENCIES Table Interpolation Module.

Implements pure-function 1D and 2D interpolation with constant boundary clamping,
fully compatible with Numba njit (fastmath=True, cache=True).
Complies with Abaqus theory manual specification for tabular material dependencies.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def eval_table_1d_numba(x_coords: np.ndarray, y_values: np.ndarray, x: float) -> float:
    """
    1D piecewise-linear table interpolation with constant extrapolation (clamp).

    Parameters
    ----------
    x_coords : np.ndarray (1D, float64)
        Sorted independent variable coordinates (e.g. Temperature T)
    y_values : np.ndarray (1D, float64)
        Dependent material parameter values (e.g. Elastic Modulus E or Nu)
    x : float
        Evaluation point

    Returns
    -------
    float
        Interpolated value, exact at nodal points, clamped beyond range.
    """
    n = len(x_coords)
    if n == 0:
        return 0.0
    if n == 1:
        return y_values[0]

    # Clamp below lower bound
    if x <= x_coords[0]:
        return y_values[0]
    # Clamp above upper bound
    if x >= x_coords[n - 1]:
        return y_values[n - 1]

    # Binary search for interval [x_i, x_{i+1}]
    low = 0
    high = n - 1
    while low <= high:
        mid = (low + high) // 2
        if x_coords[mid] <= x:
            if mid + 1 < n and x < x_coords[mid + 1]:
                low = mid
                break
            low = mid + 1
        else:
            high = mid - 1

    idx = low
    if idx >= n - 1:
        return y_values[n - 1]

    x0 = x_coords[idx]
    x1 = x_coords[idx + 1]
    y0 = y_values[idx]
    y1 = y_values[idx + 1]

    dx = x1 - x0
    if abs(dx) < 1e-15:
        return y0

    t = (x - x0) / dx
    return y0 + t * (y1 - y0)


@njit(fastmath=True, cache=True)
def eval_table_2d_numba(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    z_grid: np.ndarray,
    x: float,
    y: float
) -> float:
    """
    2D bilinear table interpolation on a regular tensor-product grid with clamping.

    Used for Abaqus *PLASTIC yield stress sigma_y = f(eps_pl, T).

    Parameters
    ----------
    grid_x : np.ndarray (1D, float64, size Nx)
        Plastic strain coordinates
    grid_y : np.ndarray (1D, float64, size Ny)
        Temperature coordinates
    z_grid : np.ndarray (2D, float64, shape (Nx, Ny))
        Yield stress values at grid vertices
    x : float
        Evaluation plastic strain
    y : float
        Evaluation temperature

    Returns
    -------
    float
        Bilinearly interpolated value.
    """
    nx = len(grid_x)
    ny = len(grid_y)

    if nx == 0 or ny == 0:
        return 0.0
    if nx == 1 and ny == 1:
        return z_grid[0, 0]

    # Clamp x to [grid_x[0], grid_x[-1]]
    x_clamped = min(max(x, grid_x[0]), grid_x[nx - 1])
    # Clamp y to [grid_y[0], grid_y[-1]]
    y_clamped = min(max(y, grid_y[0]), grid_y[ny - 1])

    # Find x interval
    ix = 0
    for i in range(nx - 1):
        if grid_x[i] <= x_clamped <= grid_x[i + 1]:
            ix = i
            break

    # Find y interval
    iy = 0
    for j in range(ny - 1):
        if grid_y[j] <= y_clamped <= grid_y[j + 1]:
            iy = j
            break

    x0 = grid_x[ix]
    x1 = grid_x[ix + 1]
    y0 = grid_y[iy]
    y1 = grid_y[iy + 1]

    dx = x1 - x0
    dy = y1 - y0

    xi = (x_clamped - x0) / dx if abs(dx) > 1e-15 else 0.0
    eta = (y_clamped - y0) / dy if abs(dy) > 1e-15 else 0.0

    f00 = z_grid[ix, iy]
    f10 = z_grid[ix + 1, iy]
    f01 = z_grid[ix, iy + 1]
    f11 = z_grid[ix + 1, iy + 1]

    return (
        (1.0 - xi) * (1.0 - eta) * f00
        + xi * (1.0 - eta) * f10
        + (1.0 - xi) * eta * f01
        + xi * eta * f11
    )
