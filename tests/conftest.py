"""Pytest configuration and fixtures for WHT_DispFoldSolver tests."""

import pytest


def pytest_addoption(parser):
    """Add --elem_jit option to pytest command line."""
    parser.addoption(
        "--elem_jit",
        action="store",
        default="jax",
        choices=["jax", "numba", "numpy"],
        help="Element JIT backend: jax (default), numba, or numpy"
    )


@pytest.fixture(scope="session")
def elem_jit_backend(request):
    """Fixture providing the selected elem_jit backend."""
    return request.config.getoption("--elem_jit")
