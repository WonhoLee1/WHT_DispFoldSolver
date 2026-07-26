"""Postprocessing: result persistence, interlayer-shear tracking, 2D viewer."""

from .animate import animate, plot_slip_history, plot_step, save_frame
from .interlayer import LayerSlipTracker
from .result_io import Result, ResultWriter, load_result, to_vtkhdf

__all__ = [
    "Result",
    "ResultWriter",
    "load_result",
    "to_vtkhdf",
    "LayerSlipTracker",
    "animate",
    "plot_step",
    "save_frame",
    "plot_slip_history",
    "PostprocessViewer",
    "launch_from_solver",
]


def __getattr__(name):
    """Import the Qt viewer lazily.

    `viewer` pulls in PySide6, which is a heavy (and, on headless
    machines, unavailable) dependency. Loading a saved result for
    analysis must not require a GUI toolkit, so the viewer is only
    imported if something actually asks for it.
    """
    if name in ("PostprocessViewer", "launch_from_solver"):
        from . import viewer
        return getattr(viewer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
