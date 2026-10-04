"""dispsolver.output
=================
Modern Python Field Output, History Output, and ODB data management framework for FEA.
Provides Abaqus API parity, NumPy vectorization, HDF5 persistence, and VTKHDF export.
"""

from __future__ import annotations

from dispsolver.output.variables import (
    FieldPosition,
    FieldType,
    FieldVar,
    HistoryVar,
    FIELD_VARIABLE_COMPONENTS,
    HISTORY_ENERGY_VARIABLES,
    HISTORY_NODAL_VARIABLES,
    HISTORY_SECTION_VARIABLES,
)
from dispsolver.output.requests import (
    FieldOutputRequest,
    HistoryOutputRequest,
    OutputTimingController,
)
from dispsolver.output.field_data import (
    FieldOutput,
    FieldValue,
    FieldLocation,
)
from dispsolver.output.history_data import (
    HistoryOutput,
    HistoryRegion,
    EnergyTracker,
)
from dispsolver.output.odb import (
    Odb,
    OdbStep,
    OdbFrame,
)
from dispsolver.output.vtkhdf_writer import (
    TransientVTKHDFWriter,
)
from dispsolver.output.manager import (
    OutputManager,
)


def open_odb(filepath: str) -> Odb:
    """Convenience function to open and load an ODB from an HDF5 binary file."""
    return Odb.open_odb(filepath)


__all__ = [
    "FieldPosition",
    "FieldType",
    "FieldVar",
    "HistoryVar",
    "FieldOutputRequest",
    "HistoryOutputRequest",
    "OutputTimingController",
    "FieldOutput",
    "FieldValue",
    "FieldLocation",
    "HistoryOutput",
    "HistoryRegion",
    "EnergyTracker",
    "Odb",
    "OdbStep",
    "OdbFrame",
    "open_odb",
    "TransientVTKHDFWriter",
    "OutputManager",
]
