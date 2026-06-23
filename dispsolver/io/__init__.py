"""
dispsolver.io
=============
Abaqus .inp file import pipeline for dispsolver.

Usage
-----
>>> from dispsolver.io import read_abaqus_input
>>> result = read_abaqus_input("model.inp")
>>> mesh = result.mesh
>>> solver = DynamicSolver(mesh, result.materials[1],
...                        rho=result.solver_config["density"],
...                        material_params=result.material_params.get(1))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

from .abaqus_lexer import AbaqusKeywordBlock, tokenize, tokenize_string
from .abaqus_model import (
    AbaqusAmplitude,
    AbaqusBoundary,
    AbaqusContactPair,
    AbaqusContactProperty,
    AbaqusDload,
    AbaqusElement,
    AbaqusLoad,
    AbaqusMaterial,
    AbaqusModel,
    AbaqusMpc,
    AbaqusNode,
    AbaqusSection,
    AbaqusStep,
    AbaqusSurface,
    AbaqusTie,
)
from .abaqus_parser import AbaqusParser
from .model_builder import ModelBuilder, ModelBuilderResult


def read_abaqus_input(filepath: str) -> ModelBuilderResult:
    """Read an Abaqus .inp file and build dispsolver-native objects.

    This is the main entry point for the Abaqus import pipeline.

    Pipeline
    --------
    1. tokenize(filepath) → list of AbaqusKeywordBlock
    2. AbaqusParser.parse(blocks) → AbaqusModel
    3. ModelBuilder(abq_model).build() → ModelBuilderResult

    Parameters
    ----------
    filepath : str
        Path to the .inp file.

    Returns
    -------
    ModelBuilderResult with fields:
        mesh : Mesh
        materials : dict[int, object]
        material_params : dict[int, dict]
        constraints : list
        solver_config : dict
        amplitudes : dict[str, Amplitude]
        dload_configs : list[dict]
        contact_pairs : list[dict]
        contact_surfaces : list[dict]

    Examples
    --------
    >>> result = read_abaqus_input("tests/fixtures/simple_quad.inp")
    >>> result.mesh.node_count()
    4

    Raises
    ------
    NotImplementedError
        For unsupported element types (C3D8, etc.) or features.
    FileNotFoundError
        If the .inp file does not exist.
    """
    blocks = tokenize(filepath)
    parser = AbaqusParser()
    abq_model = parser.parse(blocks)
    builder = ModelBuilder(abq_model)
    result = builder.build()
    return result


__all__ = [
    # Public API
    "read_abaqus_input",
    "ModelBuilderResult",
    # Parser data model
    "AbaqusModel",
    "AbaqusNode",
    "AbaqusElement",
    "AbaqusMaterial",
    "AbaqusSection",
    "AbaqusBoundary",
    "AbaqusLoad",
    "AbaqusStep",
    "AbaqusAmplitude",
    "AbaqusDload",
    "AbaqusMpc",
    "AbaqusTie",
    "AbaqusContactPair",
    "AbaqusContactProperty",
    "AbaqusSurface",
    # Lexer
    "AbaqusKeywordBlock",
    "tokenize",
    "tokenize_string",
    # Parser
    "AbaqusParser",
    # Builder
    "ModelBuilder",
]
