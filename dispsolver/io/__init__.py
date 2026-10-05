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
from .legacy_adapter import model_to_builder_result, ModelBuilderResult, AbaqusBoundary, ABAQUS_ELEMENT_MAP
from .abaqus_reader import _parse_source, detect_dim


def read_abaqus_input(filepath: str) -> ModelBuilderResult:
    """Read an Abaqus .inp file and build dispsolver-native objects.

    This is the main entry point for the Abaqus import pipeline,
    powered by the production reader and legacy_adapter.

    Pipeline
    --------
    1. _parse_source(filepath, dim=detect_dim(filepath)) → reader, model
    2. model_to_builder_result(model) → ModelBuilderResult
    """
    dim = detect_dim(filepath)
    reader, model = _parse_source(filepath, dim=dim, strict=False)
    return model_to_builder_result(model)


__all__ = [
    # Public API
    "read_abaqus_input",
    "ModelBuilderResult",
    "AbaqusBoundary",
    "ABAQUS_ELEMENT_MAP",
    # Lexer
    "AbaqusKeywordBlock",
    # New Production Reader & Executor Architecture
    "read_inp",
    "execute_inp",
    "execute_inp_smoke",
    "AbaqusInputReader",
    "GLOBAL_KEYWORD_REGISTRY",
    "KeywordRegistry",
    "KeywordContext",
    "KeywordSpec",
    "ParamSpec",
    "RowSpec",
    "normalize_param_name",
    "keyword_modules",
]

from .abaqus_reader import read_inp, execute_inp, execute_inp_smoke, AbaqusInputReader
from .keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordRegistry,
    KeywordContext,
    KeywordSpec,
    ParamSpec,
    RowSpec,
    normalize_param_name,
)
from . import keywords as keyword_modules

