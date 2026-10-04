"""Material module — Hyperelastic models with JAX autodiff stress/tangent."""

from .base import MaterialModel
from .neohookean import NeoHookean
from .yeoh import Yeoh
from .arruda_boyce import ArrudaBoyce
from .viscoelastic import ViscoelasticMaterial
from .linear_viscoelastic import LinearViscoelastic
from .plastic import J2Plasticity
from .catalog import (
    MaterialSpec,
    MATERIALS,
    material_id,
    pack_props,
    is_linear_elastic,
)

__all__ = [
    "MaterialModel",
    "NeoHookean",
    "Yeoh",
    "ArrudaBoyce",
    "ViscoelasticMaterial",
    "LinearViscoelastic",
    "J2Plasticity",
    "MaterialSpec",
    "MATERIALS",
    "material_id",
    "pack_props",
    "is_linear_elastic",
]
