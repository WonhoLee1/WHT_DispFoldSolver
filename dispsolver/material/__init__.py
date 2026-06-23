"""Material module — Hyperelastic models with JAX autodiff stress/tangent."""

from .base import MaterialModel
from .neohookean import NeoHookean
from .yeoh import Yeoh
from .arruda_boyce import ArrudaBoyce
from .viscoelastic import ViscoelasticMaterial
from .linear_viscoelastic import LinearViscoelastic
from .plastic import J2Plasticity

__all__ = [
    "MaterialModel",
    "NeoHookean",
    "Yeoh",
    "ArrudaBoyce",
    "ViscoelasticMaterial",
    "LinearViscoelastic",
    "J2Plasticity",
]
