"""
type_tags.py
============
Canonical material "type" tag strings, shared by every layer that needs
to classify a material without depending on isinstance() checks or
re-deriving it from raw property dicts:

- dispsolver/fold_model_config.py -- MaterialDef.type (config source)
- dispsolver/io/model_builder.py -- tags each pid parsed from a .inp
  deck's *MATERIAL block, so the .inp-parsed path and the pure-Python
  config-built path (examples/ex13_unified_model_io.py::run_build())
  agree on the same tag strings for the same physical material.
- dispsolver/postprocess/model_review.py -- picks representative
  properties to print per tag.

Defined here (not in examples/) so dispsolver/io/model_builder.py (core
library code) never has to import from examples/ -- only examples/
imports from dispsolver/, never the other way around.
"""

J2_PLASTIC = "j2-plasticity"                      # PET
ARRUDA_BOYCE_VISCO = "arruda-boyce-viscoelastic"  # PSA
NEOHOOKEAN = "neohookean"                         # STEEL
