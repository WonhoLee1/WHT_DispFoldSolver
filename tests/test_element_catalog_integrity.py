"""The element catalog is the only element dispatch, and it matches the code.

model/element_catalog.py decides which kernel runs an element type; the
solvers call ElementCatalog.resolve and read KERNELS for co-rotation,
connectivity trimming and the elastic-only refusal. These tests keep the
catalog and the code from drifting apart again (the previous catalog was
read by no solver and already disagreed with it).
"""
import inspect
import re

import numpy as np
import pytest

from dispsolver.model.element_catalog import KERNELS, ElementCatalog, cr_wrapped_kernels


def _assembly_branches(module) -> set:
    src = inspect.getsource(module)
    body = src[src.index("def assemble_system"):]
    body = body[:body.index("\n    def ", 10)]
    return {int(k) for k in re.findall(r"\bk == (\d+):", body)}


def test_every_kernel_in_the_table_has_an_assembly_branch_and_vice_versa():
    import dispsolver.solver2d.dynamic2d as d2
    import dispsolver.solver3d.dynamic3d as d3
    assert _assembly_branches(d3) == set(KERNELS[3])
    assert _assembly_branches(d2) == set(KERNELS[2])


def test_every_descriptor_points_at_existing_kernels():
    for d in ElementCatalog.all_descriptors():
        for k in (d.kernel_nlgeom_yes, d.kernel_nlgeom_no, d.kernel_yes_incompressible,
                  d.kernel_no_incompressible, d.kernel_yes_inelastic, d.kernel_no_inelastic):
            assert k is None or k in KERNELS[d.dimension], (d.type_name, k)


def test_solvers_take_their_co_rotated_set_from_the_catalog():
    import dispsolver.solver2d.dynamic2d as d2
    import dispsolver.solver3d.dynamic3d as d3
    assert d3._CR_WRAPPED_KERNELS == cr_wrapped_kernels(3)
    assert d2._CR_WRAPPED_KERNELS_2D == cr_wrapped_kernels(2)


def test_no_hand_written_dispatch_chain_is_left():
    from dispsolver.solver2d.dynamic2d import DynamicSolver2D
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D
    for cls in (DynamicSolver2D, DynamicSolver3D):
        src = inspect.getsource(cls._build_kernel_groups)
        assert "ElementCatalog.resolve" in src and 'et in ["' not in src, cls.__name__


@pytest.mark.parametrize("et,dim,nlgeom,nu,lin,expect", [
    ("C3D8", 3, True, 0.3, True, 1), ("C3D8", 3, False, 0.3, True, 16),
    ("C3D10", 3, False, 0.49, True, 4), ("C3D8I", 3, True, 0.3, False, 0),
    ("C3D8I", 3, False, 0.3, True, 18), ("CPE4", 2, True, 0.3, True, 5), ("CPE4", 2, False, 0.3, True, 0),
    ("CAX4", 2, False, 0.3, True, 17),
])
def test_resolve_examples(et, dim, nlgeom, nu, lin, expect):
    assert ElementCatalog.resolve(et, dim, nlgeom, nu=nu, linear_elastic=lin)[0] == expect


@pytest.mark.parametrize("et,dim,nlgeom", [("S4R", 3, True), ("S8R", 3, False), ("CAX4", 2, True),
                                           ("C3D4H", 3, True), ("C3D8XYZ", 3, False)])
def test_resolve_refuses(et, dim, nlgeom):
    with pytest.raises(NotImplementedError):
        ElementCatalog.resolve(et, dim, nlgeom)
