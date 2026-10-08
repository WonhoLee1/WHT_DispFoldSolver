"""
test_abaqus_contact_api.py
==========================
Comprehensive validation suite for Abaqus-grade Contact General and Part Exterior Contact.

Verifications:
1. 3D Exterior Face Detection on C3D8 (Hex) and C3D4 (Tet) meshes.
2. Abaqus Python API parity (ContactProperty, ContactStd, SurfaceToSurfaceContactStd).
3. Part.get_exterior_faces() and part.exterior property.
4. End-to-end 3D Contact General solve: Two approaching parts contact and compress
   with zero manual surface selection (ALL EXTERIOR automatic detection).
"""

import numpy as np
import pytest

from dispsolver.mesh3d import Mesh3D
from dispsolver.model.model import Model
from dispsolver.model.part import Part
from dispsolver.model.interaction import ContactProperty, GeneralContact, ContactStd, SurfaceToSurfaceContactStd
from dispsolver.contact.contact_surface import auto_detect_exterior_3d, ContactSurface3D
from dispsolver.contact.general_contact3d import GeneralContact3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.solver.step_executor import MultiStepExecutor


def test_auto_detect_exterior_3d_c3d8():
    """Verify exterior face extraction and outward normal calculation on a 2x2x2 C3D8 grid."""
    mesh = Mesh3D()
    # 2x2x2 elements -> 3x3x3 = 27 nodes
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 2.0]
    zs = [0.0, 1.0, 2.0]

    grid = {}
    nid = 1
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                mesh.add_node(nid, x, y, z)
                grid[(i, j, k)] = nid
                nid += 1

    eid = 1
    for k in range(2):
        for j in range(2):
            for i in range(2):
                conn = [
                    grid[(i, j, k)],
                    grid[(i + 1, j, k)],
                    grid[(i + 1, j + 1, k)],
                    grid[(i, j + 1, k)],
                    grid[(i, j, k + 1)],
                    grid[(i + 1, j, k + 1)],
                    grid[(i + 1, j + 1, k + 1)],
                    grid[(i, j + 1, k + 1)],
                ]
                mesh.add_element(eid, conn, "C3D8")
                eid += 1

    assert mesh.num_nodes == 27
    assert mesh.num_elements == 8

    # Detect exterior faces
    surf = auto_detect_exterior_3d(mesh)

    # A 2x2x2 cube has 6 faces, each face made of 2x2 = 4 element quads -> exactly 24 exterior faces!
    assert len(surf.faces) == 24
    assert len(surf.normals) == 24

    # All exterior nodes: 27 total nodes - 1 center internal node = 26 exterior boundary nodes
    assert len(surf.node_ids) == 26
    center_nid = grid[(1, 1, 1)]
    assert center_nid not in surf.node_ids

    # Outward normal orientation check: centroid of cube is (1, 1, 1)
    center = np.array([1.0, 1.0, 1.0])
    for face, normal in zip(surf.faces, surf.normals):
        face_coords = np.array([
            [mesh.nodes[nid].x, mesh.nodes[nid].y, mesh.nodes[nid].z]
            for nid in face
        ])
        face_center = np.mean(face_coords, axis=0)
        n = np.array(normal)
        # Vector from cube center to face center must point in the same direction as normal!
        assert np.dot(face_center - center, n) > 0.0


def test_auto_detect_exterior_3d_c3d4():
    """Verify exterior face extraction on 2 C3D4 tetrahedral elements sharing 1 face."""
    mesh = Mesh3D()
    # 5 nodes: shared base (1, 2, 3), apex 1 at (0, 0, 1), apex 2 at (0, 0, -1)
    mesh.add_node(1, 0.0, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0, 0.0)
    mesh.add_node(3, 0.0, 1.0, 0.0)
    mesh.add_node(4, 0.2, 0.2, 1.0)   # Top tet apex
    mesh.add_node(5, 0.2, 0.2, -1.0)  # Bottom tet apex

    # Tet 1 (top): nodes (1, 2, 3, 4)
    mesh.add_element(1, [1, 2, 3, 4], "C3D4")
    # Tet 2 (bottom): nodes (1, 3, 2, 5) sharing base face (1, 2, 3)
    mesh.add_element(2, [1, 3, 2, 5], "C3D4")

    surf = auto_detect_exterior_3d(mesh)

    # 2 tets * 4 faces = 8 total face occurrences. Shared face (1, 2, 3) canceled -> 6 exterior Tri3 faces!
    assert len(surf.faces) == 6
    for face in surf.faces:
        assert len(face) == 3
        # Shared face (1, 2, 3) must NOT be in exterior faces
        assert set(face) != {1, 2, 3}


def test_part_exterior_methods():
    """Verify Part.get_exterior_faces() and Part.exterior property."""
    part = Part(name="TestBlock", dim=3)
    # 8-node cube
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    for i, c in enumerate(coords):
        part.Node(i + 1, c[0], c[1], c[2])

    part.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    ext_faces = part.get_exterior_faces()
    assert len(ext_faces) == 6

    ext_surf = part.exterior
    assert isinstance(ext_surf, ContactSurface3D)
    assert len(ext_surf.faces) == 6
    assert len(ext_surf.node_ids) == 8


def test_abaqus_contact_python_api_declarations():
    """Verify Abaqus-identical Python API scripting declarations on Model."""
    model = Model(name="AbaqusContactModel", dim=3)

    # 1. ContactProperty (*SURFACE INTERACTION)
    prop = model.ContactProperty(
        name="IntProp-Hard",
        normal_behavior="HARD",
        softness_scale=1.0
    )
    assert prop.name == "IntProp-Hard"
    assert prop.normal_behavior == "HARD"
    assert "IntProp-Hard" in model.contact_properties

    # 2. Step
    step1 = model.Step(name="Step-1", procedure="STATIC", time_period=1.0)

    # 3. General Contact (*CONTACT, GENERAL / ContactStd)
    gc = model.ContactStd(
        name="GlobalGeneralContact",
        createStepName="Step-1",
        useAllExterior=True,
        interactionProperty="IntProp-Hard"
    )
    assert isinstance(gc, ContactStd)
    assert isinstance(gc, GeneralContact)
    assert gc.use_all_exterior is True
    assert gc.interaction_property is prop
    assert "GlobalGeneralContact" in model.interactions
    assert "GlobalGeneralContact" in step1.interactions

    # 4. Surface-to-surface Contact Pair (*CONTACT PAIR / SurfaceToSurfaceContactStd)
    cp = model.SurfaceToSurfaceContactStd(
        name="PairContact",
        createStepName="Step-1",
        main="SURF_MASTER",
        secondary="SURF_SLAVE",
        interactionProperty="IntProp-Hard",
        sliding="FINITE"
    )
    assert isinstance(cp, SurfaceToSurfaceContactStd)
    assert cp.sliding == "FINITE"
    assert "PairContact" in model.interactions
    assert "PairContact" in step1.interactions


def test_general_contact_all_exterior_two_cubes_solve():
    """Full 3D FEA solve of two approaching deformable cubes using ContactStd(useAllExterior=True).

    Geometry:
    - Bottom Cube: [0, 10] x [0, 10] x [-10, 0]
    - Top Cube:    [0, 10] x [0, 10] x [0.2, 10.2]  (initial separation gap = 0.2 mm)
    - Boundary Conditions:
      * Bottom Cube bottom face fixed (ux=uy=uz=0 at z = -10).
      * Top Cube top face prescribed uz = -0.35 mm (closing gap 0.2 mm and penetrating by 0.15 mm).
    - Interaction:
      * ONLY model.ContactStd(name="AllExteriorContact", createStepName="Step-1", useAllExterior=True)
      * Zero manual surface selection!
    """
    model = Model(name="TwoCubeGeneralContact", dim=3)

    L = 10.0
    gap = 0.2

    # Part 1: Top Cube
    part_top = model.Part(name="TopCube")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, gap), (L, 0.0, gap), (L, L, gap), (0.0, L, gap),
        (0.0, 0.0, gap + L), (L, 0.0, gap + L), (L, L, gap + L), (0.0, L, gap + L),
    ]):
        part_top.Node(nid + 1, x, y, z)
    part_top.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    # Part 2: Bottom Cube
    part_bot = model.Part(name="BottomCube")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, -L), (L, 0.0, -L), (L, L, -L), (0.0, L, -L),
        (0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0),
    ]):
        part_bot.Node(nid + 1, x, y, z)
    part_bot.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    # Material & Section
    mat = model.Material(name="Steel")
    mat.Elastic(E=200000.0, nu=0.3)
    model.HomogeneousSolidSection(name="SolidSec", material="Steel")
    part_top.SectionAssignment(region="ALL", section_name="SolidSec")
    part_bot.SectionAssignment(region="ALL", section_name="SolidSec")

    # Assembly
    inst_top = model.root_assembly.Instance(name="TopInst", part=part_top)
    inst_bot = model.root_assembly.Instance(name="BotInst", part=part_bot)

    # Step
    step = model.Step(name="Step-1", procedure="STATIC", time_period=1.0, dt_init=1.0, dt_min=1e-5, dt_max=1.0)

    # Contact General: ALL EXTERIOR
    k_contact = 200000.0 * L  # E * L = 2.0e6
    model.ContactProperty(name="IntProp", normal_behavior="HARD")
    model.ContactStd(
        name="GeneralContact",
        createStepName="Step-1",
        useAllExterior=True,
        gamma=k_contact,
    )

    # Build solver system
    solver, sys = model.create_solver3d(step_name="Step-1")

    # Verify GeneralContact3D was created and attached
    assert len(solver.constraints) >= 1
    assert any(isinstance(c, GeneralContact3D) for c in solver.constraints)
    gc_constraint = next(c for c in solver.constraints if isinstance(c, GeneralContact3D))
    assert gc_constraint.use_all_exterior is True
    # 2 separate cubes -> 6 faces each = 12 total exterior faces
    assert len(gc_constraint.faces) == 12

    # Boundary Conditions
    # Fix bottom face of bottom cube (z = -L) in X, Y, Z
    bot_nids = [sys.node_local_to_global[("BotInst", nid)] for nid in [1, 2, 3, 4]]
    for gid in bot_nids:
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)
        solver.fix_dof(gid, 2, 0.0)

    # Rollers on X and Y for all other nodes to ensure clean 1D compression
    for gid in range(1, 17):
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)

    # Prescribe downward displacement on top face of top cube (z = gap + L)
    # Move down by delta_z = -0.35 mm (closing initial gap 0.2 mm, penetrating 0.15 mm)
    top_nids = [sys.node_local_to_global[("TopInst", nid)] for nid in [5, 6, 7, 8]]
    for gid in top_nids:
        solver.fix_dof(gid, 2, -0.35)

    # Solve Newton step
    converged, iters = solver.solve_step(dt=1.0, max_iters=30)
    assert converged is True, f"Contact General solve failed to converge in {iters} iterations."
    assert iters < 20, f"Newton took too many iterations ({iters})."

    # Verify contact mechanics
    stats = gc_constraint.last_stats
    assert stats["n_active"] > 0, "No contact quadrature points were activated!"
    assert stats["total_normal_force"] > 0.0, "Contact force must be strictly positive!"

    # Verify bottom face of top cube has penetrated / closed gap onto top face of bottom cube
    top_bottom_nids = [sys.node_local_to_global[("TopInst", nid)] for nid in [1, 2, 3, 4]]
    bot_top_nids = [sys.node_local_to_global[("BotInst", nid)] for nid in [5, 6, 7, 8]]

    nid_map = sys.nid_to_idx
    u_top_bottom = np.mean([solver.u[3 * nid_map[g] + 2] for g in top_bottom_nids])
    u_bot_top = np.mean([solver.u[3 * nid_map[g] + 2] for g in bot_top_nids])

    # Initial positions: top_bottom at z = 0.2, bot_top at z = 0.0
    z_final_top_bot = 0.2 + u_top_bottom
    z_final_bot_top = 0.0 + u_bot_top

    # Interface must have contacted: difference between final heights must be <= 0.02 mm (near zero gap)
    interface_gap = abs(z_final_top_bot - z_final_bot_top)
    assert interface_gap < 0.02, f"Interface gap was {interface_gap:.4f} mm, contact not properly closed."


def test_general_contact_part_exterior_pairs():
    """Verify ContactStd with explicit Part exterior pairs in includedPairs."""
    model = Model(name="PartPairContactModel", dim=3)
    L = 5.0
    gap = 0.1

    part_top = model.Part(name="PartTop")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, gap), (L, 0.0, gap), (L, L, gap), (0.0, L, gap),
        (0.0, 0.0, gap + L), (L, 0.0, gap + L), (L, L, gap + L), (0.0, L, gap + L),
    ]):
        part_top.Node(nid + 1, x, y, z)
    part_top.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    part_bot = model.Part(name="PartBot")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, -L), (L, 0.0, -L), (L, L, -L), (0.0, L, -L),
        (0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0),
    ]):
        part_bot.Node(nid + 1, x, y, z)
    part_bot.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    mat = model.Material(name="Rubber")
    mat.Elastic(E=1000.0, nu=0.45)
    model.HomogeneousSolidSection(name="RubberSec", material="Rubber")
    part_top.SectionAssignment(region="ALL", sectionName="RubberSec")
    part_bot.SectionAssignment(region="ALL", sectionName="RubberSec")

    inst_top = model.root_assembly.Instance(name="Top", part=part_top)
    inst_bot = model.root_assembly.Instance(name="Bot", part=part_bot)

    model.Step(name="Step-1", procedure="STATIC", time_period=1.0)
    model.ContactProperty(name="IntProp", normal_behavior="HARD")

    # User passes part_top.exterior and part_bot.exterior directly in includedPairs!
    model.ContactStd(
        name="PartContact",
        createStepName="Step-1",
        useAllExterior=True,
        includedPairs=[(part_top.exterior, part_bot.exterior)],
        gamma=1000.0 * L,
    )

    solver, sys = model.create_solver3d(step_name="Step-1")
    assert any(isinstance(c, GeneralContact3D) for c in solver.constraints)

    # Fix bottom of Bot
    for nid in [1, 2, 3, 4]:
        gid = sys.node_local_to_global[("Bot", nid)]
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)
        solver.fix_dof(gid, 2, 0.0)

    # Rollers on X, Y
    for gid in range(1, 17):
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)

    # Push top of Top down by -0.2 (gap is 0.1, penetration 0.1)
    for nid in [5, 6, 7, 8]:
        gid = sys.node_local_to_global[("Top", nid)]
        solver.fix_dof(gid, 2, -0.2)

    converged, iters = solver.solve_step(dt=1.0, max_iters=30)
    assert converged is True
    gc = next(c for c in solver.constraints if isinstance(c, GeneralContact3D))
    assert gc.last_stats["n_active"] > 0
    assert gc.last_stats["total_normal_force"] > 0.0


def test_contact_property_abaqus_submethods_and_constants():
    """Verify Abaqus Python API submethods: NormalBehavior, TangentialBehavior, Damping."""
    from dispsolver.model import (
        Model,
        HARD,
        TABULAR,
        EXPONENTIAL,
        LINEAR,
        PENALTY,
        FRICTIONLESS,
        AUTOMATIC,
        DAMPING_COEFFICIENT,
        CRITICAL_DAMPING_FRACTION,
    )

    model = Model(name="SubmethodModel")

    # 1. Verify interactionProperties alias
    assert model.interactionProperties is model.contact_properties

    # 2. Define ContactProperty using Abaqus chained sub-methods
    prop = model.ContactProperty("SoftCushion")
    prop.NormalBehavior(
        pressureOverclosure=TABULAR,
        table=(
            (0.0, -0.001),     # Pre-contact clearance at 1 um: zero pressure
            (1.0, 0.0),        # Mathematical touch at gap=0: 1.0 MPa cushion
            (1000.0, 0.1),     # 0.1 mm penetration: 1000.0 MPa hard resistance
        ),
    )
    prop.TangentialBehavior(
        formulation=PENALTY,
        table=((0.3,), ),
    )
    prop.Damping(
        definition=DAMPING_COEFFICIENT,
        dampingCoefficient=0.002,
        clearance=0.05,
        tangentFraction=0.15,
    )

    # Check parsed fields
    assert prop.normal_behavior == "SOFT_TABULAR"
    assert prop.tabular_pressure == [0.0, 1.0, 1000.0]
    assert prop.tabular_overclosure == [-0.001, 0.0, 0.1]
    assert prop.friction == 0.3
    assert prop.damping_definition == "DAMPING_COEFFICIENT"
    assert prop.damping_coefficient == 0.002
    assert prop.damping_clearance == 0.05
    assert prop.tangent_fraction == 0.15

    # 3. Test Exponential & Linear submethods
    prop_exp = model.ContactProperty("ExpProp")
    prop_exp.NormalBehavior(pressureOverclosure=EXPONENTIAL, table=((0.02, 10.0),))
    assert prop_exp.normal_behavior == "SOFT_EXPONENTIAL"
    assert prop_exp.exponential_c0 == 0.02
    assert prop_exp.exponential_p0 == 10.0

    prop_lin = model.ContactProperty("LinProp")
    prop_lin.NormalBehavior(pressureOverclosure=LINEAR, contactStiffness=5.0e4)
    assert prop_lin.normal_behavior == "SOFT_LINEAR"
    assert prop_lin.linear_stiffness == 5.0e4


def test_contact_stabilization_and_controls_binding():
    """Verify ContactControl & ContactStabilization binding to GeneralContact3D."""
    from dispsolver.model import (
        Model,
        TABULAR,
        PENALTY,
        AUTOMATIC,
        DAMPING_COEFFICIENT,
    )
    from dispsolver.constraint3d.pressure_overclosure import TabularSoftLaw

    model = Model(name="StabModel", dim=3)
    L = 2.0
    gap = 0.002

    part = model.Part(name="Block")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0),
        (0.0, 0.0, L), (L, 0.0, L), (L, L, L), (0.0, L, L),
    ]):
        part.Node(nid + 1, x, y, z)
    part.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    mat = model.Material(name="Mat")
    mat.Elastic(E=10000.0, nu=0.3)
    model.HomogeneousSolidSection(name="Sec", material="Mat")
    part.SectionAssignment(region="ALL", sectionName="Sec")

    model.root_assembly.Instance(name="Inst1", part=part)

    model.Step(name="Step-1", procedure="STATIC")

    # Define ContactProperty with pre-contact tabular cushion
    prop = model.ContactProperty("SoftProp")
    prop.NormalBehavior(
        pressureOverclosure=TABULAR,
        table=((0.0, -0.001), (1.0, 0.0), (100.0, 0.05)),
    )

    # Define ContactControl and ContactStabilization
    ctrl = model.ContactControl(
        name="ContCtrl-1",
        stabilize=AUTOMATIC,
        factor=2.5,
        tangentFraction=0.25,
        dampingCoefficient=0.015,
        clearance=0.08,
    )
    assert ctrl.name in model.contact_controls

    stab = model.ContactStabilization(
        name="StabConfig-1",
        definition=AUTOMATIC,
        dampingFactor=1.8,
    )
    assert stab.name in model.contact_stabilizations

    # Attach to GeneralContact via ContactStd
    model.ContactStd(
        name="GC_Stabilized",
        createStepName="Step-1",
        useAllExterior=True,
        interactionProperty="SoftProp",
        contactControls="ContCtrl-1",
    )

    solver, sys = model.create_solver3d("Step-1")
    gc = next(c for c in solver.constraints if isinstance(c, GeneralContact3D))

    # Verify law is attached and is TabularSoftLaw
    assert gc.law is not None
    assert isinstance(gc.law, TabularSoftLaw)
    assert np.allclose(gc.law.h_pts, [-0.001, 0.0, 0.05])
    assert np.allclose(gc.law.p_pts, [0.0, 1.0, 100.0])

    # Verify stabilization parameters are transferred
    assert gc.damping_coeff == 0.015
    assert gc.damping_clearance == 0.08
    assert gc.tangent_fraction == 0.25


def test_contact_stabilization_and_soft_tangent_solve():
    """Verify solve convergence with pre-contact soft tangent cushion and contact stabilization."""
    from dispsolver.model import (
        Model,
        TABULAR,
        PENALTY,
        AUTOMATIC,
        DAMPING_COEFFICIENT,
    )

    model = Model(name="StabilizedSolveModel", dim=3)
    L = 5.0
    initial_gap = 0.05  # 50 um initial gap

    # Part 1: Top Block
    part_top = model.Part(name="TopBlock")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, initial_gap), (L, 0.0, initial_gap), (L, L, initial_gap), (0.0, L, initial_gap),
        (0.0, 0.0, initial_gap + L), (L, 0.0, initial_gap + L), (L, L, initial_gap + L), (0.0, L, initial_gap + L),
    ]):
        part_top.Node(nid + 1, x, y, z)
    part_top.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    # Part 2: Bottom Block
    part_bot = model.Part(name="BotBlock")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, -L), (L, 0.0, -L), (L, L, -L), (0.0, L, -L),
        (0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0),
    ]):
        part_bot.Node(nid + 1, x, y, z)
    part_bot.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    mat = model.Material(name="Mat")
    mat.Elastic(E=50000.0, nu=0.3)
    model.HomogeneousSolidSection(name="Sec", material="Mat")
    part_top.SectionAssignment(region="ALL", sectionName="Sec")
    part_bot.SectionAssignment(region="ALL", sectionName="Sec")

    model.root_assembly.Instance(name="TopInst", part=part_top)
    model.root_assembly.Instance(name="BotInst", part=part_bot)

    model.Step(name="Step-1", procedure="STATIC", time_period=1.0)

    # Abaqus-parity Contact Property with pre-contact soft tangent
    prop = model.ContactProperty("AbaqusSoftCushion")
    prop.NormalBehavior(
        pressureOverclosure=TABULAR,
        table=(
            (0.0, -0.01),      # Pre-contact clearance at 10 um: begins soft tangent cushion
            (0.5, 0.0),        # Gap closure (touch): 0.5 MPa cushion
            (1000.0, 0.05),    # Penetration: 1000 MPa resistance
        ),
    )
    prop.TangentialBehavior(formulation=PENALTY, table=((0.2,), ))

    # Abaqus-parity Contact Stabilization Controls
    ctrl = model.ContactControl(
        name="AutoStabCtrl",
        stabilize=AUTOMATIC,
        factor=1.0,
        dampingCoefficient=100.0,
        clearance=0.1,
    )

    model.ContactStd(
        name="AllExteriorGC",
        createStepName="Step-1",
        useAllExterior=True,
        interactionProperty="AbaqusSoftCushion",
        contactControls="AutoStabCtrl",
    )

    solver, sys = model.create_solver3d("Step-1")

    # Boundary conditions
    bot_nids = [sys.node_local_to_global[("BotInst", nid)] for nid in [1, 2, 3, 4]]
    for gid in bot_nids:
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)
        solver.fix_dof(gid, 2, 0.0)

    for gid in range(1, 17):
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)

    # Displace top block down by -0.08 mm (initial gap 0.05 mm -> 0.03 mm penetration)
    top_nids = [sys.node_local_to_global[("TopInst", nid)] for nid in [5, 6, 7, 8]]
    for gid in top_nids:
        solver.fix_dof(gid, 2, -0.08)

    # Solve
    converged, iters = solver.solve_step(dt=1.0, max_iters=30)
    assert converged is True
    assert iters < 20

    gc = next(c for c in solver.constraints if isinstance(c, GeneralContact3D))
    assert gc.last_stats["n_active"] > 0
    assert gc.last_stats["total_normal_force"] > 0.0


def test_adaptive_power_soft_law_math_and_tangent_consistency():
    """Verify mathematical C1-continuity and tangent consistency of AdaptivePowerSoftLaw."""
    from dispsolver.constraint3d.pressure_overclosure import AdaptivePowerSoftLaw

    k_target = 5.0e4
    c0 = 0.002
    delta = 0.02
    power = 2.0

    law = AdaptivePowerSoftLaw(k_target=k_target, c0=c0, delta=delta, power=power)

    # 1. Before activation: h <= -c0
    p, dp = law.evaluate(-0.003)
    assert p == 0.0 and dp == 0.0

    # 2. At activation boundary: h = -c0
    p, dp = law.evaluate(-c0)
    assert p == 0.0 and dp == 0.0

    # 3. At transition boundary: h = delta
    p_d, dp_d = law.evaluate(delta)
    assert np.isclose(dp_d, k_target)

    # 4. In post-transition linear regime: h > delta
    h_deep = delta + 0.01
    p_deep, dp_deep = law.evaluate(h_deep)
    assert np.isclose(dp_deep, k_target)
    assert np.isclose(p_deep, p_d + k_target * 0.01)

    # 5. Tangent Consistency (Derivative check via finite difference) across transition zone
    eps = 1.0e-7
    test_h = np.linspace(-c0 + 0.0005, delta + 0.01, 20)
    for h in test_h:
        p_val, dp_val = law.evaluate(h)
        p_plus, _ = law.evaluate(h + eps)
        dp_fd = (p_plus - p_val) / eps
        rel_err = abs(dp_val - dp_fd) / (k_target + 1.0)
        assert rel_err < 1.0e-4, f"Tangent inconsistency at h={h}: analytical={dp_val}, fd={dp_fd}"


def test_adaptive_soft_contact_auto_derivation_and_solve():
    """Verify end-to-end solve with AdaptiveSoftBehavior (auto-tuned power-law soft contact)."""
    from dispsolver.model import (
        Model,
        ADAPTIVE_SOFT,
        NITSCHE,
        AUTOMATIC,
    )
    from dispsolver.constraint3d.pressure_overclosure import AdaptivePowerSoftLaw

    model = Model(name="AdaptiveModel", dim=3)
    L = 4.0
    gap = 0.02  # 20 um gap

    part_top = model.Part(name="Top")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, gap), (L, 0.0, gap), (L, L, gap), (0.0, L, gap),
        (0.0, 0.0, gap + L), (L, 0.0, gap + L), (L, L, gap + L), (0.0, L, gap + L),
    ]):
        part_top.Node(nid + 1, x, y, z)
    part_top.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    part_bot = model.Part(name="Bot")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, -L), (L, 0.0, -L), (L, L, -L), (0.0, L, -L),
        (0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0),
    ]):
        part_bot.Node(nid + 1, x, y, z)
    part_bot.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    mat = model.Material(name="Steel")
    mat.Elastic(E=100000.0, nu=0.3)
    model.HomogeneousSolidSection(name="Solid", material="Steel")
    part_top.SectionAssignment(region="ALL", sectionName="Solid")
    part_bot.SectionAssignment(region="ALL", sectionName="Solid")

    model.root_assembly.Instance(name="TopI", part=part_top)
    model.root_assembly.Instance(name="BotI", part=part_bot)

    model.Step(name="Step-1", procedure="STATIC")

    # Define ContactProperty using our convenient AdaptiveSoftBehavior!
    # No manual pressure or tabular numbers required!
    prop = model.ContactProperty("AdaptiveProp")
    prop.AdaptiveSoftBehavior(
        lead=0.005,          # Start pre-contact cushioning 5 um before touch
        softnessScale=0.1,   # Relax initial stiffness to 10% of default
        power=2.0,           # Smooth quadratic parabolic lead-in
        allowablePenetration=0.03,  # Smooth transition to target stiffness over 30 um
    )
    prop.TangentialBehavior(formulation="FRICTIONLESS")

    model.ContactStd(
        name="AdaptiveGC",
        createStepName="Step-1",
        useAllExterior=True,
        contactEnforcement="PENALTY",
        interactionProperty="AdaptiveProp",
    )

    solver, sys = model.create_solver3d("Step-1")
    gc = next(c for c in solver.constraints if isinstance(c, GeneralContact3D))

    # Verify law is created as AdaptivePowerSoftLaw
    assert gc.law is not None
    assert isinstance(gc.law, AdaptivePowerSoftLaw)
    assert gc.law.c0 == 0.005
    assert gc.law.delta == 0.03
    assert gc.law.power == 2.0

    # BCs
    for nid in [1, 2, 3, 4]:
        gid = sys.node_local_to_global[("BotI", nid)]
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)
        solver.fix_dof(gid, 2, 0.0)

    for gid in range(1, 17):
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)

    # Displace TopI down by -0.04 mm (closes 0.02 mm gap, penetrates 0.02 mm into transition zone)
    for nid in [5, 6, 7, 8]:
        gid = sys.node_local_to_global[("TopI", nid)]
        solver.fix_dof(gid, 2, -0.04)

    # Solve Newton
    converged, iters = solver.solve_step(dt=1.0, max_iters=30)
    assert converged is True
    assert iters < 20
    assert gc.last_stats["n_active"] > 0
    assert gc.last_stats["total_normal_force"] > 0.0


def test_autonomous_contact_auto_tuning_and_guide():
    """Verify in-situ autonomous relaxation and parameter recommendation guide generation."""
    from dispsolver.model import Model, NITSCHE
    from dispsolver.contact.contact_auto_tuner import ContactAutoTuner

    model = Model(name="AutoTuneTestModel", dim=3)
    L = 3.0
    gap = 0.01

    part_top = model.Part(name="Top")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, gap), (L, 0.0, gap), (L, L, gap), (0.0, L, gap),
        (0.0, 0.0, gap + L), (L, 0.0, gap + L), (L, L, gap + L), (0.0, L, gap + L),
    ]):
        part_top.Node(nid + 1, x, y, z)
    part_top.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    part_bot = model.Part(name="Bot")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, -L), (L, 0.0, -L), (L, L, -L), (0.0, L, -L),
        (0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0),
    ]):
        part_bot.Node(nid + 1, x, y, z)
    part_bot.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    mat = model.Material(name="Mat")
    mat.Elastic(E=50000.0, nu=0.3)
    model.HomogeneousSolidSection(name="Sec", material="Mat")
    part_top.SectionAssignment(region="ALL", sectionName="Sec")
    part_bot.SectionAssignment(region="ALL", sectionName="Sec")

    model.root_assembly.Instance(name="T", part=part_top)
    model.root_assembly.Instance(name="B", part=part_bot)
    model.Step(name="Step-1", procedure="STATIC")

    prop = model.ContactProperty("AdaptiveInitialProp")
    prop.AdaptiveSoftBehavior(lead=0.001, softnessScale=0.5, power=2.0)

    model.ContactStd(
        name="AutoTuneGC",
        createStepName="Step-1",
        useAllExterior=True,
        contactEnforcement="PENALTY",
        interactionProperty="AdaptiveInitialProp",
    )

    solver, sys = model.create_solver3d("Step-1")

    # Boundary conditions
    for nid in [1, 2, 3, 4]:
        gid = sys.node_local_to_global[("B", nid)]
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)
        solver.fix_dof(gid, 2, 0.0)

    for gid in range(1, 17):
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)

    # Displace top down by -0.02 mm
    for nid in [5, 6, 7, 8]:
        gid = sys.node_local_to_global[("T", nid)]
        solver.fix_dof(gid, 2, -0.02)

    # Test ContactAutoTuner direct invocation
    tuner = solver.contact_tuner
    assert isinstance(tuner, ContactAutoTuner)

    # Trigger autonomous relaxation manually once to verify parameter adaptation
    tuned = tuner.tune_and_relax(reason="Artificial test chattering")
    assert tuned is True
    assert tuner.current_lead > 0.001  # Lead expanded
    assert tuner.current_softness < 0.5  # Softness relaxed

    # Solve with auto_tune_contact=True enabled
    converged, iters = solver.solve_step(dt=1.0, max_iters=25, auto_tune_contact=True)
    assert converged is True

    # Check generated recommendation guide
    guide = solver.get_contact_tuning_guide()
    assert guide["total_tunings_performed"] >= 1
    assert "prop.AdaptiveSoftBehavior(" in guide["code_snippet"]
    assert "lead=" in guide["code_snippet"]
    assert "softnessScale=" in guide["code_snippet"]

    # Verify print_guide does not raise
    solver.print_contact_tuning_guide()


def _make_unit_cube_part(model: Model, name: str, offset_z: float = 0.0, L: float = 1.0):
    """Helper to create a single C3D8 unit cube part."""
    part = model.Part(name=name)
    nodes = [
        (0.0, 0.0, offset_z), (L, 0.0, offset_z), (L, L, offset_z), (0.0, L, offset_z),
        (0.0, 0.0, offset_z + L), (L, 0.0, offset_z + L), (L, L, offset_z + L), (0.0, L, offset_z + L),
    ]
    for nid, (x, y, z) in enumerate(nodes):
        part.Node(nid + 1, x, y, z)
    part.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    return part


def test_contact_pair_friction_wiring():
    """Verify ContactPair properly wires friction and tangential stiffness ratio."""
    model = Model(name="FrictionWiringModel")
    part_top = _make_unit_cube_part(model, "TopPart", offset_z=0.0)
    part_bot = _make_unit_cube_part(model, "BotPart", offset_z=-1.0)
    mat = model.Material("Elastic")
    mat.Elastic(table=((1000.0, 0.3), ))
    model.HomogeneousSolidSection(name="Sec", material="Elastic")
    part_top.SectionAssignment(region="ALL", sectionName="Sec")
    part_bot.SectionAssignment(region="ALL", sectionName="Sec")
    model.root_assembly.Instance(name="T", part=part_top)
    model.root_assembly.Instance(name="B", part=part_bot)
    model.Step(name="Step-1", procedure="STATIC")

    prop = model.ContactProperty("FrictionProp")
    prop.NormalBehavior(pressureOverclosure="HARD")
    prop.TangentialBehavior(formulation="PENALTY", table=((0.35,), ))

    # Create deformable contact pair
    model.ContactPair(
        name="CP-1",
        createStepName="Step-1",
        slave="T",
        master="B",
        interactionProperty="FrictionProp",
    )

    solver, sys = model.create_solver3d("Step-1")
    assert len(solver.constraints) >= 1
    c = solver.constraints[0]
    assert hasattr(c, "mu")
    assert abs(c.mu - 0.35) < 1e-12
    assert hasattr(c, "k_t")
    assert c.k_t > 0.0


def test_contact_tangent_stabilization_damping_fraction():
    """Verify tangentFraction damping ratio in ContactProperty and ContactControl."""
    model = Model(name="TangentStabModel")
    part_top = _make_unit_cube_part(model, "TopPart", offset_z=0.05)
    part_bot = _make_unit_cube_part(model, "BotPart", offset_z=-1.0)
    mat = model.Material("Elastic")
    mat.Elastic(table=((1000.0, 0.3), ))
    model.HomogeneousSolidSection(name="Sec", material="Elastic")
    part_top.SectionAssignment(region="ALL", sectionName="Sec")
    part_bot.SectionAssignment(region="ALL", sectionName="Sec")
    model.root_assembly.Instance(name="T", part=part_top)
    model.root_assembly.Instance(name="B", part=part_bot)
    model.Step(name="Step-1", procedure="STATIC")

    prop = model.ContactProperty("StabProp")
    prop.NormalBehavior(pressureOverclosure="HARD")
    prop.Damping(definition="DAMPING_COEFFICIENT", dampingCoefficient=50.0, clearance=0.1, tangentFraction=0.2)

    ctrl = model.ContactControl(name="Ctrl-1", tangentFraction=0.4, dampingCoefficient=80.0, clearance=0.1)

    model.ContactStd(
        name="GC-1",
        createStepName="Step-1",
        useAllExterior=True,
        interactionProperty="StabProp",
        contactControls="Ctrl-1",
    )

    solver, sys = model.create_solver3d("Step-1")
    gc = solver.constraints[0]
    assert hasattr(gc, "tangent_fraction")
    assert abs(gc.tangent_fraction - 0.4) < 1e-12
    assert abs(gc.damping_coeff - 80.0) < 1e-12


def test_general_contact_coulomb_friction_stick_slip_solve():
    """Verify GeneralContact3D Coulomb friction stick-slip behavior in multi-step solve."""
    model = Model(name="GCFrictionModel")
    part_top = _make_unit_cube_part(model, "TopPart", offset_z=0.0)
    part_bot = _make_unit_cube_part(model, "BotPart", offset_z=-1.0)
    mat = model.Material("Elastic")
    mat.Elastic(table=((10000.0, 0.25), ))
    model.HomogeneousSolidSection(name="Sec", material="Elastic")
    part_top.SectionAssignment(region="ALL", sectionName="Sec")
    part_bot.SectionAssignment(region="ALL", sectionName="Sec")
    model.root_assembly.Instance(name="T", part=part_top)
    model.root_assembly.Instance(name="B", part=part_bot)
    model.Step(name="Step-1", procedure="STATIC")

    mu_val = 0.3
    prop = model.ContactProperty("FricProp")
    prop.NormalBehavior(pressureOverclosure="HARD")
    prop.TangentialBehavior(formulation="PENALTY", table=((mu_val,), ))

    k_rep = 10000.0 * 1.0  # E * L representative scale
    model.ContactStd(
        name="GC-Fric",
        createStepName="Step-1",
        useAllExterior=True,
        interactionProperty="FricProp",
        gamma=k_rep,
    )

    solver, sys = model.create_solver3d("Step-1")
    gc = solver.constraints[0]
    assert abs(gc.mu - mu_val) < 1e-12

    # Fix bottom cube completely
    bot_nids = [sys.node_local_to_global[("B", nid)] for nid in [1, 2, 3, 4]]
    for gid in bot_nids:
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 1, 0.0)
        solver.fix_dof(gid, 2, 0.0)

    # Fix lateral y dofs for all nodes
    for gid in range(1, 17):
        solver.fix_dof(gid, 1, 0.0)

    # Step 1: Compress top cube down in z-direction by -0.01 mm (normal contact active, no sliding)
    top_surf_nids = [sys.node_local_to_global[("T", nid)] for nid in [5, 6, 7, 8]]
    for gid in top_surf_nids:
        solver.fix_dof(gid, 0, 0.0)
        solver.fix_dof(gid, 2, -0.01)

    converged, iters = solver.solve_step(dt=1.0, max_iters=25)
    assert converged is True
    # During normal compression with zero lateral motion, stick holds and no slipping
    slip_set = gc.get_friction_state(solver.u)
    assert len(slip_set) == 0

    # Step 2: Impose small lateral displacement (stick condition)
    u_stick = 0.0005
    for gid in top_surf_nids:
        solver.fix_dof(gid, 0, u_stick)
        solver.fix_dof(gid, 2, -0.01)

    converged, iters = solver.solve_step(dt=1.0, max_iters=25)
    assert converged is True
    assert len(gc._anchors) > 0

    # Step 3: Impose lateral sliding (slip condition)
    u_slip = 0.008
    for gid in top_surf_nids:
        solver.fix_dof(gid, 0, u_slip)
        solver.fix_dof(gid, 2, -0.01)

    converged, iters = solver.solve_step(dt=1.0, max_iters=35)
    assert converged is True

    # Check friction force contribution
    f_contact, _, stats = gc.assemble(solver.u)
    assert stats["total_normal_force"] > 0.0
    # Horizontal resisting contact force on bottom nodes must be non-zero and oppose sliding
    bot_contact_nids = [sys.node_local_to_global[("B", nid)] for nid in [5, 6, 7, 8]]
    f_x_total = sum(f_contact[3 * sys.nid_to_idx[nid] + 0] for nid in bot_contact_nids)
    assert abs(f_x_total) > 0.0

    # Verify commit_friction_state preserves valid anchor coordinates
    gc.commit_friction_state(solver.u)
    assert len(gc._anchors) == 4
    for key, anchor_coords in gc._anchors.items():
        assert len(anchor_coords) == 2
        assert -1.5 <= anchor_coords[0] <= 1.5
        assert -1.5 <= anchor_coords[1] <= 1.5


def test_general_contact_harmonic_modulus_scaling():
    """Verify Stage 3: Material-adaptive harmonic mean interface compliance scaling.
    When a stiff body (E1=4000 MPa, PET) contacts a soft body (E2=1 MPa, PSA),
    the contact pair stiffness scales according to E_eff = 2*E1*E2/(E1+E2) ~= 1.999 MPa,
    preventing artificial ill-conditioning and locking.
    """
    from dispsolver.contact.general_contact3d import GeneralContact3D

    # Setup two interacting Quad4 faces: Face 0 on Body 1, Face 1 on Body 2
    # Face 0 (Z=0, normal +Z): Body 1 (PET, pid=1, E=4000 MPa)
    # Face 1 (Z=0.001, normal -Z): Body 2 (PSA, pid=2, E=1 MPa)
    coords = np.array([
        # Face 0 nodes (0, 1, 2, 3)
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [-0.5,  0.5, 0.0],
        # Face 1 nodes (4, 5, 6, 7) - oriented opposing (+Z vs -Z)
        [-0.5, -0.5, 0.001],
        [-0.5,  0.5, 0.001],
        [ 0.5,  0.5, 0.001],
        [ 0.5, -0.5, 0.001],
    ], dtype=np.float64)

    nid_to_idx = {i + 1: i for i in range(8)}
    faces = [
        (1, 2, 3, 4),  # Face 0 (fa_idx=0)
        (5, 6, 7, 8),  # Face 1 (fb_idx=1)
    ]
    face_to_elem = {0: 101, 1: 102}
    elem_to_pid = {101: 1, 102: 2}
    materials = {
        1: {"E": 4000.0, "nu": 0.3},  # PET
        2: {"E": 1.0, "nu": 0.499},   # PSA
    }

    gc = GeneralContact3D(
        faces=faces,
        coords=coords,
        nid_to_idx=nid_to_idx,
        gamma=1.0e6,
        face_to_elem=face_to_elem,
        elem_to_pid=elem_to_pid,
        materials=materials,
    )

    # Face effective moduli
    assert gc._face_E[0] == 4000.0
    assert gc._face_E[1] == 1.0

    # Calculate analytical harmonic mean modulus
    E1, E2 = 4000.0, 1.0
    E_eff_expected = 2.0 * E1 * E2 / (E1 + E2)
    assert abs(E_eff_expected - 1.9995) < 1e-3

    # Displace Face 1 into Face 0 by -0.005 mm -> penetration = 0.004 mm
    u = np.zeros(24, dtype=np.float64)
    for n in range(4, 8):
        u[3 * n + 2] = -0.005

    f_contact, _, stats = gc.assemble(u)
    assert stats["n_active"] > 0

    # Total normal force should be proportional to E_eff rather than stiff E1 (4000)
    # mod_scale = E_eff / E_ref (where E_ref is median(4000, 1) = 2000.5)
    mod_scale = E_eff_expected / gc.E_ref
    assert abs(mod_scale - (1.9995 / 2000.5)) < 1e-4

    # Verify contact force matches expected scaled magnitude
    # Total contact force for 1x1 area with penetration ~0.004 mm
    expected_force = gc.gamma * mod_scale * 0.004 * 1.0  # 4.0 * 1.9995 / 2000.5 ~= 0.003998
    assert abs(stats["total_normal_force"] - expected_force) < 0.001


def test_general_contact_untied_exterior_auto_gathering():
    """Verify ContactStd(useAllExterior=True) automatically excludes tied/bonded faces."""
    from dispsolver.model.model import Model
    from dispsolver.contact.general_contact3d import GeneralContact3D

    model = Model(name="TiedCubesModel", dim=3)
    L = 5.0

    # Part 1: Top Cube (z in [0, L])
    part_top = model.Part(name="PartTop")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0),
        (0.0, 0.0, L), (L, 0.0, L), (L, L, L), (0.0, L, L),
    ]):
        part_top.Node(nid + 1, x, y, z)
    part_top.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    # Part 2: Bottom Cube (z in [-L, 0])
    part_bot = model.Part(name="PartBot")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, -L), (L, 0.0, -L), (L, L, -L), (0.0, L, -L),
        (0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0),
    ]):
        part_bot.Node(nid + 1, x, y, z)
    part_bot.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    mat = model.Material(name="Steel")
    mat.Elastic(E=200000.0, nu=0.3)
    model.HomogeneousSolidSection(name="SolidSec", material="Steel")
    part_top.SectionAssignment(region="ALL", sectionName="SolidSec")
    part_bot.SectionAssignment(region="ALL", sectionName="SolidSec")

    inst_top = model.root_assembly.Instance(name="Top", part=part_top)
    inst_bot = model.root_assembly.Instance(name="Bot", part=part_bot)

    # Tie bottom of Top (nodes 1,2,3,4 at z=0) to top of Bot (nodes 5,6,7,8 at z=0)
    top_bot_nids = [inst_top.part.nodes[nid] for nid in [1, 2, 3, 4]]
    bot_top_nids = [inst_bot.part.nodes[nid] for nid in [5, 6, 7, 8]]
    model.Tie(
        name="TiedInterface",
        master="Top",
        slave="Bot",
    )

    model.Step(name="Step-1", procedure="STATIC", time_period=1.0)
    # Zero-configuration General Contact
    model.ContactStd(
        name="AllExteriorContact",
        createStepName="Step-1",
        useAllExterior=True,
    )

    solver, sys = model.create_solver3d(step_name="Step-1")
    gc = next(c for c in solver.constraints if isinstance(c, GeneralContact3D))

    # Two 8-node hex cubes have 6 faces each = 12 exterior faces.
    # The tied interface at z=0 contains 2 faces (1 on Top bottom, 1 on Bot top).
    # Zero-config useAllExterior=True MUST exclude the tied interface faces!
    # Kept untied faces should be exactly 12 - 2 = 10 faces!
    assert len(gc.faces) == 10, f"Expected 10 untied exterior faces, got {len(gc.faces)}"


def test_general_contact_excluded_pairs():
    """Verify ContactStd excludedPairs option suppresses interaction between specified pairs."""
    from dispsolver.model.model import Model
    from dispsolver.contact.general_contact3d import GeneralContact3D

    model = Model(name="ExcludedPairsModel", dim=3)
    L = 5.0
    gap = 0.1

    part_top = model.Part(name="PartTop")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, gap), (L, 0.0, gap), (L, L, gap), (0.0, L, gap),
        (0.0, 0.0, gap + L), (L, 0.0, gap + L), (L, L, gap + L), (0.0, L, gap + L),
    ]):
        part_top.Node(nid + 1, x, y, z)
    part_top.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    part_bot = model.Part(name="PartBot")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, -L), (L, 0.0, -L), (L, L, -L), (0.0, L, -L),
        (0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, L, 0.0), (0.0, L, 0.0),
    ]):
        part_bot.Node(nid + 1, x, y, z)
    part_bot.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    mat = model.Material(name="Steel")
    mat.Elastic(E=200000.0, nu=0.3)
    model.HomogeneousSolidSection(name="SolidSec", material="Steel")
    part_top.SectionAssignment(region="ALL", sectionName="SolidSec")
    part_bot.SectionAssignment(region="ALL", sectionName="SolidSec")

    model.root_assembly.Instance(name="Top", part=part_top)
    model.root_assembly.Instance(name="Bot", part=part_bot)

    model.Step(name="Step-1", procedure="STATIC", time_period=1.0)
    # Explicitly EXCLUDE Top vs Bot contact interaction
    model.ContactStd(
        name="ExcludeContact",
        createStepName="Step-1",
        useAllExterior=True,
        excludedPairs=[("Top", "Bot")],
        gamma=200000.0 * L,
    )

    solver, sys = model.create_solver3d(step_name="Step-1")
    gc = next(c for c in solver.constraints if isinstance(c, GeneralContact3D))

    # Candidate pairs between Top and Bot should be completely filtered out!
    u = np.zeros(len(solver.u), dtype=np.float64)
    # Push top down into bottom
    for nid in [5, 6, 7, 8]:
        gid = sys.node_local_to_global[("Top", nid)]
        u[3 * sys.nid_to_idx[gid] + 2] = -0.3

    f_contact, _, stats = gc.assemble(u)
    # Because Top vs Bot is excluded, active contact points should be 0!
    assert stats["n_active"] == 0, "Contact between Top and Bot was not excluded!"
    assert stats["total_normal_force"] == 0.0


def test_general_contact_enforcements_and_normal_behaviors():
    """Verify GeneralContact enforcement routing and normal behaviors."""
    from dispsolver.model.model import Model
    from dispsolver.contact.general_contact3d import GeneralContact3D
    from dispsolver.constraint3d.pressure_overclosure import (
        HardLaw, ExponentialSoftLaw, AdaptivePowerSoftLaw, LinearSoftLaw, TabularSoftLaw
    )

    model = Model(name="EnforcementModel", dim=3)
    part = model.Part(name="Part1")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]):
        part.Node(nid + 1, x, y, z)
    part.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    model.root_assembly.Instance(name="Inst1", part=part)
    model.Step(name="Step-1")

    # 1. NITSCHE + ADAPTIVE_SOFT
    # GeneralContact3D implements penalty only: NITSCHE never ran as requested -> refused.
    import pytest
    gc1 = model.ContactStd(
        name="GC_Nitsche_Adaptive",
        createStepName="Step-1",
        contactEnforcement="NITSCHE",
        normalBehavior="ADAPTIVE_SOFT",
    )
    with pytest.raises(NotImplementedError, match="NITSCHE"):
        model.create_solver3d("Step-1")
    # the ADAPTIVE_SOFT law is still checked, under the enforcement that is implemented
    model_p = Model(name="EnforcementModelP", dim=3)
    part_p = model_p.Part(name="Part1")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]):
        part_p.Node(nid + 1, x, y, z)
    part_p.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    model_p.root_assembly.Instance(name="Inst1", part=part_p)
    model_p.Step(name="Step-1")
    model_p.ContactStd(name="GC_Penalty_Adaptive", createStepName="Step-1",
                       contactEnforcement="PENALTY", normalBehavior="ADAPTIVE_SOFT")
    solver1, sys1 = model_p.create_solver3d("Step-1")
    c1 = next(c for c in solver1.constraints if isinstance(c, GeneralContact3D))
    assert c1.contact_enforcement == "PENALTY"
    assert isinstance(c1.law, AdaptivePowerSoftLaw)

    # 2. PENALTY + EXPONENTIAL
    model2 = Model(name="Model2", dim=3)
    p2 = model2.Part(name="P2")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]):
        p2.Node(nid + 1, x, y, z)
    p2.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    model2.root_assembly.Instance(name="Inst2", part=p2)
    model2.Step(name="Step-1")
    gc2 = model2.ContactStd(
        name="GC_Penalty_Exp",
        createStepName="Step-1",
        contactEnforcement="PENALTY",
        normalBehavior="EXPONENTIAL",
    )
    solver2, _ = model2.create_solver3d("Step-1")
    c2 = next(c for c in solver2.constraints if isinstance(c, GeneralContact3D))
    assert c2.contact_enforcement == "PENALTY"
    assert isinstance(c2.law, ExponentialSoftLaw)

    # 3. AUGMENTED_LAGRANGE
    model3 = Model(name="Model3", dim=3)
    p3 = model3.Part(name="P3")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]):
        p3.Node(nid + 1, x, y, z)
    p3.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    model3.root_assembly.Instance(name="Inst3", part=p3)
    model3.Step(name="Step-1")
    gc3 = model3.ContactStd(
        name="GC_AL",
        createStepName="Step-1",
        contactEnforcement="AUGMENTED_LAGRANGE",
    )
    # GeneralContact3D never read augmented_lagrange -> AL for general contact is refused.
    with pytest.raises(NotImplementedError, match="AUGMENTED_LAGRANGE"):
        model3.create_solver3d("Step-1")


def test_contact_control_penetration_tolerance():
    """Verify ContactControl penetrationTolerance and approach API."""
    model = Model(name="TestModelPenTol", dim=3)
    p = model.Part(name="P1")
    for nid, (x, y, z) in enumerate([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]):
        p.Node(nid + 1, x, y, z)
    p.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    model.root_assembly.Instance(name="Inst1", part=p)
    model.Step(name="Step-1")

    ctrl = model.ContactControl(
        name="PenCtrl",
        penetrationTolerance=0.08,
        approach=True,
    )
    assert ctrl.penetration_tolerance == 0.08
    assert ctrl.approach is True

    model.ContactStd(
        name="GC_PenTol",
        createStepName="Step-1",
        normalBehavior="ADAPTIVE",
        contactControls="PenCtrl",
    )
    solver, _ = model.create_solver3d("Step-1")
    gc = next(c for c in solver.constraints if isinstance(c, GeneralContact3D))
    assert gc.law is not None
    assert gc.law.delta == 0.08
    # Quantitative mechanics check: at h=0.04 (within 0.08), contact stiffness is buffered
    p, dp_dh = gc.law.evaluate(0.04)
    assert p > 0.0
    assert dp_dh > 0.0
    assert dp_dh < gc.law.k_target








