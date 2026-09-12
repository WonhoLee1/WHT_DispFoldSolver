"""Unit tests for Constraint (RigidBody, Tie, Coupling, MPC) and Load (ConcentratedForce, Pressure, Gravity) modules."""

import pytest
import numpy as np

from dispsolver.model.model import Model
from dispsolver.model.amplitude import SmoothStepAmplitude
from dispsolver.model.constraint import RigidBody, Tie, Coupling, KinematicCoupling, DistributingCoupling, MPC
from dispsolver.model.load import ConcentratedForce, Pressure, Gravity, BodyForce


def test_rigid_body_constraint_resolution():
    """Test RigidBody constraint declaration and associative slave node extraction."""
    model = Model(name="TestModel", dim=3)
    p = model.Part("PlatePart", dim=3)
    
    # Create a 2x2 grid of nodes
    p.add_node(1, [0.0, 0.0, 0.0])
    p.add_node(2, [1.0, 0.0, 0.0])
    p.add_node(3, [1.0, 1.0, 0.0])
    p.add_node(4, [0.0, 1.0, 0.0])
    
    # Reference Point (RP) at centroid
    p.add_node(100, [0.5, 0.5, 0.0])
    rp_set = p.create_set_from_box("RP_SET", x_range=(0.4, 0.6), y_range=(0.4, 0.6), z_range=(-0.1, 0.1), entity_type="NODES")
    plate_set = p.create_set_from_box("PLATE_NODES", x_range=(-0.1, 1.1), y_range=(-0.1, 1.1), z_range=(-0.1, 0.1), entity_type="NODES")
    
    rb = model.RigidBody(name="RB_Plate", refPoint=rp_set, tieNset=plate_set)
    assert rb.name == "RB_Plate"
    slaves = rb.get_slave_node_ids()
    # Slaves should include nodes 1, 2, 3, 4, 100
    assert 1 in slaves and 2 in slaves and 3 in slaves and 4 in slaves


def test_tie_constraint_declaration():
    """Test Tie constraint declaration and properties."""
    model = Model(name="TestModel")
    tie = model.Tie(
        name="Tie_Plate_Display",
        master="PlateInstance.TOP_SURF",
        slave="DispInstance.BOT_SURF",
        positionTolerance=0.02,
        adjust=True,
        tieRotations=False
    )
    assert tie.name == "Tie_Plate_Display"
    assert tie.position_tolerance == 0.02
    assert tie.adjust is True
    assert tie.tie_rotations is False
    assert tie.constraint_enforcement == "SURFACE_TO_SURFACE"


def test_coupling_kinematic_and_distributing():
    """Test Kinematic and Distributing Coupling declarations."""
    model = Model(name="TestModel")
    
    kc = model.KinematicCoupling(name="KC1", refPoint="RP", surface="SURF1", u1=True, u2=True, u3=True)
    assert kc.coupling_type == "KINEMATIC"
    assert kc.u1 is True and kc.u2 is True and kc.u3 is True
    
    dc = model.DistributingCoupling(
        name="DC1", refPoint="RP", surface="SURF1",
        weightingMethod="LINEAR", influenceRadius=5.0
    )
    assert dc.coupling_type == "DISTRIBUTING"
    assert dc.weighting_method == "LINEAR"
    assert dc.influence_radius == 5.0


def test_mpc_constraints():
    """Test Multi-Point Constraint (PIN, BEAM, EQUATION)."""
    model = Model(name="TestModel")
    
    mpc_pin = model.MPC(name="PinJoint", mpcType="PIN", firstPoint=10, secondPoint=20)
    assert mpc_pin.mpc_type == "PIN"
    assert mpc_pin.first_point == 10
    assert mpc_pin.second_point == 20
    
    mpc_eq = model.MPC(
        name="PeriodicPBC",
        mpcType="EQUATION",
        firstPoint=1,
        secondPoint=2,
        coefficients=[1.0, -1.0],
        constant=0.0
    )
    assert mpc_eq.mpc_type == "EQUATION"
    assert mpc_eq.coefficients == [1.0, -1.0]
    assert mpc_eq.constant == 0.0


def test_concentrated_force_with_amplitude():
    """Test ConcentratedForce load with SmoothStepAmplitude."""
    model = Model(name="TestModel")
    amp = model.SmoothStepAmplitude(name="SmoothRamp", data=[(0.0, 0.0), (1.0, 1.0)])
    
    cload = model.ConcentratedForce(
        name="TipForce",
        createStepName="Initial",
        region="TIP_NODES",
        cf2=-1000.0,
        amplitude="SmoothRamp"
    )
    
    # At t=0.0, force should be 0.0
    f_t0 = cload.get_force_vector(t=0.0, step_time=0.0, model=model)
    assert np.allclose(f_t0, [0.0, 0.0, 0.0])
    
    # At t=0.5, smootherstep(0.5) = 0.5 -> cf2 = -500.0
    f_t05 = cload.get_force_vector(t=0.5, step_time=0.5, model=model)
    assert np.isclose(f_t05[1], -500.0, atol=1e-5)
    
    # At t=1.0, force should be -1000.0
    f_t1 = cload.get_force_vector(t=1.0, step_time=1.0, model=model)
    assert np.isclose(f_t1[1], -1000.0, atol=1e-5)


def test_pressure_and_gravity_loads():
    """Test Pressure and Gravity loads."""
    model = Model(name="TestModel")
    
    press = model.Pressure(
        name="UniformPress",
        createStepName="Initial",
        region="TOP_FACE",
        magnitude=2.5
    )
    assert np.isclose(press.get_magnitude(t=0.5, model=model), 2.5)
    
    grav = model.Gravity(
        name="EarthGravity",
        createStepName="Initial",
        comp3=-9810.0
    )
    g_vec = grav.get_acceleration_vector(t=0.0, model=model)
    assert np.allclose(g_vec, [0.0, 0.0, -9810.0])


def test_model_solver3d_concentrated_force_solve():
    """Verify that ConcentratedForce directly drives non-linear deformation in DynamicSolver3D.solve."""
    model = Model(name="CantileverForceTest", dim=3)
    p = model.Part("BarPart", dim=3)

    # 1 single hex element of size 1.0 x 1.0 x 1.0
    p.add_node(1, [0.0, 0.0, 0.0])
    p.add_node(2, [1.0, 0.0, 0.0])
    p.add_node(3, [1.0, 1.0, 0.0])
    p.add_node(4, [0.0, 1.0, 0.0])
    p.add_node(5, [0.0, 0.0, 1.0])
    p.add_node(6, [1.0, 0.0, 1.0])
    p.add_node(7, [1.0, 1.0, 1.0])
    p.add_node(8, [0.0, 1.0, 1.0])
    p.add_element(1, "C3D8_COROTATIONAL", [1, 2, 3, 4, 5, 6, 7, 8])

    fixed_set = p.create_set_from_box("FIXED_FACE", x_range=(-0.1, 0.1), y_range=(-0.1, 1.1), z_range=(-0.1, 1.1), entity_type="NODES")
    tip_set = p.create_set_from_box("TIP_FACE", x_range=(0.9, 1.1), y_range=(-0.1, 1.1), z_range=(-0.1, 1.1), entity_type="NODES")

    mat = model.Material(name="Steel", mat_type="ELASTIC")
    mat.E = 210000.0
    mat.nu = 0.3
    sec = model.SolidSection(name="SolidSec", material="Steel")
    p.SectionAssignment(region=p.create_set_from_box("ALL_ELEMS", (-0.1, 1.1), (-0.1, 1.1), (-0.1, 1.1), "ELEMENTS"), sectionName="SolidSec")

    model.root_assembly.Instance(name="BAR_INST", part=p)

    step = model.Step(name="Step-1", time_period=0.1, dt_init=0.05, dt_min=1e-5, dt_max=0.05)
    step.parent_model = model

    # Clamp left face
    model.DisplacementBC(name="Clamp", createStepName="Step-1", region="BAR_INST.FIXED_FACE", u1=0.0, u2=0.0, u3=0.0)

    # Apply downward force on tip face
    amp = model.SmoothStepAmplitude(name="Ramp", data=[(0.0, 0.0), (0.1, 1.0)])
    model.ConcentratedForce(name="DownForce", createStepName="Step-1", region="BAR_INST.TIP_FACE", cf2=-5000.0, amplitude="Ramp")

    solver, sys = model.create_solver3d(step_name="Step-1")
    converged = solver.solve(step=step, sys=sys, verbose=False)
    assert converged is True

    # Tip nodes (2, 3, 6, 7) must exhibit negative Y-displacement
    tip_indices = sys.global_nsets["BAR_INST.TIP_FACE"]
    uy_vals = [solver.u[3 * idx + 1] for idx in tip_indices]
    assert len(uy_vals) == 4
    for uy in uy_vals:
        assert uy < -1e-6  # Deflected downward under force

