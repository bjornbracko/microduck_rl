"""The daemon twin must start on skates without falling during its home ramp."""
import mujoco
import numpy as np

from mjlab_microduck.sim.body_server import (
    Body, HOME_TRUNK_Z, JOINT_NAMES, ROLLER_HEIGHT, SCENES, WHEEL_FRICTION,
    World, pose_table, wheel_dofs,
)


def test_roller_home_clears_the_floor_and_maps_only_servo_joints():
    scene = SCENES / "scene_rollers.xml"
    world = World(scene)
    body = Body(world, 0)
    pose, height = pose_table(scene, "HOME")
    body.place(pose, height, 0.0)
    mujoco.mj_forward(world.model, world.data)
    assert height == HOME_TRUNK_Z + ROLLER_HEIGHT
    assert len(wheel_dofs(world.model)) == 4
    assert len(body.actuators) == 14
    assert len(body.sensors()["positions"]) == len(JOINT_NAMES)
    # The unadjusted STAND keyframe penetrated the floor by more than 20 mm.
    assert min((c.dist for c in world.data.contact), default=0.0) > -0.005
    assert np.allclose(world.model.dof_frictionloss[wheel_dofs(world.model)], WHEEL_FRICTION)


def test_roller_is_steadied_until_the_policy_takes_over():
    scene = SCENES / "scene_rollers.xml"
    world = World(scene)
    body = Body(world, 0)
    body.place(*pose_table(scene, "HOME"), offset_y=0.0)
    body.set_torque(True)
    home = body.sensors()["positions"]
    body.set_targets(home)
    assert not body.released  # A torque-on/home ramp alone does not provide balance.
    target = home.copy()
    target[0] += 0.01
    body.set_targets(target)
    assert body.released


def test_walking_startup_and_explicit_limp_are_unchanged():
    scene = SCENES / "scene.xml"
    world = World(scene)
    body = Body(world, 0)
    body.place(*pose_table(scene, "HOME"), offset_y=0.0)
    assert pose_table(scene, "HOME")[1] == HOME_TRUNK_Z
    assert not wheel_dofs(world.model)
    body.set_torque(True)
    assert body.released
    roller = Body(World(SCENES / "scene_rollers.xml"), 0, limp=True)
    assert roller.released


def test_roller_targets_follow_names_and_torque_can_be_disabled():
    world = World(SCENES / "scene_rollers.xml")
    body = Body(world, 0)
    body.place(*pose_table(SCENES / "scene_rollers.xml", "HOME"), offset_y=0.0)
    targets = np.linspace(-0.1, 0.1, len(JOINT_NAMES))
    body.set_targets(targets.tolist())
    # Wheel joints interleave the servos, and the wire also contains a mouth.
    # Neither may shift which BAM motor receives a command.
    for name, value in zip(JOINT_NAMES, targets):
        if name in world.bam.dof_to_q_target:
            assert world.bam.q_target[world.bam.dof_to_q_target[name]] == value
    body.set_gain(150)
    assert world.bam.model.actuator.kp == 150
    body.set_torque(False)
    world.step(4)
    assert np.allclose(world.data.actuator_force[body.actuator_slice], 0.0)
