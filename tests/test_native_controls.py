"""Native control invariants, without opening a window or requiring a running daemon."""
from pathlib import Path
from unittest.mock import patch
import json
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.body_server import Body, SCENES, World, pose_table, wheel_dofs
from mjlab_microduck.sim.viewer_controls import DriveKeys, NativeControls


def test_release_focus_and_space_stop_without_keyboard_repeat():
    keys = DriveKeys()
    keys.key("up", True)
    keys.key("left", True)
    assert keys.command() == (0.3, 0.0, 0.6)
    keys.key("up", False)
    assert keys.command() == (0.0, 0.0, 0.6)
    keys.focus(False)
    assert keys.command() == (0.0, 0.0, 0.0)
    keys.focus(True)
    assert keys.command() == (0.0, 0.0, 0.0)  # Focusing never resumes old input.
    keys.key("down", True)
    keys.key("stop", True)
    assert keys.command() == (0.0, 0.0, 0.0)


def test_opposite_keys_cancel_and_reset_blocks_new_drive_input():
    keys = DriveKeys()
    keys.key("up", True)
    keys.key("down", True)
    assert keys.command() == (0.0, 0.0, 0.0)
    keys.key("reset", True)
    keys.key("up", True)
    assert keys.begin_reset()
    keys.key("right", True)
    assert keys.command() == (0.0, 0.0, 0.0)
    keys.end_reset()
    assert keys.command() == (0.0, 0.0, 0.0)
    assert not keys.begin_reset()


@pytest.mark.parametrize("scene_name", ["scene.xml", "scene_rollers.xml"])
def test_reset_clears_passive_motion_and_holds_fresh_home_until_policy(scene_name):
    world = World(SCENES / scene_name)
    body = Body(world, 0)
    world.bodies.append(body)
    body.place(*pose_table(SCENES / scene_name, "HOME"), offset_y=0.0)
    world.data.time = 17.0
    world.data.qpos[body.trunk:body.trunk + 3] = [1.0, -0.5, 0.05]
    world.data.qvel[:] = 2.0
    body.released = True
    with pytest.raises(RuntimeError, match="relax"):
        world.reset_home()  # A teleport must not race a still-running controller.
    body.set_torque(False)
    world.reset_home()
    assert world.data.time == 17.0
    assert np.allclose(world.data.qvel, 0.0)
    assert np.allclose(world.data.qpos[:2], 0.0)
    assert world.data.qpos[body.trunk + 2] > 0.12
    assert not body.released
    assert not body.torque_on
    assert np.allclose(world.data.qvel[wheel_dofs(world.model)], 0.0)
    body.set_torque(True)
    home = body.sensors()["positions"]
    body.set_targets(home)
    assert not body.released  # HOME_RAMP still has no balancing feedback.
    home[0] += 0.01
    body.set_targets(home)
    assert body.released
    mujoco.mj_forward(world.model, world.data)
    assert body.sensors()["imu"]["gravity"][2] == -1.0


def test_reset_waits_for_real_torque_off_and_clears_controller_episode():
    world = World(SCENES / "scene.xml")
    body = Body(world, 0)
    world.bodies.append(body)
    body.place(*pose_table(SCENES / "scene.xml", "HOME"), offset_y=0.0)
    body.released = True
    controls = NativeControls(Path("/unused.sock"), world)
    calls = []

    def daemon(path, method, params=None):
        calls.append((method, params))
        if method == "robot.relax":
            body.set_torque(False)
        if method == "robot.enable" and params["on"]:
            assert not body.released
            assert world.data.qpos[body.trunk + 2] > 0.12
            body.set_torque(True)
            target = body.sensors()["positions"]
            target[0] += 0.01
            body.set_targets(target)
        return {"accepted": True}

    with patch("mjlab_microduck.sim.viewer_controls.rpc", side_effect=daemon):
        controls._reset()
    methods = [method for method, _ in calls]
    assert methods[:3] == ["robot.stop", "robot.enable", "robot.relax"]
    assert calls[1][1] == {"on": False}
    assert calls[-1] == ("robot.enable", {"on": True})
    assert body.released


def test_refused_reset_never_teleports_the_body():
    world = World(SCENES / "scene.xml")
    body = Body(world, 0)
    world.bodies.append(body)
    body.place(*pose_table(SCENES / "scene.xml", "HOME"), offset_y=0.0)
    world.data.qpos[body.trunk] = 0.5
    controls = NativeControls(Path("/unused.sock"), world)
    with patch("mjlab_microduck.sim.viewer_controls.rpc", side_effect=RuntimeError("offline")):
        with pytest.raises(RuntimeError, match="offline"):
            controls._reset()
    assert world.data.qpos[body.trunk] == 0.5


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS application identity")
def test_macos_bundle_preserves_venv_packages_without_manual_dylib_link(tmp_path):
    from mjlab_microduck.sim.macos_app import create_app
    binary = create_app(tmp_path)
    result = subprocess.check_output([
        str(binary), "-c", "import json, mujoco, mjlab_microduck; "
        "print(json.dumps({'mujoco': mujoco.__version__}))",
    ], text=True)
    assert json.loads(result)["mujoco"] == mujoco.__version__
    assert create_app(tmp_path) == binary  # Re-running keeps the bundle usable.
