"""A native MuJoCo window with real key-up and focus events, plus a compact HUD."""
from __future__ import annotations

import glfw
import mujoco


class NativeViewer:
    def __init__(self, world, controls):
        self.world = world
        self.controls = controls
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize the native viewer")
        # Head-camera renderers create invisible windows and leave those GLFW hints set.
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(1280, 800, "Microduck - MuJoCo", None, None)
        if not self.window:
            raise RuntimeError("GLFW could not create the native viewer")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.scene = mujoco.MjvScene(world.model, maxgeom=10000)
        self.context = mujoco.MjrContext(world.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 0.75
        self.camera.azimuth = 135.0
        self.camera.elevation = -22.0
        self._cursor = glfw.get_cursor_pos(self.window)
        self.mode = "ROLLER" if controls.world.bam is not None else "WALK"
        glfw.set_key_callback(self.window, self._key)
        glfw.set_window_focus_callback(self.window, lambda _, focused: controls.keys.focus(bool(focused)))
        glfw.set_window_close_callback(self.window, lambda _: controls.keys.focus(False))
        glfw.set_cursor_pos_callback(self.window, self._mouse)
        glfw.set_scroll_callback(self.window, self._scroll)
        controls.keys.focus(bool(glfw.get_window_attrib(self.window, glfw.FOCUSED)))

    def _key(self, window, key, scancode, action, mods) -> None:
        # Repeats are deliberately ignored: held state comes from press/release pairs.
        if action not in (glfw.PRESS, glfw.RELEASE):
            return
        if action == glfw.PRESS and mods & (glfw.MOD_CONTROL | glfw.MOD_ALT | glfw.MOD_SUPER):
            return
        names = {
            glfw.KEY_UP: "up", glfw.KEY_DOWN: "down", glfw.KEY_LEFT: "left",
            glfw.KEY_RIGHT: "right", glfw.KEY_SPACE: "stop", glfw.KEY_R: "reset",
        }
        if key in names:
            self.controls.keys.key(names[key], action == glfw.PRESS)
        elif key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.controls.keys.focus(False)
            glfw.set_window_should_close(window, True)

    def _mouse(self, window, x: float, y: float) -> None:
        old_x, old_y = self._cursor
        self._cursor = (x, y)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) != glfw.PRESS:
            return
        _, height = glfw.get_window_size(window)
        if height:
            mujoco.mjv_moveCamera(self.world.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                                 (x - old_x) / height, (y - old_y) / height,
                                 self.scene, self.camera)

    def _scroll(self, window, x: float, y: float) -> None:
        mujoco.mjv_moveCamera(self.world.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                             0.0, -0.05 * y, self.scene, self.camera)

    def poll(self) -> None:
        glfw.poll_events()

    def is_running(self) -> bool:
        return not glfw.window_should_close(self.window)

    def sync(self) -> None:
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if not width or not height:
            return
        body = self.world.bodies[0]
        sensors = body.sensors()
        self.camera.lookat[:] = sensors["trunk"]
        self.camera.lookat[2] += 0.04
        with self.world.lock:
            mujoco.mjv_updateScene(self.world.model, self.world.data, self.option,
                                   None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        status, health = self.controls.snapshot()
        loop = health.get("control_loop", {})
        hz = loop.get("achieved_hz")
        rate = f"{hz:.1f} Hz / {loop.get('missed', 0)} missed" if hz is not None else "Waiting for loop"
        upright = sensors["trunk_z"] > 0.095 and sensors["imu"]["gravity"][2] < -0.9
        vx, _, turn = self.controls.keys.command()
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, f"MICRODUCK  /  {self.mode}\n{status}\n{rate}\n"
                           f"{'Upright' if upright else 'Down - press R to reset'}  /  "
                           f"height {sensors['trunk_z']:.3f} m\n"
                           f"Forward {vx:+.2f} m/s  /  turn {turn:+.2f}", "", self.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                           viewport, "HOLD ARROWS  Drive / turn   |   Release to stop\n"
                           "SPACE  Stop   |   R  Reset upright\n"
                           "Drag  Orbit   |   Scroll  Zoom   |   Switching windows stops movement",
                           "", self.context)
        glfw.swap_buffers(self.window)

    def close(self) -> None:
        self.controls.keys.focus(False)
        glfw.make_context_current(self.window)
        self.context.free()
        self.scene.free()
        glfw.destroy_window(self.window)
