"""Window input sent to robotd, with simulator resets coordinated off the physics thread."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import threading
import time


class DriveKeys:
    """Held keys, rather than key-repeat events: release and loss of focus both stop."""

    def __init__(self):
        self.held: set[str] = set()
        self.focused = True
        self.reset_requested = False
        self.resetting = False
        self.lock = threading.Lock()

    def key(self, name: str, pressed: bool) -> None:
        with self.lock:
            if not pressed:
                self.held.discard(name)
            elif name == "stop":
                self.held.clear()
            elif name == "reset" and not self.resetting:
                self.held.clear()
                self.reset_requested = True
            elif self.focused and not self.resetting and not self.reset_requested:
                self.held.add(name)

    def focus(self, focused: bool) -> None:
        with self.lock:
            self.focused = focused
            self.held.clear()

    def stop(self) -> None:
        with self.lock:
            self.held.clear()

    def command(self) -> tuple[float, float, float]:
        with self.lock:
            if not self.focused or self.resetting or self.reset_requested:
                return (0.0, 0.0, 0.0)
            forward = int("up" in self.held) - int("down" in self.held)
            turn = int("left" in self.held) - int("right" in self.held)
            return (0.3 * forward, 0.0, 0.6 * turn)

    def begin_reset(self) -> bool:
        with self.lock:
            if not self.reset_requested:
                return False
            self.reset_requested = False
            self.resetting = True
            self.held.clear()
            return True

    def end_reset(self) -> None:
        with self.lock:
            self.resetting = False
            self.held.clear()


def rpc(path: Path, method: str, params: dict | None = None) -> dict:
    """Discrete calls have replies; keep their timeouts away from physics and rendering."""
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(0.5)
        connection.connect(str(path))
        connection.sendall((json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params or {},
        }) + "\n").encode())
        with connection.makefile("rb") as stream:
            answer = json.loads(stream.readline())
    if "error" in answer:
        raise RuntimeError(answer["error"].get("message", "controller refused the call"))
    result = answer["result"]
    if result.get("accepted") is False:
        raise RuntimeError(result.get("reason", "controller refused the call"))
    return result


class NativeControls:
    """A robotd client; it never writes servo targets or runs a policy itself."""

    def __init__(self, socket_path: Path, world):
        self.socket_path = socket_path
        self.world = world
        self.keys = DriveKeys()
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status = "Connecting to controller"
        self._health: dict = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self) -> tuple[str, dict]:
        with self._lock:
            return self._status, self._health.copy()

    def _report(self, status: str, health: dict | None = None) -> None:
        with self._lock:
            self._status = status
            if health is not None:
                self._health = health

    def _wait_for(self, predicate, timeout: float, message: str) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if self._quit.wait(0.01) or time.monotonic() >= deadline:
                raise RuntimeError(message)

    def _reset(self) -> None:
        """Let the daemon forget its episode before placing a fresh, held body at home.

        `robot.relax` resets skill/seat/recurrent state in the real controller. Waiting for
        torque-off acknowledges its control tick, not merely the RPC request being queued.
        Physics time stays monotonic, so cameras and sensor consumers keep their clocks.
        """
        self._report("Resetting - stopping controller")
        rpc(self.socket_path, "robot.stop")
        rpc(self.socket_path, "robot.enable", {"on": False})
        rpc(self.socket_path, "robot.relax")
        body = self.world.bodies[0]
        self._wait_for(lambda: not body.torque_on, 1.5, "controller did not relax")
        # Finish the same tick's controller.forget(), and let the disable reset fall handling.
        if self._quit.wait(0.08):
            return
        rpc(self.socket_path, "robot.head", {
            "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
        })
        rpc(self.socket_path, "robot.pose", {"active": False})
        self.world.reset_home()
        self._report("Resetting - starting balancing policy")
        rpc(self.socket_path, "robot.enable", {"on": True})
        self._wait_for(lambda: body.torque_on and body.released, 5.0,
                       "policy did not take over; press R to retry")
        self._report("Controller connected")

    @staticmethod
    def _send(connection: socket.socket, command: tuple[float, float, float]) -> None:
        vx, vy, vyaw = command
        connection.sendall((json.dumps({
            "jsonrpc": "2.0", "method": "robot.move",
            "params": {"vx": vx, "vy": vy, "vyaw": vyaw},
        }) + "\n").encode())

    def _run(self) -> None:
        connection = None
        previous = (0.0, 0.0, 0.0)
        health_at = 0.0
        try:
            while not self._quit.is_set():
                try:
                    if connection is None:
                        # Reconnection never resumes a key that was held across a broken link.
                        self.keys.stop()
                        connection = socket.socket(socket.AF_UNIX)
                        connection.settimeout(0.2)
                        connection.connect(str(self.socket_path))
                        health_at = 0.0
                        previous = (0.0, 0.0, 0.0)
                    if self.keys.begin_reset():
                        try:
                            self._send(connection, (0.0, 0.0, 0.0))
                            self._reset()
                        finally:
                            self.keys.end_reset()
                        health_at = 0.0
                    command = self.keys.command()
                    if any(command) or command != previous:
                        self._send(connection, command)
                    previous = command
                    if time.monotonic() >= health_at:
                        health = rpc(self.socket_path, "robot.health")
                        self._report("Controller connected", health)
                        health_at = time.monotonic() + 1.0
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    self.keys.stop()
                    self._report(f"Controller unavailable: {error}", {})
                    if connection is not None:
                        try:
                            self._send(connection, (0.0, 0.0, 0.0))
                        except OSError:
                            pass
                        connection.close()
                        connection = None
                    if self._quit.wait(0.25):
                        break
                self._quit.wait(0.05)  # 20 Hz intents; robotd retains its normal deadman.
        finally:
            if connection is not None:
                try:
                    self._send(connection, (0.0, 0.0, 0.0))
                except OSError:
                    pass
                connection.close()

    def close(self) -> None:
        self.keys.stop()
        self._quit.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
