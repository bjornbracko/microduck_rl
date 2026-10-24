"""Give the native GLFW window a macOS application identity, using this venv's Python.

The generated bundle lives in simulator state, never in source control. Python runs on its
main thread (GLFW's Cocoa requirement), with the existing venv and packages preserved.
"""
from __future__ import annotations

from pathlib import Path
import plistlib
import shutil
import sys
import sysconfig


def create_app(state: Path) -> Path:
    contents = state / "Microduck.app" / "Contents"
    binary = contents / "MacOS" / "Microduck"
    binary.parent.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable).resolve()
    if not binary.exists() or binary.read_bytes() != python.read_bytes():
        shutil.copy2(python, binary)
    binary.chmod(0o755)
    (contents / "pyvenv.cfg").write_text((Path(sys.prefix) / "pyvenv.cfg").read_text())
    library = contents / "lib"
    if library.is_symlink():
        library.unlink()
    library.mkdir(exist_ok=True)
    # Python's executable-relative dylib lookup must work even in a fresh uv venv,
    # whose lib/ does not normally contain the base interpreter's shared library.
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    shared = sysconfig.get_config_var("LDLIBRARY")
    for link, target in (
        (library / version, Path(sys.prefix) / "lib" / version),
        (library / shared, Path(sys.base_prefix) / "lib" / shared),
    ):
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(target, target_is_directory=target.is_dir())
    info = {
        "CFBundleIdentifier": "org.microduck.simulator",
        "CFBundleName": "Microduck",
        "CFBundleDisplayName": "Microduck Simulator",
        "CFBundleExecutable": "Microduck",
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1",
        "NSHighResolutionCapable": True,
    }
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(info, stream)
    return binary


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("the application bundle is only needed on macOS")
    print(create_app(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
