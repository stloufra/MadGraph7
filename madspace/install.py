#!/usr/bin/env python3
"""Install madspace either using pre-compiled binaries or built from source.

Interactive usage (no arguments):  python install.py
Non-interactive examples:
  python install.py --bin
  python install.py --source
  python install.py --source --cuda --cuda-arch "75;80;86"
  python install.py --source --cuda --hip --simd --debug
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
INSTALL_DIR = SCRIPT_DIR / "install"
SETTINGS_FILE = SCRIPT_DIR / "build" / "install_settings.json"

PACKAGE_NAME = "madspace"

DEFAULT_CUDA_ARCH = "75"
DEFAULT_HIP_ARCH = "gfx900"

# Platform-aware defaults for source-build options (mirrors CMakeLists.txt logic)
_IS_APPLE = platform.system() == "Darwin"
_PLATFORM_SOURCE_DEFAULTS: dict[str, bool] = {
    "cuda": False,
    "hip": False,
    "openblas": not _IS_APPLE,
    "simd": False,
    "debug": False,
}


# Interactive helpers


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


def ask_string(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [default: {default}]: ").strip()
    return raw if raw else default


def ask_compile_options(saved: dict | None = None) -> dict[str, bool]:
    """Multi-select menu for compile options; returns a dict of flags."""
    items = [
        ("cuda", "Build CUDA backend"),
        ("hip", "Build HIP/ROCm backend"),
        (
            "openblas",
            "Build OpenBLAS from source (recommended on Linux, not needed on Apple)",
        ),
        (
            "simd",
            "Build SIMD backend (experimental — not required to run SIMD matrix elements)",
        ),
        ("debug", "Build with debug symbols, still optimizing (RelWithDebInfo mode)"),
    ]
    saved = saved or {}
    prev = {key: saved.get(key, False) for key, _ in items}
    has_prev = any(prev.values())

    print()
    print("Compile options (select multiple, default: none):")
    for i, (key, label) in enumerate(items, 1):
        marker = " [*]" if prev[key] else ""
        print(f"  {i}. {label}{marker}")

    hint = "Enter to keep previous selection" if has_prev else "Enter for none"
    while True:
        raw = input(
            f"Enter numbers separated by commas/spaces, or press {hint}: "
        ).strip()
        if not raw:
            return prev if has_prev else {key: False for key, _ in items}
        try:
            chosen = {int(x) for x in raw.replace(",", " ").split()}
        except ValueError:
            print("  Invalid input — please enter numbers, e.g. 1,3 or 1 3")
            continue
        if not all(1 <= c <= len(items) for c in chosen):
            print(f"  Numbers must be between 1 and {len(items)}.")
            continue
        return {key: (idx in chosen) for idx, (key, _) in enumerate(items, 1)}


# Command execution


def run(cmd: list, env: dict | None = None) -> None:
    display = " ".join(str(c) for c in cmd)
    print(f"\n$ {display}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def install_build_deps() -> dict:
    """Install build-system dependencies into INSTALL_DIR and return an updated env."""
    pyproject_path = SCRIPT_DIR / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    requires = config.get("build-system", {}).get("requires", [])
    if requires:
        print("Installing build dependencies...")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                *requires,
                f"--target={INSTALL_DIR}",
            ]
        )

    env = os.environ.copy()
    pythonpath_parts = [str(INSTALL_DIR)]
    if existing := env.get("PYTHONPATH"):
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


# Main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Install mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--bin",
        action="store_true",
        default=False,
        help="Install pre-compiled package (default when interactive).",
    )
    mode.add_argument(
        "--source",
        action="store_true",
        default=False,
        help="Build and install from source.",
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Re-install non-interactively using saved settings; falls back to built-in defaults.",
    )

    # Compile option flags (each defaults to None = not specified via CLI)
    cuda_grp = parser.add_mutually_exclusive_group()
    cuda_grp.add_argument(
        "--cuda", dest="cuda", action="store_true", help="Enable CUDA backend."
    )
    cuda_grp.add_argument(
        "--no-cuda", dest="cuda", action="store_false", help="Disable CUDA backend."
    )

    hip_grp = parser.add_mutually_exclusive_group()
    hip_grp.add_argument(
        "--hip", dest="hip", action="store_true", help="Enable HIP/ROCm backend."
    )
    hip_grp.add_argument(
        "--no-hip", dest="hip", action="store_false", help="Disable HIP backend."
    )

    openblas_grp = parser.add_mutually_exclusive_group()
    openblas_grp.add_argument(
        "--openblas",
        dest="openblas",
        action="store_true",
        help="Build OpenBLAS from source (default on Linux/Windows).",
    )
    openblas_grp.add_argument(
        "--no-openblas",
        dest="openblas",
        action="store_false",
        help="Use system BLAS library (default on Apple).",
    )

    simd_grp = parser.add_mutually_exclusive_group()
    simd_grp.add_argument(
        "--simd",
        dest="simd",
        action="store_true",
        help="Enable SIMD backend (experimental).",
    )
    simd_grp.add_argument(
        "--no-simd", dest="simd", action="store_false", help="Disable SIMD backend."
    )

    debug_grp = parser.add_mutually_exclusive_group()
    debug_grp.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Build with debug symbols (RelWithDebInfo).",
    )
    debug_grp.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Build in Release mode (default).",
    )

    # Architecture overrides
    parser.add_argument(
        "--cuda-arch",
        default=None,
        metavar="ARCHS",
        help=f"Semicolon-separated CUDA compute capabilities (default: {DEFAULT_CUDA_ARCH}). "
        'Example: "75;80;86".',
    )
    parser.add_argument(
        "--hip-arch",
        default=None,
        metavar="ARCHS",
        help=f"Semicolon-separated HIP GPU architectures (default: {DEFAULT_HIP_ARCH}). "
        'Example: "gfx900;gfx906;gfx1100".',
    )

    # None = not provided by user; overridden by set_defaults below
    parser.set_defaults(cuda=None, hip=None, openblas=None, simd=None, debug=None)
    args = parser.parse_args()

    # Load saved settings when a previous installation is present
    saved = load_settings() if (INSTALL_DIR / "madspace").is_dir() else {}

    # Determine install mode
    if args.yes:
        from_source = saved.get("mode", "bin") == "source"
    elif args.bin:
        from_source = False
    elif args.source:
        from_source = True
    else:
        print("madspace installer")
        print("==================")
        default_is_bin = saved.get("mode", "bin") != "source"
        from_source = not ask_yes_no(
            "Install pre-compiled package? (recommended)", default=default_is_bin
        )

    # PyPI installation
    if not from_source:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                PACKAGE_NAME,
                f"--target={INSTALL_DIR}",
            ]
        )
        save_settings({"mode": "bin"})
        print(f"\nInstalled to: {INSTALL_DIR}")
        return

    # Source build — compile options
    compile_flags_given = any(
        getattr(args, attr) is not None
        for attr in ("cuda", "hip", "openblas", "simd", "debug")
    )

    if args.yes:
        enable_cuda = saved.get("cuda", _PLATFORM_SOURCE_DEFAULTS["cuda"])
        enable_hip = saved.get("hip", _PLATFORM_SOURCE_DEFAULTS["hip"])
        enable_openblas = saved.get("openblas", _PLATFORM_SOURCE_DEFAULTS["openblas"])
        enable_simd = saved.get("simd", _PLATFORM_SOURCE_DEFAULTS["simd"])
        enable_debug = saved.get("debug", _PLATFORM_SOURCE_DEFAULTS["debug"])
    elif compile_flags_given:
        enable_cuda = bool(args.cuda)
        enable_hip = bool(args.hip)
        enable_openblas = (
            bool(args.openblas)
            if args.openblas is not None
            else _PLATFORM_SOURCE_DEFAULTS["openblas"]
        )
        enable_simd = bool(args.simd)
        enable_debug = bool(args.debug)
    else:
        print("madspace source build")
        print("=====================")
        # Show saved source settings if available, else platform-appropriate defaults
        menu_defaults = (
            saved if saved.get("mode") == "source" else _PLATFORM_SOURCE_DEFAULTS
        )
        opts = ask_compile_options(menu_defaults)
        enable_cuda = opts["cuda"]
        enable_hip = opts["hip"]
        enable_openblas = opts["openblas"]
        enable_simd = opts["simd"]
        enable_debug = opts["debug"]

    # Compute capability prompts
    cuda_arch = saved.get("cuda_arch", DEFAULT_CUDA_ARCH)
    hip_arch = saved.get("hip_arch", DEFAULT_HIP_ARCH)

    if enable_cuda:
        if args.yes:
            cuda_arch = saved.get("cuda_arch", DEFAULT_CUDA_ARCH)
        elif args.cuda_arch is not None:
            cuda_arch = args.cuda_arch
        else:
            cuda_arch = ask_string(
                "CUDA compute capabilities (semicolon-separated, e.g. 75;80;86)",
                default=cuda_arch,
            )

    if enable_hip:
        if args.yes:
            hip_arch = saved.get("hip_arch", DEFAULT_HIP_ARCH)
        elif args.hip_arch is not None:
            hip_arch = args.hip_arch
        else:
            hip_arch = ask_string(
                "HIP GPU architectures (semicolon-separated, e.g. gfx900;gfx906;gfx1100)",
                default=hip_arch,
            )

    # Assemble pip command
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--upgrade",
        "-Cbuild-dir=build",
        ".",
        f"--target={INSTALL_DIR}",
    ]

    if enable_cuda:
        cmd += [
            "-Ccmake.define.ENABLE_CUDA=ON",
            f"-Ccmake.define.CMAKE_CUDA_ARCHITECTURES={cuda_arch}",
        ]
    if enable_hip:
        cmd += [
            "-Ccmake.define.ENABLE_HIP=ON",
            f"-Ccmake.define.CMAKE_HIP_ARCHITECTURES={hip_arch}",
        ]
    cmd.append(f"-Ccmake.define.ENABLE_OPENBLAS={'ON' if enable_openblas else 'OFF'}")
    if enable_simd:
        cmd.append("-Ccmake.define.ENABLE_SIMD=ON")
    if enable_debug:
        cmd.append("-Ccmake.build-type=RelWithDebInfo")

    env = install_build_deps()
    run(cmd, env=env)
    save_settings(
        {
            "mode": "source",
            "cuda": enable_cuda,
            "cuda_arch": cuda_arch,
            "hip": enable_hip,
            "hip_arch": hip_arch,
            "openblas": enable_openblas,
            "simd": enable_simd,
            "debug": enable_debug,
        }
    )
    print(f"\nInstalled to: {INSTALL_DIR}")


if __name__ == "__main__":
    main()
