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
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
INSTALL_DIR = SCRIPT_DIR / "install"

PACKAGE_NAME = "madspace"

DEFAULT_CUDA_ARCH = "75"
DEFAULT_HIP_ARCH = "gfx900"


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


def ask_compile_options() -> dict[str, bool]:
    """Multi-select menu for compile options; returns a dict of flags."""
    items = [
        ("cuda", "Build CUDA backend"),
        ("hip", "Build HIP/ROCm backend"),
        (
            "simd",
            "Build SIMD backend (experimental — not required to run SIMD matrix elements)",
        ),
        ("debug", "Build with debug symbols, still optimizing (RelWithDebInfo mode)"),
    ]
    print()
    print("Compile options (select multiple, default: none):")
    for i, (_, label) in enumerate(items, 1):
        print(f"  {i}. {label}")

    while True:
        raw = input(
            "Enter numbers separated by commas/spaces, or press Enter for none: "
        ).strip()
        if not raw:
            return {key: False for key, _ in items}
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


def run(cmd: list) -> None:
    display = " ".join(str(c) for c in cmd)
    print(f"\n$ {display}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        sys.exit(result.returncode)


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
    parser.set_defaults(cuda=None, hip=None, simd=None, debug=None)
    args = parser.parse_args()

    # Determine install mode
    if args.bin:
        from_source = False
    elif args.source:
        from_source = True
    else:
        print("madspace installer")
        print("==================")
        from_source = not ask_yes_no(
            "Install pre-compiled package? (recommended)", default=True
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
        print(f"\nInstalled to: {INSTALL_DIR}")
        return

    # Source build
    compile_flags_given = any(
        getattr(args, attr) is not None for attr in ("cuda", "hip", "simd", "debug")
    )

    if compile_flags_given:
        enable_cuda = bool(args.cuda)
        enable_hip = bool(args.hip)
        enable_simd = bool(args.simd)
        enable_debug = bool(args.debug)
    else:
        print("madspace source build")
        print("=====================")
        opts = ask_compile_options()
        enable_cuda = opts["cuda"]
        enable_hip = opts["hip"]
        enable_simd = opts["simd"]
        enable_debug = opts["debug"]

    # Compute capability prompts
    cuda_arch = DEFAULT_CUDA_ARCH
    hip_arch = DEFAULT_HIP_ARCH

    if enable_cuda:
        if args.cuda_arch is not None:
            cuda_arch = args.cuda_arch
        else:
            cuda_arch = ask_string(
                "CUDA compute capabilities (semicolon-separated, e.g. 75;80;86)",
                default=DEFAULT_CUDA_ARCH,
            )

    if enable_hip:
        if args.hip_arch is not None:
            hip_arch = args.hip_arch
        else:
            hip_arch = ask_string(
                "HIP GPU architectures (semicolon-separated, e.g. gfx900;gfx906;gfx1100)",
                default=DEFAULT_HIP_ARCH,
            )

    # Assemble pip command
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
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
    if enable_simd:
        cmd.append("-Ccmake.define.ENABLE_SIMD=ON")
    if enable_debug:
        cmd.append("-Ccmake.build-type=RelWithDebInfo")

    run(cmd)
    print(f"\nInstalled to: {INSTALL_DIR}")


if __name__ == "__main__":
    main()
