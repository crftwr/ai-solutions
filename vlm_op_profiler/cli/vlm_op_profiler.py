#!/usr/bin/env python3
"""vlm-op-profiler — thin launcher that injects libbackend_stats and exec's llama-mtmd-cli.

What it does:
  1. Parses profiler-specific flags (--out-dir, --steps, etc.).
  2. Locates libbackend_stats.so/.dylib relative to this script.
  3. Creates the output directory.
  4. Sets LD_PRELOAD / DYLD_INSERT_LIBRARIES so libbackend_stats is injected.
  5. Sets PROFSTATS_OUT_DIR and PROFSTATS_MAX_STEPS for the interceptor.
  6. exec's llama-mtmd-cli with the remaining (model / image / prompt) args.

Profiler-specific flags are consumed here; everything else is forwarded to
llama-mtmd-cli unchanged.
"""

import os
import sys
import argparse
import shutil

VERSION = "0.1.0-dev"

# Directory that contains this script — used to find libbackend_stats and
# llama-mtmd-cli when they are co-located (e.g. inside the Docker image).
HERE = os.path.dirname(os.path.realpath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_binary(name: str, candidates: list[str]) -> str:
    """Return the first executable path from candidates, then search PATH."""
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which(name) or ""


def prepend_preload(lib_path: str) -> None:
    """Prepend lib_path to LD_PRELOAD (or DYLD_INSERT_LIBRARIES on macOS)."""
    if sys.platform == "darwin":
        key = "DYLD_INSERT_LIBRARIES"
        # Needed for DYLD_INSERT_LIBRARIES to work across two-level namespace libs.
        os.environ["DYLD_FORCE_FLAT_NAMESPACE"] = "1"
    else:
        key = "LD_PRELOAD"
    existing = os.environ.get(key, "")
    os.environ[key] = lib_path + (":" + existing if existing else "")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vlm-op-profiler",
        description=(
            f"vlm-op-profiler {VERSION}\n\n"
            "Collect ggml tensor-operation statistics from llama.cpp while running\n"
            "a vision-language model.  Outputs trace.jsonl under --out-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "All other flags are forwarded to llama-mtmd-cli unchanged.\n\n"
            "Environment variables (also settable here as flags):\n"
            "  PROFSTATS_OUT_DIR    Same as --out-dir\n"
            "  PROFSTATS_MAX_STEPS  Same as --steps\n\n"
            "Examples:\n"
            "  # VLM with image:\n"
            "  vlm-op-profiler --out-dir results/test \\\n"
            "       --model  models/llava-1.6.Q4_K_M.gguf \\\n"
            "       --mmproj models/llava-1.6-mmproj.gguf \\\n"
            "       --image  photo.jpg \\\n"
            "       'Describe this image.'\n\n"
            "  # Text-only (no image):\n"
            "  vlm-op-profiler --out-dir results/text \\\n"
            "       --model models/mistral-7b.Q4_K_M.gguf \\\n"
            "       -p 'The capital of France is'"
        ),
    )
    p.add_argument(
        "--version", action="version", version=f"vlm-op-profiler {VERSION}"
    )
    p.add_argument(
        "--out-dir",
        default="profstats_out",
        help="Output directory (default: ./profstats_out)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop recording after N decode graphs (0=unlimited)",
    )
    p.add_argument(
        "--mtmd-cli",
        default="",
        metavar="PATH",
        help="Explicit path to llama-mtmd-cli",
    )
    p.add_argument(
        "--include-vision-encode",
        action="store_true",
        help="Also record the image-encoder graph separately (Phase 4)",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()

    # Split argv on '--': everything after it is forwarded verbatim to
    # llama-mtmd-cli and never seen by argparse.
    argv = sys.argv[1:]
    try:
        sep = argv.index("--")
        our_argv, forward_argv = argv[:sep], argv[sep + 1:]
    except ValueError:
        our_argv, forward_argv = argv, None

    args, extra = parser.parse_known_args(our_argv)

    # When there was no '--', unknown args from parse_known_args are forwarded.
    if forward_argv is None:
        forward_argv = extra

    if not forward_argv:
        parser.print_help()
        sys.exit(0)

    # ---- Locate shared library -----------------------------------------------
    lib_name = (
        "libbackend_stats.dylib" if sys.platform == "darwin"
        else "libbackend_stats.so"
    )
    lib_path = os.path.join(HERE, lib_name)
    if not os.path.exists(lib_path):
        print(
            f"vlm-op-profiler: cannot find {lib_name} (looked in {HERE}).\n"
            "Ensure make docker-build has completed.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Locate llama-mtmd-cli -----------------------------------------------
    mtmd_cli = args.mtmd_cli or find_binary("llama-mtmd-cli", [
        os.path.join(HERE, "llama-mtmd-cli"),
        "/app/llama-mtmd-cli",
        os.path.join(HERE, "bin", "llama-mtmd-cli"),
    ])
    if not mtmd_cli:
        print(
            "vlm-op-profiler: cannot find llama-mtmd-cli.\n"
            "Pass --mtmd-cli <path> or ensure it is on PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Create output directory ---------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Configure the interceptor via environment --------------------------
    os.environ["PROFSTATS_OUT_DIR"] = args.out_dir
    os.environ["PROFSTATS_MAX_STEPS"] = str(args.steps)
    # args.include_vision_encode → Phase 4: PROFSTATS_INCLUDE_VISION_ENCODE

    prepend_preload(lib_path)

    # ---- exec into llama-mtmd-cli -------------------------------------------
    exec_argv = [mtmd_cli] + forward_argv
    print(
        f"vlm-op-profiler: exec {mtmd_cli}"
        f"  (out_dir={args.out_dir}  steps={args.steps})",
        file=sys.stderr,
    )
    os.execv(mtmd_cli, exec_argv)


if __name__ == "__main__":
    main()
