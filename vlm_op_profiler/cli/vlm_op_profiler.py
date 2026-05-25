#!/usr/bin/env python3
"""vlm-op-profiler — thin launcher that injects libbackend_stats and exec's llama-mtmd-cli.

What it does:
  1. Parses profiler-specific flags (--out-dir, --steps, --include-vision-encode, ...).
  2. Locates libbackend_stats.so/.dylib relative to this script.
  3. Creates the output directory.
  4. Writes run_meta.json (provenance: commits, model/image SHA-256, prompt, ...).
  5. Sets LD_PRELOAD / DYLD_INSERT_LIBRARIES so libbackend_stats is injected.
  6. Sets PROFSTATS_OUT_DIR / PROFSTATS_MAX_STEPS / PROFSTATS_INCLUDE_VISION_ENCODE.
  7. exec's llama-mtmd-cli with the remaining (model / image / prompt) args.

Profiler-specific flags are consumed here; everything else is forwarded to
llama-mtmd-cli unchanged.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys

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


def sha256_file(path: str) -> str | None:
    """Compute SHA-256 of a file; return None if path is missing/unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def extract_arg(argv: list[str], names: tuple[str, ...]) -> str | None:
    """Find the value of the first of `names` that appears as a flag in argv.

    Supports both `--flag value` and `--flag=value` forms. Does not consume
    or rewrite argv; this is for metadata extraction only.
    """
    for i, tok in enumerate(argv):
        if tok in names and i + 1 < len(argv):
            return argv[i + 1]
        for name in names:
            prefix = name + "="
            if tok.startswith(prefix):
                return tok[len(prefix):]
    return None


def git_commit(repo_dir: str) -> str | None:
    """Return the HEAD commit SHA of repo_dir, or None if unavailable."""
    if not os.path.isdir(repo_dir):
        return None
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def resolve_profiler_commit() -> str:
    """Profiler commit: prefer baked-in env (Docker), fall back to local git."""
    env = os.environ.get("PROFILER_COMMIT")
    if env and env != "unknown":
        return env
    # When run from a source checkout, walk up from this file to find a .git.
    cur = HERE
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, ".git")):
            sha = git_commit(cur)
            if sha:
                return sha
        cur = os.path.dirname(cur)
    return env or "unknown"


def resolve_llama_commit() -> str:
    """llama.cpp commit: prefer baked-in env (Docker), fall back to submodule."""
    env = os.environ.get("LLAMA_COMMIT")
    if env and env != "unknown":
        return env
    # Look for the submodule relative to this file, then walk up.
    cur = HERE
    for _ in range(6):
        cand = os.path.join(cur, "third_party", "llama.cpp")
        if os.path.exists(os.path.join(cand, ".git")) or os.path.isdir(
            os.path.join(cand, ".git")
        ):
            sha = git_commit(cand)
            if sha:
                return sha
        cur = os.path.dirname(cur)
    return env or "unknown"


def detect_inner_backend() -> str:
    """Best-effort guess at which ggml backend will be used at runtime."""
    if sys.platform == "darwin":
        return "metal"
    if os.environ.get("CUDA_VISIBLE_DEVICES") or shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def platform_label() -> str:
    if sys.platform == "darwin":
        return "macOS"
    if sys.platform.startswith("linux"):
        return "Linux"
    return sys.platform


def write_run_meta(
    out_dir: str,
    forward_argv: list[str],
    include_vision_encode: bool,
) -> None:
    """Write run_meta.json into out_dir.

    Fields are best-effort; missing values become null. See docs/output_format.md.
    """
    model_path = extract_arg(forward_argv, ("--model", "-m"))
    mmproj_path = extract_arg(forward_argv, ("--mmproj",))
    image_path = extract_arg(forward_argv, ("--image",))
    prompt = extract_arg(forward_argv, ("--prompt", "-p"))

    meta = {
        "profiler_version": VERSION,
        "profiler_commit": resolve_profiler_commit(),
        "llama_commit": resolve_llama_commit(),
        "model_path": model_path,
        "model_sha256": sha256_file(model_path) if model_path else None,
        "mmproj_path": mmproj_path,
        "mmproj_sha256": sha256_file(mmproj_path) if mmproj_path else None,
        "gguf_metadata": {},  # reserved for later — see docs/output_format.md
        "prompt": prompt,
        "image_path": image_path,
        "image_sha256": sha256_file(image_path) if image_path else None,
        "run_id": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "platform": platform_label(),
        "arch": platform.machine(),
        "inner_backend": detect_inner_backend(),
        "include_vision_encode": include_vision_encode,
    }

    path = os.path.join(out_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"vlm-op-profiler: wrote {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vlm-op-profiler",
        description=(
            f"vlm-op-profiler {VERSION}\n\n"
            "Collect ggml tensor-operation statistics from llama.cpp while running\n"
            "a vision-language model.  Outputs trace.jsonl and run_meta.json under --out-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "All other flags are forwarded to llama-mtmd-cli unchanged.\n\n"
            "Environment variables (also settable here as flags):\n"
            "  PROFSTATS_OUT_DIR                Same as --out-dir\n"
            "  PROFSTATS_MAX_STEPS              Same as --steps\n"
            "  PROFSTATS_INCLUDE_VISION_ENCODE  Same as --include-vision-encode\n\n"
            "Examples:\n"
            "  # VLM with image:\n"
            "  vlm-op-profiler --out-dir results/test \\\n"
            "       --include-vision-encode \\\n"
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
        help=(
            "Also record vision-encoder graphs (identified by CONV_2D nodes). "
            "Default skips them so trace.jsonl reflects the LLM body only."
        ),
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

    # ---- Create output directory and write run metadata ----------------------
    os.makedirs(args.out_dir, exist_ok=True)
    write_run_meta(args.out_dir, forward_argv, args.include_vision_encode)

    # ---- Configure the interceptor via environment --------------------------
    os.environ["PROFSTATS_OUT_DIR"] = args.out_dir
    os.environ["PROFSTATS_MAX_STEPS"] = str(args.steps)
    os.environ["PROFSTATS_INCLUDE_VISION_ENCODE"] = (
        "1" if args.include_vision_encode else "0"
    )

    prepend_preload(lib_path)

    # ---- exec into llama-mtmd-cli -------------------------------------------
    exec_argv = [mtmd_cli] + forward_argv
    print(
        f"vlm-op-profiler: exec {mtmd_cli}"
        f"  (out_dir={args.out_dir}  steps={args.steps}"
        f"  include_vision_encode={args.include_vision_encode})",
        file=sys.stderr,
    )
    os.execv(mtmd_cli, exec_argv)


if __name__ == "__main__":
    main()
