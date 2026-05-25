#!/usr/bin/env python3
"""run_suite.py — drive vlm-op-profiler over the model × prompt × image matrix.

Designed to be invoked inside the Docker image (which provides llama-mtmd-cli
and libbackend_stats.so on PATH). Outputs land under
  <out-root>/<model_name>/<run_id>/{trace.jsonl,run_meta.json}
so individual combinations are re-runnable in isolation.

Each combination is launched as a subprocess; a failure in one run is logged
and the suite continues — typical when a model has known VLM-side issues
(see docs/supported_models.md).
"""

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    """A model+mmproj pair to drive through llama-mtmd-cli."""
    name: str            # short identifier (drives <results>/<name>/...)
    model: str           # path to the main GGUF, relative to --models-dir
    mmproj: str | None   # path to the mmproj GGUF (None disables image input)


# The model list intentionally mirrors fetch_models_hf.py's `default` suite,
# minus the gated/large ones; missing files are silently skipped at runtime.
DEFAULT_MODELS: list[ModelEntry] = [
    ModelEntry(
        "SmolVLM-Instruct",
        "SmolVLM-Instruct-Q4_K_M.gguf",
        "mmproj-SmolVLM-Instruct-f16.gguf",
    ),
    ModelEntry(
        "LLaVA-1.6-Mistral-7B",
        "llava-v1.6-mistral-7b.Q4_K_M.gguf",
        "mmproj-model-f16.gguf",
    ),
    ModelEntry(
        "Qwen2-VL-7B-Instruct",
        "Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
        "mmproj-Qwen2-VL-7B-Instruct-f16.gguf",
    ),
    ModelEntry(
        "MiniCPM-V-2_6",
        "ggml-model-Q4_K_M.gguf",
        "mmproj-model-f16.gguf",
    ),
    ModelEntry(
        "Phi-3.5-vision-instruct",
        "Phi-3.5-vision-instruct-Q4_K_M.gguf",
        "mmproj-Phi-3.5-vision-instruct-f16.gguf",
    ),
    ModelEntry(
        "Pixtral-12B",
        "pixtral-12b-Q4_K_M.gguf",
        "mmproj-pixtral-12b-f16.gguf",
    ),
    ModelEntry(
        "Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf",
        "mmproj-Llama-3.2-11B-Vision-Instruct-f16.gguf",
    ),
    ModelEntry(
        "Idefics3-8B",
        "Idefics3-8B-Llama3-Q4_K_M.gguf",
        "mmproj-Idefics3-8B-Llama3-f16.gguf",
    ),
]

DEFAULT_PROMPTS: list[str] = [
    "Describe this image in detail.",
    "What objects are visible in the image?",
    "Summarise what is happening in the scene.",
]

DEFAULT_IMAGES: list[str] = [
    "example_64.jpg",
    "example_224.jpg",
    "example_448.jpg",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def slugify(text: str, *, max_len: int = 20) -> str:
    """Filesystem-safe, deterministic slug — used in run_id directory names."""
    s = _SLUG_RE.sub("_", text.strip()).strip("_").lower()
    return s[:max_len] or "x"


def iso_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_profiler() -> str:
    """Locate vlm_op_profiler.py (Docker layout first, then PATH)."""
    candidates = [
        "/app/vlm_op_profiler.py",
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "..", "cli", "vlm_op_profiler.py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.realpath(c)
    found = shutil.which("vlm-op-profiler")
    if found:
        return found
    raise SystemExit(
        "ERROR: vlm_op_profiler.py not found. Run inside the Docker image, "
        "or pass --profiler <path>."
    )


# ---------------------------------------------------------------------------
# Per-combination launcher
# ---------------------------------------------------------------------------

def run_one(
    *,
    profiler: str,
    model: ModelEntry,
    prompt: str,
    image_path: str,
    out_dir: str,
    steps: int,
    n_predict: int,
    include_vision_encode: bool,
    dry_run: bool,
) -> tuple[bool, str]:
    """Execute a single (model, prompt, image) combination."""
    cmd: list[str] = [
        "python3", profiler,
        "--out-dir", out_dir,
        "--steps", str(steps),
    ]
    if include_vision_encode:
        cmd.append("--include-vision-encode")
    cmd += ["--model", model.model]
    if model.mmproj:
        cmd += ["--mmproj", model.mmproj]
    cmd += [
        "--image", image_path,
        "-n", str(n_predict),
        "-p", prompt,
    ]

    if dry_run:
        print("  [dry-run]", " ".join(shlex_quote(c) for c in cmd))
        return True, "dry-run"

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "stderr.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    return proc.returncode == 0, f"exit={proc.returncode} (log: {log_path})"


def shlex_quote(s: str) -> str:
    """Minimal shell-quoting for printable dry-run lines."""
    if not s or any(c in s for c in " \t\n\"'$\\`"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--models-dir",
        default="/app/models",
        help="Directory containing the GGUF files (default: /app/models).",
    )
    p.add_argument(
        "--images-dir",
        default="/app/assets",
        help="Directory containing the test images (default: /app/assets).",
    )
    p.add_argument(
        "--out-root",
        default="/app/results",
        help="Root output directory (default: /app/results).",
    )
    p.add_argument(
        "--profiler",
        default="",
        help="Path to vlm_op_profiler.py (auto-detected if omitted).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Cap on decode graphs recorded per run (default: 8).",
    )
    p.add_argument(
        "--n-predict",
        type=int,
        default=32,
        help="Cap on tokens generated per run, passed as -n (default: 32).",
    )
    p.add_argument(
        "--include-vision-encode",
        action="store_true",
        help="Also record vision-encoder graphs in trace.jsonl.",
    )
    p.add_argument(
        "--only",
        default="",
        metavar="SUBSTR",
        help="Filter: only run models whose name contains this substring.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands and exit without invoking inference.",
    )
    args = p.parse_args()

    profiler = args.profiler or find_profiler()
    timestamp = iso_timestamp()

    # Materialise the matrix entries, resolving paths and skipping missing files.
    runs: list[tuple[ModelEntry, str, str]] = []
    for model in DEFAULT_MODELS:
        if args.only and args.only.lower() not in model.name.lower():
            continue
        model_path = os.path.join(args.models_dir, model.model)
        if not os.path.isfile(model_path):
            print(f"[skip] model not found: {model_path}")
            continue
        mmproj_path: str | None = None
        if model.mmproj:
            cand = os.path.join(args.models_dir, model.mmproj)
            if not os.path.isfile(cand):
                print(f"[skip] mmproj not found for {model.name}: {cand}")
                continue
            mmproj_path = cand
        resolved = ModelEntry(model.name, model_path, mmproj_path)
        for image_name in DEFAULT_IMAGES:
            image_path = os.path.join(args.images_dir, image_name)
            if not os.path.isfile(image_path):
                print(f"[skip] image not found: {image_path}")
                continue
            for prompt in DEFAULT_PROMPTS:
                runs.append((resolved, prompt, image_path))

    if not runs:
        print("No runs scheduled — check --models-dir, --images-dir, and --only.",
              file=sys.stderr)
        return 1

    print(f"Planned: {len(runs)} run(s)")
    summary: list[dict] = []
    for model, prompt, image_path in runs:
        image_slug = slugify(os.path.basename(image_path).rsplit(".", 1)[0])
        prompt_slug = slugify(prompt)
        run_id = f"{timestamp}_{image_slug}_{prompt_slug}"
        out_dir = os.path.join(args.out_root, model.name, run_id)

        print(f"==> {model.name} | {image_slug} | {prompt_slug}")
        ok, msg = run_one(
            profiler=profiler,
            model=model,
            prompt=prompt,
            image_path=image_path,
            out_dir=out_dir,
            steps=args.steps,
            n_predict=args.n_predict,
            include_vision_encode=args.include_vision_encode,
            dry_run=args.dry_run,
        )
        print(f"    [{'ok' if ok else 'FAIL'}] {msg}")
        summary.append({
            "model": model.name,
            "image": image_slug,
            "prompt": prompt,
            "run_id": run_id,
            "out_dir": out_dir,
            "ok": ok,
            "message": msg,
        })

    if not args.dry_run:
        summary_path = os.path.join(args.out_root, f"suite_{timestamp}.json")
        os.makedirs(args.out_root, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        n_ok = sum(1 for r in summary if r["ok"])
        print()
        print(f"Suite finished: {n_ok}/{len(summary)} runs ok.")
        print(f"Per-run results: {args.out_root}/<model>/<run_id>/")
        print(f"Suite summary:   {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
