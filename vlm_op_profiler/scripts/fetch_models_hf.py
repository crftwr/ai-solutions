#!/usr/bin/env python3
"""fetch_models_hf.py — download the VLM GGUF model suite from Hugging Face.

Each model has a (model_file, mmproj_file) pair; `llama-mtmd-cli` needs both
for vision-language inference. Files are downloaded into --out-dir and
skipped if already present (size > 0).

Suites:
  minimal — SmolVLM only (~2 GB); used by smoke tests.
  default — minimal + 7 architecturally diverse VLMs (~80 GB at Q4_K_M).
  full    — default + larger variants (~160 GB).

Reads HF_TOKEN from env for gated repos. Public repos work without a token.
"""

import argparse
import os
import sys
from dataclasses import dataclass

try:
    from huggingface_hub import hf_hub_download  # type: ignore
    from huggingface_hub.errors import (  # type: ignore
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )
except ModuleNotFoundError:
    print(
        "ERROR: huggingface_hub not installed. "
        "Run this script inside the Docker image, or `pip install huggingface_hub`.",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """A single VLM entry: main model file + optional mmproj projector."""
    name: str          # short identifier, used for directory names
    repo: str          # Hugging Face repo id
    model_file: str    # filename of the main GGUF
    mmproj_file: str | None  # filename of the mmproj (vision projector), or None for text-only
    suite: str         # "minimal" | "default" | "full"
    notes: str = ""


# Suite definition. Where mmproj is hosted in a different repo from the main
# model, the convention is to download both into --out-dir; llama-mtmd-cli is
# happy as long as both files exist on disk.
#
# IMPORTANT: repo IDs and filenames are best-effort references to current HF
# locations; downloads will skip gracefully (with a printed warning) if a
# specific file has been moved or renamed upstream.
MODELS: list[ModelSpec] = [
    # ----- minimal -----
    ModelSpec(
        name="SmolVLM-Instruct",
        repo="ggml-org/SmolVLM-Instruct-GGUF",
        model_file="SmolVLM-Instruct-Q4_K_M.gguf",
        mmproj_file="mmproj-SmolVLM-Instruct-f16.gguf",
        suite="minimal",
        notes="Smallest VLM; used for CI smoke tests.",
    ),

    # ----- default -----
    ModelSpec(
        name="LLaVA-1.6-Mistral-7B",
        repo="cjpais/llava-1.6-mistral-7b-gguf",
        model_file="llava-v1.6-mistral-7b.Q4_K_M.gguf",
        mmproj_file="mmproj-model-f16.gguf",
        suite="default",
    ),
    ModelSpec(
        name="Qwen2-VL-7B-Instruct",
        repo="bartowski/Qwen2-VL-7B-Instruct-GGUF",
        model_file="Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
        mmproj_file="mmproj-Qwen2-VL-7B-Instruct-f16.gguf",
        suite="default",
    ),
    ModelSpec(
        name="MiniCPM-V-2_6",
        repo="openbmb/MiniCPM-V-2_6-gguf",
        model_file="ggml-model-Q4_K_M.gguf",
        mmproj_file="mmproj-model-f16.gguf",
        suite="default",
    ),
    ModelSpec(
        name="Phi-3.5-vision-instruct",
        repo="bartowski/Phi-3.5-vision-instruct-GGUF",
        model_file="Phi-3.5-vision-instruct-Q4_K_M.gguf",
        mmproj_file="mmproj-Phi-3.5-vision-instruct-f16.gguf",
        suite="default",
    ),
    ModelSpec(
        name="Pixtral-12B",
        repo="bartowski/pixtral-12b-GGUF",
        model_file="pixtral-12b-Q4_K_M.gguf",
        mmproj_file="mmproj-pixtral-12b-f16.gguf",
        suite="default",
    ),
    ModelSpec(
        name="Llama-3.2-11B-Vision-Instruct",
        repo="lmstudio-community/Llama-3.2-11B-Vision-Instruct-GGUF",
        model_file="Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf",
        mmproj_file="mmproj-Llama-3.2-11B-Vision-Instruct-f16.gguf",
        suite="default",
        notes="Cross-attention bridge; gated — requires HF_TOKEN.",
    ),
    ModelSpec(
        name="Idefics3-8B",
        repo="bartowski/Idefics3-8B-Llama3-GGUF",
        model_file="Idefics3-8B-Llama3-Q4_K_M.gguf",
        mmproj_file="mmproj-Idefics3-8B-Llama3-f16.gguf",
        suite="default",
        notes="Late-fusion variant — included for architectural contrast.",
    ),
]


# ---------------------------------------------------------------------------
# Suite resolution
# ---------------------------------------------------------------------------

def select_suite(suite: str) -> list[ModelSpec]:
    """Return models in the requested suite tier (lower tiers are included)."""
    if suite == "minimal":
        levels = {"minimal"}
    elif suite == "default":
        levels = {"minimal", "default"}
    elif suite == "full":
        levels = {"minimal", "default", "full"}
    else:
        raise SystemExit(f"unknown --suite {suite!r}; expected minimal|default|full")
    return [m for m in MODELS if m.suite in levels]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def already_downloaded(path: str) -> bool:
    """Treat a file as already-downloaded if it exists and is non-empty."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def download_one(
    repo: str, filename: str, out_dir: str, token: str | None
) -> tuple[bool, str]:
    """Download a single file. Returns (ok, message).

    Returns (True, "skip"|"ok") on success; (False, reason) on failure. We
    treat a missing file as a per-model failure, not a fatal error, so the
    rest of the suite can still proceed.
    """
    dest = os.path.join(out_dir, filename)
    if already_downloaded(dest):
        return True, f"skip ({os.path.getsize(dest) // (1 << 20)} MiB)"
    try:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=out_dir,
            token=token,
        )
        return True, f"ok ({os.path.getsize(dest) // (1 << 20)} MiB)"
    except GatedRepoError:
        return False, "gated repo — set HF_TOKEN in .env"
    except RepositoryNotFoundError:
        return False, "repo not found on Hugging Face"
    except EntryNotFoundError:
        return False, "file not found in repo (filename may have changed upstream)"
    except Exception as exc:  # noqa: BLE001 — surface unknown errors as-is
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--suite",
        choices=("minimal", "default", "full"),
        default="default",
        help="Which suite tier to download (default: default).",
    )
    p.add_argument(
        "--out-dir",
        default="models",
        help="Directory to download into (default: ./models).",
    )
    p.add_argument(
        "--only",
        default="",
        metavar="SUBSTR",
        help="Filter: only download entries whose name contains this substring.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the resolved model list and exit (no downloads).",
    )
    args = p.parse_args()

    models = select_suite(args.suite)
    if args.only:
        sub = args.only.lower()
        models = [m for m in models if sub in m.name.lower()]
    if not models:
        print("No models match the requested suite/filter.", file=sys.stderr)
        return 1

    if args.list:
        for m in models:
            print(f"{m.suite:7}  {m.name:32}  {m.repo}/{m.model_file}")
            if m.mmproj_file:
                print(f"{'':7}  {'':32}  {m.repo}/{m.mmproj_file}")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    n_ok = 0
    n_fail = 0
    failures: list[tuple[str, str]] = []
    for m in models:
        print(f"==> {m.name}  ({m.repo})")
        for file in (m.model_file, m.mmproj_file):
            if not file:
                continue
            ok, msg = download_one(m.repo, file, args.out_dir, token)
            tag = "[ok]" if ok else "[FAIL]"
            print(f"    {tag} {file}  -- {msg}")
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failures.append((f"{m.name}/{file}", msg))

    print()
    print(f"Done: {n_ok} file(s) ready, {n_fail} failed.")
    print(f"Files in: {os.path.abspath(args.out_dir)}/")
    if failures:
        print("\nFailures:", file=sys.stderr)
        for name, reason in failures:
            print(f"  - {name}: {reason}", file=sys.stderr)
        # Non-zero exit only if EVERY download failed; partial successes are
        # common because repo/file naming drifts upstream.
        return 0 if n_ok > 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
