#!/usr/bin/env python3
"""fetch_models_hf.py — download the VLM GGUF model suite from Hugging Face.

Each model has a (model_file, mmproj_file) pair; `llama-mtmd-cli` needs both
for vision-language inference. Files are downloaded into --out-dir and
skipped if already present (size > 0).

Suites:
  minimal — SmolVLM only (~2 GB); used by smoke tests.
  default — minimal + 7 architecturally diverse general-purpose VLMs (~80 GB at Q4_K_M).
  edge    — minimal + edge/robotics-oriented VLMs (InternVL2-2B, Qwen2.5-VL-3B,
            moondream2, PaliGemma 2, Florence-2, Gemma 3 4B) plus Q8_0 / IQ3_M
            quant variants for Jetson- and Qualcomm-class targets (~25 GB).
  full    — every spec in the registry (~180 GB).

Reads HF_TOKEN from env for gated repos. Public repos work without a token.

NOTE on edge entries: GGUF repo / file names for newer VLMs drift upstream.
Each entry below uses the most commonly attested HF path at the time of
writing; the fetcher graceful-skips per-file 404s so a partial download is
still useful. Update the registry as upstream conventions change.
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

    # ----- edge / physical-AI / robotics -----
    ModelSpec(
        name="InternVL2-2B",
        repo="bartowski/InternVL2-2B-GGUF",
        model_file="InternVL2-2B-Q4_K_M.gguf",
        mmproj_file="mmproj-InternVL2-2B-f16.gguf",
        suite="edge",
        notes="Most widely used VLM in robotics deployments; small enough for Jetson Orin.",
    ),
    ModelSpec(
        name="InternVL2-2B-Q8_0",
        repo="bartowski/InternVL2-2B-GGUF",
        model_file="InternVL2-2B-Q8_0.gguf",
        mmproj_file="mmproj-InternVL2-2B-f16.gguf",
        suite="edge",
        notes="int8 variant — exercises tensor-core int8 dot-product paths.",
    ),
    ModelSpec(
        name="Qwen2.5-VL-3B-Instruct",
        repo="bartowski/Qwen2.5-VL-3B-Instruct-GGUF",
        model_file="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        mmproj_file="mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
        suite="edge",
        notes="3B successor to Qwen2-VL; common edge fine-tune base.",
    ),
    ModelSpec(
        name="moondream2",
        repo="vikhyatk/moondream2",
        model_file="moondream2-text-model-f16.gguf",
        mmproj_file="moondream2-mmproj-f16.gguf",
        suite="edge",
        notes=(
            "Explicitly designed for embedded vision; ~2B params. Upstream "
            "ships f16 only — quantize locally for IQ3_M / Q4_K_M evaluation."
        ),
    ),
    ModelSpec(
        name="PaliGemma-2-3B",
        repo="bartowski/paligemma2-3b-pt-224-GGUF",
        model_file="paligemma2-3b-pt-224-Q4_K_M.gguf",
        mmproj_file="mmproj-paligemma2-3b-pt-224-f16.gguf",
        suite="edge",
        notes="Google robotics-lineage VLM (SigLIP + Gemma); 224 patch variant.",
    ),
    ModelSpec(
        name="Florence-2-base",
        repo="bartowski/Florence-2-base-ft-GGUF",
        model_file="Florence-2-base-ft-Q4_K_M.gguf",
        mmproj_file=None,
        suite="edge",
        notes=(
            "0.23B sub-1B spatial AI model (grounding / detection / OCR-with-location). "
            "No LLM body → different MAC profile from autoregressive VLMs. "
            "mmproj is not applicable; vision encoder is the whole model."
        ),
    ),
    ModelSpec(
        name="Gemma-3-4B-it",
        repo="bartowski/gemma-3-4b-it-GGUF",
        model_file="gemma-3-4b-it-Q4_K_M.gguf",
        mmproj_file="mmproj-gemma-3-4b-it-f16.gguf",
        suite="edge",
        notes="Compact Google multimodal; common base for edge fine-tunes.",
    ),
    ModelSpec(
        name="SmolVLM-Instruct-Q8_0",
        repo="ggml-org/SmolVLM-Instruct-GGUF",
        model_file="SmolVLM-Instruct-Q8_0.gguf",
        mmproj_file="mmproj-SmolVLM-Instruct-f16.gguf",
        suite="edge",
        notes="int8 SmolVLM — primary fixture for int8 dot-product profiling.",
    ),
    ModelSpec(
        name="Phi-3.5-vision-instruct-Q8_0",
        repo="bartowski/Phi-3.5-vision-instruct-GGUF",
        model_file="Phi-3.5-vision-instruct-Q8_0.gguf",
        mmproj_file="mmproj-Phi-3.5-vision-instruct-f16.gguf",
        suite="edge",
        notes="int8 Phi-3.5-V — covers a different attention shape than SmolVLM.",
    ),
    ModelSpec(
        name="SmolVLM-Instruct-IQ3_M",
        repo="ggml-org/SmolVLM-Instruct-GGUF",
        model_file="SmolVLM-Instruct-IQ3_M.gguf",
        mmproj_file="mmproj-SmolVLM-Instruct-f16.gguf",
        suite="edge",
        notes="Sub-4 GB embedded variant — exercises IQ3_M dot-product patterns.",
    ),
]


# ---------------------------------------------------------------------------
# Suite resolution
# ---------------------------------------------------------------------------

def select_suite(suite: str) -> list[ModelSpec]:
    """Resolve a --suite name to the set of ModelSpecs to fetch.

    Tiers are not strictly hierarchical: `edge` is its own axis (small / edge
    / robotics models + extra quants) and intentionally excludes the larger
    general-purpose VLMs in `default`. `full` includes everything.
    """
    if suite == "minimal":
        levels = {"minimal"}
    elif suite == "default":
        levels = {"minimal", "default"}
    elif suite == "edge":
        levels = {"minimal", "edge"}
    elif suite == "full":
        levels = {"minimal", "default", "edge", "full"}
    else:
        raise SystemExit(
            f"unknown --suite {suite!r}; expected minimal|default|edge|full"
        )
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
        choices=("minimal", "default", "edge", "full"),
        default="default",
        help=(
            "Which suite tier to download. 'edge' covers small / robotics / "
            "physical-AI VLMs plus Q8_0 / IQ3_M quants (default: default)."
        ),
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
