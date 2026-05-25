#!/usr/bin/env python3
"""generate_test_images.py — create the small synthetic JPEGs used by run_suite.

The suite needs 3 representative images of different sizes. Image content is
irrelevant for the profiler (the vision encoder graph topology depends on
image size, not pixel values), so we generate cheap synthetic patterns. The
outputs are committed under assets/ so smoke tests work without re-running
this script.

Outputs (under --out-dir):
  example_64.jpg    64x64    deterministic noise (smoke-test fixture)
  example_224.jpg   224x224  color-block grid     (typical CLIP ViT input)
  example_448.jpg   448x448  smooth radial gradient (typical LLaVA-1.6 patch)
"""

import argparse
import math
import os
import sys

try:
    from PIL import Image  # type: ignore
except ModuleNotFoundError:
    print(
        "ERROR: Pillow not installed. Run inside Docker (`make gen-test-images`).",
        file=sys.stderr,
    )
    sys.exit(2)


def make_noise(size: int, seed: int = 0xC0FFEE) -> Image.Image:
    """Deterministic per-pixel pseudo-noise. No numpy dependency."""
    # xorshift32 keeps the script self-contained; quality doesn't matter here.
    img = Image.new("RGB", (size, size))
    pix = img.load()
    assert pix is not None
    state = seed & 0xFFFFFFFF
    for y in range(size):
        for x in range(size):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= (state >> 17) & 0xFFFFFFFF
            state ^= (state << 5)  & 0xFFFFFFFF
            r = state & 0xFF
            g = (state >> 8) & 0xFF
            b = (state >> 16) & 0xFF
            pix[x, y] = (r, g, b)
    return img


def make_blocks(size: int) -> Image.Image:
    """Coarse color-block grid; visually distinct from the other fixtures."""
    img = Image.new("RGB", (size, size))
    pix = img.load()
    assert pix is not None
    cell = max(1, size // 8)
    palette = [
        (220,  60,  60), ( 60, 180,  90), ( 60, 110, 220), (220, 200,  60),
        (180,  60, 200), ( 60, 200, 200), (255, 140,   0), ( 30,  30,  30),
    ]
    for y in range(size):
        for x in range(size):
            idx = ((x // cell) + (y // cell)) % len(palette)
            pix[x, y] = palette[idx]
    return img


def make_radial(size: int) -> Image.Image:
    """Smooth radial gradient — exercises the encoder on a wider patch."""
    img = Image.new("RGB", (size, size))
    pix = img.load()
    assert pix is not None
    cx = cy = size / 2.0
    r_max = math.hypot(cx, cy)
    for y in range(size):
        for x in range(size):
            r = math.hypot(x - cx, y - cy) / r_max
            r = max(0.0, min(1.0, r))
            pix[x, y] = (
                int(255 * (1.0 - r)),
                int(255 * (0.5 + 0.5 * math.sin(6 * r))),
                int(255 * r),
            )
    return img


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out-dir",
        default="assets",
        help="Directory to write JPEGs into (default: ./assets).",
    )
    p.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality (default: 85).",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    targets = [
        ("example_64.jpg",  make_noise, 64),
        ("example_224.jpg", make_blocks, 224),
        ("example_448.jpg", make_radial, 448),
    ]

    for name, fn, size in targets:
        path = os.path.join(args.out_dir, name)
        img = fn(size)
        img.save(path, "JPEG", quality=args.quality)
        print(f"wrote {path}  ({size}x{size}, {os.path.getsize(path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
