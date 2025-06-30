#!/usr/bin/env python
"""Generate synthetic deforestation images + masks.

Example
-------
python scripts/generate_synth.py \
    --forest-dir data/raw/forest \
    --deforested-dir data/raw/deforested \
    --out-dir data/synth \
    --num-samples 30000
"""
from __future__ import annotations

import sys, os
from pathlib import Path
# Ensure project root is on PYTHONPATH BEFORE any local imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import random
import concurrent.futures
import functools
import time

import cv2
from tqdm import tqdm

import numpy as np

from src.utils.synthetic_deforestation import create_synthetic_deforestation


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic deforestation dataset")
    p.add_argument("--forest-dir", type=str, required=True, help="Directory with forest-only images")
    p.add_argument("--deforested-dir", type=str, required=True, help="Directory with deforested/bare ground images")
    p.add_argument("--out-dir", type=str, default="data/synth", help="Output root dir where images/ and masks/ are created")
    p.add_argument("--num-samples", type=int, default=10000, help="Number of synthetic samples to create")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-area", type=float, default=0.4, help="Min deforested area ratio (e.g., 0.4 -> at least 40% of image)")
    p.add_argument("--max-area", type=float, default=0.8, help="Max deforested area ratio")
    p.add_argument("--shape", type=str, default="polygon", choices=["polygon", "rect", "ellipse", "roundrect", "blob", "none", "mixed"], help="Shape of synthetic clearing")
    p.add_argument("--shape-ratios", type=str, default=None, help="Comma separated ratios e.g. 'ellipse:0.5,rect:0.4,polygon:0.1'. Overrides --shape and chooses shape per sample with these probabilities")
    p.add_argument("--scale", type=float, default=0.5, help="Optional scaling factor (e.g., 0.5 halves width/height). 1.0 = original resolution")
    return p.parse_args()


def _gen_one(idx: int, shape_choice: str, forest_dir: str, deforest_dir: str, out_images: str, out_masks: str,
             seed: int, min_area: float, max_area: float, scale: float):
    """Worker function to generate a single synthetic image.

    Runs in a separate process (picklable top-level function).
    """
    import random, cv2, numpy as np, os
    from pathlib import Path
    from src.utils import synthetic_deforestation as synth_utils  # local import for sub-process

    attempts = 0
    while attempts < 8:
        attempts += 1
        rng = random.Random(seed + idx + attempts)

        # Gather candidate paths lazily per worker (cheap)
        forest_paths = sorted(list(Path(forest_dir).glob("*.png")) + list(Path(forest_dir).glob("*.jpg")))
        deforest_paths = sorted(list(Path(deforest_dir).glob("*.png")) + list(Path(deforest_dir).glob("*.jpg")))

        # robust read attempts
        f_path = rng.choice(forest_paths)
        d_path = rng.choice(deforest_paths)
        f_img = cv2.imread(str(f_path))
        d_img = cv2.imread(str(d_path))
        if f_img is None or d_img is None:
            continue
        f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)

        synth_utils._current_shape_choice = shape_choice  # type: ignore[attr-defined]
        composite, mask = synth_utils.create_synthetic_deforestation(
            f_img, d_img, min_area_ratio=min_area, max_area_ratio=max_area
        )

        if mask.sum() == 0:
            continue  # try again

        out_name = f"synth_{idx:06d}.png"
        cv2.imwrite(os.path.join(out_images, out_name), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_masks, out_name), (mask * 255).astype(np.uint8))
        return True
    return None


def main():
    args = parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    forest_paths = sorted(list(Path(args.forest_dir).glob("*.png")) + list(Path(args.forest_dir).glob("*.jpg")))
    deforest_paths = sorted(list(Path(args.deforested_dir).glob("*.png")) + list(Path(args.deforested_dir).glob("*.jpg")))

    assert forest_paths, f"No images found in {args.forest_dir}"
    assert deforest_paths, f"No images found in {args.deforested_dir}"

    out_images = Path(args.out_dir) / "images"
    out_masks = Path(args.out_dir) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    from src import utils as _u  # noqa: F401  placeholder to silence linter
    import src.utils.synthetic_deforestation as synth_utils

    # Determine shape choices per sample
    if args.shape_ratios:
        # Parse ratios string
        ratio_pairs = [kv.strip() for kv in args.shape_ratios.split(",") if kv.strip()]
        shapes, weights = [], []
        for pair in ratio_pairs:
            try:
                k, v = pair.split(":" if ":" in pair else "=")
            except ValueError:
                raise ValueError("Invalid --shape-ratios format. Expected 'shape:ratio' pairs")
            k = k.strip().lower()
            v = float(v.strip())
            if k in ("ellipse", "oval"):
                k = "ellipse"
            elif k in ("rect", "rectangle"):
                k = "rect"
            elif k in ("blob", "irregular"):
                k = "blob"
            elif k in ("none", "empty", "forest"):
                k = "none"
            elif k in ("roundrect", "rounded_rect", "rounded-rect", "roundedrect"):
                k = "roundrect"
            elif k == "polygon":
                k = "polygon"
            else:
                raise ValueError(f"Unsupported shape '{k}' in --shape-ratios")
            shapes.append(k)
            weights.append(v)
        if not np.isclose(sum(weights), 1.0):
            # Normalise so they sum to 1
            total = sum(weights)
            weights = [w / total for w in weights]
    else:
        shapes, weights = [args.shape], [1.0]

    # Pre-generate shape list for deterministic counts matching ratios
    shape_sequence = random.choices(shapes, weights=weights, k=args.num_samples)

    start_time = time.time()

    worker = functools.partial(
        _gen_one,
        forest_dir=args.forest_dir,
        deforest_dir=args.deforested_dir,
        out_images=str(out_images),
        out_masks=str(out_masks),
        seed=args.seed,
        min_area=args.min_area,
        max_area=args.max_area,
        scale=args.scale,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        list(tqdm(ex.map(worker, range(args.num_samples), shape_sequence), total=args.num_samples, desc="Generating synthetics"))

    elapsed = time.time() - start_time
    print(f"Done. Saved {args.num_samples} samples to {args.out_dir} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main() 