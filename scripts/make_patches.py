"""Build training tiles with partial-label masks from the sparse tubule annotations.

Only a few tubules per section are annotated, so every pixel that is not known
for sure is marked as IGNORE and must be excluded from the loss.

Label values (``*_label.png``, uint8):
    0   background: thin ring outside each annotated tubule + glass (no tissue)
    1   tubule interior (annotated)
    2   tubule border: inner band along the annotated contour
    255 ignore: everything else, including unannotated tubules

``*_inst.png`` (uint16) holds the instance id of each annotated tubule (0 = none);
ids match the ``tubules.csv`` table.

Usage (from the repository root, inside the ``projeto_testiculos`` conda env)::

    python scripts/make_patches.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.strtree import STRtree
from tqdm import tqdm

BG, TUBULE, BORDER, IGNORE = 0, 1, 2, 255
PREVIEW_COLORS = {BG: (40, 90, 255), TUBULE: (0, 200, 0), BORDER: (255, 220, 0)}


def normalize_name(name: str | None) -> str:
    if not name:
        return "UNNAMED"
    return name.strip().upper().replace("Ã", "A")


def load_tubules(geojson_path: Path) -> list[dict]:
    """Polygon annotations only; keeps the largest part and fills brush holes."""
    tubules = []
    for feat in json.loads(geojson_path.read_text(encoding="utf-8"))["features"]:
        geom = shape(feat["geometry"])
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda p: p.area)
        if not isinstance(geom, Polygon):
            continue
        tubules.append({
            "object_id": feat.get("id"),
            "name": normalize_name(feat.get("properties", {}).get("name")),
            "polygon": Polygon(geom.exterior),
        })
    return tubules


def tissue_mask(slide: openslide.OpenSlide, min_area_um2: float, um_per_px0: float, sat_threshold: int):
    """Tissue mask at the lowest pyramid level.

    Glass has HSV saturation <= ~10 while pale tissue starts right above it, so a
    fixed low threshold is used: Otsu lands around 40 and drops pale tissue.
    """
    level = slide.level_count - 1
    downsample = slide.level_downsamples[level]
    rgba = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]))
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] == 0] = 255
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    mask = (sat > sat_threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    px_area_um2 = (downsample * um_per_px0) ** 2
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] * px_area_um2 >= min_area_um2
    return keep[comp].astype(np.uint8), downsample, rgb


def tissue_fragments(mask: np.ndarray, merge_px: int):
    """Connected tissue fragments after merging pieces closer than ``merge_px``."""
    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_px + 1,) * 2))
    _, comp = cv2.connectedComponents(merged, connectivity=8)
    return comp * (mask > 0)


def fragment_at(fragments: np.ndarray, x: float, y: float, radius: int = 6) -> int:
    xi, yi = int(round(x)), int(round(y))
    win = fragments[max(0, yi - radius):yi + radius + 1, max(0, xi - radius):xi + radius + 1]
    vals = win[win > 0]
    return int(np.bincount(vals).argmax()) if vals.size else 0


def crop_padded(arr: np.ndarray, r0: int, r1: int, c0: int, c1: int, fill=0) -> np.ndarray:
    out = np.full((r1 - r0, c1 - c0), fill, dtype=arr.dtype)
    sr0, sr1 = max(r0, 0), min(r1, arr.shape[0])
    sc0, sc1 = max(c0, 0), min(c1, arr.shape[1])
    if sr1 > sr0 and sc1 > sc0:
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = arr[sr0:sr1, sc0:sc1]
    return out


def glass_mask(tissue_dilated: np.ndarray, low_ds: float, x0: int, y0: int, side0: float, size_px: int) -> np.ndarray:
    """True where there is certainly no tissue, resampled to the tile grid."""
    c0, r0 = int(math.floor(x0 / low_ds)), int(math.floor(y0 / low_ds))
    c1, r1 = int(math.ceil((x0 + side0) / low_ds)), int(math.ceil((y0 + side0) / low_ds))
    low = crop_padded(tissue_dilated, r0, r1, c0, c1, fill=0)
    return cv2.resize((low == 0).astype(np.uint8), (size_px, size_px), interpolation=cv2.INTER_NEAREST) > 0


def read_tile(slide, x0: int, y0: int, size_px: int, downsample: float) -> np.ndarray:
    level = slide.get_best_level_for_downsample(downsample)
    level_ds = slide.level_downsamples[level]
    read_size = int(math.ceil(size_px * downsample / level_ds))
    region = slide.read_region((x0, y0), level, (read_size, read_size))
    canvas = Image.new("RGB", region.size, (255, 255, 255))
    canvas.paste(region, mask=region.split()[3])
    return np.asarray(canvas.resize((size_px, size_px), Image.BILINEAR))


def rasterize(poly: Polygon, x0: float, y0: float, downsample: float, size_px: int) -> np.ndarray:
    mask = np.zeros((size_px, size_px), dtype=np.uint8)
    pts = (np.asarray(poly.exterior.coords) - (x0, y0)) / downsample
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def build_label(tile_box, x0, y0, downsample, size_px, tubules, tree, glass, ring_px, border_px):
    label = np.full((size_px, size_px), IGNORE, dtype=np.uint8)
    inst = np.zeros((size_px, size_px), dtype=np.uint16)
    label[glass] = BG

    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1,) * 2)
    border_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border_px + 1,) * 2)
    masks = []
    for idx in tree.query(tile_box, predicate="intersects"):
        m = rasterize(tubules[idx]["polygon"], x0, y0, downsample, size_px)
        if m.any():
            masks.append((idx, m))
    for _, m in masks:
        ring = (cv2.dilate(m, ring_k) > 0) & (m == 0)
        label[ring] = BG
    for idx, m in masks:
        label[m > 0] = TUBULE
        inst[m > 0] = idx + 1
    for _, m in masks:
        border = (m > 0) & (cv2.erode(m, border_k, borderType=cv2.BORDER_REPLICATE) == 0)
        label[border] = BORDER
    return label, inst, [idx for idx, _ in masks]


def colorize(img: np.ndarray, label: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32).copy()
    for value, color in PREVIEW_COLORS.items():
        sel = label == value
        out[sel] = (1 - alpha) * out[sel] + alpha * np.asarray(color, dtype=np.float32)
    return out.astype(np.uint8)


def save_fragments_overview(rgb, fragments, tubules_lowres, out_path: Path):
    overlay = rgb.copy()
    rng = np.random.default_rng(0)
    for frag_id in range(1, fragments.max() + 1):
        sel = fragments == frag_id
        if sel.any():
            overlay[sel] = (0.5 * overlay[sel] + 0.5 * rng.integers(60, 255, 3)).astype(np.uint8)
            ys, xs = np.nonzero(sel)
            cv2.putText(overlay, str(frag_id), (int(xs.mean()), int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    for x, y in tubules_lowres:
        cv2.circle(overlay, (int(x), int(y)), 3, (255, 0, 0), -1)
    Image.fromarray(overlay).save(out_path)


def save_preview(samples, out_path: Path, thumb: int = 384, cols: int = 4):
    if not samples:
        return
    rows = math.ceil(len(samples) / cols)
    canvas = Image.new("RGB", (cols * thumb * 2, rows * thumb), "white")
    for i, (img, label) in enumerate(samples):
        pair = np.concatenate([img, colorize(img, label)], axis=1)
        tile = Image.fromarray(pair).resize((thumb * 2, thumb), Image.BILINEAR)
        canvas.paste(tile, ((i % cols) * thumb * 2, (i // cols) * thumb))
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations"))
    parser.add_argument("--slides", type=Path, default=Path("wsi_annotations"))
    parser.add_argument("--out", type=Path, default=Path("outputs/patches"))
    parser.add_argument("--um-per-px", type=float, default=1.0, help="tile resolution")
    parser.add_argument("--tile-px", type=int, default=1024, help="tile side in pixels at --um-per-px")
    parser.add_argument("--ring-um", type=float, default=4.0, help="background ring outside annotated tubules")
    parser.add_argument("--border-um", type=float, default=4.0, help="border band inside annotated tubules")
    parser.add_argument("--glass-margin-um", type=float, default=40.0, help="distance from tissue before glass counts as background")
    parser.add_argument("--fragment-merge-um", type=float, default=150.0, help="tissue pieces closer than this form one fragment")
    parser.add_argument("--min-tissue-um2", type=float, default=5e5, help="drop tissue specks smaller than this")
    parser.add_argument("--tissue-sat", type=int, default=15, help="HSV saturation above which a pixel is tissue")
    parser.add_argument("--preview", type=int, default=12, help="tiles per slide in the preview image")
    args = parser.parse_args()

    ring_px = max(1, round(args.ring_um / args.um_per_px))
    border_px = max(1, round(args.border_um / args.um_per_px))
    tiles_rows, tubule_rows = [], []

    for meta_path in sorted(args.annotations.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slide_id = meta["slide_id"]
        um0 = meta["pixel_width_um"]
        downsample = args.um_per_px / um0
        tile_side0 = args.tile_px * downsample
        slide_dir = args.out / slide_id
        slide_dir.mkdir(parents=True, exist_ok=True)

        slide = openslide.OpenSlide(str(args.slides / meta["image_name"]))
        tissue, low_ds, low_rgb = tissue_mask(slide, args.min_tissue_um2, um0, args.tissue_sat)
        fragments = tissue_fragments(tissue, max(1, round(args.fragment_merge_um / (low_ds * um0))))
        margin_px = max(1, round(args.glass_margin_um / (low_ds * um0)))
        tissue_dilated = cv2.dilate(tissue, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1,) * 2))
        Image.fromarray(tissue * 255).save(slide_dir / "tissue_mask.png")

        tubules = load_tubules(args.annotations / f"{slide_id}.geojson")
        tree = STRtree([t["polygon"] for t in tubules])
        centroids_low, fragment_ids = [], []
        for i, t in enumerate(tubules):
            c = t["polygon"].centroid
            centroids_low.append((c.x / low_ds, c.y / low_ds))
            fragment_ids.append(fragment_at(fragments, c.x / low_ds, c.y / low_ds))
            minx, miny, maxx, maxy = t["polygon"].bounds
            tubule_rows.append({
                "slide_id": slide_id, "instance_id": i + 1, "object_id": t["object_id"], "name": t["name"],
                "fragment_id": fragment_ids[i],
                "centroid_x": c.x, "centroid_y": c.y, "bbox_w_um": (maxx - minx) * um0, "bbox_h_um": (maxy - miny) * um0,
                "fits_in_tile": max(maxx - minx, maxy - miny) <= tile_side0,
            })
        save_fragments_overview(low_rgb, fragments, centroids_low, slide_dir / "fragments.png")

        preview_every = max(1, len(tubules) // max(1, args.preview))
        samples = []
        for i, t in enumerate(tqdm(tubules, desc=slide_id)):
            c = t["polygon"].centroid
            x0, y0 = int(round(c.x - tile_side0 / 2)), int(round(c.y - tile_side0 / 2))
            tile_box = box(x0, y0, x0 + tile_side0, y0 + tile_side0)

            img = read_tile(slide, x0, y0, args.tile_px, downsample)
            glass = glass_mask(tissue_dilated, low_ds, x0, y0, tile_side0, args.tile_px)

            label, inst, present = build_label(tile_box, x0, y0, downsample, args.tile_px,
                                               tubules, tree, glass, ring_px, border_px)

            tile_id = f"{slide_id}_{i + 1:04d}"
            Image.fromarray(img).save(slide_dir / f"{tile_id}_img.png")
            Image.fromarray(label).save(slide_dir / f"{tile_id}_label.png")
            cv2.imwrite(str(slide_dir / f"{tile_id}_inst.png"), inst)

            counts = np.bincount(label.ravel(), minlength=256)
            tiles_rows.append({
                "tile_id": tile_id, "slide_id": slide_id, "center_instance_id": i + 1, "name": t["name"],
                "fragment_id": fragment_ids[i],
                "x0": x0, "y0": y0, "downsample": downsample, "tile_px": args.tile_px,
                "n_instances": len(present),
                "frac_bg": counts[BG] / label.size, "frac_tubule": counts[TUBULE] / label.size,
                "frac_border": counts[BORDER] / label.size, "frac_ignore": counts[IGNORE] / label.size,
                "img": f"{slide_id}/{tile_id}_img.png", "label": f"{slide_id}/{tile_id}_label.png",
                "inst": f"{slide_id}/{tile_id}_inst.png",
            })
            if i % preview_every == 0 and len(samples) < args.preview:
                samples.append((img, label))
        slide.close()
        save_preview(samples, args.out / f"{slide_id}_preview.png")

    tiles = pd.DataFrame(tiles_rows)
    tubs = pd.DataFrame(tubule_rows)
    tiles.to_csv(args.out / "tiles.csv", index=False)
    tubs.to_csv(args.out / "tubules.csv", index=False)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    print(f"\nTiles: {len(tiles)}  ({args.tile_px}px at {args.um_per_px} um/px)")
    print(tiles.groupby("slide_id")[["frac_bg", "frac_tubule", "frac_border", "frac_ignore"]].mean().round(3).to_string())
    print("\nTubules per tissue fragment:")
    print(pd.crosstab([tubs["slide_id"], tubs["fragment_id"]], tubs["name"], margins=True).to_string())
    print(f"\nTubules larger than one tile: {int((~tubs['fits_in_tile']).sum())}")
    print(f"Outputs written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
