"""Segment seminiferous tubules on whole slides and export them to QuPath.

Each tissue fragment is processed separately: overlapping windows are blended
into a probability map, instances are split with the same watershed used for
validation, and every instance becomes a polygon in level-0 pixel coordinates.

Outputs per slide (``--out``):
    <slide>_tubules.geojson   import into QuPath (drag onto the open image)
    <slide>_tubules.csv       one row per tubule with morphometry
    <slide>_overview.png      low-resolution slide with the predicted contours

The SIM/NAO name is a shape rule on circularity (calibrated on the annotated
tubules of 3PAS), not a learned prediction. Irregular or low-confidence objects,
typically loose connective tissue at section borders, are classified as
``Revisar`` instead of ``Tubulo`` so they can be checked in QuPath.

Usage (from the repository root, inside the ``projeto_testiculos`` conda env)::

    python scripts/infer_wsi.py --checkpoint outputs/runs/final_all/last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from scipy import ndimage

from make_patches import crop_padded, read_tile, tissue_fragments, tissue_mask
from train import BG, BORDER, TUBULE, postprocess, to_tensor

CLASS_COLORS = {"Tubulo": [0, 200, 0], "Revisar": [230, 0, 0]}


def window_weights(size: int, floor: float = 0.05) -> np.ndarray:
    w = np.hanning(size)
    w = np.maximum(w, floor)
    return np.outer(w, w).astype(np.float32)


def window_origins(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile, stride))
    origins.append(length - tile)
    return origins


@torch.no_grad()
def predict_fragment(model, slide, device, ox0: int, oy0: int, w_px: int, h_px: int, ds: float,
                     tissue_low: np.ndarray, low_ds: float, tile: int, stride: int, min_tissue: float):
    """Blended class probabilities (3, h_px, w_px) for a region starting at level-0 (ox0, oy0)."""
    acc = np.zeros((3, h_px, w_px), dtype=np.float32)
    weight = np.zeros((h_px, w_px), dtype=np.float32)
    win = window_weights(tile)
    n_run = n_skip = 0
    for ty in window_origins(h_px, tile, stride):
        for tx in window_origins(w_px, tile, stride):
            x0, y0 = ox0 + tx * ds, oy0 + ty * ds
            c0, r0 = int(x0 / low_ds), int(y0 / low_ds)
            c1, r1 = int(math.ceil((x0 + tile * ds) / low_ds)), int(math.ceil((y0 + tile * ds) / low_ds))
            patch = tissue_low[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)]
            if patch.size == 0 or patch.mean() < min_tissue:
                n_skip += 1
                continue
            img = read_tile(slide, int(round(x0)), int(round(y0)), tile, ds)
            prob = model(to_tensor(img)[None].to(device)).softmax(1)[0].cpu().numpy()
            th, tw = min(tile, h_px - ty), min(tile, w_px - tx)
            acc[:, ty:ty + th, tx:tx + tw] += prob[:, :th, :tw] * win[:th, :tw]
            weight[ty:ty + th, tx:tx + tw] += win[:th, :tw]
            n_run += 1
    covered = weight > 0
    acc[:, covered] /= weight[covered]
    acc[BG, ~covered] = 1.0
    return acc, n_run, n_skip


def restrict_to_region(prob: np.ndarray, region_low: np.ndarray, low_ds: float, ox0: int, oy0: int, ds: float):
    """Zero tubule probability outside ``region_low`` (low-res mask) so neighbouring fragments are not duplicated."""
    h_px, w_px = prob.shape[1:]
    c0, r0 = int(ox0 / low_ds), int(oy0 / low_ds)
    c1 = int(math.ceil((ox0 + w_px * ds) / low_ds))
    r1 = int(math.ceil((oy0 + h_px * ds) / low_ds))
    crop = crop_padded(region_low, r0, r1, c0, c1, fill=0)
    inside = cv2.resize(crop, (w_px, h_px), interpolation=cv2.INTER_NEAREST) > 0
    prob[TUBULE][~inside] = 0
    prob[BORDER][~inside] = 0
    prob[BG][~inside] = 1


def instance_polygons(inst: np.ndarray, prob: np.ndarray, simplify_px: float):
    """Largest external contour and confidence of every instance label."""
    out = []
    for k, sl in enumerate(ndimage.find_objects(inst), start=1):
        if sl is None:
            continue
        mask = (inst[sl] == k).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        cnt = cv2.approxPolyDP(cnt, simplify_px, True)
        if len(cnt) < 4:
            continue
        pts = cnt[:, 0, :].astype(np.float64) + [sl[1].start, sl[0].start]
        fg_prob = prob[TUBULE][sl] + prob[BORDER][sl]
        out.append((pts, mask, float(fg_prob[mask > 0].mean())))
    return out


def morphometry(pts_um: np.ndarray, mask_area_um2: float) -> dict:
    cnt = pts_um.astype(np.float32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    return {
        "area_um2": mask_area_um2,
        "perimeter_um": perimeter,
        "circularity": 4 * math.pi * area / perimeter**2 if perimeter else 0.0,
        "equiv_diameter_um": 2 * math.sqrt(mask_area_um2 / math.pi),
        "aspect_ratio": max(rw, rh) / max(min(rw, rh), 1e-6),
        "solidity": area / hull_area if hull_area else 0.0,
    }


def shape_name(m: dict, min_circularity: float) -> str:
    return "SIM" if m["circularity"] >= min_circularity else "NAO"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/runs/final_all/last.pt"))
    parser.add_argument("--slides", type=Path, nargs="+", default=None, help="default: every .svs in wsi_annotations")
    parser.add_argument("--out", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--um-per-px", type=float, default=1.0, help="must match training resolution")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--tissue-sat", type=int, default=15)
    parser.add_argument("--min-tissue-um2", type=float, default=5e5)
    parser.add_argument("--fragment-merge-um", type=float, default=150.0)
    parser.add_argument("--fragment-margin-um", type=float, default=200.0)
    parser.add_argument("--tissue-edge-um", type=float, default=50.0, help="keep predictions this far outside the tissue mask")
    parser.add_argument("--min-window-tissue", type=float, default=0.01, help="skip windows with less tissue than this")
    parser.add_argument("--fg-thr", type=float, default=0.5)
    parser.add_argument("--seed-thr", type=float, default=0.6)
    parser.add_argument("--min-seed-px", type=int, default=300)
    parser.add_argument("--min-area-um2", type=float, default=3000.0)
    parser.add_argument("--simplify-um", type=float, default=1.0)
    parser.add_argument("--sim-min-circularity", type=float, default=0.68)
    parser.add_argument("--review-max-solidity", type=float, default=0.6, help="below this -> 'Revisar'")
    parser.add_argument("--review-min-confidence", type=float, default=0.85, help="below this -> 'Revisar'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = smp.Unet(ckpt["encoder"], encoder_weights=None, classes=3).to(device).eval()
    model.load_state_dict(ckpt["model"])
    slides = args.slides or sorted(Path("wsi_annotations").glob("*.svs"))
    args.out.mkdir(parents=True, exist_ok=True)
    min_area_px = int(args.min_area_um2 / args.um_per_px**2)
    pp = dict(fg_thr=args.fg_thr, seed_thr=args.seed_thr, min_seed_px=args.min_seed_px, min_area_px=min_area_px)

    for slide_path in slides:
        t0 = time.time()
        slide_id = slide_path.stem
        slide = openslide.OpenSlide(str(slide_path))
        um0 = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
        ds = args.um_per_px / um0
        tissue, low_ds, low_rgb = tissue_mask(slide, args.min_tissue_um2, um0, args.tissue_sat)
        fragments = tissue_fragments(tissue, max(1, round(args.fragment_merge_um / (low_ds * um0))))
        margin0 = args.fragment_margin_um / um0
        edge_px = max(1, round(args.tissue_edge_um / (low_ds * um0)))
        W0, H0 = slide.dimensions

        features, rows = [], []
        frag_ids = [f for f in np.unique(fragments) if f > 0]
        for frag_id in frag_ids:
            ys, xs = np.nonzero(fragments == frag_id)
            ox0 = int(max(0, xs.min() * low_ds - margin0))
            oy0 = int(max(0, ys.min() * low_ds - margin0))
            ox1 = int(min(W0, (xs.max() + 1) * low_ds + margin0))
            oy1 = int(min(H0, (ys.max() + 1) * low_ds + margin0))
            w_px, h_px = int(math.ceil((ox1 - ox0) / ds)), int(math.ceil((oy1 - oy0) / ds))

            frag_tissue = (fragments == frag_id).astype(np.uint8)
            prob, n_run, n_skip = predict_fragment(
                model, slide, device, ox0, oy0, w_px, h_px, ds, frag_tissue, low_ds,
                args.tile, args.tile - args.overlap, args.min_window_tissue)
            region = cv2.dilate(frag_tissue, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_px + 1,) * 2))
            restrict_to_region(prob, region, low_ds, ox0, oy0, ds)
            inst = postprocess(prob, **pp)

            for pts, mask, conf in instance_polygons(inst, prob, args.simplify_um / args.um_per_px):
                pts_um = pts * args.um_per_px
                m = morphometry(pts_um, float(mask.sum()) * args.um_per_px**2)
                name = shape_name(m, args.sim_min_circularity)
                review = m["solidity"] < args.review_max_solidity or conf < args.review_min_confidence
                cls = "Revisar" if review else "Tubulo"
                coords0 = pts * ds + [ox0, oy0]
                ring = coords0.tolist() + [coords0[0].tolist()]
                obj_id = str(uuid.uuid4())
                features.append({
                    "type": "Feature", "id": obj_id,
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "objectType": "annotation", "name": name,
                        "classification": {"name": cls, "color": CLASS_COLORS[cls]},
                        "measurements": {**m, "confidence": conf, "fragment_id": int(frag_id)},
                    },
                })
                c = coords0.mean(0)
                rows.append({"slide_id": slide_id, "object_id": obj_id, "fragment_id": int(frag_id),
                             "classification": cls, "name": name,
                             "centroid_x": c[0], "centroid_y": c[1], "confidence": conf, **m})
            print(f"  {slide_id} fragment {frag_id}: {w_px}x{h_px}px, {n_run} windows ({n_skip} skipped), "
                  f"{int(inst.max())} instances", flush=True)
            del prob, inst

        (args.out / f"{slide_id}_tubules.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
        table = pd.DataFrame(rows)
        table.to_csv(args.out / f"{slide_id}_tubules.csv", index=False)

        overview = low_rgb.copy()
        for f in features:
            ring = (np.asarray(f["geometry"]["coordinates"][0]) / low_ds).astype(np.int32)
            if f["properties"]["classification"]["name"] == "Revisar":
                color = (230, 0, 0)
            else:
                color = (0, 170, 0) if f["properties"]["name"] == "SIM" else (255, 120, 0)
            cv2.polylines(overview, [ring], True, color, 1, cv2.LINE_AA)
        Image.fromarray(overview).save(args.out / f"{slide_id}_overview.png")
        slide.close()

        if table.empty:
            print(f"{slide_id}: no tubules found")
            continue
        tub = table[table["classification"] == "Tubulo"]
        counts = tub["name"].value_counts()
        print(f"{slide_id}: {len(tub)} tubules (SIM {counts.get('SIM', 0)}, NAO {counts.get('NAO', 0)}) + "
              f"{len(table) - len(tub)} to review, {len(frag_ids)} fragments, "
              f"median diameter {tub['equiv_diameter_um'].median():.0f} um, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
