"""Quality control of the tubule annotations exported from QuPath.

Reads the GeoJSON/meta files produced by ``scripts/qupath/export_geojson.groovy`` and writes:

* ``annotations.csv``          one row per annotation with geometry/morphometry
* ``<slide>_overview.png``     low-resolution slide with every annotation drawn
* ``<slide>_gallery_<name>.png`` crops around a sample of annotated tubules

Usage (from the repository root, inside the ``projeto_testiculos`` conda env)::

    python scripts/qc_annotations.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon, shape

COLORS = {"SIM": (0, 200, 0), "NAO": (255, 140, 0)}
DEFAULT_COLOR = (255, 0, 255)


def normalize_name(name: str | None) -> str:
    if not name:
        return "UNNAMED"
    return name.strip().upper().replace("Ã", "A")


def polygon_parts(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def load_slide_annotations(geojson_path: Path, meta: dict) -> pd.DataFrame:
    features = json.loads(geojson_path.read_text(encoding="utf-8"))["features"]
    um = meta["pixel_width_um"]
    rows = []
    for feat in features:
        geom = shape(feat["geometry"])
        parts = sorted(polygon_parts(geom), key=lambda p: p.area, reverse=True)
        props = feat.get("properties", {})
        row = {
            "slide_id": meta["slide_id"],
            "object_id": feat.get("id"),
            "name": normalize_name(props.get("name")),
            "geometry_type": geom.geom_type,
            "is_polygon": bool(parts),
            "n_parts": len(parts),
        }
        if parts:
            main = parts[0]
            area = sum(p.area for p in parts)
            minx, miny, maxx, maxy = main.bounds
            row.update(
                area_um2=area * um * um,
                main_part_area_fraction=main.area / area,
                n_holes=len(main.interiors),
                perimeter_um=main.exterior.length * um,
                circularity=4 * math.pi * main.area / main.exterior.length**2,
                equiv_diameter_um=2 * math.sqrt(main.area / math.pi) * um,
                bbox_aspect=max(maxx - minx, maxy - miny) / max(1.0, min(maxx - minx, maxy - miny)),
                centroid_x=main.centroid.x,
                centroid_y=main.centroid.y,
                minx=minx, miny=miny, maxx=maxx, maxy=maxy,
                n_vertices=len(main.exterior.coords),
                is_valid=geom.is_valid,
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_geometry"] = [shape(f["geometry"]) for f in features]
    return df


def draw_geometry(draw: ImageDraw.ImageDraw, geom, scale: float, offset=(0.0, 0.0), color=DEFAULT_COLOR, width=2):
    ox, oy = offset
    for part in polygon_parts(geom):
        for ring in [part.exterior, *part.interiors]:
            pts = [((x - ox) / scale, (y - oy) / scale) for x, y in ring.coords]
            draw.line(pts + [pts[0]], fill=color, width=width)
    if geom.geom_type == "LineString":
        pts = [((x - ox) / scale, (y - oy) / scale) for x, y in geom.coords]
        draw.line(pts, fill=DEFAULT_COLOR, width=width)


def render_overview(slide: openslide.OpenSlide, df: pd.DataFrame, out_path: Path):
    level = slide.level_count - 1
    downsample = slide.level_downsamples[level]
    img = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
    draw = ImageDraw.Draw(img)
    for geom, name in zip(df["_geometry"], df["name"]):
        draw_geometry(draw, geom, downsample, color=COLORS.get(name, DEFAULT_COLOR), width=2)
    img.save(out_path)


def read_crop(slide: openslide.OpenSlide, bounds, target_um_per_px: float, um_per_px: float, margin: float = 0.25):
    minx, miny, maxx, maxy = bounds
    w, h = maxx - minx, maxy - miny
    side = max(w, h) * (1 + 2 * margin)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    target_downsample = target_um_per_px / um_per_px
    level = slide.get_best_level_for_downsample(target_downsample)
    level_ds = slide.level_downsamples[level]
    size = int(math.ceil(side / level_ds))
    region = slide.read_region((x0, y0), level, (size, size)).convert("RGB")
    out_size = max(1, int(round(side / target_downsample)))
    return region.resize((out_size, out_size), Image.BILINEAR), (x0, y0), target_downsample


def render_gallery(slide, df, name, out_path: Path, um_per_px: float, n: int, tile: int, target_um_per_px: float, seed: int):
    subset = df[(df["name"] == name) & df["is_polygon"]]
    if subset.empty:
        return
    idx = list(subset.index)
    random.Random(seed).shuffle(idx)
    idx = idx[:n]
    cols = min(6, len(idx))
    rows = math.ceil(len(idx) / cols)
    canvas = Image.new("RGB", (cols * tile, rows * tile), "white")
    for i, row_idx in enumerate(idx):
        r = df.loc[row_idx]
        crop, offset, scale = read_crop(slide, (r.minx, r.miny, r.maxx, r.maxy), target_um_per_px, um_per_px)
        draw_geometry(ImageDraw.Draw(crop), r["_geometry"], scale, offset, COLORS.get(name, DEFAULT_COLOR), width=3)
        crop = crop.resize((tile, tile), Image.BILINEAR)
        ImageDraw.Draw(crop).text((4, 4), f"{r.equiv_diameter_um:.0f}um c={r.circularity:.2f}", fill=(0, 0, 0))
        canvas.paste(crop, ((i % cols) * tile, (i // cols) * tile))
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations"))
    parser.add_argument("--slides", type=Path, default=Path("wsi_annotations"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qc"))
    parser.add_argument("--gallery-size", type=int, default=24)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--um-per-px", type=float, default=1.0, help="resolution of gallery crops")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tables = []
    for meta_path in sorted(args.annotations.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slide_id = meta["slide_id"]
        df = load_slide_annotations(args.annotations / f"{slide_id}.geojson", meta)
        slide = openslide.OpenSlide(str(args.slides / meta["image_name"]))
        um = meta["pixel_width_um"]

        render_overview(slide, df, args.out / f"{slide_id}_overview.png")
        for name in sorted(df["name"].unique()):
            render_gallery(slide, df, name, args.out / f"{slide_id}_gallery_{name}.png",
                           um, args.gallery_size, args.tile, args.um_per_px, args.seed)
        slide.close()
        tables.append(df.drop(columns="_geometry"))

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.out / "annotations.csv", index=False)

    polys = table[table["is_polygon"]]
    print("\nAnnotations per slide (all geometry types):")
    print(pd.crosstab([table["slide_id"]], [table["name"], table["geometry_type"]], margins=True).to_string())
    print("\nMorphometry of polygon annotations (median [min-max]):")
    for (slide_id, name), g in polys.groupby(["slide_id", "name"]):
        def fmt(col, p=0):
            return f"{np.median(g[col]):.{p}f} [{g[col].min():.{p}f}-{g[col].max():.{p}f}]"
        print(f"  {slide_id:8s} {name:4s} n={len(g):3d}  diam_um={fmt('equiv_diameter_um')}  "
              f"circ={fmt('circularity', 2)}  aspect={fmt('bbox_aspect', 2)}  multipart={int((g['n_parts'] > 1).sum())}")
    multi = polys[polys["n_parts"] > 1]
    if not multi.empty:
        print(f"\nMultipart polygons: {len(multi)}; main part covers "
              f"{multi['main_part_area_fraction'].min():.1%}-{multi['main_part_area_fraction'].max():.1%} of area")
    print(f"\nInvalid geometries: {int((~polys['is_valid'].astype(bool)).sum())}")
    print(f"Outputs written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
