"""Train a U-Net for seminiferous tubule segmentation from partial-label tiles.

The network predicts 3 classes (background, tubule interior, tubule border);
pixels labelled 255 by ``make_patches.py`` are excluded from the loss. Instances
are recovered with a watershed seeded on the predicted interiors.

Validation uses whole held-out tissue fragments or slides, given as
``slide:fragments`` tokens (``3PAS:*`` = whole slide, ``7PAS80x:7,8`` = fragments).

Usage (from the repository root, inside the ``projeto_testiculos`` conda env)::

    python scripts/train.py --val 3PAS:* --run-name holdout_3PAS
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

BG, TUBULE, BORDER, IGNORE = 0, 1, 2, 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


def parse_split(spec: list[str], tiles: pd.DataFrame) -> np.ndarray:
    is_val = np.zeros(len(tiles), dtype=bool)
    for token in spec:
        slide_id, _, frags = token.partition(":")
        sel = tiles["slide_id"].to_numpy() == slide_id
        if frags and frags != "*":
            sel &= tiles["fragment_id"].isin([int(f) for f in frags.split(",")]).to_numpy()
        if not sel.any():
            raise ValueError(f"validation token {token!r} matches no tile")
        is_val |= sel
    return is_val


def seed_worker(_worker_id: int):
    np.random.seed(torch.initial_seed() % 2**32)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_gray(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1))


def build_augment(crop: int) -> A.Compose:
    return A.Compose([
        A.Affine(scale=(0.8, 1.25), rotate=(-45, 45), shear=(-5, 5), border_mode=cv2.BORDER_CONSTANT,
                 fill=255, fill_mask=IGNORE, p=0.7),
        A.CenterCrop(crop, crop),
        A.D4(),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=15, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.GaussNoise(std_range=(0.02, 0.06))], p=0.3),
    ])


class TrainTiles(Dataset):
    """Random crops around annotated tubules; the pre-crop margin absorbs affine borders."""

    def __init__(self, rows: pd.DataFrame, root: Path, crop: int, samples_per_tile: int, min_fg: float):
        self.rows = rows.reset_index(drop=True)
        self.root = root
        self.crop = crop
        self.pre_crop = int(math.ceil(crop * math.sqrt(2)))
        self.samples_per_tile = samples_per_tile
        self.min_fg = min_fg
        self.augment = build_augment(crop)

    def __len__(self):
        return len(self.rows) * self.samples_per_tile

    def __getitem__(self, i):
        row = self.rows.iloc[i % len(self.rows)]
        img = read_rgb(self.root / row["img"])
        label = read_gray(self.root / row["label"])
        h, w = label.shape
        size = min(self.pre_crop, h, w)
        fg = (label == TUBULE) | (label == BORDER)
        for _ in range(20):
            y = np.random.randint(0, h - size + 1)
            x = np.random.randint(0, w - size + 1)
            if fg[y:y + size, x:x + size].mean() >= self.min_fg:
                break
        out = self.augment(image=img[y:y + size, x:x + size], mask=label[y:y + size, x:x + size])
        return to_tensor(out["image"]), torch.from_numpy(out["mask"].astype(np.int64))


def loss_fn(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    valid = (target != IGNORE).float()
    if valid.sum() == 0:
        return logits.sum() * 0
    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=IGNORE)
    prob = logits.float().softmax(1)
    fg_prob = (prob[:, TUBULE] + prob[:, BORDER]) * valid
    fg_true = ((target == TUBULE) | (target == BORDER)).float() * valid
    inter = (fg_prob * fg_true).sum()
    dice = 1 - (2 * inter + 1) / (fg_prob.sum() + fg_true.sum() + 1)
    return ce + dice


def postprocess(prob: np.ndarray, fg_thr: float, seed_thr: float, min_seed_px: int, min_area_px: int) -> np.ndarray:
    """Instance map from class probabilities (C, H, W): watershed seeded on interiors."""
    fg = (prob[TUBULE] + prob[BORDER]) > fg_thr
    seeds = ((prob[TUBULE] > seed_thr) & fg).astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=4)
    markers = np.zeros(fg.shape, dtype=np.int32)
    markers[~fg] = 1
    next_id = 2
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_seed_px:
            markers[comp == k] = next_id
            next_id += 1
    relief = np.clip((1 - prob[TUBULE]) * 255, 0, 255).astype(np.uint8)
    markers = cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)
    inst = np.where(markers > 1, markers - 1, 0).astype(np.int32)
    inst[~fg] = 0
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    for k, c in zip(ids, counts):
        if c < min_area_px:
            inst[inst == k] = 0
    return inst


@torch.no_grad()
def predict(model, img: np.ndarray, device, amp: bool) -> np.ndarray:
    x = to_tensor(img)[None].to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        logits = model(x)
    return logits.float().softmax(1)[0].cpu().numpy()


def evaluate(model, rows: pd.DataFrame, root: Path, device, pp: dict, amp: bool, preview_n: int = 0):
    model.eval()
    inter = union = fp_bg = n_bg = tp_fg = n_fg = 0
    per_tubule, previews = [], []
    for _, row in rows.iterrows():
        img = read_rgb(root / row["img"])
        label = read_gray(root / row["label"])
        gt_inst = read_gray(root / row["inst"]).astype(np.int32)
        prob = predict(model, img, device, amp)
        pred_fg = prob.argmax(0) != BG
        valid = label != IGNORE
        true_fg = (label == TUBULE) | (label == BORDER)
        inter += (pred_fg & true_fg & valid).sum()
        union += ((pred_fg | true_fg) & valid).sum()
        fp_bg += (pred_fg & (label == BG)).sum()
        n_bg += (label == BG).sum()
        tp_fg += (pred_fg & true_fg).sum()
        n_fg += true_fg.sum()

        inst = postprocess(prob, **pp)
        gt = gt_inst == row["center_instance_id"]
        hits = inst[gt]
        best_iou = 0.0
        for k in np.unique(hits[hits > 0]):
            pk = inst == k
            best_iou = max(best_iou, (pk & gt).sum() / (pk | gt).sum())
        per_tubule.append({"tile_id": row["tile_id"], "name": row["name"], "iou": best_iou})
        if len(previews) < preview_n:
            previews.append((img, label, inst))

    tub = pd.DataFrame(per_tubule)
    metrics = {
        "fg_iou_labeled": inter / max(union, 1),
        "fg_recall": tp_fg / max(n_fg, 1),
        "bg_false_positive_rate": fp_bg / max(n_bg, 1),
        "inst_mean_iou": tub["iou"].mean(),
        "inst_recall_at_0.5": (tub["iou"] >= 0.5).mean(),
        "inst_recall_at_0.75": (tub["iou"] >= 0.75).mean(),
    }
    for name, g in tub.groupby("name"):
        metrics[f"inst_mean_iou_{name}"] = g["iou"].mean()
    return metrics, tub, previews


def save_preview(previews, path: Path, thumb: int = 384):
    rng = np.random.default_rng(0)
    rows = []
    for img, label, inst in previews:
        pred = img.copy()
        for k in np.unique(inst[inst > 0]):
            m = (inst == k).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(pred, contours, -1, tuple(int(c) for c in rng.integers(0, 255, 3)), 4)
        gt = img.copy()
        gt[label == BG] = (0.5 * gt[label == BG] + [20, 45, 128]).astype(np.uint8)
        gt[label == TUBULE] = (0.5 * gt[label == TUBULE] + [0, 100, 0]).astype(np.uint8)
        gt[label == BORDER] = (0.5 * gt[label == BORDER] + [128, 110, 0]).astype(np.uint8)
        trio = [cv2.resize(a, (thumb, thumb), interpolation=cv2.INTER_AREA) for a in (img, pred, gt)]
        rows.append(np.concatenate(trio, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches", type=Path, default=Path("outputs/patches"))
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--val", nargs="+", default=["3PAS:*"], help="held-out slide:fragments tokens")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-tile", type=int, default=2, help="random crops per tile per epoch")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--border-weight", type=float, default=2.0)
    parser.add_argument("--min-fg", type=float, default=0.03, help="min labelled tubule fraction per crop")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="fp16 autocast (produces NaN on GTX 16xx GPUs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fg-thr", type=float, default=0.5)
    parser.add_argument("--seed-thr", type=float, default=0.6)
    parser.add_argument("--min-seed-px", type=int, default=300)
    parser.add_argument("--min-area-px", type=int, default=1500)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiles = pd.read_csv(args.patches / "tiles.csv")
    is_val = parse_split(args.val, tiles)
    train_rows, val_rows = tiles[~is_val], tiles[is_val]
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.runs / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"train tiles: {len(train_rows)}  val tiles: {len(val_rows)}  device: {device}")

    loader = DataLoader(
        TrainTiles(train_rows, args.patches, args.crop, args.samples_per_tile, args.min_fg),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True,
        persistent_workers=args.workers > 0, pin_memory=True,
        worker_init_fn=seed_worker,
    )
    model = smp.Unet(args.encoder, encoder_weights="imagenet", classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    class_weights = torch.tensor([1.0, 1.0, args.border_weight], device=device)
    pp = dict(fg_thr=args.fg_thr, seed_thr=args.seed_thr, min_seed_px=args.min_seed_px, min_area_px=args.min_area_px)

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y, class_weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
        record = {"epoch": epoch, "train_loss": running / len(loader), "lr": scheduler.get_last_lr()[0],
                  "epoch_s": time.time() - t0}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics, per_tubule, previews = evaluate(model, val_rows, args.patches, device, pp, amp, preview_n=6)
            record.update(metrics)
            if metrics["inst_mean_iou"] > best:
                best = metrics["inst_mean_iou"]
                torch.save({"model": model.state_dict(), "encoder": args.encoder, "epoch": epoch, "metrics": metrics},
                           run_dir / "best.pt")
                per_tubule.to_csv(run_dir / "val_per_tubule_best.csv", index=False)
                save_preview(previews, run_dir / "val_preview_best.png")
        history.append(record)
        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
        msg = f"epoch {epoch:3d}  loss {record['train_loss']:.4f}  {record['epoch_s']:.0f}s"
        if "inst_mean_iou" in record:
            msg += (f"  | val fgIoU {record['fg_iou_labeled']:.3f}  bgFP {record['bg_false_positive_rate']:.3f}"
                    f"  instIoU {record['inst_mean_iou']:.3f}  R@0.5 {record['inst_recall_at_0.5']:.3f}"
                    f"  R@0.75 {record['inst_recall_at_0.75']:.3f}")
        print(msg, flush=True)

    torch.save({"model": model.state_dict(), "encoder": args.encoder, "epoch": args.epochs}, run_dir / "last.pt")
    print(f"best val instance IoU: {best:.3f}  -> {run_dir.resolve()}")


if __name__ == "__main__":
    main()
