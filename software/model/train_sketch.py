"""
train_sketch.py
----------------
Retrains TinyTransformer for SKETCH COMPLETION instead of inpainting.

The original model (model_best.pt, trained by pixel_completer.train) learned
random-mask inpainting: it saw ~60% of a complete grid -- including real
background (off) pixels -- and filled the erased ~40%. That needs revealed
background to know where the shape ISN'T, which the drawing GUI never
provides (every un-drawn cell is left blank), so on a sketch it floods.

This trains the DIFFERENT task the GUI actually wants:

    input  = a random subset of a shape's LIT pixels, everything else OFF (0)
    target = the full clean shape

i.e. "turn on a few LEDs, the model lights up the rest of the shape."
Input is pure binary {0, 1} with no 'unknown' state -- an un-drawn cell is
just off, exactly like an unlit LED, matching how you draw in interactive.py.

The MODEL ARCHITECTURE is unchanged (embed_dim=64, num_layers=2) so the
attention weights still map onto the FPGA exactly as before. hidden_dim
only affects the PC-side FFN and never touches the board.

Trains only on the 414 clean canonical shapes (6 positional variants x 69
labels); the noisy dataset variants are skipped so completions snap to
clean shapes rather than reproducing noise.

Saves to model_sketch.pt -- it does NOT overwrite model_best.pt.

Usage:
    python train_sketch.py [--epochs 400] [--hidden-dim 32]
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pixel_completer import TinyTransformer

EMBED_DIM = 64      # fixed -- FPGA weight BRAM is wired for 64x64 attention matrices
NUM_LAYERS = 2      # fixed -- FPGA runs exactly 2 attention layers

KEEP_MIN = 0.30     # keep at least 30% of a shape's lit pixels as the sketch
KEEP_MAX = 0.70     # ...at most 70%, drawn fresh each __getitem__ for augmentation


def sketch_mask(grid: np.ndarray, keep_fraction: float):
    """
    Build one (partial-sketch input, full-shape target) pair.

    Keeps a random `keep_fraction` of the shape's ON pixels lit; every other
    cell -- dropped on-pixels AND all background -- is 0. The model must use
    the spatial pattern of the kept pixels to decide which 0s to light up.
    """
    grid = np.asarray(grid, dtype=np.float32)
    on = np.argwhere(grid == 1.0)
    inp = np.zeros_like(grid)

    if len(on) > 0:
        n_keep = max(1, int(round(len(on) * keep_fraction)))
        keep_idx = np.random.choice(len(on), size=n_keep, replace=False)
        for i in keep_idx:
            r, c = on[i]
            inp[r, c] = 1.0

    return inp, grid


class SketchDataset(Dataset):
    """Each __getitem__ redraws a fresh random partial sketch of its shape."""

    def __init__(self, grids):
        self.grids = grids

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        keep = np.random.uniform(KEEP_MIN, KEEP_MAX)
        inp, target = sketch_mask(self.grids[idx], keep)
        return torch.from_numpy(inp), torch.from_numpy(target)


def load_clean_grids(data_path: str):
    with open(data_path) as f:
        data = json.load(f)
    # 'noisy' variants carry baked-in pixel noise -- skip them so the model
    # completes to clean canonical shapes.
    return [d["grid"] for d in data if "noisy" not in d["name"]]


@torch.no_grad()
def completion_accuracy(model, grids, keep_fraction, seed):
    """Mean per-pixel accuracy of full-shape reconstruction from a partial sketch."""
    rng = np.random.RandomState(seed)
    model.eval()
    correct = total = 0
    for g in grids:
        g = np.asarray(g, dtype=np.float32)
        on = np.argwhere(g == 1.0)
        inp = np.zeros_like(g)
        if len(on):
            n_keep = max(1, int(round(len(on) * keep_fraction)))
            for i in rng.choice(len(on), size=n_keep, replace=False):
                r, c = on[i]
                inp[r, c] = 1.0
        pred = (model(torch.from_numpy(inp).unsqueeze(0)).squeeze(0).numpy() > 0.5)
        correct += (pred == (g == 1.0)).sum()
        total += g.size
    return correct / total * 100


def main():
    parser = argparse.ArgumentParser(description="Train TinyTransformer for sketch completion")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32,
                         help="FFN hidden width (PC-side only, does not affect FPGA)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "dataset.json")
    save_path = os.path.join(script_dir, "model_sketch.pt")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    grids = load_clean_grids(data_path)
    print(f"[train_sketch] {len(grids)} clean shapes, embed_dim={EMBED_DIM}, "
          f"hidden_dim={args.hidden_dim}, num_layers={NUM_LAYERS}")

    dataset = SketchDataset(grids)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TinyTransformer(embed_dim=EMBED_DIM, hidden_dim=args.hidden_dim, num_layers=NUM_LAYERS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_sketch] {n_params} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for inp, target in dataloader:
            pred = model(inp)
            loss = F.binary_cross_entropy(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "embed_dim": EMBED_DIM,
                "hidden_dim": args.hidden_dim,
                "num_layers": NUM_LAYERS,
                "epoch": epoch,
                "loss": best_loss,
                "task": "sketch_completion",
            }, save_path)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            acc50 = completion_accuracy(model, grids, 0.5, seed=args.seed)
            print(f"Epoch {epoch+1:4d}/{args.epochs} | loss {avg_loss:.4f} | "
                  f"completion acc @50%-kept: {acc50:.1f}%")

    print(f"\n[train_sketch] done. best loss {best_loss:.4f}. saved -> {save_path}")


if __name__ == "__main__":
    main()
