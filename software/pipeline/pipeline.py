"""
pipeline.py
-----------
Full inference pipeline: 16x16 grid -> embed -> FPGA layer 1 -> PC-side
residual+LayerNorm+FFN -> FPGA layer 2 -> PC-side residual+LayerNorm+FFN
-> output projection + sigmoid -> 16x16 grid.

Everything except the two attention layers runs on the PC, reusing the
exact trained submodules (embed, pos_embed, norm1/ffn/norm2, output_proj)
from pixel_completer.TinyTransformer -- nothing here reimplements the
model math, it's the same nn.Module calls demo() uses, with attn(x)
swapped out for a round trip to the FPGA.

The FPGA only ever computes raw attention (context @ W_out) for one layer
at a time -- see attention_fsm.sv. It has no concept of residual
connections, LayerNorm, or the feedforward network; those exist only in
the trained PyTorch model and have to run here, between the two FPGA
calls, or the result will not match what the model was trained to produce.

Weights must already be loaded onto the board (see load_weights.py) --
this script only sends inputs, not weights.

Usage:
    python pipeline.py --port COM3 [--model ../model/model_sketch.pt] [--pattern heart]
"""

import argparse
import json
import os
import sys

import numpy as np
import serial
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "driver"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from python_driver import FPGADriver  # noqa: E402
from pixel_completer import TinyTransformer  # noqa: E402
from create_dataset import mask_grid, print_grid  # noqa: E402

# attention_fsm.sv and weight_bram.sv hardcode 64x64 weight matrices and
# exactly 2 layers -- a mismatched checkpoint would run without error and
# produce garbage, so we refuse it up front.
EXPECTED_EMBED_DIM = 64
EXPECTED_NUM_LAYERS = 2


def load_model(model_path: str) -> TinyTransformer:
    checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")

    embed_dim = checkpoint["embed_dim"]
    num_layers = checkpoint.get("num_layers", 1)
    if embed_dim != EXPECTED_EMBED_DIM or num_layers != EXPECTED_NUM_LAYERS:
        raise ValueError(
            f"Checkpoint shape (embed_dim={embed_dim}, num_layers={num_layers}) "
            f"does not match what the FPGA is wired for (embed_dim={EXPECTED_EMBED_DIM}, "
            f"num_layers={EXPECTED_NUM_LAYERS})."
        )

    model = TinyTransformer(
        embed_dim=embed_dim,
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=num_layers,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class PixelCompletionPipeline:
    """
    Wraps a trained model + a live FPGA connection. Keeps the serial port
    open across repeated complete() calls -- interactive.py depends on
    this, since it calls complete() on a timer without reconnecting.
    """

    def __init__(self, model: TinyTransformer, driver: FPGADriver):
        self.model = model
        self.driver = driver

    @torch.no_grad()
    def complete(self, grid: np.ndarray):
        """
        Args:
            grid: (16, 16) array with values in {-1 (unknown), 0 (off), 1 (on)}.

        Returns:
            (16, 16) float32 numpy array of per-pixel probabilities, or
            None if an FPGA round trip failed.
        """
        x = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # (1, 16, 16)
        x = self.model.embed(x) + self.model.pos_embed  # (1, 16, 64)

        for layer_idx, block in enumerate(self.model.blocks):
            x_np = x.squeeze(0).numpy().astype(np.float32)  # (16, 64)

            attn_out_np = self.driver.run_attention_layer(layer_idx + 1, x_np)
            if attn_out_np is None:
                print(f"[pipeline] Layer {layer_idx + 1} FPGA round trip failed.")
                return None

            attn_out = torch.from_numpy(attn_out_np).unsqueeze(0)

            # Everything past this point -- residual, LayerNorm, FFN -- only
            # exists in the PyTorch model, never on the FPGA.
            x = block.norm1(x + attn_out)
            x = block.norm2(x + block.ffn(x))

        x = torch.sigmoid(self.model.output_proj(x))  # (1, 16, 16)
        return x.squeeze(0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Run one pixel-completion inference through the FPGA")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3")
    parser.add_argument("--baud", type=int, default=None, help="Override baud rate (default 921600)")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "..", "model", "model_sketch.pt"),
        help="Path to trained checkpoint (default: ../model/model_sketch.pt)",
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "dataset.json"),
        help="Path to dataset.json (default: ../data/dataset.json)",
    )
    parser.add_argument("--pattern", default=None, help="Pattern name/label to test (default: first entry)")
    parser.add_argument("--mask-fraction", type=float, default=0.4, help="Fraction of pixels to hide (default 0.4)")
    args = parser.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)

    item = None
    if args.pattern is not None:
        for entry in data:
            if entry["label"] == args.pattern or entry["name"] == args.pattern:
                item = entry
                break
        if item is None:
            print(f"[pipeline] No pattern named {args.pattern!r} found in {args.dataset}")
            sys.exit(1)
    else:
        item = data[0]

    masked, original = mask_grid(item["grid"], mask_fraction=args.mask_fraction)
    model = load_model(args.model)

    driver_kwargs = {"port": args.port}
    if args.baud is not None:
        driver_kwargs["baud_rate"] = args.baud

    try:
        with FPGADriver(**driver_kwargs) as driver:
            pipeline = PixelCompletionPipeline(model, driver)
            prediction = pipeline.complete(masked)
    except serial.SerialException as e:
        print(f"[pipeline] Could not open {args.port}: {e}")
        sys.exit(1)

    if prediction is None:
        print("[pipeline] Inference failed.")
        sys.exit(1)

    pred_binary = (prediction > 0.5).astype(int)
    acc = (pred_binary == np.array(original, dtype=int)).sum() / 256 * 100

    print(f"=== {item['name']} (accuracy: {acc:.0f}%) ===")
    print("Input (masked, ░░ = unknown):")
    print_grid(masked.tolist())
    print("Model output:")
    print_grid(pred_binary.tolist())


if __name__ == "__main__":
    main()
