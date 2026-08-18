"""
load_weights.py
----------------
Extracts the 8 attention weight matrices from a trained TinyTransformer
checkpoint (model_best.pt) and loads them into the FPGA's weight BRAM
over UART. Run this once after every board power-up / reprogram, before
pipeline.py or interactive.py can do inference.

Two things the checkpoint's raw tensors are NOT ready for:

  1. Orientation. nn.Linear stores its weight as (out_features, in_features)
     so that y = x @ weight.T. The FPGA's tile_controller computes
     Q = X @ W_q directly (attention_fsm.sv, PH_Q: "X x W_q -> Q"), so
     every matrix must be transposed before it's sent.

  2. Scale. FPGAAttention.forward() divides raw scores by sqrt(embed_dim)
     (= 8, since embed_dim=64) before softmax. attention_fsm.sv never
     performs that division -- there is no shift/divide anywhere in its
     score path. So the 1/8 factor is pre-baked into W_q1 and W_q2 before
     quantization. K, V, and W_out are sent unscaled.

Verified against the trained model directly: extracting + transposing +
prescaling reproduces FPGAAttention.forward()'s float output to ~1e-7
(pure float rounding noise), before any Q8.8 quantization.

Usage:
    python load_weights.py --port COM6 [--model ../model/model_sketch.pt]
"""

import argparse
import os
import sys

import numpy as np
import serial
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "driver"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from python_driver import FPGADriver, WEIGHT_ORDER, Q8_8_MIN, Q8_8_MAX  # noqa: E402
from pixel_completer import TinyTransformer  # noqa: E402

# attention_fsm.sv and weight_bram.sv hardcode 64x64 weight matrices and
# exactly 2 layers. A checkpoint trained with different dimensions would
# load without error and produce garbage, so we refuse to send it.
EXPECTED_EMBED_DIM = 64
EXPECTED_NUM_LAYERS = 2


def load_model(model_path: str) -> TinyTransformer:
    checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")

    embed_dim = checkpoint["embed_dim"]
    num_layers = checkpoint.get("num_layers", 1)
    if embed_dim != EXPECTED_EMBED_DIM or num_layers != EXPECTED_NUM_LAYERS:
        raise ValueError(
            f"Checkpoint shape (embed_dim={embed_dim}, num_layers={num_layers}) "
            f"does not match what the FPGA hardware is wired for "
            f"(embed_dim={EXPECTED_EMBED_DIM}, num_layers={EXPECTED_NUM_LAYERS})."
        )

    model = TinyTransformer(
        embed_dim=embed_dim,
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=num_layers,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"[load_weights] Loaded checkpoint from epoch {checkpoint['epoch'] + 1}, "
        f"loss {checkpoint['loss']:.4f}"
    )
    return model


def extract_weights(model: TinyTransformer) -> dict:
    """
    Pull the 8 attention matrices out of the model, transposed and scaled
    into the form the FPGA expects. Returns float32 numpy arrays, NOT yet
    Q8.8-quantized -- FPGADriver.load_weights() does that.
    """
    scale = model.blocks[0].attn.scale  # sqrt(embed_dim), same for every layer

    weights = {}
    for layer_idx, block in enumerate(model.blocks):
        layer_num = layer_idx + 1
        attn = block.attn

        w_q = attn.W_q.weight.detach().numpy().T.astype(np.float32)
        w_k = attn.W_k.weight.detach().numpy().T.astype(np.float32)
        w_v = attn.W_v.weight.detach().numpy().T.astype(np.float32)
        w_out = attn.W_out.weight.detach().numpy().T.astype(np.float32)

        w_q = w_q / scale  # bake in the 1/sqrt(embed_dim) score scaling

        weights[f"W_q{layer_num}"] = w_q
        weights[f"W_k{layer_num}"] = w_k
        weights[f"W_v{layer_num}"] = w_v
        weights[f"W_out{layer_num}"] = w_out

    return weights


def report_ranges(weights: dict) -> bool:
    """
    Print min/max for each matrix before it goes over the wire. Values
    outside the Q8.8 range are silently clamped by float_to_q8_8() -- that
    would corrupt the model without erroring, so we catch it here instead.
    Returns False if anything would clip.
    """
    print(f"\n[load_weights] Matrix ranges (Q8.8 range is [{Q8_8_MIN}, {Q8_8_MAX:.3f}]):")
    any_risk = False
    for name in WEIGHT_ORDER:
        mat = weights[name]
        lo, hi = float(mat.min()), float(mat.max())
        risk = lo < Q8_8_MIN or hi > Q8_8_MAX
        any_risk = any_risk or risk
        flag = "  <-- WOULD CLIP" if risk else ""
        print(f"    {name:8s} min={lo:8.4f} max={hi:8.4f}{flag}")

    if any_risk:
        print("[load_weights] WARNING: one or more matrices exceed the Q8.8 range.")

    return not any_risk


def main():
    parser = argparse.ArgumentParser(description="Load trained attention weights onto the FPGA")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM6 -- find yours with: python -m serial.tools.list_ports -v")
    parser.add_argument("--baud", type=int, default=None, help="Override baud rate (default 4000000)")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "..", "model", "model_sketch.pt"),
        help="Path to trained checkpoint (default: ../model/model_sketch.pt)",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    weights = extract_weights(model)

    if not report_ranges(weights):
        sys.exit(1)

    driver_kwargs = {"port": args.port}
    if args.baud is not None:
        driver_kwargs["baud_rate"] = args.baud

    try:
        with FPGADriver(**driver_kwargs) as driver:
            ok = driver.load_weights(weights)
    except serial.SerialException as e:
        print(f"[load_weights] Could not open {args.port}: {e}")
        sys.exit(1)

    if not ok:
        print("[load_weights] FAILED -- not all matrices were ACKed.")
        sys.exit(1)

    print("[load_weights] Done. Weights are resident in FPGA BRAM.")


if __name__ == "__main__":
    main()
