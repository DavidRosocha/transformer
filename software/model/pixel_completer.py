"""
pixel_completer.py
A tiny transformer that completes 16x16 binary pixel art patterns.

Architecture:
  - Input: 16x16 grid with values {-1 (unknown), 0 (off), 1 (on)}
  - Reshape to 16 row tokens of 16 values each
  - Linear embedding: 16 -> embed_dim (16)
  - Learned positional embeddings (16 positions)
  - Single-head self-attention (manual Q/K/V, no nn.MultiheadAttention)
  - Residual + LayerNorm
  - FFN: embed_dim -> hidden_dim -> embed_dim
  - Residual + LayerNorm
  - Output projection -> sigmoid
  - Output: 16x16 grid of probabilities

Designed to map cleanly to FPGA hardware later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import random
import sys

# Add data directory to path so we can import mask_grid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from create_dataset import mask_grid, generate_dataset, print_grid


# ══════════════════════════════════════════════
#  MANUAL SINGLE-HEAD ATTENTION
#  (Written explicitly so it maps to FPGA ops)
# ══════════════════════════════════════════════

class FPGAAttention(nn.Module):
    """
    Single-head self-attention with explicit Q/K/V projections.
    No fancy tricks — just the raw matrix ops the FPGA will compute.

    Operations (maps to hardware):
      1. Q = x @ W_q      (matrix multiply)
      2. K = x @ W_k      (matrix multiply)
      3. V = x @ W_v      (matrix multiply)
      4. scores = Q @ K^T  (matrix multiply)
      5. attn = softmax(scores / sqrt(d))  (softmax — LUT on FPGA)
      6. out = attn @ V    (matrix multiply)
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5

        # Explicit weight matrices — these get loaded onto FPGA BRAM
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x shape: (batch, 16, embed_dim)

        # Step 1-3: Project to Q, K, V
        Q = self.W_q(x)  # (batch, 16, embed_dim)
        K = self.W_k(x)  # (batch, 16, embed_dim)
        V = self.W_v(x)  # (batch, 16, embed_dim)

        # Step 4: Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, 16, 16)

        # Step 5: Softmax (on FPGA this will be a LUT-based approximation)
        attn = F.softmax(scores, dim=-1)  # (batch, 16, 16)

        # Step 6: Weighted sum of values
        out = torch.matmul(attn, V)  # (batch, 16, embed_dim)

        # Output projection
        out = self.W_out(out)  # (batch, 16, embed_dim)

        return out


# ══════════════════════════════════════════════
#  TRANSFORMER MODEL
# ══════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """One transformer block: attention + FFN, each with residual + layernorm."""

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.attn = FPGAAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class TinyTransformer(nn.Module):
    """
    Tiny transformer for pixel pattern completion.

    Data flow:
      input (16x16) → reshape to (16, 16) row tokens
        → embed (16 → embed_dim) + positional encoding
        → N x [self-attention + residual + layernorm + FFN + residual + layernorm]
        → output projection (embed_dim → 16) + sigmoid
      output (16x16) probabilities
    """

    def __init__(self, embed_dim=16, hidden_dim=32, num_layers=1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Embedding: project each row of 16 pixel values to embed_dim
        self.embed = nn.Linear(16, embed_dim)

        # Learned positional embeddings for 16 row positions
        self.pos_embed = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)

        # Stacked transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Output projection: embed_dim back to 16 pixel values per row
        self.output_proj = nn.Linear(embed_dim, 16)

    def forward(self, x):
        # x shape: (batch, 16, 16) — 16 rows of 16 pixel values

        # Embed each row token
        x = self.embed(x)  # (batch, 16, embed_dim)

        # Add positional embeddings so model knows row 0 from row 15
        x = x + self.pos_embed  # (batch, 16, embed_dim)

        # Pass through stacked transformer blocks
        for block in self.blocks:
            x = block(x)  # (batch, 16, embed_dim)

        # Project back to 16 values per row
        x = self.output_proj(x)  # (batch, 16, 16)

        # Sigmoid to get probabilities (0-1 per pixel)
        x = torch.sigmoid(x)  # (batch, 16, 16)

        return x


# ══════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════

class PixelArtDataset(Dataset):
    """
    Wraps the generated patterns for PyTorch training.
    Each __getitem__ returns a different random masking of the same pattern,
    so the model sees different hints each epoch.
    """

    def __init__(self, data_path=None, data_list=None):
        if data_list is not None:
            self.data = data_list
        elif data_path is not None:
            with open(data_path) as f:
                self.data = json.load(f)
        else:
            raise ValueError("Provide data_path or data_list")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grid = self.data[idx]["grid"]

        # Apply random masking — different every time
        masked, original = mask_grid(grid)

        # masked: has -1 for unknown, 0 for off, 1 for on  → model input
        # original: all 0s and 1s                           → model target
        masked = torch.tensor(masked, dtype=torch.float32)     # (16, 16)
        original = torch.tensor(original, dtype=torch.float32) # (16, 16)

        return masked, original


# ══════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════

def train(
    epochs=200,
    batch_size=32,
    lr=1e-3,
    embed_dim=16,
    hidden_dim=32,
    num_layers=1,
    data_path=None,
    save_dir=None,
):
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if data_path is None:
        data_path = os.path.join(script_dir, "..", "data", "dataset.json")
    if save_dir is None:
        save_dir = script_dir

    # Check if dataset exists, generate if not
    if not os.path.exists(data_path):
        print("Dataset not found, generating...")
        random.seed(42)
        np.random.seed(42)
        dataset = generate_dataset()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w") as f:
            json.dump(dataset, f)
        print(f"Generated {len(dataset)} patterns")

    # Load dataset
    dataset = PixelArtDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Dataset: {len(dataset)} examples")
    print(f"Model: embed_dim={embed_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print()

    # Create model
    model = TinyTransformer(embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Parameter memory (Q8.8): {total_params * 2 / 1024:.1f} KB")
    print()

    # Optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Training loop
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for masked, original in dataloader:
            # Forward pass
            prediction = model(masked)  # (batch, 16, 16)

            # Binary cross-entropy loss on full grid
            loss = F.binary_cross_entropy(prediction, original)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Step the learning rate scheduler
        scheduler.step()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Compute accuracy (threshold at 0.5)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for masked, original in dataloader:
                    pred = model(masked)
                    pred_binary = (pred > 0.5).float()
                    correct += (pred_binary == original).sum().item()
                    total += original.numel()
            accuracy = correct / total * 100
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}% | LR: {current_lr:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, "model_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "epoch": epoch,
                "loss": best_loss,
            }, save_path)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")

    return model


# ══════════════════════════════════════════════
#  INFERENCE / DEMO
# ══════════════════════════════════════════════

def demo(model_path=None):
    """Load a trained model and show some completions."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, "model_best.pt")

    # Load model
    checkpoint = torch.load(model_path, weights_only=True)
    model = TinyTransformer(
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint.get("num_layers", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']+1}, loss {checkpoint['loss']:.4f}\n")

    # Load some test patterns
    data_path = os.path.join(script_dir, "..", "data", "dataset.json")
    with open(data_path) as f:
        data = json.load(f)

    # Pick a few diverse examples
    seen_labels = set()
    examples = []
    # Prioritize recognizable patterns for demo
    demo_labels = ["heart", "smiley", "arrow_up", "skull", "cat", "rocket", "ghost", "crown", "letter_A", "tree"]
    label_lookup = {}
    for item in data:
        if "noisy" not in item["name"]:
            if item["label"] not in label_lookup:
                label_lookup[item["label"]] = item

    for label in demo_labels:
        if label in label_lookup:
            examples.append(label_lookup[label])
        if len(examples) >= 8:
            break

    # If we didn't find enough, fill from whatever is available
    for item in data:
        if len(examples) >= 8:
            break
        if item["label"] not in [e["label"] for e in examples] and "noisy" not in item["name"]:
            examples.append(item)

    for item in examples:
        grid = item["grid"]
        # Use 40% masking — realistic for actual use (user draws ~60% of hints)
        masked, original = mask_grid(grid, mask_fraction=0.4)

        # Run model
        with torch.no_grad():
            input_tensor = torch.tensor(masked, dtype=torch.float32).unsqueeze(0)  # (1, 16, 16)
            prediction = model(input_tensor).squeeze(0)  # (16, 16)
            pred_binary = (prediction > 0.5).int().numpy()

        # Calculate accuracy for this example
        original_np = np.array(original, dtype=int)
        acc = (pred_binary == original_np).sum() / 256 * 100

        print(f"=== {item['name']} (accuracy: {acc:.0f}%) ===")
        print("Original:")
        print_grid(original.tolist())
        print("Input (masked, ░░ = unknown):")
        print_grid(masked.tolist())
        print("Model output:")
        print_grid(pred_binary.tolist())
        print()


# ══════════════════════════════════════════════
#  EXPORT WEIGHTS (for FPGA team)
# ══════════════════════════════════════════════

def export_weights(model_path=None, output_dir=None):
    """
    Export model weights as numpy arrays for the FPGA team.
    They'll convert these to Q8.8 fixed-point for hardware.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, "model_best.pt")
    if output_dir is None:
        output_dir = os.path.join(script_dir, "..", "..", "test_vectors")

    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(model_path, weights_only=True)
    model = TinyTransformer(
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint.get("num_layers", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Exporting weights:")
    for name, param in model.named_parameters():
        arr = param.detach().numpy()
        filepath = os.path.join(output_dir, f"{name.replace('.', '_')}.npy")
        np.save(filepath, arr)
        print(f"  {name:30s} shape={str(arr.shape):15s} saved to {filepath}")

    print(f"\nAll weights saved to {output_dir}")
    print("FPGA team: convert these to Q8.8 format (multiply by 256, round to int16)")


# ══════════════════════════════════════════════
#  GENERATE TEST VECTORS (for hardware verification)
# ══════════════════════════════════════════════

def generate_test_vectors(model_path=None, output_dir=None, num_vectors=3):
    """
    Generate test vectors with input and expected output in Q8.8 format.
    These are used by the FPGA team to verify their hardware produces
    the same results as the PyTorch model.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, "model_best.pt")
    if output_dir is None:
        output_dir = os.path.join(script_dir, "..", "..", "test_vectors")

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    checkpoint = torch.load(model_path, weights_only=True)
    model = TinyTransformer(
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint.get("num_layers", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset for test inputs
    data_path = os.path.join(script_dir, "..", "data", "dataset.json")
    with open(data_path) as f:
        data = json.load(f)

    def float_to_q88(val):
        """Convert float to Q8.8 fixed-point (16-bit signed)."""
        fixed = int(round(val * 256))
        # Clamp to Q8.8 range
        fixed = max(-32768, min(32767, fixed))
        # Return as unsigned 16-bit for hex representation
        if fixed < 0:
            fixed = fixed + 65536
        return fixed

    print(f"Generating {num_vectors} test vectors...")

    for i in range(num_vectors):
        item = data[i * 50]  # Spread out across dataset
        grid = item["grid"]
        masked, original = mask_grid(grid, mask_fraction=0.65)

        # Run model and capture intermediates
        with torch.no_grad():
            input_tensor = torch.tensor(masked, dtype=torch.float32).unsqueeze(0)
            output_tensor = model(input_tensor).squeeze(0)

        # Write hex file
        hex_path = os.path.join(output_dir, f"test_{i:02d}.hex")
        with open(hex_path, "w") as f:
            f.write(f"// Test vector {i}: {item['name']}\n")
            f.write(f"// Format: Q8.8 fixed-point (16-bit)\n\n")

            # Input (masked grid)
            f.write("// INPUT (16x16 masked grid, -1=unknown, 0=off, 1=on)\n")
            for r in range(16):
                row_hex = []
                for c in range(16):
                    val = masked[r][c]
                    row_hex.append(f"{float_to_q88(val):04X}")
                f.write(" ".join(row_hex) + "\n")

            f.write("\n")

            # Expected output (model prediction, pre-sigmoid values would be
            # more useful for hardware but we'll include both)
            f.write("// EXPECTED OUTPUT (16x16 probabilities after sigmoid)\n")
            for r in range(16):
                row_hex = []
                for c in range(16):
                    val = output_tensor[r][c].item()
                    row_hex.append(f"{float_to_q88(val):04X}")
                f.write(" ".join(row_hex) + "\n")

            f.write("\n")

            # Target (ground truth binary grid)
            f.write("// TARGET (16x16 ground truth binary)\n")
            for r in range(16):
                row_hex = []
                for c in range(16):
                    val = original[r][c]
                    row_hex.append(f"{float_to_q88(val):04X}")
                f.write(" ".join(row_hex) + "\n")

        print(f"  Saved {hex_path} ({item['name']})")

    print("Done. FPGA team can use these to verify hardware output matches.")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pixel art pattern completer")
    parser.add_argument("command", choices=["train", "demo", "export", "test_vectors"],
                        help="What to do")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)

    args = parser.parse_args()

    if args.command == "train":
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    elif args.command == "demo":
        demo()
    elif args.command == "export":
        export_weights()
    elif args.command == "test_vectors":
        generate_test_vectors()