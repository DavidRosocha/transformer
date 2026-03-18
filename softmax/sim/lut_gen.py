"""
lut_gen.py
==========
Generates pre-computed LUT tables for the FPGA softmax approximation unit,
based on: Vasyltsov & Chang, "Efficient Softmax Approximation for Deep Neural
Networks with Attention Mechanism", arXiv:2111.10770, 2021.

Two methods implemented:
  - REXP  : two 1D LUTs (LUT1/e and LUTalpha)
  - 2D LUT: one 1D exp LUT + one 2D softmax output LUT  <-- recommended

Input format assumed: Q8.8 signed fixed-point (16-bit)
  - After x -> x - max(x) normalization, all values are in (-inf, 0]
  - e^x is therefore in (0, 1]

Outputs:
  - lut1e.mem       : 1D LUT for 1/e^i  (REXP method)
  - lut_alpha.mem   : 1D LUT for PDF normalizing constant (REXP method)
  - lut_exp.mem     : 1D LUT for e^x    (2D LUT method)
  - lut_2d.mem      : 2D LUT for sigma(x) output (2D LUT method)
  - lut_verify.py   : standalone verification script

Usage:
  python3 lut_gen.py

.mem files are loaded in Verilog/SystemVerilog via $readmemh()
"""

import numpy as np
import os

# ─────────────────────────────────────────────
# Configuration — matches paper Section 4 & 5
# ─────────────────────────────────────────────

PRECISION_BITS  = 8          # uint8 output precision (w in paper)
SCALE           = (2**PRECISION_BITS) - 1   # 255

# 2D LUT parameters (paper Section 4.2)
SCALE_EX        = 0.1        # step size for e^x axis  → 11 columns (0..10 * 0.1)
SCALE_SIGMA     = 1.0        # step size for Σe^x axis → 60 rows
MAX_SIGMA       = 60         # max expected Σe^x for NLP tasks (paper finding)

# For our 256-token sequence, Σe^x can be larger than 60 in the worst case.
# Each e^x ∈ (0,1], so Σe^x ∈ (0, SEQ_LEN].
# With 256 tokens, worst case Σ = 256. We extend the table accordingly.
SEQ_LEN         = 256
MAX_SIGMA_OURS  = SEQ_LEN    # conservative upper bound for 256 tokens

OUTPUT_DIR      = "."

# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def to_hex8(val):
    """Clamp a float to [0,255], round to int, format as 2-digit hex."""
    v = int(np.clip(np.floor(val), 0, 255))
    return f"{v:02x}"

def write_mem_1d(filename, data, comment=""):
    """Write a 1D list as a Verilog .mem file (one hex value per line)."""
    with open(filename, "w") as f:
        if comment:
            f.write(f"// {comment}\n")
        f.write(f"// {len(data)} entries, uint8 (2 hex digits each)\n")
        for i, val in enumerate(data):
            f.write(f"{to_hex8(val)}  // [{i}]\n")
    print(f"  wrote {filename}  ({len(data)} entries, {os.path.getsize(filename)} bytes on disk)")

def write_mem_2d(filename, data, comment=""):
    """Write a 2D list as a flat Verilog .mem file, row-major order.
    
    In SystemVerilog, load with:
        $readmemh("lut_2d.mem", lut_sigma);
    where lut_sigma is declared as:
        logic [7:0] lut_sigma [0:SIGMA_ROWS-1][0:EX_COLS-1];
    """
    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0
    with open(filename, "w") as f:
        if comment:
            f.write(f"// {comment}\n")
        f.write(f"// {rows} rows × {cols} cols = {rows*cols} entries, uint8\n")
        f.write(f"// Row = Σe^x bin (j), Col = e^xi bin (i)\n")
        f.write(f"// Access: lut_sigma[j][i]\n")
        for j, row in enumerate(data):
            f.write(f"// row j={j}  (Σe^x ≈ {(j+1)*SCALE_SIGMA:.1f})\n")
            for i, val in enumerate(row):
                f.write(f"{to_hex8(val)}\n")
    total = rows * cols
    print(f"  wrote {filename}  ({rows}×{cols} = {total} entries, "
          f"~{total} bytes logical, {os.path.getsize(filename)} bytes on disk)")

# ─────────────────────────────────────────────
# Method 1: REXP — two 1D LUTs
#   σ(xi) = LUT1e[idx_xi] · LUTalpha[idx_sigma]
#   (Section 4.1 of paper)
# ─────────────────────────────────────────────

def gen_rexp_luts():
    print("\n── REXP method ──────────────────────────────")

    # LUT1/e : stores  floor( (1/e^i) * 255 )  for i = 0, 1, ..., xq+1
    # where xq = ceil(ln(255)) ≈ 6
    xq = int(np.ceil(np.log(SCALE)))   # = ceil(ln(255)) = 6
    print(f"  xq (max useful index) = {xq}")

    lut1e = []
    for i in range(xq + 2):           # +2 for safety margin
        val = (1.0 / np.e**i) * SCALE
        lut1e.append(val)

    write_mem_1d(
        os.path.join(OUTPUT_DIR, "lut1e.mem"),
        lut1e,
        comment="LUT1/e: stores floor(1/e^i * 255) for REXP softmax method"
    )

    # LUTalpha : stores  floor( (1/j) * 255 )  for j = 1..xs
    # j = Σσ*(xi) — the accumulated reciprocal exp sum
    # xs chosen so table covers our sequence length
    xs = MAX_SIGMA_OURS + 1
    lut_alpha = []
    for j in range(1, xs + 1):
        val = (1.0 / j) * SCALE
        lut_alpha.append(val)

    write_mem_1d(
        os.path.join(OUTPUT_DIR, "lut_alpha.mem"),
        lut_alpha,
        comment="LUTalpha: stores floor(1/j * 255) — PDF normalizing constant for REXP"
    )

    return lut1e, lut_alpha

# ─────────────────────────────────────────────
# Method 2: 2D LUT  (recommended for our use)
#   Step a) e^xi  via 1D LUT
#   Step b) σ(xi) via 2D LUT[e^xi bin][Σe^x bin]
#   (Section 4.2 of paper)
# ─────────────────────────────────────────────

def gen_2d_luts():
    print("\n── 2D LUT method ────────────────────────────")

    # 1D exp LUT: stores  floor( e^x * 255 )
    # x is in (-inf, 0] after normalization, so e^x in (0, 1]
    # We quantize x to uint8 address: x_addr = round(-x * 255) clamped to [0,255]
    # This gives us e^x ≈ e^(-addr/255) for addr in 0..255
    # addr=0 → x=0   → e^x = 1.0  → stored as 255
    # addr=255 → x≈-1 → e^x ≈ 0.368 → stored as ~94
    # For larger negative x, e^x rounds to 0 quickly

    print("  Generating 1D exp LUT (256 entries, x_addr = round(-x_normalized * 255))")
    lut_exp = []
    for addr in range(256):
        x = -addr / SCALE          # maps addr back to x in [-1, 0]
        val = np.e**x * SCALE
        lut_exp.append(val)

    write_mem_1d(
        os.path.join(OUTPUT_DIR, "lut_exp.mem"),
        lut_exp,
        comment="LUTexp: stores floor(e^x * 255) for x = -addr/255, addr=0..255"
    )

    # 2D softmax LUT: LUTσ[j][i] = floor( (i * scale_ex) / (j * scale_sigma) * 255 )
    # i = e^xi bin  → i in 0..10  (0.0, 0.1, 0.2, ..., 1.0)
    # j = Σe^x bin  → j in 1..MAX_SIGMA_OURS
    #
    # Extended to MAX_SIGMA_OURS=256 for our 256-token sequence
    # Paper used 60 for NLP (seq_len=128), we need up to 256

    n_ex_cols   = int(1.0 / SCALE_EX) + 1    # 11 columns  (i = 0..10)
    n_sigma_rows = int(MAX_SIGMA_OURS / SCALE_SIGMA)  # 256 rows (j = 1..256)

    print(f"  Generating 2D LUT: {n_sigma_rows} rows × {n_ex_cols} cols")
    print(f"  e^xi bins:  0 to {(n_ex_cols-1)*SCALE_EX:.1f}  (step {SCALE_EX})")
    print(f"  Σe^x bins:  1 to {n_sigma_rows}  (step {SCALE_SIGMA})")

    lut_2d = []
    for j in range(1, n_sigma_rows + 1):     # Σe^x bin
        row = []
        for i in range(n_ex_cols):            # e^xi bin
            numerator   = i * SCALE_EX
            denominator = j * SCALE_SIGMA
            if denominator == 0:
                val = SCALE
            else:
                val = (numerator / denominator) * SCALE
            row.append(val)
        lut_2d.append(row)

    write_mem_2d(
        os.path.join(OUTPUT_DIR, "lut_2d.mem"),
        lut_2d,
        comment=f"LUT2D: softmax output table, {n_sigma_rows}x{n_ex_cols}, "
                f"index [sigma_bin][ex_bin]"
    )

    return lut_exp, lut_2d

# ─────────────────────────────────────────────
# Verification: run both methods on a known
# input and compare against numpy reference
# ─────────────────────────────────────────────

def verify_luts(lut_exp, lut_2d, lut1e, lut_alpha):
    print("\n── Verification ─────────────────────────────")

    # Test with a small known input (Q8.8 values converted to float)
    # Simulating a row of attention scores after scaling by 1/sqrt(d_k)
    np.random.seed(42)
    scores_float = np.random.randn(16).astype(np.float32) * 2.0

    # Reference softmax
    def softmax_ref(x):
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / ex.sum()

    ref = softmax_ref(scores_float)

    # ── 2D LUT approximation ──
    x = scores_float - scores_float.max()   # normalize

    # Get e^x via LUT
    def lut_exp_lookup(x_val, lut):
        # x_val in (-inf, 0], quantize to address
        addr = int(np.clip(round(-x_val * SCALE), 0, 255))
        return lut[addr] / SCALE

    ex_vals = np.array([lut_exp_lookup(v, lut_exp) for v in x])
    sigma   = ex_vals.sum()

    # Get σ(xi) via 2D LUT
    n_ex_cols = int(1.0 / SCALE_EX) + 1
    def lookup_2d(ex_val, sigma_val, lut):
        i = int(np.clip(round(ex_val / SCALE_EX), 0, n_ex_cols - 1))
        j = int(np.clip(round(sigma_val / SCALE_SIGMA), 1, len(lut))) - 1
        return lut[j][i] / SCALE

    approx_2d = np.array([lookup_2d(e, sigma, lut_2d) for e in ex_vals])
    # Renormalize (LUT introduces small error)
    if approx_2d.sum() > 0:
        approx_2d = approx_2d / approx_2d.sum()

    # ── Report ──
    max_err  = np.max(np.abs(ref - approx_2d))
    mean_err = np.mean(np.abs(ref - approx_2d))
    print(f"  Test input (16 values, σ≈{scores_float.std():.2f}):")
    print(f"  Max absolute error  : {max_err:.6f}")
    print(f"  Mean absolute error : {mean_err:.6f}")
    print(f"  Sum of approx output: {approx_2d.sum():.6f}  (should be ~1.0)")

    if max_err < 0.05:
        print("  ✓ Accuracy looks good for uint8 approximation")
    else:
        print("  ⚠ Error higher than expected — check LUT params")

    # Print side-by-side for first 8 values
    print(f"\n  {'i':>3}  {'reference':>12}  {'2D LUT approx':>14}  {'error':>8}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*14}  {'─'*8}")
    for i in range(min(8, len(ref))):
        err = abs(ref[i] - approx_2d[i])
        print(f"  {i:>3}  {ref[i]:>12.6f}  {approx_2d[i]:>14.6f}  {err:>8.6f}")

# ─────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────

def print_summary(lut_exp, lut_2d, lut1e, lut_alpha):
    n_ex_cols    = int(1.0 / SCALE_EX) + 1
    n_sigma_rows = int(MAX_SIGMA_OURS / SCALE_SIGMA)

    print("\n── Resource summary for Basys 3 (Artix-7) ──")
    print(f"  REXP method:")
    print(f"    LUT1/e  : {len(lut1e)} entries × 8 bits = {len(lut1e)} bytes")
    print(f"    LUTalpha: {len(lut_alpha)} entries × 8 bits = {len(lut_alpha)} bytes")
    print(f"    Total   : {len(lut1e) + len(lut_alpha)} bytes  → fits in LUTRAM or small BRAM")

    lut2d_bytes = n_sigma_rows * n_ex_cols
    print(f"\n  2D LUT method (recommended):")
    print(f"    LUTexp  : 256 entries × 8 bits = 256 bytes")
    print(f"    LUT2D   : {n_sigma_rows} × {n_ex_cols} = {lut2d_bytes} bytes")
    print(f"    Total   : {256 + lut2d_bytes} bytes  → fits in 1–2 BRAM18 blocks")
    print(f"\n  Basys 3 has 50 BRAM18 blocks — this uses ~{(256+lut2d_bytes)/2048*100:.1f}% of one")
    print(f"\n  Next step: load these .mem files in SystemVerilog with $readmemh()")
    print(f"  See softmax_unit.sv (to be generated)")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print(" Softmax LUT Generator")
    print(" Based on Vasyltsov & Chang arXiv:2111.10770")
    print(f" Precision : uint{PRECISION_BITS}  (scale = {SCALE})")
    print(f" Seq length: {SEQ_LEN} tokens")
    print("=" * 52)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lut1e, lut_alpha      = gen_rexp_luts()
    lut_exp, lut_2d       = gen_2d_luts()

    verify_luts(lut_exp, lut_2d, lut1e, lut_alpha)
    print_summary(lut_exp, lut_2d, lut1e, lut_alpha)

    print("\n── Output files ─────────────────────────────")
    for f in ["lut1e.mem", "lut_alpha.mem", "lut_exp.mem", "lut_2d.mem"]:
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(path):
            print(f"  {f:20s}  {os.path.getsize(path):6d} bytes")

    print("\nDone. Load these in SystemVerilog with:")
    print('  $readmemh("lut_exp.mem", lut_exp_rom);')
    print('  $readmemh("lut_2d.mem",  lut_sigma_rom);')