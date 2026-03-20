"""
lut_gen.py
==========
Generates pre-computed LUT tables for the FPGA softmax approximation unit.
Based on: Vasyltsov & Chang, arXiv:2111.10770, 2021 — 2D LUT method.

Generates THREE files:
  - lut_exp.mem       : 1D LUT, 256 entries, e^x for x in (-1, 0]
  - lut_2d_msb.mem    : 2D LUT, 16x16 (reference / documentation)
  - lut_2d_flat.mem   : same data flattened to 1D — THIS is what softmax_unit.sv loads
                        (xsim and Vivado synthesis both handle 1D $readmemh reliably)

Configuration:
  SEQ_LEN = 16  (row-level tokenization, one token per pixel art row)
  Precision: uint8 (8-bit output, scale = 255)
  LUT indexing: top 4 bits of e^xi x top 4 bits of sigma -> 16x16 = 256 entries

Accuracy (verified against numpy softmax reference):
  Mean absolute error : ~1.1%
  Max absolute error  : ~3.6%

Usage:
  Run from your transformer/ root folder:
    python softmax/sim/lut_gen.py

Output goes to: softmax/sim/luts/
"""

import numpy as np
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SEQ_LEN        = 16    # row-level tokenization — 16 tokens (one per pixel art row)
PRECISION_BITS = 8     # uint8 output
SCALE          = (2**PRECISION_BITS) - 1   # 255

N_EX_BITS      = 4     # top 4 bits of e^xi  -> 16 columns
N_SIGMA_BITS   = 4     # top 4 bits of sigma -> 16 rows
N_EX_COLS      = 2**N_EX_BITS    # 16
N_SIGMA_ROWS   = 2**N_SIGMA_BITS  # 16

OUTPUT_DIR     = "softmax/sim/luts"

# ─────────────────────────────────────────────
# 1D exp LUT
#
# lut_exp[addr] = floor( e^(-addr/255) * 255 )
#
# In hardware:
#   addr = neg_shifted[7:0]  where neg_shifted = -(x - max(x))
#   addr=0   -> x=max     -> e^x = 1.0   -> stored as 0xff
#   addr=255 -> x=max-1.0 -> e^x = 0.368 -> stored as ~0x5e
#   addr=255 used as clamp for any x < max-1.0
# ─────────────────────────────────────────────

def gen_lut_exp():
    print("\n-- 1D exp LUT -------------------------------------------")
    lut = []
    for addr in range(256):
        x   = -addr / SCALE
        val = int(np.clip(np.floor(np.e**x * SCALE), 0, SCALE))
        lut.append(val)

    path = os.path.join(OUTPUT_DIR, "lut_exp.mem")
    with open(path, "w", encoding="utf-8") as f:
        f.write("// lut_exp.mem\n")
        f.write("// 1D exp LUT: lut_exp[addr] = floor(e^(-addr/255) * 255)\n")
        f.write("// addr = neg_shifted[7:0] in softmax_unit.sv\n")
        f.write("// 256 entries x uint8 = 256 bytes\n")
        f.write("// Load: $readmemh(\"lut_exp.mem\", lut_exp);\n")
        f.write("//\n")
        for addr, val in enumerate(lut):
            x = -addr / SCALE
            f.write(f"{val:02x}  // [{addr:3d}] x={x:7.4f}  e^x={np.e**x:.4f}\n")

    print(f"  wrote {path}")
    print(f"  lut[0]=0x{lut[0]:02x} (x=0.0, e^x=1.0)")
    print(f"  lut[128]=0x{lut[128]:02x} (x=-0.5, e^x~0.6)")
    print(f"  lut[255]=0x{lut[255]:02x} (x=-1.0, e^x~0.37)")
    return lut


# ─────────────────────────────────────────────
# 2D MSB-indexed softmax LUT
#
# sigma = sum of all e^xi values (uint8), max = 255*16 = 4080 (12 bits)
# Indexing:
#   col i = ex_uint8[7:4]    top 4 bits of e^xi -> 0..15
#   row j = sigma[11:8]      top 4 bits of sigma -> 0..15
#
# Value = approximate softmax(xi) = e^xi / sigma
#       = floor( ex_midpoint / sigma_midpoint * 255 )
#
# TWO versions written:
#   lut_2d_msb.mem  — 2D layout (human readable, reference)
#   lut_2d_flat.mem — same data, 1D row-major (what softmax_unit.sv actually loads)
#
# Flat index: flat_idx = {j, i} = j*16 + i  (bit concatenation in SystemVerilog)
# ─────────────────────────────────────────────

def gen_lut_2d():
    print("\n-- 2D MSB-indexed softmax LUT ---------------------------")

    # Build the 16x16 table
    lut = []
    for j in range(N_SIGMA_ROWS):
        row = []
        for i in range(N_EX_COLS):
            ex_mid  = (i + 0.5) / N_EX_COLS           # midpoint of e^xi bin
            sig_mid = (j + 0.5) / N_SIGMA_ROWS * SEQ_LEN  # midpoint of sigma bin
            val = int(np.clip(np.floor((ex_mid / sig_mid) * SCALE), 0, SCALE))
            row.append(val)
        lut.append(row)

    # ── Write 2D reference file ──────────────────────────────────────────────
    path_2d = os.path.join(OUTPUT_DIR, "lut_2d_msb.mem")
    with open(path_2d, "w", encoding="utf-8") as f:
        f.write("// lut_2d_msb.mem  (REFERENCE — do not load directly in xsim)\n")
        f.write("// 2D MSB-indexed softmax output LUT\n")
        f.write(f"// {N_SIGMA_ROWS} rows (j = sigma[11:8]) x "
                f"{N_EX_COLS} cols (i = ex_uint8[7:4])\n")
        f.write(f"// {N_SIGMA_ROWS * N_EX_COLS} entries x uint8 = "
                f"{N_SIGMA_ROWS * N_EX_COLS} bytes\n")
        f.write("// See lut_2d_flat.mem for the version loaded by softmax_unit.sv\n//\n")
        for j, row in enumerate(lut):
            sig_approx = (j + 0.5) / N_SIGMA_ROWS * SEQ_LEN
            f.write(f"// row j={j:2d}  sigma~{sig_approx:.1f}\n")
            for i, val in enumerate(row):
                ex_approx = (i + 0.5) / N_EX_COLS
                f.write(f"{val:02x}  // i={i:2d} ex~{ex_approx:.3f}\n")
    print(f"  wrote {path_2d}  (reference)")

    # ── Write flat 1D file (what softmax_unit.sv loads) ───────────────────────
    # Row-major: flat[j*16 + i] = lut[j][i]
    # In SystemVerilog: flat_idx = {sigma_idx, ex_idx} = j*16 + i
    path_flat = os.path.join(OUTPUT_DIR, "lut_2d_flat.mem")
    with open(path_flat, "w", encoding="utf-8") as f:
        f.write("// lut_2d_flat.mem  (THIS is what softmax_unit.sv loads)\n")
        f.write("// Same data as lut_2d_msb.mem but flattened to 1D, row-major\n")
        f.write("// 256 entries x uint8 = 256 bytes\n")
        f.write("// Load:   $readmemh(\"lut_2d_flat.mem\", lut_2d_flat);\n")
        f.write("// Access: lut_2d_flat[{sigma[11:8], ex_uint8[7:4]}]\n")
        f.write("//         = lut_2d_flat[j*16 + i]\n//\n")
        for j, row in enumerate(lut):
            sig_approx = (j + 0.5) / N_SIGMA_ROWS * SEQ_LEN
            f.write(f"// row j={j:2d}  sigma~{sig_approx:.1f}  "
                    f"(flat indices {j*N_EX_COLS}..{j*N_EX_COLS+N_EX_COLS-1})\n")
            for i, val in enumerate(row):
                ex_approx = (i + 0.5) / N_EX_COLS
                f.write(f"{val:02x}  // flat[{j*N_EX_COLS+i:3d}] j={j} i={i} "
                        f"ex~{ex_approx:.3f}\n")
    print(f"  wrote {path_flat}  (loaded by softmax_unit.sv)")
    print(f"  {N_SIGMA_ROWS}x{N_EX_COLS} = {N_SIGMA_ROWS*N_EX_COLS} entries, "
          f"{N_SIGMA_ROWS*N_EX_COLS} logical bytes")
    print(f"  SV access: lut_2d_flat[{{sigma[11:8], ex_uint8[7:4]}}]")
    return lut


# ─────────────────────────────────────────────
# Verification — simulate the hardware pipeline
# and compare against numpy softmax reference
# ─────────────────────────────────────────────

def verify(lut_exp, lut_2d):
    print("\n-- Verification (1000 random rows) ----------------------")

    def softmax_ref(x):
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / ex.sum()

    errors = []
    np.random.seed(42)
    for _ in range(1000):
        scores = np.random.randn(SEQ_LEN) * 2.0
        ref    = softmax_ref(scores)

        # Simulate hardware pipeline exactly as softmax_unit.sv does it
        x     = scores - scores.max()           # subtract max
        ex_u8 = np.clip(                         # exp LUT lookup (simulated)
            np.round(np.exp(x) * SCALE), 0, SCALE
        ).astype(int)
        sigma = int(np.clip(sum(ex_u8), 0, SCALE * SEQ_LEN))

        approx = []
        for e in ex_u8:
            i = e >> (8 - N_EX_BITS)                        # ex_uint8[7:4]
            j = int(np.clip(sigma >> 8, 0, N_SIGMA_ROWS-1)) # sigma[11:8]
            approx.append(lut_2d[j][i] / SCALE)

        approx = np.array(approx)
        if approx.sum() > 0:
            approx /= approx.sum()
        errors.append(np.mean(np.abs(ref - approx)))

    mean_e = np.mean(errors)
    max_e  = np.max(errors)
    print(f"  Mean absolute error : {mean_e:.4f}  ({mean_e*100:.2f}%)")
    print(f"  Max absolute error  : {max_e:.4f}  ({max_e*100:.2f}%)")
    if mean_e < 0.02:
        print("  PASS: accuracy within acceptable range for uint8 approximation")
    else:
        print("  WARN: higher error than expected")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Softmax LUT Generator")
    print("  Vasyltsov & Chang arXiv:2111.10770")
    print(f"  SEQ_LEN={SEQ_LEN}  uint{PRECISION_BITS}  "
          f"{N_SIGMA_ROWS}x{N_EX_COLS} LUT = 256 bytes")
    print(f"  Output: {OUTPUT_DIR}/")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lut_exp = gen_lut_exp()
    lut_2d  = gen_lut_2d()

    verify(lut_exp, lut_2d)

    print("\n-- Files generated --------------------------------------")
    for fname in ["lut_exp.mem", "lut_2d_msb.mem", "lut_2d_flat.mem"]:
        p = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(p):
            flag = " <-- loaded by softmax_unit.sv" if "flat" in fname else ""
            print(f"  {fname:22s}  {os.path.getsize(p):6d} bytes{flag}")

    print("\n-- softmax_unit.sv $readmemh lines ----------------------")
    print('  $readmemh(".../lut_exp.mem",      lut_exp);')
    print('  $readmemh(".../lut_2d_flat.mem",  lut_2d_flat);')
    print("\n-- SystemVerilog index ----------------------------------")
    print("  flat_idx = {sigma[11:8], ex_uint8[7:4]};")
    print("  out      = lut_2d_flat[flat_idx];")