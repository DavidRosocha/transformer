"""
lut_gen.py  —  FPGA softmax LUT generator
==========================================
Generates lut_exp.mem and lut_2d_flat.mem for softmax_unit.sv.

CORRECT SCALING (both formulas matter):

  lut_exp:
    addr    = CLAMP((running_max - val) >> 2, 0, 255)
    formula = floor(exp(-addr / 64) * 255)

    The Q8.8 difference (running_max - val) is right-shifted by 2 before
    indexing, so each LUT step covers 4 Q8.8 units = 4/256 = 1/64 in float.
    This gives the LUT a total range of 0..255/64 = 0..3.98 in float, which
    covers the full practical spread of attention scores:
      addr=0   -> 0.0 below max -> exp(0)*255        = 255
      addr=64  -> 1.0 below max -> exp(-1)*255        = 93
      addr=128 -> 2.0 below max -> exp(-2)*255        = 34
      addr=192 -> 3.0 below max -> exp(-3)*255        = 12
      addr=255 -> 3.98 below max -> exp(-3.98)*255    = 4

    WHY THE SHIFT? The raw Q8.8 diff can reach ~1500 for typical inputs
    (e.g. max=2.0, min=-4.0 → diff = 6*256 = 1536). Without the shift,
    anything more than 255/256 ≈ 1.0 below max gets clamped to addr=255
    and assigned lut_exp[255]=94 instead of a near-zero value — causing
    74% of tokens to get the wrong weight.

  lut_2d_flat:
    ex_mid  = (i + 0.5) * (256 / E)      [ex_buf value in 0..255 space]
    sig_mid = (j + 0.5) * SIGMA_MAX / S  [sigma in 0..4096 space]
    formula = floor(ex_mid / sig_mid * 255)

    where SIGMA_MAX = 256 * SEQ_LEN = 4096
    Both ex_mid and sig_mid must be in the same units (absolute counts).

Usage:
    python lut_gen.py
    python lut_gen.py --out-dir path/to/luts --sigma-bits 4 --ex-bits 4
"""

import numpy as np
import argparse
import os

SEQ_LEN   = 16
SCALE     = 255
SIGMA_MAX = 256 * SEQ_LEN   # 4096 — max possible sigma (255 * 16 = 4080, round up)

# Number of bits to right-shift the Q8.8 diff before indexing lut_exp.
# Each addr step = 2^EXP_SHIFT Q8.8 units = 2^EXP_SHIFT / 256 in float.
# Shift of 2 → step = 1/64 float, range = 256/64 = 4.0 float.
EXP_ADDR_SHIFT = 2


def generate_lut_exp(depth=256):
    """
    lut_exp[addr] = floor(exp(-addr / 64) * 255)

    addr = CLAMP((running_max - val) >> 2, 0, 255)

    Each addr unit = 4 Q8.8 units = 1/64 in float, so the LUT covers
    0.0 to 3.98 below the running max. Anything further gets clamped to
    addr=255 → lut_exp[255] = 4 ≈ 0, which is correct (near-zero weight).
    """
    return [int(np.clip(int(np.floor(np.exp(-i / 64.0) * SCALE)), 0, SCALE))
            for i in range(depth)]


def generate_lut_2d_flat(sigma_bits, ex_bits):
    """
    Flattened S x E LUT where S=2^sigma_bits, E=2^ex_bits.

    Approximates: out = (ex_buf_val / sigma) * 255

    ex_mid  = midpoint of ex_buf bucket i, in 0..255 space
            = (i + 0.5) * (256 / E)
    sig_mid = midpoint of sigma bucket j, in 0..SIGMA_MAX space
            = (j + 0.5) * (SIGMA_MAX / S)

    Access in SystemVerilog:
      sigma_idx = sigma[11:8]          (top 4 bits of 12-bit sigma)
      ex_idx    = ex_buf[cnt][7:4]     (top 4 bits of uint8 ex value)
      out       = lut_2d_flat[{sigma_idx, ex_idx}]
    """
    S = 2**sigma_bits
    E = 2**ex_bits
    lut = []
    for j in range(S):
        for i in range(E):
            ex_mid  = (i + 0.5) * (256.0 / E)
            sig_mid = max((j + 0.5) * SIGMA_MAX / S, 0.5)
            val = int(np.clip(int(np.floor(ex_mid / sig_mid * SCALE)), 0, SCALE))
            lut.append(val)
    return lut


def write_mem(filepath, data, header_lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in header_lines:
            f.write(f"// {line}\n")
        f.write("//\n")
        for idx, val in enumerate(data):
            f.write(f"{val:02x}  // [{idx:3d}] = {val}\n")
    print(f"  Written: {filepath}  ({len(data)} entries)")


def verify(lut_exp, lut_2d_flat, S, E):
    print("\nVerification:")

    # lut_exp boundary checks
    assert lut_exp[0] == 255, f"lut_exp[0] should be 255, got {lut_exp[0]}"

    # addr=64 corresponds to diff>>2=64, i.e. diff=256 Q8.8 = 1.0 float
    expected_64 = int(np.floor(np.exp(-1.0) * 255))
    assert lut_exp[64] == expected_64, \
        f"lut_exp[64] should be {expected_64} (exp(-1)*255), got {lut_exp[64]}"

    print(f"  lut_exp[0]={lut_exp[0]}  lut_exp[64]={lut_exp[64]}  "
          f"lut_exp[128]={lut_exp[128]}  lut_exp[255]={lut_exp[255]}  OK")

    # Sanity: one token at max (diff=0 → addr=0 → ex=255),
    # rest at 2.0 below max (diff=512 → addr=128 → ex=lut_exp[128])
    ex_peak  = lut_exp[0]
    ex_other = lut_exp[128]   # 2.0 below max
    sigma    = ex_peak + ex_other * (SEQ_LEN - 1)
    sigma_idx     = min(sigma >> 8, S - 1)
    ex_idx_peak   = min(ex_peak  >> 4, E - 1)
    ex_idx_other  = min(ex_other >> 4, E - 1)
    out_peak  = lut_2d_flat[sigma_idx * E + ex_idx_peak]
    out_other = lut_2d_flat[sigma_idx * E + ex_idx_other]
    print(f"  peaked sanity (2.0 spread): out[peak]={out_peak}, "
          f"out[other]={out_other}, peak wins={out_peak > out_other}")
    assert out_peak > out_other, "FAIL: peak does not win!"
    print("  All checks passed.\n")


def main(out_dir, sigma_bits, ex_bits):
    S = 2**sigma_bits
    E = 2**ex_bits
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating LUTs: {S}x{E} -> {out_dir}")

    lut_exp = generate_lut_exp()
    write_mem(os.path.join(out_dir, "lut_exp.mem"), lut_exp, [
        "lut_exp.mem",
        "formula: lut_exp[addr] = floor(exp(-addr/64) * 255)",
        "addr = CLAMP((running_max - val) >> 2, 0, 255)",
        "Each addr step = 4 Q8.8 units = 1/64 float; covers 0.0..3.98 below max",
        "256 entries x uint8",
    ])

    lut_2d = generate_lut_2d_flat(sigma_bits, ex_bits)
    write_mem(os.path.join(out_dir, "lut_2d_flat.mem"), lut_2d, [
        "lut_2d_flat.mem",
        f"Flattened {S}x{E} softmax division LUT, row-major",
        "ex_mid  = (i + 0.5) * (256 / E)         [ex_buf units, 0..255]",
        "sig_mid = (j + 0.5) * (SIGMA_MAX / S)   [sigma units, 0..4096]",
        "formula: floor(ex_mid / sig_mid * 255)",
        f"SV access: sigma_idx = sigma[11:8]",
        f"           ex_idx    = ex_buf[cnt][7:4]",
        f"           out       = lut_2d_flat[{{sigma_idx, ex_idx}}]",
        f"{S*E} entries x uint8",
    ])

    verify(lut_exp, lut_2d, S, E)

    print(f"LUT size: {S}x{E} = {S*E} bytes")
    print(f"BRAM cost: {'LUTRAM (~'+str(S*E//8)+' LUT6s)' if S*E<=512 else '1x BRAM18'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",
        default=r"C:\Users\IsaiahK\Documents\School\FPGA\transformer\softmax\sim\luts")
    parser.add_argument("--sigma-bits", type=int, default=4,
        help="Log2 of number of sigma buckets (default 4 = 16 rows)")
    parser.add_argument("--ex-bits", type=int, default=4,
        help="Log2 of number of ex buckets (default 4 = 16 cols)")
    args = parser.parse_args()
    main(args.out_dir, args.sigma_bits, args.ex_bits)