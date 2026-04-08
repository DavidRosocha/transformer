"""
lut_gen.py  —  FPGA softmax LUT generator
==========================================
Generates lut_exp.mem and lut_2d_flat.mem for softmax_unit.sv.

CORRECT SCALING (both formulas matter):

  lut_exp:
    addr    = CLAMP(running_max - val, 0, 255)  [full Q8.8 subtraction]
    formula = floor(exp(-addr / 256) * 255)
    addr is in Q8.8 units where 256 = 1.0, so exp(-256/256) = exp(-1) = 0.368

  lut_2d_flat:
    ex_mid  = (i + 0.5) * (256 / E)      [ex_buf value in 0..255 space]
    sig_mid = (j + 0.5) * SIGMA_MAX / S  [sigma in 0..4096 space]
    formula = floor(ex_mid / sig_mid * 255)

    where SIGMA_MAX = 255 * SEQ_LEN = 4080 (rounded to 4096)
    Both ex_mid and sig_mid must be in the same units (absolute counts, not fractions).

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


def generate_lut_exp(depth=256):
    """
    lut_exp[addr] = floor(exp(-addr/256) * 255)

    addr is a full Q8.8 difference (running_max - val), so:
      addr=0   -> same as max  -> exp(0)*255   = 255
      addr=128 -> 0.5 below    -> exp(-0.5)*255 = 154
      addr=255 -> ~1.0 below   -> exp(-1)*255   = 93
    """
    return [int(np.clip(int(np.floor(np.exp(-i / 256.0) * SCALE)), 0, SCALE))
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
      sigma_idx = CLAMP(sigma * S / SIGMA_MAX, 0, S-1)
      ex_idx    = CLAMP(ex_buf * E / 256, 0, E-1)
      out       = lut_2d_flat[sigma_idx * E + ex_idx]
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
    assert lut_exp[0]   == 255, f"lut_exp[0] should be 255"
    assert lut_exp[128] == int(np.floor(np.exp(-0.5)*255)), f"lut_exp[128] wrong"
    print(f"  lut_exp[0]={lut_exp[0]}  lut_exp[128]={lut_exp[128]}  lut_exp[255]={lut_exp[255]}  OK")

    # Sanity: peak=3.0 at one token, rest=0
    # addr(peak)=0 -> ex=255, addr(others)=768 clamped to 255 -> ex=94
    sigma = 255 + 94*(SEQ_LEN-1)
    sigma_idx = int(np.clip(sigma * S // SIGMA_MAX, 0, S-1))
    ex_idx_peak  = int(np.clip(255 * E // 256, 0, E-1))
    ex_idx_other = int(np.clip(94  * E // 256, 0, E-1))
    out_peak  = lut_2d_flat[sigma_idx * E + ex_idx_peak]
    out_other = lut_2d_flat[sigma_idx * E + ex_idx_other]
    print(f"  peaked sanity: out[peak]={out_peak}, out[other]={out_other}, peak wins={out_peak>out_other}")
    assert out_peak > out_other, "FAIL: peak does not win in simple peaked test!"
    print("  All checks passed.\n")


def main(out_dir, sigma_bits, ex_bits):
    S = 2**sigma_bits
    E = 2**ex_bits
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating LUTs: {S}x{E} -> {out_dir}")

    lut_exp = generate_lut_exp()
    write_mem(os.path.join(out_dir, "lut_exp.mem"), lut_exp, [
        "lut_exp.mem",
        "formula: lut_exp[addr] = floor(exp(-addr/256) * 255)",
        "addr = CLAMP(running_max - val, 0, 255)  (full Q8.8 subtraction)",
        "256 entries x uint8",
    ])

    lut_2d = generate_lut_2d_flat(sigma_bits, ex_bits)
    write_mem(os.path.join(out_dir, "lut_2d_flat.mem"), lut_2d, [
        "lut_2d_flat.mem",
        f"Flattened {S}x{E} softmax division LUT, row-major",
        "ex_mid  = (i + 0.5) * (256 / E)         [ex_buf units, 0..255]",
        "sig_mid = (j + 0.5) * (SIGMA_MAX / S)   [sigma units, 0..4096]",
        "formula: floor(ex_mid / sig_mid * 255)",
        f"SV access: sigma_idx=CLAMP(sigma*{S}//4096,0,{S-1})",
        f"           ex_idx   =CLAMP(ex_buf*{E}//256, 0,{E-1})",
        f"           out      =lut_2d_flat[sigma_idx*{E}+ex_idx]",
        f"{S*E} entries x uint8",
    ])

    verify(lut_exp, lut_2d, S, E)

    print(f"LUT size: {S}x{E} = {S*E} bytes")
    print(f"BRAM cost: {'LUTRAM (~'+str(S*E//8)+' LUT6s)' if S*E<=512 else '1x BRAM18'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",
        default=r"C:\Users\lab\Downloads\testing\sim\luts")
    parser.add_argument("--sigma-bits", type=int, default=4,
        help="Log2 of number of sigma buckets (default 4 = 16 rows)")
    parser.add_argument("--ex-bits", type=int, default=4,
        help="Log2 of number of ex buckets (default 4 = 16 cols)")
    args = parser.parse_args()
    main(args.out_dir, args.sigma_bits, args.ex_bits)
