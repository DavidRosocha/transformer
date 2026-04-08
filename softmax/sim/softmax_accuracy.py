"""
softmax_accuracy.py  —  FPGA softmax argmax accuracy evaluator
==============================================================
Simulates softmax_unit.sv pipeline exactly and measures argmax accuracy
against true float softmax across many random input rows.

Pipeline simulated:
  addr      = CLAMP(running_max - val, 0, 255)
  ex_buf[i] = lut_exp[addr]
  sigma     = sum(ex_buf)
  sigma_idx = CLAMP(sigma * S // SIGMA_MAX, 0, S-1)
  ex_idx    = CLAMP(ex_buf[i] * E // 256,  0, E-1)
  out[i]    = lut_2d_flat[sigma_idx * E + ex_idx]

Usage:
    python softmax_accuracy.py
    python softmax_accuracy.py --lut-dir path/to/luts --sigma-bits 4 --ex-bits 4 --trials 8000
"""

import numpy as np
import argparse
import os

SEQ_LEN   = 16
SIGMA_MAX = 256 * SEQ_LEN   # 4096


def float_to_q8_8(x):
    return int(np.clip(int(np.round(float(x) * 256)), -32768, 32767))


def load_lut(filepath, expected_depth):
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '//' in line:
                line = line[:line.index('//')].strip()
            if line:
                values.append(int(line, 16))
    if len(values) != expected_depth:
        raise ValueError(f"{filepath}: expected {expected_depth} entries, got {len(values)}")
    return values


def fpga_softmax(row_float, lut_exp, lut_2d_flat, S, E):
    row_q = [float_to_q8_8(x) for x in row_float]
    running_max = max(row_q)
    ex_buf = [lut_exp[int(np.clip(running_max - v, 0, 255))] for v in row_q]
    sigma = sum(ex_buf)
    sigma_idx = int(np.clip(sigma * S // SIGMA_MAX, 0, S - 1))
    out = []
    for e in ex_buf:
        ex_idx   = int(np.clip(e * E // 256, 0, E - 1))
        out.append(lut_2d_flat[sigma_idx * E + ex_idx])
    return out


def true_softmax_scaled(row_float):
    x = np.array(row_float, dtype=np.float64)
    x -= x.max()
    e = np.exp(x)
    return (e / e.sum() * 255).tolist()


def _peaked(rng, base_range, boost, absolute=False):
    row = rng.uniform(*base_range, SEQ_LEN).tolist()
    p = int(rng.integers(0, SEQ_LEN))
    if absolute: row[p] = float(rng.uniform(*boost))
    else:        row[p] += float(rng.uniform(*boost))
    return row


def evaluate(lut_dir, sigma_bits, ex_bits, n_trials, seed):
    S = 2**sigma_bits
    E = 2**ex_bits

    print(f"\nLoading LUTs from: {lut_dir}")
    lut_exp     = load_lut(os.path.join(lut_dir, "lut_exp.mem"),     256)
    lut_2d_flat = load_lut(os.path.join(lut_dir, "lut_2d_flat.mem"), S * E)
    print(f"  lut_exp[0]={lut_exp[0]}  lut_exp[128]={lut_exp[128]}  lut_exp[255]={lut_exp[255]}")
    print(f"  lut_2d_flat: {S}x{E} = {S*E} entries")

    # Sanity check
    expected_128 = int(np.floor(np.exp(-0.5)*255))
    if abs(lut_exp[128] - expected_128) > 5:
        print(f"\n  *** WARNING: lut_exp[128]={lut_exp[128]}, expected ~{expected_128}")
        print(f"      Regenerate with lut_gen.py\n")

    rng = np.random.default_rng(seed)
    n_each = n_trials // 4
    cats = {k: 0 for k in ["uniform", "soft", "peaked", "extreme"]}

    for _ in range(n_each):
        row = rng.uniform(-2, 2, SEQ_LEN).tolist()
        cats["uniform"] += int(np.argmax(fpga_softmax(row,lut_exp,lut_2d_flat,S,E))) == int(np.argmax(true_softmax_scaled(row)))

        row = _peaked(rng, (-0.5, 0.5), (0.5, 1.5))
        cats["soft"]    += int(np.argmax(fpga_softmax(row,lut_exp,lut_2d_flat,S,E))) == int(np.argmax(true_softmax_scaled(row)))

        row = _peaked(rng, (-1.0, 1.0), (2.0, 4.0))
        cats["peaked"]  += int(np.argmax(fpga_softmax(row,lut_exp,lut_2d_flat,S,E))) == int(np.argmax(true_softmax_scaled(row)))

        row = _peaked(rng, (-1.0, 0.0), (3.0, 4.0), absolute=True)
        cats["extreme"] += int(np.argmax(fpga_softmax(row,lut_exp,lut_2d_flat,S,E))) == int(np.argmax(true_softmax_scaled(row)))

    overall = sum(cats.values()) / (4 * n_each) * 100

    print(f"\n{'='*55}")
    print(f"  Argmax Accuracy -- {n_trials} trials, {S}x{E} LUT (seed={seed})")
    print(f"{'='*55}")
    print(f"  {'Category':<35} {'Accuracy':>8}")
    print(f"  {'-'*44}")
    for name, c in cats.items():
        print(f"  {name:<35} {c/n_each*100:>7.2f}%")
    print(f"  {'-'*44}")
    print(f"  {'OVERALL':<35} {overall:>7.2f}%")
    print(f"{'='*55}")
    print(f"""
  Targets (16x16 LUT):
    peaked / extreme  ->  100%
    soft              ->  ~94%
    uniform           ->  ~88%
    overall           ->  ~96%

  Larger LUTs (16x32, 32x32) push uniform/soft higher
  at the cost of more memory. peaked/extreme are already 100%.
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lut-dir",
        default=r"C:/Users/IsaiahK/Documents/School/FPGA/transformer/softmax/sim/luts")
    parser.add_argument("--sigma-bits", type=int, default=4)
    parser.add_argument("--ex-bits",    type=int, default=4)
    parser.add_argument("--trials",     type=int, default=4000)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    evaluate(args.lut_dir, args.sigma_bits, args.ex_bits, args.trials, args.seed)
