"""
rtl_accuracy.py
===============
Two modes:

  --gen-inputs   Generate rtl_inputs.txt for the testbench to read.
                 Run this BEFORE launching simulation.

  (default)      Read rtl_outputs.txt written by the testbench,
                 compare against true softmax, report argmax accuracy.

Workflow:
  1. python rtl_accuracy.py --gen-inputs
  2. In Vivado: launch_simulation  →  run 5ms
  3. python rtl_accuracy.py
"""

import numpy as np
import argparse
import os

SEQ_LEN   = 16
FRAC_BITS = 8
N_ROWS    = 500
SEED      = 42

SIM_DIR   = r"C:/Users/IsaiahK/Documents/School/FPGA/transformer/softmax/sim"
INPUT_FILE  = os.path.join(SIM_DIR, "rtl_inputs.txt")
OUTPUT_FILE = os.path.join(SIM_DIR, "rtl_outputs.txt")


def float_to_q8_8(x):
    return int(np.clip(int(np.round(float(x) * 2**FRAC_BITS)), -32768, 32767))

def q8_8_to_float(x):
    return x / 2**FRAC_BITS

def true_softmax_scaled(row_q):
    """True softmax on Q8.8 integers, scaled to 0-255."""
    row_f = np.array([q8_8_to_float(v) for v in row_q], dtype=np.float64)
    row_f -= row_f.max()
    e = np.exp(row_f)
    return (e / e.sum() * 255).tolist()


def gen_inputs(n_rows, seed):
    """
    Generate random input rows in four categories and write to rtl_inputs.txt.
    Each row is SEQ_LEN signed Q8.8 integers, space-separated, one row per line.
    """
    rng = np.random.default_rng(seed)
    rows = []
    categories = []   # track category for later analysis

    n_each = n_rows // 4

    # Uniform: all tokens in [-2, 2]
    for _ in range(n_each):
        row = [float_to_q8_8(x) for x in rng.uniform(-2, 2, SEQ_LEN)]
        rows.append(row); categories.append("uniform")

    # Soft peak: base [-0.5, 0.5], one token boosted by 0.5-1.5
    for _ in range(n_each):
        row_f = rng.uniform(-0.5, 0.5, SEQ_LEN).tolist()
        p = int(rng.integers(0, SEQ_LEN))
        row_f[p] += float(rng.uniform(0.5, 1.5))
        rows.append([float_to_q8_8(x) for x in row_f]); categories.append("soft")

    # Peaked: base [-1, 1], one token boosted by 2-4
    for _ in range(n_each):
        row_f = rng.uniform(-1, 1, SEQ_LEN).tolist()
        p = int(rng.integers(0, SEQ_LEN))
        row_f[p] += float(rng.uniform(2, 4))
        rows.append([float_to_q8_8(x) for x in row_f]); categories.append("peaked")

    # Extreme: base [-1, 0], one token set to 3-4
    for _ in range(n_each):
        row_f = rng.uniform(-1, 0, SEQ_LEN).tolist()
        p = int(rng.integers(0, SEQ_LEN))
        row_f[p] = float(rng.uniform(3, 4))
        rows.append([float_to_q8_8(x) for x in row_f]); categories.append("extreme")

    os.makedirs(SIM_DIR, exist_ok=True)
    with open(INPUT_FILE, 'w') as f:
        for row in rows:
            f.write(' '.join(str(v) for v in row) + '\n')

    # Also save categories so the accuracy script can break down by type
    cat_file = os.path.join(SIM_DIR, "rtl_categories.txt")
    with open(cat_file, 'w') as f:
        for c in categories:
            f.write(c + '\n')

    print(f"Written {n_rows} input rows to: {INPUT_FILE}")
    print(f"Written categories to:          {cat_file}")
    print(f"\nNow in Vivado:")
    print(f"  launch_simulation")
    print(f"  run 5ms")
    print(f"Then: python rtl_accuracy.py")


def evaluate():
    """Read rtl_outputs.txt and compare against true softmax."""

    if not os.path.exists(OUTPUT_FILE):
        print(f"ERROR: {OUTPUT_FILE} not found.")
        print("Run the simulation first, then re-run this script.")
        return

    # Load categories if available
    cat_file = os.path.join(SIM_DIR, "rtl_categories.txt")
    categories = None
    if os.path.exists(cat_file):
        with open(cat_file) as f:
            categories = [line.strip() for line in f]

    # Parse output file
    input_rows  = []
    output_rows = []
    with open(OUTPUT_FILE) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("INPUT:"):
            vals = list(map(int, lines[i].split()[1:]))
            input_rows.append(vals)
        elif lines[i].startswith("OUTPUT:"):
            vals = list(map(int, lines[i].split()[1:]))
            output_rows.append(vals)
        i += 1

    if len(input_rows) != len(output_rows):
        print(f"WARNING: {len(input_rows)} input rows but {len(output_rows)} output rows")

    n = min(len(input_rows), len(output_rows))
    print(f"\nEvaluating {n} rows from RTL simulation...")

    # Per-category tracking
    cat_correct = {"uniform":0, "soft":0, "peaked":0, "extreme":0}
    cat_total   = {"uniform":0, "soft":0, "peaked":0, "extreme":0}
    total_correct = 0

    for idx in range(n):
        row_q   = input_rows[idx]
        fpga_out = output_rows[idx]

        ref = true_softmax_scaled(row_q)
        ref_argmax  = int(np.argmax(ref))
        fpga_argmax = int(np.argmax(fpga_out))
        correct = (ref_argmax == fpga_argmax)

        total_correct += correct

        if categories and idx < len(categories):
            cat = categories[idx]
            cat_correct[cat] += correct
            cat_total[cat]   += 1

    overall = total_correct / n * 100

    print(f"\n{'='*55}")
    print(f"  RTL Argmax Accuracy — {n} rows")
    print(f"{'='*55}")

    if categories:
        print(f"  {'Category':<35} {'Accuracy':>8}")
        print(f"  {'-'*44}")
        for cat in ["uniform", "soft", "peaked", "extreme"]:
            if cat_total[cat] > 0:
                acc = cat_correct[cat] / cat_total[cat] * 100
                print(f"  {cat:<35} {acc:>7.2f}%")
        print(f"  {'-'*44}")

    print(f"  {'OVERALL':<35} {overall:>7.2f}%")
    print(f"{'='*55}")
    print(f"""
  Compare against Python LUT simulation targets:
    peaked / extreme  ->  ~100%
    soft              ->  ~94%
    uniform           ->  ~87%
    overall           ->  ~95%

  If RTL numbers are significantly lower than Python targets,
  the sv pipeline has a timing or indexing issue.
  If they match, the hardware is behaving as designed.
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-inputs", action="store_true",
        help="Generate rtl_inputs.txt for the testbench (run before simulation)")
    parser.add_argument("--n-rows", type=int, default=N_ROWS)
    parser.add_argument("--seed",   type=int, default=SEED)
    args = parser.parse_args()

    if args.gen_inputs:
        gen_inputs(args.n_rows, args.seed)
    else:
        evaluate()
