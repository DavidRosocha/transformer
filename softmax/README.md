# Softmax Unit — FPGA Hardware Accelerator

LUT-based softmax approximation for a Transformer attention accelerator targeting the Basys 3 (Artix-7) FPGA.
Implements the **2D LUT method** from Vasyltsov & Chang (2021): no divider, no multiplier — just two small lookup tables and an adder.

---

## How It Works

Standard softmax requires division, which is expensive in hardware. This implementation avoids it entirely:

1. **Find the running max** of all 16 input tokens as they arrive
2. **1D LUT (`lut_exp`)** — for each token, compute `exp(-(max - val) / 64) × 255` via table lookup
3. **Accumulate sigma** — sum all 16 exp values
4. **2D LUT (`lut_2d_flat`)** — for each token, look up `exp_val / sigma × 255` using the top 4 bits of each as indices

The outputs are uint8 values in the range 0–255 representing softmax probabilities scaled to that range.

**Accuracy:** mean absolute error ~2.2/255 (0.87%), argmax correct ~96% across random inputs.
This matches the paper's reported accuracy for uint8 precision.

---

## Repository Structure

```
transformer/                        ← GitHub repo root
│
├── softmax/                        ← All source files (RTL, sim scripts, testbenches)
│   │
│   ├── rtl/
│   │   └── softmax_unit.sv         ← RTL design: package + module
│   │
│   ├── sim/
│   │   ├── luts/
│   │   │   ├── lut_exp.mem         ← 1D exp LUT (256 entries, generated)
│   │   │   └── lut_2d_flat.mem     ← 2D division LUT (256 entries, generated)
│   │   │
│   │   ├── lut_gen.py              ← Generates both .mem files
│   │   ├── softmax_accuracy.py     ← Python model of the RTL pipeline (no Vivado needed)
│   │   ├── rtl_accuracy.py         ← Generates test inputs & evaluates RTL outputs
│   │   ├── softmax_verify.py       ← Stub for LUT sanity checks
│   │   │
│   │   ├── rtl_inputs.txt          ← Test vectors fed to the Vivado testbench (generated)
│   │   ├── rtl_outputs.txt         ← RTL results captured by the testbench (generated)
│   │   ├── rtl_categories.txt      ← Input category labels for per-group accuracy (generated)
│   │   └── softmax_sim.vcd         ← Waveform dump from Vivado (auto-generated, ignore)
│   │
│   └── tb/
│       ├── softmax_unit_tb.sv      ← Functional testbench (3 directed tests)
│       └── softmax_accuracy_tb.sv  ← Accuracy testbench (500 random rows → rtl_outputs.txt)
│
└── softmax_sim/                    ← Vivado project (auto-generated, do not edit by hand)
    └── softmax_sim.xpr             ← Vivado project file
```

---

## What Each Python Script Does

| Script | What it does | When to run |
|---|---|---|
| `lut_gen.py` | Generates `lut_exp.mem` and `lut_2d_flat.mem` | Whenever LUT parameters change |
| `softmax_accuracy.py` | Simulates the full RTL pipeline **in Python**, no Vivado needed. Reads the `.mem` files and runs the same LUT logic as `softmax_unit.sv`. Quick way to check accuracy before even opening Vivado. | After regenerating LUTs |
| `rtl_accuracy.py --gen-inputs` | Generates `rtl_inputs.txt` — the 500 random test rows the Vivado testbench will read | Before running `softmax_accuracy_tb` in Vivado |
| `rtl_accuracy.py` (no flag) | Reads `rtl_outputs.txt` written by the Vivado testbench, compares against float softmax reference, and prints per-category argmax accuracy | After Vivado simulation completes |
| `softmax_verify.py` | Stub — intended for spot-checking LUT values | As needed |

---

## Full Workflow

### Step 1 — Generate LUTs

```bash
cd softmax/sim
python lut_gen.py
```

This writes `luts/lut_exp.mem` and `luts/lut_2d_flat.mem`.

Expected output:
```
Generating LUTs: 16x16 -> ...
  Written: lut_exp.mem      (256 entries)
  Written: lut_2d_flat.mem  (256 entries)
  lut_exp[0]=255  lut_exp[64]=93  lut_exp[128]=34  lut_exp[255]=4  OK
  All checks passed.
```

If you see `lut_exp[0]` is not 255, the path is wrong.

---

### Step 2 — Quick Python accuracy check (no Vivado)

```bash
python softmax_accuracy.py
```

This runs the RTL pipeline entirely in Python using the `.mem` files as the LUT contents.
Expected results:

```
  uniform    ~87%
  soft       ~95%
  peaked     100%
  extreme    100%
  OVERALL    ~96%
```

If numbers are significantly lower, regenerate the LUTs (`lut_gen.py`) before opening Vivado.

---

### Step 3 — Functional simulation in Vivado

In Vivado, set **`softmax_unit_tb`** as the simulation top.

```
Launch Simulation → Run All
```

Watch the Tcl console. Expected output:
```
[softmax] LUTs loaded OK (lut_exp[0]=0xFF)
-- Reset complete, starting tests --
-- Test 1: uniform (all 0.0) -------------------------
  PASS: all outputs equal
-- Test 2: one-hot (token 5 dominant) ----------------
  PASS: token 5 has highest output
-- Test 3: back-to-back rows --------------------------
  PASS: both rows peaked at correct index
-- All tests complete --------------------------------
```

If you see `WARNING: LUT load failed`, the `$readmemh` path in `softmax_unit.sv` is wrong — update the absolute path at the top of the file.

---

### Step 4 — Accuracy simulation in Vivado

First generate the input vectors:

```bash
python rtl_accuracy.py --gen-inputs
```

Then in Vivado, set **`softmax_accuracy_tb`** as the simulation top.

```
Launch Simulation → Run All
```

The testbench will process 500 rows and write `rtl_outputs.txt`. When it finishes you will see:
```
[tb] Wrote rtl_outputs.txt — run: python rtl_accuracy.py
```

Then evaluate:

```bash
python rtl_accuracy.py
```

Expected results match the Python simulation in Step 2 closely.

---

## Running Locally (New Machine Setup)

### Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | 3.9+ | |
| NumPy | any recent | `pip install numpy` |
| Vivado | 2020.1+ | Free WebPACK edition works (Artix-7 is supported) |

### Python setup

```bash
pip install numpy
```

That's the only dependency. All scripts are self-contained.

### Vivado project setup

The `softmax_sim/` folder contains the Vivado project. To open it on a new machine:

1. Open Vivado
2. **File → Open Project** → select `softmax_sim/softmax_sim.xpr`
3. The source files (`softmax_unit.sv`, both testbenches) should already be linked

**Update the hardcoded paths** — there are two places with absolute Windows paths you need to change to match your machine:

**`softmax_unit.sv`** (lines 33–34) — the `$readmemh` paths:
```systemverilog
$readmemh("C:/Users/IsaiahK/.../luts/lut_exp.mem",     lut_exp);
$readmemh("C:/Users/IsaiahK/.../luts/lut_2d_flat.mem", lut_2d_flat);
```

**`softmax_accuracy_tb.sv`** (lines 51, 103) — the input/output file paths:
```systemverilog
localparam string IN_FILE  = "C:/Users/IsaiahK/.../rtl_inputs.txt";
localparam string OUT_FILE = "C:/Users/IsaiahK/.../rtl_outputs.txt";
```

**`rtl_accuracy.py`** and **`softmax_accuracy.py`** — the `SIM_DIR` / `--lut-dir` paths at the top of each file.

> **Tip:** To avoid editing paths every time, in Vivado go to **Project Settings → Simulation** and add your `sim/luts/` folder to the simulation include directories. Then you can use bare filenames (`"lut_exp.mem"`) in `$readmemh` instead of absolute paths.

### If you are on Linux/Mac

The paths use forward slashes already in the SystemVerilog files so they'll work. In the Python scripts, replace the `SIM_DIR` string with your actual path using forward slashes or `os.path.join`.

---

## LUT Parameters

The LUT sizes are controlled by parameters in `softmax_pkg` (top of `softmax_unit.sv`) and matching arguments to `lut_gen.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `EX_BITS` | 4 | Log2 of exp LUT columns (16 buckets for ex_buf) |
| `SIGMA_BITS` | 4 | Log2 of sigma LUT rows (16 buckets for sigma) |
| `SEQ_LEN` | 16 | Number of attention tokens per row |

The default 16×16 = 256 entry 2D LUT fits entirely in LUTRAM (~32 LUT6s), using no BRAM.

To increase accuracy at the cost of more memory, increase `--sigma-bits` or `--ex-bits` in `lut_gen.py` and update the matching parameters in `softmax_unit.sv`.

---

## Reference

Vasyltsov, I. & Chang, W. (2021). *Efficient Softmax Approximation for Deep Neural Networks with Attention Mechanism.* arXiv:2111.10770.
