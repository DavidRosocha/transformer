# FPGA Transformer Accelerator

A transformer attention accelerator built from scratch on a **Basys 3 FPGA** (Artix-7 XC7A35T). A small PyTorch model completes 16×16 binary pixel-art sketches; its attention layers run on the FPGA in Q8.8 fixed-point, driven over UART from a Python host.

Draw a few pixels of a shape in the GUI and the board lights up the rest.

**Status: working end to end on hardware.** Weight load, both attention layers, softmax, and the result round trip all run on the real device.

---

## What Runs Where

The FPGA computes attention and nothing else. Everything the transformer needs around it — embedding, residuals, LayerNorm, the feedforward network, the output projection — stays on the PC, because those are cheap and attention is not.

Per layer, the host sends a 16×64 activation matrix and the FPGA returns a 16×64 result, having performed **seven matrix multiplies and a softmax** in one shot:

```
Q = X·W_q      K = X·W_k      V = X·W_v
scores = Q·Kᵀ
attn   = softmax(scores)          ← 16 rows, 16 wide
context = attn·V
output  = context·W_out
```

The host then applies residual + LayerNorm + FFN, and repeats for layer 2.

```
PC                                   FPGA
──────────────────────────────       ────────────────────────
16×16 grid
  → embed 16→64 + positional
  → ──────── 2 KB over UART ──────→  7 matmuls + softmax
                                     (layer-1 weights)
  ← ──────── 2 KB back ───────────
  → residual + LayerNorm + FFN
  → ──────── 2 KB over UART ──────→  7 matmuls + softmax
                                     (layer-2 weights)
  ← ──────── 2 KB back ───────────
  → residual + LayerNorm + FFN
  → output projection + sigmoid
16×16 completed grid
```

The `1/√64` attention scaling is pre-baked into `W_q` before the weights are sent, so the FPGA never has to divide.

---

## Model

| | |
|---|---|
| Input | 16×16 binary grid → 16 row-tokens of 16 pixels |
| Embedding | linear 16→64 + learned positional encoding |
| Blocks | 2 × (single-head attention, embed_dim 64, FFN hidden 32) |
| Output | linear 64→16 + sigmoid |
| Parameters | 44,816 |
| Numeric format | Q8.8 signed fixed-point everywhere (16-bit: 1 sign, 7 integer, 8 fractional) |

Trained for sketch completion — a random subset of a shape's lit pixels in, the full shape out. Pixel accuracy on the 1,242-pattern dataset:

| Fraction of shape shown | Accuracy |
|---|---|
| 30% | 89.5% |
| 50% | 93.0% |
| 70% | 94.5% |

---

## Hardware Design

### Systolic array — `systolic_array_8x8.sv`

An 8×8 output-stationary systolic array, 64 processing elements. Each PE holds an accumulator while operands flow through; partial sums never move. **8×8 is the largest that fits** — the XC7A35T has 90 DSP slices and the array consumes exactly 64, one per PE.

### Tile controller — `tile_controller.sv`

The array is 8×8 but the matrices are up to 64×64, so this walks 8×8 tiles across the operands and accumulates partial products into the scratch memory. It handles arbitrary M×K×N within those bounds, which is what lets one array serve all seven matmuls in the chain.

### Softmax — `softmax_unit.sv`

Softmax normally needs a divider, which is expensive. This uses the **2D LUT method** (Vasyltsov & Chang, 2021) — two small lookup tables and an adder, no divider and no multiplier. Mean absolute error ~2.2/255 (0.87%), argmax correct ~96%. See [`softmax/README.md`](softmax/README.md) and [`softmax/DESIGN_NOTES.md`](softmax/DESIGN_NOTES.md) for the derivation.

### Sequencer — `attention_fsm.sv`

The top-level module. Decodes UART frames, stores weights, and drives the whole attention chain: it reuses the single tile controller for each of the seven matmuls in sequence, feeds the 16 score rows through the softmax unit one at a time, and streams the result back out. `K` is stored transposed at write-back so the `Q·Kᵀ` matmul needs no separate transpose pass.

### Memory

| Block | Size | BRAM tiles |
|---|---|---|
| `weight_bram` | 8 matrices × 64×64 × Q8.8 = 64 KB | 16 |
| `scratch_bram` | 8,192 words = 16 KB (X, Q/K/V, scores, attn, context, out) | 8 |
| softmax LUTs | | 0.5 |

`scratch_bram` costs 8 tiles rather than 4 because it exposes three ports (one write, two reads) and a BRAM tile only has two — Vivado duplicates the array to serve both read ports independently. That dual read is what lets the tile controller stream both operands at once.

### Resource utilization

Post-implementation, XC7A35T:

| Resource | Used | Available | % |
|---|---|---|---|
| Slice LUTs | 3,334 | 20,800 | 16% |
| Slice registers | 5,551 | 41,600 | 13% |
| Slices | 2,030 | 8,150 | 25% |
| Block RAM tiles | 24.5 | 50 | 49% |
| **DSPs** | **64** | **90** | **71%** |

DSPs are the binding constraint, exactly as designed — 64 in the array, zero elsewhere. The softmax uses none, which is the whole point of the LUT approach.

---

## UART Protocol

921,600 baud, 8N1. All matrix data is Q8.8, row-major, big-endian.

```
PC → FPGA:   [0xAA] [TYPE] [payload] [XOR checksum] [0x55]
FPGA → PC:   [0xAA] [2048 bytes] [XOR checksum]
ACK:         [0x06]        — weight frames only
```

| TYPE | Meaning | Payload |
|---|---|---|
| `0x01` / `0x02` | Layer 1 / layer 2 inference | 2,048 B (16×64) |
| `0x10`–`0x17` | `W_q1 W_k1 W_v1 W_out1 W_q2 W_k2 W_v2 W_out2` | 8,192 B (64×64) |

The two directions are deliberately asymmetric:

- **The FPGA does not verify checksums.** It counts payload bytes and moves on, swallowing the trailing two. They are still sent so frames stay self-describing on a logic analyzer. There is no NAK.
- **Only weight frames are ACKed.** For inference frames the result *is* the acknowledgement — do not wait for an ACK after sending one, or you will consume the `0xAA` that opens the reply and desync.
- **The reply has no TYPE and no stop byte.** The host only ever receives one kind of frame. The host *does* verify the checksum it receives.

Weights are loaded once at startup and live in BRAM. They survive the serial port closing, but not a reset or a reprogram.

---

## Repository Layout

```
hardware/
  modules/
    attention_fsm.sv        top level — protocol FSM + attention sequencer
    systolic_array_8x8.sv   8×8 output-stationary MAC array
    tile_controller.sv      tiles matmuls up to 64×64 onto the 8×8 array
    softmax_unit.sv         2D-LUT softmax, no divider
    weight_bram.sv          64 KB weight storage
    scratch_bram.sv         16 KB intermediate storage
    uart_rx.v / uart_tx.v   serial front end
  constraints/
    basys3_tpu.xdc          pin assignments and timing

software/
  driver/python_driver.py   UART driver — framing, Q8.8, retries
  pipeline/
    load_weights.py         extract + transpose + prescale + upload weights
    pipeline.py             one-shot inference, prints ASCII grids
    interactive.py          live drawing GUI
  model/
    pixel_completer.py      TinyTransformer definition
    train_sketch.py         sketch-completion training
    model_sketch.pt         trained checkpoint (sketch task)
    model_best.pt           trained checkpoint (inpainting task)
  data/create_dataset.py    pattern dataset generator

softmax/                    softmax LUT generation, Python model, testbenches
verif/                      UVM environment and testcases
```

---

## Building and Running

### 1. Bitstream

Create a Vivado project targeting **xc7a35tcpg236-1**, add everything in `hardware/modules/`, add `hardware/constraints/basys3_tpu.xdc`, and set **`attention_fsm` as the top module**. Synthesize, implement, generate bitstream, program the board.

The port names in the XDC (`clk`, `rst`, `serial_rx`, `serial_tx`) must match `attention_fsm`'s ports. If Vivado reports `[Common 17-55] 'set_property' expects at least one object`, a constraint is naming a port that doesn't exist — the pin will end up unplaced and bitstream generation will fail on DRC `UCIO-1`/`NSTD-1`.

### 2. Find your serial port

```
cd software
.\.venv\Scripts\python.exe -m serial.tools.list_ports -v
```

Look for the FTDI hardware ID `VID:PID=0403:6010` — that's the Basys 3's FT2232H bridge. **The port number is machine-specific and changes with the USB socket**; do not assume the one in the examples below.

### 3. Load weights

Required after every power-up or reprogram — nothing else initializes weight BRAM.

```
cd software\pipeline
..\.venv\Scripts\python.exe load_weights.py --port COM6
```

Expect eight ACKs and a total time under a second. This doubles as a smoke test of the RX path, type dispatch, weight BRAM, and TX path.

### 4. Run the demo

```
..\.venv\Scripts\python.exe interactive.py --port COM6
```

Left-click to light a pixel, right-click to clear. Half a second after you stop drawing, both attention layers run on the board and the completion appears in the right-hand grid.

`--mock` runs the identical math on the CPU without opening the serial port, so it can run **simultaneously** as a side-by-side correctness check: draw the same shape in both windows and the predictions should match.

### Matching the model to the script

The checkpoint on the board must match the input format of the script driving it, or you get bad output that looks like a hardware fault:

| Script | Input format | Use with |
|---|---|---|
| `interactive.py` | binary {0,1}, un-drawn = off | `model_sketch.pt` |
| `pipeline.py` | {-1, 0, 1}, -1 = unknown | `model_best.pt` |

Both `load_weights.py` and the inference script must be pointed at the **same** `--model`, since the FPGA holds that checkpoint's attention weights while the host holds its FFN and LayerNorm.

---

## Verification

A UVM environment in `verif/` drives the design through its UART interface with a backdoor into weight BRAM for setup:

| Testcase | Covers |
|---|---|
| `tc_uart_rx` / `tc_uart_tx` | framing in both directions |
| `tc_weights_loaded` | all 8 matrices land in the right BRAM regions |
| `tc_mmul_correct` / `tc_mmul_random` | tile controller against a reference model |
| `tc_softmax_correct` | softmax unit against a Python reference |
| `tc_attention_full` / `..._layer2` | the full seven-matmul chain, both weight sets |

Run with `verif/sim/run.do`. The softmax unit additionally has standalone testbenches and a Python bit-accurate model in `softmax/`.

---

## Known Limitations

- **`W_q` quantization.** Because the `1/√64` scale is pre-baked into `W_q`, those weights span only ±0.045 — about 11 Q8.8 quantization levels, roughly 4 effective bits. Every other matrix spans ±0.3 (~90 levels). This is the first place to look if hardware output drifts from the CPU reference. Folding the scale elsewhere, or storing `W_q` at a different fixed-point scale, would recover the precision.
- **Tile controller `ACCUMULATE` state** burns 63 idle cycles per tile, doing real work only on the first. Correctness-first choice; a real latency win is available here.
- **Single head, fixed dimensions.** 64×64 weights and exactly 2 layers are hardcoded in `attention_fsm.sv` and `weight_bram.sv`. The host scripts check the checkpoint's shape and refuse a mismatch rather than producing garbage.
- **No flow control.** The design relies on the FPGA consuming bytes faster than they arrive at 921,600 baud, which holds comfortably at 100 MHz but leaves no margin at higher rates.

---

## Project History

Started as a four-person project with the matrix multiply, softmax, and UART front end as separate workstreams. Contributions from the others stopped partway through; the attention sequencer, weight storage and loading protocol, top-level integration, host pipeline, UVM environment, and hardware bring-up were completed solo.
