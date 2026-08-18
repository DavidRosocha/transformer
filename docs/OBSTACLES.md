# Obstacles and Design Challenges

A record of the problems hit while building this accelerator, why each happened, and what fixed it. Kept because the failures are more instructive than the finished design — several of these cost real time and would cost it again on the next project.

Entries marked **[bring-up]** were found on 2026-08-18 during hardware bring-up and the performance work that followed. The rest are design decisions recorded from the code and development history.

---

## 1. Toolchain and Synthesis

### 1.1 Silent constraint failure — a bitstream that couldn't build **[bring-up]**

**Symptom.** Bitstream generation failed with two DRC errors naming `serial_rx` and `serial_tx`:

```
[DRC NSTD-1] Unspecified I/O Standard: 2 out of 4 logical ports ...
[DRC UCIO-1] Unconstrained Logical Port: 2 out of 4 logical ports ...
[Vivado 12-1345] Error(s) found during DRC. Bitgen not run.
```

Much earlier in the log, easy to miss:

```
[Common 17-55] 'set_property' expects at least one object. [.../basys3_tpu.xdc:28]
```

**Cause.** The XDC constrained ports named `uart_rx_pin` and `uart_tx_pin`, but the top module `attention_fsm` declares them as `serial_rx` and `serial_tx`. The constraints file had been written against a planned `top.sv` wrapper that was never built — the FSM became the top module directly.

**Why it was hard to spot.** `get_ports` matching nothing is a **warning**, not an error. Vivado carried on, left both pins with no `LOC` and no `IOSTANDARD`, and the failure surfaced two steps later in a different tool (DRC at bitstream time) naming the RTL port names — which appear nowhere in the constraints file you'd go looking at. `clk` and `rst` matched fine, which is why it was 2 of 4 ports rather than all 4, making it look like a UART-specific problem rather than a naming problem.

**Fix.** Renamed both constraints to match the RTL.

**Takeaway.** After any top-module rename or refactor, grep the XDC against the actual port list. Treat `[Common 17-55]` as an error — it always means a constraint silently did nothing. A DRC error naming ports is usually an XDC naming problem, not a pin problem.

### 1.2 LUT initialization paths broke synthesis

The softmax unit loads its lookup tables with `$readmemh`. Relative paths that worked in simulation did not resolve under Vivado synthesis, which runs from a different working directory. Fixed in commit `869d241` ("updated paths for luts so synthesis won't fail").

**Takeaway.** `$readmemh` paths are resolved relative to the tool's working directory, and simulator and synthesizer disagree about what that is.

---

## 2. Resource Constraints That Shaped the Design

### 2.1 The DSP budget set the array size

The XC7A35T has **90 DSP slices**. An output-stationary systolic array needs one multiplier per processing element, so an N×N array costs N² DSPs. 8×8 = 64 fits with 26 to spare; 9×9 = 81 fits but leaves nothing; 10×10 = 100 does not fit at all.

**8×8 was not a preference, it was the ceiling.** Post-implementation confirms exactly 64 DSPs used, one per PE, with zero leakage into the softmax or address arithmetic. There is no room for a second array — that would need 64 more.

This is the single constraint that determined the whole architecture: because the array is 8×8 but the matrices are up to 64×64, a **tile controller** became mandatory, and because there is only one array, all seven matmuls in the attention chain must be **sequenced through it** rather than run in parallel.

### 2.2 Softmax needs a divider, and dividers are expensive

Standard softmax requires `exp(x_i) / Σexp(x_j)`. A hardware divider is large and slow, and with DSPs already fully committed to the array there was no budget for one.

**Solution.** The 2D LUT method (Vasyltsov & Chang, 2021): two small lookup tables and an adder, no divider and no multiplier. Costs 244 LUTs and half a BRAM tile, **zero DSPs**. Accuracy: mean absolute error ~2.2/255 (0.87%), argmax correct ~96%.

**Takeaway.** When one resource is saturated, the fix is often to move the operation onto a different resource entirely rather than to make it smaller.

### 2.3 Block RAM came in above budget — the three-port problem **[bring-up]**

**Symptom.** BRAM utilization was 24.5 of 50 tiles (49%), against an expected ~35%.

**Cause.** Accounting for it:

| Block | Logical size | Tiles |
|---|---|---|
| `weight_bram` | 32,768 × 16b = 512 Kb | 16 |
| `scratch_bram` | 8,192 × 16b = 128 Kb | **8** |
| softmax LUTs | | 0.5 |

`scratch_bram` should need 4 tiles for 128 Kb. It takes 8 because it exposes **three ports** — one write and two independent reads — and a BRAM tile physically has only two. Vivado's only option is to duplicate the entire array and serve one read port from each copy.

**Resolution: none needed.** That dual read is what lets the tile controller stream both matmul operands simultaneously; serializing them would slow every tile. Paying 4 extra tiles (8% of the device) for it is the right trade. Documented rather than fixed.

**Takeaway.** Port count, not capacity, is often what determines BRAM cost. Any inferred memory with more than two ports silently doubles.

### 2.4 Avoiding a transpose pass

`scores = Q · Kᵀ` needs K transposed. Rather than spend a pass reading and rewriting K, **K is written to scratch memory already transposed** at write-back time. The transpose is free — it costs only different address arithmetic on a write that was happening anyway.

---

## 3. Correctness Hazards in the RTL

### 3.1 The one-cycle TX send gap

`uart_tx` latches `send` and raises `busy` on the *following* clock edge. For exactly one cycle after a send pulse, the transmitter looks idle while already being committed. Gating the next send on `!tx_busy` alone lets a second send fire into that gap, where `uart_tx` has already left IDLE and ignores it — **the byte is silently dropped**.

**Fix.** A `tx_pending` flag covers the gap:

```systemverilog
assign tx_ready = !tx_busy && !tx_pending;
```

**Takeaway.** A "busy" flag that asserts a cycle late is not a valid interlock. This class of bug drops one byte in a 2 KB frame and shows up as a checksum mismatch, far from the cause.

### 3.2 Accumulate wastes 63 cycles per tile

The tile controller's `ACCUMULATE` state does real work only on its first cycle and idles for 63 more. Across ~576 tiles per inference that is meaningful — but it was left in deliberately.

**Reasoning at the time:** the design was already state-heavy and not latency-optimized. Pipelining requires tracking which cycle's data belongs to which operation as it flows through, especially with seven matmuls sequenced back to back, and getting that wrong produces exactly the kind of bug that is miserable to debug on hardware. Correctness first; optimize after profiling.

**Vindicated by measurement.** Bring-up showed total FPGA compute is **under 1 ms** against ~27 ms per round trip. The wasted cycles were never the bottleneck. Optimizing them first would have been wasted effort on a real risk of introducing bugs.

**Takeaway.** "Obviously inefficient" is not the same as "on the critical path." Profile before optimizing — this decision was made on instinct and turned out right, but only measurement proved it.

### 3.3 BRAM read latency — wait states over pipelining

Block RAM reads have a cycle of latency, so an address issued this cycle returns data next cycle. Two options: pipeline address generation to hide it, or insert a wait state. **Wait states won**, for the same reasoning as 3.2 — one extra cycle per fetch was negligible next to `ACCUMULATE`, and pipelining adds data-tracking complexity that is hard to debug on hardware.

---

## 4. Numerical Precision

### 4.1 Silent overflow when quantizing to Q8.8

Casting an out-of-range float directly to `int16` **wraps around silently** in numpy — `200.0` becomes garbage rather than saturating. The conversion therefore clamps as float *before* casting:

```python
flat = np.clip(flat, Q8_8_MIN, Q8_8_MAX)   # clamp first, as float
scaled = np.round(flat * 256.0).astype(np.int16)
```

**Takeaway.** Quantization must saturate explicitly. The default behaviour of a cast is wraparound, which turns a large value into a large *negative* one — the worst possible failure mode for weights.

### 4.2 W_q has only ~4 effective bits **[bring-up]**

**Symptom.** Checking weight ranges before upload:

```
W_q1     min= -0.0455 max=  0.0430
W_k1     min= -0.3895 max=  0.3748
W_out1   min= -0.2844 max=  0.2508
```

**Cause.** Attention divides scores by `√embed_dim` = 8. The FPGA has no divider in its score path, so the `1/8` is pre-baked into `W_q` on the host before quantization. That division crushes W_q into the bottom of the Q8.8 range: ±0.045 against a resolution of 1/256 ≈ 0.0039 is about **11 quantization levels**, roughly 4 effective bits. Every other matrix spans ±0.3 and gets ~90 levels.

**Status: open.** Not causing visible degradation, but it is the first place to look if hardware output ever drifts from the CPU reference. The fix would be to fold the scale somewhere else — into K, or into the softmax input — or to store W_q at a different fixed-point scale and shift after the multiply.

**Takeaway.** Pre-scaling to avoid a hardware divider is free in float and expensive in fixed point. Check the dynamic range of anything you pre-scale.

---

## 5. Protocol Design

The UART protocol is deliberately asymmetric, and each asymmetry is a decision with a trap attached.

### 5.1 The FPGA does not verify checksums

It counts payload bytes and moves on, swallowing the trailing checksum and stop byte. They are still transmitted so frames stay self-describing on a logic analyzer, but they carry no authority and there is no NAK. Verification happens only on the host, on the return path.

**Consequence.** An ACK means "8,192 bytes were counted in," not "the data is correct."

### 5.2 Only weight frames are ACKed

Inference frames get no ACK — the result frame *is* the acknowledgement. **Reading a byte after sending an inference frame steals the `0xAA` that opens the reply and desyncs every subsequent read.** With no ACK there is no handshake to resynchronize on, which is why the host flushes its input buffer before *every* attempt, not just retries: a single stale byte would be mistaken for a start byte forever after.

### 5.3 Weight loading has no mode, only ordering

There is no "weight loading mode" on the FPGA — every frame is dispatched purely on its TYPE byte. But **nothing else initializes weight BRAM**, so all eight matrices must arrive before the first inference or the result is noise. The ordering requirement is real even though the state machine has no concept of it.

Weights persist in BRAM across the serial port closing, but not across a reset button press or a reprogram.

---

## 6. Host-Side Integration

### 6.1 The COM port was never COM3 **[bring-up]**

Every script's docstring and `--help` example said `COM3`. The board enumerates as **COM6** on this machine, and COM3 does not exist at all — it was a placeholder that had never been verified because nothing had been run on hardware yet.

**Fix.** Identify the board by its FTDI hardware ID rather than assuming:

```
python -m serial.tools.list_ports -v
→ COM6   hwid: USB VID:PID=0403:6010   ← FT2232H, the Basys 3 bridge
```

`0403:6010` is FTDI's FT2232H. Windows assigns port numbers per USB socket, so moving the cable changes the number.

### 6.2 Model and input format must match — the trap that looks like a hardware fault **[bring-up]**

**The setup.** Two trained checkpoints exist for two different tasks:

| Checkpoint | Task | Input format |
|---|---|---|
| `model_best.pt` | inpainting | `{-1, 0, 1}` — -1 means "unknown" |
| `model_sketch.pt` | sketch completion | `{0, 1}` — un-drawn is simply off |

`model_sketch.pt` exists *because* the inpainting model failed at the actual demo task: it needs revealed background pixels to know where the shape *isn't*, and a drawing GUI never provides those — every un-drawn cell is blank, so the model floods.

**The trap.** `load_weights.py` defaults to `model_sketch.pt`, but `pipeline.py` builds its input with `mask_grid()`, which writes **-1** into masked cells — the inpainting format. Running `pipeline.py` with defaults feeds -1 values to a model that has never seen them. The accuracy comes back bad, and it looks exactly like a hardware fault.

**Compounding it:** the FPGA holds one checkpoint's attention weights while the host holds that same checkpoint's FFN and LayerNorm. **Both must come from the same `--model`** or the halves compute different models.

**Takeaway.** When a computation is split across two machines, the split becomes a versioning surface. A mismatch produces plausible-looking wrong answers rather than an error.

---

## 7. Performance — Where the Time Actually Went **[bring-up]**

This section is the most instructive, because reasoning was right about the mechanism and wrong about the outcome twice in a row.

### 7.1 The accelerator was never the bottleneck

Measured, at 4 Mbaud: **27.3 ms per layer round trip**, of which 10.3 ms is wire time and ~16 ms is fixed overhead. That leaves **under 1 ms for all seven matmuls and the softmax**. The FPGA does its job in about 3% of the time the system spends per inference.

### 7.2 Choosing a baud rate — the dual-divisibility constraint

Two clocks must agree, and both divide with integer truncation:

- **FT2232H bridge:** baud = 12 MHz ÷ n
- **FPGA:** `CLKS_PER_BIT` = 100 MHz ÷ baud, truncated

Exactness on both sides requires 100M/baud = m and 12M/baud = n with m, n integers, giving **25n = 3m** — so n must be a multiple of 3. The smallest is n = 3:

| Rate | Bridge | FPGA clks/bit | Verdict |
|---|---|---|---|
| 921,600 | 12M/13.02 → **+0.16% error** | 108.51 → 108 (+0.47%) | inexact on *both* sides |
| 2,000,000 | 12M/6 exact | 50.0 exact | exact |
| **4,000,000** | **12M/3 exact** | **25.0 exact** | **exact — the maximum** |
| 5,000,000 | 12M/2.4 not representable (+1.05%) | 20.0 exact | works, but inexact |
| 6,000,000 | 12M/2 exact | 16.67 → 16 (**+4.17%**) | fails |

**4 Mbaud is provably the highest rate both clocks divide exactly.** Anything faster forces n = 1 or 2, giving 8 M or 12 M, where the FPGA lands on 12.5 or 8.33 clocks per bit.

Why 6 M genuinely fails: the last data bit is sampled 8.5 bit-times after the start edge, so a 4.17% rate error displaces it by ~0.35 bit against a 0.5 bit budget — and the two-flop input synchronizer at only 16 clocks per bit adds another ~0.19. Total exceeds the budget.

Worth noting: **921,600 was itself inexact on this hardware.** It is a 1.8432 MHz-family number being approximated by a 12 MHz-family bridge. Moving to 4 M made the link *cleaner*, not more marginal.

### 7.3 The baud change didn't feel faster — and that was correct

Wire time genuinely dropped 89 ms → 20.5 ms per inference. It was imperceptible, because two fixed costs that **do not scale with baud** dominated:

| Component | Before | After baud | After all fixes |
|---|---|---|---|
| GUI debounce | 500 ms | 500 ms | 150 ms |
| Wire (2 layers) | 89 ms | 20.5 ms | 20.5 ms |
| FTDI latency timer | 32 ms | 32 ms | 2 ms |
| FPGA compute | ~2 ms | ~2 ms | ~2 ms |
| **Perceived** | **~623 ms** | **~555 ms** | **~174 ms** |

An 11% improvement is right at the threshold of human detectability. The perception was accurate; the change was real but was not where the time was.

**Confirmed by measurement** after all three fixes — 20 round trips, same board, same script:

| | 4 Mbaud, 16 ms timer | 4 Mbaud, 1 ms timer |
|---|---|---|
| Per layer | 27.30 ms | **12.00 ms** |
| Overhead above wire | 17.04 ms | **1.75 ms** |
| Full inference | 54.6 ms | **24.0 ms** |
| Jitter (max − min) | 1.78 ms | **0.22 ms** |

The jitter collapse is the clearest evidence the diagnosis was right: each round trip previously entered the 16 ms flush window at an arbitrary phase, scattering completion times across ~1.8 ms. With a 1 ms timer that scatter drops to 0.22 ms. A fixed timeout produces phase-dependent jitter; genuine compute does not.

### 7.4 The FTDI latency timer — a 16 ms tax invisible until it wasn't

**Symptom.** Benchmarking showed 27.3 ms per layer against 10.3 ms of wire time. 17 ms unaccounted for.

**Cause.** The FT2232H holds received bytes until either 62 accumulate (one USB packet) or a **latency timer** expires. That timer defaults to **16 ms**. A 2,050-byte result frame is 33 full packets plus a 4-byte remainder — and that remainder waits out the entire timer, on every single round trip.

Confirmed by reading the driver's own setting:

```
HKLM\SYSTEM\CurrentControlSet\Enum\FTDIBUS\...\Device Parameters
    PortName     : COM6
    LatencyTimer : 16
```

**Fix.** Device Manager → Ports → USB Serial Port → Properties → Port Settings → Advanced → **Latency Timer (msec) = 1**, then replug.

**The interesting part.** This cost was always present. At 921,600 it was 26% of round-trip time and invisible. At 4 Mbaud it became 59% of what remained. **Raising the baud rate promoted it into being the bottleneck.** Optimizing the largest term revealed a fixed term that had been hiding behind it.

**Takeaway.** Fixed per-transaction costs don't shrink when you speed up the variable ones — they take over. And this one was found by measuring, not reasoning: the wire math was right and still predicted the wrong outcome.

---

## 8. Process and Tooling

### 8.1 Duplicated constants across RTL and testbench **[bring-up]**

The UVM agent hardcoded `CLKS_PER_BIT = 108` in **two** places — the driver and the monitor — derived by hand from the RTL's parameter rather than shared with it. Changing the RTL baud would have left the testbench driving 921,600 at a 4 Mbaud DUT, failing every testcase for a reason unrelated to the design.

Caught by grepping for every baud reference before changing any of them. Now 25 in both places.

**Takeaway.** A constant copied from RTL into a testbench is a latent failure with a long fuse. When changing a parameter, grep the whole tree — verification included — not just the module.

### 8.2 Documentation drift **[bring-up]**

`CLAUDE.md` had gone badly stale: it listed seven completed components as "needs to be built," claimed ~69,000 model parameters (actual: **44,816**) and `hidden_dim=128` (actual: **32**), and its "known issues" section described bugs already fixed. It was deleted and replaced with a `README.md` written from measured values and read source.

**Takeaway.** Documentation that states quantities will drift silently. Anything numeric should be re-derived, not carried forward.

### 8.3 A measurement bug that looked like a model failure **[bring-up]**

While measuring accuracy for the README, the sketch model appeared to score **30.1%** — catastrophically bad. The cause was in the measurement, not the model: `TinyTransformer.forward()` already applies sigmoid (training uses `binary_cross_entropy`, not the `_with_logits` variant), so applying it again mapped everything into [0.5, 0.73] and thresholding at 0.5 predicted **every pixel on**. 30.1% was simply the fraction of lit pixels in the dataset.

Correct figures: 89.5% / 93.0% / 94.5% at 30% / 50% / 70% of the shape shown.

**Takeaway.** A wrong number that coincidentally matches some other quantity in the problem is a strong tell. "Predicted everything true" and "base rate of true" are the same number — worth recognizing on sight.

---

## Summary — Recurring Themes

1. **Silent failures cost the most.** The XDC warning, the dropped TX byte, the numpy wraparound, and the stale testbench constant all fail quietly and surface far from the cause. Every one was found by checking rather than by being told.

2. **Measure before optimizing, and again after.** The `ACCUMULATE` inefficiency looked like the obvious target and was irrelevant. The baud rate was a real 4× win that changed nothing perceptible. The actual bottleneck was a driver setting nobody had looked at.

3. **One saturated resource shapes everything.** 90 DSPs set the array at 8×8, which forced the tile controller, which forced sequencing all seven matmuls through one array, which made the design state-heavy — and made the divider-free softmax necessary rather than merely elegant.

4. **Split computation creates a versioning surface.** Half the model on the FPGA and half on the host means a mismatch produces plausible wrong answers instead of an error.
