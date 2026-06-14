# Softmax Unit — Full Design Notes
### Private reference document — not for the repo

---

## 1. What We're Building and Why

The softmax unit is step 5 in the attention accelerator pipeline:

```
x ──► MatMul(W_q) ──► Q ──┐
x ──► MatMul(W_k) ──► K ──┼──► scores = Q @ K^T ──► softmax(scores) ──► attn
x ──► MatMul(W_v) ──► V ──┘                                                 │
                                                         out = attn @ V ◄───┘
```

Softmax is defined as:

```
softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
```

The problem is that this requires:
1. An exponential function (expensive in hardware)
2. A division (very expensive in hardware — needs a multi-cycle divider or DSP)

For a Basys 3 FPGA with limited resources, we can't afford either. So we use a lookup table approximation based on the paper: **Vasyltsov & Chang, arXiv:2111.10770, 2021**.

---

## 2. The Algorithm — 2D LUT Method

### 2.1 Max normalization

Standard softmax first subtracts the max for numerical stability:

```
softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

This is critical because `exp(x)` overflows for large `x`, but after subtracting the max, all inputs are <= 0, so all exp values are in `(0, 1]`.

### 2.2 Two LUT approach

Instead of computing division directly, we precompute a 2D table indexed by:
- The numerator `exp(x_i - max)` — bucketed into the top 4 bits of the exp value
- The denominator `sum_j(exp(x_j - max))` — bucketed into the top 4 bits of sigma

```
LUT_sigma[numerator_idx][denominator_idx] = floor(numerator/denominator * 255)
```

Both indices are just the top 4 bits of their respective values — this is "free" in hardware, it's just wiring, no compute needed.

### 2.3 Step by step

```
Step 1: Find running_max as tokens stream in
Step 2: For each token i:
            diff = running_max - token_i       (Q8.8, always >= 0)
            addr = CLAMP(diff >> 2, 0, 255)    (scale to LUT range)
            ex_buf[i] = lut_exp[addr]           (~= exp(-diff/256) * 255)
Step 3: sigma = sum of all ex_buf values
Step 4: For each token i:
            sigma_idx = sigma[11:8]             (top 4 bits)
            ex_idx    = ex_buf[i][7:4]          (top 4 bits)
            out[i]    = lut_2d_flat[{sigma_idx, ex_idx}]
```

---

## 3. Fixed-Point Number Format

Inputs are **Q8.8 signed fixed-point**:
- 16 bits total: 1 sign bit, 7 integer bits, 8 fractional bits
- 256 Q8.8 units = 1.0 in float
- Range: -128.0 to +127.996

Examples:
```
256   → 1.0
-256  → -1.0
487   → 1.902
-416  → -1.625
```

Outputs are **uint8** in range 0..255, where 255 represents a softmax probability of 1.0.

---

## 4. The LUT Address Scaling Problem (and How We Solved It)

### 4.1 The original broken approach

The first version computed the LUT address as:

```systemverilog
lut_addr_exp = CLAMP(diff_full, 0, 255)
lut_exp[addr] = floor(exp(-addr / 256) * 255)
```

This looks correct at first glance but has a fatal flaw: in Q8.8, `diff=255` represents only `255/256 ≈ 1.0` in float. So the entire LUT only covered a float range of **0.0 to ~1.0 below the max**.

Real attention scores span roughly ±2.0 to ±4.0 in float, meaning the max-to-min spread can be 6+ units. A token 2.0 below the max has `diff = 2*256 = 512` in Q8.8 — this gets clamped to 255 and assigned `lut_exp[255] = 94`, the same value as a token only 1.0 below the max.

**Result:** 74.2% of tokens got the wrong weight. All "small" tokens got assigned roughly the same non-zero value instead of near-zero.

Accuracy with broken approach:
```
uniform:  86%
soft:     98%    ← seemed OK because soft inputs are tightly clustered
peaked:  100%    ← max token dominates regardless
extreme: 100%
mean absolute error: 10.4/255 (4.1%)
```

The argmax numbers looked deceptively OK because peaked/extreme inputs dominate anyway.
But the absolute values were wrong — `sum(outputs)` was 240 instead of 255, and
low-probability tokens all had value ~11 instead of near 0.

### 4.2 Why the old formula was wrong

The formula `exp(-addr/256)` treats each address step as `1/256` in float. But `addr` was the raw Q8.8 diff, where `256 units = 1.0`. So `addr=256` meant "1.0 below max" but `exp(-256/256) = exp(-1)` is correct! The math is actually right for small diffs — but then we clamp `addr` to 255 maximum, so anything further than `255/256 ≈ 0.996` below max gets the same value. The coverage is the problem, not the formula per se.

### 4.3 The fix: right-shift by 2

```systemverilog
lut_addr_exp = CLAMP(diff_full >> 2, 0, 255)
lut_exp[addr] = floor(exp(-addr / 64) * 255)
```

By shifting right by 2, each address step covers 4 Q8.8 units = `4/256 = 1/64` in float.
The full 256-entry LUT now covers `256/64 = 4.0` float units below the max.

Spot checks:
```
addr=0   → diff=0,    0.0 below max → exp(0)*255   = 255  ✓
addr=64  → diff=256,  1.0 below max → exp(-1)*255  = 93   ✓
addr=128 → diff=512,  2.0 below max → exp(-2)*255  = 34   ✓
addr=192 → diff=768,  3.0 below max → exp(-3)*255  = 12   ✓
addr=255 → diff=1020, 3.98 below max → exp(-3.98)*255 = 4 ✓ (near zero)
```

Accuracy after fix:
```
uniform:  86-87%
soft:     95-98%
peaked:  100%
extreme: 100%
mean absolute error: 2.2/255 (0.87%)
```

### 4.4 Why not shift by 3 or 8?

- Shift by 3: covers 8.0 float range, but each step is 1/32 coarse — too much quantization noise within the 0-2.0 range where most of the signal lives
- Shift by 8 (integer part only): covers 255.0 float range, wildly more than needed. Worse, all tokens within the same integer band (e.g. 1.0 to 1.999) get the same ex value, destroying argmax discrimination for soft distributions. Argmax dropped from 96% to 61%.
- Shift by 2 is the sweet spot: fine enough for discrimination, wide enough for real score ranges.

---

## 5. RTL Architecture

### 5.1 State machine

```
         in_valid && in_first
S_IDLE ─────────────────────► S_LOAD
  ▲                               │ in_valid && cnt==SEQ_LEN-2
  │                               ▼
S_OUTPUT ◄──── S_ACCUMULATE ◄── S_SUBTRACT
cnt==SEQ_LEN-1  cnt==SEQ_LEN-1   cnt==SEQ_LEN-1
```

### 5.2 Timing diagram

```
Cycle:    0    1    2  ...  15   16   17  ...  31   32  ...  47   48  ...  63
State:    IDLE LOAD LOAD... LOAD SUB  SUB ... SUB  ACC ... ACC  OUT ... OUT  IDLE

in_valid: 1    1    1  ...  1    0    0  ...  0    0  ...  0    0  ...  0
in_first: 1    0    0  ...  0
cnt:      -    0    1  ...  14   0    1  ...  15   0  ...  15   0  ...  15

                                      ┌── addr computed combinationally
                                      │    ┌── ex_buf written (pipeline delay)
                                      ▼    ▼
subtract stage:               [addr0][ex0][addr1][ex1]...[addr15][ex15]

sigma:                                                    [0][1+2+...+15 accumulated]

out_valid:                                                              1 1 1...1
```

Total latency: 16 cycles input + 16 cycles subtract + 16 cycles accumulate + 16 cycles output = **64 cycles** from first token to last output, at 100 MHz = **640 ns**.

### 5.3 The one-cycle subtract pipeline

The subtract stage has a one-cycle register between computing `addr` and writing `ex_buf`:

```
Cycle N:   cnt=i → diff_full computed combinationally → lut_addr_exp computed
Cycle N+1: addr_d = lut_addr_exp, cnt_d = cnt → ex_buf[cnt_d] = lut_exp[addr_d]
```

This pipeline register (`cnt_d`, `addr_d`, `subtract_d`) exists because LUT reads in FPGAs are typically registered for timing closure. The `subtract_d` signal gates the write so ex_buf isn't overwritten during other states.

### 5.4 Token capture: why token 0 is special

Token 0 is captured in S_IDLE (on the `in_first` cycle) rather than S_LOAD. This is because the FSM transitions from S_IDLE to S_LOAD on that same cycle — if token 0 were captured in S_LOAD, the state transition would happen before the write.

Token 0 capture: `S_IDLE, in_valid && in_first → row_buf[0] = in_data`
Tokens 1-15: `S_LOAD, in_valid → row_buf[cnt+1] = in_data` (cnt runs 0..14)

The S_LOAD transition condition `cnt == SEQ_LEN-2` (cnt=14) is correct:
- At cnt=14, `row_buf[14+1] = row_buf[15]` is written
- The transition fires, resetting cnt to 0 on the next clock
- All 16 slots are populated before S_SUBTRACT begins

### 5.5 Sigma accumulator detail

The accumulator has a special case for cnt=0:
```systemverilog
if (cnt == 4'd0)
    sigma <= {4'b0, ex_buf[0]};   // load, don't add (avoids needing reset)
else
    sigma <= sigma + {4'b0, ex_buf[cnt]};
```

This resets sigma implicitly at the start of each accumulation pass rather than requiring an explicit reset state.

---

## 6. LUT Design Details

### 6.1 lut_exp (1D, 256 entries)

```python
lut_exp[addr] = floor(exp(-addr / 64) * 255)
```

Generated by `lut_gen.py → generate_lut_exp()`. 256 bytes.

Key values:
| addr | float diff | exp value | uint8 |
|------|-----------|-----------|-------|
| 0    | 0.000     | 1.000     | 255   |
| 32   | 0.500     | 0.607     | 154   |
| 64   | 1.000     | 0.368     | 93    |
| 128  | 2.000     | 0.135     | 34    |
| 192  | 3.000     | 0.050     | 12    |
| 255  | 3.984     | 0.019     | 4     |

### 6.2 lut_2d_flat (2D, 16×16 = 256 entries)

```python
lut_2d_flat[sigma_idx][ex_idx] = floor(ex_mid / sig_mid * 255)
where:
  ex_mid  = (ex_idx + 0.5) * (256 / 16)     # midpoint of ex bucket
  sig_mid = (sigma_idx + 0.5) * (4096 / 16) # midpoint of sigma bucket
```

This approximates `ex / sigma * 255` using bucket midpoints. The flat indexing is:
```
flat_index = sigma_idx * 16 + ex_idx = {sigma_idx, ex_idx}  (bit concat)
```

Generated by `lut_gen.py → generate_lut_2d_flat()`. 256 bytes.

### 6.3 FPGA resource usage

Total LUT data: 256 + 256 = **512 bytes**. This fits in LUTRAM (distributed RAM built from LUT6 slices) with ~64 LUT6s, consuming zero BRAM18 blocks.

Scaling options:
| Config | Size | BRAM | Uniform acc | Overall acc |
|--------|------|------|-------------|-------------|
| 16×16  | 256B | LUTRAM | 87% | 96% |
| 16×32  | 512B | LUTRAM | ~91% | ~97% |
| 32×32  | 1024B | 1× BRAM18 | ~94% | ~98% |

---

## 7. Testbench Notes

### 7.1 Clock-edge driving race condition

Early testbench versions drove `in_valid/in_first/in_data` directly after `@(posedge clk)` with no delay. In Vivado xsim this creates a race: both the DUT's `always_ff` blocks and the TB's procedural assignments see the same clock edge at the same simulation time. Whether the DUT latches the old or new value is simulator-dependent.

**Fix:** All stimulus is driven with `#1` delay after the posedge:
```systemverilog
@(posedge clk); #1;
in_valid = 1'b1;
```

This puts the drive firmly in the "after clock edge" window.

### 7.2 The file-swap incident

At one point the local files were accidentally swapped — `softmax_unit.sv` contained the testbench code and `softmax_unit_tb.sv` contained the RTL. Vivado couldn't elaborate because it was trying to instantiate `softmax_unit` inside itself. Lesson: always check module names match filenames before debugging RTL.

### 7.3 Accuracy testbench: fork/join vs sequential

The original accuracy testbench used `fork/join` to run `send_row` and `capture_row` simultaneously:
```systemverilog
fork
    send_row(row);     // drives inputs
    capture_row(row);  // waits for out_valid
join
```

This seemed fine because `capture_row` just waits for `out_valid` which comes much later. But the 2-cycle gap between rows wasn't enough margin — if a row finished slowly, the next row's `send_row` could start before `capture_row` finished, causing output bleeding between rows.

**Fix:** Sequential execution with `wait(busy == 0)` between rows:
```systemverilog
send_row(row);
capture_row(row);
wait (busy == 1'b0);
repeat(2) @(posedge clk);
```

---

## 8. Accuracy Results

### 8.1 Input categories used for testing

| Category | Description | Typical real-world case |
|----------|-------------|------------------------|
| uniform | All tokens in [-2.0, 2.0] | Early training, random attention |
| soft | Base [-0.5, 0.5], one token +0.5 to +1.5 | Mild attention preference |
| peaked | Base [-1, 1], one token +2 to +4 | Strong attention to one token |
| extreme | Base [-1, 0], one token set to [3, 4] | Very strong single-token attention |

### 8.2 Final accuracy (after lut_exp scale fix)

```
uniform:  86%   ← hardest case; nearly-equal tokens, small errors flip argmax
soft:     98%
peaked:  100%   ← what real transformer attention usually looks like
extreme: 100%
OVERALL:  96%
mean absolute error: 2.2/255 (0.87%)
```

Peaked and extreme are 100% because even with approximation errors, the dominant token's weight is so much larger that no rounding error can flip the argmax.

Uniform is the hardest because when all 16 tokens are nearly equal, the true softmax values are each ~16/255 ≈ 6 per token, and an error of even 2 counts can flip which token appears largest.

### 8.3 Comparison to paper

The paper (Table 2, uint8, 2D LUT method) reports accuracy drop below 1% for NLP tasks. Our 0.87% mean error matches this. The paper uses slightly different LUT sizes (11×60) optimized for their specific input distributions.

---

## 9. Flowcharts

### 9.1 Full pipeline flowchart

```
                    ┌─────────────────────────────────────────────┐
                    │              S_IDLE                          │
                    │  Wait for in_valid && in_first               │
                    │  On arrival: row_buf[0] = in_data            │
                    │             running_max = in_data             │
                    └──────────────────┬──────────────────────────┘
                                       │ in_valid && in_first
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │              S_LOAD                          │
                    │  For each new token (cnt = 0..14):           │
                    │    row_buf[cnt+1] = in_data                  │
                    │    if in_data > running_max:                  │
                    │        running_max = in_data                  │
                    └──────────────────┬──────────────────────────┘
                                       │ in_valid && cnt==14
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │            S_SUBTRACT                        │
                    │  For cnt = 0..15:                            │
                    │    diff = running_max - row_buf[cnt]         │
                    │    addr = CLAMP(diff >> 2, 0, 255)           │
                    │    [next cycle] ex_buf[cnt] = lut_exp[addr]  │
                    └──────────────────┬──────────────────────────┘
                                       │ cnt==15
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │           S_ACCUMULATE                       │
                    │  sigma = ex_buf[0]                           │
                    │  for cnt = 1..15:                            │
                    │      sigma += ex_buf[cnt]                    │
                    └──────────────────┬──────────────────────────┘
                                       │ cnt==15
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │             S_OUTPUT                         │
                    │  For cnt = 0..15:                            │
                    │    sigma_idx = sigma[11:8]                   │
                    │    ex_idx    = ex_buf[cnt][7:4]              │
                    │    out_data  = lut_2d_flat[{sigma_idx,ex_idx}]│
                    │    out_valid = 1                             │
                    └──────────────────┬──────────────────────────┘
                                       │ cnt==15
                                       ▼
                                    S_IDLE
```

### 9.2 LUT address computation flowchart

```
  running_max (Q8.8)
        │
        │  - row_buf[cnt] (Q8.8)
        ▼
  diff_full [15:0]  (always >= 0)
        │
        │  >> 2  (right shift, free in hardware)
        ▼
  diff_full[13:0] >> 2 = diff_full[15:2]
        │
        │  Is diff_full[15:2] > 255?
       / \
      Y   N
      │   │
      │   └──► lut_addr_exp = diff_full[9:2]  (8-bit slice)
      │
      └──────► lut_addr_exp = 8'hFF           (clamp to 255)
                                    │
                                    ▼
                             lut_exp[lut_addr_exp]
                                    │
                                    ▼
                            ex_buf[cnt] (next cycle)
```

### 9.3 2D LUT lookup flowchart

```
  sigma [11:0]              ex_buf[cnt] [7:0]
       │                          │
       │  [11:8] (top 4 bits)     │  [7:4] (top 4 bits)
       ▼                          ▼
  sigma_idx [3:0]           ex_idx [3:0]
       │                          │
       └──────────┬───────────────┘
                  │  concatenate {sigma_idx, ex_idx}
                  ▼
           flat_idx [7:0]  =  sigma_idx * 16 + ex_idx
                  │
                  ▼
         lut_2d_flat[flat_idx]
                  │
                  ▼
            out_data [7:0]
```

---

## 10. Files Quick Reference

| File | Purpose |
|------|---------|
| `rtl/softmax_unit.sv` | RTL: package + module |
| `sim/lut_gen.py` | Generates lut_exp.mem and lut_2d_flat.mem |
| `sim/softmax_accuracy.py` | Python model of RTL — quick accuracy check without Vivado |
| `sim/rtl_accuracy.py` | Generates test inputs; evaluates RTL outputs after simulation |
| `sim/luts/lut_exp.mem` | 1D exp LUT, 256×uint8 |
| `sim/luts/lut_2d_flat.mem` | 2D division LUT, 256×uint8 |
| `sim/rtl_inputs.txt` | Test vectors for Vivado testbench (generated) |
| `sim/rtl_outputs.txt` | RTL results from Vivado testbench (generated) |
| `sim/rtl_categories.txt` | Input category labels (generated alongside rtl_inputs) |
| `tb/softmax_unit_tb.sv` | Functional testbench: 3 directed tests |
| `tb/softmax_accuracy_tb.sv` | Accuracy testbench: 500 random rows |
