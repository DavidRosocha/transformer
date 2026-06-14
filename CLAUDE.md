# CLAUDE.md — FPGA Transformer TPU Project

## What This Project Is

A hardware transformer attention accelerator built from scratch on a **Basys 3 FPGA (Artix-7 XC7A35T)**. A PyTorch model on the PC classifies/completes 16×16 binary pixel art patterns. The attention layers are offloaded to the FPGA over UART. Everything else (embedding, FFN, LayerNorm, output projection) runs on the PC in Python.

This is a personal project (not a class assignment). Originally a 4-person team — David (ML lead, the user), Jeffrey (matrix multiply), Isaiah (softmax), Ahmad (UART/integration). Teammates stopped contributing, so David is finishing it solo.

## Board Constraints

- **FPGA:** Artix-7 XC7A35T
- **LUTs:** 20,800
- **DSP slices:** 90 (this is why the systolic array is 8×8, not larger)
- **Block RAM:** 225 KB (50 blocks)
- **Clock:** 100 MHz
- **Connectivity:** USB-UART only (no shared memory, no CPU on board)

## Model Architecture

- 16×16 binary grid → 16 row-tokens of 16 pixels each
- Embedding: linear 16→64 + positional encoding
- 2 attention layers, single-head, embed_dim=64, hidden_dim=128
- Output: linear 64→16 + sigmoid
- ~69,000 parameters → ~136 KB in Q8.8 fixed-point
- Trained to loss 0.1124, accuracy 95.8%
- All values are **Q8.8 signed fixed-point** (16-bit: 1 sign, 7 integer, 8 fractional)

## PC ↔ FPGA Split

**PC does:** embedding + positional encoding, sends 2 KB embedded tokens, receives 2 KB result, does residual + LayerNorm + FFN (64→128→64 + ReLU), repeats for layer 2, final projection + sigmoid.

**FPGA does (per layer):** Q=input×W_q, K=input×W_k, V=input×W_v, scores=Q×Kᵀ, softmax(scores), context=attn×V, output=context×W_out. All 7 matmuls + softmax in one shot.

**BRAM budget:** 8 weight matrices (64×64 each) = 64 KB + ~15 KB scratch = ~79 KB used of 225 KB (35%).

**UART:** 921,600 baud. Startup: 64 KB weight load (once). Per inference: 8 KB (send 2 KB, receive 2 KB, twice for 2 layers). Protocol: [0xAA][type][2048 bytes][XOR checksum][0x55], ACK=0x06, NAK=0x15.

The √64 scaling factor is baked into W_q before loading — the FPGA doesn't need to divide.

## Current Status — What's Done

| Module | File | Status |
|---|---|---|
| 8×8 systolic array + PE | `hardware/modules/systolic_array_8x8.sv` | ✅ Done |
| Tile controller (up to 64×64) | `hardware/modules/tile_controller.sv` | ✅ Done |
| Softmax (2D LUT, no divider) | `softmax/rtl/softmax_unit.sv` | ✅ Done, validated |
| Softmax LUTs + Python tools | `softmax/sim/` | ✅ Done |
| UART RX/TX | `hardware/modules/uart_rx.v`, `uart_tx.v` | ✅ Done |
| Top-level protocol FSM | `hardware/top/top.sv` | ⚠️ Partial — receives frames, but nothing connected |
| Python UART driver | `software/driver/fpga_driver.py` | ✅ Done |
| PyTorch model | `software/model/pixel_completer.py` | ✅ Done |
| Dataset generator | `software/data/create_dataset.py` | ✅ Done |

## What Needs to Be Built

1. **`attention_core.sv`** — The FSM that chains: matmul(W_q) → matmul(W_k) → matmul(W_v) → matmul(Q×Kᵀ) → softmax (16 rows) → matmul(attn×V) → matmul(×W_out). Reuses the tile_controller for each matmul. This is the critical missing piece.

2. **Weight BRAM** — 64 KB storage for 8 weight matrices (64×64 × Q8.8). Currently top.sv only has 4 KB.

3. **Weight loading protocol** — Extend UART protocol so the PC can send all 8 weight matrices at startup before inference begins. Update fpga_driver.py with a `load_weights()` method.

4. **Top-level integration** — Rewrite top.sv to connect: UART RX → weight BRAM + input BRAM → attention_core → result BRAM → UART TX. Wire up `sending_result` trigger.

5. **PC pipeline script** — Orchestrator that does: embed → fpga_driver.run_inference() → LayerNorm + FFN → repeat for layer 2 → sigmoid → display.

6. **Testbenches** — For systolic array, tile controller, and attention core.

7. **Basys 3 .xdc constraints file** — Pin assignments for clock and UART.

## Known Issues / Improvements Backlog

- Tile controller ACCUMULATE state wastes 63 cycles (only does work on cycle 0)
- Softmax $readmemh paths are hardcoded to Isaiah's Windows machine — make relative
- top.sv BRAM is byte-wide (8-bit) — should be 16-bit for Q8.8 data
- UART protocol only supports 2048-byte payloads — weights are 8192 bytes each
- Result transmission path exists but `sending_result` is never set to 1
- fpga_driver.py needs `load_weights()` method

## Key Design Decisions Already Made

- 8×8 systolic array (64 PEs) — largest that fits in 90 DSPs
- Output-stationary dataflow — accumulator stays in PE, inputs flow through
- Tiling for larger matrices — tile_controller iterates 8×8 chunks
- Softmax uses 2D LUT method (Vasyltsov & Chang 2021) — no divider/multiplier, fits in LUTRAM
- Weights preloaded into BRAM at startup, not sent per-inference
- PC handles embedding, FFN, LayerNorm, sigmoid; FPGA handles attention only

## Code Conventions

- Hardware: SystemVerilog (.sv) for new modules, Verilog (.v) for UART (legacy)
- Vivado targeting Artix-7, simulation in xsim
- Python: numpy for matrix ops, pyserial for UART, PyTorch for model
- Fixed-point: Q8.8 everywhere (DATA_WIDTH=16, ACC_WIDTH=32)
