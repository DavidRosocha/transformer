`timescale 1ns / 1ps

// =============================================================================
// softmax_pkg — shared parameters
//
// To change LUT size or sequence length, update these parameters AND
// regenerate the .mem files with lut_gen.py using matching arguments.
// =============================================================================
package softmax_pkg;
    parameter int SEQ_LEN      = 16;   // number of tokens per row
    parameter int DATA_WIDTH   = 16;   // input width: signed Q8.8 fixed-point
    parameter int OUT_WIDTH    = 8;    // output width: uint8, range 0..255
    parameter int EX_BITS      = 4;    // log2 of 2D LUT columns (exp axis)
    parameter int SIGMA_BITS   = 4;    // log2 of 2D LUT rows (sigma axis)
    parameter int LUT_EX_DEPTH = 256;  // entries in lut_exp
    parameter int LUT_2D_ROWS  = 2**SIGMA_BITS;
    parameter int LUT_2D_COLS  = 2**EX_BITS;
    parameter int LUT_2D_FLAT  = LUT_2D_ROWS * LUT_2D_COLS;
endpackage

// =============================================================================
// softmax_unit — 2D-LUT softmax approximation
//
// Implements the 2D LUT method from:
//   Vasyltsov & Chang, "Efficient Softmax Approximation for Deep Neural
//   Networks with Attention Mechanism", arXiv:2111.10770, 2021.
//
// No divider or multiplier required. Two small LUTs and an adder.
//
// ── Input protocol ───────────────────────────────────────────────────────────
//   Assert in_first=1 with the first token of a new row.
//   Assert in_valid=1 for each token on successive clock cycles (no gaps).
//   After SEQ_LEN tokens the unit processes internally and streams
//   SEQ_LEN output values, one per cycle, on out_valid/out_data.
//   busy goes high on the first token and returns low after the last output.
//
// ── Algorithm ────────────────────────────────────────────────────────────────
//   S_IDLE -> S_LOAD:
//     Capture all SEQ_LEN tokens into row_buf. Track running_max in parallel.
//
//   S_LOAD -> S_SUBTRACT:
//     For each token i:
//       diff = running_max - row_buf[i]          (always >= 0, Q8.8)
//       addr = CLAMP(diff >> 2, 0, 255)          (1/64 float per step)
//       ex_buf[i] = lut_exp[addr]                (~= exp(-(diff/256)) * 255)
//     One-cycle pipeline: addr registered, ex_buf written on the next cycle.
//
//   S_SUBTRACT -> S_ACCUMULATE:
//     sigma = sum of all ex_buf values           (12-bit, max = 255*16 = 4080)
//
//   S_ACCUMULATE -> S_OUTPUT:
//     For each token i:
//       sigma_idx = sigma[11:8]                  (top 4 bits -> row in 2D LUT)
//       ex_idx    = ex_buf[i][7:4]               (top 4 bits -> col in 2D LUT)
//       out[i]    = lut_2d_flat[{sigma_idx, ex_idx}]  (~= ex_buf[i]/sigma * 255)
//
// ── LUT address scaling ───────────────────────────────────────────────────────
//   The Q8.8 diff is right-shifted by 2 before indexing lut_exp.
//   Each addr step therefore represents 4 Q8.8 units = 1/64 in float.
//   The full 256-entry LUT covers differences of 0.0 to 3.98 below the max,
//   which spans the practical range of attention scores.
//   lut_exp is generated as: floor(exp(-addr / 64) * 255)
//
// ── Accuracy (16x16 LUT, SEQ_LEN=16) ────────────────────────────────────────
//   Mean absolute error: ~2.2 / 255 (0.87%)
//   Argmax accuracy: ~96% overall (100% for peaked/extreme distributions)
// =============================================================================
module softmax_unit
    import softmax_pkg::*;
(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         in_valid,
    input  logic                         in_first,
    input  logic signed [DATA_WIDTH-1:0] in_data,
    output logic                         out_valid,
    output logic [OUT_WIDTH-1:0]         out_data,
    output logic                         busy
);

    // ── LUT memories ─────────────────────────────────────────────────────────
    // Update these paths to match your local directory, or add the luts/
    // folder to Vivado's simulation include directories so bare filenames work.
    logic [OUT_WIDTH-1:0] lut_exp     [0:LUT_EX_DEPTH-1];
    logic [OUT_WIDTH-1:0] lut_2d_flat [0:LUT_2D_FLAT-1];

    initial begin
        $readmemh("C:/Users/IsaiahK/Documents/School/FPGA/transformer/softmax/sim/luts/lut_exp.mem",     lut_exp);
        $readmemh("C:/Users/IsaiahK/Documents/School/FPGA/transformer/softmax/sim/luts/lut_2d_flat.mem", lut_2d_flat);
        if (lut_exp[0] === 8'hFF)
            $display("[softmax] LUTs loaded OK (lut_exp[0]=0x%02X)", lut_exp[0]);
        else
            $display("[softmax] WARNING: LUT load may have failed (lut_exp[0]=0x%02X) -- check $readmemh path", lut_exp[0]);
    end

    // ── Internal buffers ─────────────────────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] row_buf [0:SEQ_LEN-1];  // captured input tokens
    logic        [OUT_WIDTH-1:0]  ex_buf  [0:SEQ_LEN-1];  // lut_exp results

    logic [3:0]                   cnt;          // position counter within each state
    logic signed [DATA_WIDTH-1:0] running_max;  // max token value seen so far
    logic [11:0]                  sigma;        // sum of ex_buf (max = 255*16 = 4080)

    // Combinational subtract signals
    logic [15:0]          diff_full;
    logic [OUT_WIDTH-1:0] lut_addr_exp;

    // Output LUT index signals
    logic [SIGMA_BITS-1:0] sigma_idx;
    logic [EX_BITS-1:0]    ex_idx;
    logic [7:0]            flat_idx;

    // One-cycle pipeline registers for the subtract stage
    logic [3:0]           cnt_d;
    logic [OUT_WIDTH-1:0] addr_d;
    logic                 subtract_d;

    // ── State machine ─────────────────────────────────────────────────────────
    typedef enum logic [2:0] {
        S_IDLE, S_LOAD, S_SUBTRACT, S_ACCUMULATE, S_OUTPUT
    } state_t;
    state_t state, next_state;

    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    // S_LOAD transition: token 0 is captured in S_IDLE, tokens 1..SEQ_LEN-1
    // are captured in S_LOAD as row_buf[cnt+1] with cnt running 0..SEQ_LEN-2.
    // The transition fires when cnt==SEQ_LEN-2 (cnt=14 writes row_buf[15]),
    // ensuring all 16 tokens are captured before moving to S_SUBTRACT.
    always_comb begin
        next_state = state;
        case (state)
            S_IDLE:       if (in_valid && in_first)          next_state = S_LOAD;
            S_LOAD:       if (in_valid && cnt == SEQ_LEN-2)  next_state = S_SUBTRACT;
            S_SUBTRACT:   if (cnt == SEQ_LEN-1)              next_state = S_ACCUMULATE;
            S_ACCUMULATE: if (cnt == SEQ_LEN-1)              next_state = S_OUTPUT;
            S_OUTPUT:     if (cnt == SEQ_LEN-1)              next_state = S_IDLE;
            default:                                          next_state = S_IDLE;
        endcase
    end

    // ── Counter ──────────────────────────────────────────────────────────────
    // Resets to 0 on every state transition, then counts up within each state.
    // In S_LOAD it only advances when in_valid is asserted (backpressure safe).
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cnt <= '0;
        end else if (state != next_state) begin
            cnt <= '0;
        end else begin
            case (state)
                S_LOAD:       if (in_valid) cnt <= cnt + 1;
                S_SUBTRACT:   cnt <= cnt + 1;
                S_ACCUMULATE: cnt <= cnt + 1;
                S_OUTPUT:     cnt <= cnt + 1;
                default:      cnt <= '0;
            endcase
        end
    end

    // ── Data capture and running max ──────────────────────────────────────────
    // Token 0 is stored in S_IDLE on the cycle in_first is asserted.
    // Tokens 1..SEQ_LEN-1 are stored in S_LOAD via row_buf[cnt+1].
    // running_max is updated every cycle a new token arrives.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running_max <= 16'h8000;  // most-negative signed Q8.8 value
        end else begin
            if (state == S_IDLE && in_valid && in_first) begin
                row_buf[0]  <= in_data;
                running_max <= in_data;
            end
            if (state == S_LOAD && in_valid) begin
                row_buf[cnt + 1] <= in_data;
                if (in_data > running_max)
                    running_max <= in_data;
            end
        end
    end

    // ── Combinational subtract and LUT address ────────────────────────────────
    // diff_full is the Q8.8 distance of the current token below the running max.
    // Right-shifting by 2 maps each addr step to 1/64 in float, giving the
    // 256-entry LUT a range of 0.0 to 3.98 -- enough for all practical
    // attention score distributions. See lut_gen.py for the matching formula.
    always_comb begin
        diff_full    = running_max - row_buf[cnt];
        lut_addr_exp = (diff_full[15:2] > 16'd255) ? 8'hFF : diff_full[9:2];
    end

    // ── Pipeline registers for subtract stage ─────────────────────────────────
    // The LUT read takes one cycle, so cnt and addr are registered here and
    // ex_buf is written on the following cycle using the delayed values.
    always_ff @(posedge clk) begin
        cnt_d      <= cnt;
        addr_d     <= lut_addr_exp;
        subtract_d <= (state == S_SUBTRACT);
    end

    // ── Write ex_buf ──────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (subtract_d)
            ex_buf[cnt_d] <= lut_exp[addr_d];
    end

    // ── Sigma accumulator ─────────────────────────────────────────────────────
    // Sums all ex_buf values. The subtract pipeline ensures ex_buf is fully
    // written before S_ACCUMULATE begins.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sigma <= '0;
        end else if (state == S_ACCUMULATE) begin
            if (cnt == 4'd0)
                sigma <= {4'b0, ex_buf[0]};
            else
                sigma <= sigma + {4'b0, ex_buf[cnt]};
        end
    end

    // ── Output LUT indices ────────────────────────────────────────────────────
    // sigma_idx: top 4 bits of sigma  -> selects the row in lut_2d_flat
    // ex_idx:    top 4 bits of ex_buf -> selects the column
    // flat_idx:  bit concatenation    -> free in hardware (just wiring)
    always_comb begin
        sigma_idx = sigma[11:8];
        ex_idx    = ex_buf[cnt][7:4];
        flat_idx  = {sigma_idx, ex_idx};
    end

    // ── Output stage ──────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= '0;
        end else if (state == S_OUTPUT) begin
            out_valid <= 1'b1;
            out_data  <= lut_2d_flat[flat_idx];
        end else begin
            out_valid <= 1'b0;
        end
    end

    assign busy = (state != S_IDLE);

    // ── State transition debug display (simulation only) ─────────────────────
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (state != next_state) begin
            case (next_state)
                S_IDLE:       $display("[softmax] t=%0t -> IDLE",       $time);
                S_LOAD:       $display("[softmax] t=%0t -> LOAD",       $time);
                S_SUBTRACT:   $display("[softmax] t=%0t -> SUBTRACT",   $time);
                S_ACCUMULATE: $display("[softmax] t=%0t -> ACCUMULATE", $time);
                S_OUTPUT:     $display("[softmax] t=%0t -> OUTPUT",     $time);
            endcase
        end
    end
`endif

endmodule