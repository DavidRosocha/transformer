// =============================================================================
// softmax_unit.sv
// =============================================================================
// Pipelined softmax approximation for the attention accelerator.
//
// Algorithm: Vasyltsov & Chang arXiv:2111.10770, 2D LUT method (Section 4.2)
//
// Processing one row at a time (one softmax over SEQ_LEN values per call).
// For attention, caller feeds all 256 rows one after another.
//
// Pipeline stages:
//   Stage 1 — find max(x)          [SEQ_LEN cycles to scan]
//   Stage 2 — subtract max, clamp  [1 cycle per value]
//   Stage 3 — LUT exp lookup        [1 cycle per value]
//   Stage 4 — accumulate Σe^x      [SEQ_LEN cycles]
//   Stage 5 — 2D LUT output         [1 cycle per value]
//
// Input format:  Q8.8 signed fixed-point  (16-bit, signed)
// Output format: uint8 softmax values     (8-bit, unsigned, sum ≈ 255)
//
// Resource estimate (Artix-7 / Basys 3):
//   LUTs  : ~300  (comparator tree, subtractors, adder)
//   FFs   : ~150
//   BRAM  : 0     (LUTs fit in LUTRAM — 512 bytes total)
//
// =============================================================================

`timescale 1ns / 1ps

package softmax_pkg;
    parameter int SEQ_LEN    = 16;    // 16 row tokens (row-level tokenization)
    parameter int DATA_WIDTH = 16;    // Q8.8 signed input
    parameter int OUT_WIDTH  = 8;     // uint8 output
    parameter int EX_BITS    = 4;     // MSB bits of e^xi used as LUT column index
    parameter int SIGMA_BITS = 4;     // MSB bits of sigma used as LUT row index
    parameter int LUT_EX_DEPTH   = 256;           // 1D exp LUT entries
    parameter int LUT_2D_ROWS    = 2**SIGMA_BITS; // 16
    parameter int LUT_2D_COLS    = 2**EX_BITS;    // 16
endpackage

// =============================================================================
// Top-level module
// =============================================================================
module softmax_unit
    import softmax_pkg::*;
(
    input  logic                        clk,
    input  logic                        rst_n,

    // --- Input handshake ---
    // Assert valid + first together on the first value of a new row.
    // Keep valid high and cycle through all SEQ_LEN values consecutively.
    input  logic                        in_valid,
    input  logic                        in_first,   // high only on value [0]
    input  logic signed [DATA_WIDTH-1:0] in_data,   // Q8.8 score value

    // --- Output handshake ---
    // out_valid goes high SEQ_LEN cycles after the last input value.
    // out_data is valid for SEQ_LEN consecutive cycles.
    output logic                        out_valid,
    output logic [OUT_WIDTH-1:0]        out_data,   // uint8 softmax output

    // --- Status ---
    output logic                        busy        // high while processing a row
);

// =============================================================================
// LUT memories (loaded from .mem files at simulation / synthesis)
// =============================================================================

    logic [OUT_WIDTH-1:0] lut_exp  [0:LUT_EX_DEPTH-1];
    logic [OUT_WIDTH-1:0] lut_2d   [0:LUT_2D_ROWS-1][0:LUT_2D_COLS-1];

    initial begin
        $readmemh("sim/luts/lut_exp.mem",     lut_exp);
        $readmemh("sim/luts/lut_2d_msb.mem",  lut_2d);
    end

// =============================================================================
// Internal storage — one full row buffer
// =============================================================================

    // Buffer to hold incoming Q8.8 values for two-pass processing
    // (pass 1: find max; pass 2: compute softmax)
    logic signed [DATA_WIDTH-1:0] row_buf [0:SEQ_LEN-1];

    // e^xi results (uint8) for each position
    logic [OUT_WIDTH-1:0] ex_buf [0:SEQ_LEN-1];

// =============================================================================
// State machine
// =============================================================================

    typedef enum logic [2:0] {
        S_IDLE,        // waiting for in_first
        S_LOAD,        // receiving SEQ_LEN input values, finding max
        S_SUBTRACT,    // subtract max, look up e^xi for each value
        S_ACCUMULATE,  // sum all e^xi values → sigma
        S_OUTPUT       // read 2D LUT and stream out results
    } state_t;

    state_t state, next_state;

// =============================================================================
// Counters
// =============================================================================

    logic [$clog2(SEQ_LEN)-1:0] cnt;   // general purpose counter

// =============================================================================
// Stage 1 registers — running max during LOAD
// =============================================================================

    logic signed [DATA_WIDTH-1:0] running_max;

// =============================================================================
// Stage 4 register — accumulated sigma
// =============================================================================

    // Sigma: each e^xi is uint8 (0..255), SEQ_LEN=16 values
    // Maximum sum = 255 * 16 = 4080 -> needs 12 bits
    logic [11:0] sigma;

// =============================================================================
// Stage 1: LOAD — receive values and track running max
// =============================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running_max <= '1;   // most negative signed value
            running_max[DATA_WIDTH-1] <= 1'b1;
        end else if (state == S_IDLE && in_valid && in_first) begin
            // Reset max on first value of new row
            running_max <= in_data;
            row_buf[0]  <= in_data;
        end else if (state == S_LOAD && in_valid) begin
            row_buf[cnt] <= in_data;
            if (in_data > running_max)
                running_max <= in_data;
        end
    end

// =============================================================================
// Stage 2+3: SUBTRACT — x - max(x), then exp LUT lookup
// =============================================================================

    // Subtract and clamp: result is (x - max) which is <= 0
    // We need to convert to a LUT address:
    //   - result is in Q8.8, in range (-max_neg, 0]
    //   - we only care about values in (-1.0, 0] for e^x precision
    //   - address = round(-(x - max) * 255), clamped to [0, 255]
    //   - address=0  → x=max      → e^x = 1.0  → lut_exp[0] = ff
    //   - address=255 → x=max-1.0 → e^x ≈ 0.37 → lut_exp[255] ≈ 5e

    logic signed [DATA_WIDTH-1:0] shifted_val;
    logic        [DATA_WIDTH-1:0] neg_shifted;   // -(x - max), always >= 0
    logic        [OUT_WIDTH-1:0]  lut_addr_exp;  // 8-bit address into lut_exp
    logic        [OUT_WIDTH-1:0]  ex_val;         // lut_exp lookup result

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // nothing to reset here, combinational pipeline
        end else if (state == S_SUBTRACT) begin
            // Compute e^xi for each buffered value and store
            ex_buf[cnt] <= lut_exp[lut_addr_exp];
        end
    end

    // Combinational: address computation for current cnt position
    always_comb begin
        shifted_val  = row_buf[cnt] - running_max;   // Q8.8, <= 0
        // Negate and take integer bits + upper fractional bits as address
        // Q8.8: bit[15:8] = integer, bit[7:0] = fraction
        // We use the upper 8 bits of the negated value as address
        // This maps (-1.0, 0] to address [0, 255]
        neg_shifted  = (-shifted_val);
        // Top 8 bits of Q8.8 negated value = integer part
        // Clamp to 255 for values more negative than -1.0
        lut_addr_exp = (neg_shifted[DATA_WIDTH-1:8] >= 8'd1) 
                       ? 8'hFF 
                       : neg_shifted[7:0];  // use fractional byte as fine address
        ex_val       = lut_exp[lut_addr_exp];
    end

// =============================================================================
// Stage 4: ACCUMULATE — sum all e^xi values
// =============================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sigma <= '0;
        end else if (state == S_SUBTRACT && cnt == '0) begin
            sigma <= {9'b0, ex_buf[0]};   // reset and load first value
        end else if (state == S_ACCUMULATE) begin
            sigma <= sigma + {9'b0, ex_buf[cnt]};
        end
    end

// =============================================================================
// Stage 5: OUTPUT — 2D LUT lookup and stream out
// =============================================================================

    # Sigma index: take top SIGMA_BITS of sigma
    # sigma max = 255 * 16 = 4080 -> 12 bits
    # Top 4 bits = sigma[11:8]
    logic [SIGMA_BITS-1:0] sigma_idx;
    logic [EX_BITS-1:0]    ex_idx;

    always_comb begin
        // Top 4 bits of sigma (12-bit value for SEQ_LEN=16)
        sigma_idx = sigma[11:8];
        // Top 4 bits of ex_buf for current output position
        ex_idx    = ex_buf[cnt][7:4];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= '0;
        end else if (state == S_OUTPUT) begin
            out_valid <= 1'b1;
            out_data  <= lut_2d[sigma_idx][ex_idx];
        end else begin
            out_valid <= 1'b0;
        end
    end

// =============================================================================
// Counter logic
// =============================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= '0;
        end else begin
            case (state)
                S_IDLE:       cnt <= '0;
                S_LOAD:       if (in_valid) cnt <= cnt + 1;
                S_SUBTRACT:   cnt <= cnt + 1;
                S_ACCUMULATE: cnt <= cnt + 1;
                S_OUTPUT:     cnt <= cnt + 1;
                default:      cnt <= '0;
            endcase
        end
    end

// =============================================================================
// State machine transitions
// =============================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            S_IDLE:
                if (in_valid && in_first)
                    next_state = S_LOAD;

            S_LOAD:
                // Wait until we have received all SEQ_LEN values
                if (in_valid && cnt == SEQ_LEN - 1)
                    next_state = S_SUBTRACT;

            S_SUBTRACT:
                // One cycle per value to compute e^xi
                if (cnt == SEQ_LEN - 1)
                    next_state = S_ACCUMULATE;

            S_ACCUMULATE:
                // One cycle per value to sum
                if (cnt == SEQ_LEN - 1)
                    next_state = S_OUTPUT;

            S_OUTPUT:
                // Stream out SEQ_LEN results then return to idle
                if (cnt == SEQ_LEN - 1)
                    next_state = S_IDLE;

            default:
                next_state = S_IDLE;
        endcase
    end

// =============================================================================
// Busy signal
// =============================================================================

    assign busy = (state != S_IDLE);

// =============================================================================
// Simulation only: state name display
// =============================================================================

`ifndef SYNTHESIS
    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (state != next_state) begin
            case (next_state)
                S_IDLE:       $display("[softmax] t=%0t  → IDLE",       $time);
                S_LOAD:       $display("[softmax] t=%0t  → LOAD",       $time);
                S_SUBTRACT:   $display("[softmax] t=%0t  → SUBTRACT",   $time);
                S_ACCUMULATE: $display("[softmax] t=%0t  → ACCUMULATE", $time);
                S_OUTPUT:     $display("[softmax] t=%0t  → OUTPUT",     $time);
            endcase
        end
    end
    // synthesis translate_on
`endif

endmodule