`timescale 1ns / 1ps

package softmax_pkg;
    parameter int SEQ_LEN      = 16;
    parameter int DATA_WIDTH   = 16;
    parameter int OUT_WIDTH    = 8;
    parameter int EX_BITS      = 4;
    parameter int SIGMA_BITS   = 4;
    parameter int LUT_EX_DEPTH = 256;
    parameter int LUT_2D_ROWS  = 2**SIGMA_BITS;
    parameter int LUT_2D_COLS  = 2**EX_BITS;
    parameter int LUT_2D_FLAT  = LUT_2D_ROWS * LUT_2D_COLS;
endpackage

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

    // ── LUT memories ────────────────────────────────────────────────────────
    logic [OUT_WIDTH-1:0] lut_exp     [0:LUT_EX_DEPTH-1];
    logic [OUT_WIDTH-1:0] lut_2d_flat [0:LUT_2D_FLAT-1];

    initial begin
        $readmemh("C:/Users/IsaiahK/Documents/School/FPGA/transformer/softmax/sim/luts/lut_exp.mem",     lut_exp);
        $readmemh("C:/Users/IsaiahK/Documents/School/FPGA/transformer/softmax/sim/luts/lut_2d_flat.mem", lut_2d_flat);
    end

    // ── Internal buffers ────────────────────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] row_buf [0:SEQ_LEN-1];
    logic        [OUT_WIDTH-1:0]  ex_buf  [0:SEQ_LEN-1];

    logic [3:0]                   cnt;
    logic signed [DATA_WIDTH-1:0] running_max;
    logic [11:0]                  sigma;

    // Combinational subtract signals
    //
    // FIX v6: replaced top-byte-only approach with full Q8.8 subtraction.
    //
    // OLD approach (broken):
    //   shifted_val  = row_buf[cnt] - running_max   (negative Q8.8)
    //   neg_shifted  = -shifted_val                  (positive Q8.8)
    //   lut_addr_exp = (neg_shifted[15:8] >= 1) ? 0xFF : neg_shifted[7:0]
    //
    //   Problem 1: when integer difference >= 1, clamps to 0xFF and ignores
    //   fractional part entirely -- tokens 1.1 and 1.9 below max both get 0xFF.
    //
    //   Problem 2: neg_shifted[7:0] is the FRACTIONAL part of the difference,
    //   not the full difference -- addr only spans 0..255 within the same
    //   integer band, losing all inter-integer discrimination.
    //
    // NEW approach (correct):
    //   diff = running_max - row_buf[cnt]   (positive Q8.8, since max >= val)
    //   lut_addr_exp = diff clamped to 0..255
    //
    //   Now addr is the full Q8.8 difference in steps of 1/256.
    //   addr=0   -> same as max           -> lut_exp[0]   = 255
    //   addr=128 -> 0.5 below max         -> lut_exp[128] = 154
    //   addr=255 -> ~1.0 below max        -> lut_exp[255] = 94
    //   addr=256+ -> >1.0 below, clamp    -> lut_exp[255] = 94
    //
    //   lut_exp.mem must be generated with: lut_exp[i] = floor(exp(-i/256)*255)
    //   (run lut_gen.py to regenerate if needed)
    //
    logic [15:0]         diff_full;
    logic [OUT_WIDTH-1:0] lut_addr_exp;

    // Output LUT index signals
    logic [SIGMA_BITS-1:0] sigma_idx;
    logic [EX_BITS-1:0]    ex_idx;
    logic [7:0]            flat_idx;

    // Pipeline registers
    logic [3:0]           cnt_d;
    logic [OUT_WIDTH-1:0] addr_d;
    logic                 subtract_d;

    // ── State machine ────────────────────────────────────────────────────────
    typedef enum logic [2:0] {
        S_IDLE, S_LOAD, S_SUBTRACT, S_ACCUMULATE, S_OUTPUT
    } state_t;
    state_t state, next_state;

    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            S_IDLE:       if (in_valid && in_first)         next_state = S_LOAD;
            S_LOAD:       if (in_valid && cnt == SEQ_LEN-2) next_state = S_SUBTRACT;
            S_SUBTRACT:   if (cnt == SEQ_LEN-1)             next_state = S_ACCUMULATE;
            S_ACCUMULATE: if (cnt == SEQ_LEN-1)             next_state = S_OUTPUT;
            S_OUTPUT:     if (cnt == SEQ_LEN-1)             next_state = S_IDLE;
            default:                                         next_state = S_IDLE;
        endcase
    end

    // ── Counter ──────────────────────────────────────────────────────────────
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

    // ── Data capture and running max ─────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running_max <= 16'h8000;
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

    // ── Combinational subtract and LUT address ───────────────────────────────
    always_comb begin
        diff_full    = running_max - row_buf[cnt];
        lut_addr_exp = (diff_full > 16'd255) ? 8'hFF : diff_full[7:0];
    end

    // ── Pipeline: delay cnt and addr by one cycle ────────────────────────────
    always_ff @(posedge clk) begin
        cnt_d      <= cnt;
        addr_d     <= lut_addr_exp;
        subtract_d <= (state == S_SUBTRACT);
    end

    // ── Write ex_buf one cycle after computing address ───────────────────────
    always_ff @(posedge clk) begin
        if (subtract_d) begin
            ex_buf[cnt_d] <= lut_exp[addr_d];
        end
    end

    // ── Sigma accumulator ────────────────────────────────────────────────────
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

    // ── Output LUT index ─────────────────────────────────────────────────────
    //
    // sigma_idx = sigma[11:8]
    //   sigma max = 255*16 = 4080, fits in 12 bits.
    //   sigma[11:8] = top 4 bits, spreading 0..4095 across 16 buckets of 256.
    //   Bucket 0: sigma 0..255, Bucket 6: sigma 1536..1791, Bucket 15: 3840..4095.
    //   Real inputs land in buckets 6..12 depending on how peaked the input is.
    //
    // ex_idx = ex_buf[cnt][7:4]
    //   Top 4 bits of the uint8 ex value, range 0..15.
    //
    // flat_idx = {sigma_idx, ex_idx}
    //   Bit concatenation = sigma_idx*16 + ex_idx. Free in hardware (just wiring).
    //
    always_comb begin
        sigma_idx = sigma[11:8];
        ex_idx    = ex_buf[cnt][7:4];
        flat_idx  = {sigma_idx, ex_idx};
    end

    // ── Output stage ─────────────────────────────────────────────────────────
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
