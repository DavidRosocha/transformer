// =============================================================================
// softmax_unit.sv  v6
// Key change: sigma uses explicit sigma_clear flag instead of
// next_state comparison to avoid combinational feedback issues in xsim
// =============================================================================
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

// ── LUT memories ─────────────────────────────────────────────────────────────
    logic [OUT_WIDTH-1:0] lut_exp     [0:LUT_EX_DEPTH-1];
    logic [OUT_WIDTH-1:0] lut_2d_flat [0:LUT_2D_FLAT-1];

    initial begin
        $readmemh("C:/Users/IsaiahK/Documents/School/FPGA/transformer/softmax/sim/luts/lut_exp.mem",     lut_exp);
        $readmemh("C:/Users/IsaiahK/Documents/School/FPGA/transformer/softmax/sim/luts/lut_2d_flat.mem", lut_2d_flat);
    end

// ── All signal declarations ───────────────────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] row_buf     [0:SEQ_LEN-1];
    logic        [OUT_WIDTH-1:0]  ex_buf      [0:SEQ_LEN-1];
    logic [3:0]                   cnt;
    logic signed [DATA_WIDTH-1:0] running_max;
    logic [11:0]                  sigma;
    logic signed [DATA_WIDTH-1:0] shifted_val;
    logic        [DATA_WIDTH-1:0] neg_shifted;
    logic        [OUT_WIDTH-1:0]  lut_addr_exp;
    logic        [SIGMA_BITS-1:0] sigma_idx;
    logic        [EX_BITS-1:0]    ex_idx;
    logic        [7:0]            flat_idx;
    logic                         sigma_clear; // pulses high for one cycle to clear sigma

// ── State machine ─────────────────────────────────────────────────────────────
    typedef enum logic [2:0] {
        S_IDLE, S_LOAD, S_SUBTRACT, S_ACCUMULATE, S_OUTPUT
    } state_t;
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            S_IDLE:        if (in_valid && in_first)          next_state = S_LOAD;
            S_LOAD:        if (in_valid && cnt == SEQ_LEN-2)  next_state = S_SUBTRACT;
            S_SUBTRACT:    if (cnt == SEQ_LEN-1)              next_state = S_ACCUMULATE;
            S_ACCUMULATE:  if (cnt == SEQ_LEN-1)              next_state = S_OUTPUT;
            S_OUTPUT:      if (cnt == SEQ_LEN-1)              next_state = S_IDLE;
            default:                                           next_state = S_IDLE;
        endcase
    end

// ── Counter — resets to 0 on every state transition ──────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
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

// ── sigma_clear: registered flag, goes high one cycle after entering SUBTRACT
//    This gives SUBTRACT one cycle to settle before ACCUMULATE begins,
//    and clears sigma cleanly without depending on next_state combinational logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sigma_clear <= 1'b0;
        end else begin
            // Goes high on FIRST cycle of ACCUMULATE (cnt==0)
            sigma_clear <= (state == S_SUBTRACT && cnt == SEQ_LEN-1);
        end
    end

// ── Stage 1: buffer values, track max ────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
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

// ── Stage 2+3: subtract max, exp LUT lookup ──────────────────────────────────
    always_comb begin
        shifted_val  = row_buf[cnt] - running_max;
        neg_shifted  = -shifted_val;
        lut_addr_exp = (neg_shifted[DATA_WIDTH-1:8] >= 8'd1) ? 8'hFF : neg_shifted[7:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ex_buf initialised during S_SUBTRACT
        end else if (state == S_SUBTRACT) begin
            ex_buf[cnt] <= lut_exp[lut_addr_exp];
        end
    end

// ── Stage 4: accumulate sigma ─────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sigma <= '0;
        end else if (sigma_clear) begin
            sigma <= '0;
        end else if (state == S_ACCUMULATE) begin
            sigma <= sigma + {4'b0, ex_buf[cnt]};
        end
    end

// ── Stage 5: 2D LUT output ───────────────────────────────────────────────────
    always_comb begin
        sigma_idx = sigma[11:8];
        ex_idx    = ex_buf[cnt][7:4];
        flat_idx  = {sigma_idx, ex_idx};
    end

    always_ff @(posedge clk or negedge rst_n) begin
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

// ── Busy ──────────────────────────────────────────────────────────────────────
    assign busy = (state != S_IDLE);

// ── Debug display ─────────────────────────────────────────────────────────────
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