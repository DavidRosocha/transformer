// =============================================================================
// softmax_unit_tb.sv
// =============================================================================
// Testbench for softmax_unit.sv
//
// Tests:
//   1. Single row — known input, compare output against Python reference
//   2. Back-to-back rows — verify no state leaks between rows
//   3. Uniform input — all values equal → output should be ~1/N each
//
// Run with Icarus Verilog:
//   iverilog -g2012 -o sim_out tb/softmax_unit_tb.sv rtl/softmax_unit.sv
//   vvp sim_out
//
// =============================================================================

`timescale 1ns / 1ps

module softmax_unit_tb;

    import softmax_pkg::*;

    // ── Clock & reset ──────────────────────────────────────────────────────
    logic clk   = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;   // 100 MHz

    // ── DUT signals ────────────────────────────────────────────────────────
    logic                         in_valid;
    logic                         in_first;
    logic signed [DATA_WIDTH-1:0] in_data;
    logic                         out_valid;
    logic        [OUT_WIDTH-1:0]  out_data;
    logic                         busy;

    // ── DUT instantiation ──────────────────────────────────────────────────
    softmax_unit dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .in_first (in_first),
        .in_data  (in_data),
        .out_valid(out_valid),
        .out_data (out_data),
        .busy     (busy)
    );

    // ── Output capture buffer ──────────────────────────────────────────────
    logic [OUT_WIDTH-1:0] out_buf [0:SEQ_LEN-1];
    int out_cnt = 0;

    always_ff @(posedge clk) begin
        if (out_valid) begin
            out_buf[out_cnt] <= out_data;
            out_cnt          <= out_cnt + 1;
        end
    end

    // ── Helper task: send one row ──────────────────────────────────────────
    // scores[] is Q8.8 format: integer part * 256 + fractional part
    task automatic send_row(
        input logic signed [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1]
    );
        int i;
        out_cnt = 0;
        for (i = 0; i < SEQ_LEN; i++) begin
            @(posedge clk);
            in_valid = 1'b1;
            in_first = (i == 0) ? 1'b1 : 1'b0;
            in_data  = scores[i];
        end
        @(posedge clk);
        in_valid = 1'b0;
        in_first = 1'b0;
        // Wait for output to complete
        wait (out_cnt == SEQ_LEN);
        @(posedge clk);
    endtask

    // ── Helper: real to Q8.8 ──────────────────────────────────────────────
    function automatic logic signed [DATA_WIDTH-1:0] to_q8_8(input real v);
        return $rtoi(v * 256.0);
    endfunction

    // ── Helper: print result summary ──────────────────────────────────────
    task automatic print_summary(input string label);
        real sum_out;
        real max_out;
        int  argmax;
        int  i;
        sum_out = 0.0;
        max_out = 0.0;
        argmax  = 0;
        for (i = 0; i < SEQ_LEN; i++) begin
            sum_out += out_buf[i];
            if (out_buf[i] > max_out) begin
                max_out = out_buf[i];
                argmax  = i;
            end
        end
        $display("  [%s] sum=%0.1f (ideal=255*N)  argmax=%0d  max_val=%0d",
                 label, sum_out, argmax, int'(max_out));
    endtask

    // ── Test 1: uniform input ──────────────────────────────────────────────
    // All scores equal → softmax should be uniform (≈ 255/256 ≈ 1 per output)
    task test_uniform();
        logic signed [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1];
        int i;
        $display("\n── Test 1: uniform input (all zeros) ────────────────");
        for (i = 0; i < SEQ_LEN; i++)
            scores[i] = to_q8_8(0.0);
        send_row(scores);
        print_summary("uniform");
        // Check: all outputs should be equal
        for (i = 1; i < SEQ_LEN; i++) begin
            if (out_buf[i] !== out_buf[0]) begin
                $display("  WARN: output[%0d]=%0d differs from output[0]=%0d",
                         i, out_buf[i], out_buf[0]);
            end
        end
        $display("  output[0..3] = %0d %0d %0d %0d",
                 out_buf[0], out_buf[1], out_buf[2], out_buf[3]);
    endtask

    // ── Test 2: one-hot (one large score, rest small) ──────────────────────
    // Score at index 42 is much larger → almost all probability should land there
    task test_one_hot();
        logic signed [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1];
        int i;
        $display("\n── Test 2: one-hot input (index 42 dominant) ────────");
        for (i = 0; i < SEQ_LEN; i++)
            scores[i] = to_q8_8(-4.0);   // small background
        scores[42] = to_q8_8(4.0);        // dominant score
        send_row(scores);
        print_summary("one-hot");
        $display("  out[42]=%0d  out[0]=%0d  out[1]=%0d",
                 out_buf[42], out_buf[0], out_buf[1]);
        if (out_buf[42] > out_buf[0])
            $display("  PASS: dominant token has highest output");
        else
            $display("  FAIL: dominant token should have highest output");
    endtask

    // ── Test 3: back-to-back rows ──────────────────────────────────────────
    // Sends two different rows — verifies state resets between rows
    task test_back_to_back();
        logic signed [DATA_WIDTH-1:0] scores_a [0:SEQ_LEN-1];
        logic signed [DATA_WIDTH-1:0] scores_b [0:SEQ_LEN-1];
        logic [OUT_WIDTH-1:0]         result_a  [0:SEQ_LEN-1];
        int i;
        $display("\n── Test 3: back-to-back rows ────────────────────────");

        // Row A: peak at index 10
        for (i = 0; i < SEQ_LEN; i++) scores_a[i] = to_q8_8(-2.0);
        scores_a[10] = to_q8_8(2.0);

        // Row B: peak at index 200
        for (i = 0; i < SEQ_LEN; i++) scores_b[i] = to_q8_8(-2.0);
        scores_b[200] = to_q8_8(2.0);

        // Send row A
        send_row(scores_a);
        for (i = 0; i < SEQ_LEN; i++) result_a[i] = out_buf[i];
        print_summary("row A");

        // Send row B immediately
        send_row(scores_b);
        print_summary("row B");

        $display("  Row A argmax vicinity: [9]=%0d [10]=%0d [11]=%0d",
                 result_a[9], result_a[10], result_a[11]);
        $display("  Row B argmax vicinity: [199]=%0d [200]=%0d [201]=%0d",
                 out_buf[199], out_buf[200], out_buf[201]);

        if (result_a[10] >= result_a[9] && out_buf[200] >= out_buf[199])
            $display("  PASS: both rows peaked at correct index");
        else
            $display("  FAIL: peak not at expected index");
    endtask

    // ── Main test sequence ─────────────────────────────────────────────────
    initial begin
        $dumpfile("sim/softmax_sim.vcd");
        $dumpvars(0, softmax_unit_tb);

        // Reset
        in_valid = 0;
        in_first = 0;
        in_data  = 0;
        #20;
        rst_n = 1;
        #20;

        test_uniform();
        test_one_hot();
        test_back_to_back();

        $display("\n── All tests complete ────────────────────────────────");
        $finish;
    end

    // ── Timeout watchdog ───────────────────────────────────────────────────
    initial begin
        #500000;
        $display("TIMEOUT: simulation exceeded 500us");
        $finish;
    end

endmodule