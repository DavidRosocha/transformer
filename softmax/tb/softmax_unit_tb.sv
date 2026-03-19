// =============================================================================
// softmax_unit_tb.sv
// =============================================================================
// Testbench for softmax_unit.sv
//
// Tests:
//   1. Uniform input  — all scores equal -> all outputs should be equal
//   2. One-hot input  — one dominant score -> that index gets highest output
//   3. Back-to-back   — two rows in sequence -> no state leaks between them
//
// SEQ_LEN = 16  (row-level tokenization, 16 tokens per attention row)
//
// Run with Icarus Verilog from transformer/ root:
//   iverilog -g2012 -o sim_out tb/softmax_unit_tb.sv rtl/softmax_unit.sv
//   vvp sim_out
//
// =============================================================================

`timescale 1ns / 1ps

module softmax_unit_tb;

    import softmax_pkg::*;

    // Clock & reset
    logic clk   = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;   // 100 MHz

    // DUT signals
    logic                         in_valid;
    logic                         in_first;
    logic signed [DATA_WIDTH-1:0] in_data;
    logic                         out_valid;
    logic        [OUT_WIDTH-1:0]  out_data;
    logic                         busy;

    // DUT instantiation
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

    // Output capture buffer — SEQ_LEN=16 entries
    logic [OUT_WIDTH-1:0] out_buf [0:SEQ_LEN-1];
    int out_cnt = 0;

    always_ff @(posedge clk) begin
        if (out_valid) begin
            out_buf[out_cnt] <= out_data;
            out_cnt          <= out_cnt + 1;
        end
    end

    // Helper: send one row of SEQ_LEN=16 Q8.8 scores
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
        wait (out_cnt == SEQ_LEN);
        @(posedge clk);
    endtask

    // Helper: convert real to Q8.8
    function automatic logic signed [DATA_WIDTH-1:0] to_q8_8(input real v);
        return $rtoi(v * 256.0);
    endfunction

    // Helper: print output summary
    task automatic print_summary(input string label);
        real sum_out;
        real max_out;
        int  argmax;
        int  i;
        sum_out = 0.0; max_out = 0.0; argmax = 0;
        for (i = 0; i < SEQ_LEN; i++) begin
            sum_out += real'(out_buf[i]);
            if (real'(out_buf[i]) > max_out) begin
                max_out = real'(out_buf[i]);
                argmax  = i;
            end
        end
        $display("  [%s] sum=%0.1f  argmax=%0d  max_val=%0d",
                 label, sum_out, argmax, int'(max_out));
    endtask

    // Helper: print all 16 values
    task automatic print_all(input string label);
        int i;
        $write("  [%s] ", label);
        for (i = 0; i < SEQ_LEN; i++)
            $write("[%0d]=%0d ", i, out_buf[i]);
        $display("");
    endtask

    // Test 1: uniform — all scores equal -> all outputs should be identical
    // With SEQ_LEN=16, uint8, each output should be ~255/16 = 15
    task test_uniform();
        logic signed [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1];
        int i; int pass;
        $display("\n-- Test 1: uniform input (all 0.0) ------------------");
        for (i = 0; i < SEQ_LEN; i++) scores[i] = to_q8_8(0.0);
        send_row(scores);
        print_summary("uniform");
        print_all("uniform");
        pass = 1;
        for (i = 1; i < SEQ_LEN; i++) begin
            if (out_buf[i] !== out_buf[0]) begin
                $display("  WARN: out[%0d]=%0d != out[0]=%0d", i, out_buf[i], out_buf[0]);
                pass = 0;
            end
        end
        if (pass) $display("  PASS: all outputs equal");
    endtask

    // Test 2: one-hot — token 5 dominates (valid index for SEQ_LEN=16)
    task test_one_hot();
        logic signed [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1];
        int i;
        $display("\n-- Test 2: one-hot (index 5 dominant) ---------------");
        for (i = 0; i < SEQ_LEN; i++) scores[i] = to_q8_8(-4.0);
        scores[5] = to_q8_8(4.0);
        send_row(scores);
        print_summary("one-hot");
        print_all("one-hot");
        $display("  out[4]=%0d  out[5]=%0d  out[6]=%0d",
                 out_buf[4], out_buf[5], out_buf[6]);
        if (out_buf[5] > out_buf[0])
            $display("  PASS: token 5 has highest output");
        else
            $display("  FAIL: token 5 should dominate");
    endtask

    // Test 3: back-to-back rows — row A peaks at 3, row B peaks at 12
    task test_back_to_back();
        logic signed [DATA_WIDTH-1:0] scores_a [0:SEQ_LEN-1];
        logic signed [DATA_WIDTH-1:0] scores_b [0:SEQ_LEN-1];
        logic [OUT_WIDTH-1:0]         result_a  [0:SEQ_LEN-1];
        int i;
        $display("\n-- Test 3: back-to-back rows -------------------------");

        for (i = 0; i < SEQ_LEN; i++) scores_a[i] = to_q8_8(-2.0);
        scores_a[3] = to_q8_8(2.0);

        for (i = 0; i < SEQ_LEN; i++) scores_b[i] = to_q8_8(-2.0);
        scores_b[12] = to_q8_8(2.0);

        send_row(scores_a);
        for (i = 0; i < SEQ_LEN; i++) result_a[i] = out_buf[i];
        print_summary("row A (peak@3)");

        send_row(scores_b);
        print_summary("row B (peak@12)");

        $display("  Row A: [2]=%0d [3]=%0d [4]=%0d",
                 result_a[2], result_a[3], result_a[4]);
        $display("  Row B: [11]=%0d [12]=%0d [13]=%0d",
                 out_buf[11], out_buf[12], out_buf[13]);

        if (result_a[3] >= result_a[2] && result_a[3] >= result_a[4] &&
            out_buf[12] >= out_buf[11] && out_buf[12] >= out_buf[13])
            $display("  PASS: both rows peaked at correct index");
        else
            $display("  FAIL: peak mismatch — possible state leak between rows");
    endtask

    // Main sequence
    initial begin
        $dumpfile("sim/softmax_sim.vcd");
        $dumpvars(0, softmax_unit_tb);
        in_valid = 0; in_first = 0; in_data = 0;
        #20; rst_n = 1; #20;

        test_uniform();
        test_one_hot();
        test_back_to_back();

        $display("\n-- All tests complete --------------------------------");
        $finish;
    end

    // Timeout watchdog — 500us at 100MHz
    initial begin
        #500000;
        $display("TIMEOUT: simulation exceeded 500us");
        $finish;
    end

endmodule