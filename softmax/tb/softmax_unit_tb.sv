`timescale 1ns / 1ps
// =============================================================================
// softmax_unit_tb.sv  —  Functional unit testbench
//
// Vivado setup:
//   Sources:  softmax_unit.sv  (Design Sources)
//             softmax_unit_tb.sv  (Simulation Sources)
//   Set softmax_unit_tb as the top simulation module.
//   LUT files must be in the simulation working directory
//   (Vivado: <project>/project.sim/sim_1/behav/xsim/)
//   OR update the $readmemh paths in softmax_unit.sv.
// =============================================================================

module softmax_unit_tb;

    import softmax_pkg::*;

    // ── Clock ────────────────────────────────────────────────────────────────
    logic clk;
    initial clk = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    // ── Reset — use reg for guaranteed procedural drive in xsim ─────────────
    reg rst_n;

    // ── DUT signals ──────────────────────────────────────────────────────────
    logic                         in_valid;
    logic                         in_first;
    logic signed [DATA_WIDTH-1:0] in_data;
    logic                         out_valid;
    logic        [OUT_WIDTH-1:0]  out_data;
    logic                         busy;

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

    // ── Output capture buffer ─────────────────────────────────────────────
    logic [OUT_WIDTH-1:0] out_buf [0:SEQ_LEN-1];
    int capture_idx;

    // Capture outputs as they stream out.
    // Using blocking assignment here is intentional: capture_idx must update
    // immediately within the same time step so we don't double-capture.
    always @(posedge clk) begin
        if (out_valid && capture_idx < SEQ_LEN) begin
            out_buf[capture_idx] = out_data;
            capture_idx          = capture_idx + 1;
        end
    end

    // =========================================================================
    // Tasks
    // =========================================================================

    // Send one row of SEQ_LEN tokens to the DUT.
    // Signals are driven AFTER the clock edge (#1) to avoid setup-time races
    // in xsim where driving at the edge can be sampled in the same cycle.
    task automatic send_row(
        input logic signed [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1]
    );
        capture_idx = 0;
        for (int i = 0; i < SEQ_LEN; i++) begin
            @(posedge clk); #1;
            in_valid = 1'b1;
            in_first = (i == 0) ? 1'b1 : 1'b0;
            in_data  = scores[i];
        end
        // De-assert valid one cycle after the last token
        @(posedge clk); #1;
        in_valid = 1'b0;
        in_first = 1'b0;

        // Wait for the DUT to finish processing
        wait (busy == 1'b0);

        // A few extra cycles to let any trailing out_valid pulses settle
        repeat(4) @(posedge clk);
    endtask

    // Q8.8 conversion helper
    function automatic logic signed [DATA_WIDTH-1:0] to_q8_8(input real v);
        return $rtoi(v * 256.0);
    endfunction

    // Print summary: sum, argmax, peak value
    task automatic print_summary(input string label);
        real sum_out, max_out;
        int  argmax;
        sum_out = 0.0; max_out = 0.0; argmax = 0;
        for (int i = 0; i < SEQ_LEN; i++) begin
            sum_out += real'(out_buf[i]);
            if (real'(out_buf[i]) > max_out) begin
                max_out = real'(out_buf[i]);
                argmax  = i;
            end
        end
        $display("  [%s] sum=%0.1f  argmax=%0d  peak=%0d",
                 label, sum_out, argmax, int'(max_out));
    endtask

    // Print all outputs
    task automatic print_all(input string label);
        $write("  [%s] ", label);
        for (int i = 0; i < SEQ_LEN; i++)
            $write("[%0d]=%0d ", i, out_buf[i]);
        $display("");
    endtask

    // =========================================================================
    // Test cases
    // =========================================================================

    task test_uniform();
        logic signed [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1];
        int pass;
        $display("\n-- Test 1: uniform (all 0.0) -------------------------");
        for (int i = 0; i < SEQ_LEN; i++) scores[i] = to_q8_8(0.0);
        send_row(scores);
        print_summary("uniform");
        print_all("uniform");
        pass = 1;
        for (int i = 1; i < SEQ_LEN; i++) begin
            if (out_buf[i] !== out_buf[0]) begin
                $display("  WARN: out[%0d]=%0d != out[0]=%0d", i, out_buf[i], out_buf[0]);
                pass = 0;
            end
        end
        $display(pass ? "  PASS: all outputs equal" : "  FAIL: outputs differ");
    endtask

    task test_one_hot();
        logic signed [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1];
        $display("\n-- Test 2: one-hot (token 5 dominant) ----------------");
        for (int i = 0; i < SEQ_LEN; i++) scores[i] = to_q8_8(-4.0);
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

    task test_back_to_back();
        logic signed [DATA_WIDTH-1:0] scores_a [0:SEQ_LEN-1];
        logic signed [DATA_WIDTH-1:0] scores_b [0:SEQ_LEN-1];
        logic [OUT_WIDTH-1:0]         result_a  [0:SEQ_LEN-1];
        $display("\n-- Test 3: back-to-back rows --------------------------");

        for (int i = 0; i < SEQ_LEN; i++) scores_a[i] = to_q8_8(-2.0);
        scores_a[3] = to_q8_8(2.0);
        for (int i = 0; i < SEQ_LEN; i++) scores_b[i] = to_q8_8(-2.0);
        scores_b[12] = to_q8_8(2.0);

        send_row(scores_a);
        for (int i = 0; i < SEQ_LEN; i++) result_a[i] = out_buf[i];
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
            $display("  FAIL: peak mismatch - possible state leak");
    endtask

    // =========================================================================
    // Stimulus
    // =========================================================================
    initial begin
        rst_n       = 1'b0;
        in_valid    = 1'b0;
        in_first    = 1'b0;
        in_data     = '0;
        capture_idx = 0;

        repeat(8) @(posedge clk);
        @(negedge clk);  // release reset on negedge so it's stable by next posedge
        rst_n = 1'b1;
        repeat(4) @(posedge clk);

        $display("-- Reset complete, starting tests --");

        test_uniform();
        test_one_hot();
        test_back_to_back();

        $display("\n-- All tests complete --------------------------------");
        $finish;
    end

    // Safety timeout
    initial begin
        #2_000_000;
        $display("TIMEOUT: simulation exceeded 2 ms — check LUT paths or FSM hang");
        $finish;
    end

endmodule