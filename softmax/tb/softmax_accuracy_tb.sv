// =============================================================================
// softmax_accuracy_tb.sv  —  Accuracy testbench
//
// Drives N_ROWS random input rows through the DUT, captures outputs, and
// writes everything to a text file for Python evaluation.
//
// Prerequisites:
//   1. Run: python rtl_accuracy.py --gen-inputs
//      This produces rtl_inputs.txt in the sim directory.
//   2. Set LUT file paths in softmax_unit.sv ($readmemh).
//   3. In Vivado: set softmax_accuracy_tb as top, run simulation for 5ms.
//   4. Run: python rtl_accuracy.py  (to evaluate accuracy vs float reference)
//
// Output file format (one line per row):
//   INPUT:  v0 v1 ... v15    (signed decimal Q8.8 integers)
//   OUTPUT: o0 o1 ... o15    (unsigned decimal uint8)
// =============================================================================
`timescale 1ns / 1ps

module softmax_accuracy_tb;

    import softmax_pkg::*;

    // ── Clock & reset ────────────────────────────────────────────────────────
    logic clk;
    initial clk = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    reg rst_n;

    // ── DUT ports ────────────────────────────────────────────────────────────
    logic                          in_valid;
    logic                          in_first;
    logic signed [DATA_WIDTH-1:0]  in_data;
    logic                          out_valid;
    logic [OUT_WIDTH-1:0]          out_data;
    logic                          busy;

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

    // ── Parameters ───────────────────────────────────────────────────────────
    localparam int    N_ROWS   = 500;

    // Relative to the simulator's working directory, assumed to be verif/sim --
    // same convention as softmax_unit's LUT_DIR parameter.
    localparam string SIM_DIR  = "../../softmax/sim";
    localparam string IN_FILE  = {SIM_DIR, "/rtl_inputs.txt"};
    localparam string OUT_FILE = {SIM_DIR, "/rtl_outputs.txt"};

    // ── Storage ──────────────────────────────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] input_rows  [0:N_ROWS-1][0:SEQ_LEN-1];
    logic        [OUT_WIDTH-1:0]  output_rows [0:N_ROWS-1][0:SEQ_LEN-1];

    integer fd_in, fd_out, r;

    // =========================================================================
    // Tasks
    // =========================================================================

    // Send one row, driving signals AFTER the clock edge to avoid xsim races.
    task automatic send_row(input int row_idx);
        for (int i = 0; i < SEQ_LEN; i++) begin
            @(posedge clk); #1;
            in_valid = 1'b1;
            in_first = (i == 0) ? 1'b1 : 1'b0;
            in_data  = input_rows[row_idx][i];
        end
        @(posedge clk); #1;
        in_valid = 1'b0;
        in_first = 1'b0;
    endtask

    // Capture SEQ_LEN output values for one row.
    // Called AFTER send_row finishes; waits for out_valid then collects all outputs.
    task automatic capture_row(input int row_idx);
        int cap_cnt;
        cap_cnt = 0;
        while (!out_valid) @(posedge clk);
        while (cap_cnt < SEQ_LEN) begin
            if (out_valid) begin
                output_rows[row_idx][cap_cnt] = out_data;
                cap_cnt++;
            end
            if (cap_cnt < SEQ_LEN) @(posedge clk);
        end
    endtask

    // =========================================================================
    // Main stimulus
    // =========================================================================
    initial begin
        // Reset
        rst_n    = 1'b0;
        in_valid = 1'b0;
        in_first = 1'b0;
        in_data  = '0;
        repeat(4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat(2) @(posedge clk);

        // Load input rows
        fd_in = $fopen(IN_FILE, "r");
        if (fd_in == 0) begin
            $display("ERROR: could not open %s — run: python rtl_accuracy.py --gen-inputs", IN_FILE);
            $finish;
        end
        for (int row = 0; row < N_ROWS; row++)
            for (int i = 0; i < SEQ_LEN; i++)
                r = $fscanf(fd_in, "%d", input_rows[row][i]);
        $fclose(fd_in);
        $display("[tb] Loaded %0d input rows from %s", N_ROWS, IN_FILE);

        // Run DUT row by row (sequential: send then capture, never overlapping)
        for (int row = 0; row < N_ROWS; row++) begin
            send_row(row);
            capture_row(row);
            // Wait for DUT to return to IDLE before next row
            wait (busy == 1'b0);
            repeat(2) @(posedge clk);

            if (row % 50 == 0)
                $display("[tb] Row %0d / %0d done", row, N_ROWS);
        end

        // Write outputs
        fd_out = $fopen(OUT_FILE, "w");
        if (fd_out == 0) begin
            $display("ERROR: could not open output file %s", OUT_FILE);
            $finish;
        end
        for (int row = 0; row < N_ROWS; row++) begin
            $fwrite(fd_out, "INPUT:");
            for (int i = 0; i < SEQ_LEN; i++)
                $fwrite(fd_out, " %0d", input_rows[row][i]);
            $fwrite(fd_out, "\n");
            $fwrite(fd_out, "OUTPUT:");
            for (int i = 0; i < SEQ_LEN; i++)
                $fwrite(fd_out, " %0d", output_rows[row][i]);
            $fwrite(fd_out, "\n");
        end
        $fclose(fd_out);
        $display("[tb] Wrote %s — run: python rtl_accuracy.py", OUT_FILE);
        $finish;
    end

    // Safety timeout
    initial begin
        #10_000_000;
        $display("TIMEOUT: simulation exceeded 10 ms");
        $finish;
    end

endmodule