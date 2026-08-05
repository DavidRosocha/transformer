// The only module in the testbench. Everything else is a class.
//
// Responsibilities: make a clock, make a reset, instantiate the DUT and the
// interface, hand the interface to UVM, and start the test.

`timescale 1ns / 1ps

module tb_top;

    import uvm_pkg::*;
    import tpu_pkg::*;

    // 100 MHz -> 10 ns period -> toggle every 5 ns
    logic clk = 0;
    always #5 clk = ~clk;

    uart_if intf (clk);

    attention_fsm dut (
        .clk       (clk),
        .rst       (intf.rst),
        .serial_rx (intf.rx),
        .serial_tx (intf.tx)
    );

    // Reset lives here for now. It moves into the base test later, once the
    // test actually has sequences whose timing needs to line up with it.
    initial begin
        intf.rst = 1;          // intf.rx is owned by the driver, not touched here
        repeat (10) @(posedge clk);
        intf.rst = 0;
    end

    initial begin
        // "null, *" = visible to every component in the hierarchy under the
        // name "vif". The agent will pull it back out with a matching ::get().
        uvm_config_db #(virtual uart_if)::set(null, "*", "vif", intf);
        run_test();
    end

endmodule
