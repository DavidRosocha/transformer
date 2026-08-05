// UART pin bundle shared between the DUT and the UVM testbench.
//
// No clocking block: UART is asynchronous, so the driver and monitor time
// themselves by counting clk edges (CLKS_PER_BIT = 100 MHz / 921600 = 108)
// rather than by sampling on a clocking-block skew.

interface uart_if (input logic clk);

    logic rst;   // active high, matches attention_fsm
    logic rx;    // testbench drives  -> DUT serial_rx
    logic tx;    // DUT drives        -> testbench samples

endinterface
