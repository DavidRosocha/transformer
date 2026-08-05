// Compile glue. SystemVerilog classes have to live inside a package, so every
// UVM class in this testbench gets `include'd here in dependency order:
// base classes before anything that extends them.
//
// Only the files that are actually written go in this list. Adding an empty
// file breaks the compile, so uncomment each one as you fill it in.

package tpu_pkg;

    import uvm_pkg::*;
    `include "uvm_macros.svh"

    `include "uart_agent.sv"
    `include "tpu_env.sv"
    `include "tpu_base_test.sv"
    `include "tc_uart_rx.sv"
    `include "tc_weights_loaded.sv"
    `include "tc_uart_tx.sv"
    `include "tc_mmul_correct.sv"
    `include "tc_mmul_random.sv"
    `include "tc_attention_full.sv"
    `include "tc_attention_full_layer2.sv"
    // `include "tpu_seqs.sv"
    // `include "tc_mmul_correct.sv"

endpackage
