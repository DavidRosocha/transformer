// End-to-end compute test: load weights, send an inference frame, and check
// the result frame that comes back on serial_tx.
//
// The trick that makes this checkable without a golden model: every weight is
// zero. Then Q = X*0 = 0, K = 0, V = 0, so scores = Q*K' = 0 and
// context = attn x V = 0 regardless of what softmax did. The output is zero
// for any input, and the softmax LUT approximation cancels out completely.
//
// Weights go in through the backdoor rather than over UART -- loading 64 KB
// serially costs 708 ms of simulated time and proves nothing this test cares
// about. tc_weights_loaded is what verifies the UART weight path.

class tc_uart_tx extends tpu_base_test;
    `uvm_component_utils(tc_uart_tx)

    virtual weight_bkdr_if wbkdr;

    // Generous: the DUT runs seven matmuls plus softmax, then shifts 2050
    // bytes back out at 921600 baud (~22 ms on its own).
    const time RESULT_TIMEOUT = 100ms;

    function new(string name = "tc_uart_tx", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        if (!uvm_config_db #(virtual weight_bkdr_if)::get(this, "", "wbkdr", wbkdr))
            `uvm_fatal(get_type_name(), "no weight backdoor interface found")
    endfunction

    task run_phase(uvm_phase phase);
        bit [7:0] payload [];

        phase.raise_objection(this);

        wbkdr.fill_all(16'h0000);              // all 8 matrices zeroed
        env.sb.expect_zero_result = 1;

        payload = new[2048];                   // allocate (bit -> zero-filled)

        // Fill with deliberately non-zero data, so an all-zero reply means the
        // weights did the zeroing rather than the input being trivial.
        foreach (payload[i])
            payload[i] = 8'h20 + i[7:0];

        send_frame(8'h01, payload);            // 0x01 = layer 1 inference

        // The DUT computes for a long time before replying, so wait on the
        // reply itself rather than a fixed delay.
        fork
            begin
                wait (env.sb.tx_frame_done == 1'b1);
                `uvm_info(get_type_name(), "result frame arrived", UVM_LOW)
            end
            begin
                #RESULT_TIMEOUT;
                `uvm_error(get_type_name(),
                    $sformatf("no result frame after %0t -- got %0d of %0d bytes",
                              RESULT_TIMEOUT, env.sb.tx_bytes.size(), 2050))
            end
        join_any
        disable fork;

        phase.drop_objection(this);
    endtask

endclass
