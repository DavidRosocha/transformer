// Send one full-size inference frame and check, byte for byte, that what the
// monitor saw on serial_rx is what the frame was supposed to contain.
//
// All this test does is pick the stimulus. The frame building lives in
// tpu_base_test.send_frame(); the comparing lives in tpu_scoreboard.

class tc_uart_rx extends tpu_base_test;
    `uvm_component_utils(tc_uart_rx)

    function new(string name = "tc_uart_rx", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
        bit [7:0] payload [];

        phase.raise_objection(this);

        payload = new[2048];                   // one 16x64 Q8.8 token block
        foreach (payload[i])
            payload[i] = 8'hA0 + i[7:0];

        send_frame(8'h01, payload);            // 0x01 = layer 1 inference

        phase.drop_objection(this);
    endtask

endclass
