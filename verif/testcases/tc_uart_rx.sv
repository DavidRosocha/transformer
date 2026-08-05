// Smallest useful test: send one short frame and confirm the bytes appear on
// serial_rx with correct UART timing.
//
// Nothing self-checks yet -- there is no monitor or scoreboard. This test
// passes if it runs to completion; correctness is judged on the waveform.
// A 4-byte payload keeps the whole frame at 8 bytes (~86 us) instead of the
// 2053 bytes a real inference frame would send.

class tc_uart_rx extends tpu_base_test;
    `uvm_component_utils(tc_uart_rx)

    function new(string name = "tc_uart_rx", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
        uart_sequence seq;

        phase.raise_objection(this);

        seq = uart_sequence::type_id::create("seq");
        seq.type_byte = 8'h01;                 // layer 1 inference
        seq.payload   = new[4];                // short, so the sim is quick
        foreach (seq.payload[i])
            seq.payload[i] = 8'hA0 + i[7:0];   // A0 A1 A2 A3, easy to spot

        // Blocks until every byte of the frame is on the wire.
        seq.start(agent.sequencer);

        phase.drop_objection(this);
    endtask

endclass
