// Load all 8 weight matrices over UART and confirm each one landed in the
// right block of weight_bram with the right contents.
//
// Every matrix gets freshly randomized data. That matters: with identical
// payloads, an FSM that wrote W_k1 into W_q1's address block would still match
// byte for byte and the test would pass. Distinct data is what makes the
// address decode at attention_fsm.sv:474 actually get checked.

class tc_weights_loaded extends tpu_base_test;
    `uvm_component_utils(tc_weights_loaded)

    const int WORDS_PER_MATRIX = 4096;          // 64x64 Q8.8
    const int BYTES_PER_MATRIX = 4096 * 2;      // 2 bytes per word
    const int NUM_MATRICES     = 8;             // 0x10 .. 0x17

    // Backdoor into the DUT's weight BRAM. A package cannot hold a hierarchical
    // reference, so tb_top hands one in the same way it hands in uart_if.
    virtual weight_bkdr_if wbkdr;

    function new(string name = "tc_weights_loaded", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        if (!uvm_config_db #(virtual weight_bkdr_if)::get(this, "", "wbkdr", wbkdr))
            `uvm_fatal(get_type_name(), "no weight backdoor interface found")
    endfunction

    // White-box check: reads the DUT's BRAM by hierarchical path. A pin-level
    // monitor cannot see memory contents, so there is no black-box way to do
    // this. Cost is that renaming the weight_ram instance breaks the test.
    task check_weight_block(int idx, const ref bit [7:0] payload []);
        bit [15:0] exp, got;
        int        errs = 0;

        for (int w = 0; w < WORDS_PER_MATRIX; w++) begin
            // Big-endian: first byte received is the high byte.
            // attention_fsm.sv:479 -- wdata <= {held_high_byte, rx_data}
            exp = {payload[2*w], payload[2*w + 1]};
            got = wbkdr.read_word(idx*WORDS_PER_MATRIX + w);

            if (got !== exp) begin
                errs++;
                // A wrong address decode makes all 4096 words mismatch. Print
                // a few and stop -- the rest add nothing.
                if (errs <= 10)
                    `uvm_error(get_type_name(),
                        $sformatf("matrix %0d word %0d: expected 0x%04h, got 0x%04h",
                                  idx, w, exp, got))
            end
        end

        if (errs == 0)
            `uvm_info(get_type_name(),
                $sformatf("matrix %0d: all %0d words correct", idx, WORDS_PER_MATRIX),
                UVM_LOW)
        else
            `uvm_error(get_type_name(),
                $sformatf("matrix %0d: %0d of %0d words wrong",
                          idx, errs, WORDS_PER_MATRIX))
    endtask

    task run_phase(uvm_phase phase);
        bit [7:0] payload [];

        phase.raise_objection(this);

        payload = new[BYTES_PER_MATRIX];

        for (int i = 0; i < NUM_MATRICES; i++) begin
            foreach (payload[j])
                payload[j] = $urandom_range(0, 255);   // fresh data per matrix

            send_frame(8'h10 + i[7:0], payload);       // 0x10..0x17
            check_weight_block(i, payload);
        end

        phase.drop_objection(this);
    endtask

endclass
