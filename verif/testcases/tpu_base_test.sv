// Base test. Every tc_*.sv extends this.
//
// Builds the env, which in turn builds the agent and the scoreboard and
// connects them. Tests reach the sequencer as env.agent.sequencer and seed
// expectations with env.sb.expect_byte().
//
// No run_phase here: each tc_* supplies its own stimulus.

class tpu_base_test extends uvm_test;
    `uvm_component_utils(tpu_base_test)

    tpu_env env;

    function new(string name = "tpu_base_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        env = tpu_env::type_id::create("env", this);
    endfunction

    // Send one [0xAA][TYPE][payload][checksum][0x55] frame and tell the
    // scoreboard what to expect, byte for byte.
    //
    // const ref: the payload can be 8 KB for a weight matrix, so it is passed
    // by reference rather than copied. const because this task only reads it.
    virtual task send_frame(bit [7:0] type_byte, const ref bit [7:0] payload []);
        uart_sequence seq;
        bit [7:0]     checksum = 8'h00;

        seq = uart_sequence::type_id::create("seq");
        seq.type_byte = type_byte;
        seq.payload   = payload;               // dynamic array assign copies

        // Predict the frame the sequence is about to build.
        env.sb.expect_byte(8'hAA);
        env.sb.expect_byte(type_byte);
        foreach (payload[i]) begin
            env.sb.expect_byte(payload[i]);
            checksum ^= payload[i];
        end
        env.sb.expect_byte(checksum);
        env.sb.expect_byte(8'h55);

        seq.start(env.agent.sequencer);        // blocks until the last byte is out

        // Don't let the sim end the instant the last byte goes out -- give the
        // monitor time to finish looking at it.
        repeat (200) @(posedge env.agent.driver.vif.clk);
    endtask

endclass
