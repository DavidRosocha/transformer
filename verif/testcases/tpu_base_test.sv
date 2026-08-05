// Base test. Every tc_*.sv extends this.
//
// Builds the agent and nothing else. No env yet -- an env holding one agent
// would be a container around a container. It earns its place once the
// scoreboard exists.
//
// No run_phase here: each tc_* supplies its own stimulus.

class tpu_base_test extends uvm_test;
    `uvm_component_utils(tpu_base_test)

    uart_agent agent;

    function new(string name = "tpu_base_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        agent = uart_agent::type_id::create("agent", this);
    endfunction

endclass
