// Base test. Every tc_*.sv extends this.
//
// Right now it does nothing but prove the UVM flow works end to end.
// The env, the reset task and the backdoor weight load all land here later.

class tpu_base_test extends uvm_test;
    `uvm_component_utils(tpu_base_test)

    function new(string name = "tpu_base_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
        phase.raise_objection(this);
        `uvm_info(get_type_name(), "hello from the base test", UVM_LOW)
        #1us;
        phase.drop_objection(this);
    endtask

endclass
