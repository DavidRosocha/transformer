// Backdoor read access into the DUT's weight BRAM.
//
// Why this exists: every UVM class lives inside tpu_pkg, and SystemVerilog
// forbids hierarchical references from a package -- so a test cannot write
// tb_top.dut.weight_ram.mem[...] directly. An interface is module scope, where
// hierarchical references are legal and get resolved at elaboration instead of
// compile time. The test reaches this through a virtual interface handle, the
// same way the driver reaches uart_if.
//
// This is white-box access. It breaks if the weight_ram instance is renamed,
// which is the price of seeing inside a BRAM at all -- no pin-level monitor
// can do it.

interface weight_bkdr_if;

    function automatic bit [15:0] read_word(int addr);
        return tb_top.dut.weight_ram.mem[addr];
    endfunction

    function automatic void write_word(int addr, bit [15:0] data);
        tb_top.dut.weight_ram.mem[addr] = data;
    endfunction

    // Write every word of every matrix. Loading 64 KB over UART takes 708 ms of
    // simulated time; this takes none, which is what makes compute tests
    // practical to run.
    function automatic void fill_all(bit [15:0] value);
        foreach (tb_top.dut.weight_ram.mem[i])
            tb_top.dut.weight_ram.mem[i] = value;
    endfunction

endinterface
