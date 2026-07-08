/*
Weight storage for the 8 attention weight matrices (W_q1, W_k1, W_v1, W_out1,
W_q2, W_k2, W_v2, W_out2), loaded once at boot over UART and read repeatedly
during inference.

Each matrix is 64x64 Q8.8 values (16 bits/value) = 4096 words = 8192 bytes.
8 matrices -> 32768 words = 65536 bytes = 64 KB total.

Simple dual-port: one write port (fed by the UART RX / weight-load path in
attention_fsm), one independent read port (fed by tile_controller during
matmul). Ports never need to arbitrate against each other -- weight loading
finishes entirely before compute ever reads -- so no read/write conflict
logic is needed.

Address mapping (which matrix lives where) is NOT this module's job -- the
caller passes in a flat word address. attention_fsm is responsible for
turning a TYPE byte (0x10-0x17) into a base offset, e.g.
base_addr = (type_byte - 8'h10) * MATRIX_WORDS.
*/

module weight_bram #(
    parameter DATA_WIDTH   = 16,          // Q8.8 fixed-point
    parameter NUM_MATRICES = 8,           // W_q1,W_k1,W_v1,W_out1,W_q2,W_k2,W_v2,W_out2
    parameter MATRIX_WORDS = 64 * 64,     // 4096 words per 64x64 matrix
    parameter DEPTH        = NUM_MATRICES * MATRIX_WORDS,   // 32768 words = 64 KB
    parameter ADDR_WIDTH   = $clog2(DEPTH)                  // 15 bits
)(
    input  logic                  clk,

    // Write port -- UART RX / weight-load path
    input  logic                  we,
    input  logic [ADDR_WIDTH-1:0] waddr,
    input  logic [DATA_WIDTH-1:0] wdata,

    // Read port -- tile_controller / attention_fsm during compute
    input  logic [ADDR_WIDTH-1:0] raddr,
    output logic [DATA_WIDTH-1:0] rdata
);

    // ram_style attribute forces Vivado to map this onto Block RAM tiles
    // rather than distributed RAM (LUTRAM), even if the coding style alone
    // would be ambiguous at this size. Check the synthesis utilization
    // report's "Block RAM Tile" count to confirm it actually landed there.
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;
    end

    // Registered read -- BRAM has 1 cycle of latency between raddr and
    // rdata being valid. This is mandatory for BRAM inference, not a
    // stylistic choice: an asynchronous/combinational read here would
    // force Vivado to fall back to LUTRAM instead.
    always_ff @(posedge clk) begin
        rdata <= mem[raddr];
    end

endmodule
