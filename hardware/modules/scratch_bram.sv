/*
Scratch storage for intermediate attention results (X input, Q/K/V, scores,
attention weights, context, output). Written by the UART live-load path and by
tile_controller matmul results; read by tile_controller during compute.

Unlike weight_bram, this RAM needs TWO independent read ports. Some matmuls
take BOTH operands from this RAM at the same time:
    SCORES_MMUL  = Q x K^T   (A = Q, B = K, both here)
    VALUE_MMUL   = attn x V   (A = attn, B = V, both here)
The tile_controller reads A and B on the same cycle (LOAD_TILE asserts both
a_rd_en and b_rd_en), so a single read port cannot serve them.

1 write + 2 read (1W2R). Vivado infers this as two BRAM copies that share the
write port; the extra copy costs a few BRAM tiles, which we have to spare
(scratch is only ~16 KB and the board has 50 BRAM tiles).

Reads and writes never collide: within any one matmul the tile_controller reads
operands (LOAD_TILE) and writes results (STORE_RESULT) in different states, and
UART live-load finishes before compute begins -- so no read/write arbitration
logic is needed.
*/

module scratch_bram #(
    parameter DATA_WIDTH = 16,                    // Q8.8 fixed-point
    parameter DEPTH      = 8192,                  // 16 KB: X + Q/K/V + scores + attn + context + out
    parameter ADDR_WIDTH = $clog2(DEPTH)          // 13 bits
)(
    input  logic                  clk,

    // Write port -- UART live-load path OR tile_controller result writeback
    input  logic                  we,
    input  logic [ADDR_WIDTH-1:0] waddr,
    input  logic [DATA_WIDTH-1:0] wdata,

    // Read port A -- tile_controller A operand (X / Q / attn / context)
    input  logic [ADDR_WIDTH-1:0] raddr_a,
    output logic [DATA_WIDTH-1:0] rdata_a,

    // Read port B -- tile_controller B operand when it lives in scratch (K / V)
    input  logic [ADDR_WIDTH-1:0] raddr_b,
    output logic [DATA_WIDTH-1:0] rdata_b
);

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;
    end

    // Registered reads -- 1 cycle of latency, same as weight_bram. Both ports
    // read the shared memory; the delayed-capture logic in tile_controller
    // already accounts for this latency.
    always_ff @(posedge clk) rdata_a <= mem[raddr_a];
    always_ff @(posedge clk) rdata_b <= mem[raddr_b];

endmodule
