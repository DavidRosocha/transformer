module uart_tx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 921_600
)(
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data,
    input  wire       send,
    output reg        tx,
    output reg        busy
);
    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;  // 108

    localparam IDLE  = 2'd0;
    localparam START = 2'd1;
    localparam DATA  = 2'd2;
    localparam STOP  = 2'd3;

    reg [1:0]  state = IDLE;
    reg [15:0] clk_count = 0;
    reg [2:0]  bit_idx = 0;
    reg [7:0]  shift_reg = 0;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; tx <= 1; busy <= 0;
        end else case (state)
            IDLE: begin
                tx <= 1; busy <= 0;
                if (send) begin
                    shift_reg <= data;
                    clk_count <= 0;
                    busy  <= 1;
                    state <= START;
                end
            end
            START: begin
                tx <= 0;
                if (clk_count == CLKS_PER_BIT - 1) begin
                    clk_count <= 0; state <= DATA;
                end else clk_count <= clk_count + 1;
            end
            DATA: begin
                tx <= shift_reg[0];
                if (clk_count == CLKS_PER_BIT - 1) begin
                    clk_count <= 0;
                    shift_reg <= {1'b0, shift_reg[7:1]};
                    if (bit_idx == 7) begin bit_idx <= 0; state <= STOP; end
                    else bit_idx <= bit_idx + 1;
                end else clk_count <= clk_count + 1;
            end
            STOP: begin
                tx <= 1;
                if (clk_count == CLKS_PER_BIT - 1) begin
                    clk_count <= 0; state <= IDLE;
                end else clk_count <= clk_count + 1;
            end
        endcase
    end
endmodule