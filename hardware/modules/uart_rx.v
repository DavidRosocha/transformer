module uart_rx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 4_000_000
)(
    input  wire clk,
    input  wire rst,
    input  wire rx,
    output reg  [7:0] data,
    output reg  valid
);
    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;  // 25

    localparam IDLE  = 2'd0;
    localparam START = 2'd1;
    localparam DATA  = 2'd2;
    localparam STOP  = 2'd3;

    reg [1:0]  state = IDLE;
    reg [15:0] clk_count = 0;
    reg [2:0]  bit_idx = 0;
    reg [7:0]  shift_reg = 0;
    reg        rx_sync1, rx_sync2;

    always @(posedge clk) begin
        rx_sync1 <= rx;
        rx_sync2 <= rx_sync1;
    end

    always @(posedge clk) begin
        valid <= 0;
        if (rst) begin
            state <= IDLE; clk_count <= 0; bit_idx <= 0;
        end else case (state)
            IDLE: begin
                if (!rx_sync2) begin
                    clk_count <= 0;
                    state <= START;
                end
            end
            START: begin
                if (clk_count == (CLKS_PER_BIT/2 - 1)) begin
                    clk_count <= 0;
                    state <= DATA;
                end else clk_count <= clk_count + 1;
            end
            DATA: begin
                if (clk_count == CLKS_PER_BIT - 1) begin
                    clk_count <= 0;
                    shift_reg <= {rx_sync2, shift_reg[7:1]};
                    if (bit_idx == 7) begin bit_idx <= 0; state <= STOP; end
                    else bit_idx <= bit_idx + 1;
                end else clk_count <= clk_count + 1;
            end
            STOP: begin
                if (clk_count == CLKS_PER_BIT - 1) begin
                    if (rx_sync2) begin
                        data  <= shift_reg;
                        valid <= 1;
                    end
                    clk_count <= 0;
                    state <= IDLE;
                end else clk_count <= clk_count + 1;
            end
        endcase
    end
endmodule