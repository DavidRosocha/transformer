module top (
    input  wire clk,
    input  wire rst,
    input  wire uart_rx_pin,
    output wire uart_tx_pin
);

// ── UART wires ───────────────────────────────────────────────────────────────
wire [7:0] rx_data;
wire       rx_valid;
reg  [7:0] tx_data;
reg        tx_send;
wire       tx_busy;

uart_rx #(.CLK_FREQ(100_000_000), .BAUD_RATE(921_600)) RX (
    .clk(clk), .rst(rst),
    .rx(uart_rx_pin),
    .data(rx_data), .valid(rx_valid)
);

uart_tx #(.CLK_FREQ(100_000_000), .BAUD_RATE(921_600)) TX (
    .clk(clk), .rst(rst),
    .data(tx_data), .send(tx_send),
    .tx(uart_tx_pin), .busy(tx_busy)
);

// ── BRAM ─────────────────────────────────────────────────────────────────────
// Addresses 0    – 2047: incoming Q8.8 embedding (16x64 x 2 bytes)
// Addresses 2048 – 4095: attention core result (written by attention core)
reg [7:0]  bram [0:4095];
reg [11:0] bram_wr_addr;
reg        bram_we;
reg [7:0]  bram_wr_data;

always @(posedge clk)
    if (bram_we) bram[bram_wr_addr] <= bram_wr_data;

// ── Protocol constants ────────────────────────────────────────────────────────
localparam START_BYTE    = 8'hAA;
localparam STOP_BYTE     = 8'h55;
localparam ACK_BYTE      = 8'h06;
localparam NAK_BYTE      = 8'h15;
localparam PAYLOAD_BYTES = 16'd2048;

// ── FSM states ────────────────────────────────────────────────────────────────
localparam S_IDLE     = 3'd0;
localparam S_TYPE     = 3'd1;
localparam S_PAYLOAD  = 3'd2;
localparam S_CHECKSUM = 3'd3;
localparam S_STOP     = 3'd4;
localparam S_ACK      = 3'd5;
localparam S_DISPATCH = 3'd6;

reg [2:0]  state = S_IDLE;
reg [7:0]  type_reg;
reg [15:0] byte_count;
reg [7:0]  chk_accum;
reg        frame_ok;

// Output signals to attention core
// Uncomment and connect once attention core ports are defined
reg        layer1_start;
reg        layer2_start;

// Result transmission
reg [11:0] tx_byte_idx;
reg        sending_result;

always @(posedge clk) begin
    bram_we      <= 0;
    tx_send      <= 0;
    layer1_start <= 0;
    layer2_start <= 0;

    if (rst) begin
        state          <= S_IDLE;
        byte_count     <= 0;
        chk_accum      <= 0;
        frame_ok       <= 0;
        sending_result <= 0;
    end else begin

        // ── Frame reception FSM ──────────────────────────────────────────────
        case (state)

            S_IDLE: begin
                if (rx_valid && rx_data == START_BYTE)
                    state <= S_TYPE;
            end

            S_TYPE: begin
                if (rx_valid) begin
                    type_reg     <= rx_data;
                    byte_count   <= 0;
                    bram_wr_addr <= 0;
                    chk_accum    <= 0;
                    state        <= S_PAYLOAD;
                end
            end

            S_PAYLOAD: begin
                if (rx_valid) begin
                    bram_we      <= 1;
                    bram_wr_data <= rx_data;
                    bram_wr_addr <= byte_count[11:0];
                    chk_accum    <= chk_accum ^ rx_data;
                    byte_count   <= byte_count + 1;
                    if (byte_count == PAYLOAD_BYTES - 1)
                        state <= S_CHECKSUM;
                end
            end

            S_CHECKSUM: begin
                if (rx_valid) begin
                    frame_ok <= (chk_accum == rx_data);
                    state    <= S_STOP;
                end
            end

            S_STOP: begin
                if (rx_valid) begin
                    if (rx_data != STOP_BYTE)
                        frame_ok <= 0;
                    state <= S_ACK;
                end
            end

            S_ACK: begin
                if (!tx_busy) begin
                    tx_data <= frame_ok ? ACK_BYTE : NAK_BYTE;
                    tx_send <= 1;
                    state   <= frame_ok ? S_DISPATCH : S_IDLE;
                end
            end

            S_DISPATCH: begin
                case (type_reg)
                    8'h01: layer1_start <= 1;
                    8'h02: layer2_start <= 1;
                endcase
                state <= S_IDLE;
            end

        endcase

        // ── Result transmission ──────────────────────────────────────────────
        if (sending_result && !tx_busy) begin
            tx_data      <= bram[tx_byte_idx];
            tx_send      <= 1;
            tx_byte_idx  <= tx_byte_idx + 1;
            if (tx_byte_idx == 12'd2051)
                sending_result <= 0;
        end

    end
end
endmodule