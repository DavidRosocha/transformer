module attention_fsm (
    input  logic        clk,
    input  logic        rst,
    input  logic        serial_rx,   // 1-bit RX line pin (into internal uart_rx)
    output logic        serial_tx    // 1-bit TX line pin (from internal uart_tx)
);

// ── UART frame format ─────────────────────────────────────────────────────────
// PC -> FPGA:  [0xAA] [TYPE] [payload] [XOR checksum] [0x55]
//   The trailing checksum and stop byte are NOT verified here. After a weight
//   frame they are swallowed in IDLE (a checksum that happens to be 0xAA sends us
//   to WAIT_TYPE, where the 0x55 matches no type mask and drops us back to IDLE --
//   so recovery is guaranteed either way). After a live frame they land during the
//   matmul states, which ignore valid_rx entirely.
//
// FPGA -> PC:  [0xAA] [2048 payload bytes] [XOR checksum]
//   No type byte and no stop byte -- the PC only ever receives one kind of frame.
//
// ACK (0x06) is sent once per WEIGHT frame only. Live inference frames are not
// ACKed: the result frame itself is the acknowledgement. There is no NAK.
//
// ── Type byte encoding ────────────────────────────────────────────────────────
// Upper nibble 0x1X = weight frame; lower 3 bits = matrix index (0-7)
//   0x10 = W_q1,  0x11 = W_k1,  0x12 = W_v1,  0x13 = W_out1
//   0x14 = W_q2,  0x15 = W_k2,  0x16 = W_v2,  0x17 = W_out2
// Upper nibble 0x0X = inference frame
//   0x01 = layer 1 input,  0x02 = layer 2 input
localparam WEIGHT_TYPE_MASK = 4'h1;
localparam INFER_TYPE_MASK  = 4'h0;

localparam START_BYTE = 8'hAA;
localparam ACK_BYTE   = 8'h06;

// ── State encoding ────────────────────────────────────────────────────────────
typedef enum logic [3:0] {
    IDLE,
    WAIT_TYPE,
    STORING_WEIGHTS,
    STORING_LIVE,
    LOADING_MMUL_MATRICES,
    QKV_MMUL,
    SCORES_MMUL,
    SOFTMAX_FEED_ADDR,   // put scores[row][col] address on the read pin
    SOFTMAX_FEED_DATA,   // data is back -> feed it to the softmax unit
    SOFTMAX_COLLECT,     // write the 16 results into the attn region
    VALUE_MMUL,
    WOUT_MMUL,
    INITIALIZE_SEND,   // send 0xAA start byte; present word 0's address
    LOADING_SEND,      // let the next word's RAM address settle (1 cycle)
    SENDING_LIVE,      // send the current word's 2 bytes (hi then lo)
    SENDING_CHECKSUM   // send the XOR checksum
} state_t;

state_t state, next_state;

// ── Registers ─────────────────────────────────────────────────────────────────
reg [2:0]  matrix_idx;
reg [12:0] word_count;
reg [7:0]  held_high_byte;
reg        byte_phase;
reg [7:0]  type_reg;
reg [7:0]  chk;               // TX payload XOR checksum accumulator
reg        layer;             // 0 = layer-1 weights, 1 = layer-2 weights (from infer type byte)
reg [9:0]  live_waddr;        // latched input-word index (word_count BEFORE its increment)
reg        tx_pending;        // send asserted but uart_tx hasn't raised busy yet

// ── Matmul phase: single source of truth for which matmul LOADING sets up ──────
localparam [2:0] PH_Q      = 3'd0,   // X x W_q  -> Q
                 PH_K      = 3'd1,   // X x W_k  -> K
                 PH_V      = 3'd2,   // X x W_v  -> V
                 PH_SCORES = 3'd3,   // Q x K^T  -> scores
                 PH_VALUE  = 3'd4,   // attn x V -> context
                 PH_WOUT   = 3'd5;   // context x W_out -> output
reg [2:0]  mm_phase;  // advadnces Q -> K -> V -> SCORES -> VALUE -> WOUT

// ── Softmax sequencing ──────────────────────────────────────────────────────────
reg  [3:0]  sm_row;      // which score row (0..15)          -- outer loop
reg  [3:0]  sm_col;      // feed index within row (0..15)     -- inner loop
reg  [3:0]  sm_ocol;     // collect index within row (0..15)
logic [12:0] sm_rd_addr; // scratch read address we drive during softmax feed

// ── Softmax unit handshake ──────────────────────────────────────────────────────
logic        in_valid, in_first;   // we drive these INTO the softmax
logic [15:0] in_data;
logic        out_valid;            // these come OUT of the softmax
logic [7:0]  out_data;
logic        sm_busy;

// ── UART byte-level signals (to/from the internal uart_rx / uart_tx) ───────────
logic [7:0]  rx_data;    // uart_rx.data  -- received byte
logic        valid_rx;   // uart_rx.valid -- one pulse per received byte
logic [7:0]  tx_data;    // uart_tx.data  -- byte to transmit (FSM-driven)
logic        tx_send;    // uart_tx.send  -- pulse to start transmit (FSM-driven)
logic        tx_busy;    // uart_tx.busy  -- high while transmitting

// uart_tx latches `send` and raises `busy` on the FOLLOWING edge, so for one cycle
// after we pulse tx_send the TX looks idle while it is already committed. Gating on
// !tx_busy alone lets a second send fire into that gap, where uart_tx has left IDLE
// and ignores it -- the byte is silently dropped. tx_pending covers the gap.
logic        tx_ready;
assign tx_ready = !tx_busy && !tx_pending;

// ── UART write signals (shared between both BRAMs, only one we high at a time)
logic        weight_we;
logic        inference_we;
logic [14:0] waddr;            // 15 bits covers weight_ram (max 0x7FFF)
logic [15:0] wdata;

// ── Tile controller config signals ────────────────────────────────────────────
logic [6:0] num_rows_a;
logic [6:0] num_columns_b;
logic [6:0] shared_dimension_ab;
logic       start_matmul;
logic       matmul_done;
logic       matmul_busy;

// ── Tile controller memory interface wires ────────────────────────────────────
logic [11:0] a_rd_addr;   // tile_ctrl drives this (0..4095)
logic        a_rd_en;
logic [11:0] b_rd_addr;   // tile_ctrl drives this (0..4095)
logic        b_rd_en;
logic [11:0] c_wr_addr;   // tile_ctrl drives this
logic        c_wr_en;
logic [31:0] c_wr_data;   // ACC_WIDTH=32, shift right 8 to get Q8.8

// ── Base address registers (set per QKV pass in LOADING_MMUL_MATRICES) ────────
logic [14:0] b_base;  // WEIGHT-side B base: W_q=0x0000, W_k=0x1000, W_v=0x2000, W_out=0x3000
logic [12:0] c_base;  // scratch result start: Q=0x0400,  K=0x0800,   V=0x0C00

// ── BRAM read data outputs ────────────────────────────────────────────────────
logic [15:0] weight_rdata;   // weight_ram read port
logic [15:0] inf_rdata_a;    // scratch read port A (tile_ctrl A operand)
logic [15:0] inf_rdata_b;    // scratch read port B (tile_ctrl B operand, when in scratch)

// ── Combinational: tile_ctrl address + base offset = actual BRAM address ──────
// The tile_controller drives addresses starting from 0; we add a base offset so
// it reads from the right region. A is ALWAYS in scratch; B is in weight_ram for
// QKV/WOUT, but in scratch for SCORES (K) and VALUE (V).
logic [14:0] weight_raddr;   // weight-side B address
logic [12:0] inf_raddr_a;    // scratch A address
logic [12:0] inf_raddr_b;    // scratch B address (K / V)

// Per-matmul operand routing. A always in scratch; B source + bases by state.
logic [12:0] a_base;         // scratch A base:  X=0x0000, Q=0x0400, attn=0x1100, context=0x1200
logic [12:0] b_base_inf;     // scratch B base:  K=0x0800, V=0x0C00 (only used when b_src_inf)
logic        b_src_inf;      // 1 => B operand comes from scratch, 0 => from weight_ram
logic [15:0] b_rd_data_mux;  // the B data actually handed to the tile_controller

always_comb begin
    // B comes from scratch only for the two intra-scratch matmuls.
    b_src_inf = (state == SCORES_MMUL) || (state == VALUE_MMUL);

    // A-operand base within scratch RAM.
    case (state)
        SCORES_MMUL: a_base = 13'h0400;  // A = Q
        VALUE_MMUL:  a_base = 13'h1100;  // A = attention weights
        WOUT_MMUL:   a_base = 13'h1200;  // A = context
        default:     a_base = 13'h0000;  // QKV: A = X input
    endcase

    // B-operand base within scratch RAM (ignored unless b_src_inf=1).
    // SCORES reads B = K^T: K is already stored transposed at write-back (see the
    // c_wr_offset logic below), so this base just points at the K region and the
    // tile_controller reads it row-major with cfg_n=16.
    case (state)
        SCORES_MMUL: b_base_inf = 13'h0800;  // B = K (stored transposed)
        VALUE_MMUL:  b_base_inf = 13'h0C00;  // B = V
        default:     b_base_inf = 13'h0000;
    endcase
end

assign weight_raddr   = b_rd_addr + b_base;            // weight-side B (QKV / WOUT)
assign inf_raddr_b    = b_rd_addr + b_base_inf;        // scratch B (K / V)
assign b_rd_data_mux  = b_src_inf ? inf_rdata_b        // SCORES/VALUE: B from scratch
                                  : weight_rdata;      // QKV/WOUT:     B from weight_ram

// ── Scratch read port A: shared between tile controller and softmax ────────────
// scores[sm_row][sm_col] = 0x1000 + row*16 + col.  {sm_row, sm_col} == row*16 + col.
assign sm_rd_addr  = 13'h1000 + {sm_row, sm_col};
assign inf_raddr_a =
      (state == SOFTMAX_FEED_ADDR || state == SOFTMAX_FEED_DATA) ? sm_rd_addr   // softmax feed
    : (state == INITIALIZE_SEND || state == LOADING_SEND || state == SENDING_LIVE)
                                                              ? (13'h1600 + word_count[9:0])  // output readback
    :                                                           (a_rd_addr + a_base);          // tile controller

// ── Softmax feed: valid only during FEED_DATA; in_first on the row's element 0 ──
assign in_valid = (state == SOFTMAX_FEED_DATA);
assign in_first = (state == SOFTMAX_FEED_DATA) && (sm_col == 4'd0);
assign in_data  = inf_rdata_a;             // the score fetched last cycle

// ── Combinational: mux inference_ram write port ───────────────────────────────
// During STORING_LIVE: write UART bytes to input region (word 0..1023)
// All other times: write tile_ctrl results (Q/K/V scratch regions)
logic [12:0] inf_waddr_muxed;
logic [15:0] inf_wdata_muxed;
logic        inf_we_muxed;

// ── K is stored TRANSPOSED on write-back ──────────────────────────────────────
// SCORES = Q x K^T. Instead of transpose-addressing K at read time (a base offset
// can't reorder a stride), we scatter K's elements into transposed positions as
// the tile_controller writes them -- free, no extra cycles.
// During the K pass, tile_ctrl emits c_wr_addr = i*64 + j  (i = token 0..15,
// j = dim 0..63). We store K[i][j] at K^T offset j*16 + i, so SCORES reads it
// row-major with cfg_n=16.  j*16 + i == {c_wr_addr[5:0], c_wr_addr[9:6]}.
// Q (rc=0) and V (rc=2) are stored row-major, unchanged.
logic        k_pass;
logic [12:0] c_wr_offset;

assign k_pass      = (state == QKV_MMUL) && (mm_phase == PH_K);
assign c_wr_offset = k_pass ? {3'b0, c_wr_addr[5:0], c_wr_addr[9:6]}  // transposed
                            : {1'b0, c_wr_addr};                       // row-major

// Write port has three owners: UART live-load, softmax results, matmul results.
// The live-load port is selected by inference_we, NOT by state == STORING_LIVE:
//   - word_count has already incremented by the cycle inference_we is high, so the
//     address must come from live_waddr (latched pre-increment), not word_count.
//   - the final word (1023) retires one cycle after the state has moved on to
//     LOADING_MMUL_MATRICES, so a state-based selector would drop it. inference_we
//     is only ever set in STORING_LIVE, and the tile controller cannot assert
//     c_wr_en that early in a matmul, so the two owners can't collide.
assign inf_waddr_muxed = inference_we               ? {3'b0, live_waddr}                 // UART: input region
                       : (state == SOFTMAX_COLLECT) ? (13'h1100 + {sm_row, sm_ocol})     // attn[row][col]
                       :                              (c_wr_offset + c_base);            // matmul result (K transposed)
assign inf_wdata_muxed = inference_we               ? wdata
                       : (state == SOFTMAX_COLLECT) ? {8'b0, out_data}                   // uint8 prob -> Q8.8 fractional
                       :                              c_wr_data[23:8];                   // Q16.16 -> Q8.8 (right-shift 8)
assign inf_we_muxed    = inference_we               ? 1'b1
                       : (state == SOFTMAX_COLLECT) ? out_valid                          // write each result as it streams out
                       :                              c_wr_en;


// ── UART instantiations ───────────────────────────────────────────────────────
uart_rx reciever (
    .clk  (clk),
    .rst  (rst),
    .rx   (serial_rx),   // 1-bit line pin in
    .data (rx_data),     // 8-bit byte out
    .valid(valid_rx)     // one pulse per received byte
);

uart_tx transmitter (
    .clk  (clk),
    .rst  (rst),
    .data (tx_data),     // 8-bit byte in (FSM-driven)
    .send (tx_send),     // start pulse
    .tx   (serial_tx),   // 1-bit line pin out
    .busy (tx_busy)      // high while shifting
);

// ── BRAM instantiations ───────────────────────────────────────────────────────
weight_bram weight_ram (
    .clk   (clk),
    .we    (weight_we),
    .waddr (waddr),
    .wdata (wdata),
    .raddr (weight_raddr),   // tile_ctrl b_rd_addr + b_base
    .rdata (weight_rdata)    // feeds tile_ctrl b_rd_data
);

scratch_bram #(.DEPTH(8192)) inference_ram (
    .clk    (clk),
    .we     (inf_we_muxed),
    .waddr  (inf_waddr_muxed),
    .wdata  (inf_wdata_muxed),
    .raddr_a(inf_raddr_a),    // A operand (X / Q / attn / context)
    .rdata_a(inf_rdata_a),    // feeds tile_ctrl a_rd_data
    .raddr_b(inf_raddr_b),    // B operand when in scratch (K / V)
    .rdata_b(inf_rdata_b)     // feeds tile_ctrl b_rd_data via mux
);

// ── Tile controller instantiation ─────────────────────────────────────────────
// NOTE! If the board had more DSPs we could run 3 systolic arrays in parallel,
// cutting QKV time from ~0.09 ms to ~0.03 ms. Sequential only because we can
// only fit one 8x8 array (64 DSPs) within the Basys 3's 90 DSP limit.
tile_controller mmul (
    .clk      (clk),
    .rst      (rst),
    .cfg_m    (num_rows_a),
    .cfg_n    (num_columns_b),
    .cfg_k    (shared_dimension_ab),
    .cfg_start(start_matmul),
    .cfg_done (matmul_done),
    .cfg_busy (matmul_busy),
    .a_rd_addr(a_rd_addr),
    .a_rd_en  (a_rd_en),
    .a_rd_data(inf_rdata_a),    // scratch port A feeds A operand
    .b_rd_addr(b_rd_addr),
    .b_rd_en  (b_rd_en),
    .b_rd_data(b_rd_data_mux),  // muxed: weight_ram (QKV/WOUT) or scratch port B (SCORES/VALUE)
    .c_wr_addr(c_wr_addr),
    .c_wr_en  (c_wr_en),
    .c_wr_data(c_wr_data)       // result written to inference_ram scratch
);

softmax_unit softmax(
    .clk      (clk),
    .rst_n    (~rst),        // softmax is active-low; our rst is active-high
    .in_valid (in_valid),
    .in_first (in_first),
    .in_data  (in_data),
    .out_valid(out_valid),
    .out_data (out_data),
    .busy     (sm_busy)
);

// ── State register ────────────────────────────────────────────────────────────
always_ff @(posedge clk or posedge rst) begin
    if (rst)
        state <= IDLE;
    else
        state <= next_state;
end

// ── Next-state logic ──────────────────────────────────────────────────────────
always_comb begin
    next_state = state;

    case (state)

        IDLE: begin
            if (valid_rx && rx_data == START_BYTE)
                next_state = WAIT_TYPE;
        end

        WAIT_TYPE: begin
            if (valid_rx) begin
                if (rx_data[7:4] == WEIGHT_TYPE_MASK)
                    next_state = STORING_WEIGHTS;
                else if (rx_data[7:4] == INFER_TYPE_MASK && rx_data[3:0] != 4'h0)
                    next_state = STORING_LIVE;
                else
                    next_state = IDLE;
            end
        end

        STORING_WEIGHTS: begin
            if (valid_rx && byte_phase == 1'b1 && word_count == 13'd4095)
                next_state = IDLE;   // exit on the final low byte, not the gap before it
        end

        STORING_LIVE: begin
            if (valid_rx && byte_phase == 1'b1 && word_count == 13'd1023)
                next_state = LOADING_MMUL_MATRICES;
        end

        LOADING_MMUL_MATRICES: begin
            // mm_phase names which matmul we're setting up; go run the matching state.
            case (mm_phase)
                PH_Q, PH_K, PH_V: next_state = QKV_MMUL;
                PH_SCORES:        next_state = SCORES_MMUL;
                PH_VALUE:         next_state = VALUE_MMUL;
                PH_WOUT:          next_state = WOUT_MMUL;
                default:          next_state = QKV_MMUL;
            endcase
        end

        QKV_MMUL: begin
            if (matmul_done) begin
                next_state = LOADING_MMUL_MATRICES; // loop back for next
            end
        end

        SCORES_MMUL: begin
            if (matmul_done) next_state = SOFTMAX_FEED_ADDR;
        end

        SOFTMAX_FEED_ADDR: next_state = SOFTMAX_FEED_DATA;   // addr on pin -> data next cycle

        SOFTMAX_FEED_DATA: begin
            if (sm_col == 4'd15)
                next_state = SOFTMAX_COLLECT;    // fed all 16 elements of this row
            else
                next_state = SOFTMAX_FEED_ADDR;  // go grab the next element
        end

        SOFTMAX_COLLECT: begin
            if (out_valid && sm_ocol == 4'd15) begin   // just wrote the row's last result
                if (sm_row == 4'd15)
                    next_state = LOADING_MMUL_MATRICES;   // all 16 rows done -> LOADING (mm_phase now PH_VALUE)
                else
                    next_state = SOFTMAX_FEED_ADDR;      // start the next row
            end
        end

        VALUE_MMUL: begin
            if (matmul_done) next_state = LOADING_MMUL_MATRICES;   // -> LOADING sets up WOUT
        end

        WOUT_MMUL: begin
            if (matmul_done) next_state = INITIALIZE_SEND;
        end

        INITIALIZE_SEND: begin
            if (tx_ready) next_state = SENDING_LIVE;   // 0xAA accepted; word 0 data is ready
        end

        LOADING_SEND: next_state = SENDING_LIVE;       // address settled -> data valid next state

        SENDING_LIVE: begin
            if (tx_ready && byte_phase == 1'b1) begin  // low byte is going out now
                if (word_count == 13'd1023) next_state = SENDING_CHECKSUM;  // last word
                else                        next_state = LOADING_SEND;      // fetch next word
            end
        end

        SENDING_CHECKSUM: begin
            // The checksum byte is registered into tx_data/tx_send on this same cycle,
            // so uart_tx still picks it up after we've dropped back to IDLE.
            if (tx_ready) next_state = IDLE;
        end

        default: next_state = IDLE;

    endcase
end

// ── Output / datapath logic ───────────────────────────────────────────────────
always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        matrix_idx          <= 0;
        word_count          <= 0;
        held_high_byte      <= 0;
        byte_phase          <= 0;
        type_reg            <= 0;
        chk                 <= 0;
        layer               <= 0;
        live_waddr          <= 0;
        tx_data             <= 0;
        tx_send             <= 0;
        tx_pending          <= 0;
        mm_phase            <= PH_Q;
        sm_row              <= 0;
        sm_col              <= 0;
        sm_ocol             <= 0;
        b_base              <= 0;
        c_base              <= 0;
        weight_we           <= 0;
        inference_we        <= 0;
        start_matmul        <= 0;
        num_rows_a          <= 0;
        num_columns_b       <= 0;
        shared_dimension_ab <= 0;
        waddr               <= 0;
        wdata               <= 0;
    end else begin
        // Defaults -- cleared every cycle unless explicitly set below.
        // NOTE: mm_phase is NOT defaulted -- it's persistent state, advanced only
        // at each matmul/softmax completion.
        tx_send      <= 0;
        weight_we    <= 0;
        inference_we <= 0;
        start_matmul <= 0;

        // busy has risen -> uart_tx has committed to the byte, the send gap is closed.
        // Overridden below by any state that pulses tx_send this cycle.
        if (tx_busy) tx_pending <= 0;

        case (state)

            WAIT_TYPE: begin
                if (valid_rx) begin
                    type_reg   <= rx_data;
                    word_count <= 0;
                    byte_phase <= 0;
                    if (rx_data[7:4] == WEIGHT_TYPE_MASK)
                        matrix_idx <= rx_data[2:0];
                    else if (rx_data[7:4] == INFER_TYPE_MASK)
                        layer <= rx_data[1];   // 0x01 -> layer 0, 0x02 -> layer 1
                end
            end

            STORING_WEIGHTS: begin
                if (valid_rx) begin
                    waddr <= {matrix_idx, word_count[11:0]};
                    if (byte_phase == 0) begin
                        held_high_byte <= rx_data;
                        byte_phase     <= 1;
                    end else begin
                        wdata      <= {held_high_byte, rx_data};
                        weight_we  <= 1;
                        word_count <= word_count + 1;
                        byte_phase <= 0;
                    end
                end
                if (valid_rx && byte_phase == 1'b1 && word_count == 13'd4095 && tx_ready) begin
                    tx_data    <= ACK_BYTE;   // one pulse, on the final low byte, only if TX free
                    tx_send    <= 1;
                    tx_pending <= 1;
                end
            end

            STORING_LIVE: begin
                if (valid_rx) begin
                    if (byte_phase == 0) begin
                        held_high_byte <= rx_data;
                        byte_phase     <= 1;
                    end else begin
                        wdata        <= {held_high_byte, rx_data};
                        live_waddr   <= word_count[9:0];   // latch BEFORE the increment below
                        inference_we <= 1;
                        word_count   <= word_count + 1;
                        byte_phase   <= 0;
                    end
                end
            end

            LOADING_MMUL_MATRICES: begin
                // mm_phase selects the matmul. Each sets dims + write dest (c_base);
                // scratch A/B bases are combinational (a_base / b_base_inf). b_base
                // here is the WEIGHT-side base, used only when B is a weight (QKV, WOUT).
                // Weight bases add 0x4000 for layer 2 (layer==1); scratch c_base is the
                // same for both layers (scratch is reused one inference at a time).
                case (mm_phase)
                    PH_Q: begin  // X x W_q -> Q
                        num_rows_a <= 7'd16; num_columns_b <= 7'd64; shared_dimension_ab <= 7'd64;
                        b_base <= 15'h0000 + (layer ? 15'h4000 : 15'h0000);
                        c_base <= 13'h0400;
                    end
                    PH_K: begin  // X x W_k -> K
                        num_rows_a <= 7'd16; num_columns_b <= 7'd64; shared_dimension_ab <= 7'd64;
                        b_base <= 15'h1000 + (layer ? 15'h4000 : 15'h0000);
                        c_base <= 13'h0800;
                    end
                    PH_V: begin  // X x W_v -> V
                        num_rows_a <= 7'd16; num_columns_b <= 7'd64; shared_dimension_ab <= 7'd64;
                        b_base <= 15'h2000 + (layer ? 15'h4000 : 15'h0000);
                        c_base <= 13'h0C00;
                    end
                    PH_SCORES: begin  // Q x K^T -> scores
                        num_rows_a <= 7'd16; num_columns_b <= 7'd16; shared_dimension_ab <= 7'd64;
                        c_base <= 13'h1000;
                    end
                    PH_VALUE: begin  // attn x V -> context
                        num_rows_a <= 7'd16; num_columns_b <= 7'd64; shared_dimension_ab <= 7'd16;
                        c_base <= 13'h1200;
                    end
                    PH_WOUT: begin  // context x W_out -> output
                        num_rows_a <= 7'd16; num_columns_b <= 7'd64; shared_dimension_ab <= 7'd64;
                        b_base <= 15'h3000 + (layer ? 15'h4000 : 15'h0000);
                        c_base <= 13'h1600;
                    end
                endcase
                start_matmul <= 1;
            end

            QKV_MMUL: begin
                // advance Q -> K -> V -> SCORES (plain increment)
                if (matmul_done) mm_phase <= mm_phase + 1;
            end

            SCORES_MMUL: begin
                if (matmul_done) begin
                    // entering softmax -> start at row 0, element 0
                    // (phase stays PH_SCORES until softmax finishes)
                    sm_row  <= 0;
                    sm_col  <= 0;
                    sm_ocol <= 0;
                end
            end

            SOFTMAX_FEED_ADDR: ;   // address driven combinationally (sm_rd_addr) -- nothing to do

            SOFTMAX_FEED_DATA: begin
                // inf_rdata_a holds scores[sm_row][sm_col]; in_valid (comb) feeds it now
                sm_col <= sm_col + 1;   // advance to next element
            end

            SOFTMAX_COLLECT: begin
                if (out_valid) begin
                    sm_ocol <= sm_ocol + 1;   // write to attn[row][col] happens via we=out_valid
                    if (sm_ocol == 4'd15 && sm_row != 4'd15) begin
                        sm_row  <= sm_row + 1;   // move to next row
                        sm_col  <= 0;            // reset counters for it
                        sm_ocol <= 0;
                    end
                    else if (sm_ocol == 4'd15 && sm_row == 4'd15)
                        mm_phase <= PH_VALUE;   // softmax done -> set up attn x V
                end
            end

            // NOTE: {8'b0, out_data} reads the softmax's 0..255 output as a Q8.8 fraction,
            // i.e. divides by 256 where the LUT scaled by 255 -- all weights land 0.4% low.
            // Left uncorrected on purpose: the LUT's own bucketing error is ~±4% and is
            // signed per-distribution, so a uniform 0.4% correction is a wash. Measured.

            VALUE_MMUL: begin
                if (matmul_done) mm_phase <= PH_WOUT;   // -> LOADING sets up context x W_out
            end

            // Now we need to do the final mmul:

            WOUT_MMUL: begin
                // output lands in scratch @ 0x1600; kick off the send
                if (matmul_done) begin
                    mm_phase   <= PH_Q;   // reset phase for the next inference
                    word_count <= 0;      // send word index
                    byte_phase <= 0;
                    chk        <= 0;      // checksum accumulator
                end
            end

            INITIALIZE_SEND: begin
                // send the 0xAA start byte; word 0's address is already on the read pin
                // (driven combinationally from word_count = 0)
                if (tx_ready) begin
                    tx_data    <= START_BYTE;
                    tx_send    <= 1;
                    tx_pending <= 1;
                end
            end

            LOADING_SEND: ;   // address settling for the next word -- nothing to do

            SENDING_LIVE: begin
                // inf_rdata_a holds output[word_count]; send high byte then low byte
                if (tx_ready) begin
                    if (byte_phase == 0) begin
                        tx_data    <= inf_rdata_a[15:8];        // high byte first
                        tx_send    <= 1;
                        tx_pending <= 1;
                        chk        <= chk ^ inf_rdata_a[15:8];
                        byte_phase <= 1;
                    end else begin
                        tx_data    <= inf_rdata_a[7:0];         // low byte
                        tx_send    <= 1;
                        tx_pending <= 1;
                        chk        <= chk ^ inf_rdata_a[7:0];
                        byte_phase <= 0;
                        word_count <= word_count + 1;
                    end
                end
            end

            SENDING_CHECKSUM: begin
                if (tx_ready) begin
                    tx_data    <= chk;
                    tx_send    <= 1;
                    tx_pending <= 1;
                end
            end

            IDLE:                  ;
            default:               ;

        endcase
    end
end

endmodule
