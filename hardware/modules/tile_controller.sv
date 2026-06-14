// =============================================================
// tile_controller.sv — Tiling controller for large matrix multiplications
//
// Breaks an M×K * K×N multiplication into 8×8 tiles and feeds
// them sequentially through a single systolic_array_8x8 instance.
// Accumulates partial sums across the K-dimension.
//
// Supported dimensions (must be multiples of 8):
//   - 16×64 * 64×N   (transformer QK^T, QKV, etc.)
//   - 64×64 * 64×64  (FFN layers)
//
// Memory interface: simple dual-port BRAM-style read for A/B,
// write port for C results.
// =============================================================
module tile_controller #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 32,
    parameter TILE_SIZE  = 8,
    parameter MAX_M      = 64,  // max rows of A
    parameter MAX_N      = 64,  // max cols of B
    parameter MAX_K      = 64   // shared dimension
)(
    input  logic clk,
    input  logic rst,
 
    // --- Configuration (active dimensions, must be multiples of 8) ---
    input  logic [$clog2(MAX_M):0] cfg_m,
    input  logic [$clog2(MAX_N):0] cfg_n,
    input  logic [$clog2(MAX_K):0] cfg_k,
    input  logic                    cfg_start,
    output logic                    cfg_done,
    output logic                    cfg_busy,
 
    // --- A matrix read port (row-major storage) ---
    // Address = row * MAX_K + col
    output logic [$clog2(MAX_M*MAX_K)-1:0] a_rd_addr,
    output logic                            a_rd_en,
    input  logic signed [DATA_WIDTH-1:0]   a_rd_data,
 
    // --- B matrix read port (row-major storage) ---
    // Address = row * MAX_N + col
    output logic [$clog2(MAX_K*MAX_N)-1:0] b_rd_addr,
    output logic                            b_rd_en,
    input  logic signed [DATA_WIDTH-1:0]   b_rd_data,
 
    // --- C matrix write port (row-major storage) ---
    // Address = row * MAX_N + col
    output logic [$clog2(MAX_M*MAX_N)-1:0] c_wr_addr,
    output logic                            c_wr_en,
    output logic signed [ACC_WIDTH-1:0]    c_wr_data
);
 
    // ---------------------------------------------------------------
    // Local parameters
    // ---------------------------------------------------------------
    localparam M_TILES_MAX = MAX_M / TILE_SIZE;
    localparam N_TILES_MAX = MAX_N / TILE_SIZE;
    localparam K_TILES_MAX = MAX_K / TILE_SIZE;
    localparam COMPUTE_CYCLES = 3 * TILE_SIZE - 2;  // 22 for TILE_SIZE=8
 
    // ---------------------------------------------------------------
    // Tile iterators
    // ---------------------------------------------------------------
    logic [$clog2(M_TILES_MAX)-1:0] m_tile;
    logic [$clog2(N_TILES_MAX)-1:0] n_tile;
    logic [$clog2(K_TILES_MAX)-1:0] k_tile;
 
    logic [$clog2(M_TILES_MAX):0] m_tiles;  // cfg_m / TILE_SIZE (needs extra bit for max value)
    logic [$clog2(N_TILES_MAX):0] n_tiles;  // cfg_n / TILE_SIZE
    logic [$clog2(K_TILES_MAX):0] k_tiles;  // cfg_k / TILE_SIZE
 
    // ---------------------------------------------------------------
    // Tile buffers (loaded from memory before compute)
    // A tile: 8 rows × 8 cols, B tile: 8 rows × 8 cols
    // ---------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] a_tile_buf [0:TILE_SIZE-1][0:TILE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] b_tile_buf [0:TILE_SIZE-1][0:TILE_SIZE-1];
 
    // Accumulation buffer for one output tile (8×8)
    logic signed [ACC_WIDTH-1:0] c_acc_buf [0:TILE_SIZE-1][0:TILE_SIZE-1];
 
    // ---------------------------------------------------------------
    // Systolic array interface signals
    // ---------------------------------------------------------------
    logic signed [TILE_SIZE*DATA_WIDTH-1:0] sa_a_in;
    logic signed [TILE_SIZE*DATA_WIDTH-1:0] sa_b_in;
    logic signed [TILE_SIZE*TILE_SIZE*ACC_WIDTH-1:0] sa_acc_out;
    logic sa_rst;
    logic sa_done;
 
    // ---------------------------------------------------------------
    // Systolic array instance
    // ---------------------------------------------------------------
    systolic_array_8x8 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .SIZE      (TILE_SIZE)
    ) u_sa (
        .clk    (clk),
        .rst    (sa_rst),
        .a_in   (sa_a_in),
        .b_in   (sa_b_in),
        .acc_out(sa_acc_out),
        .done   (sa_done)
    );
 
    // ---------------------------------------------------------------
    // FSM
    // ---------------------------------------------------------------
    typedef enum logic [3:0] {
        IDLE,
        LOAD_TILE,
        FEED_SA,
        WAIT_SA,
        ACCUMULATE,
        STORE_RESULT,
        NEXT_K,
        NEXT_N,
        NEXT_M,
        DONE
    } state_t;
 
    state_t state;
 
    // Sub-counters for load/feed/store phases
    logic [$clog2(TILE_SIZE)-1:0] row_cnt;
    logic [$clog2(TILE_SIZE)-1:0] col_cnt;
    logic [$clog2(TILE_SIZE)-1:0] feed_cnt;
    logic [$clog2(TILE_SIZE*TILE_SIZE)-1:0] store_cnt;
 
    // Base addresses for current tile
    logic [$clog2(MAX_M*MAX_K)-1:0] a_base;
    logic [$clog2(MAX_K*MAX_N)-1:0] b_base;
    logic [$clog2(MAX_M*MAX_N)-1:0] c_base;
 
    // ---------------------------------------------------------------
    // FSM sequential
    // ---------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            m_tile    <= '0;
            n_tile    <= '0;
            k_tile    <= '0;
            row_cnt   <= '0;
            col_cnt   <= '0;
            feed_cnt  <= '0;
            store_cnt <= '0;
            m_tiles   <= '0;
            n_tiles   <= '0;
            k_tiles   <= '0;
            cfg_done  <= 1'b0;
            cfg_busy  <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    cfg_done <= 1'b0;
                    if (cfg_start) begin
                        m_tiles  <= cfg_m[$clog2(MAX_M):$clog2(TILE_SIZE)];
                        n_tiles  <= cfg_n[$clog2(MAX_N):$clog2(TILE_SIZE)];
                        k_tiles  <= cfg_k[$clog2(MAX_K):$clog2(TILE_SIZE)];
                        m_tile   <= '0;
                        n_tile   <= '0;
                        k_tile   <= '0;
                        cfg_busy <= 1'b1;
                        state    <= LOAD_TILE;
                        row_cnt  <= '0;
                        col_cnt  <= '0;
                    end
                end
 
                LOAD_TILE: begin
                    // Load both A and B tiles from memory (one element per cycle)
                    // Takes TILE_SIZE * TILE_SIZE cycles.
                    // Data capture is handled by a separate pipeline block below
                    // to account for 1-cycle memory read latency.
                    if (col_cnt == TILE_SIZE - 1) begin
                        col_cnt <= '0;
                        if (row_cnt == TILE_SIZE - 1) begin
                            row_cnt  <= '0;
                            feed_cnt <= '0;
                            state    <= FEED_SA;
                        end else begin
                            row_cnt <= row_cnt + 1;
                        end
                    end else begin
                        col_cnt <= col_cnt + 1;
                    end
                end
 
                FEED_SA: begin
                    // Feed one column of A-tile / one row of B-tile per cycle
                    if (feed_cnt == TILE_SIZE - 1) begin
                        state <= WAIT_SA;
                    end else begin
                        feed_cnt <= feed_cnt + 1;
                    end
                end
 
                WAIT_SA: begin
                    if (sa_done) begin
                        state     <= ACCUMULATE;
                        store_cnt <= '0;
                    end
                end
 
                ACCUMULATE: begin
                    // Add systolic array results to accumulation buffer
                    // Process all 64 elements (8×8)
                    if (store_cnt == TILE_SIZE * TILE_SIZE - 1) begin
                        state <= NEXT_K;
                    end else begin
                        store_cnt <= store_cnt + 1;
                    end
                end
 
                NEXT_K: begin
                    if (k_tile == k_tiles - 1) begin
                        // All k-tiles done for this output tile — store result
                        k_tile    <= '0;
                        store_cnt <= '0;
                        state     <= STORE_RESULT;
                    end else begin
                        k_tile  <= k_tile + 1;
                        row_cnt <= '0;
                        col_cnt <= '0;
                        state   <= LOAD_TILE;
                    end
                end
 
                STORE_RESULT: begin
                    // Write accumulated results to C memory
                    if (store_cnt == TILE_SIZE * TILE_SIZE - 1) begin
                        state <= NEXT_N;
                    end else begin
                        store_cnt <= store_cnt + 1;
                    end
                end
 
                NEXT_N: begin
                    if (n_tile == n_tiles - 1) begin
                        n_tile <= '0;
                        state  <= NEXT_M;
                    end else begin
                        n_tile  <= n_tile + 1;
                        row_cnt <= '0;
                        col_cnt <= '0;
                        state   <= LOAD_TILE;
                    end
                    // Clear accumulation buffer for next output tile
                end
 
                NEXT_M: begin
                    if (m_tile == m_tiles - 1) begin
                        state <= DONE;
                    end else begin
                        m_tile  <= m_tile + 1;
                        row_cnt <= '0;
                        col_cnt <= '0;
                        state   <= LOAD_TILE;
                    end
                end
 
                DONE: begin
                    cfg_done <= 1'b1;
                    cfg_busy <= 1'b0;
                    state    <= IDLE;
                end
 
                default: state <= IDLE;
            endcase
        end
    end
 
    // ---------------------------------------------------------------
    // Address generation
    // ---------------------------------------------------------------
    // A tile base: row = m_tile*8, col = k_tile*8
    // A[row][col] address = row * cfg_k + col
    // B tile base: row = k_tile*8, col = n_tile*8
    // B[row][col] address = row * cfg_n + col
 
    always_comb begin
        a_base = (m_tile * TILE_SIZE + row_cnt) * cfg_k + (k_tile * TILE_SIZE + col_cnt);
        b_base = (k_tile * TILE_SIZE + row_cnt) * cfg_n + (n_tile * TILE_SIZE + col_cnt);
        c_base = (m_tile * TILE_SIZE + store_cnt[$clog2(TILE_SIZE*TILE_SIZE)-1:$clog2(TILE_SIZE)])
                 * cfg_n
                 + (n_tile * TILE_SIZE + store_cnt[$clog2(TILE_SIZE)-1:0]);
    end
 
    assign a_rd_addr = a_base;
    assign b_rd_addr = b_base;
    assign a_rd_en   = (state == LOAD_TILE);
    assign b_rd_en   = (state == LOAD_TILE);
 
    // ---------------------------------------------------------------
    // Tile buffer capture — pipeline delay for 1-cycle memory latency
    // The address is presented on cycle N; the data arrives on cycle N+1.
    // We capture using delayed row/col indices so each element lands in
    // the correct buffer position.
    // ---------------------------------------------------------------
    logic [$clog2(TILE_SIZE)-1:0] cap_row_d, cap_col_d;
    logic cap_valid_d;
 
    always_ff @(posedge clk) begin
        if (rst) begin
            cap_row_d   <= '0;
            cap_col_d   <= '0;
            cap_valid_d <= 1'b0;
        end else begin
            cap_row_d   <= row_cnt;
            cap_col_d   <= col_cnt;
            cap_valid_d <= (state == LOAD_TILE);
        end
    end
 
    always_ff @(posedge clk) begin
        if (cap_valid_d) begin
            a_tile_buf[cap_row_d][cap_col_d] <= a_rd_data;
            b_tile_buf[cap_row_d][cap_col_d] <= b_rd_data;
        end
    end
 
    // ---------------------------------------------------------------
    // Systolic array input mux — feed column k of A-tile as rows,
    // row k of B-tile as columns
    // ---------------------------------------------------------------
    always_comb begin
        sa_a_in = '0;
        sa_b_in = '0;
        if (state == FEED_SA) begin
            for (int r = 0; r < TILE_SIZE; r++) begin
                sa_a_in[(r+1)*DATA_WIDTH-1 -: DATA_WIDTH] = a_tile_buf[r][feed_cnt];
            end
            for (int c = 0; c < TILE_SIZE; c++) begin
                sa_b_in[(c+1)*DATA_WIDTH-1 -: DATA_WIDTH] = b_tile_buf[feed_cnt][c];
            end
        end
    end
 
    // Keep SA in reset except during FEED/WAIT/ACCUMULATE phases.
    // This ensures the SA counter starts fresh when FEED_SA begins.
    assign sa_rst = rst || !(state == FEED_SA || state == WAIT_SA || state == ACCUMULATE);
 
    // ---------------------------------------------------------------
    // Accumulation logic
    // ---------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst || state == NEXT_N || state == NEXT_M) begin
            // Clear accumulation buffer for next output tile
            for (int r = 0; r < TILE_SIZE; r++)
                for (int c = 0; c < TILE_SIZE; c++)
                    c_acc_buf[r][c] <= '0;
        end else if (state == ACCUMULATE) begin
            // Add SA output to accumulation buffer
            for (int r = 0; r < TILE_SIZE; r++) begin
                for (int c = 0; c < TILE_SIZE; c++) begin
                    if (store_cnt == 0) begin
                        if (k_tile == 0)
                            c_acc_buf[r][c] <= signed'(sa_acc_out[((r*TILE_SIZE)+c+1)*ACC_WIDTH-1 -: ACC_WIDTH]);
                        else
                            c_acc_buf[r][c] <= c_acc_buf[r][c]
                                             + signed'(sa_acc_out[((r*TILE_SIZE)+c+1)*ACC_WIDTH-1 -: ACC_WIDTH]);
                    end
                end
            end
        end
    end
 
    // ---------------------------------------------------------------
    // C write port
    // ---------------------------------------------------------------
    logic [$clog2(TILE_SIZE)-1:0] wr_row, wr_col;
    assign wr_row = store_cnt[$clog2(TILE_SIZE*TILE_SIZE)-1:$clog2(TILE_SIZE)];
    assign wr_col = store_cnt[$clog2(TILE_SIZE)-1:0];
 
    assign c_wr_addr = c_base;
    assign c_wr_en   = (state == STORE_RESULT);
    assign c_wr_data = c_acc_buf[wr_row][wr_col];
 
endmodule