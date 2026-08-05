// Same idea as tc_mmul_correct, but with arbitrary W_v and W_out instead of
// identity -- which is what actually puts the weight-matrix indexing on trial.
//
// Identity is symmetric and sparse: if the RTL read a weight matrix transposed,
// or indexed it column-major, identity would behave identically and
// tc_mmul_correct would still pass. Random matrices have no such symmetry, so
// any layout error shows up as hundreds of wrong words.
//
// W_q and W_k stay zero, so scores are still all-zero and softmax is still
// uniform at 15 (see the derivation in tc_mmul_correct.sv). That keeps the
// prediction exact integer arithmetic with no LUT to replicate.
//
// ── The model, mirroring the hardware stage by stage ─────────────────────────
//   V[k][j]       = ( SUM_m  X[k][m]   * W_v[m][j]   ) >>> 8
//   context[i][j] = ( SUM_k  15        * V[k][j]     ) >>> 8     same for all i
//   output[i][j]  = ( SUM_m  context[m]* W_out[m][j] ) >>> 8
//
// Each stage accumulates in 32 bits and truncates once at the end, matching
// tile_controller's ACC_WIDTH=32 accumulator and its c_wr_data[23:8] writeback.
// Intermediate results are stored as 16-bit Q8.8 in scratch, so the >>> 8 after
// every stage is real, not an approximation of the hardware.
//
// ── Magnitude budget ─────────────────────────────────────────────────────────
// Everything is capped at +-64 raw (+-0.25 in Q8.8) so that even the worst case
// where every term shares a sign stays inside signed 16 bits:
//   V       <= 64*64*64/256          = 1024
//   context <= 15*16*1024/256        = 960
//   output  <= 64*960*64/256         = 15360   < 32767
// Overflow here would look like an RTL bug when it is really a test bug.

class tc_mmul_random extends tpu_base_test;
    `uvm_component_utils(tc_mmul_random)

    virtual weight_bkdr_if wbkdr;

    // Word bases inside weight_bram for layer 1 -- attention_fsm.sv:516-539
    const int W_V_BASE   = 32'h2000;
    const int W_OUT_BASE = 32'h3000;

    const int TOKENS = 16;
    const int DIM    = 64;

    // What softmax_unit emits when all SEQ_LEN scores are equal.
    const int UNIFORM_ATTN = 15;

    const int MAG = 64;              // raw magnitude cap, +-0.25 in Q8.8

    const time RESULT_TIMEOUT = 100ms;

    function new(string name = "tc_mmul_random", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        if (!uvm_config_db #(virtual weight_bkdr_if)::get(this, "", "wbkdr", wbkdr))
            `uvm_fatal(get_type_name(), "no weight backdoor interface found")
    endfunction

    task run_phase(uvm_phase phase);
        bit signed [15:0] x    [16][64];     // input tokens
        bit signed [15:0] wv   [64][64];     // W_v
        bit signed [15:0] wo   [64][64];     // W_out
        bit signed [15:0] v    [16][64];     // V = X * W_v
        bit signed [15:0] ctx  [64];         // one context row (all rows equal)
        bit signed [15:0] outv [64];         // one output row

        bit [7:0] payload [];
        int       acc;
        int       w;

        phase.raise_objection(this);

        // ── Weights ──────────────────────────────────────────────────────────
        wbkdr.fill_all(16'h0000);            // W_q and W_k stay zero

        foreach (wv[r, c]) wv[r][c] = $urandom_range(0, 2*MAG) - MAG;
        foreach (wo[r, c]) wo[r][c] = $urandom_range(0, 2*MAG) - MAG;

        // B-operand address is row*64 + col (tile_controller: row * MAX_N + col).
        for (int r = 0; r < DIM; r++) begin
            for (int c = 0; c < DIM; c++) begin
                wbkdr.write_word(W_V_BASE   + r*DIM + c, wv[r][c]);
                wbkdr.write_word(W_OUT_BASE + r*DIM + c, wo[r][c]);
            end
        end

        // ── Input ────────────────────────────────────────────────────────────
        foreach (x[k, j]) x[k][j] = $urandom_range(0, 2*MAG) - MAG;

        payload = new[TOKENS * DIM * 2];
        for (int k = 0; k < TOKENS; k++) begin
            for (int j = 0; j < DIM; j++) begin
                w = k*DIM + j;
                payload[2*w]     = x[k][j][15:8];    // high byte first
                payload[2*w + 1] = x[k][j][7:0];
            end
        end

        // ── Stage 1: V = X * W_v ─────────────────────────────────────────────
        for (int k = 0; k < TOKENS; k++) begin
            for (int j = 0; j < DIM; j++) begin
                acc = 0;
                for (int m = 0; m < DIM; m++)
                    acc += x[k][m] * wv[m][j];
                v[k][j] = acc >>> 8;
            end
        end

        // ── Stage 2: context = attn * V, attn uniform at 15 ──────────────────
        // Identical for every output row, so compute one row.
        for (int j = 0; j < DIM; j++) begin
            acc = 0;
            for (int k = 0; k < TOKENS; k++)
                acc += UNIFORM_ATTN * v[k][j];
            ctx[j] = acc >>> 8;
        end

        // ── Stage 3: output = context * W_out ────────────────────────────────
        for (int j = 0; j < DIM; j++) begin
            acc = 0;
            for (int m = 0; m < DIM; m++)
                acc += ctx[m] * wo[m][j];
            outv[j] = acc >>> 8;
        end

        // Same row repeated for all 16 tokens.
        for (int i = 0; i < TOKENS; i++)
            for (int j = 0; j < DIM; j++)
                env.sb.expected_result.push_back(outv[j]);

        // ── Go ───────────────────────────────────────────────────────────────
        send_frame(8'h01, payload);

        fork
            begin
                wait (env.sb.tx_frame_done == 1'b1);
                `uvm_info(get_type_name(), "result frame arrived", UVM_LOW)
            end
            begin
                #RESULT_TIMEOUT;
                `uvm_error(get_type_name(),
                    $sformatf("no result frame after %0t -- got %0d of %0d bytes",
                              RESULT_TIMEOUT, env.sb.tx_bytes.size(), 2050))
            end
        join_any
        disable fork;

        phase.drop_objection(this);
    endtask

endclass
