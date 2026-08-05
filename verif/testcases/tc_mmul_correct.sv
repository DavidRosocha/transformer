// Real arithmetic, with an expected answer derived by hand rather than from a
// golden model.
//
// ── The weight choice ────────────────────────────────────────────────────────
//   W_q = 0, W_k = 0        -> Q = 0, K = 0 -> scores = Q*K' = 0
//   W_v = identity          -> V = X
//   W_out = identity        -> output = context
//
// Sixteen equal scores make softmax uniform, so context is a weighted sum of
// all sixteen input rows -- the same value in every output row.
//
// ── Why 15 and not 16 ────────────────────────────────────────────────────────
// With every score equal, softmax_unit produces 15, not the ideal 16:
//   diff = 0 -> addr = 0 -> ex_buf[i] = lut_exp[0] = 255
//   sigma = 16*255 = 4080 = 0xFF0 -> sigma_idx = 0xF
//   ex_idx = 0xFF[7:4] = 0xF
//   out = lut_2d_flat[{15,15}] = lut_2d_flat[255] = 15
// So the sixteen attention weights sum to 240/256 = 0.9375, not 1.0. That is
// the 2D-LUT approximation, not a bug -- but the prediction has to use 15 or
// every word comes out ~6% off.
//
// ── Both remaining matmuls are exact ─────────────────────────────────────────
// Identity in Q8.8 is 0x0100 = 1.0, so V = X*I and output = context*I both
// multiply by 256 then shift right 8 -- no rounding error introduced.
//
// Leaving, in raw signed integers:
//   expected[i][j] = (15 * SUM_k X[k][j]) >>> 8      identical for every row i
//
// >>> is an arithmetic shift, matching c_wr_data[23:8] in tile_controller,
// which floors toward negative infinity for negative accumulators.

class tc_mmul_correct extends tpu_base_test;
    `uvm_component_utils(tc_mmul_correct)

    virtual weight_bkdr_if wbkdr;

    // Word bases inside weight_bram for layer 1 -- attention_fsm.sv:516-539
    const int W_V_BASE   = 32'h2000;
    const int W_OUT_BASE = 32'h3000;

    const int TOKENS = 16;
    const int DIM    = 64;

    // What softmax_unit emits when all SEQ_LEN scores are equal.
    const int UNIFORM_ATTN = 15;

    const time RESULT_TIMEOUT = 100ms;

    function new(string name = "tc_mmul_correct", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        if (!uvm_config_db #(virtual weight_bkdr_if)::get(this, "", "wbkdr", wbkdr))
            `uvm_fatal(get_type_name(), "no weight backdoor interface found")
    endfunction

    // 64x64 identity in Q8.8. Symmetric, so it does not matter whether the
    // matmul reads it row-major or column-major -- one less thing to get wrong.
    function void load_identity(int base);
        for (int r = 0; r < DIM; r++)
            for (int c = 0; c < DIM; c++)
                wbkdr.write_word(base + r*DIM + c, (r == c) ? 16'h0100 : 16'h0000);
    endfunction

    task run_phase(uvm_phase phase);
        bit signed [15:0] x [16][64];       // the input tokens, Q8.8
        bit [7:0]         payload [];
        int               w;
        int               colsum;
        int               prod;
        bit signed [15:0] exp_word;

        phase.raise_objection(this);

        // ── Weights ──────────────────────────────────────────────────────────
        wbkdr.fill_all(16'h0000);           // W_q and W_k stay zero
        load_identity(W_V_BASE);
        load_identity(W_OUT_BASE);

        // ── Input ────────────────────────────────────────────────────────────
        // Range +-511 raw = +-2.0 in Q8.8. Small enough that 15*colsum cannot
        // overflow a Q8.8 output, large enough that signs and truncation both
        // get exercised.
        foreach (x[k, j])
            x[k][j] = $urandom_range(0, 1022) - 511;

        payload = new[TOKENS * DIM * 2];
        for (int k = 0; k < TOKENS; k++) begin
            for (int j = 0; j < DIM; j++) begin
                w = k*DIM + j;
                payload[2*w]     = x[k][j][15:8];   // high byte first
                payload[2*w + 1] = x[k][j][7:0];
            end
        end

        // ── Prediction ───────────────────────────────────────────────────────
        // Every output row is identical, so the same 64 values repeat 16 times.
        for (int i = 0; i < TOKENS; i++) begin
            for (int j = 0; j < DIM; j++) begin
                colsum = 0;
                for (int k = 0; k < TOKENS; k++)
                    colsum += x[k][j];

                prod     = UNIFORM_ATTN * colsum;
                exp_word = prod >>> 8;          // arithmetic shift = floor
                env.sb.expected_result.push_back(exp_word);
            end
        end

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
