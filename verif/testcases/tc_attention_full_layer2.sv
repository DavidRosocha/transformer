// tc_attention_full, but for LAYER 2 -- frame type 0x02 instead of 0x01.
//
// The only functional difference in the DUT is attention_fsm.sv:516-539, which
// adds 0x4000 to every weight base when layer==1. That adder is unverified by
// every other test, and a wrong offset would mean layer 2 silently computes
// with layer 1's weights -- plausible-looking output, wrong answer.
//
// The weights here are written at 0x4000-0x7000 and NOWHERE ELSE, so the test
// only passes if the DUT applies the offset. Loading them at the layer-1 bases
// instead makes every result word come back 0x0000.
//
// Everything else matches tc_attention_full:
//   * Q = X*W_q and K = X*W_k with non-zero weights
//   * scores = Q*K' -- including the transposed-K read at attention_fsm.sv:166
//     (SCORES_MMUL: b_base_inf = 13'h0800, "B = K stored transposed")
//   * softmax on unequal inputs, hitting a spread of LUT addresses rather than
//     always addr=0
//   * output rows that DIFFER from each other
//
// The reference softmax reads the same .mem files as the RTL (softmax_ref_if),
// so the only thing under test is the algorithm and the wiring around it.
//
// ── The model, mirroring the hardware stage by stage ─────────────────────────
//   Q[k][j]       = ( SUM_m X[k][m] * W_q[m][j]   ) >>> 8
//   K[k][j]       = ( SUM_m X[k][m] * W_k[m][j]   ) >>> 8
//   scores[a][b]  = ( SUM_j Q[a][j] * K[b][j]     ) >>> 8      <- Q * K'
//   attn[a][*]    = softmax_row(scores[a][*])                   uint8, 0..255
//   V[k][j]       = ( SUM_m X[k][m] * W_v[m][j]   ) >>> 8
//   context[a][j] = ( SUM_k attn[a][k] * V[k][j]  ) >>> 8
//   out[a][j]     = ( SUM_m context[a][m] * W_o[m][j] ) >>> 8
//
// Every stage accumulates in 32 bits and truncates once, matching
// tile_controller's ACC_WIDTH=32 and its c_wr_data[23:8] writeback. Assigning
// an int to a 16-bit signed variable truncates exactly the way that bit slice
// does, so an accumulator that overflows wraps identically in both -- no
// spurious mismatch, just a less interesting test if it happens.

class tc_attention_full_layer2 extends tpu_base_test;
    `uvm_component_utils(tc_attention_full_layer2)

    virtual weight_bkdr_if  wbkdr;
    virtual softmax_ref_if  smref;

    // Word bases inside weight_bram -- attention_fsm.sv:516-539.
    // A layer-2 frame (type 0x02) sets layer=1, which adds 0x4000 to every
    // weight base, so the second set of four matrices lives at 0x4000-0x7000.
    // That adder is the whole point of this test: load the weights HERE and
    // the DUT must read them from HERE.
    const int LAYER2_OFFSET = 32'h4000;

    const int W_Q_BASE   = LAYER2_OFFSET + 32'h0000;
    const int W_K_BASE   = LAYER2_OFFSET + 32'h1000;
    const int W_V_BASE   = LAYER2_OFFSET + 32'h2000;
    const int W_OUT_BASE = LAYER2_OFFSET + 32'h3000;

    const int TOKENS = 16;
    const int DIM    = 64;

    // +-64 raw = +-0.25 in Q8.8. Chosen so typical scores land around +-0.7,
    // which spreads softmax LUT addresses over roughly 0..90 -- non-uniform
    // enough to be a real test, small enough that 16-bit intermediates hold.
    const int MAG = 64;

    const time RESULT_TIMEOUT = 100ms;

    function new(string name = "tc_attention_full_layer2", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        if (!uvm_config_db #(virtual weight_bkdr_if)::get(this, "", "wbkdr", wbkdr))
            `uvm_fatal(get_type_name(), "no weight backdoor interface found")
        if (!uvm_config_db #(virtual softmax_ref_if)::get(this, "", "smref", smref))
            `uvm_fatal(get_type_name(), "no softmax reference interface found")
    endfunction

    function void load_matrix(int base, const ref bit signed [15:0] m [64][64]);
        for (int r = 0; r < DIM; r++)
            for (int c = 0; c < DIM; c++)
                wbkdr.write_word(base + r*DIM + c, m[r][c]);   // row*64 + col
    endfunction

    task run_phase(uvm_phase phase);
        bit signed [15:0] x  [16][64];
        bit signed [15:0] wq [64][64], wk [64][64], wv [64][64], wo [64][64];
        bit signed [15:0] q  [16][64], k  [16][64], v  [16][64];
        bit signed [15:0] sc [16][16];
        bit signed [15:0] ctx[16][64];
        bit signed [15:0] o  [16][64];

        int row_scores [16];
        int row_probs  [16];
        int attn [16][16];

        bit [7:0] payload [];
        int       acc;
        int       w;
        int       addr_lo = 999, addr_hi = 0;   // observed softmax LUT spread

        phase.raise_objection(this);

        // ── Weights ──────────────────────────────────────────────────────────
        wbkdr.fill_all(16'h0000);
        foreach (wq[r, c]) wq[r][c] = $urandom_range(0, 2*MAG) - MAG;
        foreach (wk[r, c]) wk[r][c] = $urandom_range(0, 2*MAG) - MAG;
        foreach (wv[r, c]) wv[r][c] = $urandom_range(0, 2*MAG) - MAG;
        foreach (wo[r, c]) wo[r][c] = $urandom_range(0, 2*MAG) - MAG;

        load_matrix(W_Q_BASE,   wq);
        load_matrix(W_K_BASE,   wk);
        load_matrix(W_V_BASE,   wv);
        load_matrix(W_OUT_BASE, wo);

        // ── Input ────────────────────────────────────────────────────────────
        foreach (x[t, j]) x[t][j] = $urandom_range(0, 2*MAG) - MAG;

        payload = new[TOKENS * DIM * 2];
        for (int t = 0; t < TOKENS; t++) begin
            for (int j = 0; j < DIM; j++) begin
                w = t*DIM + j;
                payload[2*w]     = x[t][j][15:8];    // high byte first
                payload[2*w + 1] = x[t][j][7:0];
            end
        end

        // ── Q, K, V ──────────────────────────────────────────────────────────
        for (int t = 0; t < TOKENS; t++) begin
            for (int j = 0; j < DIM; j++) begin
                acc = 0;
                for (int m = 0; m < DIM; m++) acc += x[t][m] * wq[m][j];
                q[t][j] = acc >>> 8;

                acc = 0;
                for (int m = 0; m < DIM; m++) acc += x[t][m] * wk[m][j];
                k[t][j] = acc >>> 8;

                acc = 0;
                for (int m = 0; m < DIM; m++) acc += x[t][m] * wv[m][j];
                v[t][j] = acc >>> 8;
            end
        end

        // ── scores = Q * K' ──────────────────────────────────────────────────
        for (int a = 0; a < TOKENS; a++) begin
            for (int b = 0; b < TOKENS; b++) begin
                acc = 0;
                for (int j = 0; j < DIM; j++) acc += q[a][j] * k[b][j];
                sc[a][b] = acc >>> 8;
            end
        end

        // ── softmax, one row at a time ───────────────────────────────────────
        for (int a = 0; a < TOKENS; a++) begin
            int mx = sc[a][0];
            for (int b = 0; b < TOKENS; b++) row_scores[b] = sc[a][b];
            smref.softmax_row(row_scores, row_probs);
            for (int b = 0; b < TOKENS; b++) attn[a][b] = row_probs[b];

            // Track how far the LUT addresses actually spread, so a run that
            // accidentally produced near-uniform scores is visible rather than
            // quietly passing as a weaker test than intended.
            for (int b = 0; b < TOKENS; b++) if (sc[a][b] > mx) mx = sc[a][b];
            for (int b = 0; b < TOKENS; b++) begin
                int addr = (mx - sc[a][b]) >> 2;
                if (addr > 255) addr = 255;
                if (addr < addr_lo) addr_lo = addr;
                if (addr > addr_hi) addr_hi = addr;
            end
        end

        `uvm_info(get_type_name(),
            $sformatf("softmax LUT addresses spanned %0d..%0d (0 would mean uniform)",
                      addr_lo, addr_hi), UVM_LOW)

        // ── context = attn * V, then output = context * W_out ────────────────
        for (int a = 0; a < TOKENS; a++) begin
            for (int j = 0; j < DIM; j++) begin
                acc = 0;
                for (int t = 0; t < TOKENS; t++) acc += attn[a][t] * v[t][j];
                ctx[a][j] = acc >>> 8;
            end
        end

        for (int a = 0; a < TOKENS; a++) begin
            for (int j = 0; j < DIM; j++) begin
                acc = 0;
                for (int m = 0; m < DIM; m++) acc += ctx[a][m] * wo[m][j];
                o[a][j] = acc >>> 8;
            end
        end

        for (int a = 0; a < TOKENS; a++)
            for (int j = 0; j < DIM; j++)
                env.sb.expected_result.push_back(o[a][j]);

        // ── Go ───────────────────────────────────────────────────────────────
        send_frame(8'h02, payload);

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
