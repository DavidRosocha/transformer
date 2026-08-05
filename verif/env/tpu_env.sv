// Scoreboard + env.
//
// The scoreboard holds the checking that used to live in tc_uart_rx. Tests now
// only say WHAT they expect; the comparing happens here, once, for every test.

// A class can only have one write(). These macros mint imp types whose owner
// must instead provide write_rx() and write_tx(), which is how one scoreboard
// subscribes to two analysis ports.
`uvm_analysis_imp_decl(_rx)
`uvm_analysis_imp_decl(_tx)

class tpu_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(tpu_scoreboard)

    // Receiving ends of uart_monitor's two broadcasts.
    uvm_analysis_imp_rx #(uart_item, tpu_scoreboard) imp_rx;
    uvm_analysis_imp_tx #(uart_item, tpu_scoreboard) imp_tx;

    // What the wire should carry, in order. Seeded by the test before stimulus
    // starts, drained one byte at a time by write().
    bit [7:0]    expected [$];
    int unsigned n_checked;   // how many matched
    int unsigned byte_idx;    // position in the stream, counts every byte seen

    // ── TX side: the result frame the FPGA sends back ────────────────────────
    // Layout is [0xAA][2048 payload][XOR checksum] -- no type byte and no stop
    // byte, because the PC only ever receives one kind of frame.
    localparam int RESULT_PAYLOAD = 2048;
    localparam int RESULT_FRAME   = 1 + RESULT_PAYLOAD + 1;

    bit [7:0] tx_bytes [$];
    bit       tx_frame_done;       // polled by the test to know the reply landed
    bit       expect_zero_result;  // set by the test when all weights are zero

    // Optional word-level prediction. Left empty, only shape and checksum are
    // checked; filled by the test, every result word is compared too.
    bit signed [15:0] expected_result [$];

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        imp_rx = new("imp_rx", this);
        imp_tx = new("imp_tx", this);
    endfunction

    // ── Called by the test, before stimulus ──────────────────────────────────
    function void expect_byte(bit [7:0] b);
        expected.push_back(b);
    endfunction

    // ── Called by the monitor, once per byte on serial_rx ────────────────────
    function void write_rx(uart_item t);
        bit [7:0] exp;

        if (expected.size() == 0) begin
            `uvm_error(get_type_name(),
                $sformatf("saw byte 0x%02h but nothing more was expected",
                          t.data_byte))
            return;
        end

        exp = expected.pop_front();

        if (t.data_byte !== exp)
            `uvm_error(get_type_name(),
                $sformatf("byte %0d mismatch: expected 0x%02h, saw 0x%02h",
                          byte_idx, exp, t.data_byte))
        else
            n_checked++;

        byte_idx++;
    endfunction

    // ── Called by the monitor, once per byte on serial_tx ────────────────────
    function void write_tx(uart_item t);
        tx_bytes.push_back(t.data_byte);

        if (tx_bytes.size() == RESULT_FRAME) begin
            check_result_frame();
            tx_frame_done = 1;
        end
    endfunction

    // Shape and content checks on the reply, run once the full frame is in.
    function void check_result_frame();
        bit [7:0] checksum = 8'h00;
        int       nonzero  = 0;

        if (tx_bytes[0] !== 8'hAA)
            `uvm_error(get_type_name(),
                $sformatf("result frame started with 0x%02h, expected 0xAA",
                          tx_bytes[0]))

        for (int i = 1; i <= RESULT_PAYLOAD; i++) begin
            checksum ^= tx_bytes[i];
            if (tx_bytes[i] !== 8'h00) nonzero++;
        end

        if (checksum !== tx_bytes[RESULT_FRAME-1])
            `uvm_error(get_type_name(),
                $sformatf("result checksum mismatch: computed 0x%02h, received 0x%02h",
                          checksum, tx_bytes[RESULT_FRAME-1]))

        // With every weight zero, Q=K=V=0, so context = attn x V = 0 and the
        // output is zero for any input. The softmax approximation cancels out
        // entirely, which is what makes this checkable without a golden model.
        if (expect_zero_result && nonzero != 0)
            `uvm_error(get_type_name(),
                $sformatf("expected an all-zero result, but %0d of %0d payload bytes were non-zero",
                          nonzero, RESULT_PAYLOAD))

        check_result_values();

        `uvm_info(get_type_name(),
            $sformatf("result frame received: %0d payload bytes, checksum 0x%02h, %0d non-zero",
                      RESULT_PAYLOAD, checksum, nonzero), UVM_LOW)
    endfunction

    // Word-by-word comparison, when the test supplied a prediction.
    // Result words are big-endian, high byte first -- same packing as the
    // weight load (attention_fsm.sv, SENDING_LIVE).
    function void check_result_values();
        bit signed [15:0] got;
        int               errs = 0;

        if (expected_result.size() == 0) return;

        if (expected_result.size() != RESULT_PAYLOAD/2) begin
            `uvm_error(get_type_name(),
                $sformatf("prediction has %0d words, result frame carries %0d",
                          expected_result.size(), RESULT_PAYLOAD/2))
            return;
        end

        for (int w = 0; w < RESULT_PAYLOAD/2; w++) begin
            got = {tx_bytes[1 + 2*w], tx_bytes[2 + 2*w]};
            if (got !== expected_result[w]) begin
                errs++;
                if (errs <= 10)
                    `uvm_error(get_type_name(),
                        $sformatf("result word %0d (token %0d, dim %0d): expected %0d (0x%04h), got %0d (0x%04h)",
                                  w, w/64, w%64,
                                  expected_result[w], expected_result[w], got, got))
            end
        end

        if (errs == 0)
            `uvm_info(get_type_name(),
                $sformatf("all %0d result words match the prediction", RESULT_PAYLOAD/2),
                UVM_LOW)
        else
            `uvm_error(get_type_name(),
                $sformatf("%0d of %0d result words wrong", errs, RESULT_PAYLOAD/2))
    endfunction

    // A byte that never arrives never calls write_rx(), so comparing alone would
    // pass silently if the tail of a frame went missing. This catches it.
    virtual function void report_phase(uvm_phase phase);
        if (expected.size() != 0)
            `uvm_error(get_type_name(),
                $sformatf("%0d expected byte(s) never appeared on the wire",
                          expected.size()))
        `uvm_info(get_type_name(),
            $sformatf("%0d byte(s) checked OK", n_checked), UVM_LOW)
    endfunction

endclass


// Builds the agent and the scoreboard and wires them together. That single
// connect line is the entire reason this class exists -- without it, every
// test would repeat it.
class tpu_env extends uvm_env;
    `uvm_component_utils(tpu_env)

    uart_agent     agent;
    tpu_scoreboard sb;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        agent = uart_agent::type_id::create("agent", this);
        sb    = tpu_scoreboard::type_id::create("sb", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
        agent.monitor.ap.connect(sb.imp_rx);
        agent.monitor.ap_tx.connect(sb.imp_tx);
    endfunction

endclass
