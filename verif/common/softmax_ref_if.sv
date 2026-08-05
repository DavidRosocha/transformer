// Reference model for softmax_unit's 2D-LUT approximation.
//
// This reads THE SAME two .mem files the RTL reads. That is deliberate: a
// hand-written or Python copy of the LUT is a second source of truth that can
// silently drift when the LUTs are regenerated. Sharing the files means the
// reference and the DUT cannot disagree about LUT contents -- only about the
// algorithm around them, which is the thing actually being checked.
//
// Lives in an interface rather than a class because tpu_pkg cannot hold
// hierarchical references, and because $readmemh wants a module-scope array.
//
// Algorithm mirrors softmax_unit.sv:
//   diff = running_max - score                          (unsigned 16-bit)
//   addr = min(diff >> 2, 255)                          (line 196)
//   ex[i] = lut_exp[addr]
//   sigma = SUM ex                                      (max 255*16 = 4080)
//   out[i] = lut_2d_flat[{sigma[11:8], ex[i][7:4]}]     (lines 216-220)

interface softmax_ref_if;

    localparam int SEQ_LEN = 16;

    logic [7:0] lut_exp [0:255];
    logic [7:0] lut_2d  [0:255];

    initial begin
        $readmemh("../../softmax/sim/luts/lut_exp.mem",     lut_exp);
        $readmemh("../../softmax/sim/luts/lut_2d_flat.mem", lut_2d);
        if (lut_exp[0] !== 8'hFF)
            $error("[softmax_ref] LUT load failed (lut_exp[0]=0x%02h)", lut_exp[0]);
    end

    // scores: one row of the 16x16 score matrix, raw signed Q8.8
    // probs:  the 16 uint8 attention weights the DUT should produce
    function automatic void softmax_row(const ref int scores [16],
                                              ref int probs  [16]);
        int          mx;
        bit   [15:0] diff;
        int          addr;
        int          ex [16];
        int          sigma;
        int          sigma_idx, ex_idx;

        mx = scores[0];
        for (int i = 1; i < SEQ_LEN; i++)
            if (scores[i] > mx) mx = scores[i];

        sigma = 0;
        for (int i = 0; i < SEQ_LEN; i++) begin
            diff  = 16'(mx - scores[i]);          // always >= 0, wraps to 16 bits
            addr  = (diff >> 2) > 255 ? 255 : (diff >> 2);
            ex[i] = lut_exp[addr];
            sigma += ex[i];
        end

        sigma_idx = (sigma >> 8) & 'hF;           // sigma[11:8]
        for (int i = 0; i < SEQ_LEN; i++) begin
            ex_idx   = (ex[i] >> 4) & 'hF;        // ex_buf[i][7:4]
            probs[i] = lut_2d[(sigma_idx << 4) | ex_idx];
        end
    endfunction

endinterface
