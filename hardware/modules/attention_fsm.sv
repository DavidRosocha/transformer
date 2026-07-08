/*
Boot-time (once, on startup)
The PC sends 8 weight matrices into FPGA BRAM. These never change again:

W_q, W_k, W_v, W_out — for attention layer 1
W_q, W_k, W_v, W_out — for attention layer 2
Each is 64×64 × 2 bytes = 8,192 bytes. Total: ~64 KB loaded once.
*/


module attention_fsm #(
// Parameters here
)(
    input logic clk
    input logic rst

    input [7:0] serial_rx // pulled from uart_rx
    input logic valid_rx


    output [7:0] serial_tx // sent to uart_tx
)

// Local Parameters
typedef enum logic[3:0] {
    IDLE,
    WAIT_TYPE,
    STORING_WEIGHTS,
    STORING_LIVE,

    LOADING_MMUL_MATRICES,
    QKV_MMUL,
    SCORES_MMUL,
    SOFTMAX,
    VALUE_MMUL,
    WOUT_MMUL,

    SENDING_OUTPUT
 } state_t;

 state_t state, next_state;
 
// Registers

reg [15:0] base_addr;
reg [11:0] word_count = 0;
// The address you actually feed weight_bram.waddr is just base_addr + word_count — combinational, not its own register.

reg [15:0] held_high_byte = 0;
reg byte_phase = 0; // 0 (expecting first half)

//  UART gives you 8 bits at a time, but each BRAM word is 16 bits. So you can't write on every valid_rx pulse — only on every other one.


// Helper functions (if needed)

/* State Machine Assignment

What we need:

0) Waiting for PC to send, by polling if the start bits are sent. 

1) Upon receiving the start bits to start sending the weights to the BRAM, start writing the important bytes
(weights) from the RX - ensure the start byte is read, once fully validated,
check stop bit to end sequence. Then sends ack back (0x06 - accoridng to AI, so double check this)
Maybe here we can give different endings to each checksum to identify the last one that ocmes, ofr instance, 
the last matrix can have a special encoding that means we've now loaded all the weights and can poll
for the updated live drawings.

2) Wait for PC to send start bits.

3) Once detected, begin storing all the bytes until the end bits are detected.

4) Begin MMUL process. Let X be the live matrix, and W be the weights
   -> Load X, and W into the systolic array multipliers

5) Matrix Multiply all then load into the next stage of the pipeline

6) Transpose the Key matrix and multiply it against the query matrix

7) Load it into the Softmax value and let softmax core do its magic

8) Multiply the Softmax outplut against the value matrix

9) Multiply the Final output with the W_out matrix

10) Send out via TX, and delete all temporary values (like Q,K,V)


====================================================
Here's a summary of the matrix multiplication process:
    Step 1:  Q       = X      × W_q    →  [16,64] × [64,64] = [16,64]
    Step 2:  K       = X      × W_k    →  [16,64] × [64,64] = [16,64]
    Step 3:  V       = X      × W_v    →  [16,64] × [64,64] = [16,64]

    Step 4:  scores  = Q      × Kᵀ     →  [16,64] × [64,16] = [16,16]
                                            (K is transposed — same data, different read order)

    Step 5:  attn    = softmax(scores)  →  [16,16]  (row by row, no matmul)

    Step 6:  context = attn    × V      →  [16,16] × [16,64] = [16,64]

    Step 7:  output  = context × W_out  →  [16,64] × [64,64] = [16,64]
====================================================

    IDLE,
    STORING_WEIGHTS,
    STORING_LIVE,

    LOADING_MMUL_MATRICES,
    QKV_MMUL,
    SCORES_MMUL,
    SOFTMAX,
    VALUE_MMUL,
    WOUT_MMUL,

    SENDING_OUTPUT

*/

// State register
always_ff @(posedge clk or posedge rst) begin
    if (rst)
        state <= IDLE;
    else
        state <= next_state;
end


always_comb  // next state logic
// Default: stay in the current state
    next_state = state;

    case (state)
        IDLE: begin
            // Wait for the real frame marker before trusting anything after it.
            // serial_rx holds whatever byte uart_rx last decoded -- it never
            // clears itself, so we only ever act on it during the single
            // cycle valid_rx pulses high for a fresh byte.
            if (valid_rx && serial_rx == 8'hAA)
                next_state = WAIT_TYPE;
        end

        WAIT_TYPE: begin
            // The byte immediately after 0xAA is TYPE. Match on the upper
            // nibble: 0x1? = weight-load packet (0x10-0x17 defined),
            // 0x0? = live inference packet (0x01/0x02 defined). Anything
            // else is a malformed/unsupported TYPE -- resync to IDLE
            // instead of silently entering a storing state.
            if (valid_rx) begin
                if (serial_rx ==? 8'h1?)
                    next_state = STORING_WEIGHTS;
                else if (serial_rx ==? 8'h0?)
                    next_state = STORING_LIVE;
                else
                    next_state = IDLE;
            end
        end

        STORING_WEIGHTS: begin
            if (word_count == 12'd4095) // total of 4096 but starting from 0 - 4095
                    next_state = IDLE;
        end

        STORING_LIVE: begin
            if ()
                next_state = ;
        end

        LOADING_MMUL_MATRICES: begin
            if ()
                next_state = ;
        end

        QKV_MMUL: begin
            if ()
                next_state = ;
        end

        SCORES_MMUL: begin
            if ()
                next_state = ;
        end

        SOFTMAX: begin
            if ()
                next_state = ;
        end

        VALUE_MMUL: begin
            if ()
                next_state = ;
        end

        WOUT_MMUL: begin
            if ()
                next_state = ;
        end

        SENDING_OUTPUT: begin
            if ()
                next_state = ;
        end
    
    endcase

always_comb  // output logic
// Default: stay in the current state

    case (state)
        IDLE: begin
            if (serial_rx ==? 8'h1?) 

        end

        STORING_WEIGHTS: begin
            if ()
        end

        STORING_LIVE: begin
            if ()
        end

        LOADING_MMUL_MATRICES: begin
            if ()
        end

        QKV_MMUL: begin
            if ()
        end

        SCORES_MMUL: begin
            if ()
        end

        SOFTMAX: begin
            if ()
        end

        VALUE_MMUL: begin
            if ()
        end

        WOUT_MMUL: begin
            if ()
        end

        SENDING_OUTPUT: begin
            if ()
        end
    
    endcase


