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
    input clk
    input rst
)

// Local Parameters
// List states

localparam WAITING_FOR_WEIGHTS
localparam STORING_WEIGHTS
localparam WAITING_FOR_LIVE
localparam STORING_LIVE

localparam LOADING_MMUL_MATRICES
localparam QKV_MMUL

localparam SCORES_MMUL
localparam SOFTMAX
localparam VALUE_MMUL
localparam WOUT_MMUL
localPARAM SENDING_OUTPUT

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

*/

always (@*)
    begin


    end


always (@posedge(clk))
    begin


    end
