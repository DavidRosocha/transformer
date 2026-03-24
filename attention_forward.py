# =============================================================================
# THE FOLLOWING IS A NORMAL NON SYSTOLIC IMPLEMENTATION OF THE ATTENTION FORWARD PASS
# =============================================================================

import math

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def matmul(A, B):
    """
    Pure Python matrix multiply. No numpy, no torch.
    C[i][j] = sum over k of A[i][k] * B[k][j]

    The innermost line is one MAC (Multiply-Accumulate) — exactly what
    a single Processing Element (PE) does on the systolic array in hardware.
    """
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])

    C = [[0] * cols_B for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]   # <-- one PE, one cycle
    return C


def transpose(M):
    """
    Flip a matrix so rows become columns and columns become rows.
    Used to turn K (4x32) into Kᵀ (32x4) so Q @ Kᵀ produces a square
    score matrix.

    Note: on the FPGA we never explicitly transpose — we just change the
    BRAM read order (column-first instead of row-first). Same data, no
    extra logic.
    """
    rows = len(M)
    cols = len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def softmax(row):
    """
    Convert a row of raw scores into a probability distribution (sums to 1).
    Larger scores get exponentially more weight.

    Subtracting max_val before exp() is a numerical stability trick:
      exp(x - c) / Σ exp(xⱼ - c)  ==  exp(x) / Σ exp(xⱼ)   (c cancels out)
    This keeps values from overflowing to infinity.
    On the FPGA, this also shrinks the LUT input range to roughly [-8, 0],
    making the exp() lookup table much smaller.
    """
    max_val = max(row)
    exps    = [math.exp(x - max_val) for x in row]
    total   = sum(exps)
    return [e / total for e in exps]


def print_matrix(name, M):
    print(f"{name}:")
    for row in M:
        print("  ", [round(v, 4) for v in row])
    print()


# =============================================================================
# SCALED DOT-PRODUCT ATTENTION — 6 STEPS
# =============================================================================
#
# Input x has shape [seq_len, d_model].
# Each row is one token. The weight matrices W_q, W_k, W_v are learned
# during training and loaded into FPGA BRAM once at startup — never changed
# at inference time.
#
# Steps 1, 2, 3, 4, 6 are all matrix multiplies → run on the systolic array.
# Step 5 is softmax (uses exp) → runs on a separate LUT-based hardware block.
#
# =============================================================================

def attention(x, W_q, W_k, W_v):

    # -------------------------------------------------------------------------
    # STEP 1 — Query projection
    # Q = x @ W_q
    # Each token is projected into "what it is looking for".
    # Shape: [seq_len, d_model] @ [d_model, d_k] → [seq_len, d_k]
    # -------------------------------------------------------------------------
    Q = matmul(x, W_q)
    print_matrix("Step 1 — Q (query)", Q)

    # -------------------------------------------------------------------------
    # STEP 2 — Key projection
    # K = x @ W_k
    # Each token is projected into "what it advertises about itself".
    # Shape: [seq_len, d_model] @ [d_model, d_k] → [seq_len, d_k]
    # -------------------------------------------------------------------------
    K = matmul(x, W_k)
    print_matrix("Step 2 — K (key)", K)

    # -------------------------------------------------------------------------
    # STEP 3 — Value projection
    # V = x @ W_v
    # Each token is projected into "the information it wants to pass forward".
    # Shape: [seq_len, d_model] @ [d_model, d_k] → [seq_len, d_k]
    # -------------------------------------------------------------------------
    V = matmul(x, W_v)
    print_matrix("Step 3 — V (value)", V)

    # -------------------------------------------------------------------------
    # STEP 4 — Attention scores
    # scores = Q @ Kᵀ
    # score[i][j] = dot product of token i's query with token j's key.
    # High score = token i finds token j very relevant. Can be negative.
    # Shape: [seq_len, d_k] @ [d_k, seq_len] → [seq_len, seq_len]
    # -------------------------------------------------------------------------
    scores = matmul(Q, transpose(K))

    # Scale by 1/√d_k (scaled dot-product attention).
    # Without this, large d_k causes huge dot products → softmax collapses
    # to near one-hot → vanishing gradients. PyTorch does this automatically.
    d_k = len(K[0])
    scaled_scores = [[s / math.sqrt(d_k) for s in row] for row in scores]
    print_matrix("Step 4 — scaled scores", scaled_scores)

    # -------------------------------------------------------------------------
    # STEP 5 — Softmax
    # attn = softmax(scores)
    # Converts each row of raw scores into a probability distribution.
    # attn[i][j] = "fraction of attention token i pays to token j".
    # Every row sums to 1.0.
    # On the FPGA this step uses a LUT-based exp() approximation because
    # DSP slices can only do multiply-accumulate, not exp().
    # -------------------------------------------------------------------------
    attention_weights = [softmax(row) for row in scaled_scores]
    print_matrix("Step 5 — attention weights (post-softmax)", attention_weights)

    # -------------------------------------------------------------------------
    # STEP 6 — Weighted sum of values
    # output = attn @ V
    # Each output token is a blend of all value vectors, weighted by attention.
    # High attention weight → that token's value contributes more to the output.
    # Shape: [seq_len, seq_len] @ [seq_len, d_k] → [seq_len, d_k]
    # -------------------------------------------------------------------------
    output = matmul(attention_weights, V)
    print_matrix("Step 6 — output", output)

    return output


# =============================================================================
# QUICK MATMUL SANITY CHECK
# =============================================================================

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print("matmul sanity check:", matmul(A, B))
print("expected:           [[19, 22], [43, 50]]")
print()

# =============================================================================
# RUN ATTENTION
# =============================================================================

# x:   3 tokens, each 4-dimensional  →  shape [3, 4]
# W_q, W_k, W_v: project from d_model=4 down to d_k=2  →  shape [4, 2]
x   = [[1, 0, 1, 0],
       [0, 1, 0, 1],
       [1, 1, 0, 0]]

W_q = [[1, 0],
       [0, 1],
       [1, 0],
       [0, 1]]

W_k = [[1, 0],
       [0, 1],
       [1, 0],
       [0, 1]]

W_v = [[0, 1],
       [1, 0],
       [0, 1],
       [1, 0]]

output = attention(x, W_q, W_k, W_v)
