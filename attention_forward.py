

def matmul(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])

    C = [[0] * cols_B for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print("test matmul:", matmul(A, B), '\n')

## Need to make the 6 attention steps

"""
Take an input matrix, [sequence_length, d_model], where each word is one token,
and then you multiply it by three different weight matrices, W_q, W_k, and W_v, to get the query, key, andv values matrices

- Q Query: What you're looking for
- K Key: What you have
- V Value: Information you want to pass forward

All of these weight matrices are learned projections (ie. Q = matmul(x, W_q), etc...)
The weights are what the model learns during training and sit in the FPGA's BRAM - ONLY LOADED ONCE woaoohhhhh supa cool


Need to compute attention scores
- Attention scores basically mean how much should each token pay attention to every other token
--> After projecting through the weight matrices, each token has a query vector (what it's looking for), and a 
key vector (what it tells other tokens about itself)

The score is calculated as the dot product of token i's query with token j's query
score[i][j] = Q[i] * K[j]
High Score means very relevant, low score (can be negative) means not relevant


"""

import math

def softmax(row): #turns scores into probabilities 
    max_val = max(row)
    exps = [math.exp(x-max_val) for x in row] #stabilizer to prevent overflow
    total = sum(exps)
    return [e / total for e in exps]

def transpose(M):
    rows = len(M)
    cols = len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]

def attention(x, W_q, W_k, W_v):
    Q = matmul(x, W_q)
    K = matmul(x, W_k)
    V = matmul(x, W_v)

    scores = matmul(Q, transpose(K))

    d_k = len(K[0])  # dimension of the key vectors
    scaled_scores = [[s / math.sqrt(d_k) for s in row] for row in scores] 
    '''
    The square root here is scaled-dot product attention
    Dividing by the quare root of the key dimension stops the scores from getting too large that the softmax stops working
    Pytorch does automatically
    '''
    print("scaled scores:", scaled_scores, '\n')

    attention_weights = [softmax(row) for row in scaled_scores]
    print("attention weights;", attention_weights, '\n')

    output = matmul(attention_weights, V)
    print("output:", output, '\n')

    return output


x   = [[1,0,1,0], [0,1,0,1], [1,1,0,0]]
W_q = [[1,0],[0,1],[1,0],[0,1]]
W_k = [[1,0],[0,1],[1,0],[0,1]]
W_v = [[0,1],[1,0],[0,1],[1,0]]

output = attention(x, W_q, W_k, W_v)
