import numpy as np
import random

def create_data(N, alpha=0.5): # random > alpha, then here is a edge.
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if random.random() < alpha:
                G[i][j] = 1
    return G
G = create_data(10)
#print(G)

# GtoM
def GtoM(G, N):
    M = np.zeros((N, N))
    for i in range(N):
        D_i = sum(G[i])
        if D_i == 0:
            continue
        for j in range(N):
            M[j][i] = G[i][j] / D_i # watch out! M_j_i instead of M_i_j
    return M
M = GtoM(G, 10)

def PageRank(M, N, T=300, eps=1e-6):
    R = np.ones(N) / N
    for time in range(T):
        R_new = np.dot(M, R)
        if np.linalg.norm(R_new - R) < eps:
            break
        R = R_new.copy()
    return R_new

values = PageRank(M, 10, T=2000)
print('flow版本：', values)

# Google Formula, 跳出概率陷阱
def PageRank(M, N, T=300, eps=1e-6, beta=0.8):
    R = np.ones(N) / N
    teleport = np.ones(N) / N
    for time in range(T):
        R_new = beta * np.dot(M, R) + (1-beta)*teleport
        if np.linalg.norm(R_new - R) < eps:
            break
        R = R_new.copy()
    return R_new

values = PageRank(M, 10, T=2000)
print('Google Formula', values)



