
import numpy as np

gamma = 0.9
states = ["A", "B", "C", "D"]

# 定义reward为 shape=(state, 转移到的state)，这样A到A和A到Breward不同
R = np.array([[10, 2, 0, 0],    # 从A出发: A->A获得10，A->B获得2
              [0, 0, 3, 0],     # 从B出发: B->C获得3
              [0, 0, 0, 0],     # C->D: reward写到下一行
              [0, 0, 0, 0]])    # D->D: reward 0

P = np.array([[0.2, 0.8, 0, 0],  # A -> A(0.2), B(0.8)
              [0, 0, 1, 0],      # B -> C
              [0, 0, 0, 1],      # C -> D
              [0, 0, 0, 1]])     # D -> D

v = np.zeros(4)
for _ in range(50):
    # 每一项 sum_a P[s,a] * [R[s,a] + gamma*v[a]]
    v_new = np.zeros_like(v)
    for s in range(4):
        v_new[s] = np.sum(P[s] * (R[s] + gamma * v))
    v = v_new

print("v_pi =", np.round(v, 3))