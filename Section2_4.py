
import numpy as np

# 定义相关参数
gamma = 0.9  # 折扣因子
states = ["A", "B", "C"]  # 状态名称便于理解

# 即时奖励：每个状态出发能获得的奖励
# R[0]=1(A), R[1]=2(B), R[2]=0(C)
R = np.array([1, 2, 0])

# 状态转移概率矩阵 P[s, s']：从状态s出发，到s'的概率
#                   A  B  C
# 从A出发 P[0,...]= [0, 1, 0]  # A 只会到B
# 从B出发 P[1,...]= [0, 0, 1]  # B 只会到C
# 从C出发 P[2,...]= [0, 0, 1]  # C 只会停在C
P = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
])

# 初始化状态值估计
v = np.ones(3)   # 起始时刻每个状态的价值估计为1

print("  状态序号对应：A=0, B=1, C=2\n")

print(f"初始状态值 v = {v}")

# 迭代解：
# https://www.notion.so/Chapter-2-State-Values-and-Bellman-Equation-2a85d7050636819092b1c5e6558378a6?source=copy_link#2a85d705063681108058c785402e64b3
# ---------------------------迭代更新原理说明---------------------------
# 每一次迭代，就是根据贝尔曼期望方程（递归关系）用当前的v估计来计算下一个v（即自举），
# 本质上是让状态值函数不断逼近“满足贝尔曼方程的真实解”。
# 对每个状态，更新规则为：
#    v_new[s] = R[s] + gamma * sum_{s'} P[s, s'] * v[s']
# 这体现了两个成分：
#   - R[s]                 : 当前状态能立即获得的奖励
#   - gamma * ...          : 按概率折扣后，进入各个后继状态下的价值加权平均
# 用旧估计v计算新v_new，相当于用“自己的预测”作为未来的判断，这就叫“自举”。
# 迭代多次后，v会收敛到贝尔曼方程的唯一解，也就是在当前策略下每个状态的长期累计奖励期望。
for i in range(20):
    v_new = np.zeros_like(v)
    for s in range(3):
        v_new[s] = R[s] + gamma * np.dot(P[s], v)
    print(f"第{i+1}次迭代后 v = {np.round(v_new, 3)}")
    v = v_new

# 打印收敛结果，并做解释性输出
print("\n最终收敛的状态值函数 v_pi =")
for idx, val in enumerate(v):
    print(f"  状态 {states[idx]}: {val:.3f}")
