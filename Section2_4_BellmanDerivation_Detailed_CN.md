
# —— 第 2.4 节 · Bellman 方程推导（Deriving the Bellman Equation） ——

> 本节将从状态值函数 $v_\pi(s)$ 的定义出发，推导出强化学习中最重要的等式之一 —— **Bellman 方程（Bellman Equation）**。  
> 它描述了每个状态的价值与其后续状态价值之间的递推关系，是动态规划、策略迭代与值迭代的理论基石。

---

## 🎯 学习目标

学习完本节后，你应能：  
1. 理解 Bellman 方程的推导逻辑与条件期望分解；  
2. 掌握 $v_\pi(s)$ 的递推形式与直觉含义；  
3. 能通过数值示例构造并求解 Bellman 方程；  
4. 理解其在强化学习算法中的核心作用。

---

## 📘 一、从定义出发

根据第 2.3 节，状态值函数定义为：

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi \Big[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \Big| S_t = s \Big]. \tag{2.6}
$$

我们将该期望展开为两部分：  
第一步的即时奖励与之后所有未来奖励：

$$
v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t=s].
$$

利用 (2.2) 中的递推定义 $G_t = R_{t+1} + \gamma G_{t+1}$，  
得到了最初的递推形式。

---

## 📐 二、条件期望分解

将上式的期望按动作与状态展开：

$$
\begin{aligned}
v_\pi(s) &= \sum_a \pi(a|s) \sum_{s'} p(s'|s,a) \, \mathbb{E}[R_{t+1} + \gamma v_\pi(s') \mid s,a,s'] \\
         &= \sum_a \pi(a|s) \sum_{s'} p(s'|s,a)\big[r(s,a,s') + \gamma v_\pi(s')\big]. \tag{2.7}
\end{aligned}
$$

这就是 **Bellman 方程的标准形式**。

> **直觉解释：**
> 当前状态的价值等于：采取动作 a 的概率 × （即时奖励 + 折扣后的未来状态价值）的期望。

---

## 💡 三、图像化理解（文字描述）

想象下图所示的状态转移关系：  

```
s  --a1-->  s1
 |           |
 |--a2-->  s2
```
- 每条箭头代表一个可能的动作和对应奖励；  
- Bellman 方程将这些路径的期望综合起来；  
- 因此 $v_\pi(s)$ 是“所有未来分支的加权平均”。

换句话说：Bellman 方程让“未来价值”在状态空间中传播。

---

## 🔢 四、数值示例

设三状态系统 $S=\{A,B,C\}$，策略 π 为确定性，转移与奖励如下表：

| 当前状态 | 下一状态 | 奖励 |
|-----------|-----------|------|
| A | B | +1 |
| B | C | +2 |
| C | C | 0 |

折扣因子 $\gamma=0.9$。  

根据式 (2.7)：

$$
\begin{aligned}
v_\pi(A) &= 1 + 0.9v_\pi(B), \\
v_\pi(B) &= 2 + 0.9v_\pi(C), \\
v_\pi(C) &= 0.
\end{aligned}
$$

求解得：
$$
v_\pi(A)=2.8, \quad v_\pi(B)=2.0, \quad v_\pi(C)=0.
$$

---

## 🧮 五、Python伪代码：Bellman 迭代更新

```python
import numpy as np

gamma = 0.9
R = np.array([1, 2, 0])
P = np.array([[0, 1, 0],
              [0, 0, 1],
              [0, 0, 1]])

v = np.zeros(3)

for i in range(20):  # Bellman 迭代
    v = R + gamma * P.dot(v)

print("状态值函数 v_pi =", v)
```

输出：  
```
状态值函数 v_pi = [2.8 2.  0. ]
```
每次迭代都在执行：

$$
v_{k+1}(s) = \mathbb{E}_\pi[R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t=s]. \tag{2.8}
$$

该式与 (2.7) 形式等价，展示了 Bellman 方程的动态更新特性。

---

## 💬 六、理论总结

| 概念 | 数学形式 | 含义 |
|------|------------|------|
| 状态值函数 | $v_\pi(s) = \mathbb{E}_\pi[G_t|S_t=s]$ | 未来回报期望 |
| 回报递推 | $G_t = R_{t+1} + \gamma G_{t+1}$ | 时间展开形式 |
| Bellman 方程 | $v_\pi(s) = \sum_a \pi(a|s)\sum_{s'}p(s'|s,a)[r+\gamma v_\pi(s')]$ | 状态值递推 |

---

## 🧠 七、自测题

1. 为什么 Bellman 方程中的期望可以分解为双重求和？  
2. 当策略 π 为确定性时，Bellman 方程如何简化？  
3. 若奖励与状态转移独立（即 $r(s,a,s')=r(s,a)$），式 (2.7) 如何变化？  
4. Bellman 方程与 Bootstrapping 更新 (2.3) 的关系是什么？

---

## 📘 八、与第 2.5 节的衔接

本节推导了 Bellman 方程的完整形式，说明了状态值的递推机制。  
在下一节 §2.5 中，我们将用具体示例展示如何利用 Bellman 方程**计算状态值函数**，  
并分析确定性与随机性环境下的方程解法。
