
# —— 第 2.2 节 · 回报与自举思想（Returns and Bootstrapping）——

> 本节源自原书《Reinforcement Learning: An Introduction》第二章 “State Values and the Bellman Equation” 的 §2.2 部分。
> 这里采用“讲义式”翻译风格，对原文内容进行解释性重述与扩展说明。

---

## 🎯 学习目标

1. 理解强化学习中的“回报（Return）”定义；  
2. 掌握折扣机制（Discounting）的意义与数学形式；  
3. 理解“自举（Bootstrapping）”更新思想；  
4. 对比自举与蒙特卡洛（Monte Carlo）方法的差异。

---

## 📘 一、回报（Return）的定义

强化学习的目标是最大化智能体在环境中所获得的长期奖励。  
为此，我们引入“折扣回报（Discounted Return）”的概念：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}. \tag{2.1}
$$

其中：  
- $R_{t+k+1}$ 表示第 $k$ 步后的奖励；  
- $\gamma$ 为折扣因子（discount factor），取值 $0 \le \gamma \le 1$；  
- 较小的 $\gamma$ 让智能体更关注短期收益，而较大的 $\gamma$ 则鼓励远见。

【讲解】  
> 折扣系数可以理解为“时间价值”：越远的未来越不确定，因此对当前决策的影响越小。  
> 例如 $\gamma=0.9$ 意味着十步后的奖励仅保留原始价值的约 35%。

---

## 📐 二、回报的递推形式

将式 (2.1) 改写为递推形式：

$$
\begin{aligned}
G_t &= R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \dots) \\
    &= R_{t+1} + \gamma G_{t+1}.
\end{aligned} \tag{2.2}
$$

这被称为 **回报的递推定义（Recursive Definition of Return）**。

【讲解】  
> 式 (2.2) 的意义在于：当前的回报由两部分组成：  
> 即时奖励 + 折扣后的未来回报。  
> 这为自举更新提供了理论基础，使智能体能够逐步传播未来的信息。

---

## 💡 三、Bootstrapping（自举）思想

在现实环境中，我们通常无法获得完整的未来回报 $G_{t+1}$，  
因此只能用当前估计值来替代：

$$
v(S_t) \leftarrow R_{t+1} + \gamma v(S_{t+1}). \tag{2.3}
$$

这就是 **自举更新（Bootstrapping Update）**。

【讲解】  
> “Bootstrapping” 的字面含义是“自己拉着自己的靴带向上爬”。  
> 在强化学习中，它表示用对未来的预测（$v(S_{t+1})$）来改进当前的预测（$v(S_t)$）。  
> 动态规划（DP）、时序差分（TD）以及 Q-learning 都基于这种思想。

---

## 🧩 四、图像化理解（Figure 2.2）

设一个链式世界：

```
S0 → S1 → S2 → S3 → Goal
```

- 每次移动获得奖励 $R=-1$；  
- 到达 Goal 奖励为 $0$；  
- 折扣因子 $\gamma=0.9$。

从终点向前递推：

$$
v(S_4)=0, \quad v(S_3)=-1, \quad v(S_2)=-1.9, \quad v(S_1)=-2.71.
$$

【讲解】  
> 这张图展示了价值函数如何从目标状态逐步“回传”信息。  
> 离目标越远，累计损失越大；这正是 Bellman 方程与 Bootstrapping 的核心机制。

---

## 🔢 五、数值示例

假设：  
- $R_{t+1}=0.8$；  
- $\gamma=0.9$；  
- 当前估计 $v(S_{t+1})=1.0$。  

代入 (2.3)：

$$
v(S_t) = 0.8 + 0.9(1.0) = 1.7.
$$

若 $v(S_{t+1})$ 更新为 1.2：

$$
v(S_t) = 0.8 + 0.9(1.2) = 1.88.
$$

【讲解】  
> 随着下一状态估计的提高，当前状态值也会相应改进，从而实现收敛。

---

## 🧮 六、Python 伪代码示例

```python
gamma = 0.9
alpha = 0.1  # 学习率

v = {"S0": 0.0, "S1": 0.0, "S2": 0.0, "S3": 0.0, "Goal": 0.0}
rewards = {"S0": -1, "S1": -1, "S2": -1, "S3": 0}

for episode in range(10):
    for s in ["S0", "S1", "S2", "S3"]:
        next_s = {"S0": "S1", "S1": "S2", "S2": "S3", "S3": "Goal"}[s]
        v[s] += alpha * (rewards[s] + gamma * v[next_s] - v[s])

print(v)
```

【讲解】  
> 该代码实现了 TD(0) 学习的核心逻辑：  
> 利用下一状态的预测实时更新当前状态的估计。  
> 每次迭代都在执行式 (2.3)。

---

## 🧠 七、自测题

1. 证明式 (2.2) 可以展开回式 (2.1)。  
2. 当 $\gamma=0$ 时，Bootstrapping 更新将退化为什么？  
3. 为什么 Bootstrapping 对非终止（continuing）任务特别重要？  
4. 与蒙特卡洛方法相比，Bootstrapping 有哪些优缺点？

---

## 📘 八、小结与衔接

- 回报 $G_t$ 是强化学习的核心目标；  
- Bootstrapping 允许智能体利用当前预测递推地更新估计，而不必等到回合结束；  
- 该思想自然引出**状态值函数**的定义：

$$
v_\pi(s) = \mathbb{E}_\pi[G_t | S_t=s].
$$

这将成为下一节（§2.3）——**状态值函数 (State Value Function)** 的数学基础。
