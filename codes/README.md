# 格子世界环境代码

## 概述

我们在本书中加入了用于格子世界环境的代码。感兴趣的读者可以在该环境中开发和测试自己的算法。本项目提供了 Python 和 MATLAB 两种版本。

请注意，书中涉及的所有算法代码我们并未全部提供。这是因为这些算法作为线下教学作业，学生需要利用所给的环境自行开发实现相关算法。当然，也有一些第三方实现可以参考，感兴趣的读者可在本书主页查看相关链接。

我要感谢我的博士生米一泽和李佳楠，他们也是我线下课程的助教，对本代码贡献良多。

欢迎大家对代码提出任何意见或发现的 bug。

----

## Python 版本

### 环境需求

- 支持 Python 3.7、3.8、3.9、3.10 和 3.11。请确保已安装如下包：`numpy` 和 `matplotlib`。

### 运行默认示例的方法

运行示例代码，请依照下列步骤：

1. 切换至 `examples/` 目录

```bash
cd examples
```

2. 运行脚本：

```bash
python example_grid_world.py
```

运行后，您将看到如下动画：

- 蓝色星标表示智能体当前位置；
- 每个格子的箭头表示该状态下的策略；
- 绿色线条表示智能体的历史运动轨迹；
- 黄色格子表示障碍物；
- 蓝色格子表示目标状态；
- 各格子内的数值为对应状态的价值，起始为 0 到 10 的随机数。后续您需要设计自己的算法计算这些状态价值。
- 网格上方的数字为水平方向（x 轴）坐标；
- 网格左侧的数字为垂直方向（y 轴）坐标。

![](python_version/plots/sample4.png)

### 自定义格子世界环境参数

若需自定义格子世界环境，请打开 `examples/arguments.py` 并调整以下参数：

"**env-size**"、"**start-state**"、"**target-state**"、"**forbidden-states**"、"**reward-target**"、"**reward-forbidden**"、"**reward-step**"：

- “env-size” 表示为一个元组，第一个元素为列索引（水平坐标），第二个为行索引（垂直坐标）。
- “start-state” 表示智能体初始位置；
- “target-state” 表示目标位置；
- “forbidden-states” 表示障碍物位置；
- “reward-target”、“reward-forbidden” 和 “reward-step” 分别表示到达目标、进入障碍和每一步的奖励。

示例如下：

# 若源码目录下误加入了 `__pycache__` 文件夹到 Git 暂存区，可用如下命令移除（仅从暂存区删除，不删除本地文件）：

```bash
git rm --cached "__pycache__"
```

如果看到如下错误：

```bash
fatal: pathspec '__pycache__' did not match any files
```

说明当前目录下没有对应的 `__pycache__` 文件夹或暂存区中已无该路径。这种情况下，若 `__pycache__` 目录存在于项目的其它路径（如子目录下），可以使用如下命令递归查找并移除所有被暂存的 `__pycache__` 文件夹：

```bash
find . -type d -name '__pycache__' -exec git rm -r --cached {} +
```

如果依然提示没有匹配项，说明所有 `__pycache__` 文件夹都未加入暂存区，无需进一步操作，忽略该提示即可。



如需指定目标状态，请修改如下代码中的默认值：

```python
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=(4,4))
```

请注意，环境内各状态（如起点、终点、障碍点）采用常规 Python 坐标体系，(0, 0) 作为原点。

若想在每一步保存图像，请修改 "debug" 参数为 "True"：

```bash
parser.add_argument("--debug", type=bool, default=True)
```

### 创建环境实例

如需在格子世界环境下测试自己的强化学习算法需先创建实例，创建和交互的流程见 `examples/example_grid_world.py`：

```python
from src.grid_world import GridWorld

 	env = GridWorld()
    state = env.reset()               
    for t in range(20):
        env.render()
        action = np.random.choice(env.action_space)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, Next state: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")

```

![](python_version/plots/sample1.png)

- 策略以矩阵形式存储，可设计为确定性或随机性。下例为随机策略：

 ```python
     # 添加策略
     policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                       
     policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis] 
 ```

- 若要修改箭头形状，可在 `src/grid_world.py` 文件中更改如下代码：

 ```python
self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1+action_probability/2)*dx, dy=(0.1+action_probability/2)*dy, color=self.color_policy, width=0.001, head_width=0.05))   
 ```

![](python_version/plots/sample2.png)

-  为每个格子添加状态价值：

```python
values = np.random.uniform(0,10,(env.num_states,))
env.add_state_values(values)
```

![](python_version/plots/sample3.png)

- 渲染环境：

```python
env.render(animation_interval=3)    # 图片暂停3秒
```
