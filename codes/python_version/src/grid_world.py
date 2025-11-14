__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]

import sys    
sys.path.append("..")         
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches          
from examples.arguments import args           

class GridWorld():

    def __init__(self, env_size=args.env_size, 
                 start_state=args.start_state, 
                 target_state=args.target_state, 
                 forbidden_states=args.forbidden_states):

        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states

        self.agent_state = start_state
        self.action_space = args.action_space          
        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step

        self.canvas = None
        self.animation_interval = args.animation_interval


        self.color_forbid = (0.9290,0.6940,0.125)
        self.color_target = (0.3010,0.7450,0.9330)
        self.color_policy = (0.4660,0.6740,0.1880)
        self.color_trajectory = (0, 1, 0)
        self.color_agent = (0,0,1)



    def reset(self):
        self.agent_state = self.start_state
        self.traj = [self.agent_state] 
        return self.agent_state, {}


    def step(self, action):
        assert action in self.action_space, "Invalid action"

        next_state, reward  = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        state_store = tuple(np.array((x_store,  y_store)) + 0.2 * np.array(action))
        state_store_2 = (next_state[0], next_state[1])

        self.agent_state = next_state

        self.traj.append(state_store)   
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}   
    
        
    def _get_next_state_and_reward(self, state, action):
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))
        if y + 1 > self.env_size[1] - 1 and action == (0,1):    # down
            y = self.env_size[1] - 1
            reward = self.reward_forbidden  
        elif x + 1 > self.env_size[0] - 1 and action == (1,0):  # right
            x = self.env_size[0] - 1
            reward = self.reward_forbidden  
        elif y - 1 < 0 and action == (0,-1):   # up
            y = 0
            reward = self.reward_forbidden  
        elif x - 1 < 0 and action == (-1, 0):  # left
            x = 0
            reward = self.reward_forbidden 
        elif new_state == self.target_state:  # stay
            x, y = self.target_state
            reward = self.reward_target
        elif new_state in self.forbidden_states:  # stay
            x, y = state
            reward = self.reward_forbidden        
        else:
            x, y = new_state
            reward = self.reward_step
            
        return (x, y), reward
        

    def _is_done(self, state):
        return state == self.target_state
    

    def render(self, animation_interval=args.animation_interval):
        if self.canvas is None:
            plt.ion()                             
            self.canvas, self.ax = plt.subplots()   
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))     
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))     
            self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')          
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()                           
            self.ax.xaxis.set_ticks_position('top')           
            
            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                self.ax.text(lb, -0.75, str(lb+1), size=10, ha='center', va='center', color='black')           
            for lb in idx_labels_y:
                self.ax.text(-0.75, lb, str(lb+1), size=10, ha='center', va='center', color='black')
            self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,labeltop=False)   

            self.target_rect = patches.Rectangle( (self.target_state[0]-0.5, self.target_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_target, facecolor=self.color_target)
            self.ax.add_patch(self.target_rect)     

            for forbidden_state in self.forbidden_states:
                rect = patches.Rectangle((forbidden_state[0]-0.5, forbidden_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_forbid, facecolor=self.color_forbid)
                self.ax.add_patch(rect)

            self.agent_star, = self.ax.plot([], [], marker = '*', color=self.color_agent, markersize=20, linewidth=0.5) 
            self.traj_obj, = self.ax.plot([], [], color=self.color_trajectory, linewidth=0.5)

        # self.agent_circle.center = (self.agent_state[0], self.agent_state[1])
        self.agent_star.set_data([self.agent_state[0]],[self.agent_state[1]])       
        traj_x, traj_y = zip(*self.traj)         
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(animation_interval)
        if args.debug:
            input('press Enter to continue...')     



    def add_policy(self, policy_matrix):
        """
        在格子世界可视化环境中添加策略箭头。

        参数:
            policy_matrix: 一个形状为 (状态数, 动作数) 的二维array，policy_matrix[s, a] 代表在状态s采取动作a的概率。

        功能说明:
            - 对于每一个状态(state)，都会在对应坐标(x, y)上检查其所有动作的概率。
            - 如果某个动作概率不为0，则：
                - 取该动作的方向(dx, dy)。
                - 如果动作不是原地运动((0,0))，则在状态对应格子处画一根箭头，长度/粗细反映策略概率。
                - 如果动作是原地运动，则在格子中画一个圈，表示驻留动作。
            - 箭头(或圈)的颜色统一用 self.color_policy。

        """
        for state, state_action_group in enumerate(policy_matrix):
            # 计算格子坐标 (x, y)，x为列号，y为行号
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            # 枚举该状态下每个动作的概率
            for i, action_probability in enumerate(state_action_group):
                if action_probability != 0:
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0, 0):
                        # 在(x, y)为起点，按照(action_probability)比例绘制箭头，箭头方向为(dx, dy)
                        self.ax.add_patch(
                            patches.FancyArrow(
                                x, y,
                                dx=(0.1 + action_probability / 2) * dx,
                                dy=(0.1 + action_probability / 2) * dy,
                                color=self.color_policy,
                                width=0.001,
                                head_width=0.05
                            )
                        )
                    else:
                        # 动作为原地不动，则画一个圈
                        self.ax.add_patch(
                            patches.Circle(
                                (x, y),
                                radius=0.07,
                                facecolor=self.color_policy,
                                edgecolor=self.color_policy,
                                linewidth=1,
                                fill=False
                            )
                        )

    def add_state_values(self, values, precision=1):
        """
        在格子世界可视化环境中显示状态值。

        参数:
            values: 可迭代对象，包含每个状态的值。
            precision: 显示值的小数点精度，默认为1。

        功能说明:
            - 先将所有的状态值四舍五入到指定的精度。
            - 遍历每个状态，根据状态索引 i 计算该状态的格子(x, y)坐标，x为列，y为行。
            - 在每个格子的中心位置(x, y)处，使用 matplotlib 的 text 方法显示对应的数值，黑色字体，居中对齐。
        """
        values = np.round(values, precision)
        for i, value in enumerate(values):
            x = i % self.env_size[0]    # 列坐标
            y = i // self.env_size[0]   # 行坐标
            self.ax.text(
                x, y, str(value), 
                ha='center', va='center', 
                fontsize=10, color='black'
            )