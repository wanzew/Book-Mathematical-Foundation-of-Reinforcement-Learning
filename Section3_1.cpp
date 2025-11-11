#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main() {
  // 定义所有状态
  const std::vector<std::string> states = {"A", "B", "C"};
  // 折扣因子
  const double gamma = 0.9;

  // 奖励矩阵 R[s][a]，对应每个状态s下采取动作a获得的即时奖励
  const std::vector<std::vector<double>> R = {
      {0.0, 1.0, 0.0}, // 状态 A 的各动作奖励
      {0.0, 0.0, 2.0}, // 状态 B 的各动作奖励
      {0.0, 0.0, 0.0}  // 状态 C 的各动作奖励
  };

  // 状态转移概率矩阵 P[s][a][s']，表示状态s采取动作a后转移到状态s'的概率
  const std::vector<std::vector<std::vector<double>>> P = {
      // 状态A的转移（对每个动作，对每个s'的概率）
      {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}},
      // 状态B的转移
      {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}},
      // 状态C的转移
      {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}}};

  // 状态值函数 V，初始均为0
  std::vector<double> V(states.size(), 0.0);

  // 策略，policy[s]=a表示在状态s采取动作a，初始均为0
  std::vector<int> policy(states.size(), 0);

  // 策略迭代最大次数
  const int max_policy_iterations = 10;
  // 策略评估收敛阈值
  const double evaluation_tolerance = 1e-9;

  // 策略迭代主循环
  for (int iteration = 0; iteration < max_policy_iterations; ++iteration) {
    // 策略评估阶段，对当前策略求解状态价值函数
    while (true) {
      double delta = 0.0; // 每轮最大状态值变化
      // 遍历每个状态
      for (std::size_t s = 0; s < states.size(); ++s) {
        const int a = policy[s]; // 当前策略选定的动作a

        // 按贝尔曼方程计算新值
        double new_value = R[s][a];
        for (std::size_t sp = 0; sp < states.size(); ++sp) {
          new_value += gamma * P[s][a][sp] * V[sp];
        }

        delta = std::max(delta, std::fabs(new_value - V[s])); // 记录最大变化
        V[s] = new_value;                                     // 更新状态值
      }
      // 如果收敛则停止本轮评估
      if (delta < evaluation_tolerance) {
        break;
      }
    }

    // 策略提升阶段，对每个状态选择使价值最大的动作
    bool policy_stable = true; // 记录策略是否收敛
    // 遍历所有状态
    for (std::size_t s = 0; s < states.size(); ++s) {
      int old_action = policy[s];   // 记录旧动作
      int best_action = old_action; // 最优动作
      double best_value = -std::numeric_limits<double>::infinity();

      // 枚举状态s下所有可能动作，找最大q值
      for (int a = 0; a < static_cast<int>(R[s].size()); ++a) {
        double q_sa = R[s][a];
        for (std::size_t sp = 0; sp < states.size(); ++sp) {
          q_sa += gamma * P[s][a][sp] * V[sp];
        }
        if (q_sa > best_value) {
          best_value = q_sa;
          best_action = a;
        }
      }

      // 更新策略
      policy[s] = best_action;
      if (best_action != old_action) {
        policy_stable = false; // 策略有变，设为不稳定
      }
    }

    // 若策略未发生变化，则算法收敛，跳出主循环
    if (policy_stable) {
      break;
    }
  }

  // 输出最终改进后的策略
  std::cout << "改进后的策略: ";
  for (std::size_t s = 0; s < policy.size(); ++s) {
    std::cout << policy[s] << (s + 1 < policy.size() ? " " : "\n");
  }

  // 输出最终状态值
  std::cout << "状态值: ";
  for (std::size_t s = 0; s < V.size(); ++s) {
    std::cout << V[s] << (s + 1 < V.size() ? " " : "\n");
  }

  return 0;
}