import numpy as np

# 定义环境中的状态数和动作数
num_states = 6
num_actions = 4

# 定义Q表格，用于存储Q函数的估计值
Q = np.zeros((num_states, num_actions))

# 定义学习率、折扣因子和探索率
learning_rate = 0.8
gamma = 0.95
epsilon = 0.1

# 定义环境的奖励矩阵
rewards = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

# Q-Learning算法
def q_learning(num_episodes):
    for episode in range(num_episodes):
        # 初始化状态为随机选择的起始状态
        state = np.random.randint(0, num_states)

        while state != 5:  # 终止状态为5
            # 根据贪婪策略选择动作（或者进行随机探索）
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, num_actions)
            else:
                action = np.argmax(Q[state, :])

            # 根据选择的动作与环境交互，获取奖励和新状态
            new_state = np.argmax(rewards[state, :])
            reward = rewards[state, action]

            # 使用贝尔曼方程更新Q函数的估计值
            Q[state, action] = Q[state, action] + learning_rate * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            # 更新状态为新状态
            state = new_state

# 执行Q-Learning算法
q_learning(num_episodes=1000)

# 打印训练后的Q表格
print("Q表格:")
print(Q)
