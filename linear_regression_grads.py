import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [1.1, 1.3],
    [1.2, 0.9],
    [2.3, 2.4],
    [2.5, 3.4],
    [3.9, 4.0],
    [4.5, 3.9],
    [5.6, 5.0],
    [6.1, 5.4],
    [7.1, 9.8],
    [9.1, 12],

])
x = data[:, 0]
y = data[:, 1]
plt.scatter(x, y)
plt.show()


# 损失函数
def loss(w, b, data):
    cost = 0
    n = len(data)
    for i in range(n):
        x = data[i, 0]
        y = data[i, 1]
        cost += (y - w * x - b) ** 2
    return cost / n


a = 0.0001  # 步长
initial_w = 0  # 初始w（随便给）
initial_b = 0  # 初始b（随便给）
num_iter = 10000  # 迭代次数


def step_grad_desc(current_w, current_b, a, data):
    sum_grad_w = 0
    sum_grad_b = 0
    m = len(data)

    for i in range(m):
        x = data[i, 0]
        y = data[i, 1]
        sum_grad_w += (current_w * x + current_b - y) * x
        sum_grad_b += current_w * x + current_b - y

    grad_w = 2 / m * sum_grad_w
    grad_b = 2 / m * sum_grad_b

    updated_w = current_w - a * grad_w
    updated_b = current_b - a * grad_b

    return updated_w, updated_b


def grad_desc(data, initial_w, initial_b, a, num_iter):
    w = initial_w
    b = initial_b
    cost_list = []  # 保存损失函数，显示下降过程
    for i in range(num_iter):
        cost_list.append(loss(w, b, data))
        w, b = step_grad_desc(w, b, a, data)

    return [w, b, cost_list]


w, b, cost_list = grad_desc(data, initial_w, initial_b, a, num_iter)

plt.plot(cost_list)
plt.show()
plt.scatter(x, y)
pred_y = w * x + b
plt.plot(x, pred_y, c='y')
plt.show()
