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


# 损失函数
def loss(w, b, data):
    cost = 0
    n = len(data)
    for i in range(n):
        x = data[i, 0]
        y = data[i, 1]
        cost += (y - w * x - b) ** 2
    return cost / n


# 分别对w 和 b 求偏导
def fit(data):
    m = len(data)
    x_avg = np.mean(x)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(m):
        sum1 += y[i] * (x[i] - x_avg)
        sum2 += x[i] ** 2
    w = sum1 / (sum2 - m * (x_avg ** 2))
    for i in range(m):
        sum3 += y[i] - w * x[i]
    b = sum3 / m
    return w, b


w, b = fit(data)
print(f"w为{w}，b为{b}")
print("损失函数值：", loss(w, b, data))
pred_y = w * x + b
plt.plot(x, pred_y, c='r')
plt.show()
