import numpy as np

x = np.array([np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2]), np.array([3, 3, 3, 3])])
#x1 = np.array([1, 1, 1, 1])
#x2 = np.array([2, 2, 2, 2])
#x3 = np.array([3, 3, 3, 3])
y = np.array([2, 6, 10])
w = np.array([0, 0, 0, 0])
b = 0
f = np.zeros([3, ])
for j in range(10000):
    f = np.array([np.dot(w, x[0]) + b, np.dot(w, x[1]) + b, np.dot(w, x[2]) + b])
    a = f - y
    d1 = (a[0] * x[0][0] + a[1] * x[1][0] + a[2] * x[2][0]) / 3
    d2 = (a[0] * x[0][1] + a[1] * x[1][1] + a[2] * x[2][1]) / 3
    d3 = (a[0] * x[0][2] + a[1] * x[1][2] + a[2] * x[2][2]) / 3
    d4 = (a[0] * x[0][3] + a[1] * x[1][3] + a[2] * x[2][3]) / 3
    d = np.array([d1, d2, d3, d4])
    w = w - 0.01 * d
    b = b - 0.01 * np.mean(a)
print(f)
print(w)
print(b)
