import numpy as np

def derivative_theta_0(theta_0, theta_1, x, y):
    n = len(x)
    return (1 / n) * np.sum((theta_1 * x + theta_0 - y))

def derivative_theta_1(theta_0, theta_1, x, y):
    n = len(x)
    return (1 / n) * np.sum((theta_1 * x + theta_0 - y) * x)


theta0 = 1234
theta1 = 678
alpha = 0.01
deps = 0.001

while True:
    derv_0 = derivative_theta_0(theta0, theta1, x, y)
    derv_1 = derivative_theta_1(theta0, theta1, x, y)
    theta0 = theta0 - alpha * derv_0
    theta1 = theta1 - alpha * theta1
    if derv_0 < deps and derv_1 < deps:
        break