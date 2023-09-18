import numpy as np
from sklearn.linear_model import LinearRegression
import scipy as scp

# dataset: (theta_1, theta_2, x, y), assuming theta_1 and theta_2 are in radians?
converter = {
    0: lambda s: float(s[1:]),
    1: lambda s: float(s),
    2: lambda s: float(s),
    3: lambda s: float(s[:-1]),
}
data = np.loadtxt("noisy_acrobot.txt", delimiter=",", converters=converter)
n_samples, _ = data.shape

"""Linear Algebra way

Solve least squares problem Ax=b, where A is a 2N x 2 matrix, B is a 2N x 1 vector.
Each row of data takes up two rows in A and b. The FK equations of positions x and y
each take up one of the two rows. x is a 2 x 1 vector of joint angles. 
"""
A = np.empty((2 * n_samples, 2))
b = np.empty(2 * n_samples)
for i, (theta1, theta2, x, y) in enumerate(data):
    A[i * 2] = (np.cos(theta1), np.cos(theta1 + theta2))
    A[i * 2 + 1] = (np.sin(theta1), np.sin(theta1 + theta2))
    b[i * 2] = x
    b[i * 2 + 1] = y

x, sse, rank, s = scp.linalg.lstsq(A, b)  # same as np.linalg.inv(A.T @ A) @ A.T @ b
print(x)
