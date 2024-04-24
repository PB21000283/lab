import numpy as np

# 定义矩阵
A = np.array([[1, -1, 0], [-1, 2, 2], [0, 2, 3]])

# 计算特征值
eigenvalues = np.linalg.eigvals(A)

print("特征值：", eigenvalues)