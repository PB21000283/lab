import numpy as np

matrix = np.array([[4, -1, 1], [16, -2, -3], [16, -3, -1]])

# 进行QR分解
Q, R = np.linalg.qr(matrix)

print(Q, R
)