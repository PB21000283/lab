import numpy as np

# 定义矩阵
A = np.array([[1, -1, 0], [-1, 2, 2], [0, 2, 3]])

# 计算特征值
eigenvalues = np.linalg.eigvals(A)

print("特征值：", eigenvalues)

def jacobi_method_converge(A, tolerance=1e-10):
    n = A.shape[0]
    iteration = 0
    while True:
        # 寻找最大的非对角线元素
        largest_off_diag = 0
        for i in range(n):
            for j in range(i+1, n):
                if np.abs(A[i, j]) > np.abs(largest_off_diag):
                    largest_off_diag = A[i, j]
                    p, q = i, j
        
        # 检查是否已经收敛
        if np.abs(largest_off_diag) < tolerance:
            break
        
        # 计算旋转参数
        if A[p, p] == A[q, q]:
            t = 1
        else:
            tau = (A[q, q] - A[p, p]) / (2 * A[p, q])
            t = np.sign(tau) / (np.abs(tau) + np.sqrt(1 + tau**2))
        c = 1 / np.sqrt(1 + t**2)
        s = t * c
        
        # 构建旋转矩阵
        J = np.eye(n)
        J[p, p], J[q, q] = c, c
        J[p, q], J[q, p] = -s, s
        
        # 应用旋转
        A = J.T @ A @ J
        
        iteration += 1
        
    print(f"Converged after {iteration} iterations.")
    return A

# 使用更高的容忍度重新计算，直到收敛
final_converged_A = jacobi_method_converge(A, 1e-10)
final_converged_A