import numpy as np

def jacobi_method_for_eigenvalues(A, max_iterations=100, tolerance=1e-10):
    n = A.shape[0]
    i=1
    for _ in range(max_iterations):
        # 寻找最大的非对角线元素
        p, q = np.unravel_index(np.argmax(np.abs(np.triu(A, 1))), A.shape)
        largest_off_diag = A[p, q]
        if np.abs(largest_off_diag) < tolerance:
            break
        if i==1 :
            print(p)
            print(q)
        # 计算旋转参数
        phi = 0.5 * np.arctan2(2 * largest_off_diag, A[q, q] - A[p, p])
        c = np.cos(phi)
        s = np.sin(phi)
        if i==1 :
            print(phi)
        # 创建旋转矩阵 J (初始化为单位矩阵)
        J = np.eye(n)
        J[[p, q], [p, q]] = c
        J[p, q] = s
        J[q, p] = -s

        # 应用旋转
        A = J.T @ A @ J
        if i==1 :
            print(A)
            i=0

    # 对角线元素即为特征值
    eigenvalues = np.diag(A)
    return eigenvalues

# 重新定义矩阵 A，以避免使用上一次修改过的 A
A = np.array([[1, -1, 0], [-1, 2, 2], [0, 2, 3]])

# 计算特征值
eigenvalues_converged = jacobi_method_for_eigenvalues(A)
print(eigenvalues_converged)