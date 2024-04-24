import numpy as np

# 加载数据
data_path = 'data/bj_train_set.npz'
data = np.load(data_path)

# 查看包含的键
print(list(data.keys()))

# 访问特定数组
# 假设我们想要访问键为'array1'的数组
array1 = data['array1']
print(array1)

# 一旦完成数据操作，确保关闭文件以释放资源
data.close()