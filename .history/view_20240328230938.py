import numpy as np

# 加载数据
data_path = 'data/bj_train_set.npz'
data = np.load(data_path)


# 访问并打印特定数组的形状
train_list_shape = data['train_list'].shape
print(f"'train_list'的形状是：{train_list_shape}")

train_y_shape = data['train_y'].shape
print(f"'train_y'的形状是：{train_y_shape}")

# 完成后关闭文件
data.close()