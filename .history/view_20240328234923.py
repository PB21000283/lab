import numpy as np
import torch
# 加载数据
data_path = 'data/bj_train_set.npz'
data = np.load(data_path, allow_pickle=True)

y_idx = data['train_y']
x = [torch.LongTensor(sq) for sq in data['train_list']]

print(x[0])  # 打印第一个转换后的张量查看是否成功
# 访问并打印特定数组的形状
train_list_shape = data['train_list'].shape
print(f"'train_list'的形状是：{train_list_shape}")
print(type(data['train_list'][0]))
print(data['train_list'][0])
train_y_shape = data['train_y'].shape
print(f"'train_y'的形状是：{train_y_shape}")

# 完成后关闭文件
data.close()