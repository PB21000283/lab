import numpy as np
import torch
# 加载数据
data_path = 'data/bj_train_set.npz'
data = np.load(data_path, allow_pickle=True)
y = [torch.LongTensor(sq) for sq in data['train_y']]
print(y[0])  # 打印第一个转换后的张量查看是否成功
x2=data['train_list']
y2=data['train_y']
z=data[x2[y2[0]]]
print(z)
# 完成后关闭文件
data.close()