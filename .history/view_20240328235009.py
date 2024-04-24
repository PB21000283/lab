import numpy as np
import torch
# 加载数据
data_path = 'data/bj_train_set.npz'
data = np.load(data_path, allow_pickle=True)

y_idx = data['train_y']
x = [torch.LongTensor(sq) for sq in data['train_list']]

print(x[0])  # 打印第一个转换后的张量查看是否成功


# 完成后关闭文件
data.close()