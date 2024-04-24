import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils

data_path ='data/bj_test_set.npz'
data = np.load(data_path, allow_pickle=True)
##y = [torch.LongTensor(sq) for sq in data['train_y']]
##print(y[0])  # 打印第一个转换后的张量查看是否成功
x=[torch.LongTensor(sq) for sq in data['test_list']]

print(len(x))