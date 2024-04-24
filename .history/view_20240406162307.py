import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils

##import torch
# 加载数据
def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))



data_path = 'data/bj_train_set.npz'
data = np.load(data_path, allow_pickle=True)
##y = [torch.LongTensor(sq) for sq in data['train_y']]
##print(y[0])  # 打印第一个转换后的张量查看是否成功
x=[torch.LongTensor(sq) for sq in data['train_list']]

batch_x_lengths = [i.shape[0] for i in x]

padding_masks_x=padding_mask(torch.tensor(batch_x_lengths, dtype=torch.int16), max_len=max( batch_x_lengths))

lengths = [i.shape[0] for i in padding_masks_x]

print(lengths)
##print(max([len(sq) for sq in x]))
padding_masks_x=rnn_utils.pad_sequence(padding_masks_x, batch_first=True, padding_value=0)
# 完成后关闭文件
print(padding_masks_x[0])
data.close()











