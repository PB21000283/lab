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


def geom_noise_mask_single(L, lm, masking_ratio):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]
    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep_mask



def noise_mask(X, masking_ratio, lm=3, mode='together', distribution='random', exclude_feats=None, add_cls=True):
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)
    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    elif distribution == 'random':  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])
    else:
        mask = np.ones(X.shape, dtype=bool)
    if add_cls:
        mask[0] = True  # CLS at 0, set mask=1
    return mask

##TODO!!!,x填充是啥？对于mask来说，x应该不重要？重要的是padding,mask？所以mask_input直接用batch_x就可以，重要的是targets.long(), target_masks, padding_masks
##TODO!!!,还是不对，第三个维度相当于只有一个点

def collate_unsuperv_mask(data, max_len=None,  add_cls=True):
    batch_size = len(data)  # list of (seq_length, feat_dim)
    features , masks= zip(*data)
    #print(masks)
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    # masks related to objective
    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end] = features[i][:end]
        target_masks[i, :end] = masks[i][:end]
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks
    targets = X.clone()
    targets = targets.masked_fill_(target_masks == 0,0)

    return X.long(), targets.long(), target_masks, padding_masks














data_path = 'data/bj_train_set.npz'
data = np.load(data_path, allow_pickle=True)
##y = [torch.LongTensor(sq) for sq in data['train_y']]
##print(y[0])  # 打印第一个转换后的张量查看是否成功
x=[torch.LongTensor(sq) for sq in data['train_list']]

batch_x_lengths = [i.shape[0] for i in x]

padding_masks_x=padding_mask(torch.tensor(batch_x_lengths, dtype=torch.int16), max_len=max( batch_x_lengths))


##print(max([len(sq) for sq in x]))
padding_masks_x=rnn_utils.pad_sequence(padding_masks_x, batch_first=True, padding_value=0)
# 完成后关闭文件
##lengths = [i.shape[0] for i in padding_masks_x]

##print(lengths)

##print(padding_masks_x[0])
y=[sq for sq in data['train_list']]

print(y[0])

data_idx_array = np.array(y[0])

data_idx_transformed= np.array([[item] for item in data_idx_array])

mask=noise_mask(data_idx_transformed, 0.2, 3, "together", "random", None, True)


data_tuple=y[0:2]

print(data_tuple)

features=[torch.LongTensor(sq) for sq in data_tuple]

print(features)

data_for_mask = list(zip(features, torch.LongTensor(mask)))

masked_x, targets, target_masks, padding_masks= collate_unsuperv_mask(
        data=data_for_mask, max_len=682 , add_cls=True )


print(target_masks)


data.close()











