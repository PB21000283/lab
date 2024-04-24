import numpy as np
import torch
from pars_args import args


def collate_unsuperv_mask( tra_len, features,masks, max_len=None, vocab=None, add_cls=True):
    
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features] #计算每个特征序列的实际长度，并存储在一个列表中。
    
    if max_len is None:
        max_len = max(lengths)
     #初始化一个全零的TensorX，用于存储填充后的特征序列，形状为(batch_size, max_len, feat_dim)
    X = torch.zeros( tra_len, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    # 初始化一个与X形状相同的布尔型Tensor target_masks，用于标记需要预测的位置。
    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range( tra_len):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks.unsqueeze(-1)

    targets = X.clone()
    targets = targets.masked_fill_(target_masks == 0, vocab.pad_index)

    X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
    X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    return X.long(), targets.long(), target_masks, padding_masks




def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))




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



data_path = 'data/bj_train_set.npz'

data = np.load(data_path, allow_pickle=True)

x=data['train_list']

tra_len = x.shape[0]

print(tra_len)

mask = []

for i in range( 2):
    x_i_array = np.array(x[i])
    ##print( x_i_array)
    x_transformed= np.array([[item] for item in x_i_array])
    ##print( x_transformed)
    mask_i = noise_mask(x_transformed, 0.2, 3, "together", "random", None, True) 
    ##print( mask_i)
    mask.append(mask_i)
    ##print(mask)
##x=x[0]
##x = [np.array(item) for item in x]
x_tensors = [torch.tensor(item) for item in x][0:1]
print(x_tensors)
masked_x, targets, target_masks, padding_masks = collate_unsuperv_mask(
       tra_len , x, mask , max_len=None, vocab=None, add_cls=True)

print(masked_x[0])


