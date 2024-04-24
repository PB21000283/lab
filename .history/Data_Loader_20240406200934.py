from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from pars_args import args



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
def collate_unsuperv_mask_odd (data, max_len=None,  add_cls=True):
    batch_size = len(data)  # list of (seq_length, feat_dim)
    features , masks= zip(*data)
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    # masks related to objective
    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks.unsqueeze(-1)
    targets = X.clone()
    targets = targets.masked_fill_(target_masks == 0,0)
    X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, 3)  # loc -> mask_index,掩码的位置
    X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, 0)  # others -> pad_index，对齐长度
    return X.long(), targets.long(), target_masks, padding_masks


def collate_unsuperv_mask(data, max_len=None,  add_cls=True):
    batch_size = len(data)  # list of (seq_length, feat_dim)
    features , masks= zip(*data)
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









class MyData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label #标签
        self.idx = list(range(len(data)))#data的索引列表

    def __len__(self):
        return len(self.data)
     ##TODO,这里加上noise-masking,对每一条轨迹掩码
    
    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.label[idx], self.idx[idx])
        data_idx_array = np.array(self.data[idx])
        data_idx_transformed= np.array([[item] for item in data_idx_array])
        mask=noise_mask(data_idx_transformed, 0.2, 3, "together", "random", None, True) 
        return tuple_ , torch.LongTensor(mask)##TODO，注意mask的格式


def load_traindata(train_file):
    data = np.load(train_file, allow_pickle=True)
    x = data['train_list']
    y_idx = data['train_y']
    return x, y_idx


def load_valdata(val_file):
    data = np.load(val_file, allow_pickle=True)
    x = data['val_list']
    y_idx = data['train_y']
    return x, y_idx


def load_testdata(val_file):
    data = np.load(val_file, allow_pickle=True)
    x = data['test_list']
    y_idx = data['test_y']
    return x, y_idx


def load_poi_neighbors(poi_file):
    data = np.load(poi_file, allow_pickle=True)
    neighbors = data['neighbors']
    return neighbors


##TODO，train_list共有43803条轨迹
def TrainValueDataLoader(train_file, poi_file, batchsize):
    def collate_fn_neg(data):##用于在数据加载过程中对数据批次进行特定的预处理。
        data_tuple,masks=zip(*data)
        ##TODO!!这里先把降序排布去掉
        ##data_tuple.sort(key=lambda x: len(x[0]), reverse=True)##降序排布
        ##TODO!,这里有问题，features需要转换格式才有shape[0]?已解决
        features=[torch.LongTensor(sq[0]) for sq in data_tuple]
        #和上面的mydata对应
        ##data的格式？
        ##TODO，data是一个列表，每个列表是一条轨迹，存储一堆点，列表中的每个元素是其点的编号
        ##TODO,lbael[i]存放的是data中第i条轨迹对应的正样本轨迹的索引，如data中第一条轨迹的正样本轨迹是data中第40092条轨迹
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        label = [sq[1] for sq in data_tuple] #正样本索引？
        idx_list = [sq[2] for sq in data_tuple] #我也不知道这是啥，随机索引？

        data_neg = []  #生成负样本数据，循环通过随机选择索引生成负样本，确保负样本的索引既不是当前序列的索引也不是对应的正样本索引。
        for idx, d in enumerate(data):#遍历，获得每个元素的索引和值
            neg = np.random.randint(tra_len)
            while neg == idx_list[idx] or neg == label[idx]:#负样本的索引既不是当前序列的索引也不是对应的正样本索引。
                neg = np.random.randint(tra_len)
            data_neg.append(torch.LongTensor(train_x[neg]))

        data_label = []
        for d in label: ##label就是train_y中的数据
            data_label.append(torch.LongTensor(train_x[d]))

        data_length = [len(sq) for sq in data]
        neg_length = [len(sq) for sq in data_neg]
        label_length = [len(sq) for sq in data_label]

        poi_pos = []#轨迹中每个点的正样本
        poi_neg = []#轨迹中每个点的负样本
        for traj in data: #每条轨迹
            pos = []
            neg = []##为轨迹中的每个POI随机选择一个邻居作为正样本，
            ##并将这些正样本收集到一个列表中。
            for poi in traj: #轨迹中的每个点
                pos_id = np.random.randint(len(neighbors[poi])) 
                pos.append(neighbors[poi][pos_id])

                neg_id = np.random.randint(args.nodes)
                while neg_id in neighbors[poi] or neg_id == poi:
                    neg_id = np.random.randint(args.nodes)
                neg.append(neg_id)

            poi_pos.append(torch.LongTensor(pos))
            poi_neg.append(torch.LongTensor(neg))

            ##poi_pos是一个列表，长度与数据集中轨迹的数量相同。
            ##poi_pos中的每个元素都是一个torch.LongTensor，其长度等于对应轨迹中的点数。
            ##每个torch.LongTensor包含了对应轨迹中每个点的正样本索引。

        traj_poi_pos = []#轨迹的正样本
        traj_poi_neg = []#轨迹的负样本
        for traj in data:
            pos = []
            neg = []
            for i in range(len(traj)):
                if i == 0:
                    pos.append(traj[i + 1])
                elif i == len(traj) - 1:
                    pos.append(traj[i - 1])
                else:
                    rand = np.random.rand(1)
                    if rand <= 0.5:
                        pos.append(traj[i - 1])
                    else:
                        pos.append(traj[i + 1])
                neg_id = np.random.randint(args.nodes)
                while neg_id in traj:
                    neg_id = np.random.randint(args.nodes)
                neg.append(neg_id)
            traj_poi_pos.append(torch.LongTensor(pos))
            traj_poi_neg.append(torch.LongTensor(neg))

        ##TODO，在这里加入掩码masking
                ##mask=??
        data_for_mask = list(zip(features, masks))
        masked_x, targets, target_masks, padding_masks= collate_unsuperv_mask(
        data=data_for_mask, max_len=args.seq_len,  add_cls=args.add_cls)
        ##TODO!
        batch_x_lengths = [X.shape[0] for X in data]

        batch_y_lengths = [Y.shape[0] for Y in  data_label]

        batch_n_lengths = [N.shape[0] for N in  data_neg]

        padding_masks_x=padding_mask(torch.tensor(batch_x_lengths, dtype=torch.int16), max_len=max( batch_x_lengths))

        padding_masks_y=padding_mask(torch.tensor(batch_y_lengths, dtype=torch.int16), max_len=max( batch_y_lengths))

        padding_masks_n=padding_mask(torch.tensor(batch_n_lengths, dtype=torch.int16), max_len=max( batch_n_lengths))

        ##TODO，这里是用0填充
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        traj_poi_pos_tensor = rnn_utils.pad_sequence(traj_poi_pos, batch_first=True, padding_value=0)
        traj_poi_neg_tensor = rnn_utils.pad_sequence(traj_poi_neg, batch_first=True, padding_value=0)
        data_label = rnn_utils.pad_sequence(data_label, batch_first=True, padding_value=0)
        data_neg = rnn_utils.pad_sequence(data_neg, batch_first=True, padding_value=0)
        poi_pos_tensor = rnn_utils.pad_sequence(poi_pos, batch_first=True, padding_value=0)
        poi_neg_tensor = rnn_utils.pad_sequence(poi_neg, batch_first=True, padding_value=0)

        ##TODO!!!
        masked_x=rnn_utils.pad_sequence( masked_x, batch_first=True, padding_value=0)
        
        ##TODO!!!,这块应该删掉？
        targets=rnn_utils.pad_sequence(targets, batch_first=True, padding_value=0)
        target_masks=rnn_utils.pad_sequence(target_masks, batch_first=True, padding_value=0)
        padding_masks=rnn_utils.pad_sequence(padding_masks, batch_first=True, padding_value=0)
        padding_masks_x=rnn_utils.pad_sequence(padding_masks_x, batch_first=True, padding_value=0)
        padding_masks_y=rnn_utils.pad_sequence(padding_masks_y, batch_first=True, padding_value=0)
        padding_masks_n=rnn_utils.pad_sequence(padding_masks_n, batch_first=True, padding_value=0)

        return data, data_neg, data_label, data_length, neg_length, label_length, \
               traj_poi_pos_tensor, traj_poi_neg_tensor, poi_pos_tensor, poi_neg_tensor,\
               masked_x, targets, target_masks, padding_masks,padding_masks_x,  padding_masks_y, padding_masks_n
    
    train_x, train_y = load_traindata(train_file)
    neighbors = load_poi_neighbors(poi_file)
    tra_len = train_x.shape[0]
    data_ = MyData(train_x, train_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, ##TODO
                        collate_fn=collate_fn_neg)

    return dataset


def TrainDataValLoader(train_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        label = [torch.LongTensor([sq[1]]) for sq in data_tuple]
        idx_list = [sq[2] for sq in data_tuple]

        data_length = [len(sq) for sq in data]
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        return data, label, data_length, idx_list

    val_x, val_y = load_traindata(train_file)
    data_ = MyData(val_x, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset


def ValValueDataLoader(val_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        label = [sq[1] for sq in data_tuple]
        idx_list = [sq[2] for sq in data_tuple]
        data_length = [len(sq) for sq in data]

        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        return data, label, data_length, idx_list

    val_x, val_y = load_valdata(val_file)
    data_ = MyData(val_x, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset


def TestValueDataLoader(val_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        label = [sq[1] for sq in data_tuple]
        idx_list = [sq[2] for sq in data_tuple]

        data_length = [len(sq) for sq in data]
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        return data, label, data_length, idx_list

    val_x, val_y = load_testdata(val_file)
    data_ = MyData(val_x, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset



