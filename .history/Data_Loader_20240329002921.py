from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from pars_args import args


class MyData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label #标签
        self.idx = list(range(len(data)))#data的索引列表

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.label[idx], self.idx[idx])
        return tuple_


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
    def collate_fn_neg(data_tuple):##用于在数据加载过程中对数据批次进行特定的预处理。
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)##降序排布
        
        #和上面的mydata对应
        ##data的格式？
        ##TODO，data是一个列表，每个列表是一条轨迹，存储一堆点，列表中的每个元素是其点的编号
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

         
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        traj_poi_pos_tensor = rnn_utils.pad_sequence(traj_poi_pos, batch_first=True, padding_value=0)
        traj_poi_neg_tensor = rnn_utils.pad_sequence(traj_poi_neg, batch_first=True, padding_value=0)
        data_label = rnn_utils.pad_sequence(data_label, batch_first=True, padding_value=0)
        data_neg = rnn_utils.pad_sequence(data_neg, batch_first=True, padding_value=0)
        poi_pos_tensor = rnn_utils.pad_sequence(poi_pos, batch_first=True, padding_value=0)
        poi_neg_tensor = rnn_utils.pad_sequence(poi_neg, batch_first=True, padding_value=0)
        return data, data_neg, data_label, data_length, neg_length, label_length, \
               traj_poi_pos_tensor, traj_poi_neg_tensor, poi_pos_tensor, poi_neg_tensor##TODO，mask
               
    train_x, train_y = load_traindata(train_file)
    neighbors = load_poi_neighbors(poi_file)
    tra_len = train_x.shape[0]
    data_ = MyData(train_x, train_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

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



