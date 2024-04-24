import torch
import torch.nn as nn
import logging
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from pars_args import args
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np
import math
import copy
import random


def drop_path_func(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)
    


class MultiHeadedAttention(nn.Module):
     ##num_heads：注意力头的数量。模型会将d_model维度的特征分割成num_heads份，每份分别计算注意力。
##d_model：输入特征的维度。
##dim_out：输出特征的维度。
##attn_drop和proj_drop：注意力层和投影层的dropout率。
##add_cls：是否在序列前添加CLS（分类）标记。
##device：模型运行的设备，例如CPU或GPU。
##add_temporal_bias：是否添加时间偏置，用于处理时间序列数据。
##temporal_bias_dim：时间偏置的维度。当设置为特定值时，会通过一个小型神经网络处理时间间隔，以生成动态的时间偏置。
##use_mins_interval：是否使用分钟间隔计算时间偏置，用于处理时间序列数据。
    
    def __init__(self, num_heads, d_model, dim_out, attn_drop=0., proj_drop=0.,
                add_cls=True, device=torch.device('cpu') ):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.add_cls = add_cls
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)##注意力权重计算前的dropout，用于防止过拟合

        self.proj = nn.Linear(d_model, dim_out)##是另一个线性变换层，用于将注意力机制的输出转换到指定的输出维度dim_out。
        self.proj_drop = nn.Dropout(proj_drop)##是应用在投影层输出上的另一个dropout层，其dropout率
        ##为proj_drop。这同样是为了防止过拟合，增强模型的泛化能力。


    def forward(self, x, padding_masks, future_mask=True, output_attentions=False):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T) padding_mask
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:

        """
        batch_size, seq_len, d_model = x.shape

        ##print(x.shape)
        ##print(padding_masks.shape)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l(x) --> (B, T, d_model)
        # l(x).view() --> (B, T, head, d_k)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (x, x, x))]
        # q, k, v --> (B, head, T, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, head, T, T)
        torch.cuda.empty_cache()
        
        #TODO，用-inf填充
        if padding_masks is not None:
            scores.masked_fill_(padding_masks == 0, float('-inf'))

        if future_mask:
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            if self.add_cls:
                mask_postion[:, 0, :] = 0
            scores.masked_fill_(mask_postion, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)  # (B, head, T, T)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)  # (B, head, T, d_k)

        # 3) "Concat" using a view and apply a final linear.
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # (B, T, d_model)
        out = self.proj(out)  # (B, T, N, D)
        out = self.proj_drop(out)
        if output_attentions:
            return out, p_attn  # (B, T, dim_out), (B, head, T, T)
        else:
            return out, None  # (B, T, dim_out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        """Position-wise Feed-Forward Networks
        Args:
            in_features:
            hidden_features:
            out_features:
            act_layer:
            drop:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, drop_path,
                 attn_drop, dropout, type_ln='pre', add_cls=True,
                 device=torch.device('cpu')):
        """

        Args:
            d_model: hidden size of transformer
            attn_heads: head sizes of multi-head attention
            feed_forward_hidden: feed_forward_hidden, usually 4*d_model
            drop_path: encoder dropout rate
            attn_drop: attn dropout rate
            dropout: dropout rate
            type_ln:
        """

        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=attn_heads, d_model=d_model, dim_out=d_model,
                                              attn_drop=attn_drop, proj_drop=dropout, add_cls=add_cls,
                                              device=device )
        self.mlp = Mlp(in_features=d_model, hidden_features=feed_forward_hidden,
                       out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False):
        ##例
        ##批次大小B为2，表示我们一次处理两个序列；
        ##序列长度T为3，表示每个序列包含三个元素或时间步；
        ##模型维度d_model为4，意味着每个序列元素都被表示为一个4维的特征向量。
        
        
        """
     
        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T)
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:
            (B, T, d_model)

        """
        ##pre（先归一化再进入注意力和前馈网络）或post（先进行注意力和前馈网络计算再归一化）
        if self.type_ln == 'pre':
            attn_out, attn_score = self.attention(self.norm1(x), padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions
                                                  )
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(x, padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions
                                                 )
            x = self.norm1(x + self.drop_path(attn_out))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return x, attn_score

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """

        Args:
            hidden: output size of BERT model
            vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
    


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class GRLSTMPooler(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pooling = args.pooling
        self.add_cls = args.add_cls
        self.d_model = args.d_model
        self.linear = MLPLayer(d_model=self.d_model)

    def forward(self, bert_output, padding_masks, hidden_states=None):
        """
        Args:
            bert_output: (batch_size, seq_length, d_model) torch tensor of bert output
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
            hidden_states: list of hidden, (batch_size, seq_length, d_model)
        Returns:
            output: (batch_size, feat_dim)
        """
        token_emb = bert_output  # (batch_size, seq_length, d_model)
        if self.pooling == 'cls':
            if self.add_cls:
                return self.linear(token_emb[:, 0, :])  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'cls_before_pooler':
            if self.add_cls:
                return token_emb[:, 0, :]  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'mean':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == 'max':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            token_emb[input_mask_expanded == 0] = float('-inf')  # Set padding tokens to large negative value
            max_over_time = torch.max(token_emb, 1)[0]
            return max_over_time  # (batch_size, feat_dim)
        elif self.pooling == "avg_first_last":
            first_hidden = hidden_states[0]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (first_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == "avg_top2":
            second_last_hidden = hidden_states[-2]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (second_last_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        else:
            raise ValueError('Error pooling type {}'.format(self.pooling))


##TODO，用transformer编码器
        ##TODO,即使掩码，也要传入所有完整轨迹构造图注意力gat？
class GRLSTM(nn.Module):
    def __init__(self, args, device, batch_first=True):
        super(GRLSTM, self).__init__()
        self.nodes = args.nodes
        self.latent_dim = args.latent_dim   ##128
        self.device = device
        self.batch_first = batch_first
        self.num_heads = args.num_heads
        self.future_mask=args.future_mask
        self.d_model = args.d_model
        self.n_layers =args.n_layers
        self.attn_heads = args.attn_heads
        self.mlp_ratio =args.mlp_ratio
        self.dropout = args.dropout
        self.drop_path = args.drop_path
        self.lape_dim = args.lape_dim
        self.attn_drop = args.attn_drop
        self.type_ln = args.type_ln
        self.seq_len=args.seq_len##最长轨迹长度
        self.add_cls=args.add_cls
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.d_model * self.mlp_ratio


        logging.info('Initializing model: latent_dim=%d' % self.latent_dim)

        self.poi_neighbors = np.load(
            args.poi_file, allow_pickle=True)['neighbors']

        self.poi_features = torch.randn( ##随机初始化一个形状为(self.nodes, self.latent_dim)的张量，用于表示每个节点（POI）的特征或嵌入
            self.nodes, self.latent_dim).to(self.device)
        
        self.gat = GATConv(in_channels=self.latent_dim, out_channels=16,##TODO!,out_channel和d_model应该是一个东西
                           heads=8, dropout=0.1, concat=True)
        ##TODO
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_layers)]  # stochastic depth decay rule
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, attn_heads=self.attn_heads,
                              feed_forward_hidden=self.feed_forward_hidden, drop_path=enc_dpr[i],
                              attn_drop=self.attn_drop, dropout=self.dropout,
                              type_ln=self.type_ln, add_cls=self.add_cls,
                              device=self.device) for i in range(self.n_layers)])
        ##TODO
    def _construct_edge_index(self, batch_x_flatten):
        batch_x_flatten = batch_x_flatten.cpu().numpy()
        neighbors = self.poi_neighbors[batch_x_flatten]
        batch_x_flatten = batch_x_flatten.repeat(neighbors.shape[1])
        neighbors = neighbors.reshape(-1)
        edge_index = np.vstack((neighbors, batch_x_flatten))
        batch_x_flatten = np.vstack((batch_x_flatten, batch_x_flatten))
        edge_index = np.concatenate((batch_x_flatten, edge_index), axis=1)
        edge_index = np.unique(edge_index, axis=1)
        return torch.tensor(edge_index).to(self.device)
    ##每一列代表一条边，数组有两行：
    ##第一行包含起点的索引。
    ##第二行包含终点的索引。
    def forward(self, fps , data_input ,padding_masks,output_hidden_states=False, output_attentions=False,
                pos=True):
        if pos :
            ## batch_x, batch_x_len用来构造图注意力
            batch_x, batch_x_len, _, _ = fps #batch_x可能具有形状(batch_size, seq_len)，这里的batch_size是批次中轨迹的数量，seq_len是轨迹长度。
            batch_x_flatten = batch_x.reshape(-1)
            batch_x_flatten = torch.unique(batch_x_flatten)
            batch_edge_index = self._construct_edge_index(batch_x_flatten)
            embedding_weight = self.gat(self.poi_features, batch_edge_index)##获取每个点的poi
            batch_emb = embedding_weight[data_input]##batch_emb是一个新的张量，形状为(批次大小, 序列长度, 嵌入维度)
            ##print(batch_emb.shape)
            ##print(data_input.shape)
            print(0)
            print(data_input)
            print(batch_emb)
            ##TODO ,之后把 batch_emb传入transformer编码器，padding_mask怎么办？
            ##TODO，只有mask部分的padding_mask才有用
            ##TODO,改padding_mask
            ##TODO!!!
            ##batch_emb_pack = rnn_utils.pack_padded_sequence(
                ##batch_emb, batch_x_len, batch_first=self.batch_first)
            ##TODO!,对齐batch_emb_pack的长度,直接用batch_emb_pack也行？0
            
            padding_masks_input = padding_masks.unsqueeze(1).repeat(1,  self.seq_len, 1).unsqueeze(1)  # (B, 1, T, T)
        # running over multiple transformer blocks
            all_hidden_states = [ batch_emb] if output_hidden_states else None
            all_self_attentions = [] if output_attentions else None
            for transformer in self.transformer_blocks:
                batch_emb, attn_score = transformer.forward(
                x= batch_emb, padding_masks=padding_masks_input,
                future_mask=self.future_mask, output_attentions=output_attentions)  # (B, T, d_model)
            if output_hidden_states:
                all_hidden_states.append( batch_emb)
            if output_attentions:
                all_self_attentions.append(attn_score)
            traj_poi_emb = batch_emb  #原始的轨迹嵌入
            poi_emb = batch_emb
            print(1)
            print()
            print(batch_x_len)
            
            print(batch_emb.shape)
        else :
            batch_x, batch_x_len, poi, batch_traj_poi = fps  #atch_x可能具有形状(batch_size, seq_len)，这里的batch_size是批次中轨迹的数量，seq_len是轨迹长度。
            batch_x_flatten = batch_x.reshape(-1)
            batch_x_flatten = torch.unique(batch_x_flatten)
            batch_edge_index = self._construct_edge_index(batch_x_flatten)
            embedding_weight = self.gat(self.poi_features, batch_edge_index)##获取每个点的poi
            batch_emb = embedding_weight[data_input]##batch_emb是一个新的张量，形状为(批次大小, 序列长度, 嵌入维度)
            ##TODO ,之后把 batch_emb传入transformer编码器，padding_mask怎么办？
            ##TODO
            padding_masks_input = padding_masks.unsqueeze(1).repeat(1,self.seq_len, 1).unsqueeze(1)  # (B, 1, T, T)
        # running over multiple transformer blocks
            ##数据重排

            all_hidden_states = [ batch_emb] if output_hidden_states else None
            all_self_attentions = [] if output_attentions else None
            for transformer in self.transformer_blocks:
                batch_emb, attn_score = transformer.forward(
                x= batch_emb, padding_masks=padding_masks_input,
                future_mask=self.future_mask, output_attentions=output_attentions)  # (B, T, d_model)
            if output_hidden_states:
                all_hidden_states.append( batch_emb)
            if output_attentions:
                all_self_attentions.append(attn_score)

            ##数据再排回来
            


            if batch_traj_poi is not None:
                batch_traj_poi_flatten = batch_traj_poi.reshape(-1)
                batch_traj_poi_flatten = torch.unique(batch_traj_poi_flatten)
                batch_traj_poi_edge_index = self._construct_edge_index(
                    batch_traj_poi_flatten)
                embedding_weight = self.gat(
                    self.poi_features, batch_traj_poi_edge_index)
                traj_poi_emb = embedding_weight[batch_traj_poi]
                print(2)

                print(traj_poi_emb[0])
                print(traj_poi_emb.shape)
            else:
                print(5)
                traj_poi_emb = None

            if poi is not None:
                poi_flatten = poi.reshape(-1)
                poi_flatten = torch.unique(poi_flatten)
                poi_edge_index = self._construct_edge_index(poi_flatten)
                embedding_weight = self.gat(self.poi_features, poi_edge_index)
                poi_emb = embedding_weight[poi]
                print(3)
                print(poi.shape)
                print(poi_emb.shape)
            else:
                print(6)
                poi_emb = None


    
        return  batch_emb, all_hidden_states, all_self_attentions , traj_poi_emb,  poi_emb # (B, T, d_model), list of (B, T, d_model), list of (B, head, T, T)
            ##TODO




class JHL_Model(nn.Module):

    def __init__(self, args, device, batch_first=True):
        super().__init__()
        self.vocab_size = args.vocab_size##TODO,args
        self.d_model =  args.d_model 
        self.pooling = args.pooling##池化方式
        self.grlstm = GRLSTM( args, device, batch_first)#TODO
        self.mask_l = MaskedLanguageModel(self.d_model, self.vocab_size)
        self.pooler = GRLSTMPooler(args)#池化#TODO

    def forward(self, data_forgat,data_forgat_len, poi, batch_traj_poi,data_input,##TODO，argument_methods==false,代表不再进行数据增强
                padding_masks,pos,type,
                output_hidden_states=False, output_attentions=False):
        """
        Args:
            data: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            contra_view2: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            masked_input: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
            graph_dict(dict):
        Returns:
            output: (batch_size, seq_length, vocab_size)
        """
        if self.pooling in ['avg_first_last', 'avg_top2']:
            output_hidden_states = True
        ##d_model 是指输入到 BERT 模型中的特征向量的维度。feat_dim 则是指池化后输出的特征向量的维度。
        if type ==True :##一个是处理数据，一个是掩码
            out_view1, hidden_states1, _,traj_poi_emb,  poi_emb = self.grlstm((data_forgat,data_forgat_len ,poi, batch_traj_poi), data_input,padding_masks=padding_masks, ##把bert改了？
                                                output_hidden_states=output_hidden_states,
                                                output_attentions=output_attentions, pos=pos)  # (B, T, d_model)
            a = self.pooler(bert_output=out_view1, padding_masks=padding_masks,
                                    hidden_states=hidden_states1)  # (B, d_model)
        else :
            mask_output, _, _, traj_poi_emb,  poi_emb = self.grlstm((data_forgat,data_forgat_len, poi, batch_traj_poi), data_input,padding_masks=padding_masks,
                                    output_hidden_states=output_hidden_states,
                                    output_attentions=output_attentions, pos=pos)  # (B, T, d_model)
            a = self.mask_l(mask_output)##(batch_size, seq_length, vocab_size)

        return  a  , traj_poi_emb,  poi_emb#(batch_size, seq_length, vocab_size)

