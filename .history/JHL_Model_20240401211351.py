import torch
import torch.nn as nn
import logging
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from pars_args import args
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np


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
                 add_cls=True, device=torch.device('cpu'), add_temporal_bias=False,
                 temporal_bias_dim=64, use_mins_interval=False):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.add_cls = add_cls
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_mins_interval = use_mins_interval

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)##注意力权重计算前的dropout，用于防止过拟合

        self.proj = nn.Linear(d_model, dim_out)##是另一个线性变换层，用于将注意力机制的输出转换到指定的输出维度dim_out。
        self.proj_drop = nn.Dropout(proj_drop)##是应用在投影层输出上的另一个dropout层，其dropout率
        ##为proj_drop。这同样是为了防止过拟合，增强模型的泛化能力。

        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
            elif self.temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T) padding_mask
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:

        """
        batch_size, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l(x) --> (B, T, d_model)
        # l(x).view() --> (B, T, head, d_k)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (x, x, x))]
        # q, k, v --> (B, head, T, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, head, T, T)

        if self.add_temporal_bias:
            if self.use_mins_interval:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) +
                    (batch_temporal_mat / torch.tensor(60.0).to(self.device)))
            else:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + batch_temporal_mat)
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                batch_temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(
                    self.temporal_mat_bias_1(batch_temporal_mat.unsqueeze(-1)),
                    negative_slope=0.2)).squeeze(-1)  # (B, T, T)
            if self.temporal_bias_dim == -1:
                batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias.expand((1, seq_len, seq_len))
            batch_temporal_mat = batch_temporal_mat.unsqueeze(1)  # (B, 1, T, T)
            scores += batch_temporal_mat  # (B, 1, T, T)

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
                 device=torch.device('cpu'), add_temporal_bias=False,
                 temporal_bias_dim=64, use_mins_interval=False):
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
                                              device=device, add_temporal_bias=add_temporal_bias,
                                              temporal_bias_dim=temporal_bias_dim,
                                              use_mins_interval=use_mins_interval)
        self.mlp = Mlp(in_features=d_model, hidden_features=feed_forward_hidden,
                       out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
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
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(x, padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
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
    
class BERTPooler(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.pooling = self.config.get('pooling', 'mean')
        self.add_cls = self.config.get('add_cls', True)
        self.d_model = self.config.get('d_model', 768)
        self.linear = MLPLayer(d_model=self.d_model)

        self._logger = getLogger()
        self._logger.info("Building BERTPooler model")

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
class GRLSTM(nn.Module):
    def __init__(self, args, device, batch_first=True):
        super(GRLSTM, self).__init__()
        self.nodes = args.nodes
        self.latent_dim = args.latent_dim   ##128
        self.device = device
        self.batch_first = batch_first
        self.num_heads = args.num_heads

        logging.info('Initializing model: latent_dim=%d' % self.latent_dim)

        self.poi_neighbors = np.load(
            args.poi_file, allow_pickle=True)['neighbors']

        self.poi_features = torch.randn( ##随机初始化一个形状为(self.nodes, self.latent_dim)的张量，用于表示每个节点（POI）的特征或嵌入
            self.nodes, self.latent_dim).to(self.device)

        self.lstm_list = nn.ModuleList([
            nn.LSTM(input_size=self.latent_dim, hidden_size=self.latent_dim,
                    num_layers=1, batch_first=True)
            for _ in range(args.lstm_layers)
        ])

        self.gat = GATConv(in_channels=self.latent_dim, out_channels=16,
                           heads=8, dropout=0.1, concat=True)
        
        ##TODO
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_layers)]  # stochastic depth decay rule
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, attn_heads=self.attn_heads,
                              feed_forward_hidden=self.feed_forward_hidden, drop_path=enc_dpr[i],
                              attn_drop=self.attn_drop, dropout=self.dropout,
                              type_ln=self.type_ln, add_cls=self.add_cls,
                              device=self.device, add_temporal_bias=self.add_temporal_bias,
                              temporal_bias_dim=self.temporal_bias_dim,
                              use_mins_interval=self.use_mins_interval) for i in range(self.n_layers)])
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

    def forward(self, fps, pos=True):
        if pos :
            batch_x, batch_x_len, _, _ = fps #atch_x可能具有形状(batch_size, seq_len)，这里的batch_size是批次中轨迹的数量，seq_len是轨迹长度。
            batch_x_flatten = batch_x.reshape(-1)
            batch_x_flatten = torch.unique(batch_x_flatten)
            batch_edge_index = self._construct_edge_index(batch_x_flatten)
            embedding_weight = self.gat(self.poi_features, batch_edge_index)##获取每个点的poi
            batch_emb = embedding_weight[batch_x]##batch_emb是一个新的张量，形状为(批次大小, 序列长度, 嵌入维度)
            ##TODO ,之后把 batch_emb传入transformer编码器，padding_mask怎么办？
            ##TODO，只有mask部分的padding_mask才有用
            batch_emb_pack = rnn_utils.pack_padded_sequence(
                batch_emb, batch_x_len, batch_first=self.batch_first)
            
            ##TODO
            padding_masks_input = padding_masks.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (B, 1, T, T)
        # running over multiple transformer blocks
            all_hidden_states = [ batch_emb_pack] if output_hidden_states else None
            all_self_attentions = [] if output_attentions else None
            for transformer in self.transformer_blocks:
                batch_emb_pack, attn_score = transformer.forward(
                x= batch_emb_pack, padding_masks=padding_masks_input,
                future_mask=self.future_mask, output_attentions=output_attentions,
                batch_temporal_mat=batch_temporal_mat)  # (B, T, d_model)
            if output_hidden_states:
                all_hidden_states.append( batch_emb_pack)
            if output_attentions:
                all_self_attentions.append(attn_score)
        return  batch_emb_pack, all_hidden_states, all_self_attentions  # (B, T, d_model), list of (B, T, d_model), list of (B, head, T, T)
            ##TODO




class JHL_Model(nn.Module):

    def __init__(self, args, data_feature):
        super().__init__()

        self.vocab_size = args.vocab_size##TODO
        self.usr_num = data_feature.get('usr_num')
        self.d_model =  args.d_model 
        self.pooling = self.config.get('pooling', 'mean')

        self.grlstm = GRLSTM(config, data_feature)

        self.mask_l = MaskedLanguageModel(self.d_model, self.vocab_size)
        self.pooler = BERTPooler(config, data_feature)#池化

    def forward(self, contra_view1, contra_view2, argument_methods1,
                argument_methods2, masked_input, padding_masks,
                batch_temporal_mat, padding_masks1=None, padding_masks2=None,
                batch_temporal_mat1=None, batch_temporal_mat2=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        """
        Args:
            contra_view1: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            contra_view2: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            masked_input: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
            graph_dict(dict):
        Returns:
            output: (batch_size, seq_length, vocab_size)
        """
        if self.pooling in ['avg_first_last', 'avg_top2']:
            output_hidden_states = True
        if padding_masks1 is None:
            padding_masks1 = padding_masks
        if padding_masks2 is None:
            padding_masks2 = padding_masks
        if batch_temporal_mat1 is None:
            batch_temporal_mat1 = batch_temporal_mat
        if batch_temporal_mat2 is None:
            batch_temporal_mat2 = batch_temporal_mat

        out_view1, hidden_states1, _ = self.grlstm(x=contra_view1, padding_masks=padding_masks1, ##把bert改了？
                                                 batch_temporal_mat=batch_temporal_mat1,
                                                 argument_methods=argument_methods1, graph_dict=graph_dict,
                                                 output_hidden_states=output_hidden_states,
                                                 output_attentions=output_attentions)  # (B, T, d_model)
        pool_out_view1 = self.pooler(bert_output=out_view1, padding_masks=padding_masks1,
                                     hidden_states=hidden_states1)  # (B, d_model)

        out_view2, hidden_states2, _ = self.grlstm(x=contra_view2, padding_masks=padding_masks2,
                                                 batch_temporal_mat=batch_temporal_mat2,
                                                 argument_methods=argument_methods2, graph_dict=graph_dict,
                                                 output_hidden_states=output_hidden_states,
                                                 output_attentions=output_attentions)  # (B, T, d_model)
        pool_out_view2 = self.pooler(bert_output=out_view2, padding_masks=padding_masks2,
                                     hidden_states=hidden_states2)  # (B, d_model)

        bert_output, _, _ = self.grlstm(x=masked_input, padding_masks=padding_masks,
                                      batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                                      output_hidden_states=output_hidden_states,
                                      output_attentions=output_attentions)  # (B, T, d_model)
        a = self.mask_l(bert_output)
        return pool_out_view1, pool_out_view2, a

