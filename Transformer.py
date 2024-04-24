import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import copy
import random

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
        return x, attn_score##可能的注意力权重