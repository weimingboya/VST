"""
From original at https://github.com/aimagelab/meshed-memory-transformer/blob/master/models/transformer/attention.py
Original copyright of AImageLab code below, modifications by Yehao Li and Jianjie Luo Copyright 2021.	
"""
# Copyright (c) 2019, AImageLab
import torch
import torch.nn as nn
import numpy as np 
from einops import rearrange

__all__ = [
    "MultiHeadAttention", 
    "ScaledDotProductAttention", 
    "ScaledDotProductAttentionMemory",
    "OSAttention"
]

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(
        self,
        *, 
        d_model: int, 
        d_k: int, 
        d_v: int, 
        h: int):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :param history_states: Use history_states as key and value to speed up inference.
        :return:
        '''
        attention_weights = kwargs.pop('attention_weights', None)
        history_states = kwargs.pop('history_states', None)

        b_s, nq = queries.shape[:2]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if history_states is not None:
            nk = history_states.shape[1]
            k = self.fc_k(history_states).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(history_states).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        else:
            nk = keys.shape[1]
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out, att

class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(
        self, 
        *,
        d_model: int, 
        d_k: int, 
        d_v: int, 
        num_head: int,  # number of heads
        dropout: float,
        shortcut=True,
        attention_module=None, 
        attention_module_kwargs=None
    ): 

        super(MultiHeadAttention, self).__init__()
        
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=num_head, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=num_head)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=num_head)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.shortcut = shortcut

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        out, _ = self.attention(queries, keys, values, attention_mask, **kwargs)
        out = self.dropout(out)
        if self.shortcut:
            out = queries + out
        out = self.layer_norm(out)
        return out

class ScaledDotProductAttentionMemory(ScaledDotProductAttention):
    '''
    Scaled dot-product attention with memory
    '''

    def __init__(
        self, 
        *,
        d_model: int, 
        d_k: int, 
        d_v: int, 
        h: int, 
        m: int):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param m: Number of memory slots
        '''
        super(ScaledDotProductAttentionMemory, self).__init__(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.m_k = nn.Parameter(torch.FloatTensor(1, m, h * d_k))
        self.m_v = nn.Parameter(torch.FloatTensor(1, m, h * d_v))

        self.m = m

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.m_k, 0, 1 / self.d_k)
        nn.init.normal_(self.m_v, 0, 1 / self.m)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        attention_weights = kwargs.pop('attention_weights', None)

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        m_k = np.sqrt(self.d_k) * self.m_k.expand(b_s, self.m, self.h * self.d_k)
        m_v = np.sqrt(self.m) * self.m_v.expand(b_s, self.m, self.h * self.d_v)

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = torch.cat([self.fc_k(keys), m_k], 1).view(b_s, nk + self.m, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = torch.cat([self.fc_v(values), m_v], 1).view(b_s, nk + self.m, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = torch.cat([att[:, :, :, :nk] * attention_weights, att[:, :, :, nk:]], -1)
        if attention_mask is not None:
            att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out, att

class TopDownAttention(ScaledDotProductAttention):
    '''
    TopDownAttention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        super(TopDownAttention, self).__init__(d_model=d_model, d_k=d_k, d_v=d_v, h=h)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        '''
        queries: (bs, n_q, d_model)
        keys: (bs, n_q, n_k, d_model)
        values: (bs, n_q, n_k, d_model)
        '''

        b_s, nq = queries.shape[:2]
        nk = keys.shape[-2]

        q = self.fc_q(queries)  # (b_s, nq, h*d_k)
        q = rearrange(q, 'b q (h d) -> b h q d', h=self.h, d=self.d_k)  # (b_s, h, nq, d_k)
        q = q.unsqueeze(-2).expand(b_s, self.h, nq, nk, self.d_k)  # (b_s, h, nq, nk, d_k)

        k = self.fc_k(keys.view(b_s, nq*nk, -1)) # (b_s, nq*nk, h*d_k)
        v = self.fc_v(values.view(b_s, nq*nk, -1)) # (b_s, nq*nk, h*d_v)
        k = rearrange(k, 'b (q k) (h d) -> b h q k d', q=nq, k=nk, h=self.h, d=self.d_k) # (b_s, h, nq, nk, d_k)
        v = rearrange(v, 'b (q k) (h d) -> b h q k d', q=nq, k=nk, h=self.h, d=self.d_v) # (b_s, h, nq, nk, d_v)

        att = torch.sum(q * k, dim=-1) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -1e9)

        att = torch.softmax(att, -1)  # (b_s, h, nq, nk)
        att = att.unsqueeze(-1).expand(*att.shape, self.d_v) # (b_s, h, nq, nk, d_v)
        out = torch.sum(att * v, dim=-2) # (b_s, h, nq, d_v)
        out = rearrange(out, 'b h q d -> b q (h d)')  # (b_s, nq, h*d_v)

        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out, att

class ScaledDotProductWithBoxAttention(ScaledDotProductAttention):
    '''
    ScaledDotProductWithBoxAttention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        super(ScaledDotProductWithBoxAttention, self).__init__(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        attention_weights = kwargs.pop('attention_weights', None)
        relative_geometry_weights = kwargs.pop('relative_geometry_weights', None)

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        w_g = relative_geometry_weights
        w_a = att

        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        w_mn = torch.softmax(w_mn, -1)  ## bs * 8 * r * r

        att = self.dropout(w_mn)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out, att


class RPEAttention(nn.Module):
    '''
    Normalized Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        super(RPEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.fc_gq = nn.Linear(d_model, h * d_k)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        gq = self.fc_gq(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).unsqueeze(-1)  # (b_s, h, nq, d_k, 1)
        geometric_bias = torch.matmul(attention_weights, gq).squeeze(-1)  # (b_s, h, nq, nk)
        att = att + geometric_bias

        # k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
        # q = q.unsqueeze(-2).expand(b_s, self.h, nq, nk, self.d_k)  # (b_s, h, nq, nk, d_k)
        # k = k.unsqueeze(-3).expand(b_s, self.h, nq, nk, self.d_k)  # (b_s, h, nq, nk, d_k)
        # q = q + attention_weights
        # att = torch.sum(q * k, dim=-1) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -1e9)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out, att