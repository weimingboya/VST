# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import math
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from xmodaler.modeling.embedding.position_embedding import PolarRPE
from xmodaler.modeling.layers.bert import BertIntermediate, BertOutput, BertSelfOutput
from .build import ENCODER_REGISTRY

__all__ = ["VanillaEncoder"]

class BertSelfAttention(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob
    ):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.fc_gq = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "num_attention_heads": cfg.MODEL.BERT.NUM_ATTENTION_HEADS,
            "attention_probs_dropout_prob": cfg.MODEL.BERT.ATTENTION_PROBS_DROPOUT_PROB
        }

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)

        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)
        #return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, rpe, attention_mask, history_states=None):
        mixed_query_layer = self.query(hidden_states)
        
        if history_states is not None:            
            mixed_key_layer = self.key(history_states)
            mixed_value_layer = self.value(history_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        gq = self.transpose_for_scores(self.fc_gq(hidden_states)).unsqueeze(-1)  # (b_s, h, nq, d_k, 1)
        geometric_bias = torch.matmul(rpe, gq).squeeze(-1)  # (b_s, h, nq, nk)
        attention_scores = attention_scores + geometric_bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()
        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs

class BertAttention(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        bert_self_attention,
        bert_self_output
    ):
        super(BertAttention, self).__init__()
        self.self = bert_self_attention
        self.output = bert_self_output

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_self_attention": BertSelfAttention(cfg),
            "bert_self_output": BertSelfOutput(cfg),
        }

    def forward(self, input_tensor, rpe, attention_mask, history_states=None):
        self_output, attention_probs = self.self(input_tensor, rpe, attention_mask, history_states)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs

class BertLayer(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        bert_attention, 
        bert_intermediate,
        bert_output
    ):
        super(BertLayer, self).__init__()
        self.attention = bert_attention
        self.intermediate = bert_intermediate
        self.output = bert_output

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_attention": BertAttention(cfg),
            "bert_intermediate": BertIntermediate(cfg),
            "bert_output": BertOutput(cfg)
        }

    def forward(self, hidden_states, rpe, attention_mask=None, history_states=None):
        attention_output, attention_probs = self.attention(hidden_states, rpe, attention_mask, history_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

@ENCODER_REGISTRY.register()
class VanillaEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        hidden_size: int,
        bert_layers,
    ):
        super(VanillaEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = bert_layers
        self.rpe = PolarRPE(k=3, h=8, d_k=64, d_r=hidden_size // 2, window_size=(9, 9))

    @classmethod
    def from_config(cls, cfg):
        bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        )
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "bert_layers": bert_layers,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            grid_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK]

            rpe = self.rpe(grid_feats)

            for layer_module in self.layers:
                grid_feats, _ = layer_module(grid_feats, rpe)

            ret.update( {kfg.ATT_FEATS_WO_MASK: grid_feats })
        return ret