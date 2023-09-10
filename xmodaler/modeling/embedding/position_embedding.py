# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import torch
from torch import nn

from xmodaler.utils.registry import Registry

POSITION_ENC_REGISTRY = Registry("POSITION_ENC")
POSITION_ENC_REGISTRY.__doc__ = """
Registry for positional encoding
"""

__all__ = ["SinusoidEncoding", "NNEmbeddingEncoding"]

def build_position_encoding(cfg, dim, max_len):
    name = cfg.MODEL.TOKEN_EMBED.POSITION
    return POSITION_ENC_REGISTRY.get(name)(dim, max_len)

@POSITION_ENC_REGISTRY.register()
class SinusoidEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(SinusoidEncoding, self).__init__()   
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() *
                             -(math.log(max_len * 2.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if isinstance(x, int):
            return self.pe[:, x]
        else:
            x_size = x.size(1)
            return self.pe[:, :x_size]

@POSITION_ENC_REGISTRY.register()
class NNEmbeddingEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(NNEmbeddingEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)

    def forward(self, x):
        if isinstance(x, int):
            position_embeddings = self.position_embeddings(torch.tensor([x], dtype=torch.long).cuda())
        else:
            x_size = x.size(1)
            position_ids = torch.arange(x_size, dtype=torch.long, device=x.device)
            position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings

def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


def GridRelationalEmbedding(batch_size, grid_size=7, dim_g=64, wave_len=1000, trignometric_embedding=True):
    # make grid
    a = torch.arange(0, grid_size).float().cuda()
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(batch_size, -1, -1)
    x_min, y_min, x_max, y_max = f(c1), f(c2), f(c3), f(c4)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


def get_normalized_grids(bs, grid_size=7):
    a = torch.arange(0, grid_size).float().cuda()
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(bs, -1, -1) / grid_size
    x_min, y_min, x_max, y_max = f(c1), f(c2), f(c3), f(c4)
    return y_min, x_min, y_max, x_max


def AllRelationalEmbedding(f_g, grid_size=7, dim_g=64, wave_len=1000, trignometric_embedding=True, require_all_boxes=False):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)
    grid_x_min, grid_y_min, grid_x_max, grid_y_max = get_normalized_grids(batch_size, grid_size=grid_size)

    x_min = torch.cat([x_min, grid_x_min], dim=1)
    y_min = torch.cat([y_min, grid_y_min], dim=1)
    x_max = torch.cat([x_max, grid_x_max], dim=1)
    y_max = torch.cat([y_max, grid_y_max], dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    if require_all_boxes:
        all_boxes = torch.cat([x_min, y_min, x_max, y_max], dim=-1)
        return (embedding), all_boxes
    return (embedding)

class GridPESine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        not_mask = (mask == False)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # .permute(0, 3, 1, 2)
        pos = pos.flatten(1, 2)
        return pos

class GridPELearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class PolarRPE(nn.Module):
    def __init__(self, k=3, h=8, d_k=64, d_r=256, window_size = (9, 9)):
        super(PolarRPE, self).__init__()
        Wh, Ww = window_size
        self.h = h
        self.d_k = d_k
        self.num_seq = Wh * Ww
        # num_direction =  4 * k + 1
        num_direction = 4 * k
        num_distance = math.floor(math.sqrt(Wh*Wh + Ww*Ww))

        # define a parameter table of relative position
        # self.relative_direction_table = nn.Embedding(num_direction, d_r)
        # self.relative_distance_table = nn.Embedding(num_distance, d_r)
        self.relative_table = nn.Embedding(num_direction * num_distance, d_r)
        self.projection = nn.Linear(d_r, h * d_k)
        # self.projection = nn.Linear(d_r, h)
        # self.act = nn.ReLU()
        # self.projection = nn.Linear(d_r * 2, h * d_k)

        # get pair-wise relative position index for each token inside the window
        coords_h, coords_w = torch.arange(Wh), torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]), dim=-1)  # Wh, Ww, 2
        coords_flatten = coords.view(-1, 2)  # Wh*Ww, 2
        relative_coords = coords_flatten.unsqueeze(1) - coords_flatten.unsqueeze(0)  # Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords.view(-1, 2).float() # N*N, 2

        # relative_distance_pos
        norm_relative_distance = torch.norm(relative_coords, dim=-1)
        self.relative_distance_pos = norm_relative_distance.int()  # N*N

        # relative_direction_pos
        unit_direction_x = torch.cos(torch.arange(num_direction - 1) * math.pi / 2 / k)
        unit_direction_y = torch.sin(torch.arange(num_direction - 1) * math.pi / 2 / k)
        unit_direction = torch.stack([unit_direction_x, unit_direction_y])  # 2, 4k

        relative_direction = torch.matmul(relative_coords, unit_direction)
        self.relative_direction_pos = torch.argmax(relative_direction, dim=-1)  # N*N
        # relative_direction_pos = relative_direction_pos.masked_fill(norm_relative_distance == 0, num_direction-1)

        self.relative_pos = self.relative_direction_pos * num_distance + self.relative_distance_pos
        self.relative_pos = self.relative_pos.masked_fill(norm_relative_distance == 0, num_direction * num_distance - 1)

        self.init_weights()

    def init_weights(self):
        # nn.init.uniform_(self.relative_direction_table.weight, b=0.2)
        # nn.init.uniform_(self.relative_distance_table.weight, b=0.2)
        nn.init.uniform_(self.relative_table.weight, b=0.2)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

    def forward(self, input):

        bs = input.size(0)
        # self.relative_direction_pos = self.relative_direction_pos.to(input.device)
        # self.relative_distance_pos = self.relative_distance_pos.to(input.device)
        self.relative_pos = self.relative_pos.to(input.device)

        # direction + distance
        # relative_direction_emb = self.relative_direction_table(self.relative_direction_pos)
        # relative_distance_emb = self.relative_distance_table(self.relative_distance_pos)
        # relative_emb = relative_direction_emb + relative_distance_emb # (n*n, d_r)

        # relative_emb = torch.cat([relative_direction_emb, relative_distance_emb], dim=-1)  # (n*n, d_r * 2)
        relative_emb = self.relative_table(self.relative_pos)
        relative_emb = self.projection(relative_emb).view(-1, self.h, self.d_k)  # (n*n, h, d_k)

        # relative_emb = self.projection(relative_emb)  # (n*n, h)
        # relative_emb = self.act(relative_emb)

        # direction
        # relative_direction_emb = self.relative_direction_table(self.relative_direction_pos) # (n*n, d_r)
        # relative_emb = self.projection(relative_direction_emb).view(-1, self.h, self.d_k)  # (n*n, h, d_k)

        # distance
        # relative_distance_emb = self.relative_distance_table(self.relative_distance_pos) # (n*n, d_r)
        # relative_emb = self.projection(relative_distance_emb).view(-1, self.h, self.d_k)  # (n*n, h, d_k)

        relative_emb = relative_emb.view(self.num_seq, self.num_seq, self.h, self.d_k).permute(2, 0, 1, 3)
        relative_emb = relative_emb.unsqueeze(0).expand(bs, self.h, self.num_seq, self.num_seq, self.d_k)  # (b_s, h, n, n, d_k)
        
        # relative_emb = relative_emb.view(self.num_seq, self.num_seq, self.h).permute(2, 0, 1)
        # relative_emb = relative_emb.unsqueeze(0).expand(bs, self.h, self.num_seq, self.num_seq)  # (b_s, h, n, n)

        return relative_emb