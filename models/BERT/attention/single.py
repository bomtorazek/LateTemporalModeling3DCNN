import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            masking_value = -1e9 if scores.dtype == torch.float32 else -1e4
            scores = scores.masked_fill(mask == 0, masking_value) 

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    

    
class Attention2(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, value, p_attn, dropout=None):

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
