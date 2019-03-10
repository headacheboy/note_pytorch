import numpy as np
import torch
from torch.nn import *
import torch.nn.functional as F

class MultiHead(Module):
    def __init__(self, config):
        super(MultiHead, self).__init__()
        self.config = config
        self.qLinear = Linear(config.d_model, config.d_model)
        self.kLinear = Linear(config.d_model, config.d_model)
        self.vLinear = Linear(config.d_model, config.d_model)
        self.lNorm = LayerNorm(config.d_model)
        self.drop = Dropout(config.attn_drop)
        self.fc = Linear(config.d_model, config.d_model)

    def forward(self, queries, keys, values, mask=None):
        '''

        :param queries:
        :param keys:
        :param values:
        :param num_heads:
        :param drop: drop rate
        :param masked: True for decoder, masked the words behind the predicting word
        :return:
        '''
        batchSize = queries.size(0)
        Q = self.qLinear(queries)   # [N, L_q, d_q]
        K = self.kLinear(keys)      # [N, L_k, d_k]
        V = self.vLinear(values)    # [N, L_v, d_v]

        # split and concat
        Q_ = torch.cat(torch.split(Q, self.config.d_model//self.config.heads, dim=2), dim=0)     # [N*h, L_q, d_q/h]
        K_ = torch.cat(torch.split(K, self.config.d_model//self.config.heads, dim=2), dim=0)     # [N*h, L_k, d_k/h]
        V_ = torch.cat(torch.split(V, self.config.d_model//self.config.heads, dim=2), dim=0)     # [N*h, L_v, d_v/h]

        # Attention
        d_k = Q.shape[-1]
        outputs = torch.matmul(Q_, K_.transpose(1, 2))  # [N*h, L_q, L_k]
        outputs /= d_k**0.5

        # masking: key mask by setting negative infinite value
        # masking: query mask by setting 0      different masking due to key is the last dimension
        # therefore, first masking key  and if necessary, mask future position, then softmax. after that, mask query
        # here mask is a 3d tensor  [N, L_q, L_k]

        if mask is not None:
            mask = mask.repeat(self.config.heads, 1, 1) # dim 0 repeat num_heads times, while other dims do not repeat
            outputs = outputs.masked_fill(mask, -np.inf)

        outputs = F.softmax(outputs, dim=2)
        outputs = self.drop(outputs)
        V_ = torch.matmul(outputs, V_)

        # concat and fc
        V_ = torch.cat(torch.split(V_, batchSize, dim=0), dim=2)

        V_ = self.drop(F.relu(self.fc(V_)))

        # residual
        values = self.lNorm(values + V_)

        return values


class FC(Module):
    def __init__(self, config):
        super(FC, self).__init__()
        self.config = config
        self.fc1 = Linear(config.d_model, config.d_in)
        self.fc2 = Linear(config.d_in, config.d_model)
        self.drop = Dropout(config.fc_drop)
        self.lNorm = LayerNorm(config.d_model)

    def forward(self, x):
        outputs = self.fc2(F.relu(self.fc1(x)))
        outputs = self.drop(outputs)
        ret = self.lNorm(x+outputs)

        return ret