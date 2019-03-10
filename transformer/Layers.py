import MultiHead
import torch
from torch.nn import *
import torch.nn.functional as F

class EncodeLayer(Module):
    def __init__(self, config):
        super(EncodeLayer, self).__init__()
        self.config = config
        self.selfAtt = MultiHead.MultiHead(config)
        self.fc = MultiHead.FC(config)

    def forward(self, x, non_pad_mask=None, attn_mask=None):
        encOutput = self.selfAtt(x, x, x, attn_mask)

        encOutput *= non_pad_mask       # what's non_pad_mask?

        encOutput = self.fc(encOutput)
        encOutput *= non_pad_mask

        return encOutput

class DecodeLayer(Module):
    def __init__(self, config):
        super(DecodeLayer, self).__init__()
        self.config = config
        self.selfAtt = MultiHead.MultiHead(config)
        self.encAtt = MultiHead.MultiHead(config)
        self.fc = MultiHead.FC(config)

    def forward(self, x, encOutput, non_pad_mask=None, attn_mask = None, dec_enc_mask=None):
        decOutput = self.selfAtt(x, x, x, attn_mask)
        decOutput *= non_pad_mask

        decOutput = self.encAtt(decOutput, encOutput, encOutput, dec_enc_mask)  # decOutput are queries
        decOutput *= non_pad_mask

        decOutput = self.fc(decOutput)
        decOutput *= non_pad_mask

        return decOutput
