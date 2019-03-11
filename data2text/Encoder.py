import torch
from torch.nn import *
import torch.nn.functional as F

class Encoder(Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.emb = Embedding(self.config.maxAttr, self.config.attrEmbDim)
        self.fc1 = Linear(self.config.attrEmbDim, self.config.dHidden, bias=False)
        self.fc2 = Linear(3*self.config.dHidden, self.config.lstmLayer*self.config.dHidden*2)
        self.drop = Dropout(self.config.drop)

    def forward(self, x):
        x = self.emb(x)
        hid = self.fc1(x).view(-1, self.config.dHidden*3)    # batch, 3*512
        hid = F.tanh(self.fc2(hid)).view(-1, 2, self.config.dHidden*2)
        # batch, 2*1024
        hidLS = torch.split(hid, self.config.dHidden, dim=2)
        # [batch, 2, 512] [batch, 2, 512]
        assert  len(hidLS) == 2
        retLS = []
        for i in range(len(hidLS)):
            retLS.append(hidLS[i].transpose(0, 1).contiguous())
        return retLS