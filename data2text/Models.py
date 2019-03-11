import torch
from torch.nn import *
import torch.nn.functional as F
import Encoder
import Decoder

class Data2Text(Module):
    def __init__(self, config):
        super(Data2Text, self).__init__()
        self.config = config
        self.encoder = Encoder.Encoder(config)
        self.decoder = Decoder.Decoder(config)

        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0., std=0.08)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias.data, mean=0., std=0.08)
            elif isinstance(m, torch.nn.LSTM):
                torch.nn.init.normal_(m.all_weights[0][0], mean=0., std=0.08)
                torch.nn.init.normal_(m.all_weights[0][1], mean=0., std=0.08)
                torch.nn.init.normal_(m.all_weights[1][0], mean=0., std=0.08)
                torch.nn.init.normal_(m.all_weights[1][1], mean=0., std=0.08)
        self.apply(weights_init)

    def forward(self, x, tgt):
        encOutput = self.encoder(x)
        wordProj = self.decoder(encOutput, tgt)
        return wordProj