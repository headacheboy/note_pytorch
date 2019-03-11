import torch
from torch.nn import *
import torch.nn.functional as F

class Decoder(Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.drop = Dropout(self.config.drop)
        self.wordEmb = Embedding(self.config.vocab, self.config.wordEmbDim, padding_idx=0)
        self.lstm = LSTM(self.config.wordEmbDim, self.config.dHidden, num_layers=self.config.lstmLayer,
                         batch_first=True, dropout=self.config.drop, bidirectional=False
                         )
        self.wordProj = Linear(self.config.dHidden, self.config.vocab, bias=False)

    def forward(self, encOutputLS, tgt):
        h0, c0 = encOutputLS
        wordEmb = self.wordEmb(tgt)
        out, (hn, cn) = self.lstm(wordEmb, (h0, c0))
        wordProj = self.wordProj(out)
        return wordProj