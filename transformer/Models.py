import numpy as np
import Layers
import torch
from torch.nn import *

def getPosEmb(length, hid, padding_idx=None):
    def cal_angle(pos, hid_idx):
        return pos / np.power(10000, 2*(hid_idx // 2) / hid)
    def get_posi(pos):
        return [cal_angle(pos, i) for i in range(hid)]

    table = np.array([get_posi(pos_i) for pos_i in range(length)])

    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])

    if padding_idx is not None:
        table[padding_idx] = 0.

    return torch.FloatTensor(table)

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1).cuda()    # batch * L * 1    tensor.ne(0) replace non-zero value by one

def get_subseq_mask(seq):
    batch, length = seq.size()
    subsequent_mask = torch.triu(torch.ones((length, length), dtype=torch.uint8), diagonal=1)   # 上三角
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch, -1, -1)

    return subsequent_mask.cuda()

def get_attn_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)  # batch * L       tensor.eq(0) set 0 to 1 and set non-0 to 0
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)      # why eq? masked_fill(mask, -np.inf) set
                                                                        # non-zero value to -np.inf
    return padding_mask.cuda()

class Encoder(Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.wordEmb = Embedding(config.srcVocabSize, config.embDim, padding_idx=0)
        self.posEmb = Embedding.from_pretrained(getPosEmb(config.maxSouLen, config.d_model, padding_idx=0), freeze=True)
        self.layers = ModuleList([Layers.EncodeLayer(config) for _ in range(config.encLayers)])

    def forward(self, x, pos):
        # prepare masks
        selfAttMask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask = get_non_pad_mask(x)

        encOutput = self.wordEmb(x) + self.posEmb(pos)
        for encLayer in self.layers:
            encOutput = encLayer(encOutput, non_pad_mask=non_pad_mask, attn_mask=selfAttMask)

        return encOutput

class Decoder(Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.wordEmb = Embedding(config.tarVocabSize, config.embDim, padding_idx=0)
        self.posEmb = Embedding.from_pretrained(getPosEmb(config.maxTarLen, config.d_model, padding_idx=0), freeze=True)
        self.layers = ModuleList([Layers.DecodeLayer(config) for _ in range(config.decLayers)])

    def forward(self, x, pos, srcX, encOutput):
        # prepare masks
        non_pad_mask = get_non_pad_mask(x)      # what's the input x of decoder?
        slf_attn_mask_subseq = get_subseq_mask(x)                           # 上三角能mask住q的第q位和k的第k位的att  q>k
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        slf_attn_mask = (slf_attn_mask_subseq+slf_attn_mask_keypad).gt(0)   # since value 1 will be set to -np.inf,
                                                                            # therefore the plus operation can combine
                                                                            # two masks into one
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=srcX, seq_q=x)

        # --forward
        decOutput = self.wordEmb(x) + self.posEmb(pos)

        for layer in self.layers:
            decOutput = layer(decOutput, encOutput, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)

        return decOutput


class Transformer(Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.word_proj = Linear(config.d_model, config.tarVocabSize, bias=False)
        def weights_init(m):
            if isinstance(m, Linear):
                torch.nn.init.normal_(m.weight.data, mean=0., std=0.1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.01)
            elif isinstance(m, LSTM):
                torch.nn.init.normal_(m.all_weights[0][0], mean=0., std=0.1)
                torch.nn.init.normal_(m.all_weights[0][1], mean=0., std=0.1)
                torch.nn.init.normal_(m.all_weights[1][0], mean=0., std=0.1)
                torch.nn.init.normal_(m.all_weights[1][1], mean=0., std=0.1)
        self.apply(weights_init)    # self.apply is inherited from torch.nn.Module, which will recursively apply
                                    # weights_init function to all modules in Transformer

        assert config.embDim == config.d_model  # ensure residual connection

    def forward(self, src_seq, src_pos, tar_seq, tar_pos):

        encOutput = self.encoder(src_seq, src_pos)
        decOutput = self.decoder(tar_seq, tar_pos, src_seq, encOutput)
        seqLogit = self.word_proj(decOutput)    # batch * L * tarVocabSize

        #return seqLogit.view(-1, seqLogit.size(2))  # ?
        return seqLogit
