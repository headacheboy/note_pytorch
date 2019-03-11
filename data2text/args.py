from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="pytorch/seq2seq/testing")
    parser.add_argument('--maxLen', type=int, default=65)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dHidden', type=int, default=512)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--maxAttr', type=int, default=99936)
    parser.add_argument('--attrEmbDim', type=int, default=64)
    parser.add_argument('--seqHid', type=int, default=512)
    parser.add_argument('--wordEmbDim', type=int, default=512)
    parser.add_argument('--lstmLayer', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--validStep', type=int, default=1)
    parser.add_argument('--decayEpoch', type=int, default=1)
    parser.add_argument('--lr_decay_thre', type=int, default=10)
    parser.add_argument('--maxClip', type=float, default=5.0)
    args = parser.parse_args()
    return args