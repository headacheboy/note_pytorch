from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="pytorch/transformer/testing")
    parser.add_argument('--maxSouLen', type=int, default=10)
    parser.add_argument('--maxTarLen', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=20000)
    parser.add_argument('--batchSize', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--encLayers', type=int, default=6)
    parser.add_argument('--decLayers', type=int, default=6)
    parser.add_argument('--embDim', type=int, default=512)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--fc_drop', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.04)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--smooth', type=int, default=0)
    parser.add_argument('--d_in', type=int, default=2048)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    args = parser.parse_args()
    return args