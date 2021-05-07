import torch

# this function average the subword representation when these subwords consist of a word. e.g. vector[victim] = vector[vic##] + vector[tim]

def average_pooling(encoded_layers, token_subword_index, pad_token_id):
    batch_size, num_tokens, num_subwords = token_subword_index.size()
    batch_index = torch.arange(batch_size).view(-1, 1, 1).type_as(token_subword_index)
    token_index = torch.arange(num_tokens).view(1, -1, 1).type_as(token_subword_index)
    _, num_total_subwords, hidden_size = encoded_layers.size()
    expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
        batch_size, num_tokens, num_total_subwords, hidden_size)
    # [batch_size, num_tokens, num_subwords, hidden_size]
    token_reprs = expanded_encoded_layers[batch_index, token_index, token_subword_index]
    subword_pad_mask = token_subword_index.eq(pad_token_id).unsqueeze(3).expand(
        batch_size, num_tokens, num_subwords, hidden_size)
    token_reprs.masked_fill_(subword_pad_mask, 0)
    # [batch_size, num_tokens, hidden_size]
    sum_token_reprs = torch.sum(token_reprs, dim=2)
    # [batch_size, num_tokens]
    num_valid_subwords = token_subword_index.ne(pad_token_id).sum(dim=2)
    pad_mask = num_valid_subwords.eq(pad_token_id).long()
    # Add ones to arrays where there is no valid subword.
    divisor = (num_valid_subwords + pad_mask).unsqueeze(2).type_as(sum_token_reprs)
    # [batch_size, num_tokens, hidden_size]
    avg_token_reprs = sum_token_reprs / divisor
    return avg_token_reprs

if __name__ == '__main__':
    token_subword_index = torch.Tensor([[[0, 1, 2, 0, 0], [0, 0, 0, 3, 4]]]).long() # 0 is <pad>, this means that
    # batch=1, num_token=2, num_subword=5, the first token is subword_1 and subword_2, the second token is subword_3 and
    # subword_4
    # the order can be random. for example, [0,1,2,0,0] is the same as [0,0,2,1,0]
    encoded_layers = torch.rand(1, 5, 1)
    output = average_pooling(encoded_layers, token_subword_index)
    print(output)
    print(output.shape)
    print(encoded_layers)
