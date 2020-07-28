

def copy(vocab_dist, att_dist, p_gen, batch_enc_vocab):
	# vocab_dist, att_dist are vectors after softmax
	# p_gen is [batch, 1], you can get it from output hidden state
	vocab_dist = p_gen * vocab_dist	# [batch, dec_vocab]
	att_dist = (1-p_gen) * att_dist # [batch, enc_len]
	
	vocab_dist = vocab_dist.scatter_add(1, batch_enc_vocab, att_dist)
	# batch_enc_vocab: [batch, enc_len]		# batch_enc_vocab[i][j] = the index of dec_vocab of jth word in ith batch  
        # this copy function can be used when src vocab == tgt vocab

def dynamic_copy_loss(vocab_dist, ground_truth_idx, src_copy_tgt):
        # vocab_dist: [batch, tgt_len, gen_vocab+src_vocab], src_vocab is the maximum src length of these batch (this is why it call dynamic), gen_vocab is the size of generation vocabulary. This is the concatenation of generate vocabulary and attention score
	# ground_truth_idx: [batch, tgt_len], ground_truth_idx[b][i] is the idx of gen_vocab
        # src_copy_tgt: [batch, tgt_len], src_copy_tgt[b][i] is the idx of src_vocab, which means that src_copy_tgt[b][i] <= src_vocab.
	# if src_copy_tgt[b][i] == CONSTANT_PAD, it means that this token doesn't copy from source
        src_copy_tgt_mask = src_copy_tgt.ne(CONSTANT_PAD)
	ground_truth_mask = ground_truth_idx.ne(CONSTANT_PAD) & ground_truth_idx.ne(CONSTANT_UNK)  # if ground_truth is <pad> or <unk>, the decode process must copy from source at this step
        src_copy_tgt_offset = src_copy_tgt.unsqueeze(2) + gen_vocab
        src_copy_prob = vocab_dist.gather(dim=2, index=src_copy_tgt_offset).squeeze(2).masked_fill(~src_copy_tgt_mask, 0)  # [batch, tgt_len]
        gen_prob = vocab_dist.gather(dim=2, index=ground_truth_idx.unsqueeze(2)).suqeeze(2)     # [batch, tgt_len]
        final_prob = src_copy_prob + gen_prob.masked_fill(~ground_truth_mask, 0)                 
                                              # the mask guarantees that generate <pad> or <unk> should not be consider when training
        tgt_pad_mask = ground_truth_idx.ne(CONSTANT_PAD)  # tgt_pad_mask differs to ground_truth_mask, since <pad> should be ignored when calculate loss but <unk> should be taken into consideration
        loss = -final_prob.log().masked_fill(~tgt_pad_mask, 0)
        loss = loss.sum() / tgt_pad_mask.sum()
        return loss

        #
	# example: (batch=1), input=['hello', 'how', 'are', 'you', "pikachu", '<pad>'], output_vocab=["<pad>", "<unk>", "i", "am", "fine", "and", 'hello', 'how', 'are', 'you']
	# ground_truth_output = ["i", "am", "fine", "and", "you", "<unk>"(should be "pikachu" but output_vocab do, "<pad>"]
	# then ground_truth_idx=[2, 3, 4, 5, 9, 1, 0], src_copy_tgt=[-1, -1, -1, -1, 3, 4, 5]
	# vocab_dist.shape=[1, 15]
	# gen_vocab=10
	# src_cop_tgt_mask = [0, 0, 0, 0, 1, 1, 1]
	# ground_truth_mask = [1, 1, 1, 1, 1, 0, 0]
	# tgt_pad_mask = [1, 1, 1, 1, 1, 1, 0]
	# for token "you", the prob is gen_prob[4] + src_copy_prob[4], for token "pikachu", gen_prob[5] = 0
	# for <pad>, it is masked when calculating loss
	# for other token, src_copy_prob[i] = 0
	#
	
	
	
def gather_usage(a, b, c):
	# a: [2, 5], b: [2, 3](int), c: [2, 3]
	torch.gather(a, 1, b)
	# ret[i][j] = a[i][b[i][j]]
        # gather is used to change the <dim> according to <index>. In this example, assume a=[[1,2,3,4,5],[2,3,4,5,6]], b=[[0,1,2],[4,3,2]], we will get [[1,2,3],[6,5,4]
	# gather() use only <a> to get new vector, while scatter() use <c> to change <a> according to <index>(<b>)
	

# gather, scatter_add functions are important but hard to remember. Please refer to docs when you have problems in dealing with particular problems

def label2onehot(label, vocab_size):
	# label: [batch, seq_len], label[i][j]=k means the i-th batch and the j-th token is the k-th vocab index
	# we want to change it to onehot vector: [batch, seq_len, vocab_size]
	# we use scatter
	batch, seq_len = label.shape
	onehot = torch.zeros(batch, seq_len, vocab_size)
	label = label.unsqueeze(2)
	onehot.scatter_(dim=2, index=label, src=1)
	# scatter with dim=2: self[i][j][index[i][j][k]] = src[i][j][k], here src[i][j][k] is always 1. Therefore, self[i][j][k] = 1 (recall that label[i][j][1]=k)