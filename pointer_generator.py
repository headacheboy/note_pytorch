

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
        # src_copy_tgt: [batch, tgt_len], src_copy_tgt[b][i] is the idx of src_vocab, if src_copy_tgt[b][i] == CONSTANT_PAD, it means that this token doesn't copy from source
        src_copy_tgt_mask = src_copy_tgt.ne(CONSTANT_PAD)
        src_copy_tgt_offset = src_copy_tgt.unsqueeze(2) + gen_vocab
        src_copy_prob = vocab_dist.gather(dim=2, index=src_copy_tgt_offset).squeeze(2).masked_fill(~src_copy_tgt_mask, 0)  # [batch, tgt_len]
        gen_prob = vocab_dist.gather(dim=2, index=ground_truth_idx.unsqueeze(2)).suqeeze(2)     # [batch, tgt_len]
        final_prob = src_copy_prob + gen_prob.masked_fill(src_copy_tgt_mask, 0)                 
                                              # the mask guarantees that copy and generate will not calculate the prob at the same time
        tgt_pad_mask = ground_truth_idx.ne(CONSTANT_PAD)
        loss = -final_prob.log().masked_fill(~tgt_pad_mask, 0)
        loss = loss.sum() / tgt_pad_mask.sum()
        return loss
	
	
	
def gather_usage(a, b, c):
	# a: [2, 5], b: [2, 3](int), c: [2, 3]
	torch.gather(a, 1, b)
	# ret[i][j] = a[i][b[i][j]]
        # gather is used to change the <dim> according to <index>. In this example, assume a=[[1,2,3,4,5],[2,3,4,5,6]], b=[[0,1,2],[4,3,2]], we will get [[1,2,3],[6,5,4]
	# gather() use only <a> to get new vector, while scatter() use <c> to change <a> according to <index>(<b>)
	

# gather, scatter_add functions are important but hard to remember. Please refer to docs when you have problems in dealing with particular problems
