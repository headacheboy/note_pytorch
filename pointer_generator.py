

def copy(vocab_dist, att_dist, p_gen, batch_enc_vocab):
	# vocab_dist, att_dist are vectors after softmax
	# p_gen is [batch, 1], you can get it from output hidden state
	vocab_dist = p_gen * vocab_dist	# [batch, dec_vocab]
	att_dist = (1-p_gen) * att_dist # [batch, enc_len]
	
	vocab_dist = vocab_dist.scatter_add(1, batch_enc_vocab, att_dist)
	# batch_enc_vocab: [batch, enc_len]		# batch_enc_vocab[i][j] = the index of dec_vocab of jth word in ith batch 
	
	
	
def gather_usage(a, b, c):
	# a: [2, 5], b: [2, 3](int), c: [2, 3]
	torch.gather(a, 1, b)
	# ret[i][j] = a[i][b[j]]
	

# gather, scatter_add functions are important but hard to remember. Please refer to docs when you have problems in dealing with particular problems