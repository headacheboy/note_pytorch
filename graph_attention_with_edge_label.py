import torch
batch=1
length=2
dim=3
edge_len=2

#邻接表表示：b[:, i, j]=k表示第i个点的邻接表中的第j个，那个点的序号是k

a = torch.rand(1, 2, 3)	# batch, length, dim
b = torch.Tensor([[[1, -1], [0, 0]]])	# shape=[batch,length, edge_len]
mask = b < 0
b_fill = b.masked_fill(mask=mask, value=0)
b_fill = b_fill.unsqueeze(3).repeat(1, 1, 1, dim)	# batch, length, edge_len, dim
c = torch.gather(a.unsqueeze(2).repeat(1, 1, edge_len, 1), dim=1, index=b_fill) # batch, length, edge_len, dim, 此时c[:, i, j, :]就表示第i个点指向的点（即第k个点）的hidden state
c_masked = c.masked_fill(mask=mask, value=0)	# 不要的边mask成0
a_new = linear(c_masked.view(batch, length, edge_len*dim))

# 然后view(batch, length, edge_len*dim), 再一个linear就可以接回原来的形式了，注意mask掉不要的边
# 也可以换一种mask和获取a_new的方式如下：

c_score = linear_1(c)   # batch, length, edge_len, 1
c_score = c_score.masked_fill(mask=mask, value=-1e18)
c_score = F.softmax(c_score, dim=2)
# 可以对c拼接上需要的edge label embedding
a_new = torch.matmul(c_score.permute(0, 1, 3, 2), c)