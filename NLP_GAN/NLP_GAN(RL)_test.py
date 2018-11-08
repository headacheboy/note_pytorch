import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

inDim = 20
hidDim = 10
labelNum = 2
batchSize = 2

class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, pred, dLoss):
        lossfunction = -pred.max(dim=1)[0]*dLoss
        return torch.mean(lossfunction)


class TestG(nn.Module):
    def __init__(self):
        super(TestG, self).__init__()
        self.Fc1 = nn.Linear(inDim, hidDim)
        self.Fc2 = nn.Linear(hidDim, labelNum)

    def forward(self, x):
        h1 = F.relu(self.Fc1(x))
        out = F.softmax(self.Fc2(h1), dim=-1)
        return out

class TestD(nn.Module):
    def __init__(self):
        super(TestD, self).__init__()
        self.Fc1 = nn.Linear(inDim, hidDim)
        self.Fc2 = nn.Linear(hidDim, labelNum)

    def forward(self, x):
        h1 = F.relu(self.Fc1(x))
        out = self.Fc2(h1)
        return out

lossfunc = nn.CrossEntropyLoss()
lossfunc2 = myLoss()
generator = TestG()
discriminator = TestD()
opt_gen = torch.optim.Adam(generator.parameters(), lr=0.1)
opt_dis = torch.optim.Adam(discriminator.parameters(), lr=0.01)

x = [torch.Tensor(np.random.randn(batchSize, inDim)) for i in range(2)]
y = torch.Tensor([[0, 1], [1, 0]])

z = generator(x[0])
cho = list(z.argmax(1).numpy())
print(z)
print(cho)
new_x = torch.cat([x[0]]+[x[cho[i]] for i in range(2)], dim=0)
new_y = torch.cat([y[0]]+[y[cho[i]] for i in range(2)], dim=0).long()

output = discriminator(new_x)
loss_d = lossfunc(output, new_y)

tmpFc1 = list(discriminator.Fc1.parameters())
tmpFc2 = list(generator.Fc1.parameters())
print(tmpFc1[0][0])
print(tmpFc2[0][0])

opt_dis.zero_grad()
loss_d.backward()
opt_dis.step()
print(tmpFc1[0][0])
print(tmpFc2[0][0])

tmpNewOutput = discriminator(new_x)
loss_g = lossfunc2(z, loss_d.detach())

opt_gen.zero_grad()
loss_g.backward()
opt_gen.step()
print(tmpFc1[0][0])
print(tmpFc2[0][0])
