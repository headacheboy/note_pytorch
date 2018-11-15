import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

inDim = 20
hidDim = 10
labelNum = 2
batchSize = 2

class TestD(nn.Module):
    def __init__(self):
        super(TestD, self).__init__()
        self.Fc1 = nn.Linear(inDim, hidDim)
        self.Fc2 = nn.Linear(hidDim, labelNum)
        self.Fc3 = nn.Linear(hidDim, labelNum)

    def forward(self, x, ind):
        h1 = F.relu(self.Fc1(x))
        if ind == 0:
            out = self.Fc2(h1)
        else:
            out = self.Fc3(h1)
        return out

model = TestD()

x = torch.Tensor(np.random.randn(batchSize, inDim))
z = torch.Tensor(np.random.randn(batchSize, inDim))
yy = torch.Tensor([1, 0]).long()
y = torch.Tensor([0, 1]).long()

lossfunc = nn.CrossEntropyLoss()
opt1 = torch.optim.Adam(list(model.Fc1.parameters()) + list(model.Fc2.parameters()), lr=0.1)
opt2 = torch.optim.Adam(list(model.Fc1.parameters()) + list(model.Fc3.parameters()), lr=0.01)
print(list(model.Fc1.parameters()), "fc1")
print(list(model.Fc2.parameters()), "fc2")
print(list(model.Fc3.parameters()), "fc3")


logit1 = model(x, 0)

loss1 = lossfunc(logit1, y)
opt1.zero_grad()
loss1.backward()
opt1.step()

print(list(model.Fc1.parameters()), "fc1")
print(list(model.Fc2.parameters()), "fc2")
print(list(model.Fc3.parameters()), "fc3")

logit2 = model(x, 1)
loss2 = lossfunc(logit2, yy)
opt2.zero_grad()
loss2.backward()
opt2.step()

print(list(model.Fc1.parameters()), "fc1")
print(list(model.Fc2.parameters()), "fc2")
print(list(model.Fc3.parameters()), "fc3")