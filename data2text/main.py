import numpy as np
import torch
from torch.nn import *
import torch.nn.functional as F
import args
import Models
import dataLoader
from torch.utils.data import Dataset, DataLoader, TensorDataset

class Main():
    def __init__(self):
        self.config = args.get_args()
        self.texti = dataLoader.TextIterator(self.config)
        self.model = Models.Data2Text(self.config)
        self.model.cuda()
        self.lossfunc = CrossEntropyLoss(reduction='sum', ignore_index=0).cuda()
        self.optimizer = torch.optim.Adam(lr=self.config.lr, params=self.model.parameters(), weight_decay=
                                          self.config.weight_decay)

    def cal_loss(self, pred, gold):
        batch = pred.size(0)
        pred_flatten = pred.view(-1, pred.size(2))  # [batch*length, vocab]
        gold_flatten = gold.view(-1)                        # [batch*length]
        loss = self.lossfunc(pred_flatten, gold_flatten)
        return loss / batch

    def training(self):
        trainDataLoader = DataLoader(dataset=dataLoader.MyDataSet(self.config, self.texti, 0),
                                     batch_size=self.config.batchSize, shuffle=True,
                                     num_workers=0, drop_last=True)

        for epoch in range(self.config.epoch):
            total_loss = 0.0
            i = 0
            for data in trainDataLoader:
                i += 1
                self.optimizer.zero_grad()
                x, labels = data
                x = x.long().cuda()
                labels = labels.long().cuda()
                pred = self.model(x, labels)
                loss = self.cal_loss(pred, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.maxClip)
                self.optimizer.step()
                total_loss += loss.item()
                print(i)
            total_loss /= i
            print("epoch: ", epoch, " avg loss: ", total_loss)
            if (epoch+1) % self.config.validStep == 0:
                self.model.eval()
                self.valid(1)
                self.model.train()
            if (epoch+1) > self.config.lr_decay_thre and (epoch+1) % self.config.decayEpoch == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.config.lr_decay

    def valid(self, i):
        validDataLoader = DataLoader(dataset=dataLoader.MyDataSet(self.config, self.texti, i),
                                     batch_size=self.config.batchSize, shuffle=False,
                                     num_workers=0, drop_last=False)
        for data in validDataLoader:
            x, gold = data
            x = x.long().cuda()
            labels = torch.ones(x.size(0), self.config.maxLen).long().cuda()
            for i in range(1, self.config.maxLen):
                pred = self.model(x, labels)
                predLabel = pred.argmax(dim=2)
                labels[:, i] = predLabel[:, i - 1]
            print("gold: ")
            for i in range(self.config.maxLen):
                if gold[0][i] == self.texti.vocab["<eos>"]:
                    break
                else:
                    print(self.texti.vocabInd[gold[0][i].item()], end=" ")
            print("\nmine: ")
            for i in range(1, self.config.maxLen):
                if labels[0][i] == self.texti.vocab["<eos>"]:
                    break
                else:
                    print(self.texti.vocabInd[labels[0][i].item()], end=" ")
            print("")




if __name__ == '__main__':
    m = Main()
    m.training()