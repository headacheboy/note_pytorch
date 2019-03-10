import torch
import numpy as np
import args
import dataLoader
import Models
from torch.utils.data import Dataset, DataLoader, TensorDataset

class Main():
    def __init__(self):
        self.config = args.get_args()
        self.texti = dataLoader.TextIterator(self.config)
        self.model = Models.Transformer(self.config)
        self.model.cuda()
        self.lossfunc = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum').cuda()
        self.optimizer = torch.optim.Adam(lr=self.config.lr, params=self.model.parameters(), weight_decay=
                                          self.config.weight_decay)

    def cal_loss(self, pred, gold, smooth):
        gold = gold.view(-1)    # flatten
        if smooth == 1:
            pass
        elif smooth == 0:
            batch_size = pred.size(0)
            pred = pred.view(-1, pred.size(2))  # flatten
            return self.lossfunc(pred, gold) / batch_size

    def cal_performance(self, pred, gold, smooth):
        loss = self.cal_loss(pred, gold, smooth)

        pred = pred.view(-1, pred.size(2))

        pred = pred.max(1)[1]
        print(pred.view(-1, self.config.maxTarLen))
        print(gold)
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(0)
        n_word = non_pad_mask.sum().item()
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct, n_word

    def training(self):
        trainDataLoader = DataLoader(dataset=dataLoader.MyDataSet(self.config, self.texti, 0),
                                     batch_size=self.config.batchSize, shuffle=True,
                                     num_workers=0, drop_last=True)
        print(self.config.srcVocabSize, self.config.tarVocabSize)
        for epoch in range(self.config.epoch):
            total_loss = 0.0
            n_word_total = 0
            n_word_correct = 0
            for i, data in enumerate(trainDataLoader):
                self.optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.long().cuda()
                labels = labels.long().cuda()
                posInput = torch.from_numpy(
                    np.array([list(range(1, self.config.maxSouLen+1)) for _ in range(self.config.batchSize)])
                ).long().cuda()
                posInput = posInput.masked_fill(inputs.le(0), 0)
                posOutput = torch.from_numpy(
                    np.array([list(range(1, self.config.maxTarLen+1)) for _ in range(self.config.batchSize)])
                ).long().cuda()
                posOutput = posOutput.masked_fill(labels.le(0), 0)
                output = self.model(inputs, posInput, labels, posOutput)
                loss, n_correct, n_word = self.cal_performance(output, labels, smooth=self.config.smooth)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_word_correct += n_correct
                n_word_total += n_word
            print("epoch: ", epoch, " total loss: ", total_loss, " correct word rate: ", n_word_correct / n_word_total)
            if (epoch+1) % 50 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.config.lr_decay

    def valid(self):
        testDataLoader = DataLoader(dataset=dataLoader.MyDataSet(self.config, self.texti, 2),
                                    batch_size=self.config.batchSize, shuffle=False,
                                    num_workers=0, drop_last=False)
        self.model.eval()
        total_loss = 0.0
        n_word_total = 0
        n_word_correct = 0
        for i, data in enumerate(testDataLoader):
            self.optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.long().cuda()
            labels = labels.long().cuda()
            posInput = torch.from_numpy(
                np.array([list(range(1, self.config.maxSouLen+1)) for _ in range(self.config.batchSize)])
            ).long().cuda()
            posInput = posInput.masked_fill(inputs.le(0), 0)
            posOutput = torch.from_numpy(
                np.array([list(range(1, self.config.maxTarLen+1)) for _ in range(self.config.batchSize)])
            ).long().cuda()
            posOutput = posOutput.masked_fill(labels.le(0), 0)
            output = self.model(inputs, posInput, labels, posOutput)
            loss, n_correct, n_word = self.cal_performance(output, labels, smooth=self.config.smooth)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_word_correct += n_correct
            n_word_total += n_word
        print("total loss: ", total_loss, " correct word rate: ", n_word_correct / n_word_total)


if __name__ == '__main__':
    m = Main()
    m.training()
    print('-------------')
    print(m.valid())