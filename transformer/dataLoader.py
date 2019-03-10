import numpy as np
import torch
import args
from torch.utils.data import Dataset, DataLoader, TensorDataset

class TextIterator():
    def __init__(self, config):
        self.config = config
        self.vocabSou = {}
        self.vocabIndSou = {}
        self.vocabTar = {}
        self.vocabIndTar = {}
        self.readData()

        for key in self.vocabSou:
            tmp = self.vocabSou[key]
            self.vocabIndSou[tmp] = key
        for key in self.vocabTar:
            tmp = self.vocabTar[key]
            self.vocabIndTar[tmp] = key

        self.config.vocabSou = len(self.vocabSou)
        self.config.vocabTar =len(self.vocabTar)
        self.testX = self.trainX
        self.testY = self.trainY

    def readData(self):
        trainX = []
        trainY = []
        f = open("data.txt", encoding='utf-8')
        sou = 3
        self.vocabSou["<pad>"] = 0
        self.vocabSou["<eos>"] = 1
        self.vocabSou["<unk>"] = 2
        tar = 3
        self.vocabTar["<pad>"] = 0
        self.vocabTar["<eos>"] = 1
        self.vocabTar["<unk>"] = 2
        for line in f:
            ls = line.split('\t')
            ls[0] = ls[0].split()
            ls[1] = ls[1].split()
            for ele in ls[0]:
                if ele not in self.vocabSou:
                    self.vocabSou[ele] = sou
                    sou += 1
            for ele in ls[1]:
                if ele not in self.vocabTar:
                    self.vocabTar[ele] = tar
                    tar += 1
            tmpLS = [0] * self.config.maxSouLen
            minLen = min(len(ls[0]), self.config.maxSouLen-1)
            for i in range(minLen):
                tmpLS[i] = self.vocabSou[ls[0][i]] if ls[0][i] in self.vocabSou else self.vocabSou["<unk>"]
            tmpLS[minLen] = 1
            trainX.append(tmpLS)

            tmpLS = [0] * self.config.maxTarLen
            minLen = min(len(ls[1]), self.config.maxTarLen-1)
            for i in range(minLen):
                tmpLS[i] = self.vocabTar[ls[1][i]] if ls[1][i] in self.vocabTar else self.vocabTar["<unk>"]
            tmpLS[minLen] = 1
            trainY.append(tmpLS)
        self.trainX = torch.from_numpy(np.array(trainX))
        self.trainY = torch.from_numpy(np.array(trainY))

        self.config.srcVocabSize = len(self.vocabSou)
        self.config.tarVocabSize = len(self.vocabTar)

class MyDataSet():
    def __init__(self, config, texti, i):
        self.config = config
        self.texti = texti
        self.i = i

    def __getitem__(self, item):
        if self.i == 0:
            return self.texti.trainX[item], self.texti.trainY[item]
        elif self.i == 1:
            return self.texti.validX[item], self.texti.validY[item]
        elif self.i == 2:
            return self.texti.testX[item], self.texti.testY[item]

    def __len__(self):
        return self.texti.trainX.shape[0]

if __name__ == "__main__":
    config = args.get_args()
    texti = TextIterator(config)
    trainDataLoader = DataLoader(dataset=MyDataSet(config, texti, 0), batch_size=config.batchSize, shuffle=True, num_workers=0, drop_last=True)
    testDataLoader = DataLoader(dataset=MyDataSet(config, texti, 2), batch_size=config.batchSize, shuffle=False, num_workers=0, drop_last=True)
    for epoch in range(2):
        for i, data in enumerate(trainDataLoader):
            inputs, labels = data
            print("epoch: ", epoch, " ", inputs, " ", inputs.shape, " ", labels.shape)