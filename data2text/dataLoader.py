import numpy as np
import torch
import args
from torch.utils.data import Dataset, DataLoader, TensorDataset

class TextIterator():
    def __init__(self, config):
        self.config = config
        self.config.maxU = 0
        self.config.maxP = 0
        self.config.maxR = 0
        self.vocab = {}
        self.vocabInd = {}
        self.devX = None
        self.devY = None
        self.testX = None
        self.testY = None
        self.getVocab(open("train.txt", encoding='utf-8'))
        self.trainX, self.trainY = self.readData(open("train.txt", encoding='utf-8'))
        self.devX, self.devY = self.readData(open("dev.txt", encoding='utf-8'))
        self.testX, self.testY = self.readData(open("test.txt", encoding='utf-8'))
        self.config.vocab =len(self.vocab)

    def getVocab(self, f):
        self.vocab["<pad>"] = 0
        self.vocab["<go>"] = 1
        self.vocab["<eos>"] = 2
        self.vocab["<unk>"] = 3
        self.vocabInd[0] = "<pad>"
        self.vocabInd[1] = "<go>"
        self.vocabInd[2] = "<eos>"
        self.vocabInd[3] = "<unk>"
        tgtWord = 4
        wordNum = {}
        for line in f:
            senLS = line.split('\t')
            senLS[3] = senLS[3].split()
            for word in senLS[3]:
                if word not in wordNum:
                    wordNum[word] = 1
                else:
                    wordNum[word] += 1
        for key in wordNum:
            if wordNum[key] >= 10:
                self.vocab[key] = tgtWord
                self.vocabInd[tgtWord] = key
                tgtWord += 1
        print(tgtWord)

    def readData(self, f):
        lsX = []
        lsY = []
        for line in f:
            senLS = line.split('\t')
            senLS[0] = int(senLS[0]) - 1
            senLS[1] = int(senLS[1]) + 19674    # max user ID is 19675
            senLS[2] = int(float(senLS[2])) + 99930 # max user ID + product ID is 99930
            senLS[3] = senLS[3].split()
            lsX.append(senLS[:3])
            tmpLS = [0] * self.config.maxLen
            tmpLS[0] = 1
            assert self.config.maxLen > len(senLS[3]) + 2
            for ind, word in enumerate(senLS[3]):
                if word in self.vocab:
                    tmpLS[ind+1] = self.vocab[word]
                else:
                    tmpLS[ind+1] = self.vocab["<unk>"]
            tmpLS[len(senLS[3])+1] = self.vocab["<eos>"]
            lsY.append(tmpLS)
        print("----------")
        tX = torch.from_numpy(np.array(lsX))
        tY = torch.from_numpy(np.array(lsY)).long()
        return tX, tY

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