import os
from collections import defaultdict
import torch
import numpy as np

train_inter_dict = defaultdict(int)
train_link_dict = defaultdict(int)

def loadTrainData(datapath, n, batch):
    test_path = './data/'
    with open(train_path, 'r') as f:
        train_data = f.read().replace("\n", ",").split(",")

    for i in range(len(train_data)-1):
        record = train_data[i]
        lparts = record.split('\t')
        n1 = lparts[0]
        n2 = lparts[1]
        k = lparts[4]
        key = n1 + '_' + n2
        if k == '0':
            train_link_dict[key] += 1
        if k == '1':
            train_inter_dict[key] += 1

    train_data = [[float(i) for i in train_data[p].split("\t")] for p in range(len(train_data)-1)]
    train_data = torch.from_numpy(np.array(train_data, dtype=float))
    train_batch_num = train_data.size(0) // batch
    train_data = train_data.narrow(0, 0, train_batch_num * batch)
    train_data = train_data.view(train_batch_num, batch, -1)

    return train_data, train_batch_num, train_inter_dict, train_link_dict

def loadTestData(datapath, eval):
    test_path = './data/'
    with open(test_path, 'r') as f:
        test_data = f.read().replace("\n", ",").split(",")
    test_data = [[float(i) for i in test_data[p].split("\t")] for p in range(len(test_data)-1)]
    test_data = torch.from_numpy(np.array(test_data, dtype=float))
    test_batch_num = test_data.size(0) // eval
    test_data = test_data.narrow(0, 0, test_batch_num * eval)
    test_data = test_data.view(test_batch_num, eval, -1)

    return test_data, test_batch_num
