import torch
from torch.autograd import Variable
from scipy.stats import rankdata
import numpy as np


def getBatchSequence(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, batch, evaluation=False):
    seq_len = min(batch, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation)
    return data
