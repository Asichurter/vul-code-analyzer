import pickle
import numpy as np
import torch

from downstream.model.devign.data_loader import n_identifier, g_identifier, l_identifier
import inspect
from datetime import datetime


def load_default_identifiers(n, g, l):
    if n is None:
        n = n_identifier
    if g is None:
        g = g_identifier
    if l is None:
        l = l_identifier
    return n, g, l


def initialize_batch(entries, batch_size, shuffle=False):
    total = len(entries)
    indices = np.arange(0, total - 1, 1)
    if shuffle:
        np.random.shuffle(indices)
    batch_indices = []
    start = 0
    end = len(indices)
    curr = start
    while curr < end:
        c_end = curr + batch_size
        if c_end > end:
            c_end = end
        batch_indices.append(indices[curr:c_end])
        curr = c_end
    return batch_indices[::-1]


def tally_param(model):
    total = 0
    for param in model.parameters():
        total += param.data.nelement()
    return total


def debug(*msg, sep='\t'):
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    print('[' + str(time) + '] File \"' + file_name + '\", line ' + str(ln) + '  ', end='\t')
    for m in msg:
        print(m, end=sep)
    print('')

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def make_mask_from_lens(lens):
    max_pad_len = lens.max().item()
    bsz = len(lens)
    idxes = torch.arange(0,max_pad_len,device=lens.device).repeat(bsz,1)
    mask = idxes < lens.unsqueeze(-1)
    return mask

def mask_mean(tensor, mask, dim, eps=1e-5):
    while tensor.ndim > mask.ndim:
        mask = mask.unsqueeze(-1)
    tensor *= mask
    tensor_sum = tensor.sum(dim=dim)
    tensor_count = mask.sum(dim=dim) + eps
    return tensor_sum / tensor_count


