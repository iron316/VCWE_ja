import numpy as np
import torch
from numba import jit


class MaxPad:
    def __init__(self, neg_size=5):
        self.neg_size = neg_size

    def __call__(self, batch):
        idx, pos_idx, pos_img, neg_idx, neg_img = zip(*batch)

        idx = torch.LongTensor(idx)
        neg_idx = torch.LongTensor(neg_idx)
        pos_idx = torch.LongTensor(pos_idx)
        pos_img, neg_img = maxpad(pos_img, neg_img, self.neg_size)
        pos_img = torch.from_numpy(pos_img).type(torch.float32)
        neg_img = torch.from_numpy(neg_img).type(torch.float32)

        return idx, pos_idx, pos_img, neg_idx, neg_img


@jit(nopython=True)
def maxpad(pos_img, neg_img, neg_size):
    bs = len(pos_img)

    pos_len = max([len(img) for img in pos_img])
    pos_pad = np.zeros((bs, pos_len, 40, 40))
    for i in np.arange(bs):
        pos_pad[i, :len(pos_img[i])] = pos_img[i]

    neg_len = max([imgs.shape[1] for imgs in neg_img])
    neg_pad = np.zeros((bs, neg_size, neg_len, 40, 40))
    for i in np.arange(bs):
        neg_pad[i, :, :neg_img[i].shape[1]] = neg_img[i]

    return pos_pad, neg_pad
