import torch
import numpy as np


def max_pad(batch):
    print('start collete')
    bs = len(batch)
    idx = [i for i, _, _, _, _ in batch]
    idx = torch.LongTensor(idx)

    pos_idx = [i for _, i, _, _, _ in batch]
    pos_idx = torch.LongTensor(pos_idx)

    pos_img = [i for _, _, i, _, _ in batch]
    pos_len = max([len(img) for img in pos_img])
    pos_pad = np.zeros((bs, pos_len, 40, 40))
    for i, img in enumerate(pos_img):
        pos_pad[i, :len(img)] = np.array(img)
    pos_pad = torch.Tensor(pos_pad).type(torch.float32)

    neg_idx = [i for _, _, _, i, _ in batch]
    neg_idx = torch.LongTensor(neg_idx)

    neg_img = [i for _, _, _, _, i in batch]
    neg_len = max([len(img) for img in neg_img])
    neg_pad = np.zeros((bs, neg_len, 40, 40))
    for i, img in enumerate(neg_img):
        neg_pad[i, :len(img)] = np.array(img)

    print('end collete')
    return idx, pos_idx, pos_img, neg_idx, neg_img
