import numpy as np
import torch


class MaxPad():
    def __init__(self, neg_size=5):
        self.neg_size = neg_size

    def __call__(self, batch):
        bs = len(batch)
        idx, pos_idx, pos_img, neg_idx, neg_img = zip(*batch)

        idx = torch.LongTensor(idx)

        pos_idx = torch.LongTensor(pos_idx)

        pos_len = max([len(img) for img in pos_img])
        pos_pad = np.zeros((bs, pos_len, 40, 40))
        for i, img in enumerate(pos_img):
            pos_pad[i, :len(img)] = img
        pos_pad = torch.from_numpy(pos_pad).type(torch.float32)

        neg_idx = torch.LongTensor(neg_idx)

        neg_len = max([imgs.shape[1] for imgs in neg_img])
        neg_pad = np.zeros((bs, self.neg_size, neg_len, 40, 40))
        for i, img in enumerate(neg_img):
            neg_pad[i, :, :img.shape[1]] = img
        neg_pad = torch.from_numpy(neg_pad).type(torch.float32)

        return idx, pos_idx, pos_pad, neg_idx, neg_pad
