import torch
import torch.nn as nn

from models.img_encoder import ImgEncoder


class LSTMEncoder(nn.Module):
    def __init__(self, encode_dim, char_dim, atten_hidden, neg_size):
        self.encode_dim = encode_dim
        super(LSTMEncoder, self).__init__()
        self.img_encoder = ImgEncoder(char_dim, neg_size)
        self.lstm = nn.LSTM(char_dim, int(encode_dim / 2),
                            bidirectional=True, batch_first=True)
        self.atten1 = nn.Linear(encode_dim, atten_hidden)
        self.atten2 = nn.Linear(atten_hidden, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def attention(self, h):
        h = self.tanh(self.atten1(h))
        h = self.softmax(self.atten2(h))
        return h

    def forward(self, x):
        h = self.img_encoder(x)
        h, _ = self.lstm(h)
        a = self.attention(h)
        h = torch.sum(a * h, dim=1)
        return h
