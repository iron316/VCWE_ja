import torch.nn as nn
import torch

from models.lstm_encoder import LSTMEncoder


class VCWE(nn.Module):
    def __init__(self, encode_dim, n_vocab, char_dim, atten_hidden, ):
        super(VCWE, self).__init__()
        self.u_embed = nn.Embedding(n_vocab, encode_dim)
        self.v_embed = nn.Embedding(n_vocab, encode_dim)
        self.lstm_encoder = LSTMEncoder(encode_dim, char_dim, atten_hidden)
        self.sigmoid = nn.LogSigmoid()

    def forward(self, pos_u, pos_v, pos_img, neg_v, neg_img):
        embed_u = self.u_embed(pos_u)
        embed_pos_v = self.v_embed(pos_v)
        img_pos = self.lstm_encoder(pos_img)
        embed_neg_v = self.v_embed(neg_v)
        img_neg = self.lstm_encoder(neg_img)

        score = torch.sum(torch.mul(embed_u, embed_pos_v), dim=1)
        score = -self.sigmoid(score)

        img_score = torch.sum(torch.mul(embed_u, img_pos))
        img_score = -self.sigmoid(img_score)

        neg_score = torch.bmm(embed_neg_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(-self.sigmoid(-neg_score))

        img_neg_score = torch.bmm(img_neg, embed_u.unsqueeze(2)).squeeze()
        img_neg_score = torch.sum(-self.sigmoid(-img_neg_score))

        return torch.mean(score + img_score + neg_score + img_neg_score)
