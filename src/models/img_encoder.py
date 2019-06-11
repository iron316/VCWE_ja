import torch.nn as nn


class ImgEncoder(nn.Module):
    def __init__(self, char_dim):
        self.char_encode = char_dim
        super(ImgEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(2048, char_dim)
        self.bn3 = nn.BatchNorm1d(char_dim)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU

    def forward(self, imgs):
        batch = imgs.size(0)
        imgs = imgs.view(-1, 1, 40, 40)
        h = self.maxpool(self.bn1(self.conv1(imgs)))
        h = self.maxpool(self.bn2(self.conv2(h)))
        h = h.view(-1, 2048)
        h = self.relu(self.bn3(self.fc(h)))
        h = h.view(batch, -1, self.char_dim)
        return h
