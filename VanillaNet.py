import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class VanillaNet(nn.Module):
    def __init__(self, n_c=3, n_f=32):
        super(VanillaNet, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            Swish(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            Swish(),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),
            Swish(),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),
            Swish(),
            nn.Conv2d(n_f*8, n_f*16, 4, 2, 1),
            Swish(),
            nn.Conv2d(n_f*16, 1, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()
