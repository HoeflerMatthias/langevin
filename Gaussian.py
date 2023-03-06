import torch
import torch.utils.data

import torch.nn as nn
import torch.nn.parallel


class Symmetric(nn.Module):
    def forward(self, x):
        return x.triu() + x.triu(1).transpose(-1, -2)


class GaussianModel(nn.Module):
    def __init__(self, device, in_chan=1, symmetrized=True):

        super(GaussianModel, self).__init__()
        self.bilinear = nn.Bilinear(in_chan, in_chan, 1, bias=False)
        if symmetrized:
            nn.utils.parametrize.register_parametrization(self.bilinear, "weight", Symmetric())

        noise = torch.rand(torch.Size([in_chan]), device=device)
        noise.uniform_(-1., 1.)
        self.bias = nn.Parameter(noise.data)

        self.flatten = nn.Flatten()

    def forward(self, x):
        y = x + self.bias
        z = self.bilinear(y, y)
        return 0.5 * z
