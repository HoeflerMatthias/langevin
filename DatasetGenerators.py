import torch
import torch.utils.data
from torch.utils.data import Dataset

import torch.nn.parallel


class NormalDataset(Dataset):
    def __init__(self, device):
        super(NormalDataset, self).__init__()
        self._device = device

        self.size = None
        self.mean = None
        self.covariance = None
        self.data = None

    def calculate(self, size, mean, covariance):
        self.size = size
        self.mean = mean
        self.covariance = covariance
        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
        self.data = m.sample((size,)).to(self._device)

    def load(self, filename):
        tmp = torch.load(filename)
        self.size = tmp['size']
        self.mean = tmp['mean']
        self.covariance = tmp['covariance']
        self.data = tmp['data'].to(self._device)

    def save(self, filename):
        torch.save({
            'size': self.size,
            'mean': self.mean,
            'covariance': self.covariance,
            'data': self.data
        }, filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0.


class NormalCutDataset(Dataset):
    def __init__(self, device):
        super(NormalCutDataset, self).__init__()
        self._device = device

        self.size = None
        self.mean = None
        self.covariance = None
        self.min_vec = None
        self.max_vec = None
        self.data = None

    def calculate(self, size, mean, covariance, min_vec, max_vec):
        self.size = size
        self.mean = mean
        self.covariance = covariance
        self.min_vec = min_vec
        self.max_vec = max_vec

        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
        data = torch.empty([size]+list(mean.size()), dtype=torch.float)
        i = 0
        while i < size:
            d = m.sample()
            if torch.all(torch.gt(d, min_vec)) and torch.all(torch.gt(max_vec, d)):
                data[i] = d
                i += 1
        self.data = data.to(self._device)

    def load(self, filename):
        tmp = torch.load(filename)
        self.size = tmp['size']
        self.mean = tmp['mean']
        self.covariance = tmp['covariance']
        self.data = tmp['data'].to(self._device)

    def getall(self):
        return self.data

    def save(self, filename):
        torch.save({
            'size': self.size,
            'mean': self.mean,
            'covariance': self.covariance,
            'min_vec': self.min_vec,
            'max_vec': self.max_vec,
            'data': self.data
        }, filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0.
