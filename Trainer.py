# Imports
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Optimizer

import torch.nn.parallel

from Langevin import LangevinSampler
from Buffer import Buffer


class Trainer:
    """ Maximum likelihood training of an energy based model base training algorithm

    Implements the training algorithm for maximum likelihood training of an energy based model
    """
    def __init__(self, dataloader: DataLoader,
                 model: torch.nn,
                 optimizer: Optimizer,
                 sampler: LangevinSampler,
                 criterion: torch.nn,
                 device: torch.device,
                 buffer: Buffer,
                 scheduler: Optimizer = None):
        """ Initialisation

        :param dataloader: dataloader for access to training data
        :param model: energy based model
        :param optimizer: optimizer for maximum likelihood optimisation
        :param sampler: Langevin sampler
        :param criterion: loss function used for optimisation
        :param device: torch device
        :param buffer: sample replay buffer to store start samples for Langevin sampling
        :param scheduler: optional learning rate scheduler
        """
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._dataloader = dataloader
        self._device = device
        self._sampler = sampler

        self._buffer = buffer
        self._scheduler = scheduler

        self.saveIntervall = 4

        self.independent_sampling = False
        self.batch_size = None

        self._save_function = None
        self._print_function = None

    @property
    def save_function(self):
        """ Returns save function used to store progress
        """
        return self._save_function

    @save_function.setter
    def save_function(self, save_function):
        self._save_function = save_function

    @property
    def print_function(self):
        """ Returns print function used for output
        """
        return self._print_function

    @print_function.setter
    def print_function(self, print_function):
        self._print_function = print_function

    def train(self, num_epochs, burn_in, do_print=False):
        """ Basic training algorithm

        :param num_epochs: amount of total iterations of the whole dataset
        :param burn_in: burn in duration used for Langevin sampling
        :param do_print: optional output of training progress
        """
        langevin = self._sampler
        store = {'loss': [], 'evaluation': []}

        for epoch in range(num_epochs):
            last_loss = []
            # Training
            for i, (X, y) in enumerate(self._dataloader):

                # Sample
                x_plus = X.to(self._device)

                if self.independent_sampling:
                    buffer_index, x_minus = self._buffer.get_item(self.batch_size)
                    x_minus = langevin.sample(x_minus, burn_in=burn_in)

                else:
                    buffer_index, x0 = self._buffer.get_item(1)
                    x_minus = langevin.sample_subsequent(x0, burn_in=burn_in, batch_size=X.size(dim=0))

                # Compute prediction and loss
                self._optimizer.zero_grad()

                rx_plus = self._model(x_plus)

                rx_minus = self._model(x_minus)

                loss = self._criterion(rx_plus, rx_minus)

                # Backpropagation
                loss.backward()

                self._optimizer.step()
                last_loss.append(loss.item())

                # Buffer maintainance
                self._buffer.add_item(x_minus, buffer_index)

                # Save
                if do_print:
                    total_loss = np.mean(last_loss)
                    print('[%d/%d]\tLoss: %.4f' % (epoch, num_epochs, total_loss.item()))
                    if self._print_function is not None:
                        self._print_function(self._model)
                    last_loss = []
                if i % self.saveIntervall == 0 and self._save_function is not None:
                    total_loss = np.mean(last_loss)
                    self._save_function(self._model, total_loss, store, i, epoch)
                    last_loss = []

            # Adjusting
            if self._scheduler is not None:
                self._scheduler.step()
