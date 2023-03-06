# Imports
import torch
import torch.utils.data
import torch.nn.parallel


class Buffer:
    """ Sample replay buffer

    Implements functionality to choose startsamples for Langevin Sampling
    """
    def __init__(self, shape: torch.Size,
                 size: int = 1000,
                 reinitialisation_chance: float = 0.05,
                 device: torch.device = torch.cuda.current_device()):
        """ Initialisation

        :param shape: shape of one sample
        :param size: total amount of items stored in buffer
        :param reinitialisation_chance: probability for reinitialisation
        :param device: torch device used
        """
        self._device = device
        self._reinitialisationChance = reinitialisation_chance
        self._size = size
        self._shape = shape

        self._noise = torch.empty(size, device=self._device)
        self._buffer = torch.empty(torch.Size([self._size] + list(self._shape)))

        self.bufferReinitialisation = torch.empty(1, device=self._device)

    @property
    def shape(self) -> torch.Size:
        """ Returns shape of one sample
        """
        return self._shape

    def get_item(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """ Returns items from buffer

        :param batch_size: sample size of returned items
        :return: batch of items from buffer
        """

        idx = torch.randperm(self._size)[:batch_size]
        return idx, self._buffer[idx].detach().clone()

    def add_item(self, x: torch.Tensor, idx):
        """ Replaces existing items from buffer

        :param x: new items
        :param idx: indicies of old items
        :return:
        """

        reinit_probs = torch.rand_like(idx, dtype=torch.float64)

        p = 1-self._reinitialisationChance
        reinit_idx = torch.nonzero(reinit_probs > p)
        reuse_idx = torch.nonzero(reinit_probs <= p)

        self._buffer[reinit_idx].uniform_(0, 1)
        self._buffer[reuse_idx] = x[reuse_idx].detach().clone()
