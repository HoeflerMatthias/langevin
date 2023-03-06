import torch
import math


class LangevinSampler:
    """ Langevin Monte Carlo Sampler

    Construct samples according to the Euler-Maruyama discretization of the Langevin stochastic differential equation.
    Possibility to create samples simultaneously or to subsequently with given distance between two samples.
    """
    def __init__(self, model: torch.nn,
                 h: float,
                 device,
                 clipping: bool = False,
                 clip_limits: tuple[float, float] = (-1., 1.)):
        """ Initialisation

        :param model: energy model of target distribution
        :param h: step size used to discretize timescale
        :param device: device used
        :param clipping: created samples will be clipped to be in the range specified by clip_limits
        :param clip_limits: range of valid values if clipping is applied
        """
        self._model = model
        self.__h = h
        self._device = device

        self.__clipping = clipping
        self.__clip_limits = clip_limits

    def sample_subsequent(self, x0: torch.Tensor,
                          burn_in: int = 1,
                          batch_size: int = 1,
                          sample_distance: int = 1) -> torch.Tensor:
        """ Creates a batch of subsequent samples.

        :param x0: starting point
        :param burn_in: amount of time steps after when to consider samples to be from target distribution
        :param batch_size: size of batch of subsequent samples
        :param sample_distance: amount of time steps left between two subsequent samples
        :return: tensor containing all samples with first dimension indicating batch numbering
        """
        samples = torch.empty(torch.Size([batch_size] + list(x0.shape)[1:]), device=self._device)

        x = self.sample(x0, burn_in)
        samples[0] = x

        for i in range(batch_size-1):
            x = self.sample(x, sample_distance)
            samples[i] = x.clone().detach()

        return samples

    def sample(self, x0: torch.Tensor, burn_in: int = 1) -> torch.Tensor:
        """ Creates one sample

        :param x0: starting point
        :param burn_in: amount of time steps after when to consider samples to be from target distribution
        :return:
        """

        # Initialization
        x = x0.to(self._device)
        x.requires_grad = True

        noise = torch.empty_like(x)

        for p in self._model.parameters():
            p.requires_grad = False

        # Sample loop
        for k in range(burn_in):
            # Calculate model output
            y = self._model(x)

            # Calculate gradient
            grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)
            grad = grad[0]

            # Add gradient to the input
            x.data.add_(grad.data, alpha=-1. * self.__h)

            # Add noise to the input
            noise.normal_(0, 1)
            x.data.add_(noise.data, alpha=math.sqrt(2 * self.__h))

            # Apply input clipping
            if self.__clipping:
                x.data.clamp_(self.__clip_limits[0], self.__clip_limits[1])

        # Overall cleanup
        for p in self._model.parameters():
            p.requires_grad = True

        x.requires_grad = False
        # Output
        return x

    def __str__(self):
        return "LangevinSampler(model, '%s', '%s', '%s', '%s')" \
            % (self.__h, self._device, self.__clipping, self.__clip_limits)

    def __repr__(self):
        return str(self)
