# Langevin Monte Carlo Sampling with Applications to Energy Based Modelling
This repository implements various algorithms related to langevin monte carlo sampling (LMC).
The implementations are based on PyTorch.

<!-- TOC -->
1. [Introduction](#introduction)
2. [Langevin Monte Carlo Samling Algorithm](#langevin-monte-carlo-samling-algorithm)
3. [Maximum Likelihood Training](#maximum-likelihood-training)
4. [Further Usages](#further-usages)
   1. [Image Reconstruction](#image-reconstruction)
   2. [Inpainting](#inpainting)
<!-- TOC -->

## Introduction

LMC is especially useful in the context of energy based models (EBMs) - parametrized smooth probability density functions
which are only specified up to a multiplicative constant.

The first section is devoted on the sampling from an EBM with fixed parameters. In the second section, fitting parameters
with respect to a given dataset is considered. Finally, some further usages of EBMs like image restoration and classifiction
are considered.
The "advanced" examples are based on the MNIST dataset.

## Langevin Monte Carlo Samling Algorithm

The <code>LangevinSampler</code> class from Langevin.py implements the basic unadjusted LMC algorithm according to the 
iteration scheme 
$$x_{k+1}=x_k-\nabla f_\theta(x_k)+\sqrt{2h}z,$$
where $z \sim \mathcal{N}(0,1)$ are independent standard normally distributed and $f_\theta(x)$ denotes the energy function.

## Maximum Likelihood Training

The <code>Trainer</code> class from Trainer.py implements a maximum-likelihood based learning algorithm in order to fit 
the parameters of an EBM.

In addition, the <code>Buffer</code> class from Buffer.py implements sample replay buffering of samples created from past 
model distributions.

Basic use cases can be found in main_func.py. It contains configurations for three different scenarios:

- Fitting a quadratic energy function to a dataset sampled from a normal distribution.
Involved classes are <code>NormalDataset</code> from DatasetGenerators.py and <code>GaussianModel</code> from Gaussian.py.
- The second scenario considers the same dataset but restriced to a quadratic box. This needs more complicated models which
can capture the non-normal behaviour, however, the repository does not provide these models yet.
- The last scenario considers the MNIST dataset together with a convolutional neural network model for the energy function.

## Further Usages

This section deals with the MNIST dataset together with an already trained model according to the previous section.

### Image Reconstruction

The idea is to use the energy function as $f_\theta$ regularization in order to solve for the reconstruction problem
$$\min_x \sigma \| x - y \| ^2 + \lambda f_\theta (x).$$

An example is given in reconstruction.ipynb. The below figure shows a particular result. Note that the model and MNIST 
dataset path need to be specified for reconstruction.ipynb to work.

![Example of image reconstructed from original with added white noise](img/img_reconstruction.png "Image reconstruction")

### Inpainting

Given some mask as on the figure below, can restrict the energy function to a masked area and consider only that area for 
optimization. Example is given in inpainting.ipynb.

![Example of image inpainting from masked original](img/img_inpainting.png "Image inpainting")

![Example of image inpainting from masked original](img/img_inpainting.gif "Image inpainting")