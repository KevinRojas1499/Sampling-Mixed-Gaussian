import numpy
import scipy.fftpack
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import math
from tqdm import tqdm


class GaussianRF(object):

    def __init__(self, dim, size, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", device=None):
        """
        dim : dimension of output functions
        size : ?
        alpha : ?
        tau : ?
        sigma : ?
        """
        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N, mul=1):

        coeff = torch.randn(N, *self.size, 2, device=self.device)*mul

        coeff[...,0] = self.sqrt_eig*coeff[...,0] #real
        coeff[...,1] = self.sqrt_eig*coeff[...,1] #imag

        ##########torch 1.7###############
        #u = torch.ifft(coeff, self.dim, normalized=False)
        #u = u[...,0]
        ##################################

        #########torch latest#############
        coeff_new = torch.complex(coeff[...,0],coeff[...,1])
        #print(coeff_new.size())
        u = torch.fft.ifft2(coeff_new, dim = (-2,-1), norm=None)
        u = u.real
        return u


def generate_sin_dist(res=1024, n_samples=10000, low_res=64):
    grid = torch.linspace(-1, 1, res)
    amp_freqs = torch.repeat_interleave(torch.abs(torch.rand((n_samples, 2))), res, dim=1).view(n_samples, 2, res)
    # Scale max freq to 10. Keep amplitude small to avoid training issues
    amp_freqs[:, 1] = 10*amp_freqs[:, 1]
    samples = amp_freqs[:, 0]*torch.sin(amp_freqs[:, 1]*grid)
    torch.save(samples, "datasets/random_sin_1024.pt")

    coarsenings = []
    # Generating coarsenings
    for sample in tqdm(samples):
        inds = torch.randperm(res)[:low_res]
        ord_inds, _ = torch.sort(inds)
        coarsenings.append(sample[ord_inds])
    coarsenings = torch.stack(coarsenings)
    print(coarsenings.shape)
    torch.save(samples, "datasets/random_sin_64.pt")

    # Generate uniform coarsening
    uniform_coarsening = samples[:, ::(res // low_res)]
    print(uniform_coarsening.shape)
    torch.save(uniform_coarsening, "datasets/random_sin_64_uniform.pt")


def visualize_sin(res=1024):
    grid = torch.linspace(-1, 1, res)[::(1024//64)]
    print(grid.shape)
    #samples = torch.load("datasets/random_sin_64_uniform.pt")
    samples = torch.load("samples/random_sin_64_uniform_samples.pt")
    print(samples.shape)

    plt.clf()
    plt.plot(grid, samples[0].flatten().cpu())
    plt.savefig("figs/syn_sin.png")



if __name__ == "__main__":
    #generate_sin_dist()
    visualize_sin()