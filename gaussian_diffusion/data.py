import numpy
import scipy.fftpack
import numpy as np
import cv2

import torch
import math


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


if __name__ == "__main__":
    grf = GaussianRF(1, 10)
    print(grf.sample(1))