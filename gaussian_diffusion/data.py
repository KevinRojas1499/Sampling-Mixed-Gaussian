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


def subsample(data, res):
    print("Data shape: ", data.shape)
    assert data.shape[-1] % res == 0
    new_data = data[:, ::(data.shape[-1] // res)]
    print("Subsampled shape: ", new_data.shape)
    return new_data

def generate_piecewise_func(res = 64, n_samples = 10000, num_intervals=5,save=False):
    piecewise_func = torch.zeros((n_samples,res))
    delta = 1/res
    for j in range(n_samples):
        rand_partition = torch.sort(torch.rand(num_intervals))[0]
        rand_values = torch.rand(num_intervals)*2
        k = 0
        for i in range(res):
            if k+1 < num_intervals and delta*i > rand_partition[k+1]:
                k+=1
            piecewise_func[j][i] = rand_values[k]
    if save:
        torch.save(piecewise_func, "datasets/random_piecewise_func.pt")
    return piecewise_func


def generate_sin_dist(res=1024, n_samples=10000, low_res=64, save=True, min_amplitude=0):
    grid = torch.linspace(-1, 1, res)
    amp_freqs = torch.repeat_interleave(torch.rand((n_samples, 2)), res, dim=1).view(n_samples, 2, res)
    amp_freqs+=min_amplitude
    # Scale max freq to 10. Keep amplitude small to avoid training issues
    samples = amp_freqs[:, 0]*torch.sin(10*amp_freqs[:, 1]*grid)

    uniform = "uniform" if min_amplitude == 0 else "nonuniform"
    file_name = f"datasets/random_{uniform}_sin_1024.pt"
    torch.save(samples, file_name)

    # Generate uniform coarsening
    uniform_coarsening = samples[:, ::(res // low_res)]
    print(uniform_coarsening.shape)
    if save :
        torch.save(uniform_coarsening, file_name)
    return uniform_coarsening

def generate_piecewise_sin_dist(res,n_samples,low_res, save=True):
    sin_dist = generate_sin_dist(res,n_samples,low_res,save=False)
    piece_dist = generate_piecewise_func(low_res,n_samples,5,save=False)
    sin_piece_dist = sin_dist+piece_dist
    if save: 
        torch.save(sin_piece_dist,"datasets/random_piecewise_sin_64.pt")
    return sin_piece_dist


def visualize_sin(sins, file_name):
    res = sins.shape[-1]
    grid = torch.linspace(-1, 1, res)

    plt.clf()
    for i in range(5):
        plt.plot(grid, sins[i].flatten().cpu())
    plt.savefig("figs/{}.png".format(file_name))

    for i in range(5):
        plt.clf()
        plt.plot(grid, sins[i].flatten().cpu())
        print("Plotting {}_{}.png".format(file_name, i))
        plt.savefig("figs/{}_{}.png".format(file_name, i))



if __name__ == "__main__":
    #generate_sin_dist(low_res=64)
    #visualize_sins()
    generate_piecewise_sin_dist(64,1,64,False)
    # subsampled = subsample(torch.load("datasets/random_sin_1024.pt"), 128)
    # torch.save(subsampled, "datasets/random_sin_128.pt")