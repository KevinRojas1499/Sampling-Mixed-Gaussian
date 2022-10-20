from bisect import bisect_left
from cmath import sqrt
from distutils.log import error
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_sample_from_multi_gaussian(lambda_,gamma_,mean):
    # Here lambda and gamma are the eigen decomposition of the corresponding covariance matrix
    dimensions = len(lambda_)
    # sampling from normal distribution
    x_normal = np.random.randn(dimensions)
    # transforming into multivariate distribution
    x_multi = (x_normal*lambda_) @ gamma_ + mean
    return x_multi

def plot_histogram(samples):
    fig, axs = plt.subplots(1, 1,
                            figsize =(10, 7),
                            tight_layout = True)
    
    axs.hist(samples, bins = 200)
    plt.show()

num_samples = 1000

cov = [[1,2],[2,1]]
mean = [0,0]

def plot_2d_gaussian_distribution(covariance,mean):
    samples_x = []
    samples_y = []

    for i in range(num_samples): 
        x,y = get_sample_from_multi_gaussian(covariance,mean)
        samples_x.append(x)
        samples_y.append(y)

    plt.scatter(samples_x,samples_y)
    plt.show()

# plot_2d_gaussian_distribution(cov,mean)

# This is the case 1D
# c = [1/2,1/6,1/3]
# means = [0,5,15]
# variances = [1,1,1]

# def get_samples_from_mixed_gaussian(c,means,variances):
#     n = len(c)
#     accum = np.zeros(n)
#     print(accum)
#     accum[0] = c[0]
#     for i in range(1,n):
#         accum[i] = accum[i-1]+c[i]
    
#     samples = []
#     for i in range(num_samples):
#         idx = bisect_left(accum,np.random.rand(1)[0])
#         samples.append(np.random.randn(1)[0]*variances[idx]+means[idx])
#     return samples


# This is the case 2D
c = [1/2,1/6,1/3]
means = [[0.5,0.5],[15,15], [8,8]]
variances = [[[1,0],[0,1]], [[1, 3],[3,1]] , [[1, 2],[2,1]]]

def get_samples_from_mixed_gaussian(c,means,variances):
    n = len(c)
    accum = np.zeros(n)
    accum[0] = c[0]
    for i in range(1,n):
        accum[i] = accum[i-1]+c[i]
    lambdas = []
    gamma = []
    for i in range(n):
        lambda_, gamma_ = np.linalg.eig(np.array(variances[i]))
        lambdas.append(lambda_)
        gamma.append(gamma_)
    samples = []
    for i in range(num_samples):
        idx = bisect_left(accum,np.random.rand(1)[0])
        samples.append(get_sample_from_multi_gaussian(lambda_=lambdas[idx],gamma_=gamma[idx],mean=means[idx]))
    return samples


def scatter_plot(points):
    plt.scatter(points[:,0],points[:,1])
    plt.show()



def alpha(t):
    return 3

def drift(x,t):
    return -alpha(t)*x

def diffusion(t):
    A = np.array([[.1,.1],[.1,.1]])
    return sqrt(2*alpha(t))

import sde_lib 
import model
import training


sde = sde_lib.SDE(100,1,drift, diffusion)
samples = np.array(get_samples_from_mixed_gaussian(c,means,variances))
from sklearn.datasets import make_swiss_roll

# generate the swiss roll dataset
xnp, _ = make_swiss_roll(1000, noise=1.0)
samples = torch.as_tensor(xnp[:, [0, 2]] / 10.0, dtype=torch.float32)

# diffusedSamples = []
# for sample in samples:
#     newSample = sde.discretize(sample)
#     diffusedSamples.append([newSample[0],newSample[1]])

# diffusedSamples = np.array(diffusedSamples)


device = 'cpu'
print(torch.cuda.is_available())
score_function = model.Score(2)
# checkpoint = torch.load('./coefficients.pth')
# score_function.load_state_dict(checkpoint)
score_function.to(device=device)

# samples = torch.tensor(samples)
samples.to(device=device)

train = True
# train = False


if train:
    errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=10000)
    plt.plot(np.linspace(1,len(errors),len(errors)),errors)
    plt.show()
else:
    # We generate some samples
    generatedSamples = []
    reverse = sde_lib.ReverseSDE(sde=sde, score=score_function)
    i = 0
    samples = []
    for j in range(100):
        samples.append(np.random.randn(2)*(1-np.math.exp(-2*alpha(1))))

    samples = np.array(samples)
    plt.scatter(samples[:,0],samples[:,1])
    plt.show()
    for sample in samples:
        newSample = reverse.discretize(sample)
        i+=1
        generatedSamples.append([newSample[0],newSample[1]])

    generatedSamples = np.array(generatedSamples)
    plt.scatter(generatedSamples[:,0],generatedSamples[:,1])
    plt.show()