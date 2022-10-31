from bisect import bisect_left
from cmath import sqrt
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import torch
import sde_lib 
import model
import training


def get_sample_from_multi_gaussian(lambda_,gamma_,mean):
    # Here lambda and gamma are the eigen decomposition of the corresponding covariance matrix
    dimensions = len(lambda_)
    # sampling from normal distribution
    x_normal = np.random.randn(dimensions)
    # transforming into multivariate distribution
    x_multi = (x_normal*lambda_) @ gamma_ + mean
    return x_multi

num_samples = 1000

c = [1/2,1/6,1/3]
means = [[0.5,0.5],[-15,15], [8,8]]
variances = [[[1,0],[0,1]], [[5, -2],[-2,5]] , [[1, 2],[2,1]]]

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

def beta(t):
    return 20*t

def drift(x,t):
    return -beta(t)*x/2

def diffusion(t):
    return sqrt(beta(t))


sde = sde_lib.SDE(100,1,beta=beta(1))
samplesBeforeFFT = torch.tensor(get_samples_from_mixed_gaussian(c,means,variances))
samples = torch.fft.fft(samplesBeforeFFT,norm="forward")


plt.scatter(samples[:,0],samples[:,1],color='red')
plt.show()

device = 'cpu'
score_function = model.Score(2)
checkpoint = torch.load('./coefficients.pth')
score_function.load_state_dict(checkpoint)


score_function.to(device=device)
samples.to(device=device)

train = True
train = False


def generate_samples(score_network: torch.nn.Module, nsamples: int) -> torch.Tensor:
    x_t = torch.randn((nsamples, 2))
    time_pts = torch.linspace(1, 0, 1000)
    beta = lambda t: 0.1 + (20 - 0.1) * t
    for i in range(len(time_pts) - 1):
        t = time_pts[i]
        dt = time_pts[i + 1] - t
        score = score_network(x_t,t.expand(x_t.shape[0], 1)).detach()
        tot_drift = drift(x_t,t) - diffusion(t)**2 * score
        tot_diffusion = diffusion(t)

        # euler-maruyama step
        x_t = x_t + tot_drift * dt + tot_diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
    return x_t

if train:
    errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=150001)
    plt.plot(np.linspace(1,len(errors),len(errors)),errors)
    plt.show()
else:
    generatedSamplesBeforeFFT = generate_samples(score_network=score_function, nsamples=1000)

    plt.scatter(samples[:,0],samples[:,1],color='red')
    plt.scatter(generatedSamplesBeforeFFT[:,0],generatedSamplesBeforeFFT[:,1],color='blue')
    plt.show()

    generatedSamples = torch.fft.ifft(generatedSamplesBeforeFFT,norm="forward")
    plt.scatter(samplesBeforeFFT[:,0],samplesBeforeFFT[:,1],color='red')
    plt.scatter(generatedSamples[:,0],generatedSamples[:,1],color='blue')
    plt.show()