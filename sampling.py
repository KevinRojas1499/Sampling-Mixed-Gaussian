from bisect import bisect_left
from cmath import sqrt
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import torch
import sde_lib 
import model
import training
import ot
import generateSamples

num_samples = 1000

c = [1/2,1/6,1/3]
means = [[0.5,0.5],[-15,15], [8,8]]
variances = [[[1,0],[0,1]], [[5, -2],[-2,5]] , [[1, 2],[2,1]]]

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
samplesBeforeFFT = torch.tensor(generateSamples.get_samples_from_mixed_gaussian(c,means,variances,num_samples))
samples = torch.fft.fft(samplesBeforeFFT,norm="forward")


# plt.scatter(samples[:,0],samples[:,1],color='red')
# plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
score_function = model.Score(2)
checkpoint = torch.load('./coefficientsFFT.pth')
score_function.load_state_dict(checkpoint)


score_function = score_function.to(device=device)
samples = samples.to(device=device)

train = True
train = False


def generate_samples(score_network: torch.nn.Module, nsamples: int) -> torch.Tensor:
    x_t = torch.randn((nsamples, 2))
    time_pts = torch.linspace(1, 0, 1000)
    beta = lambda t:  beta(t)
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
    errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=150001,device=device)
    plt.plot(np.linspace(1,len(errors),len(errors)),errors)
    plt.show()
else:
    generatedSamplesFFT = generate_samples(score_network=score_function, nsamples=1000)

    # plt.scatter(samples[:,0],samples[:,1],color='red')
    # plt.scatter(generatedSamplesFFT[:,0],generatedSamplesFFT[:,1],color='blue')
    # plt.show()

    generatedSamples = torch.fft.ifft(generatedSamplesFFT,norm="forward")

    realPart = generatedSamples.real.type(torch.double)
    ab = torch.ones(1000) / 1000
    M = ot.dist(samplesBeforeFFT,realPart, metric='euclidean')
    print(samplesBeforeFFT.size(),realPart.size())
    print(ot.emd2(ab,ab,M))


    plt.scatter(samplesBeforeFFT[:,0],samplesBeforeFFT[:,1],color='red')
    plt.scatter(generatedSamples[:,0].real,generatedSamples[:,1].real,color='blue')
    plt.show()