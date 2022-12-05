from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import torch
import sde_lib
import model
import training
import ot
import wandb
import generateSamples


num_samples = 1000

c = [1/2,1/6,1/3]
means = [[0.5,0.5],[-15,15], [8,8]]
variances = [[[1,0],[0,1]], [[5, -2],[-2,5]] , [[1, 2],[2,1]]]

def beta(t):
    return 20*t

def drift(x,t):
    return -beta(t)*x/2

def diffusion(t):
    return sqrt(beta(t))

sde = sde_lib.SDE(100,1,beta=beta(1))

device = 'cpu'
score_function = model.Score(2)
checkpoint = torch.load('./coefficientsNoFFT.pth')
score_function.load_state_dict(checkpoint)
score_function.to(device=device)


samples = torch.tensor(generateSamples.get_samples_from_mixed_gaussian(c,means,variances,num_samples))
samples = samples.to(device=device)

#train = True
train = False


def generate_samples(score_network: torch.nn.Module, nsamples: int) -> torch.Tensor:
    x_t = torch.randn((nsamples, 2))
    time_pts = torch.linspace(1, 0, 1000)
    beta = lambda t: beta(t)
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
    generatedSamples = generate_samples(score_network=score_function, nsamples=1000)
    print(generatedSamples.type())


    ab = torch.ones(1000) / 1000
    realPart = generatedSamples.real.type(torch.double)
    M = ot.dist(samples,realPart, metric='euclidean')
    print(ot.emd2(ab,ab,M))

    plt.scatter(samples[:,0],samples[:,1], color='red')
    plt.scatter(generatedSamples[:,0],generatedSamples[:,1])
    plt.show()
    plt.savefig("samples.png")