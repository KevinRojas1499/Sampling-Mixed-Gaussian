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


sde = sde_lib.LinearSDE(beta=20)
samplesBeforeFFT = torch.tensor(generateSamples.get_samples_from_mixed_gaussian(c,means,variances,num_samples))
samples = torch.fft.fft(samplesBeforeFFT,norm="forward")

device = 'cpu'
score_function = model.Score(2)
checkpoint = torch.load('./coefficientsNoFFT.pth')
score_function.load_state_dict(checkpoint)
score_function.to(device=device)


samples = torch.tensor(generateSamples.get_samples_from_mixed_gaussian(c,means,variances,num_samples))
samples = samples.to(device=device)

train = True
train = False


if train:
    errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=150001)
    plt.plot(np.linspace(1,len(errors),len(errors)),errors)
    plt.show()
else:
    generatedSamplesFFT = sde.generate_samples_reverse(score_network=score_function, nsamples=1000)


    ab = torch.ones(1000) / 1000
    realPart = generatedSamplesFFT.real.type(torch.double)
    M = ot.dist(samples,realPart, metric='euclidean')
    print(ot.emd2(ab,ab,M))

    plt.scatter(samples[:,0],samples[:,1], color='red')
    plt.scatter(generatedSamplesFFT[:,0],generatedSamplesFFT[:,1])
    plt.show()