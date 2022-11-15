import os
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

# Now we are in 4D instead of 2D
samples = torch.cat((samples.real,samples.imag),dim=1) 

# plt.scatter(samples[:,0],samples[:,1],color='red')
# plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
score_function = model.Score(4)

checkpointPath = './withFFTLayer.pth'
if os.path.exists(checkpointPath):
    checkpoint = torch.load(checkpointPath)
    score_function.load_state_dict(checkpoint)


score_function = score_function.to(device=device)
samples = samples.to(device=device)

train = True
train = False


if train:
    errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=150001, fileToSave=checkpointPath, device=device)
    errors.detach()
    plt.plot(np.linspace(1,len(errors),len(errors)),errors)
    plt.show()
else:
    # This are in 4D
    generatedSamplesFFT = sde.generate_samples_reverse(score_network=score_function, nsamples=1000)
    real, imaginary = torch.chunk(generatedSamplesFFT,2,dim=1)

    # Back to 2D in frequency space
    complexGenerated = torch.complex(real,imaginary)

    # Back to original space (hopefully)
    generatedSamples = torch.fft.ifft(complexGenerated,norm="forward")

    print(generatedSamples)

    # plt.scatter(samples[:,0],samples[:,1],color='red')
    # plt.scatter(generatedSamplesFFT[:,0],generatedSamplesFFT[:,1],color='blue')
    # plt.show()

    realPart = generatedSamples.real.type(torch.double)
    ab = torch.ones(1000) / 1000
    M = ot.dist(samplesBeforeFFT,realPart, metric='euclidean')
    print(samplesBeforeFFT.size(),realPart.size())
    print(ot.emd2(ab,ab,M))

    plt.scatter(samplesBeforeFFT[:,0],samplesBeforeFFT[:,1],color='red')
    plt.scatter(generatedSamples[:,0].real,generatedSamples[:,1].real,color='blue')

    plt.show()