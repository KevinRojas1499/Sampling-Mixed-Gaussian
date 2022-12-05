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

fft = False
fft = True


c = [1/2,1/6,1/3]
means = [[0.5,0.5,0.5],[-15,-20,0], [30,10,20]]
variances = [[[1,0,0],[0,1,0],[0,0,1]], [[5,1,-2],[1,1,3],[-2,3,5]] , [[1, 2,3],[2,5,6],[3,6,1]]]

sde = sde_lib.LinearSDE(beta=20)
samplesBeforeFFT = torch.tensor(generateSamples.get_samples_from_mixed_gaussian(c,means,variances,num_samples))
samples = samplesBeforeFFT


if fft:
    # Move to frequency space
    samples = torch.fft.fft(samplesBeforeFFT,norm="forward")

    # Now we are in 6D instead of 3D
    samples = torch.cat((samples.real,samples.imag),dim=1) 
    print(samples)


# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")

# ax.scatter3D(samplesBeforeFFT[:,0], samplesBeforeFFT[:,1], samplesBeforeFFT[:,2], color = "green")
# plt.title("simple 3D scatter plot")
# plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

dim = 6 if fft else 3
score_function = model.Score(dim)

checkpointPath = './3DFFT.pth'
if os.path.exists(checkpointPath):
    checkpoint = torch.load(checkpointPath)
    score_function.load_state_dict(checkpoint)


score_function = score_function.to(device=device)
samples = samples.to(device=device)

train = True
# train = False


if train:
    errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=150001, fileToSave=checkpointPath, device=device)
else:
    generatedSamples = sde.generate_samples_reverse(score_network=score_function, dimension = dim, nsamples=1000)
    
    if fft:
        real, imaginary = torch.chunk(generatedSamples,2,dim=1)

        # Back to 2D in frequency space
        complexGenerated = torch.complex(real,imaginary)

        # Back to original space (hopefully)
        generatedSamples = torch.fft.ifft(complexGenerated,norm="forward")
        print(generatedSamples)

    generatedSamples = generatedSamples.to(device='cpu')
    # plt.scatter(samples[:,0],samples[:,1],color='red')
    # plt.scatter(generatedSamplesFFT[:,0],generatedSamplesFFT[:,1],color='blue')
    # plt.show()

    # realPart = generatedSamples.real.type(torch.double)
    # ab = torch.ones(1000) / 1000
    # M = ot.dist(samplesBeforeFFT,realPart, metric='euclidean')
    # print(samplesBeforeFFT.size(),realPart.size())
    # print(ot.emd2(ab,ab,M))

    # plt.scatter(samplesBeforeFFT[:,0],samplesBeforeFFT[:,1],color='red')
    # plt.scatter(generatedSamples[:,0].real,generatedSamples[:,1].real,color='blue')


    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    ax.scatter3D(samplesBeforeFFT[:,0], samplesBeforeFFT[:,1], samplesBeforeFFT[:,2], color = "green")
    ax.scatter3D(generatedSamples[:,0], generatedSamples[:,1], generatedSamples[:,2], color = "blue")
    
    plt.title("simple 3D scatter plot")
    plt.show()
