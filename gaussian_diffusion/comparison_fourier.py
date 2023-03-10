import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sde_lib 
import model
import training
import ot
import wandb
import gaussian_diffusion.generate_samples as generate_samples
from utils import *



num_samples = 1000

# 3D Example
# c = [1/3,1/4,1/5,1/6,1/20]
# means = [[40,40,0.],[-15,-20,0], [30,10,20],[-3,-3,-3],[0,0,0]]
# variances = [[[1,0,0],[0,1,0],[0,0,1]], [[5,1,-2],[1,1,3],[-2,3,5]] , [[1, 2,3],[2,5,6],[3,6,1]],[[5,1,-2],[1,1,3],[-2,3,5]] , [[1, 2,3],[2,5,6],[3,6,1]]]


c = [1/3,1/4,1/5,1/6,1/20]
means = [[0.5,0.5],[-15,-20], [30,10], [7,15], [-6, 15]]
variances = [[[1,0],[0,1]], [[5,1],[1,1]] , [[1, 2],[2,5]], [[-1,-2],[-2,-5]],[[1,2],[2,4]]]


sde = sde_lib.LinearSDE(beta=20)
np.random.seed(0)
samplesBeforeFFT = generate_samples.get_samples_from_mixed_gaussian(c,means,variances,num_samples)

samplesFFT = torch.fft.fft(samplesBeforeFFT)

dualPlot(samplesBeforeFFT,samplesFFT,'Comparison')

path = './checkpoints2D/'
methods = ['normal']
# methods = ['fft']


def train_all():
    for method in methods:
        samples = samplesBeforeFFT
        dim = samples.shape[1]
        if method == 'fft':
            # Move to frequency space and train both real and imaginary
            samples = torch.fft.rfft(samplesBeforeFFT,n=dim, norm="forward")
            for i,sample in enumerate(samples):
                samples[i] =  getTransform(sample)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        score_function = model.Score(dim)

        path = './checkpoints/' if dim == 3 else './checkpoints2D/'
        checkpointPath = path+method
        if os.path.exists(checkpointPath):
            checkpoint = torch.load(checkpointPath)
            score_function.load_state_dict(checkpoint)


        score_function = score_function.to(device=device)
        samples = samples.to(device=device)

        errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=150001, file_to_save=checkpointPath, device=device,lr=0.01,wd=0,epochs=150001)
        
# train_all()

def sample(method,dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    score_function = model.Score(dim)
    path = './checkpoints/' if dim == 3 else './checkpoints2D/'
    checkpointPath = path+method
    if method == 'sampleFourier':
        checkpointPath = path+'normal'

    if os.path.exists(checkpointPath):
        checkpoint = torch.load(checkpointPath)
        score_function.load_state_dict(checkpoint)

    score_function = score_function.to(device=device)
    if method == 'sampleFourier':
        generatedSamples, trajectories = sde.generate_samples_reverse_fft(score_network=score_function, shape=[dim], nsamples=1000,ode=False)
        plotTrajectories(trajectories,"Trajectories Fourier")
    else: 
        generatedSamples, trajectories = sde.generate_samples_reverse(score_network=score_function, shape=[dim] , nsamples=1000,ode=False)
        plotTrajectories(trajectories,"Trajectories Normal")

    if method != 'normal':
        for i , samp in enumerate(generatedSamples):
            generatedSamples[i] = torch.fft.irfft(getInverseTransform(samp,dim),n=dim,norm="forward")
        
    generatedSamples = generatedSamples.to('cpu')
    print(generatedSamples.shape)
    return generatedSamples


# for method in methods:
#     np.random.seed(2)
#     newSamples = sample(method,2)
#     title = "Diffusion using "+method
#     # dual3DPlot(samplesBeforeFFT,newSamples, title)
#     dualPlot(samplesBeforeFFT,newSamples,title)
#     realPart = newSamples.real.type(torch.double)
#     ab = torch.ones(1000) / 1000
#     M = ot.dist(samplesBeforeFFT,realPart, metric='euclidean')
#     print(f"METHOD {method} {ot.emd2(ab,ab,M)}")

def fourierSample2D():
    newSamples = sample('sampleFourier',2)
    title = "Diffusion using "
    dualPlot(samplesBeforeFFT,newSamples,title)

    realPart = newSamples.real.type(torch.double)
    ab = torch.ones(1000) / 1000
    M = ot.dist(samplesBeforeFFT,realPart, metric='euclidean')
    print(f"METHOD {'sampleFourier'} {ot.emd2(ab,ab,M)}")

def fourierSample3D():
    newSamples = sample('sampleFourier',3)
    title = "Diffusion using "
    dual3DPlot(samplesBeforeFFT,newSamples,title)

# fourierSample3D()
# fourierSample2D()

samples = sample('normal',2)
dualPlot(samplesBeforeFFT,samples,"Dual")