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



def plot3D(samples,ax,title):
    ax.scatter3D(samples[:,0], samples[:,1], samples[:,2], color = "green")
    plt.title(title)

def dual3DPlot(samples1,samples2,title):    
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(1, 2, 1, projection='3d')    
    plot3D(samples1, ax,"Samples Before")

    ax = fig.add_subplot(1, 2, 2, projection='3d')    
    plot3D(samples2, ax, "Generated Samples")

    plt.show()  

def dualPlot(samples1, samples2,title):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(1, 2, 1)    

    plt.title("Samples Before")
    ax.scatter(samples1[:,0],samples1[:,1],color='red')


    ax = fig.add_subplot(1, 2, 2)    
    plt.title("Samples After")
    ax.scatter(samples2[:,0],samples2[:,1],color='blue')

    plt.show()


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
samplesBeforeFFT = torch.tensor(generateSamples.get_samples_from_mixed_gaussian(c,means,variances,num_samples))

samplesFFT = torch.fft.fft(samplesBeforeFFT)

dualPlot(samplesBeforeFFT,samplesFFT,'Comparison')

path = './checkpoints2D/'
methods = ['normal','fft']
# methods = ['fft']

def getTransform(ft):
  a = []
  for c in ft:
    a.append(torch.real(c))
    if torch.imag(c).item() != 0 :
        a.append(torch.imag(c))
  return torch.tensor(a)

def getInverseTransform(ft,dim):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  newT = torch.zeros(dim,dtype=torch.cfloat,device=device)
  k = 0
  for i,val in enumerate(ft):
      if i == 0:
          newT[k] = val
          k+=1
          continue
      if(i%2 == 1):
          newT[k] += val 
      else :
          newT[k] += torch.complex(torch.tensor(0.,device=device),val)
          k+=1
  return newT

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

        errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=150001, fileToSave=checkpointPath, device=device)
        
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

    k = 1000
    score_function = score_function.to(device=device)
    if method == 'sampleFourier':
        generatedSamples, trajectories = sde.generate_samples_reverse_fft(score_network=score_function, dimension = dim, nsamples=1000,ode=True)
        
        plt.title("Trajectories Fourier")
        for i in range(len(trajectories)):
            s = trajectories[i].to('cpu').numpy()
            s = np.array(np.split(s,s.shape[0]//2))
            plt.plot(s[:k,0],s[:k,1])
        plt.show()
    else: 
        generatedSamples, trajectories = sde.generate_samples_reverse(score_network=score_function, dimension = dim, nsamples=1000,ode=True)

        plt.title("Trajectories Normal")
        for i in range(len(trajectories)):
            s = trajectories[i].to('cpu').numpy()
            s = np.array(np.split(s,s.shape[0]//2))
            plt.plot(s[:k,0],s[:k,1])
        plt.show()

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
fourierSample2D()

samples = sample('normal',2)
dualPlot(samplesBeforeFFT,samples,"Dual")