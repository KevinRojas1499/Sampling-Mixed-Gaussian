import matplotlib.pyplot as plt 
import torch
import numpy as np





############## Plotting ##############

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


def plotTrajectories(trajectories, title):
    plt.title("Trajectories Fourier")
    for i in range(len(trajectories)):
        s = trajectories[i].to('cpu').numpy()
        s = np.array(np.split(s,s.shape[0]//2))
        plt.plot(s[:,0],s[:,1])
    plt.show()

############## Fourier Transforms ##############


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
