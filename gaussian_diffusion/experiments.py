import numpy as np 
import torch 

# dimension = 2

# def getJacobianOfFourierTransform(dimension):
#     jacobianOfFourierTransform = np.zeros((dimension,dimension))
#     for i in range(dimension):
#       for j in range(dimension):
#         if i%2 == 1 or i == 0:
#            jacobianOfFourierTransform[i,j] = np.cos(2*np.pi*i*j/dimension)
#         else:
#            jacobianOfFourierTransform[i,j] = np.sin(2*np.pi*i*j/dimension)
#     return jacobianOfFourierTransform

# print(getJacobianOfFourierTransform(3))

# t = torch.arange(6)
# print(t)
# ft = torch.fft.rfft(t)

# print(ft)
# a = []

# for c in ft:
#     a.append(torch.real(c))
#     if torch.imag(c).item() != 0 :
#         a.append(torch.imag(c))
# print(a)

# tt = torch.tensor([ 0.7626,  1.1540, -1.0174])
# print(torch.fft.irfft(tt,3))

val = torch.complex(torch.tensor([1.]),torch.tensor([2.]))
print(val)
def getInverseTransform(ft,dim):
    n  = ft.shape[0]
    newT = torch.zeros(dim,dtype=torch.cfloat)
    k = 0
    for i,val in enumerate(ft):
        if i == 0:
            newT[k] = val
            k+=1
            continue
        if(i%2 == 1):
            newT[k] += val 
        else :
            newT[k] += torch.complex(torch.tensor(0.),val)
            k+=1
    return newT

print(getInverseTransform(torch.tensor([3,2]),2))