import numpy as np 
import torch 
import time
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from PIL import Image 

import numpy as np 
import torch 
import torchvision.transforms.functional as F

def modifiedRFFT(a):
    # Only working for even N
    n = a.shape[0]
    b = torch.fft.rfft(a,norm="ortho")
    re, im = b.real, b.imag
    k = re.shape[0]
    c = torch.zeros(n)
    c[0] = re[0]
    c[1:1+2*(k-2):2] = re[1:-1]
    c[2:2+2*(k-2):2] = im[1:-1]
    c[-1] = re[-1]
    return c

def modifiedIRFFT(a):
    # Only working for even N
    n = a.shape[0]
    k = (n+2)//2
    real = torch.zeros(k)
    im = torch.zeros(k)
    real[0] = a[0]
    real[-1] = a[-1]
    real[1:-1] = a[1:n-2:2]
    im[1:-1] = a[2:n-1:2]
    fft = torch.complex(real,im)
    return torch.fft.irfft(fft,norm="ortho")

# TODO (kevin) : Parallelize this functions

def modifiedRFFT2(image):
    newImage = image.clone()
    # newImage = functorch.vmap(modifiedRFFT)(newImage)
    # newImage = functorch.vmap(modifiedRFFT,in_dims=1)(newImage)
    newImage = torch.stack([modifiedRFFT(row) for row in newImage])
    newImage = torch.stack([modifiedRFFT(col) for col in newImage.T],dim=1)
    return newImage

def modifiedIRFFT2(image):
    newImage = image.clone()
    newImage = torch.stack([modifiedIRFFT(col) for col in newImage.T],dim=1)
    newImage = torch.stack([modifiedIRFFT(row) for row in newImage])
    return newImage

def completeRFFT2(image):
    rfft = torch.zeros(image.shape)
    for i in range(3):
        rfft[i,:,:] = modifiedRFFT2(image[i,:,:])
    return rfft

def completeIRFFT2(image):
    irfft = torch.zeros(image.shape)
    for i in range(3):
        irfft[i,:,:] = modifiedIRFFT2(image[i,:,:])
    return irfft




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

# val = torch.complex(torch.tensor([1.]),torch.tensor([2.]))
# print(val)
# def getInverseTransform(ft,dim):
#     n  = ft.shape[0]
#     newT = torch.zeros(dim,dtype=torch.cfloat)
#     k = 0
#     for i,val in enumerate(ft):
#         if i == 0:
#             newT[k] = val
#             k+=1
#             continue
#         if(i%2 == 1):
#             newT[k] += val 
#         else :
#             newT[k] += torch.complex(torch.tensor(0.),val)
#             k+=1
#     return newT

# print(getInverseTransform(torch.tensor([3,2]),2))

# torch.seed()
# n = 32
# A = torch.rand((n,n))
# print(A)

# B = torch.fft.rfft2(A)

# def transform(complexImage : torch.Tensor):
#     complexImage[-14:,0] = 0
#     complexImage[-14:,-1] = 0
#     torch.set_printoptions(precision=2, linewidth=3*100)
#     print(complexImage)
#     values = torch.zeros(32*32)
#     real, imaginary = complexImage.real , complexImage.imag
#     k = 0
#     for val in real.flatten():
#         if val == 0:
#             continue
#         values[k] = val 
#         k+=1
#     print(k)
#     print(sum(imaginary.flatten() == 0))
#     for i,val in enumerate(imaginary.flatten()):
#         if k == 1024:
#             print(i)
#         if val == 0:
#             continue
#         values[k] = val 
#         k+=1
#     print(values)
#     return values.reshape((32,32))

# print(A)
# transform(B)


im2 = Image.open("truck10.png")
im = np.asarray(im2)
im = torch.tensor(im)
im = torch.permute(im,(2,0,1))


im = completeRFFT2(im)
im = torch.permute(im,(1,2,0))

# im += torch.randn_like(im)


im = torch.permute(completeIRFFT2(torch.permute(im, (2, 0,1))),(1,2,0))
im = Image.fromarray(np.uint8(im))
im.show()


# Your code as above

im = Image.open("truck10.png")
im = F.to_tensor(im)
im = F.normalize(im,[0.5],[0.5])

# im*=255

im = completeRFFT2(im)
im = torch.permute(im,(1,2,0)) # [32,32,3]
print(im.shape)

for i in range(3):
    im[:, -8:, -8:]+= torch.randn_like(im[:, -8:, -8:])/255


im = completeIRFFT2(torch.permute(im, (2, 0,1)))
im = im*.5 + .5

# im/=255
im = F.to_pil_image(im)
im.show()



im = Image.open("noise.png")
im = F.to_tensor(im)

im = completeRFFT2(im)

im = 2*(im - 0.5) 

im = completeIRFFT2(im)
im = im*.5 + .5
im = F.to_pil_image(im)
im.show()

