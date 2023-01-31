import numpy as np 
import torch 
import functorch


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



# a = torch.randn(8)
# print(a)
# fft = modifiedRFFT(a)
# print(modifiedIRFFT(fft))

# b = torch.randn(((8,8)))
# print(b)
# c = modifiedRFFT2(b)
# print(modifiedIRFFT2(c))

# n = 4
# d = torch.stack([torch.randn((4,4)) for i in range(3)])
# print(d.shape)
# print(d)
# e = completeRFFT2(d)
# print(completeIRFFT2(e))

# d = torch.stack([torch.randn((4,5)) for i in range(3)])
# print(d.shape)
# x = torch.permute(d, (2, 0,1))
# print(x.shape)
# print(torch.permute(x, (1,2,0)).shape)


# d = torch.stack([torch.randn((4,4)) for i in range(6)])
# print(d.shape)
# print(d)