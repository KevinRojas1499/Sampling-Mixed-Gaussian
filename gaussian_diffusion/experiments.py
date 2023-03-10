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

