from cleanfid import fid
from datasets import load_dataset
from PIL import Image
import numpy as np
import pywt
import matplotlib.pyplot as plt

# This is for computing fid using the clean fid library
# names = ["local-unet-cifar10-32", "dual-fno-cifar10-32", "fno-cifar10-32",]
# 
# for name in names:
#     path = f"../datasets/{name}"
#     dataset = load_dataset(f"Dahoas/{name}",cache_dir=path,split="train")
#     for i, image in enumerate(dataset):
#         image = np.array(image['images'])
#         image*=256
#         image//=1
#         image = image.astype('uint8')
#         image = np.transpose(image, (1, 2, 0))
#         img = Image.fromarray(image,'RGB')
#         img.save(f"../images/{name}/{i}.png")

#     score = fid.compute_fid('../images/{name}/', dataset_name="cifar10",dataset_res=32,dataset_split="test")

#     print(f"FID SCORE for {name : 20} : {score}")


names = ["local-unet-cifar10-32", "dual-fno-cifar10-32", "fno-cifar10-32",]

dataset = load_dataset("Dahoas/unet-lsun-256")

dataset = dataset['train']
k = 256

def upscaleFourier(image,zerosToPad):
    n = image.shape[0]
    nEnd = n  + 2*zerosToPad
    ffImg64 = np.zeros((nEnd,nEnd,3),dtype='complex128')
    for j in range(3):
        a = np.fft.fft2(image[:,:,j],norm="backward")
        a = np.fft.fftshift(a)

        # b = np.random.randn(nEnd,nEnd).astype('complex128')
        b = np.zeros((nEnd,nEnd)).astype('complex128')

        b[zerosToPad:zerosToPad+n,zerosToPad:zerosToPad+n] = a

        b = np.fft.ifftshift(b)
        b = np.fft.ifft2(b,norm="backward")
        b*=(nEnd/n)**2

        ffImg64[:,:,j] = b
    return ffImg64.real

def upscaleFourier1D(image,zerosToPad):
    n = image.shape[0]
    nEnd = n  + 2*zerosToPad
    ffImg64 = np.zeros(nEnd ,dtype='complex128')

    a = np.fft.fft(image ,norm="backward")
    a = np.fft.fftshift(a)

    # b = np.random.randn(nEnd,nEnd).astype('complex128')
    b = np.zeros(nEnd).astype('complex128')

    b[zerosToPad:zerosToPad+n] = a

    b = np.fft.ifftshift(b)
    b = np.fft.ifft(b,norm="backward")
    b*=(nEnd/n)

    ffImg64 = b
    return ffImg64.real


# def upscaleWavelets(image,nStart,nEnd):
#     ffImg64 = np.zeros((nEnd,nEnd,3),dtype='complex128')
#     for j in range(3):
#         coeffs = pywt.dwt2(image[:,:,j],norm="ortho")
#         # b = np.random.randn(nEnd,nEnd).astype('complex128')
#         b = np.zeros((nEnd,nEnd)).astype(a.dtype)

#         b[:nStart,:nStart] = a
#         b = np.fft.ifft2(b,norm="ortho")
#         ffImg64[:,:,j] = b
#     return ffImg64.real

def saveImage(im ,path):
    image = 256*im.copy()
    image//=1


    image = image.astype('uint8')
    img = Image.fromarray(image,'RGB')
    img.save(path)


def simpleExample():
    L = 5
    a = np.linspace(-L,L,100)
    c = np.linspace(-L,L,200)

    b = 1/(2+np.sin(a))

    plt.plot(a,b)
    plt.plot(c,upscaleFourier1D(b,50))
    plt.show()

simpleExample()

for i, image in enumerate(dataset):
    if i>20: 
        break
    image = np.array(image['images'])
    image = np.transpose(image, (1, 2, 0))

    transition = [256,512]


    currImage = image.copy()
    for k in range(len(transition)-1):
        saveImage(currImage,f"../imagesHigherQuality/church/transition{k}.png")
        currImage = upscaleFourier(currImage,(transition[k+1]-transition[k])//2)

    saveImage(image,f"../images/church/{i}.png")
    saveImage(currImage,f"../imagesHigherQuality/church/{i}.png")