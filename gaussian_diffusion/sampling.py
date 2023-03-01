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
import argparse


def run(mode, fft, num_samples, checkpoint_path, save_path, use_autoencoder, n_layers, hidden_channels, hidden_dim, n_modes, data_path, verbose, sample_path, lr, wd, num_steps, epochs):

    c = [1/2,1/6,1/3]
    means = [[0.5,0.5,0.5],[-15,-20,0], [30,10,20]]
    variances = [[[1,0,0],[0,1,0],[0,0,1]], [[5,1,-2],[1,1,3],[-2,3,5]] , [[1, 2,3],[2,5,6],[3,6,1]]]

    sde = sde_lib.LinearSDE(beta=20)
    if data_path is None:
        samplesBeforeFFT = torch.tensor(generateSamples.get_samples_from_mixed_gaussian(c,means,variances,num_samples))
    else:
        samplesBeforeFFT = torch.load(data_path)
    samples = samplesBeforeFFT
    print("Sample shape: ", samples.shape)

    if fft:
        # Move to frequency space
        samples = torch.fft.rfft(samplesBeforeFFT,norm="forward")
        # Now we are in 6D instead of 3D
        samples = torch.cat((samples.real,samples.imag),dim=1) 
        print(samples.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    score_function = model.FNOScore(n_layers, hidden_channels, hidden_dim, n_modes, verbose=verbose)#model.Score(dim)

    if checkpoint_path != "" and os.path.exists(checkpoint_path):
        print("Loading ckpt...")
        checkpoint = torch.load(checkpoint_path)
        score_function.load_state_dict(checkpoint)

    score_function = score_function.to(device=device)
    samples = samples.to(device=device)[:, None, :]
    shape = list(samples.shape[1:])

    if mode == "train":
        print("Learning score...")
        errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=num_steps, file_to_save=save_path, device=device, lr=lr, wd=wd, epochs=epochs)
    else:
        print("Generating samples...")
        generatedSamples = sde.generate_samples_reverse(score_network=score_function, shape=shape, nsamples=1000)[0]
        print(generatedSamples.shape)
        torch.save(generatedSamples, sample_path) if sample_path != "" else None
        exit()
        if fft:
            real, imaginary = torch.chunk(generatedSamples,2,dim=1)

            # Back to 2D in frequency space
            complexGenerated = torch.complex(real,imaginary)
            # Back to original space (hopefully)
            generatedSamples = torch.fft.irfft(complexGenerated,n=3, norm="forward")
            print(generatedSamples.shape)

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
        plt.savefig("results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--fft", action="store_true")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--num_samples", default=1000)
    parser.add_argument("--num_steps", default=150001)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--save_path", default="")
    parser.add_argument("--use_autoencoder", action="store_true")
    parser.add_argument("--n_layers", default=4)
    parser.add_argument("--hidden_channels", default=1)
    parser.add_argument("--hidden_dim", default=64)
    parser.add_argument("--n_modes", default=32)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sample_path", default="")
    parser.add_argument("--lr", default=5e-5)
    parser.add_argument("--wd", default=0)
    parser.add_argument("--epochs", default=1000)
    args = parser.parse_args()

    run(**vars(args))

    