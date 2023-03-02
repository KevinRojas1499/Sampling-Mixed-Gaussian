import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sde_lib
import model
import training
import argparse


def run(mode, fft, num_samples, checkpoint_path, save_path, use_autoencoder, n_layers, hidden_channels, hidden_dim, n_modes, data_path, verbose, sample_path, lr, wd, num_steps, epochs):
    sde = sde_lib.LinearSDE(beta=20)

    samples = torch.load(data_path)
    print("Sample shape: ", samples.shape)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    score_function = model.FNOScore(n_layers, hidden_channels, hidden_dim, n_modes, verbose=verbose)#model.Score(dim)

    score_function = model.SimpleScore(64)
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

    