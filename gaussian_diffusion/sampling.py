import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sde_lib
import model
import training
import argparse
import json


def run(args):
    sde = sde_lib.LinearSDE(beta=20)

    samples = torch.load(args.data_path)
    print("Sample shape: ", samples.shape)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if args.model_type == "fno":
        score_function = model.FNOScore(args.n_layers, args.hidden_channels, args.hidden_dim, args.n_modes, verbose=args.verbose)#model.Score(dim)
    elif args.model_type == "simple":
        score_function = model.SimpleScore(samples.shape[-1])
    else:
        raise ValueError("{} model type not supported".format(args.model_type))

    if args.checkpoint_path != "" and os.path.exists(args.checkpoint_path):
        print("Loading ckpt...")
        checkpoint = torch.load(args.checkpoint_path)
        score_function.load_state_dict(checkpoint)

    score_function = score_function.to(device=device)
    samples = samples.to(device=device)[:, None, :]
    shape = list(samples.shape[1:])

    if args.mode == "train":
        print("Printing input args...")
        print(json.dumps(vars(args), indent=4))
        print("Learning score...")
        errors = training.train(sde=sde, score_model=score_function,data=samples, number_of_steps=args.num_steps, file_to_save=args.save_path, device=device, lr=args.lr, wd=args.wd, epochs=args.epochs, batch_size=args.batch_size)
    else:
        print("Generating samples...")
        generatedSamples = sde.generate_samples_reverse(score_network=score_function, shape=shape, nsamples=1000)[0]
        print(generatedSamples.shape)
        torch.save(generatedSamples, args.sample_path) if args.sample_path != "" else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--fft", action="store_true")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--num_steps", default=150001, type=int)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--save_path", default="")
    parser.add_argument("--use_autoencoder", action="store_true")
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--hidden_channels", default=1)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--n_modes", default=32, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sample_path", default="")
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--wd", default=0, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--model_type", default="simple")
    parser.add_argument("--batch_size", default=1024, type=int)
    args = parser.parse_args()

    run(args)

    