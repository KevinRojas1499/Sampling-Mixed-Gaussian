import sklearn
import sklearn.datasets
from sklearn.datasets import load_digits


from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import torch
import sde_lib
from models.model import SimpleScore
import ot
import wandb
import generateSamples
import argparse
import time


def beta(t):
    return 100*t

def drift(x,t):
    return -beta(t)*x/2

def diffusion(t):
    return torch.sqrt(beta(t))


def loss_function(sde, data, score, eps = .0001):
    random_t = torch.rand((data.shape[0], 1), device=data.device) * (sde.T - eps) + eps
    z = torch.randn_like(data)
    mean = sde.marginal_prob_mean(data, random_t)
    std = sde.marginal_prob_var(random_t).view(z.shape[0], 1, 1, 1)**.5
    perturbed_data = mean+z*(std)
    m_output = score(perturbed_data, random_t)
    loss = torch.mean(std*(std*m_output+z)**2)
    return loss


def train(sde, score, dataloader, epochs):
    # optimizer = torch.optim.SGD(score_model.parameters(),lr=0.01)
    optimizer = torch.optim.Adam(score.parameters(), lr=1e-4, weight_decay=1e-3)
    errors = []
    t0 = time.time()
    iter_cnt = 0
    for e in range(epochs):
        for batch in dataloader:
            batch = batch.to(score.device)
            optimizer.zero_grad()
            loss = loss_function(sde, batch, score)
            loss.backward()
            optimizer.step()

            if(iter_cnt%1000 == 0):
                print(f"Step number {iter_cnt} ({time.time() - t0}s) \nError : {loss}")
                errors.append(loss)
                #torch.save(score.state_dict(), 'conv_mnist.pth')
            iter_cnt += 1
    torch.save(score.state_dict(), 'conv_mnist.pth')
    return errors


def generate_samples(score_network: torch.nn.Module, noise_shape = (10, 1, 8, 8)) -> torch.Tensor:
    x_t = torch.randn(noise_shape).to(score_network.device)
    time_pts = torch.linspace(1, 0, 1000).to(score_network.device)
    for i in tqdm(range(len(time_pts) - 1)):
        t = time_pts[i]
        dt = time_pts[i + 1] - t
        score = score_network(x_t, t.expand(x_t.shape[0], 1)).detach()
        tot_drift = drift(x_t, t) - diffusion(t)**2 * score
        tot_diffusion = diffusion(t)

        # euler-maruyama step
        x_t = x_t + tot_drift * dt + tot_diffusion * torch.randn_like(x_t) * (torch.abs(dt) ** 0.5)
    return x_t.cpu()


def sample(score):
    generatedSamples = generate_samples(score_network=score)
    sample = generatedSamples[0].view(8, 8)
    plt.imshow(sample, cmap="gray")
    plt.savefig("num.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    mnist = torch.tensor(load_digits()["data"], dtype=torch.float32).view(-1, 1, 8, 8)
    batch_size = 64
    epochs = 10000
    dataloader = torch.utils.data.DataLoader(mnist, batch_size, shuffle=True)

    c = [1/2,1/6,1/3]
    means = [[0.5,0.5],[-15,15], [8,8]]
    variances = [[[1,0],[0,1]], [[5, -2],[-2,5]] , [[1, 2],[2,1]]]

    sde = sde_lib.SDE(100, 1, beta=beta(1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    score = SimpleScore(1)

    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        score.load_state_dict(checkpoint)
    score.to(device=device)
    score.device = device

    if args.mode == "train":
        errors = train(sde=sde, score=score, dataloader=dataloader, epochs=epochs)
        plt.plot(np.linspace(1,len(errors),len(errors)),errors)
        plt.show()
        plt.savefig("losses.png")
    elif args.mode == "eval":
        sample(score)
    else:
        raise ValueError(args.mode)
