import torch
from torch import nn
import tqdm
from model.ContextUnet import ContextUnet
from dataloader import spriteDataset, get_transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from utils_draw import plot_sample
import argparse
import os

def denoise(x_t, t, pred_noise, beta_t, alpha_t, alpha_bar_t, z=None):
    if z is None:
        z = torch.randn_like(x_t)
    noise = torch.sqrt(beta_t[t]) * z # this is a approximation in the paper 3.2
    mean = (x_t - pred_noise * ((1 - alpha_t[t]) / (1 - alpha_bar_t[t]).sqrt())) / alpha_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample(n_sample, save_rate=20):
    samples = torch.randn(n_sample, (1 if dataset=="MNIST" else 3), picture_shape, picture_shape, device=device)
    intermediate = []
    for i in range(timesteps, 0, -1):
        t = torch.tensor([i/timesteps])[:, None, None, None].to(device)
        z = torch.randn_like(samples) if i > 1 else 0
        epsilons = model(samples, t)
        samples = denoise(samples, i, epsilons, beta_t, alpha_t, alpha_bar_t, z)
        if i % save_rate == 0 or timesteps == i or i <= 8 * save_rate:
            intermediate.append(samples.detach().cpu().numpy())
    intermediate = np.stack(intermediate)
    return samples, intermediate

if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--beta", type=float, nargs=2, default=[1e-4, 0.02])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--context_dim", type=int, default=10)
    parser.add_argument("--picture_shape", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["sprites_v1", "CIFA10", "MNIST"])
    parser.add_argument("--path", type=str, default="/home/d3ac/Desktop/dataset")
    parser.add_argument("--save_rate", type=int, default=5)
    parser.add_argument("--n_sample", type=int, default=16)
    parser.add_argument("--row", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="results")

    args = parser.parse_args()
    timesteps = args.timesteps
    beta = args.beta
    hidden_dim = args.hidden_dim
    context_dim = args.context_dim
    picture_shape = args.picture_shape
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = torch.device(args.device)
    dataset = args.dataset
    path = args.path
    save_rate = args.save_rate
    n_sample = args.n_sample
    row = args.row
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # construct noise
    beta_t = (beta[1] - beta[0]) * torch.linspace(0, 1, timesteps + 1, device=device) + beta[0]
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.cumsum(alpha_t.log(), dim=0).exp()
    alpha_bar_t[0]= 1

    # model
    model = ContextUnet(in_channels=(1 if dataset=="MNIST" else 3), hidden_dim=hidden_dim, context_dim=context_dim, picture_shape=picture_shape).to(device)
    model.load_state_dict(torch.load(f"results/model_{dataset}.pth", weights_only=True))

    # test
    model.eval()
    n_display = 16
    samples, intermediate_ddpm = sample(n_display, save_rate=save_rate)
    samples = (samples - samples.min()) / (samples.max() - samples.min()) # clip to [0, 1]
    intermediate_ddpm = (intermediate_ddpm - intermediate_ddpm.min()) / (intermediate_ddpm.max() - intermediate_ddpm.min()) # clip to [0, 1]
    
    # draw
    fig, axes = plt.subplots(row, n_display//row, figsize=(n_sample, 4))
    for i in range(row):
        for j in range(n_display//row):
            axes[i, j].imshow(samples[i*n_display//row+j].permute(1, 2, 0).cpu().numpy(), cmap='gray')
            axes[i, j].axis('off')
    plt.show()
    
    np.save(f"results/intermediate_ddpm_{dataset}.npy", intermediate_ddpm)