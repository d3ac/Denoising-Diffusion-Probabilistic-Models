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

def denoise(x, t, pred_noise, beta_t, alpha_t, alpha_bar_t, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = torch.sqrt(beta_t[t]) * z
    mean = (x - pred_noise * ((1 - alpha_t[t]) / (1 - alpha_bar_t[t]).sqrt())) / alpha_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample(n_sample, save_rate=20):
    samples = torch.randn(n_sample, 3, picture_shape, picture_shape, device=device)
    intermediate = []
    for i in range(timesteps, 0, -1):
        t = torch.tensor([i/timesteps])[:, None, None, None].to(device)
        z = torch.randn_like(samples) if i > 1 else 0
        epsilons = model(samples, t)
        samples = denoise(samples, i, epsilons, beta_t, alpha_t, alpha_bar_t, z)
        if i % save_rate == 0 or timesteps == i or i <= 8:
            intermediate.append(samples.detach().cpu().numpy())
    intermediate = np.stack(intermediate)
    return samples, intermediate

if __name__ == '__main__':
    # parameters
    timesteps = 500
    beta = [1e-4, 0.02]
    hidden_dim = 64
    context_dim = 5
    picture_shape = 16
    batch_size = 128
    epochs = 100
    lr = 1e-3
    device = torch.device("cuda:0")

    # construct noise
    beta_t = (beta[1] - beta[0]) * torch.linspace(0, 1, timesteps + 1, device=device) + beta[0]
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.cumsum(alpha_t.log(), dim=0).exp()

    # model
    model = ContextUnet(in_channels=3, hidden_dim=hidden_dim, context_dim=context_dim, picture_shape=picture_shape).to(device)
    model.load_state_dict(torch.load("model.pth"))

    # dataset
    feature_file = "/home/d3ac/Desktop/dataset/sprites_v1/sprites_1788_16x16.npy"
    label_file = "/home/d3ac/Desktop/dataset/sprites_v1/sprite_labels_nc_1788_16x16.npy"
    dataset = spriteDataset(feature_file, label_file, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=7)

    # test
    model.eval()
    plt.clf()
    samples, intermediate_ddpm = sample(32)
    animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, 'img', "ani_run", None, save=True)