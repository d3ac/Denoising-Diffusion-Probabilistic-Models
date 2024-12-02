import torch
from torch import nn
import torchvision.transforms as transforms
import tqdm
from model.ContextUnet import ContextUnet
import torchvision.datasets as datasets
from dataloader import spriteDataset, get_transforms
from torch.utils.data import DataLoader
import pandas as pd

def add_noise(x0, epsilon, t, alpha_bar_t): # equation 68
    return torch.sqrt(alpha_bar_t[t, None, None, None]) * x0 + torch.sqrt(1 - alpha_bar_t[t, None, None, None]) * epsilon 

if __name__ == '__main__':
    # parameters
    timesteps = 500
    beta = [1e-4, 0.02]
    hidden_dim = 128
    context_dim = 10
    picture_shape = 32
    batch_size = 128
    epochs = 32
    lr = 1e-3
    device = torch.device("cuda:0")
    # dataset = "sprites_v1"
    # dataset = "CIFA10"
    dataset = "MNIST"

    # construct noise
    beta_t = (beta[1] - beta[0]) * torch.linspace(0, 1, timesteps + 1, device=device) + beta[0]
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.cumsum(alpha_t.log(), dim=0).exp()
    alpha_bar_t[0]= 1

    # model
    model = ContextUnet(in_channels=(1 if dataset=="MNIST" else 3), hidden_dim=hidden_dim, context_dim=context_dim, picture_shape=picture_shape).to(device)
    model.load_state_dict(torch.load(f"results/model_{dataset}.pth", weights_only=True))