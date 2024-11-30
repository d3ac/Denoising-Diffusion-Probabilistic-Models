import torch
from torch import nn
import tqdm
from model.ContextUnet import ContextUnet
from dataloader import spriteDataset, get_transforms
from torch.utils.data import DataLoader

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

    # construct noise
    beta_t = (beta[1] - beta[0]) * torch.linspace(0, 1, timesteps + 1, device=torch.device("cuda:0")) + beta[0]
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.cumsum(alpha_t.log(), dim=0).exp()

    # model
    model = ContextUnet(in_channels=3, hidden_dim=hidden_dim, context_dim=context_dim, picture_shape=picture_shape).to(torch.device("cuda:0"))
    
    # dataset
    feature_file = "/home/d3ac/Desktop/dataset/sprites_v1/sprites_1788_16x16.npy"
    label_file = "/home/d3ac/Desktop/dataset/sprites_v1/sprite_labels_nc_1788_16x16.npy"
    dataset = spriteDataset(feature_file, label_file, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    