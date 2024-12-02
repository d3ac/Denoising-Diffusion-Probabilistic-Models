import torch
from torch import nn
import torchvision.transforms as transforms
import tqdm
from model.ContextUnet import ContextUnet
import torchvision.datasets as datasets
from dataloader import spriteDataset, get_transforms
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import os

def add_noise(x0, epsilon, t, alpha_bar_t): # equation 68
    return torch.sqrt(alpha_bar_t[t, None, None, None]) * x0 + torch.sqrt(1 - alpha_bar_t[t, None, None, None]) * epsilon 

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
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists("results"):
        os.makedirs("results")

    # construct noise
    beta_t = (beta[1] - beta[0]) * torch.linspace(0, 1, timesteps + 1, device=device) + beta[0]
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.cumsum(alpha_t.log(), dim=0).exp()
    alpha_bar_t[0]= 1

    # model
    model = ContextUnet(in_channels=(1 if dataset=="MNIST" else 3), hidden_dim=hidden_dim, context_dim=context_dim, picture_shape=picture_shape).to(device)
    
    # dataset
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset == "sprites_v1":
        feature_file = "/home/d3ac/Desktop/dataset/sprites_v1/sprites_1788_16x16.npy"
        label_file = "/home/d3ac/Desktop/dataset/sprites_v1/sprite_labels_nc_1788_16x16.npy"
        dataloader = DataLoader(spriteDataset(feature_file, label_file, transform=get_transforms()), batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
    elif dataset == "CIFA10":
        dataloader = DataLoader(datasets.CIFAR10(root=path, train=True, download=True, transform=trans), batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
    elif dataset == "MNIST":
        dataloader = DataLoader(datasets.MNIST(root=path, train=True, download=True, transform=trans), batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
    
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()

    model.train()
    trange = tqdm.tqdm(range(epochs))
    mean_loss_list = []
    for epoch in trange:
        optimizer.param_groups[0]['lr'] = lr * (1 - epoch/epochs)
        mean_loss = 0
        for x, y in dataloader:
            x = x.to(device)
            # add noise
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_t = add_noise(x, noise, t, alpha_bar_t)
            # predict noise
            pred_noise = model(x_t, t/timesteps)
            # backprop
            l = loss(pred_noise, noise)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            mean_loss += l.item()
        torch.save(model.state_dict(), f"results/model_{dataset}.pth")
        mean_loss_list.append(mean_loss/len(dataloader))
        pd.DataFrame(mean_loss_list).to_csv(f"results/loss_{dataset}.csv", index=False, header=None)
        trange.set_postfix({"Loss": mean_loss/len(dataloader)})