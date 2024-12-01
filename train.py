import torch
from torch import nn
import tqdm
from model.ContextUnet import ContextUnet
from dataloader import spriteDataset, get_transforms
from torch.utils.data import DataLoader

def add_noise(x0, epsilon, t, alpha_bar_t): # equation 68
    return torch.sqrt(alpha_bar_t[t, None, None, None]) * x0 + torch.sqrt(1 - alpha_bar_t[t, None, None, None]) * epsilon 

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
    
    # dataset
    feature_file = "/home/d3ac/Desktop/dataset/sprites_v1/sprites_1788_16x16.npy"
    label_file = "/home/d3ac/Desktop/dataset/sprites_v1/sprite_labels_nc_1788_16x16.npy"
    dataset = spriteDataset(feature_file, label_file, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=7)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr, end_factor=lr*1e-3, total_iters=epochs)
    loss = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        trange = tqdm.tqdm(dataloader)
        mean_loss = 0
        for x, y in trange:
            x = x.to(device)
            # add noise
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_t = add_noise(x, noise, t, alpha_bar_t)
            # predict noise
            pred_noise = model(x_t, t/timesteps)
            # backprop
            loss_val = loss(pred_noise, noise)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            mean_loss += loss_val.item()
        torch.save(model.state_dict(), f"model.pth")
        scheduler.step()
        print(f"Loss: {mean_loss/len(dataloader)}")
        print("-------------------------------------\n")