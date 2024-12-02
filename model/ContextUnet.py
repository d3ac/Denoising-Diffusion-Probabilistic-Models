import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_res=False):
        super().__init__()
        self.use_res = use_res
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        if self.use_res:
            if hasattr(self, 'shortcut'):
                return self.shortcut(x) + self.conv2(self.conv1(x))
            else:
                return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )
    
    def forward(self, x, skip):
        return self.model(torch.cat([x, skip], dim=1))

class Embeding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, x):
        return self.model(x.view(-1, self.input_dim))

class ContextUnet(nn.Module):
    def __init__(self, in_channels, hidden_dim, context_dim, picture_shape):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.picture_shape = picture_shape

        # Encoder
        self.init_conv_down = ResidualBlock(in_channels, hidden_dim, use_res=True)
        self.down1 = UnetDown(hidden_dim, hidden_dim)
        self.down2 = UnetDown(hidden_dim, hidden_dim*2)

        # embedding
        self.time_embedding1 = Embeding(1, hidden_dim*2)
        self.time_embedding2 = Embeding(1, hidden_dim)
        self.context_embedding1 = Embeding(context_dim, hidden_dim*2)
        self.context_embedding2 = Embeding(context_dim, hidden_dim)

        # Decoder
        self.init_conv_up = nn.Sequential(
            nn.AvgPool2d((4)),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim*2, self.picture_shape//4, self.picture_shape//4),
            nn.GroupNorm(8, hidden_dim*2),
            nn.ReLU(),
            nn.AvgPool2d((2)),
        )
        self.up1 = UnetUp(hidden_dim*4, hidden_dim)
        self.up2 = UnetUp(hidden_dim*2, hidden_dim)

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, 3, 1, 1)
        )
    
    def forward(self, x, t, context=None):
        # Encoder
        x = self.init_conv_down(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        # embedding
        if context is None:
            context = torch.zeros(x.shape[0], self.context_dim).to(x)
        context_embedding1 = self.context_embedding1(context).view(-1, self.hidden_dim * 2, 1, 1)
        context_embedding2 = self.context_embedding2(context).view(-1, self.hidden_dim, 1, 1)
        time_embedding1 = self.time_embedding1(t).view(-1, self.hidden_dim * 2, 1, 1)
        time_embedding2 = self.time_embedding2(t).view(-1, self.hidden_dim, 1, 1)
        # Decoder
        up1 = self.init_conv_up(down2)
        up2 = self.up1(context_embedding1 * up1 + time_embedding1, down2)
        up3 = self.up2(context_embedding2 * up2 + time_embedding2, down1)
        # Output
        return self.output(torch.cat((up3, x), dim=1))