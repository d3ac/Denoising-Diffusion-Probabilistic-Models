import torch
import torch.nn as nn

class StdUnet(nn.Module):
    def __init__(self):
        super(StdUnet, self).__init__()
        
        # left side
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)  
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)  
        self.relu2_2 = nn.ReLU(inplace=True)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)  
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)  
        self.relu3_2 = nn.ReLU(inplace=True)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)  
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0) 
        self.relu4_2 = nn.ReLU(inplace=True)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # bottom
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0) 
        self.relu5_2 = nn.ReLU(inplace=True)
        self.up_conv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        
        # right side
        self.conv6_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0) 
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0) 
        self.relu6_2 = nn.ReLU(inplace=True)
        self.up_conv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)  
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)  
        self.relu7_2 = nn.ReLU(inplace=True)
        self.up_conv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0) 

        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)  
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)  
        self.relu8_2 = nn.ReLU(inplace=True)
        self.up_conv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0) 

        self.conv9_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0) 
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)  
        self.relu9_2 = nn.ReLU(inplace=True)
        self.conv_10 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)

    def crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.relu1_1(x1)
        x2 = self.conv1_2(x1)
        x2 = self.relu1_2(x2)
        down1 = self.maxpool_1(x2)

        x3 = self.conv2_1(down1)
        x3 = self.relu2_1(x3)
        x4 = self.conv2_2(x3)
        x4 = self.relu2_2(x4)
        down2 = self.maxpool_2(x4)

        x5 = self.conv3_1(down2)
        x5 = self.relu3_1(x5)
        x6 = self.conv3_2(x5)
        x6 = self.relu3_2(x6)
        down3 = self.maxpool_3(x6)

        x7 = self.conv4_1(down3)
        x7 = self.relu4_1(x7)
        x8 = self.conv4_2(x7)
        x8 = self.relu4_2(x8)
        down4 = self.maxpool_4(x8)

        x9 = self.conv5_1(down4)
        x9 = self.relu5_1(x9)
        x10 = self.conv5_2(x9)
        x10 = self.relu5_2(x10)

        up1 = self.up_conv_1(x10)  
        crop1 = self.crop_tensor(x8, up1)
        up_1 = torch.cat([crop1, up1], dim=1)

        y1 = self.conv6_1(up_1)
        y1 = self.relu6_1(y1)
        y2 = self.conv6_2(y1)
        y2 = self.relu6_2(y2)

        up2 = self.up_conv_2(y2)
        crop2 = self.crop_tensor(x6, up2)
        up_2 = torch.cat([crop2, up2], dim=1)

        y3 = self.conv7_1(up_2)
        y3 = self.relu7_1(y3)
        y4 = self.conv7_2(y3)
        y4 = self.relu7_2(y4)

        up3 = self.up_conv_3(y4)
        crop3 = self.crop_tensor(x4, up3)
        up_3 = torch.cat([crop3, up3], dim=1)

        y5 = self.conv8_1(up_3)
        y5 = self.relu8_1(y5)
        y6 = self.conv8_2(y5)
        y6 = self.relu8_2(y6)

        up4 = self.up_conv_4(y6)
        crop4 = self.crop_tensor(x2, up4)
        up_4 = torch.cat([crop4, up4], dim=1)

        y7 = self.conv9_1(up_4)
        y7 = self.relu9_1(y7)
        y8 = self.conv9_2(y7)
        y8 = self.relu9_2(y8)

        out = self.conv_10(y8)
        return out