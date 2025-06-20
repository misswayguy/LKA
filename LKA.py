import torch
import torch.nn as nn
import torch.nn.functional as F

class LKA(nn.Module): #LKA for Swin or ConvNext
    def __init__(self,
                 in_dim=768,
                 in_channels=8):  
        super().__init__()

        self.project1 = nn.Linear(in_dim, in_channels)  
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(in_channels, in_dim)  

        self.dropout = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels)

    def forward(self, x, hw_shapes=None):
        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.conv2(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return project2
    
class LKA_ViT(nn.Module): #LKA for ViT
    def __init__(self, dim=8):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim=dim, dim=dim, kernel_size=7, stride=1, padding=3, groups=dim)

        self.adapter_down = nn.Linear(768, dim)  
        self.adapter_up = nn.Linear(dim, 768)  
        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down) 
        return x_up
