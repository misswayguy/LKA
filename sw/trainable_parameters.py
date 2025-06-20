import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model
class Mona(nn.Module):
    def __init__(self,
                 in_dim=768,
                 in_channels=8):  # 确保 in_channels 和 project1 输出的通道数一致
        super().__init__()

        self.project1 = nn.Linear(in_dim, in_channels)  # 输出通道数是 9
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(in_channels, in_dim)  # 输入通道数是 9

        self.dropout = nn.Dropout(p=0.1)

        # Conv2d 的输入通道数和输出通道数都应为 9
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels)

    def forward(self, x, hw_shapes=None):
        project1 = self.nonlinear(self.project1(x))

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.conv2(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return project2

# Instantiate the model
model = Mona()

# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')
