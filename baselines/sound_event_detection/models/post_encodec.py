import torch
import torch.nn as nn
import torch.nn.functional as F
from encodec import modules as m

import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock2D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class PostEncodecEnhancer2D(nn.Module):
    def __init__(self, in_channels, base_channels=24, num_blocks=6):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock2D(base_channels) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv2d(base_channels, in_channels, kernel_size=1)  # project back to in_channels (typically 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.out_proj(x)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return x


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        )

    def forward(self, x):
        return x + self.block(x)


# class PostEncodecEnhancer2D(nn.Module):
#     def __init__(self, in_channels=1, base_channels=32, num_blocks=4, input_dim=128):
#         """
#         Outputs shape: [B, T, input_dim]
#         """
#         super().__init__()
#         self.input_dim = input_dim
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.res_blocks = nn.Sequential(
#             *[ResidualBlock2D(base_channels) for _ in range(num_blocks)]
#         )

#         self.out_proj = nn.Conv2d(base_channels, input_dim, kernel_size=1)  # project to mel dim

#     def forward(self, x):
#         """
#         Args:
#             x: [B, 1, F, T] (mel-spectrogram)
#         Returns:
#             [B, T, input_dim] for AENet
#         """
#         x = self.conv1(x)        # [B, C, F, T]
#         x = self.res_blocks(x)   # [B, C, F, T]
#         x = self.out_proj(x)     # [B, input_dim, F, T]

#         x = x.mean(dim=2)        # average over F → [B, input_dim, T]
#         x = x.transpose(1, 2)    # → [B, T, input_dim]
#         return x

class PostEncodecEnhancer1D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_blocks=4):
        """
        Input:  [B, 1, T]
        Output: [B, 1, T] (residual-enhanced waveform)
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(base_channels, dilation=3) for _ in range(num_blocks)]
        )

        self.out_proj = nn.Conv1d(base_channels, in_channels, kernel_size=1)  # project back to 1 channel

    def forward(self, x):
        """
        Args:
            x: [B, 1, T]
        Returns:
            [B, 1, T]
        """
        x = self.conv1(x)         # [B, C, T]
        x = self.res_blocks(x)    # [B, C, T]
        x = self.out_proj(x)      # [B, 1, T]
        return x
    
class PostEncodecEnhancerSConv(nn.Module): 
    def __init__(self, n_filters, channels, last_kernel_size, norm, norm_params, causal, pad_mode):
        super().__init__()
        
        self.extra_layer = nn.Sequential(
            m.SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode),
            nn.Tanh()  # or any final activation
        )

    def forward(self, z):
        x = self.extra_layer(z)
        return x