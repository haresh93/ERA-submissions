from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from utils import get_correct_pred_count
from tqdm import tqdm

def conv_block(in_channels, out_channels, kernel_size = 3, pool=False, padding=0, activation=True):
    layers = [
        nn.Conv2d(in_channels = in_channels[0], out_channels = out_channels[0], kernel_size = kernel_size, padding = padding),
        nn.BatchNorm2d(out_channels[0]),
        nn.ReLU(inplace=True),
        nn.Dropout(0.05),
        nn.Conv2d(in_channels = in_channels[1], out_channels = out_channels[1], kernel_size = kernel_size, padding = padding),
        nn.BatchNorm2d(out_channels[1]),
        nn.ReLU(inplace=True),
        nn.Dropout(0.05),
        nn.Conv2d(in_channels = in_channels[2], out_channels = out_channels[2], kernel_size = kernel_size, stride=2, padding = padding)
    ]
    
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = conv_block((3, 16, 16), (16, 16, 16), kernel_size = 3, pool = False, padding=1)
        self.conv2 = conv_block((16, 32, 32), (32, 32, 32), kernel_size = 3, pool = False, padding=1)
        self.conv3 = conv_block((32, 64, 64), (64, 64, 64), kernel_size = 3, pool = True, padding=1)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv11 = conv_block(64, 10, None, kernel_size = 1, pool = False, activation = False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = self.conv11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)