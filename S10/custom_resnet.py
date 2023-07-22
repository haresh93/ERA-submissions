from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from utils import get_correct_pred_count, get_lr
from tqdm import tqdm

def res_block(channels, kernel_size = 3, padding=1):
    layers = [
        nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = kernel_size, padding = padding),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = kernel_size, padding = padding),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    ]
    
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.prep_conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.res_block1 = nn.Sequential(
            res_block(128)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.res_block2 = nn.Sequential(
            res_block(512)
        )

        self.classification_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size = 4, stride = 2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.prep_conv(x)
        x = self.layer1(x)
        x = x + self.res_block1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x + self.res_block2(x)
        x = self.classification_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def model_summary(model, input_size):
    summary(model, input_size = input_size)
