from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from utils import get_correct_pred_count
from tqdm import tqdm

GROUP_SIZE = 8

class Normalization(Enum):
    BATCH_NORMALIZATION = 1
    LAYER_NORMALIZATION = 2
    GROUP_NORMALIZATION = 3

def conv_block(in_channels, out_channels, normalization, kernel_size = 3, pool=False, padding=0, activation=True):
    layers = [
        nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding),
    ]

    # Adding normalization layer
    if normalization is Normalization.BATCH_NORMALIZATION:
        layers.append(nn.BatchNorm2d(out_channels))
    elif normalization is Normalization.LAYER_NORMALIZATION:
        layers.append(nn.GroupNorm(1, out_channels))
    elif normalization is Normalization.GROUP_NORMALIZATION:
        layers.append(nn.GroupNorm(GROUP_SIZE, out_channels))
    
    # Adding activation function
    if activation:
        layers.append(nn.ReLU(inplace=True))

    # Adding Dropout
    layers.append(nn.Dropout(0.05))

    # Checking if Pooling Layer needs to be added
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    
    return nn.Sequential(*layers)

# The Network Architecture
class NetBN(nn.Module):
    def __init__(self):
        super(NetBN, self).__init__()
        self.conv1 = conv_block(3, 8, Normalization.BATCH_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv2 = conv_block(8, 16, Normalization.BATCH_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv3 = conv_block(16, 8, Normalization.BATCH_NORMALIZATION, kernel_size = 1, pool = True, padding=1)

        self.conv4 = conv_block(8, 16, Normalization.BATCH_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv5 = conv_block(16, 32, Normalization.BATCH_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv6 = conv_block(32, 32, Normalization.BATCH_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv7 = conv_block(32, 16, Normalization.BATCH_NORMALIZATION, kernel_size = 1, pool = True, padding=1)

        self.conv8 = conv_block(16, 32, Normalization.BATCH_NORMALIZATION, kernel_size = 3, pool = False)
        self.conv9 = conv_block(32, 32, Normalization.BATCH_NORMALIZATION, kernel_size = 3, pool = False)
        self.conv10 = conv_block(32, 32, Normalization.BATCH_NORMALIZATION, kernel_size = 3, pool = False)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv11 = conv_block(32, 10, None, kernel_size = 1, pool = False, activation = False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        x = self.conv11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class NetLN(nn.Module):
    def __init__(self):
        super(NetLN, self).__init__()
        self.conv1 = conv_block(3, 8, Normalization.LAYER_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv2 = conv_block(8, 16, Normalization.LAYER_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv3 = conv_block(16, 8, Normalization.LAYER_NORMALIZATION, kernel_size = 1, pool = True, padding=1)

        self.conv4 = conv_block(8, 16, Normalization.LAYER_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv5 = conv_block(16, 32, Normalization.LAYER_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv6 = conv_block(32, 32, Normalization.LAYER_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv7 = conv_block(32, 16, Normalization.LAYER_NORMALIZATION, kernel_size = 1, pool = True, padding=1)

        self.conv8 = conv_block(16, 32, Normalization.LAYER_NORMALIZATION, kernel_size = 3, pool = False)
        self.conv9 = conv_block(32, 32, Normalization.LAYER_NORMALIZATION, kernel_size = 3, pool = False)
        self.conv10 = conv_block(32, 32, Normalization.LAYER_NORMALIZATION, kernel_size = 3, pool = False)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv11 = conv_block(32, 10, None, kernel_size = 1, pool = False, activation = False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        x = self.conv11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class NetGN(nn.Module):
    def __init__(self):
        super(NetGN, self).__init__()
        self.conv1 = conv_block(3, 8, Normalization.GROUP_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv2 = conv_block(8, 16, Normalization.GROUP_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv3 = conv_block(16, 8, Normalization.GROUP_NORMALIZATION, kernel_size = 1, pool = True, padding=1)

        self.conv4 = conv_block(8, 16, Normalization.GROUP_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv5 = conv_block(16, 32, Normalization.GROUP_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv6 = conv_block(32, 32, Normalization.GROUP_NORMALIZATION, kernel_size = 3, pool = False, padding=1)
        self.conv7 = conv_block(32, 16, Normalization.GROUP_NORMALIZATION, kernel_size = 1, pool = True, padding=1)

        self.conv8 = conv_block(16, 32, Normalization.GROUP_NORMALIZATION, kernel_size = 3, pool = False)
        self.conv9 = conv_block(32, 32, Normalization.GROUP_NORMALIZATION, kernel_size = 3, pool = False)
        self.conv10 = conv_block(32, 32, Normalization.GROUP_NORMALIZATION, kernel_size = 3, pool = False)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv11 = conv_block(32, 10, None, kernel_size = 1, pool = False, activation = False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        x = self.conv11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
train_losses = []
test_losses = []
train_acc = []
test_acc = []
misclassified_images = []

def model_train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        output = model(data)

        # Calculate loss
        loss = F.nll_loss(output, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1,keepdim=True)
        
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), # Convolution layer 1 - input 28 x 28 x 1 : Output 28 x 28 x 32 : RF 3 x 3
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, padding=1), # Convolution Layer 2 - input 28 x 28 x 32 : Output 28 x 28 x 64 : RF 5 x 5
            nn.ReLU(), 
            nn.MaxPool2d(2, 2) # Max Pooling Layer - input 28 x 28 x 64 : Output 14 x 14 x 64 : RF 6 x 6
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), # Convolution Layer 3 - input 14 x 14 x 64 : Output 14 x 14 x 128 : RF 10 x 10
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), # Convolution Layer 4 - input 14 x 14 x 128 : Output 14 x 14 x 256 : RF 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Max Pooling Layer - input 14 x 14 x 256 : Output 7 x 7 x 256 : RF 16 x 16
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3), # Convolution Layer 5 - input 7 x 7 x 256 : Output 5 x 5 x 512 : RF 24 x 24
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3), # Convolution Layer 6 - input 5 x 5 x 512 : Output 3 x 3 x 1024 : RF 32 x 32
            nn.ReLU(),
            nn.Conv2d(1024, 10, 3), # Convolution Layer 7 - input 3 x 3 x 1024 : Output 1 x 1 x 10 : RF 40 x 40
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3), # Convolution layer 1 - input 28 x 28 x 1 : Output 26 x 26 x 8 : RF 3 x 3
            nn.ReLU(), 
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3), # Convolution Layer 2 - input 26 x 26 x 8 : Output 24 x 24 x 8 : RF 5 x 5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3), # Convolution Layer 3 - input 24 x 24 x 8 : Output 22 x 22 x 8 : RF 7 x 7
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2) # Max Pooling Layer - input 22 x 22 x 8 : Output 11 x 11 x 8 : RF 8 x 8
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3), # Convolution Layer 4 - input 11 x 11 x 8 : Output 9 x 9 x 16 : RF 12 x 12
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3), # Convolution Layer 5 - input 9 x 9 x 16 : Output 7 x 7 x 16 : RF 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 10, 1), # Convolution Layer 7 - input 7 x 7 x 16 : Output 7 x 7 x 10 : RF 20 x 20
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 7), # Convolution Layer 8 - input 5 x 5 x 10 : Output 1 x 1 x 10 : RF 28 x 28
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3), # Convolution layer 1 - input 28 x 28 x 1 : Output 26 x 26 x 8 : RF 3 x 3
            nn.ReLU(), 
            nn.BatchNorm2d(8),
            nn.Dropout(0.05),
            nn.Conv2d(8, 8, 3), # Convolution Layer 2 - input 26 x 26 x 8 : Output 24 x 24 x 16 : RF 5 x 5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.05),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2) # Max Pooling Layer - input 24 x 24 x 8 : Output 12 x 12 x 8 : RF 6 x 6
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3), # Convolution Layer 4 - input 12 x 12 x 8 : Output 10 x 10 x 16 : RF 10 x 10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 16, 3), # Convolution Layer 5 - input 10 x 10 x 16 : Output 8 x 8 x 32 : RF 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2) # Max Pooling Layer - input 24 x 24 x 8 : Output 12 x 12 x 8 : RF 6 x 6
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, 3), # Convolution Layer 5 - input 10 x 10 x 16 : Output 8 x 8 x 32 : RF 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.AdaptiveAvgPool2d(1), # Average Pooling Layer - to reduce the dimensions to 1 - input 6 x 6 x 16 : Output 1 x 1 x 16 : RF 30 x 30
            nn.Conv2d(16, 10, 1), # Convolution Layer 8 - input 1 x 1 x 16 : Output 1 x 1 x 10 : RF 34 x 34
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


def model_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            for i in range(len(pred)):
              if pred[i] != target[i]:
                misclassified_images.append([data[i], pred[i], target[i]])

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    test_acc.append(accuracy)

def draw_graphs():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0,0].plot(train_losses)
    axs[0,0].set_title("Training Loss")
    axs[1,0].plot(train_acc)
    axs[1,0].set_title("Training Accuracy")

    axs[0,1].plot(test_losses)
    axs[0,1].set_title("Testing Loss")
    axs[1,1].plot(train_acc)
    axs[1,1].set_title("Testing Accuracy")

def plot_misclassified_images():
  fig = plt.figure(figsize = (10,10))
  for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        plt.imshow(misclassified_images[i][0].cpu().numpy().squeeze().T)
        
        sub.set_title("Pred={}, Act={}".format(str(misclassified_images[i][1].data.cpu().numpy()),str(misclassified_images[i][2].data.cpu().numpy())))
        
  plt.tight_layout()

  plt.show()

def model_summary(model, input_size):
    summary(model, input_size = input_size)