from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from utils import get_correct_pred_count
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=(3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, groups = 32, kernel_size = (3,3), padding = 1, bias = False),
            nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size = (1,1), padding=0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5)
        )

        # Transition Block 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = (3,3), padding=2, stride=2, dilation=2, bias = False)
        )

        # Convolution Block 2
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, groups = 64, kernel_size = (3,3), padding=1, bias = False),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (1,1), padding = 0,bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, groups = 128, kernel_size = (3,3), padding = 1, bias = False),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (1,1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5)
        )

        # Transition Block 2
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size=(3,3), padding = 2, dilation = 2, stride = 2, bias = False)
        )

        # Convolution Block 3
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, groups = 64, kernel_size = (3,3), padding = 1, bias = False),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, groups = 64, kernel_size = (3,3), padding = 1, bias = False),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5)
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, groups = 64, kernel_size = (3,3), padding = 1, bias = False),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 10, kernel_size = (1,1), padding = 0, bias = False),
        )

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
        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def model_summary(model, input_size):
    summary(model, input_size = input_size)


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
