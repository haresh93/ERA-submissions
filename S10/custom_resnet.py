from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from utils import get_correct_pred_count
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
            nn.Linear(256, 10)
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
