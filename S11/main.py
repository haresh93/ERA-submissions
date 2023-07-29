#Importing required modules 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
from enum import Enum
from torchsummary import summary
from torchvision.utils import save_image

from models.resnet import ResNet18,ResNet34
from dataset import Cifar10SearchDataset

train_losses = []
lrs=[]
test_losses = []
train_acc = []
test_acc = []
misclassified_images = []

MISCLASSIFIED_IMAGES_DIR = "~/misclassifed_dir"

def create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

class DataModels(Enum):
    RESNET18='RESNET18'
    RESNET34='RESNET34'

def get_model_summary(model_name):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if(model_name == DataModels.RESNET18.value):
        model= ResNet18().to(device)
    elif(model_name == DataModels.RESNET34.value):
        model = ResNet34().to(device)
    summary(model, input_size = (3, 32, 32))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#Building the model
def run_model(model_name, epochs):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if(model_name == DataModels.RESNET34.value):
        model=ResNet34().to(device)
    elif(model_name == DataModels.RESNET18.value):
        model= ResNet18().to(device)

    train_loader, test_loader = getDataLoaders()

    # get_model_summary(model_name, device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay = 0.0001)
    criterion = nn.CrossEntropyLoss()

    scheduler = OneCycleLR(
        optimizer,
        max_lr = 0.018,
        steps_per_epoch=len(train_loader),
        epochs = epochs,
        pct_start = 5/epochs,
        div_factor=100,
        three_phase = False,
        final_div_factor=1000,
        anneal_strategy='linear'
    )



    #Test and Train the data model
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        model_train(model, device, train_loader, optimizer, scheduler, criterion, epoch)
        model_test(model, device, test_loader, criterion)
    
    return misclassified_images
    #save_misclassified_images()
    
    
def model_train(model, device, train_loader, optimizer, scheduler, criterion, epoch):
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
        loss = criterion(output, target)
        lrs.append(get_lr(optimizer))
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = output.argmax(dim=1,keepdim=True)
        
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))


def model_test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
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

def save_misclassified_images():
    create_output_folder(MISCLASSIFIED_IMAGES_DIR)
    for i in range(misclassified_images):
        img_path = os.path.join(MISCLASSIFIED_IMAGES_DIR, f"misclassified_{i}.jpg")
        save_image(misclassified_images[i][0].cpu().numpy().squeeze())

def getDataLoaders():
   # Train and Test Transforms
    train_transforms = A.Compose([
            A.PadIfNeeded(min_height=40, min_width=40, value=(0.4914, 0.4822, 0.4465), p=1), # Padding with a border of 4 pixels
            A.RandomCrop(width=32, height=32),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes = 1, max_height=8, max_width=8,
                        min_holes = 1, min_height=8, min_width=8, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None),
            A.Normalize( (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ToTensorV2(),
    ])

    test_transforms = A.Compose([
        A.Normalize( (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ])

    # Train and Test Datasets
    train_dataset = Cifar10SearchDataset('~/data/CIFAR10', train=True, download=True,
                        transform=train_transforms)
    test_dataset = Cifar10SearchDataset('~/data/CIFAR10', train=False, download=True,
                        transform=test_transforms)
    
    dataloader_args = dict(shuffle = True, batch_size = 512, num_workers = 4, pin_memory = True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return  train_dataloader, test_dataloader
