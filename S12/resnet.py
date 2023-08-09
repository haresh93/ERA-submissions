import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import Cifar10SearchDataset


PATH_DATASETS  = '~/data/CIFAR10'
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
EPOCHS = 24

class LitResnet(LightningModule):
    def __init__(self, transforms = None, data_dir=PATH_DATASETS, learning_rate=0.01):
        super().__init__()
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
            self._res_block(128)
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
            self._res_block(512)
        )

        self.classification_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size = 4, stride = 2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 10)
        )

        self.misclassified_images = []
        self.accuracy = Accuracy(task='multiclass', num_classes = 10)
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        if transforms is None:
            self.training_transforms = A.Compose([
                A.PadIfNeeded(min_height=40, min_width=40, value=(0.4914, 0.4822, 0.4465), p=1), # Padding with a border of 4 pixels
                A.RandomCrop(width=32, height=32),
                A.HorizontalFlip(p=0.5),
                A.CoarseDropout(max_holes = 1, max_height=8, max_width=8,
                            min_holes = 1, min_height=8, min_width=8, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None),
                A.Normalize( (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ToTensorV2(),
            ])
        else:
            self.training_transforms = transforms

        self.test_transforms = A.Compose([
            A.Normalize( (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ToTensorV2(),
        ])
    
    def _res_block(self, channels, kernel_size = 3, padding=1):
        layers = [
            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
    
        return nn.Sequential(*layers)

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


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # for i in range(len(pred)):
        #     if pred[i] != target[i]:
        #         self.misclassified_images.append([data[i], pred[i], target[i]])

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

        criterion = nn.CrossEntropyLoss()
        lr_finder = LRFinder(self, optimizer, criterion, device = 'cuda')
        lr_finder.range_test(self.train_dataloader(), end_lr = 10, num_iter = 200, step_mode = "exp")
        lr_finder.plot()
        optimal_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]
        lr_finder.reset()

        scheduler = OneCycleLR(
            optimizer,
            max_lr = 0.018,
            steps_per_epoch=len(self.train_dataloader()),
            epochs = EPOCHS,
            pct_start = 5/EPOCHS,
            div_factor=100,
            three_phase = False,
            final_div_factor=1000,
            anneal_strategy='linear'
        )

        return { "optimizer": optimizer, "lr_scheduler": scheduler }

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        Cifar10SearchDataset(self.data_dir, train=True, download=True)
        Cifar10SearchDataset(self.data_dir, train=False, download=True)

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            cifar_full = Cifar10SearchDataset(self.data_dir, train = True, transform = self.training_transforms)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [48000, 2000])

        if stage == 'test' or stage is None:
            self.cifar_test = Cifar10SearchDataset(self.data_dir, train=False, transform = self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size = BATCH_SIZE, num_workers = os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size = BATCH_SIZE, num_workers = os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size = BATCH_SIZE, num_workers = os.cpu_count())