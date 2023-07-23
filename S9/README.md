## CIFAR10 Classification using Advanced Convolutions and Albumentations

In this repository, we have created a model with CNN using depthwise Convolution and Grouped Convolutions and using Albumentations image augmentation library and got an accuracy of **86.10%**

### Reposity Walkthrough

The repository is structured in a modular way, where we have the following files `model.py`, `dataset.py`,  `utils.py`:
- `model.py`  - This file consists of the model with depthwise convolutions and Dilated and Strided convolutions without using any MaxPooling layer 
- `dataset.py` - It consists a `Cifar10SearchDataset` class which extends the `torchvision.datasets.CIFAR10` class and overrides the `__getitem__` method by applying the albumentations transforms
- `utils.py` - This file consists of the some utility functions which are used for displaying the graphs of traning and testing losses and accuracies over epochs and so on.

### CNN Model 

We have used a pure convolutional network using Depthwise convolutions and in place of Max Pooling layer we have used Dilated and Strided Convolutions together

**Total Parameters used: 194880**

## Image Augmentation Techniques used
The following Image Augmentation techniques were used:

- **HorizontalFlip()** - For Horizontal Flipping the image as the CIFAR dataset the objects can flipped horizontally and then also we should be able to identity it as the same label for example Car when flipped horizontal looks like a car only

- **ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=0.5)** - Shifting the image, scaling the image and rotating the 15 degrees

- **CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None)** - This is the Cutout strategy 

