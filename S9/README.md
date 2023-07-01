## CIFAR10 Classification using Advanced Convolutions and Albumentations

In this repository, we have created a model with CNN using depthwise Convolution and Grouped Convolutions and using Albumentations image augmentation library.

The repository is structured in a modular way, where we have `model.py`, `dataset.py` and `utils.py` files which contain the model,  the dataset and the utility functions respectively.

There are 3 models in the `model.py` file with following names:

`NetBN` - Network with Batch Normalization

`NetLN` - Network with Layer Normalization

`NetGN` - Network with Group Normalization

We have used a standard Network for the architecture as follows:

**Total Parameters used: 41026**

<img width="476" alt="Network" src="https://github.com/haresh93/ERA-submissions/assets/9997345/cfb222ad-a121-460b-925a-57c356339cf3">

The following Image Augmentation techniques were used:

**RandomCrop(32, padding=4, padding_mode='reflect')** - This will crop at a random location in the image by padding it by 4px and with output size at 32 x 32

**RandomHorizontalFlip()** - Horizontally flips the image randomly with a probability of 0.5 

### Observations

1. Batch Normalization has performed very well among all the 3 models with the same architecture and parameters where it was able to get to 77.28% test accuracy.
2. BN achieved an accuracy of 70% in the 6th epoch itself, but in the Layer Normalization it was at 16th epoch, in Group Normalization at 13th epoch.
3. Group Normalization has performed better than Layer Normalization as it achieved both better training and test accuracy.
4. This shows that for Convolutional Neural Networks Batch Normalization is a better one to use.


### Model using Batch Normalization

Training Accuracy: 73.70%

Test Accuracy: 77.28%

Results:
<img width="986" alt="BN-results" src="https://github.com/haresh93/ERA-submissions/assets/9997345/2f3f9b2d-3e07-4bad-adbb-4f54cd2c27c6">

Below are the misclassified images:

<img width="1031" alt="BN-misclassified" src="https://github.com/haresh93/ERA-submissions/assets/9997345/1439f57a-6d76-4401-af82-136a465eb8bf">

### Model using Layer Normalization

Training Accuracy: 67.72%

Test Accuracy: 72.75%

Results:

<img width="986" alt="LN-results" src="https://github.com/haresh93/ERA-submissions/assets/9997345/c4212c69-def6-4a3e-8912-26af028bfba5">

Below are the misclassified images:

<img width="825" alt="LN-misclassified" src="https://github.com/haresh93/ERA-submissions/assets/9997345/a235260b-b963-4f8b-a1b3-2674a3e82b8b">

### Model using Group Normalization with Group Size 8

Training Accuracy: 68.08%

Test Accuracy: 74.06

Results: 

<img width="977" alt="GN-results" src="https://github.com/haresh93/ERA-submissions/assets/9997345/8473cfde-18fd-471b-94e8-cc258813539e">

Below are the misclassified images: 

<img width="813" alt="GN-misclassified" src="https://github.com/haresh93/ERA-submissions/assets/9997345/45af5690-3cff-4532-9294-9baa75e37ab0">

