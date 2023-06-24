## CIFAR10 Classification using DNN 

In this repository, we have created models with CNN and different normalizations to compare how the 3 normalizations have performed over the CIFAR10 dataset.

The repository is structured in a modular way, where we have `model.py` and `utils.py` files which contain all the models and the utility functions respectively.

There are 3 models in the `model.py` file with following names:

`NetBN` - Network with Batch Normalization

`NetLN` - Network with Layer Normalization

`NetGN` - Network with Group Normalization

We have used a standard Network for the architecture as follows:

**Total Parameters used: 41026**

The following Image Augmentation techniques were used:

**RandomCrop(32, padding=4, padding_mode='reflect')** - This will crop at a random location in the image by padding it by 4px and with output size at 32 x 32

**RandomHorizontalFlip()** - Horizontally flips the image randomly with a probability of 0.5 

### Observations

1. Batch Normalization has performed very well among all the 3 models with the same architecture and parameters where it was able to get to 77.28% test accuracy.
2. BN achieved an accuracy of 70% in the 6th epoch itself, but in the Layer Normalization it was at 16th epoch, in Group Normalization at 13th epoch.
3. This shows that for Convolutional Neural Networks Batch Normalization is a better one to use.



### Model using Batch Normalization

Training Accuracy: 73.70%

Test Accuracy: 77.28%

Results:

Below are the misclassified images:



### Model using Layer Normalization

Training Accuracy: 67.72%

Test Accuracy: 72.75%

Results:

Below are the misclassified images:


### Model using Group Normalization

Training Accuracy: 68.08%

Test Accuracy: 74.06

Results: 

Below are the misclassified images: 

