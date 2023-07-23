## CIFAR10 Classification using Custom Resnet Model and One Cycle LR Policy

In this repository, we have created a model with CNN using residual connections and One Cycle LR policy and tested the model on CIFAR10 Dataset and successfully achieved an Accuracy of **93.3%** in 24 Epochs.

### Reposity Walkthrough

The repository is structured in a modular way, where we have the following files `custom_resnet.py`, `dataset.py`, `train.py`,  `utils.py`:
- `custom-resnet.py`  - the Custom Resnet model will discuss about it in detail in the next section
- `dataset.py` - It consists a `Cifar10SearchDataset` class which extends the `torchvision.datasets.CIFAR10` class and overrides the `__getitem__` method by applying the albumentations transforms
- `train.py` - This file consists of the `model_train` and `model_test` functions which train the model and test the model respectively.
- `utils.py` - This file consists of the some utility functions which are used for displaying the graphs of traning and testing losses and accuracies over epochs and so on.

### The Custom Resnet Model

We are using a custom resnet model with 2 residual blocks and skip connections in the model, as shown in the below diagram.

<img width="849" alt="Screenshot 2023-07-23 at 5 41 57 AM" src="https://github.com/haresh93/ERA-submissions/assets/9997345/e7b116bd-fff5-417d-b4d0-a39147e7150d">

The total number of parameters used in the model are **6575370** parameters. Below is the model summary:

<img width="567" alt="Screenshot 2023-07-23 at 7 41 11 AM" src="https://github.com/haresh93/ERA-submissions/assets/9997345/977229a7-95ce-4141-9d5a-4b5dab6ce247">

### Training

We have used Adam Optimizer with Cross Entropy Loss and One Cycle LR scheduler. We Will discuss about the One Cycle LR scheduler in detail in the next section, for the Adam Optimizer the following are the parameters set:

- Initial LR: **0.01**
- Weight Decay: **0.0001**

Observation: Initially we have tried with a higher learning rate of **0.1** and Weight Decay of **0.3** which resulted in test accuracy jumping here and there a lot, after setting the optimizer with the above parameters there was a steady increase in the test accuracy along the epochs.

### One Cycle LR Policy

We have used One Cycle LR scheduler for faster training of the model and we have used the `torch_lr_finder` package for finding the Maximum LR, below is output of the range test from the `torch_lr_finder`:

<img width="1118" alt="Screenshot 2023-07-23 at 7 41 49 AM" src="https://github.com/haresh93/ERA-submissions/assets/9997345/717614ef-6ed9-4bca-8428-ae83d3b9c901">

Hence here are the parameters for the One Cycle LR:

- Max LR: **0.018**
- Min LR: **0.00018** (Div Factor of 100)
- PCT Start: 5/24

Observation: Using the One Cycle LR scheduler the training time has reduced drastically and we were able to hit an accuracy of **83.2%** in the 7th Epoch itself, and finally the accuracy of **93.3%** in 24 Epochs.
