import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def print_train_data_stats():
    simple_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                      #  transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])
    exp = datasets.CIFAR10('../data', train=True, download=True, transform=simple_transforms)
    exp_data = exp.data.astype(float)
    #exp_data = exp.transform(exp_data.numpy())
    exp_tensor = torch.as_tensor(exp_data)

    print('[Train]')
    print(' - Numpy Shape:', exp_data.shape)
    print(' - min:', torch.min(exp_tensor))
    print(' - max:', torch.max(exp_tensor))
    print(' - mean:', torch.mean(exp_tensor))
    print(' - std:', torch.std(exp_tensor))
    print(' - var:', torch.var(exp_tensor))

def plot_train_data(train_loader):
    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def get_correct_pred_count(prediction, labels):
    return prediction.argmax(dim=1).eq(labels).sum().item()
