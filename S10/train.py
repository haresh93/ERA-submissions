import torch
import matplotlib.pyplot as plt
from utils import get_correct_pred_count, get_lr
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []
misclassified_images = []
lrs = []

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
