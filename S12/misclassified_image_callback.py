import os
import torch
import torchvision
from pytorch_lightning import LightningModule, Callback
from torchvision.transforms import ToPILImage

class MisclassifiedImagesCallback(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.to_pil = ToPILImage()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the model and dataloader
        model = pl_module.model
        dataloader = pl_module.val_dataloader()

        # Set the model to evaluation mode
        model.eval()

        # Loop through the validation dataloader
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Forward pass
            with torch.no_grad():
                outputs = model(images)
                predicted_labels = torch.argmax(outputs, dim=1)

            # Check for misclassified samples
            misclassified_indices = (predicted_labels != labels).nonzero().squeeze()

            for index in misclassified_indices:
                image = self.to_pil(images[index].cpu())
                true_label = labels[index].item()
                predicted_label = predicted_labels[index].item()
                image_path = os.path.join(self.save_dir, f"misclassified_{index}_true_{true_label}_pred_{predicted_label}.jpg")
                image.save(image_path)