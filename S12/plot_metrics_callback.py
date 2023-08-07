class PlotMetricsCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.losses = []
        self.val_losses = []
        self.val_accs = []
    
    def on_epoch_end(self, trainer, pl_module):
        self.losses.append(trainer.progress_bar_dict['loss'])
        self.val_losses.append(trainer.progress_bar_dict['val_loss'])
        self.val_accs.append(trainer.progress_bar_dict['val_acc'])
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()