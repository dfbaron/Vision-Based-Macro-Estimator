import torch
import torch.nn as nn
from pathlib import Path

class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0, path: Path = MODEL_SAVE_PATH):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module, optimizer: optim.Optimizer, epoch: int):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, optimizer: optim.Optimizer, epoch: int):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': val_loss,
        }
        torch.save(checkpoint_data, self.path)
        self.val_loss_min = val_loss