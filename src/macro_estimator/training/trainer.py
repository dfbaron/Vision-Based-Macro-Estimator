import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from pathlib import Path
import timm
from tqdm import tqdm
from typing import Tuple, Dict, Any

from ..models.vit_regressor import ViTRegressor
from ..datasets import Nutrition5kDataset
from ..utils import EarlyStopping

class ModelTrainer:
    """
    A class to encapsulate the entire training, validation, and testing pipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ModelTrainer with a configuration dictionary.
        
        Args:
            config (Dict[str, Any]): A dictionary containing all necessary configurations.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")

        # Initialize core components to None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.early_stopper = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def _prepare_dataloaders(self):
        """Prepares and splits the dataset into train, validation, and test DataLoaders."""
        print("--- Preparing DataLoaders ---")
        temp_model = timm.create_model(self.config['model_params']['model_name'], pretrained=True)
        data_config = timm.data.resolve_data_config(model=temp_model)
        transforms = timm.data.create_transform(**data_config)
        del temp_model

        full_dataset = Nutrition5kDataset(self.config['data_paths']['images_csv'], self.config['data_paths']['labels_csv'], transform=transforms)
        
        dish_ids = full_dataset.data_frame['dish_id'].unique()
        np.random.shuffle(dish_ids)

        n_dishes = len(dish_ids)
        test_split_idx = int(n_dishes * float(self.config['data_split']['test_split']))
        val_split_idx = test_split_idx + int(n_dishes * float(self.config['data_split']['val_split']))
        
        train_dish_ids = dish_ids[val_split_idx:]
        val_dish_ids = dish_ids[test_split_idx:val_split_idx]
        test_dish_ids = dish_ids[:test_split_idx]

        train_indices = full_dataset.data_frame[full_dataset.data_frame['dish_id'].isin(train_dish_ids)].index.tolist()
        val_indices = full_dataset.data_frame[full_dataset.data_frame['dish_id'].isin(val_dish_ids)].index.tolist()
        test_indices = full_dataset.data_frame[full_dataset.data_frame['dish_id'].isin(test_dish_ids)].index.tolist()

        train_dataset, val_dataset, test_dataset = Subset(full_dataset, train_indices), Subset(full_dataset, val_indices), Subset(full_dataset, test_indices)
        
        self.train_loader = DataLoader(train_dataset, batch_size=int(self.config['training_params']['batch_size']), shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=int(self.config['training_params']['batch_size']), shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=int(self.config['training_params']['batch_size']), shuffle=False, num_workers=4)
        
        print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

    def _prepare_model(self):
        """Initializes the model, optimizer, criterion, and early stopper."""
        print("--- Preparing Model, Optimizer, and Callbacks ---")
        self.model = ViTRegressor(model_name=self.config['model_params']['model_name'], n_outputs=int(self.config['model_params']['n_outputs'])).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=float(self.config['training_params']['learning_rate']), weight_decay=float(self.config['training_params']['weight_decay']))
        self.criterion = nn.MSELoss()
        self.early_stopper = EarlyStopping(
            patience=int(self.config['callbacks']['early_stopping_patience']),
            verbose=True,
            path=self.config['model_paths']['model_save_path']
        )

    def _load_checkpoint(self) -> Tuple[int, float]:
        """Loads a training checkpoint if specified."""
        path = Path(self.config['model_paths']['resume_checkpoint_path'])
        if path and path.exists():
            print(f"--- Resuming training from checkpoint: {path} ---")
            checkpoint = torch.load(path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.early_stopper.val_loss_min = best_val_loss
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}. Best val loss: {best_val_loss:.4f}")
            return start_epoch
        return 0

    def _train_epoch(self) -> float:
        """Trains the model for one epoch with a progress bar."""
        self.model.train()
        running_loss = 0.0
        
        # 1. Envolver el DataLoader con tqdm para crear la barra de progreso
        progress_bar = tqdm(
            self.train_loader, 
            desc="  Training", 
            leave=False, # No deja una barra duplicada al final
            ncols=100 # Ancho de la barra
        )
        
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # 2. Actualizar la barra de progreso con la pérdida actual
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(self.train_loader)

    def _validate_epoch(self) -> Tuple[float, float]:
        """Validates the model for one epoch with a progress bar."""
        self.model.eval()
        running_loss = 0.0
        running_mae = 0.0
        
        # 1. Envolver el DataLoader con tqdm
        progress_bar = tqdm(
            self.val_loader, 
            desc="  Validating", 
            leave=False,
            ncols=100
        )
        
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                running_mae += torch.abs(outputs - labels).sum().item()

        avg_loss = running_loss / len(self.val_loader)
        
        # Aquí el cálculo de `total_predictions` puede ser un poco delicado si
        # el último batch es más pequeño. Una forma más robusta es esta:
        total_items = len(self.val_loader.dataset)
        num_targets = len(self.val_loader.dataset.dataset.target_columns) # Accede al dataset original
        avg_mae = running_mae / (total_items * num_targets)
        
        return avg_loss, avg_mae

    def train(self):
        """
        The main training loop that orchestrates the entire process.
        It handles epoch iteration, validation, and early stopping.
        """
        # Step 1: Initialize all components
        self._prepare_dataloaders()
        self._prepare_model()
        start_epoch = self._load_checkpoint()
        
        print("\n--- Starting Training ---")
        
        # Step 2: Main training loop
        for epoch in range(start_epoch, int(self.config['training_params']['epochs'])):
            print(f"\nEpoch {epoch+1}/{int(self.config['training_params']['epochs'])}")
            
            # Run one epoch of training
            train_loss = self._train_epoch()
            
            # Run one epoch of validation
            val_loss, val_mae = self._validate_epoch()
            
            # Log the results
            self._log_epoch_summary(epoch, train_loss, val_loss, val_mae)
            
            # Step 3: Check for early stopping and save best model
            # The EarlyStopping object handles the logic of saving the best model.
            self.early_stopper(val_loss, self.model, self.optimizer, epoch)
            if self.early_stopper.early_stop:
                print("Early stopping triggered!")
                break
        
        print("\n--- Training Finished ---")
        
        # Step 4: Run final evaluation on the test set with the best model
        self.test()

    def _log_epoch_summary(self, epoch: int, train_loss: float, val_loss: float, val_mae: float):
        """Prints a summary of the epoch's performance."""
        print(f"\r✓ Epoch {epoch+1} Summary:")
        print(f"  - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")
        
    def test(self):
        """Evaluates the final model on the unseen test set."""
        print("\n--- Evaluating on Test Set ---")
        # Load the best performing model
        checkpoint = torch.load(self.config['model_paths']['model_save_path'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_mae = self._validate_epoch()  # We can reuse the validation logic for testing
        print(f"  - Test Loss (MSE): {test_loss:.4f}")
        print(f"  - Test MAE: {test_mae:.2f} (avg. error per nutrient)")