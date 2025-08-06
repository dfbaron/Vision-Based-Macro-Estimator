import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from PIL import Image
from pathlib import Path
import timm

class ViTRegressor(nn.Module):
    """
    A wrapper around a pre-trained ViT model from timm, with a custom regression head.
    """
    def __init__(self, model_name: str, n_outputs: int, pretrained: bool = True):
        super().__init__()
        # Load the pre-trained model without its classification head
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Freeze the backbone parameters
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Get the number of features from the pre-trained model's output
        n_features = self.vit.embed_dim
        
        # Create a new regression head
        self.regressor_head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, n_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through the ViT backbone
        features = self.vit(x)
        # Pass features through our custom regression head
        outputs = self.regressor_head(features)
        return outputs