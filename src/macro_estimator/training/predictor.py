# src/macro_estimator/training/predictor.py
import torch
import timm
from pathlib import Path
from PIL import Image
import io
import numpy as np
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importar la arquitectura del modelo
from ..models.vit_regressor import ViTRegressor

class Predictor:
    """
    Encapsulates a trained model and provides an interface for making predictions.
    The model is loaded once and reused for efficiency.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Predictor using device: {self.device} ---")
        self.transforms = self._load_transforms()
        self.model = self._load_model()
        
    def _load_transforms(self):
        model_name = self.config['model_params']['model_name']
        temp_model = timm.create_model(model_name, pretrained=True)
        data_config = timm.data.resolve_data_config(model=temp_model)
        transforms = timm.data.create_transform(**data_config)
        del temp_model
        return transforms

    def _load_model(self) -> torch.nn.Module:
        model_path = Path(self.config['model_paths']['model_save_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        print(f"Loading model from: {model_path}")
        model = ViTRegressor(
            model_name=self.config['model_params']['model_name'],
            n_outputs=int(self.config['model_params']['n_outputs'])
        )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, float]:
        """
        Makes a nutritional prediction on an image provided as bytes.
        This is the primary method for API use.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not open image from bytes: {e}")
        
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(image_tensor)
            
        macros = prediction.cpu().numpy()[0]
        
        if len(macros) == 4:
            return {
                'calories': max(0, round(float(macros[0]), 2)),
                'fat_grams': max(0, round(float(macros[1]), 2)),
                'carb_grams': max(0, round(float(macros[2]), 2)),
                'protein_grams': max(0, round(float(macros[3]), 2))
            }
        elif len(macros) == 1:
            return {
                'carb_grams': max(0, round(float(macros[0]), 2))
            }
        else:
            raise TypeError(f"Unexpected type for prediction output: {type(macros)}. Expected torch.Tensor or np.ndarray.")

    def predict_from_file(self, image_path: Path) -> Dict[str, float]:
        """
        Makes a prediction on an image from a file path.
        Useful for command-line scripts.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found at {image_path}")
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return self.predict_from_bytes(image_bytes)
    
    def predict_on_loader(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs predictions on an entire dataset provided by a DataLoader.
        This is the primary method for batch evaluation.

        Args:
            dataloader (DataLoader): The DataLoader containing the test data.

        Returns:
            A tuple (y_true, y_preds) of NumPy arrays.
        """
        self.model.eval()  # Asegurarse de que el modelo esté en modo de evaluación
        
        y_true = []
        y_preds = []

        # Envolver el dataloader con tqdm para una barra de progreso
        progress_bar = tqdm(dataloader, desc="Predicting on dataset")
        
        with torch.no_grad():
            for images, labels in progress_bar:
                # Mover las imágenes al dispositivo correcto
                images = images.to(self.device)
                
                # Obtener las predicciones del modelo
                outputs = self.model(images)
                
                # Mover los resultados a la CPU y guardarlos
                y_preds.append(outputs.cpu().numpy())
                y_true.append(labels.cpu().numpy())
        
        # Concatenar los resultados de todos los lotes en arrays únicos
        y_preds = np.vstack(y_preds)
        y_true = np.vstack(y_true)
        
        return y_true, np.maximum(y_preds, 0)