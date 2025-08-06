import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Tuple, Any

class Nutrition5kDataset(Dataset):
    """
    Custom PyTorch Dataset for the Nutrition5k data.
    Connects image paths with their corresponding nutritional information.
    """
    def __init__(self, images_csv_path: str, labels_csv_path: str, transform: Any = None):
        """
        Args:
            images_csv_path (str): Path to the CSV file with image paths and dish_ids.
            labels_csv_path (str): Path to the CSV file with dish_ids and nutritional labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        
        # 1. Load data and merge the two CSVs into a single master DataFrame
        # This is efficient and keeps all information for a sample in one row.
        df_images = pd.read_csv(images_csv_path)
        df_labels = pd.read_csv(labels_csv_path)
        
        # Merge based on the common 'dish_id' column
        self.data_frame = pd.merge(df_images, df_labels, on="dish_id")
        
        self.transform = transform
        
        # 2. Define the target columns for our regression task
        self.target_columns = ['total_calories', 'total_fat', 'total_carb', 'total_protein']

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the sample at the given index.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            A tuple (image, label) where:
            - image is the transformed image tensor.
            - label is a tensor of the nutritional values.
        """
        # 1. Get the data row for the given index
        sample_row = self.data_frame.iloc[idx]
        
        # 2. Load the image from the path
        image_path = Path(sample_row["path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Skipping.")
            # Return a dummy tensor if you want to handle this gracefully,
            # or you can just let it error out.
            # For this example, we'll try the next item if possible,
            # though in a real Dataloader this might be complex.
            # A cleaner approach is to ensure data integrity beforehand.
            return self.__getitem__((idx + 1) % len(self)) # Be careful with this approach

        # 3. Extract the label (nutritional info) and convert to a tensor
        # This creates a vector like [calories, fat, carbs, protein]
        label_values = sample_row[self.target_columns].values.astype('float32')
        label = torch.from_numpy(label_values)
        
        # 4. Apply transformations to the image, if any
        if self.transform:
            image = self.transform(image)
        
        return image, label