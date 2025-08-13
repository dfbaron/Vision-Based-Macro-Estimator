"""
A class to encapsulate and manage the data preparation tasks for the Nutrition5k dataset.
This includes scanning image directories, parsing metadata, merging data, and creating
separate, reproducible train, validation, and test sets.
"""
import csv
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Generator, Tuple
from sklearn.model_selection import train_test_split

class DataPreparer:
    """
    Manages the preprocessing, merging, and splitting of the Nutrition5k dataset.
    """
    # Class-level constants for schema definitions
    DISH_COLUMNS: List[str] = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']
    INGREDIENT_COLUMNS: List[str] = ['id', 'name', 'grams', 'calories', 'fat', 'carb', 'protein']
    TYPE_SCHEMA: Dict[str, Any] = {
        'total_calories': float, 'total_mass': float, 'total_fat': float,
        'total_carb': float, 'total_protein': float, 'grams': float,
        'calories': float, 'fat': float, 'carb': float, 'protein': float
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparer with a configuration dictionary.

        Args:
            config (Dict[str, Any]): A dictionary containing all necessary configurations.
        """
        self.config = config
        
        # Paths are derived from the config for clarity
        self.image_dir = Path(config['data_paths']['image_dir'])
        self.metadata_paths = [Path(path) for path in config['data_paths']['metadata_path'].split(',')]
        self.output_dir = Path(config['data_paths']['output_dir'])
        
        self.image_glob_pattern = config['settings']['image_glob_pattern']
        self.random_seed = config['settings']['random_seed']
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("DataPreparer initialized.")

    def run(self):
        """
        Executes the full data preparation pipeline:
        1. Load and parse raw data.
        2. Merge into a master DataFrame.
        3. Split the data by dish_id.
        4. Save the train, validation, and test sets as separate CSV files.
        """
        print("\n--- Starting Data Preparation Process ---")
        
        # Step 1: Load and parse the raw data sources
        df_images = self._create_image_path_df()
        df_labels = self._parse_dish_metadata_df()
        
        if df_images is None or df_labels.empty:
            print("Aborting preparation due to missing data.")
            return

        # Step 2: Merge into a single master DataFrame
        master_df = pd.merge(df_images, df_labels, on='dish_id')
        print(f"Created master DataFrame with {len(master_df)} total samples.")

        # Step 3: Split the data into train, validation, and test sets
        train_df, val_df, test_df = self._split_data(master_df)

        # Step 4: Save each DataFrame to its own CSV file
        self._save_splits(train_df, val_df, test_df)

        print("\n--- Data Preparation Finished ---")

    def _create_image_path_df(self) -> pd.DataFrame | None:
        """Scans for images and creates a DataFrame of paths and dish_ids."""
        print(f"Scanning for images in '{self.image_dir}'...")
        image_paths = list(self.image_dir.rglob(self.image_glob_pattern))
        
        if not image_paths:
            print(f"Warning: No images found with pattern '{self.image_glob_pattern}'.")
            return None

        df = pd.DataFrame(image_paths, columns=['path'])
        df['dish_id'] = df['path'].apply(lambda p: p.parent.parent.name)
        return df

    def _parse_dish_metadata_df(self) -> pd.DataFrame:
        """Parses the complex dish metadata CSV."""
        parsed_dishes = []
        for metadata_path in self.metadata_paths:
            print(f"Parsing dish metadata from '{metadata_path}'...")
            try:
                with metadata_path.open(mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row: continue
                        dish_data = self._process_metadata_row(row)
                        parsed_dishes.append(dish_data)
            except FileNotFoundError:
                print(f"Error: Metadata file not found at '{metadata_path}'")
        
        return pd.DataFrame(parsed_dishes)

    def _process_metadata_row(self, row: List[str]) -> Dict[str, Any]:
        """Processes a single row from the metadata CSV."""
        dish_data = dict(zip(self.DISH_COLUMNS, row))
        dish_data = self._convert_types(dish_data, self.TYPE_SCHEMA)
        
        ingredient_data_flat = row[len(self.DISH_COLUMNS):]
        ingredients_list = [
            self._convert_types(dict(zip(self.INGREDIENT_COLUMNS, chunk)), self.TYPE_SCHEMA)
            for chunk in self._chunker(ingredient_data_flat, len(self.INGREDIENT_COLUMNS))
        ]
        dish_data['ingredients'] = json.dumps(ingredients_list)
        return dish_data

    def _split_data(self, master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the master DataFrame by dish_id into train, val, and test sets."""
        print("Splitting data into train, validation, and test sets by dish_id...")
        
        dish_ids = master_df['dish_id'].unique()
        val_size = float(self.config['data_split']['val_split'])
        test_size = float(self.config['data_split']['test_split'])
        
        # First split: separate test set from the rest
        train_ids, test_val_ids = train_test_split(
            dish_ids,
            test_size=test_size+val_size,
            random_state=int(self.random_seed)
        )
        
        # Second split: separate train and validation sets
        # Calculate the correct proportion for the validation set from the remaining data
        test_ids, val_ids = train_test_split(
            test_val_ids,
            test_size=0.5,
            random_state=int(self.random_seed)
        )
        
        # Filter the master DataFrame to create the final splits
        train_df = master_df[master_df['dish_id'].isin(train_ids)]
        val_df = master_df[master_df['dish_id'].isin(val_ids)]
        test_df = master_df[master_df['dish_id'].isin(test_ids)]
        
        print(f"Split complete: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")
        return train_df, val_df, test_df

    def _save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Saves the split DataFrames to separate CSV files."""
        train_df.to_csv(self.output_dir / "train_dataset.csv", index=False)
        val_df.to_csv(self.output_dir / "validation_dataset.csv", index=False)
        test_df.to_csv(self.output_dir / "test_dataset.csv", index=False)
        print(f"Datasets saved successfully in '{self.output_dir}'")

    @staticmethod
    def _chunker(sequence: List[Any], size: int) -> Generator[List[Any], None, None]:
        """Yields successive size-sized chunks from a sequence."""
        for i in range(0, len(sequence), size):
            yield sequence[i:i + size]

    @staticmethod
    def _convert_types(data_dict: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Safely converts string values in a dictionary to numeric types based on a schema."""
        converted = {}
        for key, value in data_dict.items():
            if key in schema:
                try:
                    converted[key] = schema[key](value) if value and value.strip() else 0.0
                except (ValueError, TypeError):
                    converted[key] = 0.0  # Default to 0.0 on conversion error
            else:
                converted[key] = value
        return converted