"""
A class to encapsulate and manage the data preparation tasks for the Nutrition5k dataset.
This includes scanning image directories, extracting dish_ids, and parsing complex
metadata CSVs into clean, structured CSVs for downstream use.
"""
import csv
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Generator

class DataPreparer:
    """
    Manages the preprocessing and organization of Nutrition5k dataset files.

    It handles:
    - Scanning image directories to create a mapping of image paths to dish IDs.
    - Parsing the complex dish metadata CSV into a flattened, structured format.
    - Saving the processed data into standardized CSV files.
    """

    # --- Configuration and Schema Definitions ---
    # These are class-level constants. They can be overridden by instance attributes
    # if passed in __init__ or modified directly via the class.
    DISH_COLUMNS: List[str] = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']
    INGREDIENT_COLUMNS: List[str] = ['id', 'name', 'grams', 'calories', 'fat', 'carb', 'protein']
    TYPE_SCHEMA: Dict[str, Any] = {
        'total_calories': float, 'total_mass': float, 'total_fat': float,
        'total_carb': float, 'total_protein': float, 'grams': float,
        'calories': float, 'fat': float, 'carb': float, 'protein': float
    }

    def __init__(self, 
                 base_data_path: Path = Path("data/Extracted_Files/nutrition5k_dataset"),
                 output_dir: Path = Path("data/csv_files"),
                 image_glob_pattern: str = "frames_sampled70/*.jpeg",
                 metadata_filename: str = "dish_metadata_cafe1.csv",
                 images_csv_filename: str = "images.csv",
                 labels_csv_filename: str = "labels.csv"):
        """
        Initializes the DataPreparer with paths and configuration.

        Args:
            base_data_path (Path): Root path to the extracted Nutrition5k dataset.
            output_dir (Path): Directory where processed CSVs will be saved.
            image_glob_pattern (str): Pattern to find image files (e.g., "frames_sampled70/*.jpeg").
            metadata_filename (str): Name of the metadata CSV file.
            images_csv_filename (str): Name of the output CSV for image paths.
            labels_csv_filename (str): Name of the output CSV for parsed labels.
        """
        self.base_data_path = base_data_path
        self.image_dir = self.base_data_path / "imagery/side_angles"
        self.metadata_path = self.base_data_path / "metadata" / metadata_filename
        
        self.output_dir = output_dir
        self.output_images_csv_path = self.output_dir / images_csv_filename
        self.output_labels_csv_path = self.output_dir / labels_csv_filename
        
        self.image_glob_pattern = image_glob_pattern
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"DataPreparer initialized with base_data_path: {self.base_data_path}")
        print(f"Output directory: {self.output_dir}")

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

    def create_image_path_csv(self) -> None:
        """
        Scans for processed JPEG images based on the configured pattern and
        creates a CSV file mapping image paths to their dish_ids.
        """
        print(f"Scanning for images in '{self.image_dir}' using pattern '{self.image_glob_pattern}'...")
        
        image_paths = list(self.image_dir.rglob(self.image_glob_pattern))
        
        if not image_paths:
            print("Warning: No images found. Check the image_dir and image_glob_pattern.")
            return

        df = pd.DataFrame(image_paths, columns=['path'])
        
        # Extract dish_id from the folder structure (e.g., .../dish_ID/frames_sampled70/image.jpeg)
        df['dish_id'] = df['path'].apply(lambda p: p.parent.parent.name)
        
        df.to_csv(self.output_images_csv_path, index=False)
        print(f"Successfully created image path CSV with {len(df)} entries at '{self.output_images_csv_path}'")

    def parse_dish_metadata_to_df(self) -> pd.DataFrame:
        """
        Parses the complex dish metadata CSV into a clean pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with one row per dish and ingredients as a JSON string.
        """
        print(f"Parsing dish metadata from '{self.metadata_path}'...")
        parsed_dishes = []
        
        try:
            with self.metadata_path.open(mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue

                    # Use class constants for column names
                    dish_data = dict(zip(self.DISH_COLUMNS, row))
                    dish_data = self._convert_types(dish_data, self.TYPE_SCHEMA)

                    ingredient_data_flat = row[len(self.DISH_COLUMNS):]
                    ingredients_list = [
                        self._convert_types(dict(zip(self.INGREDIENT_COLUMNS, chunk)), self.TYPE_SCHEMA)
                        for chunk in self._chunker(ingredient_data_flat, len(self.INGREDIENT_COLUMNS))
                    ]
                    
                    dish_data['ingredients'] = json.dumps(ingredients_list)
                    parsed_dishes.append(dish_data)
                    
        except FileNotFoundError:
            print(f"Error: Metadata file not found at '{self.metadata_path}'")
            return pd.DataFrame()

        print(f"Successfully parsed {len(parsed_dishes)} dishes.")
        return pd.DataFrame(parsed_dishes)

    def run_preparation(self) -> None:
        """
        Orchestrates the full data preparation process:
        1. Creates the image paths CSV.
        2. Parses the metadata CSV and saves it.
        """
        print("\n--- Starting Data Preparation Process ---")
        
        # Task 1: Create the image paths CSV
        self.create_image_path_csv()
        
        print("\n" + "-"*40 + "\n")
        
        # Task 2: Parse metadata and save as CSV
        df_labels = self.parse_dish_metadata_to_df()
        if not df_labels.empty:
            df_labels.to_csv(self.output_labels_csv_path, index=False)
            print(f"Successfully saved labels CSV at '{self.output_labels_csv_path}'")
            print("Labels DataFrame head:")
            print(df_labels.head())

        print("\n--- Data Preparation Finished ---")