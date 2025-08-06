"""
This script performs two main data preparation tasks for the Nutrition5k dataset:
1. It scans a directory of processed images, extracts the dish_id from the
   folder structure, and creates a CSV file mapping image paths to their dish_id.
2. It parses a complex, non-standard CSV of dish metadata (including nested
   ingredient information) and saves it as a structured, clean CSV.
"""
import os
import csv
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Generator

# --- 1. Configuration and Constants ---
# Using Path objects for robust, OS-agnostic path handling
BASE_DATA_PATH = Path("data/Extracted_Files/nutrition5k_dataset")

IMAGE_DIR = BASE_DATA_PATH / "imagery/side_angles"
METADATA_PATH = BASE_DATA_PATH / "metadata/dish_metadata_cafe1.csv"

OUTPUT_DIR = Path("data/csv_files")
OUTPUT_IMAGES_CSV = OUTPUT_DIR / "images.csv"
OUTPUT_LABELS_CSV = OUTPUT_DIR / "labels.csv"

# Schema definitions
DISH_COLUMNS = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']
INGREDIENT_COLUMNS = ['id', 'name', 'grams', 'calories', 'fat', 'carb', 'protein']
TYPE_SCHEMA = {
    'total_calories': float, 'total_mass': float, 'total_fat': float,
    'total_carb': float, 'total_protein': float, 'grams': float,
    'calories': float, 'fat': float, 'carb': float, 'protein': float
}


# --- 2. Functional Core ---

def create_image_path_csv(root_dir: Path, output_path: Path) -> None:
    """
    Scans for processed JPEG images and creates a CSV of their paths and dish_ids.

    Args:
        root_dir (Path): The root directory to search for images.
        output_path (Path): The path to save the resulting CSV file.
    """
    print(f"Scanning for images in '{root_dir}'...")
    
    # pathlib.Path.rglob is a generator, making it memory-efficient for many files.
    # It finds all files matching the pattern in the directory and its subdirectories.
    image_paths = list(root_dir.rglob("frames_sampled70/*.jpeg"))
    
    if not image_paths:
        print("Warning: No images found. Check the path and file structure.")
        return

    # Create a DataFrame from the path objects
    df = pd.DataFrame(image_paths, columns=['path'])
    
    # Use pathlib's attributes for robust parsing. .parent.parent.name gets the dish_id folder name.
    df['dish_id'] = df['path'].apply(lambda p: p.parent.parent.name)
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully created image path CSV with {len(df)} entries at '{output_path}'")


def _chunker(sequence: List[Any], size: int) -> Generator[List[Any], None, None]:
    """Yields successive size-sized chunks from a sequence."""
    for i in range(0, len(sequence), size):
        yield sequence[i:i + size]


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


def parse_dish_metadata_to_df(file_path: Path) -> pd.DataFrame:
    """
    Parses complex dish metadata CSV into a clean pandas DataFrame.

    Args:
        file_path (Path): Path to the metadata CSV.

    Returns:
        pd.DataFrame: A DataFrame with one row per dish and ingredients as a JSON string.
    """
    print(f"Parsing dish metadata from '{file_path}'...")
    parsed_dishes = []
    
    try:
        with file_path.open(mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue

                dish_data = dict(zip(DISH_COLUMNS, row))
                dish_data = _convert_types(dish_data, TYPE_SCHEMA)

                ingredient_data_flat = row[len(DISH_COLUMNS):]
                ingredients_list = [
                    _convert_types(dict(zip(INGREDIENT_COLUMNS, chunk)), TYPE_SCHEMA)
                    for chunk in _chunker(ingredient_data_flat, len(INGREDIENT_COLUMNS))
                ]
                
                # Store ingredients as a JSON string for easy CSV storage and retrieval
                dish_data['ingredients'] = json.dumps(ingredients_list)
                parsed_dishes.append(dish_data)
                
    except FileNotFoundError:
        print(f"Error: Metadata file not found at '{file_path}'")
        return pd.DataFrame()

    print(f"Successfully parsed {len(parsed_dishes)} dishes.")
    return pd.DataFrame(parsed_dishes)


# --- 3. Main Execution Block ---

def main():
    """Main function to orchestrate the data preparation process."""
    print("--- Starting Data Preparation Script ---")
    
    # Task 1: Create the image paths CSV
    create_image_path_csv(IMAGE_DIR, OUTPUT_IMAGES_CSV)
    
    print("\n" + "-"*40 + "\n")
    
    # Task 2: Parse metadata and save as CSV
    df_labels = parse_dish_metadata_to_df(METADATA_PATH)
    if not df_labels.empty:
        df_labels.to_csv(OUTPUT_LABELS_CSV, index=False)
        print(f"Successfully saved labels CSV at '{OUTPUT_LABELS_CSV}'")
        print("Labels DataFrame head:")
        print(df_labels.head())

    print("\n--- Data Preparation Finished ---")


if __name__ == '__main__':
    main()