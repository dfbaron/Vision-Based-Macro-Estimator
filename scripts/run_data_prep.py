import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import the DataPreparer class
from src.macro_estimator.data_preprocessing import DataPreparer

def main():
    """Entry point for running the data preparation process."""
    # Instantiate the DataPreparer class.
    # You can pass custom paths/filenames here if needed,
    # otherwise it will use the defaults defined in the class.
    data_preparer = DataPreparer(
        base_data_path=Path("data/Extracted_Files/nutrition5k_dataset"),
        output_dir=Path("data/csv_files"),
        image_glob_pattern="frames_sampled70/*.jpeg" # Ensure this matches your frame extraction
    )
    
    # Run the full data preparation pipeline
    data_preparer.run_preparation()

if __name__ == '__main__':
    main()