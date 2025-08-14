# scripts/predict.py
import yaml
import argparse
import json
from pathlib import Path
import sys
import configparser

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.macro_estimator.training.predictor import Predictor

def main():
    """
    Main function to run the prediction script from the command line.
    """
    parser = argparse.ArgumentParser(description="Estimate nutritional content from a food image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading or parsing config file: {e}")
        return

    try:
        predictor = Predictor(config)
        result = predictor.predict_from_file(Path(args.image_path))
        
        if result:
            print("\n--- Prediction Results ---")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")

if __name__ == '__main__':
    main()