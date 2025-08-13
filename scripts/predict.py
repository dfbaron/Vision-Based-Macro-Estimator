# scripts/predict.py
import yaml
import argparse
import json
from ..src.macro_estimator.prediction.predictor import Predictor
from pathlib import Path

def main():
    """
    Main function to run the prediction script from the command line.
    """
    # 1. Configurar el parser de argumentos de la línea de comandos
    parser = argparse.ArgumentParser(description="Estimate nutritional content from a food image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    # 2. Cargar la configuración
    config_path = Path(args.config)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading or parsing config file: {e}")
        return

    # 3. Inicializar el predictor y hacer la predicción
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