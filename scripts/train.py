import sys
import os
from pathlib import Path
import configparser

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import the ModelTrainer class
from src.macro_estimator.training.trainer import ModelTrainer

def main():
    """
    Defines the configuration and runs the model training pipeline.
    """
    # --- Centralized Configuration Dictionary ---
    config = configparser.ConfigParser()
    config.read('config/config.yaml')

    # Instantiate and run the trainer
    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()