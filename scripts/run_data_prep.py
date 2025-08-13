# scripts/run_data_prep.py
import sys
import configparser
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.macro_estimator.data_preprocessing import DataPreparer

def load_config(config_path="config/config.yaml") -> dict:
    """Loads and parses the INI configuration file."""
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser

def main():
    """Entry point for running the data preparation process."""
    config = load_config()
    data_preparer = DataPreparer(config)
    data_preparer.run()

if __name__ == '__main__':
    main()