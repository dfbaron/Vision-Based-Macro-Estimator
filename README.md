# Vision-Based Macro Estimator

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![App](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning project to estimate the nutritional content (calories, fat, carbohydrates, and protein) directly from an image of a meal. This repository showcases a full MLOps pipeline, from reproducible data preprocessing and model training to a fully interactive Streamlit web application with user authentication and persistent data storage.

---

### 🎥 Interactive Demo

The final model is deployed in a user-friendly Streamlit web application. It features user registration/login, persistent meal history via a SQLite database, and interactive charts to track nutritional progress.

![App Demo](img/Demo.gif)

---

## 📋 Table of Contents
1.  [Problem Statement](#-problem-statement)
2.  [Features](#-features)
3.  [Tech Stack & Architecture](#-tech-stack--architecture)
4.  [Project Structure](#-project-structure)
5.  [Setup & Installation](#-setup--installation)
6.  [Usage - A Reproducible Pipeline](#-usage---a-reproducible-pipeline)
7.  [Model Performance](#-model-performance)
8.  [Future Work](#-future-work)
9.  [License](#-license)

---

## 🎯 Problem Statement

Tracking nutritional intake is crucial for health and fitness, but manual logging of meals is tedious, time-consuming, and often inaccurate. This project automates this process by leveraging computer vision. By simply taking a photo of a meal, the system provides a robust estimate of its macronutrient content within a full-featured application, making dietary tracking effortless and insightful.

## ✨ Features

-   **AI-Powered Nutritional Estimation**: Predicts **Calories, Fat, Carbs, and Protein** from a single image.
-   **Full-Featured Web Application**: An interactive **Streamlit** app provides a complete user experience.
-   **User Authentication**: Secure user registration and login system with password hashing.
-   **Persistent Data Storage**: Utilizes a **SQLite** database to store user profiles, goals, and meal history.
-   **Interactive Dashboard**: Users can track their daily progress against personal goals with dynamic charts and metrics.
-   **Meal History & Trends**: A detailed log of all meals with visualizations to track nutritional trends over time.
-   **End-to-End MLOps Pipeline**:
    -   **Reproducible Data Splits**: A dedicated script creates fixed train/val/test sets based on `dish_id` to prevent data leakage.
    -   **Configuration-Driven**: All hyperparameters and paths are managed via a `config.ini` file.
    -   **Dependency Management**: A Conda `environment.yml` file ensures a consistent development environment.
-   **Robust Training**: Implements **Early Stopping** to prevent overfitting and save the best model.

---

## 🛠️ Tech Stack & Architecture

This project is built with industry-standard tools for machine learning and application development.

-   **Modeling & Data Science**:
    -   ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
    -   ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
    -   ![Timm](https://img.shields.io/badge/-Timm-grey) (for SOTA vision models)
    -   ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) & ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
    -   ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
-   **Application & Deployment**:
    -   ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
    -   ![SQLite](https://img.shields.io/badge/-SQLite-003B57?logo=sqlite&logoColor=white)
    -   ![Docker](https://img.shields.io/badge/-Docker-2496ED?logo=docker&logoColor=white) (for containerizing the application)
-   **Development Tools**:
    -   ![Conda](https://img.shields.io/badge/-Conda-44A833?logo=conda-forge&logoColor=white)
    -   ![Git](https://img.shields.io/badge/-Git-F05032?logo=git&logoColor=white) & ![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)

---

## 📁 Project Structure

The repository follows professional Object-Oriented and modular design principles.

```
vision-based-macro-estimator/
├── artifacts/                # Stores model checkpoints (.pth files)
├── configs/                  # Configuration files (config.ini, training_config.yaml)
├── data/                     # Data directory (ignored by Git)
├── scripts/                  # Executable scripts for running the pipeline
│   ├── run_data_prep.py      # Prepares and splits the data
│   ├── train.py              # Runs the model training
│   └── app.py                # Runs the Streamlit application
├── src/
│   └── macro_estimator/      # Source code package
│       ├── __init__.py
│       ├── data_preprocessing.py # DataPreparer class
│       ├── database_utils.py     # Database interaction class
│       ├── datasets.py           # Custom PyTorch Dataset class
│       ├── models/
│       │   └── vit_regressor.py  # Model architecture
│       ├── training/
│       │   ├── trainer.py        # ModelTrainer class
│       │   └── predictor.py      # Predictor class
│       └── utils.py              # Helper classes (EarlyStopping)
├── .dockerignore
├── .gitignore
├── Dockerfile                # Instructions to build the production image
├── environment.yml           # Conda environment definition
└── README.md
```

---

## ⚙️ Setup & Installation

Follow these steps to set up the project environment.

**1. Clone the Repository**
```bash
git clone https://github.com/dfbaron/Vision-Based-Macro-Estimator.git
cd Vision-Based-Macro-Estimator
```

**2. Create the Conda Environment**
Use the provided `environment.yml` file to create a consistent Conda environment.
```bash
# This command will create an environment named 'macro-estimator'
conda env create -f environment.yml

# Activate the environment
conda activate macro-estimator
```

**3. Download the Dataset**
This project uses the **Nutrition5k** dataset. Download the raw data and place its contents in a `data/raw/` directory (you may need to create this folder).

---

## 🚀 Usage - A Reproducible Pipeline

The pipeline is executed in a sequence of scripts.

**Step 1: Prepare the Data**
This script parses metadata, finds image files, creates the train/validation/test splits, and saves them as clean CSVs.
```bash
python scripts/run_data_prep.py
```

**Step 2: Train the Model**
This script loads the processed data, initializes the model, and runs the training loop. All parameters are controlled by `configs/training_config.yaml`.
```bash
python scripts/train.py
```
The best model checkpoint is saved in `artifacts/models/`.

**Step 3: Run the Streamlit Application**
Launch the interactive web application.
```bash
streamlit run scripts/app.py
```
Your application will be available at **http://localhost:8501**.

---

## 📊 Model Performance

*[**Action Required:** Update this table with your final model's results on the test set. MAE (Mean Absolute Error) is a highly interpretable metric for this problem.]*

The final model was evaluated on the unseen test set.

| Metric        | Total Calories (kcal) | Fat (g) | Carbs (g) | Protein (g) |
|---------------|-----------------------|---------|-----------|-------------|
| **MAE**       |         152.90        | 8.58	  | 7.72    | 7.80      |
| **MSE**       |        48846.40       | 161.67  | 116.20    | 118.66      |
| **R² Score**  |      221.01      |12.71|10.78|10.89

---

## 💡 Future Work

-   **Improve Model Accuracy**: Experiment with different pre-trained backbones (e.g., ConvNeXt, EfficientNetV2) and more complex regression heads.
-   **Implement Food Segmentation**: Extend the model to first perform semantic segmentation to identify individual food items on the plate, allowing for more granular and editable nutritional estimation.
-   **Cloud Deployment**: Containerize the Streamlit application with Docker and deploy it on a cloud platform like Streamlit Community Cloud, Heroku, or AWS for public access.
-   **CI/CD Pipeline**: Implement a GitHub Actions workflow to automatically test, build, and deploy the application on new commits.

---

## 📜 License
This project is licensed under the MIT License.
