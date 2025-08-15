# Vision-Based Macro Estimator

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning project to estimate the nutritional content (calories, fat, carbohydrates, and protein) directly from an image of a meal. This repository showcases a full MLOps pipeline, from reproducible data preprocessing and model training to a containerized API for inference.

---

### 🎥 Demo

This project is deployed as a containerized FastAPI service. An interactive demo can be built on top using tools like Gradio or a simple web front-end.

*[**Acción requerida:** Graba un GIF corto mostrando cómo envías una imagen al endpoint de la API (usando la documentación de FastAPI en http://localhost:8080/docs) y muestras el resultado JSON. Un GIF es extremadamente efectivo.]*

![Demo GIF Placeholder](https://i.imgur.com/example.gif)
*(Ejemplo: Sube una imagen a la API y obtén las macros estimadas al instante)*

---

## 📋 Table of Contents
1.  [Problem Statement](#-problem-statement)
2.  [Features](#-features)
3.  [Tech Stack & Architecture](#-tech-stack--architecture)
4.  [Project Structure](#-project-structure)
5.  [Setup & Installation](#-setup--installation)
6.  [Usage - A Reproducible Pipeline](#-usage---a-reproducible-pipeline)
7.  [Dockerization - Build & Run](#-dockerization---build--run)
8.  [Model Performance](#-model-performance)
9.  [Future Work](#-future-work)
10. [License](#-license)

---

## 🎯 Problem Statement

Tracking nutritional intake is crucial for health and fitness, but manual logging of meals is tedious, time-consuming, and often inaccurate. This project automates this process by leveraging computer vision. By simply taking a photo of a meal, the system provides a robust estimate of its macronutrient content, making dietary tracking effortless and accessible.

## ✨ Features

-   **Nutritional Estimation**: Predicts four key nutritional values: **Total Calories (kcal), Fat (g), Carbohydrates (g), and Protein (g)**.
-   **Deep Learning Model**: Fine-tunes a pre-trained **Vision Transformer (ViT-B/16 on ImageNet-21k)** for the regression task.
-   **End-to-End MLOps Pipeline**: A complete, script-driven pipeline from raw data to a trained, deployable model.
-   **Reproducibility First**:
    -   **Configuration-Driven**: All hyperparameters and paths are managed via a `config.ini` file, separating code from configuration.
    -   **Reproducible Data Splits**: A dedicated script creates fixed train, validation, and test sets based on `dish_id` to prevent data leakage.
    -   **Dependency Management**: A Conda `environment.yml` file ensures a consistent development environment.
    -   **Containerization**: A `Dockerfile` is provided to create a portable and reproducible production environment for the API.
-   **Robust Training**: Implements **Early Stopping** to prevent overfitting and saves the best-performing model based on validation loss.
-   **API for Inference**: A **FastAPI** service exposes the trained model via a clean, documented REST API endpoint for easy integration.

---

## 🛠️ Tech Stack & Architecture

This project is built with industry-standard tools for machine learning and software development.

-   **Backend & Modeling**:
    -   ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
    -   ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
    -   ![Timm](https://img.shields.io/badge/-Timm-grey) (for easy access to state-of-the-art vision models)
    -   ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) & ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
    -   ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
-   **Deployment & API**:
    -   ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white)
    -   ![Uvicorn](https://img.shields.io/badge/-Uvicorn-green)
    -   ![Docker](https://img.shields.io/badge/-Docker-2496ED?logo=docker&logoColor=white)
-   **Development Tools**:
    -   ![Conda](https://img.shields.io/badge/-Conda-44A833?logo=conda-forge&logoColor=white)
    -   ![Git](https://img.shields.io/badge/-Git-F05032?logo=git&logoColor=white) & ![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)

---

## 📁 Project Structure

The repository is organized following professional Object-Oriented and modular design principles.

```
vision-based-macro-estimator/
├── artifacts/                # Stores model checkpoints (.pth files)
├── configs/                  # Configuration files (config.ini, etc.)
├── data/                     # Data directory (ignored by Git)
├── scripts/                  # Executable scripts for running the pipeline
│   ├── run_data_prep.py
│   ├── train.py
│   └── server_api.py
├── src/
│   └── macro_estimator/      # Source code package
│       ├── __init__.py
│       ├── data_preprocessing.py # DataPreparer class
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
# It is recommended to use Mamba for faster installation
# conda install -n base mamba -c conda-forge

mamba env create -f environment.yml

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
This script loads the processed data, initializes the model, and runs the training loop. All parameters are controlled by `configs/training_config.ini`.
```bash
python scripts/train.py
```
The best model checkpoint is saved in `artifacts/models/`.

**Step 3: Run the API Locally**
Serve the model with a FastAPI endpoint.
```bash
uvicorn scripts.serve_api:app --reload
```
You can now access the interactive API documentation at **http://127.0.0.1:8000/docs**.

---

## 🐳 Dockerization - Build & Run

The application is fully containerized for easy and reproducible deployment.

**1. Build the Docker Image**
From the root of the project, run:
```bash
docker build -t macro-estimator-api .
```

**2. Run the Docker Container**
This command starts the container and maps your local port 8080 to the container's port 8000.
```bash
docker run -p 8080:8000 macro-estimator-api
```

Your API is now running inside a Docker container and is accessible at **http://localhost:8080/docs**.

---

## 📊 Model Performance

*[**Acción requerida:** Actualiza esta tabla con los resultados finales de tu modelo en el conjunto de test. El MAE (Error Absoluto Medio) es una métrica muy interpretable para este problema.]*

The final model was evaluated on the unseen test set.

| Metric        | Total Calories (kcal) | Fat (g) | Carbs (g) | Protein (g) |
|---------------|-----------------------|---------|-----------|-------------|
| **MAE**       |         152.90        | 8.58	  | 7.72    | 7.80      |
| **MSE**       |        48846.40       | 161.67  | 116.20    | 118.66      |

---

## 💡 Future Work

-   **Improve Model Accuracy**: Experiment with different pre-trained backbones (e.g., ConvNeXt, EfficientNetV2) and more complex regression heads.
-   **Implement Segmentation**: Extend the model to first perform semantic segmentation to identify individual food items on the plate for more granular estimation.
-   **Cloud Deployment**: Deploy the containerized FastAPI service on a cloud platform like Google Cloud Run or AWS Elastic Beanstalk for a scalable, real-world service.
-   **CI/CD Pipeline**: Implement a GitHub Actions workflow to automatically test, build, and push the Docker image on new commits.

---

## 📜 License
This project is licensed under the MIT License.