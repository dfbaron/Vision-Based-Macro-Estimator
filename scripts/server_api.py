# scripts/serve_api.py
import sys
from pathlib import Path
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Añadir el directorio src al path de Python
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Importar la clase Predictor
from src.macro_estimator.training.predictor import Predictor

# --- 1. Cargar Configuración y Modelo (¡se hace una sola vez al iniciar!) ---

def load_config(config_path="configs/training_config.yaml") -> dict:
    """Carga la configuración desde un archivo YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Cargar la configuración
config = load_config()

# Crear una instancia global del Predictor.
# Esto carga el modelo en memoria una vez, al iniciar la API.
try:
    predictor = Predictor(config)
    print("✅ Model loaded successfully. API is ready.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    predictor = None

# --- 2. Crear la Aplicación FastAPI ---

app = FastAPI(
    title="Vision-Based Macro Estimator API",
    description="An API to estimate nutritional values from food images.",
    version="1.0.0"
)

# --- 3. Definir los Endpoints de la API ---

@app.get("/", tags=["Health Check"])
async def root():
    """Endpoint raíz para verificar que la API está funcionando."""
    return {"message": "Welcome to the Macro Estimator API!"}

@app.post("/predict/", tags=["Prediction"])
async def predict_image(file: UploadFile = File(..., description="An image file of a meal.")):
    """
    Recibe una imagen, la procesa con el modelo y devuelve las macros estimadas.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    # Asegurarse de que el archivo es una imagen
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Leer el contenido del archivo en bytes
        image_bytes = await file.read()
        
        # Obtener las predicciones usando el método de bytes
        predictions = predictor.predict_from_bytes(image_bytes)
        
        return JSONResponse(content=predictions)
        
    except Exception as e:
        # Capturar cualquier error durante la predicción
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# --- 4. Ejecutar el Servidor (si el script se ejecuta directamente) ---

if __name__ == '__main__':
    # Esto permite ejecutar la API con: python scripts/serve_api.py
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)