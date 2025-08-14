# --- Etapa 1: Base ---
# Usar una imagen oficial de Python. 'slim-bullseye' es ligera y estable.
FROM python:3.10-slim-bullseye

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# --- Etapa 2: Instalar Dependencias ---
# Copiar solo el archivo de requerimientos primero para aprovechar el caché de Docker
# Esto evita reinstalar todo cada vez que cambias el código.
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Etapa 3: Copiar la Aplicación ---
# Copiar el código fuente, la configuración y los artefactos del modelo
COPY ./src /app/src
COPY ./scripts /app/scripts
COPY ./config /app/config
COPY ./artifacts /app/artifacts

# --- Etapa 4: Configuración de Red y Ejecución ---
# Exponer el puerto en el que se ejecuta la API
EXPOSE 8000

# Comando para ejecutar la aplicación cuando se inicie el contenedor
# Usar 0.0.0.0 para que la API sea accesible desde fuera del contenedor.
CMD ["uvicorn", "scripts.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]