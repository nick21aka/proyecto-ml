# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Evitar caché y escritura de bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias (opcional pero recomendable)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Comando por defecto: ejecutar pipeline de clasificación
CMD ["kedro", "run", "--pipeline=classification"]
