# Usamos la imagen oficial de Python 3.11
FROM python:3.13.3-slim-bookworm

# Establecemos el directorio de trabajo en /app
WORKDIR /app

# Copiamos el archivo de dependencias
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY data/ data/

# Establecemos el comando por defecto para iniciar el contenedor
CMD ["python"]