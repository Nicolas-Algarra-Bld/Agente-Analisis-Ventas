FROM python:3.10-slim

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias del sistema necesarias para matplotlib
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar version mas reciente de pip y aumentar timeout
RUN pip install --upgrade pip \
    && pip install pyarrow>=7.0 --timeout 300 --retries 10

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la app
COPY app.py .
COPY timeline.py .
COPY agent_graph.py .

# Crear carpeta para archivos generados
RUN mkdir -p outputs

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando de arranque
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
