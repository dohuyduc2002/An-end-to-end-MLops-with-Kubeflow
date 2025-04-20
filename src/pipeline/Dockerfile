# Use the correct image tag (with colon, not @sha)
FROM ghcr.io/kubeflow/kubeflow/notebook-servers/jupyter-scipy:sha-4a26c7b5e9575410613faf7df6735aa1883a2d24

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy your requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optionally define entrypoint
# CMD ["python", "main.py"]
