FROM python:3.11.11-slim

# Set arguments
ARG JUPYTERLAB_VERSION=4.3.5
ARG NODE_MAJOR_VERSION=20
ARG PIP_VERSION=24.3.1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR_VERSION}.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install JupyterLab
RUN pip install --upgrade pip==${PIP_VERSION} && \
    pip install jupyterlab==${JUPYTERLAB_VERSION}

# Copy and install Python dependencies from src/kfp/requirements.txt
COPY src/kfp/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
